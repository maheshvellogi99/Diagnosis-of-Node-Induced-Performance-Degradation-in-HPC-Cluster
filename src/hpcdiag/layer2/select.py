"""Selection over time: e-BH on node e-processes and baselines B1-B5.

PLAN 4.3, 4.4; F17; D50; D55.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from hpcdiag.stats.calibrators import mixture_calibrator
from hpcdiag.stats.ebh import bh, ebh_log

DAY_NS = 24 * 3600 * 10**9


def ebh_over_time(logM: np.ndarray, alpha: float, m: int) -> np.ndarray:
    """Flags (n_evals, n_nodes) from e-BH with FIXED m at each evaluation."""
    return np.vstack([ebh_log(row, alpha, m) for row in logM])


def ville_flags(logM: np.ndarray, alpha: float) -> np.ndarray:
    """B5: per-node threshold M >= 1/alpha, no multiplicity control."""
    return logM >= np.log(1.0 / alpha)


def fisher_bh_flags(
    fisher_snap: np.ndarray, count_snap: np.ndarray, alpha: float, m: int
) -> np.ndarray:
    """B3: per-node Fisher combination of its p-values so far (chi2 with 2k df), BH with fixed m.
    Nodes without jobs get p = 1."""
    out = np.zeros(fisher_snap.shape, bool)
    for i in range(fisher_snap.shape[0]):
        k = count_snap[i]
        pn = np.ones(len(k))
        has = k > 0
        pn[has] = stats.chi2.sf(fisher_snap[i, has], 2 * k[has])
        out[i] = bh(pn, alpha, m)
    return out


def averaged_e_flags(
    sum_f_snap: np.ndarray, count_snap: np.ndarray, alpha: float, m: int
) -> np.ndarray:
    """B4: node e-value = mean of the per-job e-values f(p_j) (an e-value); e-BH with fixed m."""
    e = np.where(count_snap > 0, sum_f_snap / np.maximum(count_snap, 1), 1.0)
    return np.vstack([ebh_log(np.log(row), alpha, m) for row in e])


def cumulative_node_stats(pairs: pd.DataFrame, p_job: np.ndarray, kappas) -> pd.DataFrame:
    """Per update: running count, sum of -2 log p (Fisher) and sum of f(p) (B4) per node."""
    pk = pairs["k"].to_numpy()
    pj = p_job[pk]
    use = np.isfinite(pj)
    d = pd.DataFrame(
        {
            "node_id": pairs["node_id"].to_numpy()[use],
            "end_ns": pairs["end_ns"].to_numpy()[use],
            "fis": -2.0 * np.log(pj[use]),
            "f": mixture_calibrator(pj[use], kappas),
        }
    )
    g = d.groupby("node_id", sort=False)
    d["cnt"] = g.cumcount() + 1
    d["fis"] = g["fis"].cumsum()
    d["f"] = g["f"].cumsum()
    return d


def residual_baselines(
    pairs: pd.DataFrame,
    rel_resid: np.ndarray,
    users: np.ndarray,
    day_ns: np.ndarray,
    n_nodes: int,
    cfg: dict,
) -> dict[str, np.ndarray]:
    """B1 (guide heuristic, D13/D55) and B2 (top 5%) on trailing 7-day windows evaluated daily.
    rel_resid, users: per job (pop order). Returns flags (n_days, n_nodes) for 'B1' and 'B2'."""
    b = cfg["baselines"]
    win = int(b["window_days"] * DAY_NS)
    pk = pairs["k"].to_numpy()
    d = pd.DataFrame(
        {
            "node": pairs["node_id"].to_numpy(),
            "e": pairs["end_ns"].to_numpy(),
            "r": rel_resid[pk],
            "u": users[pk],
        }
    ).sort_values("e", kind="stable")
    e = d["e"].to_numpy()
    b1_cond = np.zeros((len(day_ns), n_nodes), bool)
    b2 = np.zeros((len(day_ns), n_nodes), bool)
    k_top = int(np.ceil(b["b2_top_frac"] * n_nodes))
    for i, t in enumerate(day_ns):
        lo, hi = np.searchsorted(e, t - win, side="right"), np.searchsorted(e, t, side="right")
        w = d.iloc[lo:hi]
        if w.empty:
            continue
        g = w.groupby("node").agg(
            med=("r", "median"), mean=("r", "mean"), n=("r", "size"), users=("u", "nunique")
        )
        c = g[
            (g["med"] > b["b1"]["median_rel_resid"])
            & (g["n"] >= b["b1"]["min_jobs"])
            & (g["users"] >= b["b1"]["min_users"])
        ]
        b1_cond[i, c.index.to_numpy()] = True
        top = g.sort_values(["mean"], ascending=False, kind="stable").head(k_top)
        b2[i, top.index.to_numpy()] = True
    persist = b["b1"]["persist"]
    b1 = np.zeros_like(b1_cond)
    for i in range(persist - 1, len(day_ns)):
        b1[i] = b1_cond[i - persist + 1 : i + 1].all(axis=0)
    return {"B1": b1, "B2": b2}
