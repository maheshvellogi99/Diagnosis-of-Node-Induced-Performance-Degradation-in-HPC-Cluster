"""Layer-1 inputs for Layer 2 on every scenario (PLAN.md §9 Phase 4; D22, D28, D34, D38, D41).

No model is refitted per scenario. The frozen Mode E model (Phase 2, PM100, L = 45) is refitted
ONCE, deterministically, on exactly its Phase 2 training rows and checked against the stored
Phase 2 q_hat; injection never touches those rows (onsets >= freeze + 7 d, D31).
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sortedcontainers import SortedList

from hpcdiag.features.encoding import encode_frame
from hpcdiag.layer1.baseline import make_model
from hpcdiag.layer1.quantile import fit_quantile, log_runtime
from hpcdiag.layer1.scores import mode_e_pvalues
from hpcdiag.stats.conformal import sliding_window_pvalues
from hpcdiag.utils.run import load_config, resolve

H_NS = 3600 * 10**9


@dataclass
class Frozen:
    freeze: pd.Timestamp
    preds: pd.DataFrame  # row_id, q_hat (tau 0.9, log), q50 (log), that (XGB seconds)
    check: dict


def frozen_predictions(cfg: dict, jobs: pd.DataFrame, log=None) -> Frozen:
    """q_hat, q50 (LightGBM, Phase 2 settings) and T_hat (Phase-1-style XGBoost) for every
    eligible row, from models frozen on the pre-freeze rows. Cached in data/interim/phase4."""
    d = cfg["data"]
    p2 = load_config(resolve(d["phase2_config"]))
    e_run = resolve(d["phase2_mode_e_run"])
    freeze = pd.Timestamp(json.loads((e_run / "metrics.json").read_text())["freeze_time"])
    cache = resolve(d["cache_dir"]) / "frozen_predictions.parquet"
    elig = jobs[
        jobs["state_raw"].isin(cfg["population"]["states"]) & jobs["walltime_req_s"].notna()
    ].reset_index(drop=True)
    if cache.exists():
        preds = pd.read_parquet(cache)
        check = json.loads((cache.parent / "frozen_check.json").read_text())
        return Frozen(freeze, preds, check)
    f = p2["features"]
    pop = elig[elig["run_time_s"] > cfg["population"]["min_run_time_s_exclusive"]]
    tr = pop[pop["end_time"] < freeze]  # = Phase 2 Mode E training rows
    xtr, xall, _ = encode_frame(tr, elig, f["numeric"], f["categorical"], f["unseen_code"])
    ytr = log_runtime(tr["run_time_s"])
    qp, seed = p2["quantile"]["lgbm_params"], p2["seed"]
    m90 = fit_quantile(xtr, ytr, p2["quantile"]["taus"]["upper"], qp, seed)
    m50 = fit_quantile(xtr, ytr, p2["quantile"]["taus"]["median"], qp, seed)
    # Phase-1-style point model for baselines B1/B2: all eligible rows ending before the freeze
    trx = elig[elig["end_time"] < freeze]
    xb, xball, _ = encode_frame(trx, elig, f["numeric"], f["categorical"], f["unseen_code"])
    xgb = make_model(cfg["baselines"]["xgb_params"]).fit(xb, trx["run_time_s"].to_numpy(float))
    preds = pd.DataFrame(
        {
            "row_id": elig["row_id"].to_numpy(),
            "q_hat": m90.predict(xall),
            "q50": m50.predict(xall),
            "that": xgb.predict(xball).astype(float),
        }
    )
    # exact check against the stored Phase 2 Mode E q_hat
    stored = pd.read_parquet(e_run / "scores_eval.parquet", columns=["row_id", "q_hat"])
    mm = stored.merge(preds[["row_id", "q_hat"]], on="row_id", suffixes=("_p2", "_new"))
    diff = float(np.abs(mm["q_hat_p2"] - mm["q_hat_new"]).max())
    check = {
        "n_compared": int(len(mm)),
        "n_stored": int(len(stored)),
        "max_abs_diff": diff,
        "train_rows": int(len(tr)),
        "xgb_train_rows": int(len(trx)),
        "freeze": str(freeze),
    }
    if len(mm) != len(stored) or diff > 1e-9:
        raise RuntimeError(f"frozen model does not reproduce Phase 2 q_hat: {check}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(cache, index=False)
    (cache.parent / "frozen_check.json").write_text(json.dumps(check, indent=2))
    if log:
        log.info("frozen predictions cached: %s", check)
    return Frozen(freeze, preds, check)


def theta_by_row(n_rows: int, seed: int) -> np.ndarray:
    """theta_j indexed by row_id (same theta for a job in every scenario)."""
    return np.random.default_rng(seed).uniform(size=n_rows)


def scenario_population(
    jobs: pd.DataFrame, fr: Frozen, theta: np.ndarray, cfg: dict
) -> pd.DataFrame:
    """Main population AFTER injection (D34): eligible states, run_time > 30 s, ending at/after the
    freeze; D14 order (end_time, row_id); with score S = log T - q_hat."""
    pc = cfg["population"]
    j = jobs[
        jobs["state_raw"].isin(pc["states"])
        & (jobs["run_time_s"] > pc["min_run_time_s_exclusive"])
        & (jobs["end_time"] >= fr.freeze)
    ]
    cols = [
        "row_id",
        "user",
        "partition",
        "submit_time",
        "start_time",
        "end_time",
        "run_time_s",
        "nodes_req",
        "walltime_req_s",
    ]
    j = j[cols].merge(fr.preds, on="row_id", how="inner", validate="one_to_one")
    j = j.sort_values(["end_time", "row_id"], kind="stable").reset_index(drop=True)
    j["score"] = log_runtime(j["run_time_s"].to_numpy()) - j["q_hat"].to_numpy()
    j["theta"] = theta[j["row_id"].to_numpy()]
    j["end_ns"] = j["end_time"].to_numpy().astype("datetime64[ns]").astype("int64")
    return j


def sliding_median_prev(s: np.ndarray, end_ns: np.ndarray, window_ns: int) -> np.ndarray:
    """Median of the scores of EARLIER jobs (processing order) with end >= end_j - window;
    0 if there is none (guard (a), D41)."""
    sl, dq = SortedList(), deque()
    out = np.zeros(len(s))
    for j in range(len(s)):
        lo = end_ns[j] - window_ns
        while dq and dq[0][0] < lo:
            sl.remove(dq.popleft()[1])
        n = len(sl)
        if n:
            out[j] = sl[n // 2] if n % 2 else 0.5 * (sl[n // 2 - 1] + sl[n // 2])
        sl.add(s[j])
        dq.append((end_ns[j], s[j]))
    return out


def mode_p(
    pop: pd.DataFrame, window_hours: float, min_ref: int, detrend_hours: float | None = None
) -> np.ndarray:
    """Mode P p-values (D22) in pop order; optional guard (a) detrending first (D41)."""
    s = pop["score"].to_numpy()
    e = pop["end_ns"].to_numpy()
    if detrend_hours:
        s = s - sliding_median_prev(s, e, int(detrend_hours * H_NS))
    p, _ = sliding_window_pvalues(s, e, pop["theta"].to_numpy(), int(window_hours * H_NS), min_ref)
    return p


def mode_e_asrun(pop: pd.DataFrame, n_min: int = 30) -> np.ndarray:
    """Mode E (fallback grouping, D19) p-values on this scenario's population (B6 ablation)."""
    return mode_e_pvalues(
        pop["score"].to_numpy(),
        pop["user"].astype(str).to_numpy(),
        pop["partition"].astype(str).to_numpy(),
        pop["theta"].to_numpy(),
        n_min,
    ).p_fallback


def load_placement(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(resolve(path), columns=["row_id", "node_id", "n_nodes"])


def add_centred(pop: pd.DataFrame, window_hours: float, min_n: int = 3) -> pd.DataFrame:
    """Adds score_c (D60) and centre_level (0 shape, 1 user, 2 none) to a population frame."""
    from hpcdiag.layer2.centred import centre_scores

    sc, lev = centre_scores(
        pop["score"].to_numpy(),
        pop["end_ns"].to_numpy(),
        pop["user"].astype(str).to_numpy(),
        pop["nodes_req"].to_numpy(),
        pop["walltime_req_s"].to_numpy(),
        int(window_hours * H_NS),
        min_n,
    )
    return pop.assign(score_c=sc, centre_level=lev)


def mode_p_on(
    pop: pd.DataFrame,
    col: str,
    window_hours: float,
    min_ref: int,
    detrend_hours: float | None = None,
    query_shift: float = 0.0,
) -> np.ndarray:
    """Mode P on any score column; optional guard (a) detrending; optional query shift (D61:
    p-value of S_j + shift against the unshifted reference set)."""
    s = pop[col].to_numpy()
    e = pop["end_ns"].to_numpy()
    if detrend_hours:
        s = s - sliding_median_prev(s, e, int(detrend_hours * H_NS))
    p, _ = sliding_window_pvalues(
        s,
        e,
        pop["theta"].to_numpy(),
        int(window_hours * H_NS),
        min_ref,
        query_scores=s + query_shift if query_shift else None,
    )
    return p
