"""Per-node evidence (PLAN.md §9 Phase 4.1–4.2; D40, D50, D51, D56).

Layer 2 state at evaluation time t uses only jobs that ended at or before t (PLAN §7.4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hpcdiag.stats.calibrators import power_calibrator
from hpcdiag.stats.evalues import (
    blame_logM,
    jumper_init,
    jumper_step,
    log_bet,
    log_bet_funcs,
    log_total,
    node_logM_by_rank,
    relative_evalues,
    relative_logM,
)

H_NS = 3600 * 10**9


def make_pairs(pop: pd.DataFrame, placement: pd.DataFrame) -> pd.DataFrame:
    """One row per (scored job, node), in pop (D14) order; `k` = position of the job in pop."""
    x = pop[["row_id", "user", "start_time", "end_ns"]].reset_index(names="k")
    x = x.merge(placement, on="row_id", how="inner", validate="one_to_many")
    return x.sort_values(["k", "node_id"], kind="stable").reset_index(drop=True)


def decoy_counts(placement: pd.DataFrame, jobs: pd.DataFrame, hours: float) -> pd.DataFrame:
    """c_{u,n} for every (job, node) in the placement (D40): number of jobs of the same user that
    STARTED on node n in [start_j - hours, start_j) — strictly before job j started."""
    x = placement[["row_id", "node_id"]].merge(jobs[["row_id", "user", "start_time"]], on="row_id")
    x["st"] = x["start_time"].to_numpy().astype("datetime64[ns]").astype("int64")
    x["g"] = x.groupby(["node_id", x["user"].astype(str)], sort=False).ngroup()
    x = x.sort_values(["g", "st"], kind="stable").reset_index(drop=True)
    span = int(x["st"].max() - x["st"].min()) + 10 * int(hours * H_NS) + 1
    key = x["g"].to_numpy().astype(np.int64) * span + (x["st"].to_numpy() - x["st"].min())
    lo = np.searchsorted(key, key - int(hours * H_NS), side="left")
    hi = np.searchsorted(key, key, side="left")  # excludes jobs starting at the same time
    x["c"] = hi - lo
    return x[["row_id", "node_id", "c"]]


def weights(pairs: pd.DataFrame, decoy: pd.DataFrame | None = None) -> np.ndarray:
    """w = 1/|N_j| (PLAN 4.1) or, with decoy counts, 1/(|N_j| max(1, c_{u,n})) (D40)."""
    w = 1.0 / pairs["n_nodes"].to_numpy(float)
    if decoy is None:
        return w
    c = pairs[["row_id", "node_id"]].merge(decoy, on=["row_id", "node_id"], how="left")["c"]
    return w / np.maximum(1.0, c.fillna(0).to_numpy(float))


def guard_b_frozen(p: np.ndarray, end_ns: np.ndarray, J: float, ratio: float) -> np.ndarray:
    """D56 + D51: global one-sided Simple-Jumper martingale on pooled p-values
    (f = 1 + eps (p - 1/2), eps in {-1, 0}); signal while S_t / min_{s<=t} S_s >= ratio
    (min reset when the signal ends).
    Freeze decision for job j = signal state after all jobs that ended STRICTLY before j."""
    n = len(p)
    sig_after = np.zeros(n, bool)
    logC = jumper_init(1, 2)
    log_min, on = 0.0, False
    for i in range(n):
        if np.isfinite(p[i]):
            lg = np.array([[0.0, np.log(1.5 - p[i])]])
            logC = jumper_step(logC, lg, J)
            ls = float(log_total(logC)[0])
            log_min = min(log_min, ls)
            now = ls - log_min >= np.log(ratio)
            if on and not now:
                log_min = ls  # reset when the signal ends
            on = now
        sig_after[i] = on
    last_before = np.searchsorted(end_ns, end_ns, side="left") - 1
    return np.where(last_before >= 0, sig_after[np.maximum(last_before, 0)], False)


def node_evidence(
    pairs: pd.DataFrame,
    p_job: np.ndarray,
    w: np.ndarray,
    kappas,
    J: float | None,
    frozen_job: np.ndarray | None = None,
) -> pd.DataFrame:
    """log M of each node after each of its updates. Pairs whose job has no p-value (warm-up) or
    is frozen (guard b) do not update. Returns rows (node_id, end_ns, logM) for updates only."""
    pk = pairs["k"].to_numpy()
    pj = p_job[pk]
    use = np.isfinite(pj)
    if frozen_job is not None:
        use &= ~frozen_job[pk]
    if len(kappas) and callable(kappas[0]):
        lb = log_bet_funcs(pj[use], w[use], kappas)  # D61 histogram bets
    else:
        lb = log_bet(pj[use], w[use], kappas)
    node = pairs["node_id"].to_numpy()[use]
    lm = node_logM_by_rank(node, lb, J)
    return pd.DataFrame({"node_id": node, "end_ns": pairs["end_ns"].to_numpy()[use], "logM": lm})


def node_evidence_blame(
    pairs: pd.DataFrame,
    p_job: np.ndarray,
    bets,
    J: float,
    frozen_job: np.ndarray | None = None,
) -> pd.DataFrame:
    """D71: as node_evidence with w_j replaced by lambda_{j,n} = M_n / sum_{k in N_j} M_k, taken
    from the state before the job's tie block. Same output plus the lambda of each update (`lam`).
    Every pair of a job has the same p-value, so a job updates all of its nodes or none."""
    pk = pairs["k"].to_numpy()
    pj = p_job[pk]
    use = np.isfinite(pj)
    if frozen_job is not None:
        use &= ~frozen_job[pk]
    pu = pj[use]
    node = pairs["node_id"].to_numpy()[use]
    end = pairs["end_ns"].to_numpy()[use]
    if len(bets) and callable(bets[0]):
        g = np.column_stack([f(pu) for f in bets])
    else:
        g = np.column_stack([power_calibrator(pu, k) for k in bets])
    m = int(pairs["node_id"].max()) + 1
    lm, lam = blame_logM(pk[use], end, node, g, J, m)
    return pd.DataFrame({"node_id": node, "end_ns": end, "logM": lm, "lam": lam})


def block_grid(eval_ns: np.ndarray, first_end_ns: int) -> np.ndarray:
    """Block ends for D78: the e-BH evaluation times, extended backwards at the same step so that
    jobs ending before the first evaluation also fall in 6-h blocks. A block is (t_{b-1}, t_b]."""
    step = int(eval_ns[1] - eval_ns[0])
    back = max(0, int(np.ceil((eval_ns[0] - first_end_ns) / step)))
    return np.r_[eval_ns[0] - step * np.arange(back, 0, -1), eval_ns]


def node_evidence_relative(
    pairs: pd.DataFrame,
    p_job: np.ndarray,
    part_job: np.ndarray,
    bets,
    J: float,
    lam: float,
    grid: np.ndarray,
    denom: str = "node",
) -> pd.DataFrame:
    """D78: one update per (block, node with a pair in the block), dated at the block end. Pairs
    without a p-value and jobs ending after the last block do not count. Same output columns as
    node_evidence. A node must belong to one partition only."""
    pk = pairs["k"].to_numpy()
    pj = p_job[pk]
    block = np.searchsorted(grid, pairs["end_ns"].to_numpy(), side="left")
    use = np.isfinite(pj) & (block < len(grid))
    pu, q, node = pj[use], part_job[pk[use]], pairs["node_id"].to_numpy()[use]
    if len(np.unique(np.c_[node, q], axis=0)) != len(np.unique(node)):
        raise ValueError("a node appears in more than one partition")
    f = np.column_stack([g(pu) for g in bets])
    m = int(pairs["node_id"].max()) + 1
    b, nd, e = relative_evalues(block[use], q, node, f, m, denom)
    return pd.DataFrame({"node_id": nd, "end_ns": grid[b], "logM": relative_logM(nd, e, lam, J)})


def snapshots(
    ev: pd.DataFrame, eval_ns: np.ndarray, n_nodes: int, col: str = "logM", fill: float = 0.0
) -> np.ndarray:
    """Matrix (n_evals, n_nodes): value of `col` after the node's last update with end <= t."""
    out = np.full((len(eval_ns), n_nodes), fill)
    cur = np.full(n_nodes, fill)
    e = ev["end_ns"].to_numpy()
    order = np.argsort(e, kind="stable")  # stable: keeps per-node update order
    e, nd, v = e[order], ev["node_id"].to_numpy()[order], ev[col].to_numpy()[order]
    pos = 0
    for i, t in enumerate(eval_ns):
        hi = np.searchsorted(e, t, side="right")
        if hi > pos:
            seg = pd.DataFrame({"n": nd[pos:hi], "v": v[pos:hi]}).drop_duplicates("n", keep="last")
            cur[seg["n"].to_numpy()] = seg["v"].to_numpy()
            pos = hi
        out[i] = cur
    return out
