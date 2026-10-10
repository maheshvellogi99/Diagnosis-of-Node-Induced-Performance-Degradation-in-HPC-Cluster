"""Layer-2 metrics over time (PLAN.md §9 Phase 4.5; D44–D47, D57)."""

from __future__ import annotations

import numpy as np
import pandas as pd

DAY_NS = 24 * 3600 * 10**9


def null_metrics(flags: np.ndarray, eval_ns: np.ndarray) -> dict:
    """For scenarios where every flag is false (null, decoy, shock)."""
    any_t = flags.any(axis=1)
    span_days = (eval_ns[-1] - eval_ns[0]) / DAY_NS if len(eval_ns) > 1 else np.nan
    first = int(np.argmax(any_t)) if any_t.any() else None
    return {
        "ever_flagged": bool(any_t.any()),
        "first_flag_eval": first,
        "first_flag_time": str(pd.Timestamp(int(eval_ns[first]))) if first is not None else None,
        "node_evals_flagged": int(flags.sum()),
        "flags_per_day": float(flags.sum() / span_days) if span_days else np.nan,
        "distinct_nodes_ever": int(flags.any(axis=0).sum()),
        "n_evals": int(len(eval_ns)),
    }


def node_metrics(
    flags: np.ndarray,
    eval_ns: np.ndarray,
    truth: pd.DataFrame,
    affected_main_nodes: set[int],
    coalloc_nodes: set[int],
    node_jobs_end: dict[int, np.ndarray],
    horizon_days: float,
    offsets_days: list[float],
) -> dict:
    """Recall (a) all degraded nodes, (b) degraded nodes with >= 1 affected main-population job;
    detection time = first evaluation at/after onset with the node flagged; delay in hours and in
    scored jobs on the node; FDP at every evaluation (true set = degraded nodes past onset), its
    mean, and at each degraded node's own onset + offsets (D57); false flags split (D46)."""
    deg = truth["node_id"].astype(int).to_numpy()
    on = truth["onset_time"].to_numpy().astype("datetime64[ns]").astype("int64")
    n_eval, n_nodes = flags.shape
    true_mask = np.zeros((n_eval, n_nodes), bool)
    for n, o in zip(deg, on, strict=True):
        true_mask[eval_ns >= o, n] = True
    nflag = flags.sum(axis=1)
    false = flags & ~true_mask
    fdp = np.where(nflag > 0, false.sum(axis=1) / np.maximum(nflag, 1), 0.0)
    det = []
    for n, o in zip(deg, on, strict=True):
        idx = np.flatnonzero((eval_ns >= o) & flags[:, n])
        t = int(eval_ns[idx[0]]) if len(idx) else None
        jobs = node_jobs_end.get(int(n), np.array([], dtype=np.int64))
        det.append(
            {
                "node_id": int(n),
                "onset_ns": int(o),
                "detected": t is not None,
                "delay_h": (t - o) / 3.6e12 if t is not None else np.nan,
                "delay_jobs": int(((jobs > o) & (jobs <= t)).sum()) if t is not None else None,
                "within_horizon": t is not None and (t - o) <= horizon_days * DAY_NS,
                "has_affected_main": int(n) in affected_main_nodes,
                # D68: right-censoring time (last evaluation) and the node's pair rate after onset
                "censor_h": (int(eval_ns[-1]) - o) / 3.6e12,
                "n_pairs_after": int((jobs > o).sum()),
                "pair_rate_per_day": float((jobs > o).sum())
                / max((int(eval_ns[-1]) - o) / DAY_NS, 1e-9),
            }
        )
    det = pd.DataFrame(det)
    out = {
        "n_degraded": int(len(deg)),
        "n_degraded_with_affected_main": int(det["has_affected_main"].sum()),
    }
    for lab, sub in (("a", det), ("b", det[det["has_affected_main"]])):
        out[f"recall_{lab}"] = float(sub["detected"].mean()) if len(sub) else np.nan
        out[f"recall_{lab}_within_{int(horizon_days)}d"] = (
            float(sub["within_horizon"].mean()) if len(sub) else np.nan
        )
    d = det[det["detected"]]
    out["median_delay_h"] = float(d["delay_h"].median()) if len(d) else np.nan
    out["median_delay_jobs"] = float(d["delay_jobs"].median()) if len(d) else np.nan
    out["fdp_mean_over_time"] = float(fdp.mean())
    out["fdp_mean_when_flags"] = float(fdp[nflag > 0].mean()) if (nflag > 0).any() else np.nan
    for off in offsets_days:
        vals = []
        for o in on:
            i = np.searchsorted(eval_ns, o + int(off * DAY_NS), side="left")
            if i < n_eval:
                vals.append(fdp[i])
        out[f"fdp_at_onset_plus_{int(off)}d"] = float(np.mean(vals)) if vals else np.nan
    ever_false = false.any(axis=0)
    fn = set(np.flatnonzero(ever_false).tolist())
    out["false_nodes_ever"] = len(fn)
    out["false_nodes_coallocated"] = len(fn & coalloc_nodes)
    out["false_nodes_unrelated"] = len(fn - coalloc_nodes)
    out["false_node_evals_coallocated"] = (
        int(false[:, sorted(fn & coalloc_nodes)].sum()) if fn & coalloc_nodes else 0
    )
    out["false_node_evals_unrelated"] = (
        int(false[:, sorted(fn - coalloc_nodes)].sum()) if fn - coalloc_nodes else 0
    )
    cum = np.cumsum(flags, axis=0) > 0  # descriptive: cumulative flagged set
    cum_true = cum & true_mask
    ncum = cum.sum(axis=1)
    out["fdp_cumulative_final"] = (
        float(1 - cum_true[-1].sum() / max(ncum[-1], 1)) if ncum[-1] else 0.0
    )
    out["detections"] = det.to_dict(orient="records")
    return out
