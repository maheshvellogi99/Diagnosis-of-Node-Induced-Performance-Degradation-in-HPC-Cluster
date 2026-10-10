"""Power diagnostics for Layer 2 (D59, D62).

D59: paired check — an affected job's p-value in a node scenario vs the SAME job's p-value in the
matching null; the median paired difference must be negative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def paired_check(p_scen: pd.Series, p_null: pd.Series, affected: np.ndarray) -> dict:
    """p_scen, p_null indexed by row_id; affected = row_ids changed by injection.
    Uses affected jobs that are scored (finite p) in both."""
    common = pd.Index(affected).intersection(p_scen.index).intersection(p_null.index)
    a, b = p_scen.reindex(common).to_numpy(), p_null.reindex(common).to_numpy()
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    return {"n_pairs": int(ok.sum()), "median_diff": float(np.median(d)) if len(d) else np.nan,
            "mean_diff": float(np.mean(d)) if len(d) else np.nan,
            "share_negative": float(np.mean(d < 0)) if len(d) else np.nan,
            "share_zero": float(np.mean(d == 0)) if len(d) else np.nan,
            "median_p_null": float(np.median(b[ok])) if len(d) else np.nan,
            "median_p_scen": float(np.median(a[ok])) if len(d) else np.nan}


def before_after(p: pd.Series, end_ns: pd.Series, onset_ns: int) -> dict:
    """Median p of a node's jobs before vs after onset (same p-values, no pairing)."""
    e = end_ns.to_numpy()
    v = p.to_numpy()
    ok = np.isfinite(v)
    pre, post = v[ok & (e <= onset_ns)], v[ok & (e > onset_ns)]
    return {"n_before": int(len(pre)), "n_after": int(len(post)),
            "median_before": float(np.median(pre)) if len(pre) else np.nan,
            "median_after": float(np.median(post)) if len(post) else np.nan}


def gate2_l2_for(p: np.ndarray, phase2_cfg: dict) -> dict:
    """Gate 2-L2 (D26 thresholds, Phase 2 code) for a set of p-values (NaN = warm-up, dropped)."""
    from hpcdiag.layer1.post_scores import TailCache, _basic, gate2_l2

    m = _basic(p, phase2_cfg, TailCache(phase2_cfg))
    m["gate2_l2"] = gate2_l2(m, phase2_cfg)
    return m
