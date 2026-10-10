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
    return {
        "n_pairs": int(ok.sum()),
        "median_diff": float(np.median(d)) if len(d) else np.nan,
        "mean_diff": float(np.mean(d)) if len(d) else np.nan,
        "share_negative": float(np.mean(d < 0)) if len(d) else np.nan,
        "share_zero": float(np.mean(d == 0)) if len(d) else np.nan,
        "median_p_null": float(np.median(b[ok])) if len(d) else np.nan,
        "median_p_scen": float(np.median(a[ok])) if len(d) else np.nan,
    }


def before_after(p: pd.Series, end_ns: pd.Series, onset_ns: int) -> dict:
    """Median p of a node's jobs before vs after onset (same p-values, no pairing)."""
    e = end_ns.to_numpy()
    v = p.to_numpy()
    ok = np.isfinite(v)
    pre, post = v[ok & (e <= onset_ns)], v[ok & (e > onset_ns)]
    return {
        "n_before": int(len(pre)),
        "n_after": int(len(post)),
        "median_before": float(np.median(pre)) if len(pre) else np.nan,
        "median_after": float(np.median(post)) if len(post) else np.nan,
    }


def gate2_l2_for(p: np.ndarray, phase2_cfg: dict) -> dict:
    """Gate 2-L2 (D26 thresholds, Phase 2 code) for a set of p-values (NaN = warm-up, dropped)."""
    from hpcdiag.layer1.post_scores import TailCache, _basic, gate2_l2

    m = _basic(p, phase2_cfg, TailCache(phase2_cfg))
    m["gate2_l2"] = gate2_l2(m, phase2_cfg)
    return m


def detectability(
    pop: pd.DataFrame, placement: pd.DataFrame, cfg: dict, bets: dict, mode_p_on
) -> list[dict]:
    """D62: expected log-growth per job-node pair under a shift alternative, and implied days to
    reach log(m / alpha) at the median node's pair rate. For each s, every job's p-value is
    recomputed with its score shifted by log(s) against the unshifted reference (as if only that
    job were slowed). Rates: kappa mixture = max_kappa E log(1 + w (f_kappa(p) - 1)) (a mixture of
    products grows at its best component's rate); LR jumper = max_s0 E log(1 + w (g_s0(p) - 1))
    (jumper switching costs ignored); oracle = same with g = histogram of the true alternative."""
    from hpcdiag.layer2.bets import hist_bet
    from hpcdiag.stats.calibrators import power_calibrator

    l1, lc, dc = cfg["layer1"], cfg["layer2"], cfg["d62"]
    x = pop[["row_id", "end_ns"]].reset_index(names="k").merge(placement, on="row_id")
    w = 1.0 / x["n_nodes"].to_numpy(float)
    k = x["k"].to_numpy()
    span_days = (pop["end_ns"].max() - pop["end_ns"].min()) / (24 * 3600 * 1e9)
    rate_node = float(x.groupby("node_id").size().median() / span_days)
    target = float(np.log(lc["m_fixed"] / lc["alpha"]))
    out = []
    for lab, col in (("S", "score"), ("Sc", "score_c")):
        for s in dc["s"]:
            p = mode_p_on(pop, col, l1["window_hours"], l1["min_ref"], query_shift=float(np.log(s)))
            pp = p[k]
            ok = np.isfinite(pp)
            pp, ww = pp[ok], w[ok]
            kap = max(
                float(np.mean(np.log1p(ww * (power_calibrator(pp, kk) - 1.0))))
                for kk in lc["kappas"]
            )
            lr = max(float(np.mean(np.log1p(ww * (g(pp) - 1.0)))) for g in bets[lab])
            orc_g = hist_bet(p, cfg["shift_bets"]["bins"], "oracle")
            with np.errstate(divide="ignore"):
                orc = float(np.mean(np.log1p(ww * (orc_g(pp) - 1.0))))
            row = {
                "score": lab,
                "s": s,
                "n_pairs": int(ok.sum()),
                "pairs_per_day_median_node": rate_node,
                "target_log": target,
                "median_p_alt": float(np.median(p[np.isfinite(p)])),
            }
            for name, r in (("kappa", kap), ("lr_jumper", lr), ("oracle", orc)):
                row[f"rate_{name}"] = r
                row[f"days_{name}"] = target / (r * rate_node) if r > 0 else float("inf")
            out.append(row)
    return out
