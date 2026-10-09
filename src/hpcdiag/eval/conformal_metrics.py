"""Layer-1 conformal evaluation (PLAN.md §9 Phase 2.6, Gate 2; PROMPT 2 addendum E).

False-alarm rate (FA) = share of jobs with an alarm, on unmodified data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from hpcdiag.eval.metrics import bin_labels_for, tare_metrics


def ks_distance(p: np.ndarray) -> float:
    """sup_u |F_hat(u) - u| against U(0,1) (PLAN §10)."""
    p = np.asarray(p, float)
    return float(stats.kstest(p, "uniform").statistic) if len(p) else float("nan")


def fa_by_alpha(p: np.ndarray, alphas: list[float]) -> dict[str, float]:
    p = np.asarray(p, float)
    return {f"{a:g}": float(np.mean(p <= a)) for a in alphas}


def fa_by_bin(p, y, alpha: float, edges_h, labels) -> list[dict]:
    p, y = np.asarray(p, float), np.asarray(y, float)
    b = bin_labels_for(y, edges_h, labels)
    return [{"bin": lab, "n": int((b == lab).sum()),
             "fa": float(np.mean(p[b == lab] <= alpha)) if (b == lab).any() else float("nan")}
            for lab in labels]


def rate_by_bin(alarm, y, edges_h, labels) -> list[dict]:
    """Alarm rate per actual-runtime bin for a boolean alarm column."""
    alarm, y = np.asarray(alarm, bool), np.asarray(y, float)
    b = bin_labels_for(y, edges_h, labels)
    return [{"bin": lab, "n": int((b == lab).sum()),
             "fa": float(alarm[b == lab].mean()) if (b == lab).any() else float("nan")}
            for lab in labels]


def fa_by_resource_decile(p, nodes_req, walltime_s, alpha: float) -> dict:
    """Top decile of nodes_req x walltime_req_s (known at submission) vs the rest."""
    key = np.asarray(nodes_req, float) * np.asarray(walltime_s, float)
    p = np.asarray(p, float)
    thr = float(np.quantile(key, 0.9))
    top = key >= thr
    return {"threshold_node_seconds": thr, "n_top": int(top.sum()), "n_rest": int((~top).sum()),
            "fa_top": float(np.mean(p[top] <= alpha)) if top.any() else float("nan"),
            "fa_rest": float(np.mean(p[~top] <= alpha)) if (~top).any() else float("nan")}


def timeout_share(alarm: np.ndarray, state_raw: np.ndarray) -> dict:
    alarm = np.asarray(alarm, bool)
    to = np.asarray(state_raw) == "TIMEOUT"
    return {"n_alarms": int(alarm.sum()), "rate": float(alarm.mean()) if len(alarm) else np.nan,
            "timeout_share_of_alarms": float(to[alarm].mean()) if alarm.any() else float("nan")}


def ks_top_users(p, users, k: int) -> list[dict]:
    s = pd.Series(np.asarray(p, float))
    u = pd.Series(np.asarray(users, dtype=object))
    top = u.value_counts().head(k)
    return [{"user": str(name), "n": int(n), "ks": ks_distance(s[u == name].to_numpy())}
            for name, n in top.items()]


def evaluate(df: pd.DataFrame, pcol: str, cfg: dict, alarm_cols: list[str]) -> dict:
    """Metrics for one set of p-values. `df` needs run_time_s, state_raw, nodes_req,
    walltime_req_s, user and `pcol`; optional q50_log (TARE), ub_log (CQR bound)."""
    a, ev = cfg["alarms"], cfg["evaluation"]
    p = df[pcol].to_numpy()
    y = df["run_time_s"].to_numpy()
    out = {
        "n": int(len(df)),
        "timeout_share_of_jobs": float((df["state_raw"] == "TIMEOUT").mean()),
        "ks": ks_distance(p),
        "fa": fa_by_alpha(p, a["alphas"]),
        "fa_by_bin_alpha_job": fa_by_bin(p, y, a["alpha_job"], ev["bins_hours"], ev["bin_labels"]),
        "fa_by_resource_decile_alpha_job": {
            "key": ev["resource_key"],
            **fa_by_resource_decile(p, df["nodes_req"], df["walltime_req_s"], a["alpha_job"])},
        "ks_top_users": ks_top_users(p, df["user"], ev["top_users_ks"]),
        "alarms": {c: timeout_share(df[c].to_numpy(), df["state_raw"].to_numpy())
                   for c in alarm_cols},
        "alarms_by_bin": {c: rate_by_bin(df[c].to_numpy(), y, ev["bins_hours"], ev["bin_labels"])
                          for c in alarm_cols},
    }
    if "q50_log" in df:
        out["tare_median_model"] = tare_metrics(
            y, np.exp(df["q50_log"].to_numpy()), df["nodes_req"].to_numpy(),
            ev["tare_min_run_time_s_exclusive"], ev["tare_clip_pred_min_s"])
    if "ub_log" in df:
        ub = df["ub_log"].to_numpy()
        fin = np.isfinite(ub)
        out["upper_bound"] = {
            "alpha": cfg["mode_a"]["bound_alpha"],
            "coverage": float(np.mean(np.log(np.maximum(y, 1.0)) <= ub)),
            "share_infinite": float(np.mean(~fin)),
            "median_bound_h": float(np.median(np.exp(ub[fin])) / 3600) if fin.any() else np.nan,
            "median_bound_over_median_pred": _median_ratio(ub, df, fin),
        }
    if "group_key" in df:
        gk = df["group_key"].astype(str)
        out["group_level_share"] = {
            "user": float(gk.str.startswith("user:").mean()),
            "partition": float(gk.str.startswith("partition:").mean()),
            "global": float((gk == "global").mean())}
    return out


def gate2_mode_e(m: dict, cfg: dict) -> dict:
    g = cfg["gate2"]
    fa_ok = {k: bool(abs(v - float(k)) <= g["fa_tol"]) for k, v in m["fa"].items()}
    ks_ok = bool(m["ks"] <= g["ks_max"])
    return {"ks": m["ks"], "ks_ok": ks_ok, "fa": m["fa"], "fa_ok": fa_ok,
            "pass": bool(ks_ok and all(fa_ok.values()))}


def gate2_aci(rate: float, cfg: dict) -> dict:
    a = cfg["alarms"]["alpha_job"]
    ok = bool(abs(rate - a) <= cfg["gate2"]["aci_tol"])
    return {"rate": rate, "target": a, "tol": cfg["gate2"]["aci_tol"], "pass": ok}


def _median_ratio(ub: np.ndarray, df: pd.DataFrame, fin: np.ndarray) -> float:
    if not fin.any() or "q50_log" not in df:
        return float("nan")
    return float(np.median(np.exp(ub[fin] - df["q50_log"].to_numpy()[fin])))
