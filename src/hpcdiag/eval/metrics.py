"""Runtime-prediction metrics (PLAN.md §10; F20; Phase 1 addendum D, Gate 1 rule).

y = actual run time (s), yhat = predicted run time (s).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

H = 3600.0


def rmse_h(y: np.ndarray, yhat: np.ndarray) -> float:
    """sqrt(mean((yhat - y)^2)) / 3600."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.sqrt(np.mean((yhat - y) ** 2)) / H)


def mae_h(y: np.ndarray, yhat: np.ndarray) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.mean(np.abs(yhat - y)) / H)


def mean_signed_error_h(y: np.ndarray, yhat: np.ndarray) -> float:
    """mean(yhat - y) / 3600; negative = underestimation."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.mean(yhat - y) / H)


def a_geo(y: np.ndarray, yhat: np.ndarray, nodes: np.ndarray) -> float:
    """TARE A_geo = prod a_i^(c_i / sum c), a_i = min(yhat/y, y/yhat), c_i = nodes * y (F20).

    Requires y > 0 and yhat > 0 (callers clip yhat and filter y)."""
    y, yhat, nodes = (np.asarray(v, float) for v in (y, yhat, nodes))
    if (y <= 0).any() or (yhat <= 0).any():
        raise ValueError("a_geo needs y > 0 and yhat > 0")
    a = np.minimum(yhat / y, y / yhat)
    c = nodes * y
    return float(np.exp(np.sum(c * np.log(a)) / np.sum(c)))


def under_rate(y: np.ndarray, yhat: np.ndarray) -> float:
    """mean 1[yhat < y] (F20)."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.mean(yhat < y))


def tare_metrics(y, yhat, nodes, min_y_exclusive: float = 30.0, clip_min: float = 1.0) -> dict:
    """A_geo and UnderRate on jobs with y > min_y_exclusive, with yhat clipped to >= clip_min."""
    y, yhat, nodes = (np.asarray(v, float) for v in (y, yhat, nodes))
    m = y > min_y_exclusive
    yc = np.maximum(yhat[m], clip_min)
    return {"n": int(m.sum()), "a_geo": a_geo(y[m], yc, nodes[m]) if m.any() else float("nan"),
            "under_rate": under_rate(y[m], yc) if m.any() else float("nan")}


def log_ratio(y, yhat) -> np.ndarray:
    """r = log(max(yhat, 1) / y) (Gate 1 qualitative rule; requires y > 0)."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return np.log(np.maximum(yhat, 1.0) / y)


def bin_labels_for(y: np.ndarray, edges_h: list[float], labels: list[str]) -> np.ndarray:
    """Bin of actual run time: [e_k, e_{k+1}) in hours, last bin open-ended."""
    edges = np.asarray(edges_h, float) * H
    idx = np.searchsorted(edges, np.asarray(y, float), side="right") - 1
    if (idx < 0).any():
        raise ValueError("run time below first bin edge")
    return np.asarray(labels, dtype=object)[idx]


def per_bin_table(y, yhat, edges_h: list[float], labels: list[str],
                  r_min_y_exclusive: float = 30.0) -> pd.DataFrame:
    """Per actual-runtime bin: n, RMSE h, MAE h, mean signed error h (all jobs, unclipped yhat);
    median r, UnderRate (jobs with y > r_min_y_exclusive; r uses max(yhat, 1))."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    b = bin_labels_for(y, edges_h, labels)
    rows = []
    for lab in labels:
        m = b == lab
        mr = m & (y > r_min_y_exclusive)
        rows.append({
            "bin": lab, "n": int(m.sum()),
            "rmse_h": rmse_h(y[m], yhat[m]) if m.any() else np.nan,
            "mae_h": mae_h(y[m], yhat[m]) if m.any() else np.nan,
            "mean_signed_error_h": mean_signed_error_h(y[m], yhat[m]) if m.any() else np.nan,
            "mean_actual_h": float(y[m].mean() / H) if m.any() else np.nan,
            "mean_pred_h": float(yhat[m].mean() / H) if m.any() else np.nan,
            "n_r": int(mr.sum()),
            "median_r": float(np.median(log_ratio(y[mr], yhat[mr]))) if mr.any() else np.nan,
            "under_rate_r": under_rate(y[mr], np.maximum(yhat[mr], 1.0)) if mr.any() else np.nan,
        })
    return pd.DataFrame(rows)


def qualitative_gate(table: pd.DataFrame, long_bin: str = ">2d") -> dict:
    """Pass if median r in the long bin is negative and the lowest of all bins."""
    med = table.set_index("bin")["median_r"]
    v = float(med[long_bin])
    lowest = bool(np.isfinite(v) and v == med.min(skipna=True))
    return {"median_r_long_bin": v, "negative": bool(v < 0), "lowest": lowest,
            "pass": bool(v < 0 and lowest)}
