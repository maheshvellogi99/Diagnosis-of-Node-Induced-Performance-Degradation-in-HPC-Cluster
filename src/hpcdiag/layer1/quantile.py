"""LightGBM quantile models on log runtime (PLAN.md §9 Phase 2.1; F10, D20)."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np


def log_runtime(run_time_s: np.ndarray) -> np.ndarray:
    """y = log(max(run_time, 1)) (CLAUDE.md conventions)."""
    return np.log(np.maximum(np.asarray(run_time_s, dtype="float64"), 1.0))


def fit_quantile(X, y, tau: float, params: dict, seed: int) -> lgb.LGBMRegressor:
    """LightGBM with objective='quantile', alpha=tau."""
    m = lgb.LGBMRegressor(objective="quantile", alpha=tau, random_state=seed, verbose=-1, **params)
    return m.fit(X, y)


def fit_l2(X, y, params: dict, seed: int) -> lgb.LGBMRegressor:
    m = lgb.LGBMRegressor(objective="regression", random_state=seed, verbose=-1, **params)
    return m.fit(X, y)


def pinball_loss(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    """mean(max(tau (y - q), (tau - 1)(y - q)))."""
    d = np.asarray(y, float) - np.asarray(q, float)
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))
