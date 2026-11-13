"""Layer-1 alarm baselines (PLAN.md §9 Phase 2.5).

B-guide: relative residual r = (T - T_hat)/T_hat > 0.20 (hpc_project.pdf), T_hat clipped >= 1 s.
B-UARP:  T > Q0.99 + max(0.2 Q0.99, 0.5 sigma), sigma(x) = sqrt(E[(y - Q0.50)^2 | x]) (F10),
         all in seconds; LightGBM with the UARP settings.
B-user:  T > walltime_req_s (context only).
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np

from hpcdiag.layer1.quantile import fit_l2, fit_quantile


def guide_alarm(T: np.ndarray, T_hat: np.ndarray, threshold: float = 0.20,
                clip_min: float = 1.0) -> np.ndarray:
    th = np.maximum(np.asarray(T_hat, float), clip_min)
    return (np.asarray(T, float) - th) / th > threshold


def uarp_alarm(T: np.ndarray, q99: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    q99 = np.asarray(q99, float)
    margin = np.maximum(0.2 * q99, 0.5 * np.asarray(sigma, float))
    return np.asarray(T, float) > q99 + margin


def user_alarm(T: np.ndarray, walltime_s: np.ndarray) -> np.ndarray:
    return np.asarray(T, float) > np.asarray(walltime_s, float)


@dataclass
class UARPModel:
    q50: lgb.LGBMRegressor
    q99: lgb.LGBMRegressor
    var: lgb.LGBMRegressor

    @classmethod
    def fit(cls, X, y_seconds: np.ndarray, params: dict, seed: int) -> UARPModel:
        y = np.asarray(y_seconds, float)
        q50 = fit_quantile(X, y, 0.50, params, seed)
        q99 = fit_quantile(X, y, 0.99, params, seed)
        resid2 = (y - q50.predict(X)) ** 2          # residual model fitted on training rows
        var = fit_l2(X, resid2, params, seed)
        return cls(q50, q99, var)

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        """(Q0.99, sigma) in seconds; sigma = sqrt(max(predicted squared residual, 0))."""
        return self.q99.predict(X), np.sqrt(np.maximum(self.var.predict(X), 0.0))
