"""e-BH (F17) and BH selection.

e-BH: with e-values e_1..e_m sorted decreasingly e_[1] >= ... >= e_[m], reject the k* largest,
k* = max{k : k e_[k] / m >= 1/alpha} (0 if none). Controls FDR <= alpha under arbitrary dependence.
BH: with p-values sorted increasingly, k* = max{k : p_(k) <= k alpha / m}.
"""

from __future__ import annotations

import numpy as np


def ebh(e: np.ndarray, alpha: float, m: int | None = None) -> np.ndarray:
    """Boolean rejection mask. m defaults to len(e); pass a FIXED m (D50) when some hypotheses
    are not in `e` (they count as e = 1 and are never rejected)."""
    e = np.asarray(e, float)
    m = len(e) if m is None else int(m)
    if m < len(e):
        raise ValueError("m must be >= number of e-values")
    order = np.argsort(-e, kind="stable")
    k = np.arange(1, len(e) + 1)
    ok = k * e[order] / m >= 1.0 / alpha
    rej = np.zeros(len(e), bool)
    if ok.any():
        kstar = int(np.flatnonzero(ok).max()) + 1
        rej[order[:kstar]] = True
    return rej


def ebh_log(log_e: np.ndarray, alpha: float, m: int | None = None) -> np.ndarray:
    """e-BH on log e-values (avoids overflow): k log-comparisons."""
    le = np.asarray(log_e, float)
    m = len(le) if m is None else int(m)
    order = np.argsort(-le, kind="stable")
    k = np.arange(1, len(le) + 1)
    ok = np.log(k) + le[order] - np.log(m) >= -np.log(alpha)
    rej = np.zeros(len(le), bool)
    if ok.any():
        rej[order[: int(np.flatnonzero(ok).max()) + 1]] = True
    return rej


def bh(p: np.ndarray, alpha: float, m: int | None = None) -> np.ndarray:
    p = np.asarray(p, float)
    m = len(p) if m is None else int(m)
    order = np.argsort(p, kind="stable")
    k = np.arange(1, len(p) + 1)
    ok = p[order] <= k * alpha / m
    rej = np.zeros(len(p), bool)
    if ok.any():
        rej[order[: int(np.flatnonzero(ok).max()) + 1]] = True
    return rej
