"""Restart mixture for probe evidence (D109; PLAN.md §9 Phase 5, 5.3). All in log space.

For one node, investigations k = 1, 2, ... in time order, F_i = product of f over the probes of
investigation i, N = investigations finished so far:

    P = sum_{k <= N} w_k prod_{i=k..N} F_i + (1 - J)^N,   w_k = J (1 - J)^(k - 1),   P(0) = 1.

Recursion: A <- F (A + J R), R <- (1 - J) R, P = A + R, with A_0 = 0, R_0 = 1. Each restart term is
a test supermartingale equal to 1 before it starts and the weights sum to 1, so P is a test
supermartingale when every F is conditionally valid. J = 1 gives P = prod F (the old formula).
"""

from __future__ import annotations

import numpy as np


def restart_init(n: int) -> tuple[np.ndarray, np.ndarray]:
    """(log A, log R) for n nodes before any investigation: A = 0, R = 1."""
    return np.full(n, -np.inf), np.zeros(n)


def restart_step(logA, logR, logF, J: float) -> tuple[np.ndarray, np.ndarray]:
    """One finished investigation with log F (array or scalar, elementwise)."""
    if not 0.0 < J <= 1.0:
        raise ValueError("J must be in (0, 1]")
    log1mJ = -np.inf if J == 1.0 else np.log1p(-J)
    logA = np.asarray(logF, float) + np.logaddexp(logA, np.log(J) + np.asarray(logR, float))
    return logA, np.asarray(logR, float) + log1mJ


def restart_logP(logA, logR) -> np.ndarray:
    return np.logaddexp(logA, logR)


def restart_logP_seq(logF: np.ndarray, J: float) -> np.ndarray:
    """Running log P after each investigation of ONE node (reference for tests and reports)."""
    logA, logR = -np.inf, 0.0
    out = np.empty(len(logF))
    for i, x in enumerate(logF):
        logA, logR = restart_step(logA, logR, x, J)
        out[i] = restart_logP(logA, logR)
    return out
