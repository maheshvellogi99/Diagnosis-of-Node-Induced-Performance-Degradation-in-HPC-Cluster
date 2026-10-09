"""Adaptive conformal inference for job alarms (PLAN.md §9 Phase 2.4; F19).

Per group: alarm_t = 1[p_t <= alpha_t]; alpha_{t+1} = alpha_t + gamma * (alpha - err_t),
err_t = alarm_t, alpha_1 = alpha. Jobs must be given in processing (end-time) order.
"""

from __future__ import annotations

import numpy as np


def aci_alarms(p: np.ndarray, groups: np.ndarray, alpha: float,
               gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (alarm bool array, alpha_t used for each job). `groups` keys the ACI state."""
    p = np.asarray(p, dtype="float64")
    groups = np.asarray(groups, dtype=object)
    if p.shape != groups.shape:
        raise ValueError("p and groups must have the same shape")
    state: dict = {}
    alarm = np.zeros(len(p), dtype=bool)
    used = np.empty(len(p), dtype="float64")
    for i, (pi, g) in enumerate(zip(p, groups, strict=True)):
        a = state.get(g, alpha)
        used[i] = a
        err = pi <= a
        alarm[i] = err
        state[g] = a + gamma * (alpha - float(err))
    return alarm, used
