"""Layer-1 scores and p-values (PLAN.md §9 Phase 2.2–2.3; D17, D19).

Score S = log T - q_hat_0.9(x). Mode A: split-conformal p-value against the group's calibration
scores. Mode E: smoothed online p-values over streams in (end_time, row_id) order.
Groups: user -> partition -> global with n_min (D7).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hpcdiag.stats.conformal import (
    OnlineStreams,
    conformal_upper_offset,
    smoothed_split_pvalues,
)

GLOBAL = "global"


def user_key(u) -> str:
    return f"user:{u}"


def partition_key(p) -> str:
    return f"partition:{p}"


@dataclass
class ModeAResult:
    p: np.ndarray
    group_key: np.ndarray
    n_cal: np.ndarray
    upper_offset: np.ndarray  # CQR offset at alpha_bound for the chosen group


def mode_a_pvalues(
    cal_score: np.ndarray,
    cal_user: np.ndarray,
    cal_part: np.ndarray,
    test_score: np.ndarray,
    test_user: np.ndarray,
    test_part: np.ndarray,
    n_min: int,
    theta: np.ndarray,
    alpha_bound: float = 0.1,
) -> ModeAResult:
    """Smoothed split-conformal p-values (D23); calibration group = test job's user if it has
    >= n_min calibration jobs, else its partition if >= n_min, else all calibration jobs."""
    cal = pd.DataFrame({"s": cal_score, "u": cal_user, "p": cal_part})
    by_user = {k: v.to_numpy() for k, v in cal.groupby("u", sort=False)["s"]}
    by_part = {k: v.to_numpy() for k, v in cal.groupby("p", sort=False)["s"]}
    glob = cal["s"].to_numpy()

    n = len(test_score)
    p = np.empty(n)
    key = np.empty(n, dtype=object)
    ncal = np.empty(n, dtype="int64")
    off = np.empty(n)
    test = pd.DataFrame({"s": test_score, "u": test_user, "p": test_part})
    for (u, part), idx in test.groupby(["u", "p"], sort=False).indices.items():
        if u in by_user and len(by_user[u]) >= n_min:
            ref, k = by_user[u], user_key(u)
        elif part in by_part and len(by_part[part]) >= n_min:
            ref, k = by_part[part], partition_key(part)
        else:
            ref, k = glob, GLOBAL
        p[idx] = smoothed_split_pvalues(ref, test["s"].to_numpy()[idx], np.asarray(theta)[idx])
        key[idx] = k
        ncal[idx] = len(ref)
        off[idx] = conformal_upper_offset(ref, alpha_bound) if len(ref) else np.inf
    return ModeAResult(p, key, ncal, off)


@dataclass
class ModeEResult:
    p_fallback: np.ndarray
    key_fallback: np.ndarray
    p_user: np.ndarray
    n_user: np.ndarray


def mode_e_pvalues(
    score: np.ndarray, user: np.ndarray, part: np.ndarray, theta: np.ndarray, n_min: int
) -> ModeEResult:
    """Smoothed online p-values; inputs must already be in (end_time, row_id) order.

    Every job updates its user, partition and global stream. Fallback (D19 default): report the
    user-stream p if the user had >= n_min earlier jobs, else partition if >= n_min, else global.
    Variant: the user-stream p for every job (pure per-user Mondrian)."""
    streams = OnlineStreams()
    n = len(score)
    pf, pu = np.empty(n), np.empty(n)
    kf = np.empty(n, dtype=object)
    nu = np.empty(n, dtype="int64")
    for i, (s, u, pt, th) in enumerate(zip(score, user, part, theta, strict=True)):
        uk, pk = user_key(u), partition_key(pt)
        n_u_prev, n_p_prev = streams.count(uk), streams.count(pk)
        p_u, nu[i] = streams.update(uk, s, th)
        p_p, _ = streams.update(pk, s, th)
        p_g, _ = streams.update(GLOBAL, s, th)
        pu[i] = p_u
        if n_u_prev >= n_min:
            pf[i], kf[i] = p_u, uk
        elif n_p_prev >= n_min:
            pf[i], kf[i] = p_p, pk
        else:
            pf[i], kf[i] = p_g, GLOBAL
    return ModeEResult(pf, kf, pu, nu)
