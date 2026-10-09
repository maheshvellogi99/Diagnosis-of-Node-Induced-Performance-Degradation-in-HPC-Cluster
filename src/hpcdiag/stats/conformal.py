"""Conformal p-values (PLAN.md §9 Phase 2.3; F11, F12).

Large score = late job. Both functions return p-values for the hypothesis "this job is
exchangeable with the reference jobs"; small p = abnormally late.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable

import numpy as np
from sortedcontainers import SortedList


def smoothed_online_pvalues(scores: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Smoothed online conformal p-values for ONE stream in processing order (F12):

        p_n = (|{i <= n : S_i > S_n}| + theta_n * |{i <= n : S_i = S_n}|) / n

    i runs over the stream so far INCLUDING n. Under exchangeability p_1, p_2, ... are IID U(0,1).
    """
    scores = np.asarray(scores, dtype="float64")
    theta = np.asarray(theta, dtype="float64")
    if scores.shape != theta.shape:
        raise ValueError("scores and theta must have the same shape")
    stream = OnlineStreams()
    return np.array([stream.update(0, s, th)[0] for s, th in zip(scores, theta, strict=True)])


class OnlineStreams:
    """Many independent online conformal streams keyed by any hashable (user, partition, ...)."""

    def __init__(self) -> None:
        self._s: dict[Hashable, SortedList] = {}

    def count(self, key: Hashable) -> int:
        """Number of scores already in the stream (before the next update)."""
        sl = self._s.get(key)
        return 0 if sl is None else len(sl)

    def update(self, key: Hashable, score: float, theta: float) -> tuple[float, int]:
        """Add `score` to stream `key` and return (smoothed p-value of this score, n incl. it)."""
        if not np.isfinite(score):
            raise ValueError("score must be finite")
        sl = self._s.get(key)
        if sl is None:
            sl = self._s[key] = SortedList()
        sl.add(score)
        n = len(sl)
        right = sl.bisect_right(score)
        n_gt = n - right
        n_eq = right - sl.bisect_left(score)  # >= 1 (includes this score)
        return (n_gt + theta * n_eq) / n, n


def split_pvalues(cal_scores: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    """Split-conformal p-values (Mode A): p_j = (1 + |{i in C : S_i >= S_j}|) / (|C| + 1)."""
    cal = np.sort(np.asarray(cal_scores, dtype="float64"))
    t = np.asarray(test_scores, dtype="float64")
    n_ge = len(cal) - np.searchsorted(cal, t, side="left")
    return (1.0 + n_ge) / (len(cal) + 1.0)


def conformal_upper_offset(cal_scores: np.ndarray, alpha: float) -> float:
    """One-sided CQR offset (F11): the ceil((1 - alpha)(n + 1))-th smallest calibration score,
    +inf if that rank exceeds n. Upper bound = q_hat(x) + offset."""
    cal = np.sort(np.asarray(cal_scores, dtype="float64"))
    n = len(cal)
    k = int(np.ceil((1 - alpha) * (n + 1)))
    return float("inf") if k > n else float(cal[k - 1])


def smoothed_split_pvalues(cal_scores: np.ndarray, test_scores: np.ndarray,
                           theta: np.ndarray) -> np.ndarray:
    """Smoothed split-conformal p-values (Mode A, D23):
    p_j = (#{i in C : S_i > S_j} + theta_j (1 + #{i in C : S_i = S_j})) / (|C| + 1)."""
    cal = np.sort(np.asarray(cal_scores, dtype="float64"))
    t = np.asarray(test_scores, dtype="float64")
    th = np.asarray(theta, dtype="float64")
    if t.shape != th.shape:
        raise ValueError("test_scores and theta must have the same shape")
    right = np.searchsorted(cal, t, side="right")
    n_gt = len(cal) - right
    n_eq = right - np.searchsorted(cal, t, side="left")
    return (n_gt + th * (1.0 + n_eq)) / (len(cal) + 1.0)


def sliding_window_pvalues(scores: np.ndarray, end_ns: np.ndarray, theta: np.ndarray,
                           window_ns: int, min_ref: int,
                           keys: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Mode P (D22): pooled sliding-window smoothed conformal p-values.

    Inputs must be in D14 order (end_time, then row_id). For job j the reference set R_j is all
    EARLIER jobs (same key, if keys are given) with end_time >= end_time_j - window:
        p_j = (#{i in R_j : S_i > S_j} + theta_j (1 + #{i in R_j : S_i = S_j})) / (|R_j| + 1).
    If |R_j| < min_ref, p_j = NaN (warm-up). Returns (p, |R_j|)."""
    s = np.asarray(scores, dtype="float64")
    e = np.asarray(end_ns, dtype="int64")
    th = np.asarray(theta, dtype="float64")
    if not (s.shape == e.shape == th.shape):
        raise ValueError("scores, end_ns and theta must have the same shape")
    if len(e) > 1 and np.any(np.diff(e) < 0):
        raise ValueError("end times must be non-decreasing (D14 order)")
    k = np.zeros(len(s), dtype=np.int64) if keys is None else np.asarray(keys)
    state: dict = {}
    p = np.full(len(s), np.nan)
    nref = np.zeros(len(s), dtype="int64")
    for j in range(len(s)):
        st = state.get(k[j])
        if st is None:
            st = state[k[j]] = (SortedList(), deque())
        sl, dq = st
        lo = e[j] - window_ns
        while dq and dq[0][0] < lo:
            sl.remove(dq.popleft()[1])
        m = len(sl)
        nref[j] = m
        if m >= min_ref:
            right = sl.bisect_right(s[j])
            n_gt = m - right
            n_eq = right - sl.bisect_left(s[j])
            p[j] = (n_gt + th[j] * (1.0 + n_eq)) / (m + 1.0)
        sl.add(s[j])
        dq.append((e[j], s[j]))
    return p, nref
