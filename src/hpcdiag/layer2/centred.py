"""Centred score (D60).

S_c(j) = S(j) - median(S of the same user's jobs with the same nodes_req and walltime_req_s that
ended in the 7 days before end_time(j), if >= 3 such jobs); else the same user's jobs in that
window (>= 3); else S(j). Only jobs that ended STRICTLY before j are used (equal end times are
excluded). Level: 0 = same user and shape, 1 = same user, 2 = not centred.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from sortedcontainers import SortedList


class _Win:
    """Sliding multiset of scores keyed by group, with eviction by end time."""

    def __init__(self) -> None:
        self.sl: dict = {}
        self.dq: dict = {}

    def evict(self, key, lo: int) -> SortedList | None:
        sl = self.sl.get(key)
        if sl is None:
            return None
        dq = self.dq[key]
        while dq and dq[0][0] < lo:
            sl.remove(dq.popleft()[1])
        return sl

    def add(self, key, end: int, s: float) -> None:
        if key not in self.sl:
            self.sl[key], self.dq[key] = SortedList(), deque()
        self.sl[key].add(s)
        self.dq[key].append((end, s))


def _median(sl: SortedList) -> float:
    n = len(sl)
    return sl[n // 2] if n % 2 else 0.5 * (sl[n // 2 - 1] + sl[n // 2])


def centre_scores(
    score: np.ndarray,
    end_ns: np.ndarray,
    user: np.ndarray,
    nodes_req: np.ndarray,
    walltime: np.ndarray,
    window_ns: int,
    min_n: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Inputs in D14 order (end_ns non-decreasing). Returns (S_c, level)."""
    n = len(score)
    if n > 1 and np.any(np.diff(end_ns) < 0):
        raise ValueError("end times must be non-decreasing")
    shape, usr = _Win(), _Win()
    sc = np.asarray(score, float).copy()
    level = np.full(n, 2, dtype=np.int8)
    i = 0
    while i < n:
        j = i
        while j < n and end_ns[j] == end_ns[i]:
            j += 1
        lo = int(end_ns[i]) - window_ns
        for k in range(i, j):  # query the state before this end time
            ks, ku = (user[k], nodes_req[k], walltime[k]), user[k]
            a = shape.evict(ks, lo)
            if a is not None and len(a) >= min_n:
                sc[k] = score[k] - _median(a)
                level[k] = 0
                continue
            b = usr.evict(ku, lo)
            if b is not None and len(b) >= min_n:
                sc[k] = score[k] - _median(b)
                level[k] = 1
        for k in range(i, j):  # then insert the whole tie group
            shape.add((user[k], nodes_req[k], walltime[k]), int(end_ns[k]), float(score[k]))
            usr.add(user[k], int(end_ns[k]), float(score[k]))
        i = j
    return sc, level
