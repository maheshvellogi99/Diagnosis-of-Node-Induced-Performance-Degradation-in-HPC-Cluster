"""Time-ordered train/test splits (PLAN.md §9 Phase 1, F6; CLAUDE.md rule 8).

train(t) = jobs with end_time in [t - train_days, t)
test(t)  = jobs with submit_time in [t, t + test_days)
Split times: first = earliest submit_time + offset, then every `every` days, while
t + test_days <= max submit_time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def split_times(submit_time: pd.Series, offset_days: float, every_days: float,
                test_days: float) -> list[pd.Timestamp]:
    """All split times for a dataset (addendum C)."""
    start, end = submit_time.min(), submit_time.max()
    t = start + pd.Timedelta(days=offset_days)
    out = []
    while t + pd.Timedelta(days=test_days) <= end:
        out.append(t)
        t += pd.Timedelta(days=every_days)
    return out


def fast_subset(times: list[pd.Timestamp], n: int) -> list[pd.Timestamp]:
    """n split times evenly spaced (by index) over the full list."""
    if n >= len(times):
        return list(times)
    idx = np.unique(np.round(np.linspace(0, len(times) - 1, n)).astype(int))
    return [times[i] for i in idx]


@dataclass
class TimeIndex:
    """Positions of jobs sorted by end_time and by submit_time, for fast window slicing."""

    end_order: np.ndarray
    end_sorted: np.ndarray
    submit_order: np.ndarray
    submit_sorted: np.ndarray

    @classmethod
    def build(cls, end_time: pd.Series, submit_time: pd.Series) -> TimeIndex:
        e = end_time.to_numpy(dtype="datetime64[us]")
        s = submit_time.to_numpy(dtype="datetime64[us]")
        if np.isnat(e).any() or np.isnat(s).any():
            raise ValueError("end_time/submit_time must not be null in the split frame")
        eo = np.argsort(e, kind="stable")
        so = np.argsort(s, kind="stable")
        return cls(eo, e[eo], so, s[so])

    def train_positions(self, t: pd.Timestamp, train_days: float) -> np.ndarray:
        """Row positions with end_time in [t - train_days, t)."""
        lo = np.datetime64(t - pd.Timedelta(days=train_days), "us")
        hi = np.datetime64(t, "us")
        a = np.searchsorted(self.end_sorted, lo, side="left")
        b = np.searchsorted(self.end_sorted, hi, side="left")  # excludes end_time == t
        return np.sort(self.end_order[a:b])

    def test_positions(self, t: pd.Timestamp, test_days: float) -> np.ndarray:
        """Row positions with submit_time in [t, t + test_days)."""
        lo = np.datetime64(t, "us")
        hi = np.datetime64(t + pd.Timedelta(days=test_days), "us")
        a = np.searchsorted(self.submit_sorted, lo, side="left")
        b = np.searchsorted(self.submit_sorted, hi, side="left")
        return np.sort(self.submit_order[a:b])


def mode_a_windows(index: TimeIndex, t: pd.Timestamp, train_days: float, cal_days: float,
                   test_days: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mode A (PLAN 2.3, addendum B): train end_time in [t - cal - train, t - cal),
    calibration end_time in [t - cal, t), test submit_time in [t, t + test).
    Train and calibration are disjoint by construction."""
    cal_start = t - pd.Timedelta(days=cal_days)
    train = index.train_positions(cal_start, train_days)
    cal = index.train_positions(t, cal_days)
    test = index.test_positions(t, test_days)
    return train, cal, test
