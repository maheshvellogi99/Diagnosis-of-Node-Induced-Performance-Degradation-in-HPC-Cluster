"""Telemetry classifier helpers (D94)."""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew

from hpcdiag.layer3 import classifier as C


def test_window_stats_by_hand():
    v = np.arange(60, dtype=float)
    x = np.stack([v, 2 * v + 1], axis=1)[None]
    s = C.window_stats(x)
    assert s.shape == (1, 22)
    a = s[0, :11]
    assert a[0] == 0 and a[1] == 59 and a[2] == 29.5
    assert a[3] == v.var()
    assert a[4] == skew(v) and a[5] == kurtosis(v)
    assert np.allclose(a[6:], np.percentile(v, [5, 25, 50, 75, 95]))
    assert s[0, 11] == 1  # second metric starts after the first's 11 statistics


def test_window_stats_constant_has_no_nan():
    assert np.isfinite(C.window_stats(np.ones((2, 60, 3)))).all()


def test_cumulative_detection():
    assert C.is_cumulative(np.arange(100.0))
    assert not C.is_cumulative(np.sin(np.arange(100.0)))
    assert not C.is_cumulative(np.ones(100))


def test_per_second_ffill_limit_and_grid():
    t = np.array([100.2, 100.9, 101.1, 110.0])
    v = np.array([[1.0], [2.0], [3.0], [4.0]])
    o = C.per_second(t, v, 100, 12)[:, 0]
    assert o[0] == 2 and o[1] == 3  # last sample of the second wins
    assert (o[2:7] == 3).all() and np.isnan(o[7])  # carried at most 5 s
    assert o[10] == 4


def test_pure_windows():
    f = np.array(["None"] * 60 + ["leak"] * 30 + ["None"] * 30 + ["ddot"] * 60 + ["x"] * 10)
    assert C.pure_windows(f).tolist() == ["None", "", "ddot"]


def test_fault_type_drops_core():
    import pandas as pd

    assert C.fault_type(pd.Series(["leak_0", "None", "memeater_12"])).tolist() == [
        "leak",
        "None",
        "memeater",
    ]


def test_per_core_groups():
    g, p = C.per_core_groups(["per_core_x0", "per_core_x12", "per_core_y1", "user", "temp0.1"])
    assert g == {"per_core_x": ["per_core_x0", "per_core_x12"], "per_core_y": ["per_core_y1"]}
    assert p == ["user", "temp0.1"]


def test_time_folds_contiguous_disjoint_purged():
    t = np.arange(0, 100_000, 60.0)
    folds = C.time_folds(t, 5, 3600)
    seen = np.concatenate([te for _, te in folds])
    assert sorted(seen.tolist()) == list(range(len(t)))  # every window tested once
    for tr, te in folds:
        assert np.array_equal(te, np.arange(te[0], te[-1] + 1))  # contiguous
        dist = np.abs(t[tr][:, None] - t[te][None, :]).min()
        assert dist >= 3600
