"""Time-order tests for Phase 1 splits (CLAUDE.md rule 8, PLAN.md §7.4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from hpcdiag.layer1.splits import TimeIndex, fast_subset, split_times

T0 = pd.Timestamp("2020-01-01")


def _jobs(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    submit = T0 + pd.to_timedelta(rng.integers(0, 200 * 86400, n), unit="s")
    wait = pd.to_timedelta(rng.integers(0, 3 * 86400, n), unit="s")
    dur = pd.to_timedelta(rng.integers(0, 5 * 86400, n), unit="s")
    return pd.DataFrame({"submit_time": submit, "end_time": submit + wait + dur})


def test_split_times_spacing_and_last_full_test_day():
    s = pd.Series(pd.to_datetime(["2020-01-01", "2020-01-20"]))
    ts = split_times(s, offset_days=5, every_days=6, test_days=1)
    assert ts == [pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-12"),
                  pd.Timestamp("2020-01-18")]
    # t + 1 d == max submit is allowed (full test window)
    s2 = pd.Series(pd.to_datetime(["2020-01-01", "2020-01-19"]))
    assert split_times(s2, 5, 6, 1)[-1] == pd.Timestamp("2020-01-18")
    s3 = pd.Series([pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-18 23:59:59")])
    assert split_times(s3, 5, 6, 1)[-1] == pd.Timestamp("2020-01-12")


def test_fast_subset_evenly_spaced():
    full = [T0 + pd.Timedelta(days=6 * i) for i in range(244)]
    sub = fast_subset(full, 12)
    assert len(sub) == 12 and sub[0] == full[0] and sub[-1] == full[-1]
    idx = [full.index(t) for t in sub]
    assert np.all(np.diff(idx) > 0) and max(np.diff(idx)) - min(np.diff(idx)) <= 1
    assert fast_subset(full[:5], 12) == full[:5]


def _check_split(df: pd.DataFrame, idx: TimeIndex, t: pd.Timestamp, train_days=100, test_days=1):
    tr = idx.train_positions(t, train_days)
    te = idx.test_positions(t, test_days)
    end, sub = df["end_time"], df["submit_time"]
    lo = t - pd.Timedelta(days=train_days)
    # no training job ends at/after t, none ends before t - 100 d
    assert (end.iloc[tr] < t).all()
    assert (end.iloc[tr] >= lo).all()
    # exactly the brute-force sets
    exp_tr = np.flatnonzero(((end >= lo) & (end < t)).to_numpy())
    exp_te = np.flatnonzero(((sub >= t) & (sub < t + pd.Timedelta(days=test_days))).to_numpy())
    np.testing.assert_array_equal(tr, exp_tr)
    np.testing.assert_array_equal(te, exp_te)


def test_train_test_windows_match_brute_force():
    df = _jobs(5000, 1)
    idx = TimeIndex.build(df["end_time"], df["submit_time"])
    for t in split_times(df["submit_time"], 100, 6, 1):
        _check_split(df, idx, t)


def test_job_ending_exactly_at_t_is_excluded_from_training():
    t = pd.Timestamp("2020-05-01")
    df = pd.DataFrame({"submit_time": [t - pd.Timedelta(days=1)] * 3 + [t],
                       "end_time": [t - pd.Timedelta(seconds=1), t, t + pd.Timedelta(hours=1),
                                    t + pd.Timedelta(hours=2)]})
    idx = TimeIndex.build(df["end_time"], df["submit_time"])
    assert idx.train_positions(t, 100).tolist() == [0]
    assert idx.test_positions(t, 1).tolist() == [3]


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 10_000), n=st.integers(1, 400),
       offset=st.floats(0, 150), train_days=st.floats(0.5, 120))
def test_property_no_training_job_ends_at_or_after_t(seed, n, offset, train_days):
    df = _jobs(n, seed)
    idx = TimeIndex.build(df["end_time"], df["submit_time"])
    t = T0 + pd.Timedelta(days=offset)
    _check_split(df, idx, t, train_days=train_days)


def test_null_times_rejected():
    df = pd.DataFrame({"submit_time": [T0, T0], "end_time": [T0, pd.NaT]})
    try:
        TimeIndex.build(df["end_time"], df["submit_time"])
    except ValueError:
        return
    raise AssertionError("expected ValueError")
