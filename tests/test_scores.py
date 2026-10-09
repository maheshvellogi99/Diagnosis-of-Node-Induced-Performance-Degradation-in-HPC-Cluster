"""Layer-1 scoring: Mode A windows and groups, Mode E streams, baselines (PLAN.md Phase 2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hpcdiag.layer1.baselines import guide_alarm, uarp_alarm, user_alarm
from hpcdiag.layer1.quantile import log_runtime, pinball_loss
from hpcdiag.layer1.scores import GLOBAL, mode_a_pvalues, mode_e_pvalues
from hpcdiag.layer1.splits import TimeIndex, mode_a_windows
from hpcdiag.stats.conformal import smoothed_online_pvalues, split_pvalues

T0 = pd.Timestamp("2020-01-01")


def _jobs(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    submit = T0 + pd.to_timedelta(rng.integers(0, 300 * 86400, n), unit="s")
    end = submit + pd.to_timedelta(rng.integers(0, 4 * 86400, n), unit="s")
    return pd.DataFrame({"submit_time": submit, "end_time": end})


@pytest.mark.parametrize("t_days", [130, 150, 200, 280])
def test_mode_a_windows_disjoint_and_ordered(t_days):
    df = _jobs()
    idx = TimeIndex.build(df["end_time"], df["submit_time"])
    t = T0 + pd.Timedelta(days=t_days)
    tr, cal, te = mode_a_windows(idx, t, 100, 14, 1)
    assert len(tr) and len(cal) and len(te)
    assert not set(tr) & set(cal)                                    # no overlap (addendum B)
    e, s = df["end_time"], df["submit_time"]
    assert (e.iloc[tr] >= t - pd.Timedelta(days=114)).all()
    assert (e.iloc[tr] < t - pd.Timedelta(days=14)).all()
    assert (e.iloc[cal] >= t - pd.Timedelta(days=14)).all() and (e.iloc[cal] < t).all()
    assert (s.iloc[te] >= t).all() and (s.iloc[te] < t + pd.Timedelta(days=1)).all()
    # rule 7.4: every calibration job ended before every test job ended (and was submitted)
    assert e.iloc[cal].max() < s.iloc[te].min() <= e.iloc[te].min()


def test_mode_a_group_fallback():
    rng = np.random.default_rng(1)
    cal_u = np.array(["u1"] * 30 + ["u2"] * 5 + ["u9"] * 2)
    cal_p = np.array(["P"] * 35 + ["Q"] * 2)
    cal_s = rng.standard_normal(37)
    test_s = np.array([0.1, 0.2, 0.3, 0.4])
    test_u = np.array(["u1", "u2", "u3", "u4"])
    test_p = np.array(["P", "P", "Q", "R"])
    r = mode_a_pvalues(cal_s, cal_u, cal_p, test_s, test_u, test_p, n_min=30)
    assert r.group_key.tolist() == ["user:u1", "partition:P", GLOBAL, GLOBAL]
    assert r.n_cal.tolist() == [30, 35, 37, 37]
    assert r.p[0] == pytest.approx(split_pvalues(cal_s[:30], test_s[:1])[0])
    assert r.p[1] == pytest.approx(split_pvalues(cal_s[:35], test_s[1:2])[0])
    assert r.p[2] == pytest.approx(split_pvalues(cal_s, test_s[2:3])[0])


def test_mode_e_streams_and_switch():
    rng = np.random.default_rng(2)
    n = 120
    users = np.array(["a"] * 40 + ["b"] * 80)
    rng.shuffle(users)
    parts = np.where(users == "a", "P", "Q")
    s = rng.standard_normal(n)
    th = rng.uniform(size=n)
    r = mode_e_pvalues(s, users, parts, th, n_min=30)
    for u in ("a", "b"):
        m = users == u
        np.testing.assert_allclose(r.p_user[m], smoothed_online_pvalues(s[m], th[m]))
        keys = r.key_fallback[m]
        # first 30 jobs of a user cannot use the user stream (fewer than 30 earlier jobs)
        assert all(k != f"user:{u}" for k in keys[:30])
        assert all(k == f"user:{u}" for k in keys[30:])
    first = np.flatnonzero(r.key_fallback == GLOBAL)
    np.testing.assert_allclose(r.p_fallback[first], smoothed_online_pvalues(s, th)[first])


def test_mode_e_online_past_unchanged():
    rng = np.random.default_rng(3)
    n = 300
    u = rng.choice(["a", "b", "c"], n)
    p = rng.choice(["P", "Q"], n)
    s, th = rng.standard_normal(n), rng.uniform(size=n)
    a = mode_e_pvalues(s, u, p, th, 30)
    s2 = s.copy()
    s2[200:] += 50
    b = mode_e_pvalues(s2, u, p, th, 30)
    np.testing.assert_array_equal(a.p_fallback[:200], b.p_fallback[:200])
    np.testing.assert_array_equal(a.p_user[:200], b.p_user[:200])


def test_baselines_hand():
    assert guide_alarm([130, 120, 2], [100, 100, 0]).tolist() == [True, False, True]
    # margin = max(0.2*100, 0.5*10) = 20 -> alarm if T > 120; sigma 100 -> margin 50
    assert uarp_alarm([121, 120], [100, 100], [10, 10]).tolist() == [True, False]
    assert uarp_alarm([151, 150], [100, 100], [100, 100]).tolist() == [True, False]
    assert user_alarm([10, 11], [10, 10]).tolist() == [False, True]


def test_pinball_and_log_runtime():
    # tau=0.9: y-q = [1, -1] -> [0.9, 0.1] -> mean 0.5
    assert pinball_loss([2.0, 0.0], [1.0, 1.0], 0.9) == pytest.approx(0.5)
    np.testing.assert_allclose(log_runtime([0, 1, np.e]), [0, 0, 1])
