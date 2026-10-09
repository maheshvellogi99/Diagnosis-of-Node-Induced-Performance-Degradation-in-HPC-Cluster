"""Conformal p-values: hand examples, uniformity, online property (PLAN.md §7.5)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from hpcdiag.stats.conformal import (
    OnlineStreams,
    conformal_upper_offset,
    smoothed_online_pvalues,
    split_pvalues,
)

SEEDS = [0, 1, 2, 3, 4]


def test_smoothed_hand_example():
    # n=1: (0 + .5*1)/1; n=2 (3): (0 + .5*1)/2; n=3 (2): (1 + .5*1)/3; n=4 (3): (0 + .5*2)/4
    p = smoothed_online_pvalues(np.array([1.0, 3.0, 2.0, 3.0]), np.full(4, 0.5))
    np.testing.assert_allclose(p, [0.5, 0.25, 0.5, 0.25])


def test_smoothed_theta_extremes():
    s = np.array([5.0, 5.0, 5.0])
    np.testing.assert_allclose(smoothed_online_pvalues(s, np.zeros(3)), [0, 0, 0])
    np.testing.assert_allclose(smoothed_online_pvalues(s, np.ones(3)), [1, 1, 1])


@pytest.mark.parametrize("seed", SEEDS)
def test_smoothed_uniform_on_iid_continuous(seed):
    rng = np.random.default_rng(seed)
    p = smoothed_online_pvalues(rng.standard_normal(10_000), rng.uniform(size=10_000))
    assert stats.kstest(p, "uniform").pvalue > 0.01


@pytest.mark.parametrize("seed", SEEDS)
def test_smoothed_uniform_with_heavy_ties(seed):
    rng = np.random.default_rng(seed)
    scores = rng.integers(0, 4, 10_000).astype(float)   # only 4 distinct values
    p = smoothed_online_pvalues(scores, rng.uniform(size=10_000))
    assert stats.kstest(p, "uniform").pvalue > 0.01


def test_online_past_unchanged_by_future():
    rng = np.random.default_rng(7)
    s, th = rng.standard_normal(500), rng.uniform(size=500)
    a = smoothed_online_pvalues(s, th)
    s2 = s.copy()
    s2[300:] = rng.standard_normal(200) * 100
    b = smoothed_online_pvalues(s2, th)
    np.testing.assert_array_equal(a[:300], b[:300])


def test_streams_are_independent():
    st = OnlineStreams()
    assert st.update("u1", 1.0, 0.5) == (0.5, 1)
    assert st.update("u2", 9.0, 0.5) == (0.5, 1)
    p, n = st.update("u1", 0.0, 0.5)          # 1 greater, 1 equal (itself)
    assert n == 2 and p == pytest.approx((1 + 0.5) / 2)
    assert st.count("u1") == 2 and st.count("u2") == 1 and st.count("x") == 0
    with pytest.raises(ValueError):
        st.update("u1", float("nan"), 0.5)


def test_split_pvalues_hand():
    p = split_pvalues(np.array([4.0, 1.0, 3.0, 2.0]), np.array([2.5, 4.0, 0.0, 5.0]))
    np.testing.assert_allclose(p, [3 / 5, 2 / 5, 5 / 5, 1 / 5])
    assert split_pvalues(np.array([]), np.array([1.0]))[0] == 1.0


def test_split_pvalues_super_uniform():
    rng = np.random.default_rng(3)
    n_cal, trials = 49, 20_000
    s = rng.standard_normal((trials, n_cal + 1))
    p = np.array([split_pvalues(row[:-1], row[-1:])[0] for row in s])
    for a in (0.02, 0.05, 0.1):
        se = np.sqrt(a * (1 - a) / trials)
        assert np.mean(p <= a) <= a + 3 * se
        # exact level for continuous scores: floor(a (n+1)) / (n+1)
        assert np.mean(p <= a) == pytest.approx(np.floor(a * (n_cal + 1)) / (n_cal + 1), abs=4 * se)


def test_conformal_upper_offset():
    cal = np.arange(1.0, 10.0)                     # n = 9
    assert conformal_upper_offset(cal, 0.1) == 9.0  # k = ceil(0.9 * 10) = 9
    assert conformal_upper_offset(cal, 0.2) == 8.0  # k = ceil(0.8 * 10) = 8
    assert conformal_upper_offset(cal, 0.05) == float("inf")


def test_upper_bound_coverage_simulated():
    rng = np.random.default_rng(11)
    hits = []
    for _ in range(4000):
        s = rng.standard_normal(100)
        hits.append(s[-1] <= conformal_upper_offset(s[:-1], 0.1))
    assert np.mean(hits) >= 0.9 - 3 * np.sqrt(0.09 / 4000)
