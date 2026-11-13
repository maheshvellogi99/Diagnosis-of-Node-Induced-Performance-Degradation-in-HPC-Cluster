"""Conformal p-values: hand examples, uniformity, online property (PLAN.md §7.5)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from hpcdiag.stats.conformal import (
    OnlineStreams,
    conformal_upper_offset,
    sliding_window_pvalues,
    smoothed_online_pvalues,
    smoothed_split_pvalues,
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


# ---------------------------------------------------------------- D23 smoothed split p-values


def test_smoothed_split_hand():
    cal = np.array([1.0, 2.0, 2.0, 3.0])
    # S=2: 1 greater, 2 equal -> (1 + .5 * 3) / 5 ; S=0: 4 greater -> (4 + .5) / 5 ; S=9 -> .5/5
    p = smoothed_split_pvalues(cal, np.array([2.0, 0.0, 9.0]), np.full(3, 0.5))
    np.testing.assert_allclose(p, [2.5 / 5, 4.5 / 5, 0.5 / 5])


@pytest.mark.parametrize("seed", SEEDS)
def test_smoothed_split_uniform_with_heavy_ties(seed):
    rng = np.random.default_rng(seed)
    n_cal, trials = 40, 10_000
    s = rng.integers(0, 3, (trials, n_cal + 1)).astype(float)   # 3 distinct values
    th = rng.uniform(size=trials)
    p = np.array([smoothed_split_pvalues(r[:-1], r[-1:], th[i:i + 1])[0]
                  for i, r in enumerate(s)])
    assert stats.kstest(p, "uniform").pvalue > 0.01
    # the unsmoothed version is far from uniform on the same draws (the reason for D23)
    q = np.array([split_pvalues(r[:-1], r[-1:])[0] for r in s])
    assert stats.kstest(q, "uniform").statistic > 0.1


# ---------------------------------------------------------------- D22 sliding-window (Mode P)


def _brute_sliding(s, e, th, w, min_ref, keys=None):
    n = len(s)
    k = np.zeros(n) if keys is None else keys
    p = np.full(n, np.nan)
    for j in range(n):
        ref = [i for i in range(j) if k[i] == k[j] and e[i] >= e[j] - w]
        if len(ref) >= min_ref:
            r = s[ref]
            p[j] = (np.sum(r > s[j]) + th[j] * (1 + np.sum(r == s[j]))) / (len(r) + 1)
    return p


def _stream(n, seed, ties=False):
    rng = np.random.default_rng(seed)
    e = np.sort(rng.integers(0, 50_000, n))           # many equal end times
    s = rng.integers(0, 5, n).astype(float) if ties else rng.standard_normal(n)
    return s, e, rng.uniform(size=n)


@pytest.mark.parametrize("ties", [False, True])
def test_sliding_equals_brute_force(ties):
    s, e, th = _stream(2000, 4, ties)
    p, nref = sliding_window_pvalues(s, e, th, 3000, 30)
    np.testing.assert_allclose(p, _brute_sliding(s, e, th, 3000, 30), equal_nan=True)
    keys = np.random.default_rng(1).integers(0, 3, 2000)
    pk, _ = sliding_window_pvalues(s, e, th, 3000, 30, keys)
    np.testing.assert_allclose(pk, _brute_sliding(s, e, th, 3000, 30, keys), equal_nan=True)


def test_sliding_reference_excludes_later_and_tied_larger_rowid():
    """R_j never contains a job that ends after j, nor a job with the same end time that comes
    later in D14 order (larger row_id): changing such jobs leaves p_j unchanged."""
    s, e, th = _stream(600, 5)
    e[300:310] = e[300]                              # a block of equal end times
    p, _ = sliding_window_pvalues(s, e, th, 10**9, 30)
    s2 = s.copy()
    s2[305:] = 1e6                                   # later-or-tied-larger-row_id jobs change
    p2, _ = sliding_window_pvalues(s2, e, th, 10**9, 30)
    np.testing.assert_array_equal(p[:305], p2[:305])


def test_sliding_warm_up_nan_and_window_drop():
    s = np.arange(100.0)
    e = np.arange(100) * 10
    p, nref = sliding_window_pvalues(s, e, np.full(100, 0.5), 10**9, 30)
    assert np.isnan(p[:30]).all() and np.isfinite(p[30:]).all()
    assert nref.tolist() == list(range(100))
    # window 95: jobs ending >= e_j - 95 -> at most the 9 previous jobs; min_ref 5
    p2, nref2 = sliding_window_pvalues(s, e, np.full(100, 0.5), 95, 5)
    assert nref2.max() == 9 and np.isfinite(p2[9:]).all() and np.isnan(p2[:5]).all()
    with pytest.raises(ValueError):
        sliding_window_pvalues(s, e[::-1], np.full(100, 0.5), 10, 5)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("ties", [False, True])
def test_sliding_uniform_on_iid(seed, ties):
    s, e, th = _stream(10_000 + 30, seed, ties)
    p, _ = sliding_window_pvalues(s, e, th, 5000, 30)
    p = p[np.isfinite(p)]
    assert len(p) >= 9000
    assert stats.kstest(p, "uniform").pvalue > 0.01


def test_sliding_query_scores_shift_matches_brute_force():
    s, e, th = _stream(800, 9)
    q = s + 0.7
    p, _ = sliding_window_pvalues(s, e, th, 3000, 30, query_scores=q)
    exp = np.full(len(s), np.nan)
    for j in range(len(s)):
        ref = [i for i in range(j) if e[i] >= e[j] - 3000]
        if len(ref) >= 30:
            r = s[ref]
            exp[j] = (np.sum(r > q[j]) + th[j] * (1 + np.sum(r == q[j]))) / (len(r) + 1)
    np.testing.assert_allclose(p, exp, equal_nan=True)
    p0, _ = sliding_window_pvalues(s, e, th, 3000, 30)
    assert np.nanmedian(p) < np.nanmedian(p0)            # a positive shift lowers p-values
