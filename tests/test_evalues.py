"""E-processes and e-BH (PLAN.md §7.5; D42, D49, D50)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate

from hpcdiag.stats.calibrators import KAPPAS
from hpcdiag.stats.conformal import sliding_window_pvalues
from hpcdiag.stats.ebh import bh, ebh, ebh_log
from hpcdiag.stats.evalues import (
    jumper_logM,
    log_bet,
    node_logM_by_rank,
    plain_mixture_logM,
)

ALPHA = 0.1


@pytest.mark.parametrize("w", [1.0, 0.5, 0.1])
@pytest.mark.parametrize("kappa", KAPPAS)
def test_bet_integrates_to_one(w, kappa):
    # int over p of 1 + w(f(p) - 1) = 1 + w(int f - 1) = 1; f via quad's 'alg' weight
    fint = integrate.quad(lambda x: kappa, 0, 1, weight="alg", wvar=(kappa - 1, 0))[0]
    assert abs((1 - w) + w * fint - 1) < 1e-6


def test_w_one_gives_plain_product():
    p = np.array([0.5, 0.01, 0.9])
    lb = log_bet(p, 1.0)
    k = KAPPAS[0]
    np.testing.assert_allclose(lb[:, 0], np.log(k * p ** (k - 1)))
    lm = plain_mixture_logM(lb)
    prods = np.cumprod(np.column_stack([k * p ** (k - 1) for k in KAPPAS]), axis=0)
    np.testing.assert_allclose(lm, np.log(prods.mean(axis=1)))


def test_jumper_with_J0_is_mixture_with_no_bet_state():
    p = np.random.default_rng(0).uniform(size=50)
    lb = log_bet(p, 0.3)
    # J = 0: capital never moves between states -> (1 + sum of products) / (K + 1)
    lm = jumper_logM(lb, 0.0)
    prods = np.exp(np.cumsum(lb, axis=0))
    np.testing.assert_allclose(lm, np.log((1 + prods.sum(axis=1)) / (len(KAPPAS) + 1)))


def test_vectorised_by_rank_matches_reference():
    rng = np.random.default_rng(1)
    n = 3000
    node = rng.integers(0, 40, n)
    lb = log_bet(rng.uniform(size=n), rng.uniform(0.05, 1, n))
    for J in (None, 0.01):
        got = node_logM_by_rank(node, lb, J)
        for nd in (0, 7, 39):
            m = node == nd
            ref = plain_mixture_logM(lb[m]) if J is None else jumper_logM(lb[m], J)
            np.testing.assert_allclose(got[m], ref, rtol=1e-10, atol=1e-10)


def _one_step_expectation(logC, w, J):
    """E_p[total capital after one jumper step], p ~ U(0,1), by quadrature with p = u^20
    (dp = 20 u^19 du) so the p^(kappa - 1) singularities (kappa >= 0.05) become bounded."""
    from scipy import integrate as itg

    from hpcdiag.stats.evalues import jumper_step, log_total

    def h(u):
        p = u ** 20
        lg = np.concatenate([[0.0], log_bet(np.array([max(p, 1e-300)]), w)[0]])[None, :]
        return np.exp(log_total(jumper_step(logC[None, :], lg, J))[0]) * 20 * u ** 19

    return itg.quad(h, 0, 1, limit=400, epsabs=1e-12, epsrel=1e-10)[0]


@pytest.mark.parametrize("J", [0.0, 0.01, 0.3])
@pytest.mark.parametrize("w", [1.0, 0.4])
def test_one_step_is_fair_by_quadrature(J, w):
    """Martingale property checked exactly (PLAN 7.5): E[M_t | past] = M_{t-1} for the jumper
    (J = 0 includes the plain mixture plus a no-bet state)."""
    rng = np.random.default_rng(int(J * 100) + int(w * 10))
    logC = np.log(rng.dirichlet(np.ones(len(KAPPAS) + 1)) * 3.7)
    before = np.log(np.exp(logC).sum())
    assert abs(np.log(_one_step_expectation(logC, w, J)) - before) < 1e-6


def test_mc_mean_finite_variance_bet():
    """Monte-Carlo mean of M_T = 1 within 3 SE, on a bet with FINITE variance (kappa = 0.75,
    E f^2 = kappa^2/(2 kappa - 1) = 1.125). The project's kappas (<= 0.5) have infinite variance,
    so a sample-mean test on them is not meaningful (reported in the Phase 4 report)."""
    rng = np.random.default_rng(6)
    sims, T = 20_000, 20
    p = rng.uniform(size=(sims, T))
    node = np.repeat(np.arange(sims), T)
    for J in (None, 0.01):
        lb = log_bet(p.ravel(), 0.5, kappas=(0.75,))
        MT = np.exp(node_logM_by_rank(node, lb, J).reshape(sims, T)[:, -1])
        assert abs(MT.mean() - 1) <= 3 * MT.std(ddof=1) / np.sqrt(sims)


@pytest.mark.parametrize("J", [None, 0.01])
def test_ville_on_null_streams(J):
    """P(max_t M_t >= 1/alpha) <= alpha + Monte-Carlo error, project kappas, mixed weights."""
    rng = np.random.default_rng(2)
    sims, T = 10_000, 200
    p = rng.uniform(size=(sims, T))
    w = rng.choice([1.0, 0.5, 0.25], size=(sims, T))
    node = np.repeat(np.arange(sims), T)
    lm = node_logM_by_rank(node, log_bet(p.ravel(), w.ravel()), J).reshape(sims, T)
    hit = np.mean(lm.max(axis=1) >= np.log(1 / ALPHA))
    assert hit <= ALPHA + 3 * np.sqrt(ALPHA * (1 - ALPHA) / sims)


def test_tail_mode_p_null_streams():
    """D42: p-values from Mode P on IID scores; P(max M >= 1/alpha) <= alpha + MC error."""
    rng = np.random.default_rng(3)
    sims, T = 2000, 300
    hits = 0
    for _ in range(sims):
        s = rng.standard_normal(T + 30)
        e = np.arange(T + 30) * 10
        pp, _ = sliding_window_pvalues(s, e, rng.uniform(size=T + 30), 10**9, 30)
        pp = pp[np.isfinite(pp)]
        lm = node_logM_by_rank(np.zeros(len(pp), int), log_bet(pp, 1.0), 0.01)
        hits += lm.max() >= np.log(1 / ALPHA)
    assert hits / sims <= ALPHA + 3 * np.sqrt(ALPHA * (1 - ALPHA) / sims)


def test_ebh_hand_example_and_bruteforce():
    e = np.array([50.0, 30.0, 12.0, 1.0, 0.5])
    # m = 5, alpha = 0.1: k e_[k]/m >= 10 -> k=1: 10 ok; k=2: 12 ok; k=3: 7.2 no -> k* = 2
    assert ebh(e, 0.1).tolist() == [True, True, False, False, False]
    # fixed larger m (D50): m = 10 -> k=1: 5 no; k=2: 6 no -> nothing
    assert not ebh(e, 0.1, m=10).any()
    rng = np.random.default_rng(4)
    for _ in range(300):
        ev = rng.exponential(5, 8)
        m = 8
        srt = np.sort(ev)[::-1]
        ks = [k for k in range(1, m + 1) if k * srt[k - 1] / m >= 1 / 0.1]
        exp = max(ks) if ks else 0
        got = ebh(ev, 0.1)
        assert got.sum() == exp
        if exp:
            assert ev[got].min() >= srt[exp - 1]
        np.testing.assert_array_equal(ebh_log(np.log(ev), 0.1), got)


def test_ebh_fdr_on_null_evalues():
    rng = np.random.default_rng(5)
    m, sims = 50, 4000
    fdp = []
    for _ in range(sims):
        e = 0.5 * rng.uniform(size=m) ** (-0.5)    # kappa = 0.5 calibrator of U: E = 1
        fdp.append(float(ebh(e, ALPHA).any()))      # all nulls: FDP = 1{any rejection}
    assert np.mean(fdp) <= ALPHA + 3 * np.sqrt(ALPHA * (1 - ALPHA) / sims)


def test_bh_hand():
    p = np.array([0.001, 0.02, 0.03, 0.5])
    # alpha 0.1, m 4: thresholds .025 .05 .075 .1 -> p(3) = .03 <= .075 -> k* = 3
    assert bh(p, 0.1).tolist() == [True, True, True, False]
