"""D109 restart mixture for probe evidence (PLAN.md §9 Phase 5, 5.3).

P(t) = sum_{k <= N} w_k prod_{i=k..N} F_i + (1 - J)^N,  w_k = J (1 - J)^(k - 1),  P(0) = 1.
Recursion: A <- F (A + J R), R <- (1 - J) R, P = A + R, A_0 = 0, R_0 = 1.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpcdiag.stats.restart import restart_init, restart_logP, restart_logP_seq, restart_step


def explicit_P(F: np.ndarray, J: float) -> np.ndarray:
    """Reference: the explicit sum after each of the N = 1..len(F) investigations (no recursion)."""
    out = []
    for N in range(1, len(F) + 1):
        s = sum(J * (1 - J) ** (k - 1) * np.prod(F[k - 1 : N]) for k in range(1, N + 1))
        out.append(s + (1 - J) ** N)
    return np.array(out)


def test_P0_is_one():
    logA, logR = restart_init(3)
    assert np.all(restart_logP(logA, logR) == 0.0)


@pytest.mark.parametrize("J", [0.01, 0.1, 0.5, 1.0])
def test_recursion_equals_explicit_sum(J):
    rng = np.random.default_rng(0)
    F = np.exp(rng.normal(0, 1.5, size=40))
    got = np.exp(restart_logP_seq(np.log(F), J))
    assert np.allclose(got, explicit_P(F, J), rtol=1e-10, atol=0)


@pytest.mark.parametrize("J", [0.01, 0.3, 1.0])
def test_one_step_fair_when_f_integrates_to_one(J):
    """From any state (A, R), E[P'] = P exactly when E[F] = 1; here F takes the values of a bet on
    bins with probabilities equal to the bin widths (p uniform)."""
    edges = np.array([0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1])
    width = np.diff(edges)
    g = np.array([40.0, 30, 20, 10, 6, 3, 1.5, 0.8, 0.4, 0.2])
    g = g / (g @ width)  # density integrating to 1
    f = 0.1 + 0.9 * g
    assert abs(f @ width - 1) < 1e-12
    rng = np.random.default_rng(1)
    for _ in range(20):
        logA, logR = np.log(rng.uniform(0, 5)), np.log(rng.uniform(0, 1))
        P = np.exp(np.logaddexp(logA, logR))
        nxt = [restart_step(logA, logR, np.log(fb), J) for fb in f]
        EP = sum(w * np.exp(restart_logP(a, r)) for w, (a, r) in zip(width, nxt, strict=True))
        assert abs(EP - P) < 1e-12 * max(1.0, P)


def test_supermartingale_when_f_subfair():
    """E[F] < 1 gives E[P'] < P (the mixture is a test supermartingale)."""
    logA, logR = np.log(0.7), np.log(0.4)
    P = 1.1
    vals = [0.5, 1.2]  # mean 0.85
    EP = np.mean([np.exp(restart_logP(*restart_step(logA, logR, np.log(v), 0.01))) for v in vals])
    assert EP < P


def test_J_one_is_the_old_formula_exactly():
    """J = 1: P = prod of all F, i.e. log P = running sum of log F, bit for bit."""
    rng = np.random.default_rng(2)
    logF = rng.normal(-1, 2, size=200)
    got = restart_logP_seq(logF, 1.0)
    ref, s = [], 0.0
    for x in logF:
        s += x
        ref.append(s)
    assert np.array_equal(got, np.array(ref))


def test_debt_is_bounded():
    """After many healthy investigations (F < 1), P stays above (1 - J)^N + ..., and one later
    investigation with a large F lifts P to at least w_{N+1} F; with J = 1 the debt is unbounded."""
    J = 0.01
    logF = np.full(25, np.log(0.12))  # about -2.1 nats each, as measured before onset
    logA, logR = restart_init(1)
    for x in logF:
        logA, logR = restart_step(logA, logR, x, J)
    big = np.log(400.0)
    a2, r2 = restart_step(logA, logR, big, J)
    w_next = J * (1 - J) ** 25
    assert restart_logP(a2, r2)[0] >= np.log(w_next * 400.0) - 1e-12
    assert restart_logP_seq(np.append(logF, big), 1.0)[-1] < np.log(w_next * 400.0)


def test_vectorised_matches_scalar():
    rng = np.random.default_rng(3)
    logF = rng.normal(0, 1, size=(30, 4))
    logA, logR = restart_init(4)
    for row in logF:
        logA, logR = restart_step(logA, logR, row, 0.01)
    for j in range(4):
        assert np.isclose(restart_logP(logA, logR)[j], restart_logP_seq(logF[:, j], 0.01)[-1])


def test_bad_J_rejected():
    with pytest.raises(ValueError):
        restart_step(*restart_init(1), np.zeros(1), 0.0)
    with pytest.raises(ValueError):
        restart_step(*restart_init(1), np.zeros(1), 1.5)
