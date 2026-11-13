"""Calibrators integrate to 1 (PLAN §7.5) and the D24 tail check behaves under the null."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate

from hpcdiag.stats.calibrators import (
    KAPPAS,
    mixture_calibrator,
    null_mean_f,
    power_calibrator,
    tail_checks,
)


@pytest.mark.parametrize("kappa", KAPPAS)
def test_power_calibrator_integrates_to_one(kappa):
    # integrate kappa * x^(kappa-1) using quad's algebraic weight (exact singularity handling)
    val, _ = integrate.quad(lambda x: kappa, 0, 1, weight="alg", wvar=(kappa - 1, 0))
    assert abs(val - 1) < 1e-6


def test_mixture_integrates_to_one_and_hand_value():
    vals = [
        integrate.quad(lambda x, k=k: k, 0, 1, weight="alg", wvar=(k - 1, 0))[0] for k in KAPPAS
    ]
    assert abs(np.mean(vals) - 1) < 1e-6
    # f(1) = mean(kappa) ; f(0.01) for kappa = .5 is .5 * .01^-.5 = 5
    assert mixture_calibrator(np.array([1.0]))[0] == pytest.approx(np.mean(KAPPAS))
    assert power_calibrator(np.array([0.01]), 0.5)[0] == pytest.approx(5.0)
    with pytest.raises(ValueError):
        power_calibrator(np.array([0.5]), 1.0)


def test_null_mean_f_typically_below_one_and_tail_check_on_uniform():
    sims = null_mean_f(20_000, 200, seed=1)
    assert np.median(sims) < 1.0  # heavy right tail of f(U)
    rng = np.random.default_rng(9)
    out = tail_checks(rng.uniform(size=20_000), n_sims=200, seed=2)
    assert out["mean_f"] <= out["null_mean_f"]["p99"]
    assert out["p_le"]["0.001"] == pytest.approx(0.001, abs=0.001)


def test_tail_check_flags_heavy_tail():
    rng = np.random.default_rng(3)
    p = rng.uniform(size=20_000)
    p[:200] = 1e-6  # 1% extreme p-values
    out = tail_checks(p, n_sims=200, seed=2)
    assert not out["mean_f_le_null_p95"]
    assert out["p_le"]["0.0001"] >= 0.01
