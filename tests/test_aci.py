"""ACI (F19): hand example and long-run alarm rate on a drifting stream."""

from __future__ import annotations

import numpy as np
import pytest

from hpcdiag.stats.aci import aci_alarms


def test_aci_hand_example():
    # a1=.05: .01 alarm -> a2 = .05 + .1(.05-1) = -.045; .5 no alarm -> a3 = -.045 + .005 = -.04
    alarm, used = aci_alarms(np.array([0.01, 0.5, 0.04]), np.array(["g"] * 3), 0.05, 0.1)
    assert alarm.tolist() == [True, False, False]
    np.testing.assert_allclose(used, [0.05, -0.045, -0.04])


def test_aci_groups_have_separate_state():
    alarm, used = aci_alarms(np.array([0.01, 0.01, 0.5]), np.array(["a", "b", "a"]), 0.05, 0.1)
    np.testing.assert_allclose(used, [0.05, 0.05, -0.045])


@pytest.mark.parametrize("gamma, T", [(0.001, 200_000), (0.005, 50_000), (0.01, 30_000)])
def test_aci_long_run_rate_under_drift(gamma, T):
    """Gibbs & Candès Prop. 4.1: |mean err - alpha| <= (max(alpha_1, 1 - alpha_1) + gamma)/(gamma T)
    for ANY sequence, here p-values that drift from too small towards uniform."""
    rng = np.random.default_rng(5)
    a = np.linspace(0.3, 1.0, T)  # Beta(a, 1), a < 1: p-values too small, drifting to U(0, 1)
    p = rng.beta(a, 1.0)
    alpha = 0.05
    alarm, _ = aci_alarms(p, np.zeros(T, dtype=int), alpha, gamma)
    bound = (max(alpha, 1 - alpha) + gamma) / (gamma * T)
    assert abs(alarm.mean() - alpha) <= bound
    # the fixed rule is far off on this stream (sanity: drift is real)
    assert abs(np.mean(p <= alpha) - alpha) > 3 * bound


def test_aci_shape_mismatch():
    with pytest.raises(ValueError):
        aci_alarms(np.array([0.1, 0.2]), np.array(["g"]), 0.05, 0.01)
