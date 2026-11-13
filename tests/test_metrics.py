"""Hand-computed checks of PLAN.md §10 metrics and the Gate 1 qualitative rule."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from hpcdiag.eval import metrics as M

LABELS = ["0-1h", "1-6h", "6-24h", "1-2d", ">2d"]
EDGES = [0, 1, 6, 24, 48]


def test_rmse_mae_signed_hours():
    y, yhat = np.array([3600.0, 7200.0]), np.array([0.0, 7200.0])
    assert M.rmse_h(y, yhat) == pytest.approx(1 / math.sqrt(2))  # sqrt(3600^2/2)/3600
    assert M.mae_h(y, yhat) == pytest.approx(0.5)
    assert M.mean_signed_error_h(y, yhat) == pytest.approx(-0.5)


def test_a_geo_unweighted_and_weighted():
    assert M.a_geo([100, 200], [50, 400], [1, 1]) == pytest.approx(0.5)
    # a = [1, 0.5], c = [3*100, 1*100] -> 0.5 ** (100/400)
    assert M.a_geo([100, 100], [100, 50], [3, 1]) == pytest.approx(0.5**0.25)
    assert M.a_geo([10, 20], [10, 20], [1, 5]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        M.a_geo([10], [0], [1])


def test_under_rate_strict():
    assert M.under_rate([10, 20, 30], [5, 20, 40]) == pytest.approx(1 / 3)


def test_tare_metrics_filter_and_clip():
    # y=20 dropped (<= 30 s); yhat clipped to >= 1: [0.5 -> 1, 400]
    out = M.tare_metrics([20, 100, 200], [-5, 0.5, 400], [1, 1, 1], 30, 1.0)
    a = (1 / 100) ** (100 / 300) * 0.5 ** (200 / 300)
    assert out["n"] == 2
    assert out["a_geo"] == pytest.approx(a)
    assert out["under_rate"] == pytest.approx(0.5)


def test_bins_edges():
    y = np.array([0, 3599, 3600, 6 * 3600, 24 * 3600 - 1, 48 * 3600, 1e7])
    got = M.bin_labels_for(y, EDGES, LABELS).tolist()
    assert got == ["0-1h", "0-1h", "1-6h", "6-24h", "6-24h", ">2d", ">2d"]


def test_log_ratio_uses_max_yhat_1():
    assert M.log_ratio([100], [0.2])[0] == pytest.approx(math.log(1 / 100))
    assert M.log_ratio([100], [200])[0] == pytest.approx(math.log(2))


def test_per_bin_table_hand_example():
    h = 3600.0
    y = np.array([10.0, 1800.0, 2 * h, 50 * h, 60 * h])
    yhat = np.array([20.0, 900.0, 4 * h, 25 * h, 30 * h])
    t = M.per_bin_table(y, yhat, EDGES, LABELS, 30).set_index("bin")
    assert t.loc["0-1h", "n"] == 2 and t.loc["0-1h", "n_r"] == 1  # y=10 s excluded from r
    assert t.loc["0-1h", "median_r"] == pytest.approx(math.log(0.5))
    assert t.loc["1-6h", "median_r"] == pytest.approx(math.log(2))
    assert t.loc[">2d", "median_r"] == pytest.approx(math.log(0.5))
    assert t.loc[">2d", "mean_signed_error_h"] == pytest.approx(-27.5)
    assert t.loc[">2d", "under_rate_r"] == 1.0
    assert t.loc["6-24h", "n"] == 0 and np.isnan(t.loc["6-24h", "median_r"])


def _table(meds):
    return pd.DataFrame({"bin": LABELS, "median_r": meds})


def test_qualitative_gate():
    assert M.qualitative_gate(_table([-0.1, 0.0, 0.1, -0.2, -0.5]))["pass"] is True
    # negative but not the lowest
    assert M.qualitative_gate(_table([-0.9, 0.0, 0.1, -0.2, -0.5]))["pass"] is False
    # lowest but not negative
    assert M.qualitative_gate(_table([0.3, 0.2, 0.4, 0.5, 0.1]))["pass"] is False
    # empty bins (NaN) are ignored when finding the minimum
    assert M.qualitative_gate(_table([np.nan, 0.0, 0.1, np.nan, -0.5]))["pass"] is True
