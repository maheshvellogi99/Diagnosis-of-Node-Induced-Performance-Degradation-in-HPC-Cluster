"""Phase 2B post-processing: Gate 2-L2 logic, Mode P variants, node check (synthetic)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hpcdiag.layer1.post_scores import gate2_l2, mode_p, node_check
from hpcdiag.stats.conformal import sliding_window_pvalues

CFG = {
    "alarms": {"alpha_job": 0.05, "aci_gamma_default": 0.005, "alphas": [0.01, 0.05, 0.1]},
    "gate2_l2": {"ks_max": 0.02, "fa_tol": 0.01, "p_le_1e3_max": 0.002, "aci_tol": 0.005},
    "mode_p": {"min_ref": 30},
}


def _m(ks=0.01, fa=(0.01, 0.05, 0.1), p3=0.001, mf=0.8, p95=1.1, aci=0.05):
    return {"ks": ks, "fa": {"0.01": fa[0], "0.05": fa[1], "0.1": fa[2]},
            "tail": {"p_le": {"0.001": p3}, "mean_f": mf, "null_mean_f": {"p95": p95},
                     "mean_f_le_null_p95": mf <= p95},
            "aci_global": {"0.005": aci}}


@pytest.mark.parametrize("kw, ok", [
    ({}, True), ({"ks": 0.021}, False), ({"fa": (0.021, 0.05, 0.1)}, False),
    ({"p3": 0.0021}, False), ({"mf": 1.2}, False), ({"aci": 0.056}, False),
    ({"fa": (0.0, 0.06, 0.11)}, True), ({"aci": 0.0451}, True)])
def test_gate2_l2(kw, ok):
    assert gate2_l2(_m(**kw), CFG)["pass"] is ok


def _df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    end = pd.Timestamp("2020-01-01") + pd.to_timedelta(np.sort(rng.integers(0, 30 * 86400, n)),
                                                       unit="s")
    return pd.DataFrame({"row_id": np.arange(n), "end_time": end,
                         "score": rng.standard_normal(n),
                         "user": rng.choice(["a", "b"], n, p=[.8, .2]),
                         "dec_qhat": rng.integers(0, 10, n)})


def test_mode_p_global_and_user_first():
    df = _df()
    th = np.full(len(df), 0.5)
    pg, ng = mode_p(df, CFG, 24 * 7, "global", th)
    e = df["end_time"].to_numpy().astype("datetime64[ns]").astype("int64")
    ref, _ = sliding_window_pvalues(df["score"].to_numpy(), e, th, 7 * 24 * 3600 * 10**9, 30)
    np.testing.assert_allclose(pg, ref, equal_nan=True)
    pf, _ = mode_p(df, CFG, 24 * 7, "user_first", th)
    pu, _ = sliding_window_pvalues(df["score"].to_numpy(), e, th, 7 * 24 * 3600 * 10**9, 30,
                                   df["user"].to_numpy())
    use_u = np.isfinite(pu)
    np.testing.assert_allclose(pf[use_u], pu[use_u])
    np.testing.assert_allclose(pf[~use_u], pg[~use_u], equal_nan=True)


def test_node_check_dispersion_one_under_binomial(tmp_path):
    rng = np.random.default_rng(1)
    n_jobs, n_nodes = 40_000, 100
    jn = pd.DataFrame({"row_id": np.arange(n_jobs), "node_id": rng.integers(0, n_nodes, n_jobs)})
    jn.to_parquet(tmp_path / "jn.parquet")
    df = pd.DataFrame({"row_id": np.arange(n_jobs), "user": rng.choice(list("abcdef"), n_jobs)})
    p = rng.uniform(size=n_jobs)
    cfg = {"node_check": {"min_jobs": 50, "alpha": 0.05, "job_nodes": str(tmp_path / "jn.parquet")}}
    out = node_check(df, p, cfg)
    assert out["n_nodes"] == n_nodes
    assert 0.6 < out["dispersion_ratio"] < 1.5          # single-node jobs, no node effect
    assert "NOT valid tests" in out["warning"]


def test_ks_location_hand_and_matches_scipy():
    from scipy import stats as sst

    from hpcdiag.eval.conformal_metrics import ks_location
    # 4 points all at 0.1: F_n jumps to 1 at 0.1 -> D+ = 1 - 0.1 = 0.9, sign +
    r = ks_location(np.array([0.1, 0.1, 0.1, 0.1]))
    assert r["sign"] == "+" and r["p_at_max"] == pytest.approx(0.1)
    assert r["ks"] == pytest.approx(0.9)
    r = ks_location(np.array([0.9, 0.95]))                  # too few small p -> sign -
    assert r["sign"] == "-" and r["p_at_max"] == pytest.approx(0.9)
    p = np.random.default_rng(0).beta(0.9, 1, 5000)
    assert ks_location(p)["ks"] == pytest.approx(sst.kstest(p, "uniform").statistic)
