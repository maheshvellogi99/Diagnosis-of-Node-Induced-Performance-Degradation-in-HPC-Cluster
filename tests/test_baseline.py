"""Phase 1 baseline: defaults only, deterministic, leakage guard, duplicate detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import xgboost

from hpcdiag.layer1 import data as D
from hpcdiag.layer1.baseline import base_score, config_fingerprint, make_model, run_split
from hpcdiag.layer1.splits import TimeIndex

NUM = ["walltime_req_s", "procs_req", "mem_req", "nodes_req", "gpus_req"]
CAT = ["user", "partition"]


def _df(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2020-01-01")
    submit = t0 + pd.to_timedelta(rng.integers(0, 150 * 86400, n), unit="s")
    wall = rng.choice([600.0, 3600.0, 86400.0], n)
    rt = np.minimum(wall, rng.exponential(wall / 3))
    return pd.DataFrame(
        {
            "row_id": np.arange(n),
            "submit_time": submit,
            "end_time": submit + pd.to_timedelta(rt.astype(int) + 1, unit="s"),
            "run_time_s": rt,
            "walltime_req_s": wall,
            "procs_req": rng.integers(1, 64, n),
            "mem_req": rng.choice([0.0, 1000.0], n),
            "nodes_req": rng.integers(1, 4, n),
            "gpus_req": rng.integers(0, 2, n),
            "user": rng.choice(list("abcdefg"), n),
            "partition": rng.choice(["short", "std"], n),
        }
    )


def test_make_model_uses_xgboost_defaults():
    assert make_model().get_params() == xgboost.XGBRegressor().get_params()
    assert make_model({}).get_params() == xgboost.XGBRegressor().get_params()


def test_run_split_deterministic_and_config_saved():
    df = _df()
    idx = TimeIndex.build(df["end_time"], df["submit_time"])
    t = pd.Timestamp("2020-04-15")
    r1 = run_split(df, idx, t, 100, 1, NUM, CAT, "run_time_s")
    r2 = run_split(df, idx, t, 100, 1, NUM, CAT, "run_time_s")
    assert r1 is not None and r1.test_rows > 0
    np.testing.assert_array_equal(r1.predictions["yhat"], r2.predictions["yhat"])
    assert r1.train_end_max < t
    assert "learner" in r1.config  # booster save_config() content
    assert set(r1.predictions.columns) == {"row_id", "y", "yhat", "nodes_req"}


def test_config_fingerprint_ignores_only_base_score():
    df = _df()
    idx = TimeIndex.build(df["end_time"], df["submit_time"])
    a = run_split(df, idx, pd.Timestamp("2020-04-15"), 100, 1, NUM, CAT, "run_time_s").config
    b = run_split(df, idx, pd.Timestamp("2020-05-15"), 100, 1, NUM, CAT, "run_time_s").config
    assert base_score(a) != base_score(b)  # estimated from data in xgboost >= 2
    assert config_fingerprint(a) == config_fingerprint(b)
    c = run_split(
        df, idx, pd.Timestamp("2020-04-15"), 100, 1, NUM, CAT, "run_time_s", params={"max_depth": 3}
    ).config
    assert config_fingerprint(a) != config_fingerprint(c)


def test_non_default_param_changes_predictions():
    """Guard: params given to run_split reach the fitted model (Phase 1 sens_approx check)."""
    df = _df()
    idx = TimeIndex.build(df["end_time"], df["submit_time"])
    t = pd.Timestamp("2020-04-15")
    a = run_split(df, idx, t, 100, 1, NUM, CAT, "run_time_s")
    b = run_split(df, idx, t, 100, 1, NUM, CAT, "run_time_s", params={"max_depth": 2})
    c = run_split(df, idx, t, 100, 1, NUM, CAT, "run_time_s", params={"tree_method": "exact"})
    assert not np.array_equal(a.predictions["yhat"], b.predictions["yhat"])
    assert not np.array_equal(a.predictions["yhat"], c.predictions["yhat"])


def test_run_split_empty_returns_none():
    df = _df()
    idx = TimeIndex.build(df["end_time"], df["submit_time"])
    assert run_split(df, idx, pd.Timestamp("2019-01-01"), 100, 1, NUM, CAT, "run_time_s") is None


def test_exact_duplicate_row_ids(tmp_path):
    t = pd.Timestamp("2020-01-01")
    raw = pd.DataFrame(
        {
            "job_id": [1, 1, 1, 2, 2, 3],
            "name": ["a", "a", "a", None, None, "c"],
            "submit_time": [t] * 6,
            "run_time": [5.0, 5.0, 6.0, 7.0, 7.0, 8.0],
        }
    )
    p = tmp_path / "raw.parquet"
    pq.write_table(pa.Table.from_pandas(raw, preserve_index=False), p)
    # row 1 copies row 0; row 4 copies row 3 (nulls compare equal); row 2 differs in run_time
    assert D.exact_duplicate_row_ids(p).tolist() == [1, 4]


def test_apply_variant_filters():
    df = pd.DataFrame({"row_id": [0, 1, 2, 3], "run_time_s": [10.0, 30.0, 31.0, 100.0]})
    assert D.apply_variant(df, {}, None)["row_id"].tolist() == [0, 1, 2, 3]
    assert D.apply_variant(df, {"min_run_time_s_exclusive": 30}, None)["row_id"].tolist() == [2, 3]
    out = D.apply_variant(df, {"drop_exact_duplicates": True}, np.array([1, 3]))
    assert out["row_id"].tolist() == [0, 2]
    with pytest.raises(ValueError):
        D.apply_variant(df, {"drop_exact_duplicates": True}, None)
