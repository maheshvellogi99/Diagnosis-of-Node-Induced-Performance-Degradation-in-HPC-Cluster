"""Tests for hpcdiag.io (PLAN.md §3.1 contracts, Phase 0).

Two kinds:
- unit tests on tiny hand-made frames (always run);
- `realdata` tests on data/processed/*.parquet (skipped if the build has not been run).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from hpcdiag.io import eagle, pm100
from hpcdiag.io.contract import (
    JOB_NODES_SCHEMA,
    JOBS_SCHEMA,
    normalise_state,
    to_table,
    validate_jobs,
)
from hpcdiag.utils.run import resolve

# ----------------------------------------------------------------------------- fixtures


def _eagle_raw() -> pd.DataFrame:
    t = pd.Timestamp("2020-01-01 00:00:00")
    return pd.DataFrame(
        {
            "job_id": [1, 2, 3, 4],
            "user": ["u1", "u1", "u2", "u3"],
            "account": ["a", "a", "b", None],
            "partition": ["short", "short", "gpu", "std"],
            "qos": ["normal"] * 4,
            "wallclock_req": [3600.0, 600.0, None, 60.0],
            "nodes_req": [1, 2, 1, 1],
            "processors_req": [36, 72, 1, 1],
            "gpus_req": [0, 0, 2, 0],
            "mem_req": [0.0, 1000.0, 0.0, 0.0],
            "submit_time": [t] * 4,
            "start_time": [t, t, t, pd.NaT],
            "end_time": [t + pd.Timedelta(seconds=100), t + pd.Timedelta(seconds=600), t, pd.NaT],
            "run_time": [100.0, 600.0, 0.0, 0.0],
            "state": ["COMPLETED", "TIMEOUT", "CANCELLED", "PENDING"],
            "name": ["n1", "n2", "n3", "n4"],
        }
    )


def _pm100_raw() -> pd.DataFrame:
    t = pd.Timestamp("2020-06-01 10:00:00", tz="UTC")
    return pd.DataFrame(
        {
            "job_id": [10, 11, 12],
            "user_id": [7, 7, 8],
            "partition": ["1", "1", "0"],
            "qos": ["1", "11", "1"],
            "submit_time": [t] * 3,
            "start_time": [t] * 3,
            "end_time": [t + pd.Timedelta(seconds=s) for s in (60, 120, 0)],
            "run_time": [60, 120, 0],
            "time_limit": [30, 1440, 1],
            "num_nodes_req": [1, 3, 2],
            "num_cores_req": [4, 384, 256],
            "num_gpus_req": [0, 12, 8],
            "mem_req": [1000, 2000, 3000],
            "job_state": ["COMPLETED", "TIMEOUT", "FAILED"],
            "nodes": [np.array([5]), np.array([1, 2, 3]), np.array([8, 9])],
        }
    )


# ----------------------------------------------------------------------------- unit tests


def test_eagle_contract_columns_dtypes_and_values():
    out = eagle.to_contract(_eagle_raw())
    assert list(out.columns) == JOBS_SCHEMA.names
    tbl = to_table(out, JOBS_SCHEMA)  # raises if any column cannot be cast to the contract
    assert tbl.schema.equals(JOBS_SCHEMA)
    assert len(out) == 4  # lossless (D8): PENDING and CANCELLED rows kept
    assert out["state"].tolist() == ["COMPLETED", "TIMEOUT", "other", "other"]
    assert out["state_raw"].tolist() == ["COMPLETED", "TIMEOUT", "CANCELLED", "PENDING"]
    assert out["censored"].tolist() == [False, True, False, False]
    assert out["walltime_req_s"].iloc[0] == 3600.0  # seconds, no conversion (F4)
    assert np.isnan(out["walltime_req_s"].iloc[2])
    assert (out["system"] == "eagle").all()


def test_pm100_contract_time_limit_tz_and_user():
    out = pm100.to_contract(_pm100_raw())
    assert list(out.columns) == JOBS_SCHEMA.names
    assert to_table(out, JOBS_SCHEMA).schema.equals(JOBS_SCHEMA)
    assert out["walltime_req_s"].tolist() == [1800.0, 86400.0, 60.0]  # minutes x 60
    assert out["user"].tolist() == ["7", "7", "8"]
    assert out["account"].isna().all() and out["name"].isna().all()
    # tz-aware UTC 10:00 -> naive 10:00 (not shifted to local time)
    assert out["start_time"].dt.tz is None
    assert out["start_time"].iloc[0] == pd.Timestamp("2020-06-01 10:00:00")
    assert out["censored"].tolist() == [False, True, False]


@pytest.mark.parametrize(
    "values, expected",
    [
        (pd.Series([1, 30, 1440]), [60.0, 1800.0, 86400.0]),
        (pd.Series(["30", "INFINITE", "infinite ", None]), [1800.0, np.nan, np.nan, np.nan]),
        (pd.Series([10.0, np.nan]), [600.0, np.nan]),
    ],
)
def test_time_limit_to_seconds(values, expected):
    got = pm100.time_limit_to_seconds(values)
    np.testing.assert_array_equal(got.to_numpy(dtype="float64"), np.array(expected))


def test_time_limit_negative_rejected():
    with pytest.raises(ValueError):
        pm100.time_limit_to_seconds(pd.Series([-1]))


def test_parse_nodes_lists_of_ints():
    s = pd.Series([np.array([3, 1]), [7], "[4, 5, 6]", "2"])
    got = pm100.parse_nodes(s)
    assert got.tolist() == [[3, 1], [7], [4, 5, 6], [2]]
    assert all(type(x) is int for lst in got for x in lst)


def test_job_nodes_rowcount_equals_sum_of_nodes():
    raw = _pm100_raw()
    jn = pm100.job_nodes(raw)
    assert list(jn.columns) == JOB_NODES_SCHEMA.names
    assert to_table(jn, JOB_NODES_SCHEMA).schema.equals(JOB_NODES_SCHEMA)
    assert len(jn) == sum(len(x) for x in raw["nodes"]) == 6
    assert jn[jn.job_id == 11]["node_id"].tolist() == [1, 2, 3]
    assert (jn.groupby("job_id")["n_nodes"].first() == jn.groupby("job_id").size()).all()


def test_job_nodes_empty_list_rejected():
    raw = _pm100_raw()
    raw.at[0, "nodes"] = np.array([], dtype="int64")
    with pytest.raises(ValueError):
        pm100.job_nodes(raw)


def test_negative_run_time_rejected():
    raw = _eagle_raw()
    raw.loc[0, "run_time"] = -5.0
    with pytest.raises(ValueError, match="negative"):
        eagle.to_contract(raw)


def test_validate_jobs_rejects_bad_censoring():
    out = eagle.to_contract(_eagle_raw())
    out.loc[0, "censored"] = True
    with pytest.raises(ValueError, match="censored"):
        validate_jobs(out)


def test_normalise_state():
    s = pd.Series(["COMPLETED", "TIMEOUT", "FAILED", "NODE_FAIL", "OUT_OF_MEMORY"])
    assert normalise_state(s).tolist() == ["COMPLETED", "TIMEOUT", "other", "other", "other"]


# ----------------------------------------------------------------------------- real data

PROCESSED = {
    "eagle": resolve("data/processed/jobs_eagle.parquet"),
    "pm100": resolve("data/processed/jobs_pm100.parquet"),
}
JOB_NODES = resolve("data/processed/job_nodes_pm100.parquet")

realdata = pytest.mark.realdata


def _need(path):
    if not path.exists():
        pytest.skip(f"{path} not built; run `python -m hpcdiag.io.build`")


@realdata
@pytest.mark.parametrize("system", ["eagle", "pm100"])
def test_processed_jobs_schema(system):
    path = PROCESSED[system]
    _need(path)
    assert pq.read_schema(path).remove_metadata().equals(JOBS_SCHEMA)


@realdata
@pytest.mark.parametrize("system", ["eagle", "pm100"])
def test_processed_jobs_values(system):
    path = PROCESSED[system]
    _need(path)
    t = pq.read_table(path, columns=["run_time_s", "state", "state_raw", "censored", "system",
                                     "job_id"])
    rt = t["run_time_s"].to_numpy(zero_copy_only=False)
    assert not (rt < 0).any(), "negative run times"
    assert set(t["system"].unique().to_pylist()) == {system}
    assert set(t["state"].unique().to_pylist()) <= {"COMPLETED", "TIMEOUT", "other"}
    df = t.select(["state", "censored"]).to_pandas()
    assert (df["censored"] == (df["state"] == "TIMEOUT")).all()


@realdata
def test_processed_rows_equal_raw_rows():
    """Lossless loaders (D8): processed row count equals raw row count."""
    for system, raw in (("eagle", "data/raw/eagle/eagle_data.parquet"),
                        ("pm100", "data/raw/pm100/job_table.parquet")):
        _need(PROCESSED[system])
        assert (pq.ParquetFile(PROCESSED[system]).metadata.num_rows
                == pq.ParquetFile(resolve(raw)).metadata.num_rows)


@realdata
def test_pm100_job_id_unique():
    _need(PROCESSED["pm100"])
    ids = pq.read_table(PROCESSED["pm100"], columns=["job_id"])["job_id"]
    assert len(ids) == len(ids.unique())


@realdata
@pytest.mark.xfail(strict=True, reason="Eagle job_id repeats in the raw file (Phase 0 finding); "
                   "contract says unique — awaiting decision, see reports/phase_0.md")
def test_eagle_job_id_unique():
    _need(PROCESSED["eagle"])
    ids = pq.read_table(PROCESSED["eagle"], columns=["job_id"])["job_id"]
    assert len(ids) == len(ids.unique())


@realdata
def test_pm100_raw_nodes_parse_to_int_lists():
    raw = pq.read_table(resolve("data/raw/pm100/job_table.parquet"), columns=["nodes"])
    assert pa.types.is_list(raw.schema.field("nodes").type)
    assert pa.types.is_int64(raw.schema.field("nodes").type.value_type)
    parsed = pm100.parse_nodes(raw.to_pandas()["nodes"].head(2000))
    assert all(isinstance(lst, list) and len(lst) > 0 for lst in parsed)
    assert all(type(x) is int for lst in parsed for x in lst)


@realdata
def test_job_nodes_schema_and_rowcount():
    _need(JOB_NODES)
    assert pq.read_schema(JOB_NODES).remove_metadata().equals(JOB_NODES_SCHEMA)
    jn = pq.read_table(JOB_NODES, columns=["job_id", "n_nodes"]).to_pandas()
    raw = pq.read_table(resolve("data/raw/pm100/job_table.parquet"), columns=["job_id", "nodes"])
    sum_nodes = int(pc.sum(pc.list_value_length(raw["nodes"])).as_py())
    assert len(jn) == sum_nodes
    per_job = jn.groupby("job_id")["n_nodes"].agg(["first", "size"])
    assert (per_job["first"] == per_job["size"]).all()
    assert len(jn) == int(per_job["first"].sum())
    assert set(per_job.index) == set(raw["job_id"].to_pylist())
