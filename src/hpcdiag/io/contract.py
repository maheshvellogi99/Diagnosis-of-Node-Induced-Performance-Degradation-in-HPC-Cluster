"""Data contracts from PLAN.md §3.1 (with D8/D9: lossless loaders, state_raw kept)."""

from __future__ import annotations

import pandas as pd
import pyarrow as pa

TS = pa.timestamp("us")  # UTC-naive (CLAUDE.md conventions)

JOBS_SCHEMA = pa.schema(
    [
        ("job_id", pa.int64()),
        ("system", pa.string()),
        ("user", pa.string()),
        ("partition", pa.string()),
        ("qos", pa.string()),
        ("account", pa.string()),
        ("name", pa.string()),
        ("submit_time", TS),
        ("start_time", TS),
        ("end_time", TS),
        ("run_time_s", pa.float64()),
        ("walltime_req_s", pa.float64()),
        ("nodes_req", pa.int64()),
        ("procs_req", pa.int64()),
        ("gpus_req", pa.int64()),
        ("mem_req", pa.float64()),
        ("state", pa.string()),
        ("state_raw", pa.string()),
        ("censored", pa.bool_()),
    ]
)

JOB_NODES_SCHEMA = pa.schema(
    [
        ("job_id", pa.int64()),
        ("node_id", pa.int64()),
        ("n_nodes", pa.int64()),
        ("start_time", TS),
        ("end_time", TS),
    ]
)

# Columns that must never be null (everything else may be null; D8).
JOBS_NOT_NULL = ["job_id", "system", "user", "partition", "qos", "submit_time",
                 "nodes_req", "procs_req", "gpus_req", "state", "state_raw",
                 "censored"]

KEPT_STATES = ("COMPLETED", "TIMEOUT")


def normalise_state(raw: pd.Series) -> pd.Series:
    """COMPLETED / TIMEOUT kept as is; every other raw state -> 'other' (PLAN.md §3.1)."""
    return raw.where(raw.isin(KEPT_STATES), "other")


def to_utc_naive(s: pd.Series) -> pd.Series:
    """tz-aware -> UTC then drop tz; tz-naive is assumed to be UTC already and kept."""
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s.astype("datetime64[us]")


def to_table(df: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    """Select contract columns in order and cast to the contract schema (raises on mismatch)."""
    return pa.Table.from_pandas(df[schema.names], schema=schema, preserve_index=False)


def validate_jobs(df: pd.DataFrame) -> None:
    """Raise ValueError if a jobs frame breaks the contract rules checked at load time."""
    missing = [c for c in JOBS_SCHEMA.names if c not in df.columns]
    if missing:
        raise ValueError(f"missing contract columns: {missing}")
    nulls = {c: int(df[c].isna().sum()) for c in JOBS_NOT_NULL if df[c].isna().any()}
    if nulls:
        raise ValueError(f"nulls in not-null columns: {nulls}")
    neg = int((df["run_time_s"] < 0).sum())
    if neg:
        raise ValueError(f"{neg} negative run_time_s values")
    bad_state = set(df["state"].unique()) - {*KEPT_STATES, "other"}
    if bad_state:
        raise ValueError(f"unexpected normalised states: {bad_state}")
    if not (df["censored"] == (df["state"] == "TIMEOUT")).all():
        raise ValueError("censored must equal state == 'TIMEOUT'")
