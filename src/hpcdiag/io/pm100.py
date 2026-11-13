"""PM100 (Marconi100) loader: raw `job_table.parquet` -> jobs + job_nodes contracts.

PLAN.md §3.1, F8.

Units: run_time is seconds; time_limit is minutes, 'INFINITE' -> null (F8).
Timestamps are tz-aware UTC in the raw file; they are converted to UTC-naive.
PM100 has no account and no job-name field (Phase 0 finding), so both are null.
mem_req: units unknown, kept as-is (D12). row_id = position in the raw file (D10).
Lossless (D8): every raw row is kept.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hpcdiag.io.contract import (
    JOB_NODES_SCHEMA,
    JOBS_SCHEMA,
    normalise_state,
    raw_row_ids,
    to_utc_naive,
    validate_jobs,
)

RAW_COLUMNS = [
    "job_id",
    "user_id",
    "partition",
    "qos",
    "submit_time",
    "start_time",
    "end_time",
    "run_time",
    "time_limit",
    "num_nodes_req",
    "num_cores_req",
    "num_gpus_req",
    "mem_req",
    "job_state",
    "nodes",
]

INFINITE = "INFINITE"


def time_limit_to_seconds(tl: pd.Series) -> pd.Series:
    """Minutes -> seconds as float64. 'INFINITE' (any case) and nulls -> NaN.

    Accepts integer, float or string input (the 287 MB file stores int64 minutes; the
    documentation allows the string 'INFINITE').
    """
    if pd.api.types.is_numeric_dtype(tl):
        minutes = tl.astype("float64")
    else:
        s = tl.astype("str").str.strip()
        s = s.mask(s.str.upper() == INFINITE)
        minutes = pd.to_numeric(s, errors="raise").astype("float64")
    if (minutes < 0).any():
        raise ValueError("negative time_limit")
    return minutes * 60.0


def parse_nodes(nodes: pd.Series) -> pd.Series:
    """Each element -> list[int]. Accepts arrays/lists of ints or strings like '[1, 2]'."""

    def one(v) -> list[int]:
        if isinstance(v, str):
            v = v.strip().strip("[]")
            return [int(x) for x in v.replace(",", " ").split()]
        return [int(x) for x in v]

    return nodes.map(one)


def to_contract(raw: pd.DataFrame) -> pd.DataFrame:
    """Map raw PM100 columns onto the jobs contract. `raw` = whole file in file order."""
    out = pd.DataFrame(
        {
            "row_id": raw_row_ids(raw),
            "job_id": raw["job_id"].astype("int64"),
            "system": "pm100",
            "user": raw["user_id"].astype("int64").astype("str"),
            "partition": raw["partition"].astype("str"),
            "qos": raw["qos"].astype("str"),
            "account": pd.Series(None, index=raw.index, dtype="str"),
            "name": pd.Series(None, index=raw.index, dtype="str"),
            "submit_time": to_utc_naive(raw["submit_time"]),
            "start_time": to_utc_naive(raw["start_time"]),
            "end_time": to_utc_naive(raw["end_time"]),
            "run_time_s": raw["run_time"].astype("float64"),
            "walltime_req_s": time_limit_to_seconds(raw["time_limit"]),
            "nodes_req": raw["num_nodes_req"].astype("int64"),
            "procs_req": raw["num_cores_req"].astype("int64"),
            "gpus_req": raw["num_gpus_req"].astype("int64"),
            "mem_req": raw["mem_req"].astype("float64"),
            "state_raw": raw["job_state"].astype("str"),
        }
    )
    out["state"] = normalise_state(out["state_raw"])
    out["censored"] = out["state"] == "TIMEOUT"
    out = out[JOBS_SCHEMA.names]
    validate_jobs(out)
    return out


def job_nodes(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per (job, allocated node): row_id, job_id, node_id, n_nodes, start, end."""
    nodes = parse_nodes(raw["nodes"])
    n = nodes.map(len).astype("int64")
    if (n == 0).any():
        raise ValueError(f"{int((n == 0).sum())} jobs with an empty node list")
    rep = np.repeat(np.arange(len(raw)), n.to_numpy())
    out = pd.DataFrame(
        {
            "row_id": raw_row_ids(raw)[rep],
            "job_id": raw["job_id"].to_numpy()[rep].astype("int64"),
            "node_id": np.fromiter(
                (x for lst in nodes for x in lst), dtype="int64", count=int(n.sum())
            ),
            "n_nodes": n.to_numpy()[rep],
            "start_time": to_utc_naive(raw["start_time"]).to_numpy()[rep],
            "end_time": to_utc_naive(raw["end_time"]).to_numpy()[rep],
        }
    )
    return out[JOB_NODES_SCHEMA.names]


def read_raw(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns or RAW_COLUMNS)


def load(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the raw PM100 file and return (jobs, job_nodes) contract frames."""
    raw = read_raw(path)
    return to_contract(raw), job_nodes(raw)
