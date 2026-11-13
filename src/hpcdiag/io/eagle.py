"""NREL Eagle loader: raw `eagle_data.parquet` -> jobs contract (PLAN.md §3.1, F2, F4).

Units (F4): run_time and wallclock_req are seconds. mem_req: units unknown, kept as-is (D12).
Unique key: row_id = position in the raw file (D10); job_id repeats in Eagle.
Lossless (D8): every raw row is kept; nothing is filtered here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hpcdiag.io.contract import (
    JOBS_SCHEMA,
    normalise_state,
    raw_row_ids,
    to_utc_naive,
    validate_jobs,
)

RAW_COLUMNS = [
    "job_id", "user", "account", "partition", "qos", "wallclock_req", "nodes_req",
    "processors_req", "gpus_req", "mem_req", "submit_time", "start_time", "end_time",
    "run_time", "state", "name",
]  # work_dir and submit_line are not needed by any phase


def read_raw(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns or RAW_COLUMNS)


def to_contract(raw: pd.DataFrame) -> pd.DataFrame:
    """Map raw Eagle columns onto the jobs contract. `raw` = whole file in file order."""
    out = pd.DataFrame(
        {
            "row_id": raw_row_ids(raw),
            "job_id": raw["job_id"].astype("int64"),
            "system": "eagle",
            "user": raw["user"].astype("str"),
            "partition": raw["partition"].astype("str"),
            "qos": raw["qos"].astype("str"),
            "account": raw["account"],
            "name": raw["name"],
            "submit_time": to_utc_naive(raw["submit_time"]),
            "start_time": to_utc_naive(raw["start_time"]),
            "end_time": to_utc_naive(raw["end_time"]),
            "run_time_s": raw["run_time"].astype("float64"),
            "walltime_req_s": raw["wallclock_req"].astype("float64"),  # seconds (F4)
            "nodes_req": raw["nodes_req"].astype("int64"),
            "procs_req": raw["processors_req"].astype("int64"),
            "gpus_req": raw["gpus_req"].astype("int64"),
            "mem_req": raw["mem_req"].astype("float64"),
            "state_raw": raw["state"].astype("str"),
        }
    )
    out["state"] = normalise_state(out["state_raw"])
    out["censored"] = out["state"] == "TIMEOUT"
    out = out[JOBS_SCHEMA.names]
    validate_jobs(out)
    return out


def load(path: str | Path) -> pd.DataFrame:
    """Read the raw Eagle file and return the contract frame."""
    return to_contract(read_raw(path))
