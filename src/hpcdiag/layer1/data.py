"""Phase 1 data: filtered Eagle jobs and exact-duplicate row ids (addendum A, D10)."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE_COLUMNS = ["row_id", "job_id", "submit_time", "end_time", "run_time_s", "state_raw"]


def exact_duplicate_row_ids(raw_path: str | Path) -> np.ndarray:
    """row_id of every raw row that is identical (all columns) to an earlier raw row.

    The first occurrence is kept; later copies are returned. row_id = 0-based file position,
    which duckdb exposes as file_row_number (same order as the Phase 0 loader).
    """
    path = str(raw_path).replace("'", "''")
    cols = pq.read_schema(raw_path).names
    part = ", ".join(f'"{c}"' for c in cols)
    con = duckdb.connect()
    con.sql("SET TimeZone='UTC'")
    sql = f"""
        select file_row_number as row_id from (
            select file_row_number,
                   row_number() over (partition by {part} order by file_row_number) k
            from read_parquet('{path}', file_row_number = true))
        where k > 1 order by 1"""
    return con.sql(sql).fetchnumpy()["row_id"].astype("int64")


def build_dup_table(raw_path: Path, out_path: Path) -> int:
    ids = exact_duplicate_row_ids(raw_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"row_id": pa.array(ids, pa.int64())}), out_path)
    return len(ids)


def load_jobs(jobs_path: str | Path, features: list[str], states: list[str],
              require_walltime: bool) -> pd.DataFrame:
    """Contract jobs with only the needed columns, filtered by state and walltime."""
    cols = list(dict.fromkeys(BASE_COLUMNS + features + ["walltime_req_s", "nodes_req"]))
    df = pd.read_parquet(jobs_path, columns=cols)
    m = df["state_raw"].isin(states)
    if require_walltime:
        m &= df["walltime_req_s"].notna()
    return df.loc[m].reset_index(drop=True)


def apply_variant(df: pd.DataFrame, variant: dict, dup_row_ids: np.ndarray | None) -> pd.DataFrame:
    """Extra row filters of a variant (addendum A)."""
    m = np.ones(len(df), dtype=bool)
    if "min_run_time_s_exclusive" in variant:
        m &= df["run_time_s"].to_numpy() > float(variant["min_run_time_s_exclusive"])
    if variant.get("drop_exact_duplicates"):
        if dup_row_ids is None:
            raise ValueError("variant needs exact duplicate row ids")
        m &= ~np.isin(df["row_id"].to_numpy(), dup_row_ids)
    return df.loc[m].reset_index(drop=True)
