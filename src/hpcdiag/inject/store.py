"""Scenario storage: data/processed/scenarios/<scenario_id>/{changes,truth}.parquet + scenario.json.

Only affected rows are stored (D35 storage choice); `materialize` rebuilds the full jobs table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hpcdiag.io.contract import validate_jobs

CHANGED = ["run_time_s", "end_time", "state", "censored"]


def write_scenario(out_dir: Path, changes: pd.DataFrame, truth: pd.DataFrame, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    changes.to_parquet(out_dir / "changes.parquet", index=False)
    truth.to_parquet(out_dir / "truth.parquet", index=False)
    (out_dir / "scenario.json").write_text(json.dumps(meta, indent=2, default=str))


def materialize(
    base_jobs: pd.DataFrame, changes: pd.DataFrame, validate: bool = True
) -> pd.DataFrame:
    """Full jobs table = base with the changed rows replaced. Base must be in row_id order
    (row_id = position, D10). Unchanged rows are copied untouched."""
    if not np.array_equal(base_jobs["row_id"].to_numpy(), np.arange(len(base_jobs))):
        raise ValueError("base jobs must be in row_id order with row_id = position")
    out = base_jobs.copy()
    if len(changes):
        pos = changes["row_id"].to_numpy()
        for c in CHANGED:
            col = out.columns.get_loc(c)
            out.iloc[pos, col] = changes[c].to_numpy()
    if validate:
        validate_jobs(out)
    return out


def load_scenario(
    scen_dir: Path, base_jobs: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(materialized jobs, truth, meta). The node table to use is meta['job_nodes']."""
    meta = json.loads((scen_dir / "scenario.json").read_text())
    changes = pd.read_parquet(scen_dir / "changes.parquet")
    truth = pd.read_parquet(scen_dir / "truth.parquet")
    return materialize(base_jobs, changes), truth, meta
