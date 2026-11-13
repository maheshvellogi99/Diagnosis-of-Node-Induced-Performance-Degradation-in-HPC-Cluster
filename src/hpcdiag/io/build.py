"""Build the processed contract tables from raw data (PLAN.md §9 Phase 0, tasks 4 and 6).

    uv run python -m hpcdiag.io.build --config configs/phase0_data.yaml
"""

from __future__ import annotations

import argparse
import sys

import pyarrow.parquet as pq

from hpcdiag.io import eagle, pm100
from hpcdiag.io.contract import JOB_NODES_SCHEMA, JOBS_SCHEMA, to_table
from hpcdiag.utils.checksums import verify
from hpcdiag.utils.run import Run, resolve, run_guarded, start_run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    return run_guarded(start_run("phase0", args.config, tag="build"), _build)


def _build(run: Run) -> int:
    cfg, log = run.config, run.log

    problems = verify(resolve(cfg["raw_dir"]), resolve(cfg["checksums_file"]))
    if problems:
        log.error("raw checksum problems: %s", problems)
        run.finish("failed: checksums")
        return 1
    log.info("raw checksums OK")

    out = {}
    jobs = eagle.load(resolve(cfg["eagle"]["raw"]))
    p = resolve(cfg["eagle"]["out_jobs"])
    pq.write_table(to_table(jobs, JOBS_SCHEMA), p)
    out["jobs_eagle_rows"] = len(jobs)
    log.info("wrote %s (%d rows)", p, len(jobs))
    del jobs

    jobs, nodes = pm100.load(resolve(cfg["pm100"]["raw"]))
    p = resolve(cfg["pm100"]["out_jobs"])
    pq.write_table(to_table(jobs, JOBS_SCHEMA), p)
    log.info("wrote %s (%d rows)", p, len(jobs))
    p = resolve(cfg["pm100"]["out_job_nodes"])
    pq.write_table(to_table(nodes, JOB_NODES_SCHEMA), p)
    log.info("wrote %s (%d rows)", p, len(nodes))
    out["jobs_pm100_rows"] = len(jobs)
    out["job_nodes_pm100_rows"] = len(nodes)

    run.write_json("metrics.json", out)
    run.finish("ok", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
