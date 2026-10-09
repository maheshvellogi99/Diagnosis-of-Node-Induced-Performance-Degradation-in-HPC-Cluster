"""Phase 0 profiling: metrics.json, Eagle F1–F5 check, data cards (PLAN.md §9 Phase 0).

    uv run python -m hpcdiag.io.profile --config configs/phase0_data.yaml

All numbers in the data cards and the F-check table are rendered from metrics.json
(PLAN.md §7.6); nothing is typed by hand.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import duckdb
import pyarrow.parquet as pq

from hpcdiag.io import eagle as eagle_io
from hpcdiag.io import pm100 as pm100_io
from hpcdiag.utils.checksums import read_checksums, verify
from hpcdiag.utils.run import resolve, start_run

# ----------------------------------------------------------------------------- helpers


def _con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.sql("SET TimeZone='UTC'")
    return con


def _rows(con, sql: str) -> list[dict[str, Any]]:
    rel = con.sql(sql)
    cols = rel.columns
    return [dict(zip(cols, r, strict=True)) for r in rel.fetchall()]


def _one(con, sql: str) -> dict[str, Any]:
    return _rows(con, sql)[0]


def _q(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def common_profile(con, jobs: Path, quantiles: list[float], top_users: int) -> dict[str, Any]:
    """Profile of a processed jobs_<system>.parquet (contract columns)."""
    J = _q(jobs)
    schema = pq.read_schema(jobs)
    m: dict[str, Any] = {"schema": [(f.name, str(f.type)) for f in schema]}
    m["counts"] = _one(con, f"""
        select count(*) n_rows, count(distinct job_id) distinct_job_ids,
               count(distinct "user") users, count(distinct partition) partitions,
               count(distinct qos) qos_values from {J}""")
    m["date_range"] = _one(con, f"""
        select min(submit_time) submit_min, max(submit_time) submit_max,
               min(start_time) start_min, max(start_time) start_max,
               min(end_time) end_min, max(end_time) end_max from {J}""")
    m["state_raw_counts"] = _rows(con, f"""
        select state_raw, count(*) n from {J} group by 1 order by 2 desc, 1""")
    m["state_counts"] = _rows(con, f"""
        select state, count(*) n from {J} group by 1 order by 2 desc, 1""")
    m["missing"] = _one(con, "select " + ", ".join(
        f'count(*) filter (where "{c}" is null) "{c}"' for c in schema.names) + f" from {J}")
    qs = ", ".join(f"quantile_cont(run_time_s, {q}) \"q{q:g}\"" for q in quantiles)
    m["run_time_quantiles_all"] = _one(con, f"""
        select count(*) n, min(run_time_s) min, {qs}, max(run_time_s) max,
               count(*) filter (where run_time_s = 0) n_zero,
               count(*) filter (where run_time_s < 0) n_negative from {J}""")
    m["run_time_quantiles_completed_timeout"] = _one(con, f"""
        select count(*) n, min(run_time_s) min, {qs}, max(run_time_s) max,
               count(*) filter (where run_time_s = 0) n_zero
        from {J} where state in ('COMPLETED', 'TIMEOUT')""")
    m["walltime_req_s"] = _one(con, f"""
        select count(walltime_req_s) n_non_null, min(walltime_req_s) min,
               median(walltime_req_s) median, max(walltime_req_s) max from {J}""")
    m["mem_req"] = _one(con, f"""
        select min(mem_req) min, median(mem_req) median, max(mem_req) max,
               count(*) filter (where mem_req = 0) n_zero from {J}""")
    m["jobs_per_user"] = _one(con, f"""
        with u as (select "user", count(*) n from {J} group by 1)
        select count(*) users, min(n) min, quantile_cont(n, 0.10) p10, median(n) median,
               quantile_cont(n, 0.90) p90, max(n) max,
               count(*) filter (where n < 30) users_lt_30_jobs from u""")
    m["jobs_per_user_completed_timeout"] = _one(con, f"""
        with u as (select "user", count(*) n from {J}
                   where state in ('COMPLETED', 'TIMEOUT') group by 1)
        select count(*) users, min(n) min, quantile_cont(n, 0.10) p10, median(n) median,
               quantile_cont(n, 0.90) p90, max(n) max,
               count(*) filter (where n < 30) users_lt_30_jobs from u""")
    m["top_users"] = _rows(con, f"""
        select "user", count(*) n, count(*) / (select count(*) from {J}) share
        from {J} group by 1 order by 2 desc, 1 limit {int(top_users)}""")
    m["partition_counts"] = _rows(con, f"""
        select partition, count(*) n from {J} group by 1 order by 2 desc, 1""")
    m["qos_counts"] = _rows(con, f"""
        select qos, count(*) n from {J} group by 1 order by 2 desc, 1""")
    m["duplicate_job_ids"] = _one(con, f"""
        with d as (select job_id, count(*) k from {J} group by 1 having count(*) > 1)
        select count(*) ids_repeated, coalesce(sum(k), 0) rows_on_repeated_ids,
               coalesce(max(k), 0) max_rows_per_id from d""")
    return m


# ----------------------------------------------------------------------------- Eagle


def eagle_profile(con, raw: Path) -> dict[str, Any]:
    R = _q(raw)
    m: dict[str, Any] = {"raw_columns": pq.read_schema(raw).names}
    m["raw_schema"] = [(f.name, str(f.type)) for f in pq.read_schema(raw)]
    m["exact_duplicate_rows"] = _one(con, f"""
        with d as (select * from {R}
                   where job_id in (select job_id from {R} group by 1 having count(*) > 1))
        select count(*) rows_on_repeated_ids,
               count(*) - (select count(*) from (select distinct * from d)) exact_copies
        from d""")
    m["F"] = _one(con, f"""
        with ct as (select * from {R} where state in ('COMPLETED', 'TIMEOUT')),
             c as (select * from {R} where state = 'COMPLETED')
        select
          (select count(*) from {R}) F1_rows,
          (select cast(min(submit_time) as date) from {R}) F1_submit_min_date,
          (select cast(max(submit_time) as date) from {R}) F1_submit_max_date,
          (select count(distinct "user") from {R}) F1_users,
          (select count(distinct partition) from {R}) F1_partitions,
          (select avg(case when abs(epoch(end_time) - epoch(start_time) - run_time) <= 1
                           then 1.0 else 0.0 end) from c) F4_runtime_matches_end_minus_start,
          (select count(*) filter (where end_time is null or start_time is null) from c)
              F4_completed_missing_start_or_end,
          (select avg(case when run_time > wallclock_req then 1.0 else 0.0 end) from c)
              F4_runtime_gt_wallclock,
          (select count(*) from {R} where wallclock_req is null) F4_wallclock_null_rows,
          (select count(*) from ct) F5_completed_timeout_rows,
          (select datediff('day', min(submit_time), max(submit_time)) from ct)
              F5_span_days_datediff,
          (select floor((epoch(max(submit_time)) - epoch(min(submit_time))) / 86400)::bigint
             from ct) F5_span_days,
          (select median(run_time) from ct) F5_median_run_time_s,
          (select avg(case when nodes_req = 1 then 1.0 else 0.0 end) from c)
              F5_one_node_frac_completed,
          (select avg(case when run_time < 30 then 1.0 else 0.0 end) from ct) F5_lt30s_frac
    """)
    f = m["F"]
    f["F5_jobs_per_day"] = f["F5_completed_timeout_rows"] / f["F5_span_days"]
    f["F2_columns"] = m["raw_columns"]
    f["F2_has_node_list"] = any("node" in c and c != "nodes_req" for c in m["raw_columns"])
    f["F3_states"] = {r["state"]: r["n"] for r in _rows(
        con, f"select state, count(*) n from {R} group by 1 order by 2 desc")}
    return m


def check_facts(got: dict[str, Any], exp: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare computed Eagle facts with PLAN.md F1–F5 (expected values from config)."""
    rows = []
    for key, e in exp.items():
        g = got[key]
        if isinstance(e, dict) and "value" in e:
            d = e["decimals"]
            ok = round(float(g), d) == round(float(e["value"]), d)
            rows.append({"fact": key, "expected": e["value"], "got": g,
                         "compared": f"rounded to {d} decimals: {round(float(g), d)}", "match": ok})
        elif key == "F3_states":
            ok = dict(g) == dict(e)
            rows.append({"fact": key, "expected": e, "got": g, "compared": "exact", "match": ok})
        elif key == "F2_columns":
            ok = list(g) == list(e)
            rows.append({"fact": key, "expected": e, "got": g, "compared": "exact, ordered",
                         "match": ok})
        else:
            ok = str(g) == str(e) if "date" in key else g == e
            rows.append({"fact": key, "expected": e, "got": g, "compared": "exact", "match": ok})
    rows.append({"fact": "F2_no_node_list", "expected": False, "got": got["F2_has_node_list"],
                 "compared": "no column naming nodes other than nodes_req",
                 "match": got["F2_has_node_list"] is False})
    return rows


# ----------------------------------------------------------------------------- PM100


def pm100_profile(con, raw: Path, jobs: Path, job_nodes: Path) -> dict[str, Any]:
    R, J, N = _q(raw), _q(jobs), _q(job_nodes)
    raw_schema = pq.read_schema(raw)
    m: dict[str, Any] = {
        "raw_schema": [(f.name, str(f.type)) for f in raw_schema],
        "raw_rows": pq.ParquetFile(raw).metadata.num_rows,
        "columns_used": pm100_io.RAW_COLUMNS,
        "name_like_columns": [c for c in raw_schema.names if "name" in c.lower()],
    }
    m["raw_checks"] = _one(con, f"""
        select count(*) filter (where abs(epoch(end_time) - epoch(start_time) - run_time) <= 1)
                 run_time_eq_end_minus_start,
               count(*) filter (where num_nodes_req <> num_nodes_alloc) nodes_req_ne_alloc,
               count(*) filter (where len(nodes) <> num_nodes_alloc) len_nodes_ne_alloc,
               count(*) filter (where nodes is null or len(nodes) = 0) nodes_empty,
               min(time_limit) time_limit_min_minutes, max(time_limit) time_limit_max_minutes,
               count(*) filter (where time_limit is null) time_limit_null,
               sum(len(nodes)) sum_len_nodes,
               count(distinct group_id) distinct_group_id
        from {R}""")
    m["time_limit_raw_type"] = str(raw_schema.field("time_limit").type)
    m["shared_counts"] = _rows(
        con, f"select shared, count(*) n from {R} group by 1 order by 2 desc")
    m["job_nodes"] = _one(con, f"""
        select count(*) n_rows, count(distinct node_id) distinct_nodes,
               count(distinct job_id) distinct_jobs, min(node_id) min_node_id,
               max(node_id) max_node_id from {N}""")
    m["job_nodes_vs_jobs"] = _one(con, f"""
        select (select count(*) from {N}) job_nodes_rows,
               (select sum(n) from (select any_value(n_nodes) n from {N} group by job_id))
                 sum_n_nodes_per_job,
               (select count(*) from (select job_id from {N} group by job_id
                                      having count(*) <> any_value(n_nodes)))
                 jobs_rowcount_ne_n_nodes,
               (select count(*) from (select job_id, node_id from {N} group by 1, 2
                                      having count(*) > 1)) repeated_job_node_pairs""")
    m["multi_node"] = _one(con, f"""
        with j as (select job_id, any_value(n_nodes) n from {N} group by 1)
        select count(*) jobs, count(*) filter (where n > 1) multi_node_jobs,
               avg(case when n > 1 then 1.0 else 0.0 end) multi_node_frac,
               max(n) max_nodes from j""")
    m["multi_node_completed_timeout"] = _one(con, f"""
        select count(*) jobs, avg(case when nodes_req > 1 then 1.0 else 0.0 end) frac_nodes_req_gt1
        from {J} where state in ('COMPLETED', 'TIMEOUT')""")
    # jobs per node per day: job counted on each of its nodes, on the UTC day it started.
    m["jobs_per_node_per_day"] = _one(con, f"""
        with c as (select node_id, cast(start_time as date) d, count(*) n from {N} group by 1, 2),
             days as (select unnest(generate_series(
                         (select min(cast(start_time as date)) from {N}),
                         (select max(cast(start_time as date)) from {N}),
                         interval 1 day))::date d),
             nodes as (select distinct node_id from {N}),
             grid as (select nodes.node_id, days.d, coalesce(c.n, 0) n
                      from nodes cross join days left join c using (node_id, d))
        select (select count(*) from days) days, (select count(*) from nodes) nodes,
               count(*) node_days,
               quantile_cont(n, 0.10) p10_incl_zero, median(n) median_incl_zero,
               quantile_cont(n, 0.90) p90_incl_zero, avg(n) mean_incl_zero,
               count(*) filter (where n = 0) zero_node_days,
               (select quantile_cont(n, 0.10) from c) p10_nonzero,
               (select median(n) from c) median_nonzero,
               (select quantile_cont(n, 0.90) from c) p90_nonzero,
               (select avg(n) from c) mean_nonzero,
               (select min(d) from days) first_day, (select max(d) from days) last_day
        from grid""")
    # Q2 overlaps: per node, sort by (start, end); job overlaps if it starts strictly before the
    # latest end time among earlier jobs on that node.
    ov_base = f"""
        with s as (select job_id, node_id, start_time, end_time,
                          max(end_time) over (partition by node_id order by start_time, end_time,
                                              job_id rows between unbounded preceding
                                              and 1 preceding) prev_end
                   from {N}),
             o as (select *, epoch(least(prev_end, end_time)) - epoch(start_time) ov_s
                   from s where start_time < prev_end)"""
    m["overlaps"] = _one(con, ov_base + """
        select count(*) overlapping_rows, count(distinct node_id) nodes_with_overlap,
               count(distinct job_id) jobs_involved,
               coalesce(max(ov_s), 0) max_overlap_s, median(ov_s) median_overlap_s,
               count(*) filter (where ov_s <= 1) ov_le_1s,
               count(*) filter (where ov_s > 1 and ov_s <= 60) ov_1_to_60s,
               count(*) filter (where ov_s > 60) ov_gt_60s from o""")
    m["overlap_examples"] = _rows(con, ov_base + """
        select node_id, job_id, start_time, end_time, prev_end, ov_s from o
        order by ov_s desc, node_id, job_id limit 5""")
    return m


# ----------------------------------------------------------------------------- rendering


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        if v.is_integer() and abs(v) < 1e15:
            return f"{int(v):,}"
        return f"{v:.6g}"
    if isinstance(v, int) and not isinstance(v, bool):
        return f"{v:,}"
    return str(v)


def _table(rows: list[dict[str, Any]], cols: list[str] | None = None) -> str:
    if not rows:
        return "_(none)_\n"
    cols = cols or list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(_fmt(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out) + "\n"


def _kv(d: dict[str, Any]) -> str:
    return _table([{"field": k, "value": v} for k, v in d.items()])


def _common_sections(c: dict[str, Any]) -> str:
    s = "## Row counts\n\n" + _kv(c["counts"])
    s += "\nRepeated job_id values:\n\n" + _kv(c["duplicate_job_ids"])
    s += "\n## Date range (UTC-naive)\n\n" + _kv(c["date_range"])
    s += "\n## Raw state counts (`state_raw`)\n\n" + _table(c["state_raw_counts"])
    s += "\n## Normalised state counts (`state`)\n\n" + _table(c["state_counts"])
    s += "\n## Missing values per contract column\n\n" + _kv(c["missing"])
    s += "\n## Runtime quantiles (`run_time_s`, seconds)\n\nAll rows:\n\n"
    s += _kv(c["run_time_quantiles_all"])
    s += "\nCOMPLETED + TIMEOUT only:\n\n" + _kv(c["run_time_quantiles_completed_timeout"])
    s += "\n## Requested walltime (`walltime_req_s`, seconds)\n\n" + _kv(c["walltime_req_s"])
    s += "\n## `mem_req` (source units)\n\n" + _kv(c["mem_req"])
    s += "\n## Per-user job counts\n\nAll rows:\n\n" + _kv(c["jobs_per_user"])
    s += "\nCOMPLETED + TIMEOUT only:\n\n" + _kv(c["jobs_per_user_completed_timeout"])
    s += f"\nTop {len(c['top_users'])} users (all rows):\n\n" + _table(c["top_users"])
    s += "\n## Partition counts\n\n" + _table(c["partition_counts"])
    s += "\n## QOS counts\n\n" + _table(c["qos_counts"])
    return s


def _schema_section(c: dict[str, Any], mapping: dict[str, str]) -> str:
    rows = [{"column": n, "type": t, "source / units": mapping.get(n, "")} for n, t in c["schema"]]
    return "## Schema (`" + "jobs contract, PLAN.md §3.1" + "`)\n\n" + _table(rows)


EAGLE_MAP = {
    "job_id": "job_id (NOT unique, see below)",
    "system": "constant 'eagle'",
    "user": "user", "partition": "partition", "qos": "qos", "account": "account",
    "name": "name",
    "submit_time": "submit_time (stored tz-naive; kept as is)",
    "start_time": "start_time", "end_time": "end_time",
    "run_time_s": "run_time, seconds (F4)",
    "walltime_req_s": "wallclock_req, seconds (F4)",
    "nodes_req": "nodes_req", "procs_req": "processors_req", "gpus_req": "gpus_req",
    "mem_req": "mem_req, units UNKNOWN — needs check (not stated in sources we hold)",
    "state": "state -> COMPLETED / TIMEOUT / other", "state_raw": "state (D9)",
    "censored": "state == TIMEOUT",
}

PM100_MAP = {
    "job_id": "job_id",
    "system": "constant 'pm100'",
    "user": "user_id (int, anonymised; cast to str)",
    "partition": "partition (anonymised code)", "qos": "qos (anonymised code)",
    "account": "null (no account field; group_id exists, not mapped)",
    "name": "null (no job-name-like field)",
    "submit_time": "submit_time (raw tz-aware UTC -> UTC-naive)",
    "start_time": "start_time (UTC -> naive)", "end_time": "end_time (UTC -> naive)",
    "run_time_s": "run_time, seconds (F8)",
    "walltime_req_s": "time_limit minutes x 60; INFINITE -> null (F8)",
    "nodes_req": "num_nodes_req", "procs_req": "num_cores_req", "gpus_req": "num_gpus_req",
    "mem_req": "mem_req, units UNKNOWN — needs check (F8 does not state units)",
    "state": "job_state -> COMPLETED / TIMEOUT / other", "state_raw": "job_state (D9)",
    "censored": "state == TIMEOUT",
}


def render_eagle_card(m: dict[str, Any], run_id: str, sums: dict) -> str:
    c, e = m["eagle"]["common"], m["eagle"]
    s = "# Data card — NREL Eagle jobs\n\n"
    s += f"_Generated by `hpcdiag.io.profile` from `runs/phase0/{run_id}/metrics.json`. "
    s += "Do not edit by hand._\n\n"
    s += "## Source\n\n"
    s += "- File: `data/raw/eagle/eagle_data.parquet` (read-only)\n"
    s += "- Publisher: NREL, OEDI submission 5860 (https://data.openei.org/submissions/5860)\n"
    s += "- Licence: CC-BY 4.0 (PLAN.md §4). Cite the dataset and Menear et al. (PEARC'23).\n"
    sha = sums.get("eagle/eagle_data.parquet")
    if sha:
        s += f"- SHA-256: `{sha[0]}`; MD5: `{sha[1]}`; size {sha[2]:,} bytes\n"
    s += "- Processed: `data/processed/jobs_eagle.parquet` (lossless, D8)\n\n"
    s += "## Raw schema\n\n" + _table([{"column": n, "type": t} for n, t in e["raw_schema"]])
    s += "\nNot loaded: `work_dir`, `submit_line` (no phase needs them). "
    s += "No node list (F2).\n\n"
    s += _schema_section(c, EAGLE_MAP)
    s += "\n## Units\n\n- Times: UTC-naive timestamps as stored.\n"
    s += "- `run_time_s`, `walltime_req_s`: seconds (F4).\n"
    s += "- `mem_req`: source units, UNKNOWN — needs check.\n\n"
    s += _common_sections(c)
    s += "\n## Repeated job_id (contract says unique within system)\n\n"
    s += _kv(e["exact_duplicate_rows"])
    s += ("\n`exact_copies` = rows on repeated ids that are identical to another row in all 18 "
          "raw columns. Rows on a repeated id share user and name; they look like Slurm array "
          "tasks with the array index missing, but the dataset documentation we hold does not "
          "say so — UNKNOWN — needs check.\n")
    return s


def render_pm100_card(m: dict[str, Any], run_id: str, sums: dict, licence: str) -> str:
    c, p = m["pm100"]["common"], m["pm100"]
    s = "# Data card — PM100 (Marconi100)\n\n"
    s += f"_Generated by `hpcdiag.io.profile` from `runs/phase0/{run_id}/metrics.json`. "
    s += "Do not edit by hand._\n\n"
    s += "## Source\n\n"
    s += "- File: `data/raw/pm100/job_table.parquet` (read-only)\n"
    s += "- Publisher: Zenodo record 10127767 (https://zenodo.org/records/10127767); "
    s += "code https://github.com/francescoantici/PM100-data\n"
    s += f"- Licence: {licence}\n"
    sha = sums.get("pm100/job_table.parquet")
    if sha:
        s += (f"- SHA-256: `{sha[0]}`; MD5: `{sha[1]}` (equals the published Zenodo MD5); "
              f"size {sha[2]:,} bytes\n")
    s += ("- Processed: `data/processed/jobs_pm100.parquet`, "
          "`data/processed/job_nodes_pm100.parquet` (lossless, D8)\n")
    s += (f"- Raw rows: {p['raw_rows']:,}. PLAN.md F7 (Zenodo page) says 231,116 jobs — "
          f"{'match' if p['raw_rows'] == 231116 else 'MISMATCH'}.\n\n")
    s += "## Raw schema\n\n" + _table([{"column": n, "type": t} for n, t in p["raw_schema"]])
    s += "\nColumns loaded: " + ", ".join(f"`{x}`" for x in p["columns_used"]) + ".\n\n"
    s += _schema_section(c, PM100_MAP)
    s += "\n## Units\n\n- Times: raw tz-aware UTC, converted to UTC-naive.\n"
    s += "- `run_time_s`: seconds (F8).\n"
    s += (f"- `walltime_req_s` = `time_limit` (minutes) x 60. Raw `time_limit` type is "
          f"`{p['time_limit_raw_type']}`; range {p['raw_checks']['time_limit_min_minutes']}–"
          f"{p['raw_checks']['time_limit_max_minutes']} min; "
          f"{p['raw_checks']['time_limit_null']} nulls; no `INFINITE` values can occur in an "
          "int64 column. The converter still maps `INFINITE` -> null (unit-tested).\n")
    s += "- `mem_req`: source units, UNKNOWN — needs check.\n\n"
    s += "## Raw consistency checks\n\n" + _kv(p["raw_checks"])
    s += "\n`shared` column (raw) counts:\n\n" + _table(p["shared_counts"])
    s += _common_sections(c)
    s += "\n## Nodes (`job_nodes_pm100.parquet`)\n\n" + _kv(p["job_nodes"])
    s += "\nRow count check:\n\n" + _kv(p["job_nodes_vs_jobs"])
    s += "\n## Multi-node jobs (allocated nodes, all rows)\n\n" + _kv(p["multi_node"])
    s += "\nCOMPLETED + TIMEOUT, by `nodes_req`:\n\n" + _kv(p["multi_node_completed_timeout"])
    s += "\n## Jobs per node per day\n\n"
    s += ("Definition: a job counts once on each node it was allocated, on the UTC day of its "
          "`start_time`. `incl_zero` = over every (node, day) pair from the first to the last "
          "start day, zeros included; `nonzero` = only pairs with at least one job.\n\n")
    s += _kv(p["jobs_per_node_per_day"])
    s += "\n## Overlap of jobs on the same node (Q2)\n\n"
    s += ("Definition: per node, jobs sorted by (start, end, job_id); a job overlaps if its "
          "start is strictly before the latest end among earlier jobs on that node. "
          "Overlap seconds = min(prev_end, end) − start.\n\n")
    s += _kv(p["overlaps"])
    s += "\nLargest overlaps:\n\n" + _table(p["overlap_examples"])
    s += "\n## Job-name-like field\n\n"
    s += (f"Raw columns containing 'name': {p['name_like_columns'] or 'none'}. "
          "No job-name field exists, so contract `name` is null. For baseline B1 "
          "(PLAN.md §9 Phase 4.4) the fallback is distinct partitions/qos — decision for "
          "Mahesh (not taken in Phase 0).\n")
    return s


def render_fcheck(rows: list[dict[str, Any]]) -> str:
    out = ["| fact | expected | got | compared | match |", "|---|---|---|---|---|"]
    for r in rows:
        exp, got = r["expected"], r["got"]
        if r["fact"] in ("F2_columns", "F3_states"):
            exp, got = "(see config)", ("identical" if r["match"] else str(got))
        out.append(f"| {r['fact']} | {_fmt(exp)} | {_fmt(got)} | {r['compared']} | "
                   f"{'yes' if r['match'] else '**NO**'} |")
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    run = start_run("phase0", args.config, tag="profile")
    cfg, log = run.config, run.log

    raw_dir, sums_file = resolve(cfg["raw_dir"]), resolve(cfg["checksums_file"])
    problems = verify(raw_dir, sums_file)
    if problems:
        log.error("raw checksum problems: %s", problems)
        run.finish("failed: checksums")
        return 1
    sums = read_checksums(sums_file)

    con = _con()
    pcfg = cfg["profile"]
    m: dict[str, Any] = {"checksums": sums}

    log.info("profiling Eagle")
    m["eagle"] = eagle_profile(con, resolve(cfg["eagle"]["raw"]))
    m["eagle"]["columns_used"] = eagle_io.RAW_COLUMNS
    m["eagle"]["common"] = common_profile(con, resolve(cfg["eagle"]["out_jobs"]),
                                          pcfg["quantiles"], pcfg["top_users"])
    fcheck = check_facts(m["eagle"]["F"], cfg["eagle_facts"])
    m["eagle_fact_check"] = fcheck

    log.info("profiling PM100")
    m["pm100"] = pm100_profile(con, resolve(cfg["pm100"]["raw"]),
                               resolve(cfg["pm100"]["out_jobs"]),
                               resolve(cfg["pm100"]["out_job_nodes"]))
    m["pm100"]["common"] = common_profile(con, resolve(cfg["pm100"]["out_jobs"]),
                                          pcfg["quantiles"], pcfg["top_users"])
    m["pm100"]["licence"] = cfg["pm100"]["licence"]
    run.write_json("metrics.json", m)

    cards = resolve("docs/data_cards")
    cards.mkdir(parents=True, exist_ok=True)
    (cards / "eagle.md").write_text(render_eagle_card(m, run.run_id, sums))
    (cards / "pm100.md").write_text(render_pm100_card(m, run.run_id, sums, cfg["pm100"]["licence"]))
    (run.dir / "eagle_fact_check.md").write_text(render_fcheck(fcheck))
    log.info("wrote data cards and %s", run.dir / "eagle_fact_check.md")

    n_bad = sum(not r["match"] for r in fcheck)
    run.finish("ok", {"eagle_fact_mismatches": n_bad,
                      "pm100_overlapping_rows": m["pm100"]["overlaps"]["overlapping_rows"],
                      "pm100_distinct_nodes": m["pm100"]["job_nodes"]["distinct_nodes"]})
    print(render_fcheck(fcheck))
    return 0


if __name__ == "__main__":
    sys.exit(main())
