"""Phase 3 generator: all scenarios to data/processed/scenarios/ (PLAN.md §9 Phase 3; D29–D35).

    uv run python -m hpcdiag.inject.run_inject --config configs/phase3_inject.yaml

Every scenario is checked before it is written: the affected set and multipliers are re-derived
from the truth table alone (inject.verify), the materialized table passes the jobs contract, and
unaffected rows are compared with the base.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from hpcdiag.inject import scenarios as S
from hpcdiag.inject.placement import random_placement
from hpcdiag.inject.store import materialize, write_scenario
from hpcdiag.inject.verify import check_changes
from hpcdiag.utils.checksums import verify
from hpcdiag.utils.run import Run, resolve, run_guarded, start_run

KIND_CODE = {"node": 1, "node_partial": 1, "user_decoy": 2, "system_shock": 3}  # paired draws


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    return run_guarded(start_run("phase3", a.config, tag="inject"), _run)


def _rng(root: int, base_code: int, kind: str, k_pct: int, seed: int) -> np.random.Generator:
    """Keyed generator. The key excludes s and node vs node_partial, so for a given base, K and
    seed every s and both node kinds use the same degraded nodes, onsets and causes (paired)."""
    return np.random.default_rng(np.random.SeedSequence([root, base_code, KIND_CODE[kind], k_pct,
                                                         seed]))


def _per_node_counts(jobs: pd.DataFrame, jn: pd.DataFrame, eligible: np.ndarray) -> dict:
    x = jn[jn["row_id"].isin(jobs.loc[eligible, "row_id"])].groupby("node_id").size()
    return {"nodes": int(len(x)), "min": int(x.min()), "p10": float(x.quantile(0.1)),
            "median": float(x.median()), "p90": float(x.quantile(0.9)), "max": int(x.max())}


def _run(run: Run) -> int:
    cfg, log = run.config, run.log
    problems = verify(resolve(cfg["raw_dir"]), resolve(cfg["checksums_file"]))
    if problems:
        raise RuntimeError(f"raw checksum problems: {problems}")
    d = cfg["data"]
    jobs = pd.read_parquet(resolve(d["jobs"]))
    if not np.array_equal(jobs["row_id"].to_numpy(), np.arange(len(jobs))):
        raise RuntimeError("jobs table not in row_id order")
    jn_real = pd.read_parquet(resolve(d["job_nodes"]))
    freeze = pd.Timestamp(json.loads(resolve(d["freeze_from_run"]).read_text())["freeze_time"])
    gap_rows = json.loads(resolve(d["gaps_from_run"]).read_text())["pm100"]["common"][
        "largest_submit_gaps"]
    gaps = [(pd.Timestamp(g["last_submit_before_gap"]), pd.Timestamp(g["next_submit"]))
            for g in gap_rows]
    eligible = jobs["state_raw"].isin(cfg["eligible_states"]).to_numpy()
    main_pop = eligible & (jobs["run_time_s"] > cfg["main_population_min_run_time_s_exclusive"]
                           ).to_numpy()
    trace_end = jobs.loc[eligible, "end_time"].max()
    oc = cfg["onset"]
    window = S.Window(freeze + pd.Timedelta(days=oc["after_freeze_days"]),
                      trace_end - pd.Timedelta(days=oc["before_trace_end_days"]), gaps)
    log.info("freeze %s; onset window [%s, %s]; trace end %s; %d gaps", freeze, window.start,
             window.end, trace_end, len(gaps))
    out_root = resolve(d["out_dir"])
    if out_root.exists() and any(out_root.iterdir()):
        raise RuntimeError(f"{out_root} is not empty; refusing to overwrite (ask first)")

    # ---- bases (D29, D35)
    bases: list[tuple[str, int, pd.DataFrame, str, list[int]]] = []
    bases.append(("real", 1000, jn_real, d["job_nodes"], list(range(cfg["seeds"]))))
    for k in range(cfg["bases"]["random_seeds"]):
        jn = random_placement(jn_real, jobs, seed=k)
        p = out_root / "_bases" / f"random_{k:02d}" / "job_nodes.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        jn.to_parquet(p, index=False)
        rel = str(p.relative_to(resolve("."))) if p.is_relative_to(resolve(".")) else str(p)
        bases.append((f"rand{k:02d}", k, jn, rel, [k]))
    log.info("bases written: real + %d random", cfg["bases"]["random_seeds"])

    counts_per_node = {b[0]: _per_node_counts(jobs, b[2], eligible) for b in bases}
    rows = []
    t_start = time.time()
    for bname, bcode, jn, jn_path, seeds in bases:
        ends = jobs.loc[eligible & (jobs["end_time"] >= window.start).to_numpy(), "row_id"]
        active = np.sort(jn.loc[jn["row_id"].isin(ends), "node_id"].unique())
        scen_list: list[tuple] = [("null", None, None, None)]          # D35: one null per base
        for seed in seeds:
            for kind in ("node", "node_partial"):
                for kp in cfg["kinds"][kind]["k_pct"]:
                    for s in cfg["kinds"][kind]["s"]:
                        scen_list.append((kind, kp, s, seed))
            for kind in ("user_decoy", "system_shock"):
                for s in cfg["kinds"][kind]["s"]:
                    scen_list.append((kind, None, s, seed))
        for kind, kp, s, seed in scen_list:
            info: dict = {"kind": kind, "base": bname,
                          "base_seed": None if bname == "real" else bcode,
                          "job_nodes": jn_path, "K_pct": kp, "s": s, "seed": seed}
            if kind == "null":
                sid = f"null__{bname}"
                ch, truth = S.empty_changes(), S.empty_truth()
            elif kind in ("node", "node_partial"):
                K = int(np.floor(kp / 100 * len(active) + 0.5))
                sid = f"{kind}__{bname}__K{kp}__s{s}__seed{seed:02d}"
                rng = _rng(cfg["seed"], bcode, kind, kp, seed)
                deg, red = S.draw_degraded_nodes(rng, active, K, window, cfg["causes"],
                                                 oc["max_redraws"])
                ch, truth = S.node_scenario(jobs, jn, deg, s, kind == "node_partial", eligible,
                                            trace_end, sid)
                info.update({"K": K, "n_active_nodes": int(len(active)), "onset_redraws": red})
            elif kind == "user_decoy":
                sid = f"{kind}__{bname}__s{s}__seed{seed:02d}"
                rng = _rng(cfg["seed"], bcode, kind, 0, seed)
                t0, user, red = S.decoy_draw(rng, jobs, main_pop, window,
                                             cfg["kinds"][kind]["min_user_jobs_after_t0"],
                                             oc["max_redraws"])
                ch, truth = S.decoy_scenario(jobs, eligible, t0, user, s, trace_end, sid)
                info.update({"t0": t0, "user": user, "t0_redraws": red})
            else:
                sid = f"{kind}__{bname}__s{s}__seed{seed:02d}"
                rng = _rng(cfg["seed"], bcode, kind, 0, seed)
                t1, red = window.draw(rng, oc["max_redraws"])
                ch, truth = S.shock_scenario(jobs, eligible, t1,
                                             cfg["kinds"][kind]["window_hours"], s, sid)
                info.update({"t1": t1, "t1_redraws": red})
            if kind != "null":
                probs = check_changes(jobs, jn, truth, ch, cfg["eligible_states"])
                if probs:
                    raise AssertionError(f"{sid}: {probs}")
            out = materialize(jobs, ch)               # validates the jobs contract
            keep = ~jobs["row_id"].isin(ch["row_id"]).to_numpy()
            if not out[keep].equals(jobs[keep]):
                raise AssertionError(f"{sid}: unaffected rows changed")
            T0 = jobs["run_time_s"].to_numpy()[ch["row_id"].to_numpy().astype(int)]
            T1 = ch["run_time_s"].to_numpy()
            aff_nodes = set(jn.loc[jn["row_id"].isin(ch["row_id"]), "node_id"].tolist())
            deg_nodes = set(truth["node_id"].dropna().astype(int).tolist())
            row = {"scenario_id": sid, **info,
                   "n_affected": int(len(ch)),
                   "n_affected_main_after": int((T1 > 30).sum()),
                   "n_cross_30s": int(((T0 <= 30) & (T1 > 30)).sum()),
                   "n_censored_by_injection": int(ch["censored_by_injection"].sum()),
                   "n_nodes_touched": len(aff_nodes),
                   "n_degraded_nodes_with_affected_jobs": len(aff_nodes & deg_nodes),
                   "mean_factor_applied": float(ch["factor_applied"].mean()) if len(ch) else np.nan,
                   "main_population_size_after": int((out["state_raw"].isin(cfg["eligible_states"])
                                                      & (out["run_time_s"] > 30)).sum())}
            write_scenario(out_root / sid, ch, truth, {**info, **{k: v for k, v in row.items()
                                                                  if k.startswith("n_")}})
            rows.append(row)
        log.info("base %s done: %d scenarios so far (%.0fs)", bname, len(rows),
                 time.time() - t_start)

    idx = pd.DataFrame(rows)
    idx.to_parquet(out_root / "index.parquet", index=False)
    idx.to_csv(run.dir / "scenario_counts.csv", index=False)
    grid = idx.groupby(["kind", idx["base"].where(idx["base"] == "real", "random")]).size()
    m = {"n_scenarios": int(len(idx)), "freeze_time": str(freeze),
         "onset_window": [str(window.start), str(window.end)], "trace_end": str(trace_end),
         "gaps": [[str(a), str(b)] for a, b in gaps],
         "grid": {f"{k}|{b}": int(v) for (k, b), v in grid.items()},
         "per_node_eligible_job_counts": counts_per_node,
         "base_main_population_size": int(main_pop.sum()),
         "seconds": time.time() - t_start}
    run.write_json("metrics.json", m)
    run.finish("ok", {"n_scenarios": len(idx)})
    log.info("wrote %d scenarios", len(idx))
    return 0


if __name__ == "__main__":
    sys.exit(main())
