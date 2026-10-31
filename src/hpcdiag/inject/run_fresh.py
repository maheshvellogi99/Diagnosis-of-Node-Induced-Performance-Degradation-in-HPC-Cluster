"""Fresh evaluation scenarios for Phase 4b (D70): seeds 20-39 and null bases 200-399.

    uv run python -m hpcdiag.inject.run_fresh --config configs/phase4b_fresh.yaml

Separate driver so that run_inject.py and configs/phase3_inject.yaml stay unchanged: same keyed RNG
(_rng), same draw and scenario functions, same checks before every write. Nothing that exists is
overwritten, and the new scenarios get their own index file.
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
from hpcdiag.inject.run_inject import _rng
from hpcdiag.inject.store import materialize, write_scenario
from hpcdiag.inject.verify import check_changes
from hpcdiag.utils.checksums import verify
from hpcdiag.utils.run import Run, load_config, resolve, run_guarded, start_run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    return run_guarded(start_run("phase4b", a.config, tag="fresh"), _run)


def _scenario_list(bname: str, seeds: list[int], p3: dict, kinds: list[str]) -> list[tuple]:
    out: list[tuple] = []
    nd = p3["kinds"]["node"]
    for seed in seeds:
        if "node" in kinds:
            for kp in nd["k_pct"]:
                for s in nd["s"] + nd["s_extra"]:
                    out.append(("node", kp, s, seed))
        for kind in ("user_decoy", "system_shock", "user_decoy_heavy"):
            if kind in kinds:
                for s in p3["kinds"][kind]["s"]:
                    out.append((kind, None, s, seed))
    return out


def _run(run: Run) -> int:
    cfg, log = run.config, run.log
    p3 = load_config(resolve(cfg["phase3_config"]))
    if p3["seed"] != cfg["seed"]:
        raise RuntimeError("root seed must equal the Phase 3 root seed")
    problems = verify(resolve(p3["raw_dir"]), resolve(p3["checksums_file"]))
    if problems:
        raise RuntimeError(f"raw checksum problems: {problems}")
    d = p3["data"]
    jobs = pd.read_parquet(resolve(d["jobs"]))
    if not np.array_equal(jobs["row_id"].to_numpy(), np.arange(len(jobs))):
        raise RuntimeError("jobs table not in row_id order")
    jn_real = pd.read_parquet(resolve(d["job_nodes"]))
    freeze = pd.Timestamp(json.loads(resolve(d["freeze_from_run"]).read_text())["freeze_time"])
    gap_rows = json.loads(resolve(d["gaps_from_run"]).read_text())["pm100"]["common"][
        "largest_submit_gaps"
    ]
    gaps = [
        (pd.Timestamp(g["last_submit_before_gap"]), pd.Timestamp(g["next_submit"]))
        for g in gap_rows
    ]
    eligible = jobs["state_raw"].isin(p3["eligible_states"]).to_numpy()
    main_pop = (
        eligible & (jobs["run_time_s"] > p3["main_population_min_run_time_s_exclusive"]).to_numpy()
    )
    trace_end = jobs.loc[eligible, "end_time"].max()
    oc = p3["onset"]
    window = S.Window(
        freeze + pd.Timedelta(days=oc["after_freeze_days"]),
        trace_end - pd.Timedelta(days=oc["before_trace_end_days"]),
        gaps,
    )
    root = resolve(d["out_dir"])
    index_path = root / cfg["index_file"]
    if not (root / "index.parquet").exists():
        raise RuntimeError("Phase 3 scenarios missing")
    if index_path.exists():
        raise RuntimeError(f"{index_path} exists; refusing to overwrite")
    old_ids = set(pd.read_parquet(root / "index.parquet", columns=["scenario_id"])["scenario_id"])

    def base_path(k: int):
        return root / "_bases" / f"random_{k:02d}" / "job_nodes.parquet"

    def rel(p) -> str:
        return str(p.relative_to(resolve(".")))

    lo, hi = cfg["seeds"]
    seeds = list(range(lo, hi))
    bases: list[tuple[str, int, pd.DataFrame, str, list[int], bool]] = [
        ("real", 1000, jn_real, d["job_nodes"], seeds, False)
    ]
    for k in seeds:  # existing null bases of Phase 3: read back and checked against their seed
        jn = pd.read_parquet(base_path(k))
        if not jn.equals(random_placement(jn_real, jobs, seed=k)):
            raise AssertionError(f"stored base {k} differs from its seed")
        bases.append((f"rand{k:02d}", k, jn, rel(base_path(k)), [k], False))
    for k in range(*cfg["null_bases"]):
        p = base_path(k)
        if p.exists():
            raise RuntimeError(f"{p} exists; refusing to overwrite")
        jn = random_placement(jn_real, jobs, seed=k)
        p.parent.mkdir(parents=True, exist_ok=True)
        jn.to_parquet(p, index=False)
        bases.append((f"rand{k:02d}", k, jn, rel(p), [], True))
    log.info("%d bases ready", len(bases))

    rows = []
    t_start = time.time()
    for bname, bcode, jn, jn_path, b_seeds, null_only in bases:
        ends = jobs.loc[eligible & (jobs["end_time"] >= window.start).to_numpy(), "row_id"]
        active = np.sort(jn.loc[jn["row_id"].isin(ends), "node_id"].unique())
        scen = [("null", None, None, None)] if null_only else []
        scen += _scenario_list(bname, b_seeds, p3, cfg["kinds"])
        for kind, kp, s, seed in scen:
            info: dict = {
                "kind": kind,
                "base": bname,
                "base_seed": None if bname == "real" else bcode,
                "job_nodes": jn_path,
                "K_pct": kp,
                "s": s,
                "seed": seed,
            }
            if kind == "null":
                sid = f"null__{bname}"
                ch, truth = S.empty_changes(), S.empty_truth()
            elif kind == "node":
                K = int(np.floor(kp / 100 * len(active) + 0.5))
                sid = f"node__{bname}__K{kp}__s{s}__seed{seed:02d}"
                rng = _rng(cfg["seed"], bcode, kind, kp, seed)
                deg, red = S.draw_degraded_nodes(
                    rng, active, K, window, p3["causes"], oc["max_redraws"]
                )
                ch, truth = S.node_scenario(jobs, jn, deg, s, False, eligible, trace_end, sid)
                info.update({"K": K, "n_active_nodes": int(len(active)), "onset_redraws": red})
            elif kind == "user_decoy":
                sid = f"{kind}__{bname}__s{s}__seed{seed:02d}"
                rng = _rng(cfg["seed"], bcode, kind, 0, seed)
                t0, user, red = S.decoy_draw(
                    rng,
                    jobs,
                    main_pop,
                    window,
                    p3["kinds"][kind]["min_user_jobs_after_t0"],
                    oc["max_redraws"],
                )
                ch, truth = S.decoy_scenario(jobs, eligible, t0, user, s, trace_end, sid)
                info.update({"t0": t0, "user": user, "t0_redraws": red})
            elif kind == "user_decoy_heavy":
                sid = f"{kind}__{bname}__s{s}__seed{seed:02d}"
                rng = _rng(cfg["seed"], bcode, kind, 0, seed)
                t0, user, rank, n_user, red = S.heavy_decoy_draw(
                    rng, jobs, main_pop, window, p3["kinds"][kind]["top_users"], oc["max_redraws"]
                )
                ch, truth = S.decoy_scenario(
                    jobs, eligible, t0, user, s, trace_end, sid, kind="user_decoy_heavy"
                )
                info.update(
                    {
                        "t0": t0,
                        "user": user,
                        "user_rank_after_t0": rank,
                        "user_main_jobs_after_t0": n_user,
                        "t0_redraws": red,
                    }
                )
            else:
                sid = f"{kind}__{bname}__s{s}__seed{seed:02d}"
                rng = _rng(cfg["seed"], bcode, kind, 0, seed)
                t1, red = window.draw(rng, oc["max_redraws"])
                ch, truth = S.shock_scenario(
                    jobs, eligible, t1, p3["kinds"][kind]["window_hours"], s, sid
                )
                info.update({"t1": t1, "t1_redraws": red})
            if sid in old_ids or (root / sid).exists():
                raise RuntimeError(f"scenario {sid} exists; refusing to overwrite")
            if kind != "null":
                probs = check_changes(jobs, jn, truth, ch, p3["eligible_states"])
                if probs:
                    raise AssertionError(f"{sid}: {probs}")
            out = materialize(jobs, ch)  # validates the jobs contract
            keep = ~jobs["row_id"].isin(ch["row_id"]).to_numpy()
            if not out[keep].equals(jobs[keep]):
                raise AssertionError(f"{sid}: unaffected rows changed")
            T0 = jobs["run_time_s"].to_numpy()[ch["row_id"].to_numpy().astype(int)]
            T1 = ch["run_time_s"].to_numpy()
            aff_nodes = set(jn.loc[jn["row_id"].isin(ch["row_id"]), "node_id"].tolist())
            deg_nodes = set(truth["node_id"].dropna().astype(int).tolist())
            row = {
                "scenario_id": sid,
                **info,
                "n_affected": int(len(ch)),
                "n_affected_main_after": int((T1 > 30).sum()),
                "n_cross_30s": int(((T0 <= 30) & (T1 > 30)).sum()),
                "n_censored_by_injection": int(ch["censored_by_injection"].sum()),
                "n_nodes_touched": len(aff_nodes),
                "n_degraded_nodes_with_affected_jobs": len(aff_nodes & deg_nodes),
                "mean_factor_applied": float(ch["factor_applied"].mean()) if len(ch) else np.nan,
            }
            write_scenario(
                root / sid,
                ch,
                truth,
                {**info, **{k: v for k, v in row.items() if k.startswith("n_")}},
            )
            rows.append(row)
        if not null_only or bcode % 50 == 0:
            log.info("base %s done: %d scenarios (%.0fs)", bname, len(rows), time.time() - t_start)
    idx = pd.DataFrame(rows)
    idx.to_csv(run.dir / "scenario_counts.csv", index=False)
    idx.to_parquet(index_path, index=False)
    base_kind = idx["base"].where(idx["base"] == "real", "random")
    grid = idx.groupby(["kind", base_kind]).size()
    run.write_json(
        "metrics.json",
        {
            "n_scenarios": int(len(idx)),
            "grid": {f"{k}|{b}": int(v) for (k, b), v in grid.items()},
            "onset_window": [str(window.start), str(window.end)],
            "seconds": time.time() - t_start,
        },
    )
    run.finish("ok", {"n_scenarios": len(idx)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
