"""Phase 5 runner: probing policies on Phase 3 scenarios (docs/PHASE5_DESIGN.md §5-§7, D86-D91).

    uv run python -m hpcdiag.layer3.run_phase5 --config configs/phase5.yaml --mode pilot|full \
        [--workers 7]

Reads the Phase 3 index only. A sealed seed (D77) in the task list raises.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import sys
import time
import zlib

import numpy as np
import pandas as pd

from hpcdiag.layer2 import accumulate as A
from hpcdiag.layer2 import metrics as MT
from hpcdiag.layer2 import prep as P
from hpcdiag.layer2 import run_layer2 as R
from hpcdiag.layer2 import select as SEL
from hpcdiag.layer3 import setup as ST
from hpcdiag.layer3 import simulate as SIM
from hpcdiag.layer3.probes import HYPOTHESES
from hpcdiag.utils.checksums import verify
from hpcdiag.utils.run import Run, load_config, resolve, run_guarded, start_run

BATTERY = {"probes": "battery"}
POLICIES: dict[str, dict] = {
    "ours": {},  # IG per second, rank order, D86
    "ours_design_rule": {"exclusion": "all"},  # D86 sensitivity row
    "ours_skipdone": {"exclusion": "cleared_skipdone"},  # disclosed extra row, see report
    "P-all": dict(BATTERY),
    "P-IG": {"probes": "ig_unit"},
    "P-random": {"probes": "random"},
    "P-randnode": {"nodes": "random"},
    "P-sweep": {"nodes": "sweep", "exclusion": "none", **BATTERY},
}
BASELINES = ("P-all", "P-IG", "P-random", "P-randnode", "P-sweep")
L: dict = {}  # per-process state


def key(x: str) -> int:
    return zlib.crc32(x.encode())


def check_sealed(sids: list[str]) -> None:
    """D77: seeds 20-39 and nulls 200-399 stay sealed."""
    for sid in sids:
        seed = re.search(r"seed(\d+)", sid)
        base = re.search(r"rand(\d+)", sid)
        if (seed and int(seed.group(1)) > 19) or (base and int(base.group(1)) > 199):
            raise RuntimeError(f"sealed scenario in the task list (D77): {sid}")


def _init(config_path: str) -> None:
    cfg = load_config(config_path)
    R._init(cfg["layer2_config"])
    L.update(cfg=cfg, beliefs={}, null=None)


def _belief(name: str) -> SIM.Belief:
    if name not in L["beliefs"]:
        L["beliefs"][name] = ST.make_belief(L["cfg"], ST.VARIANTS[name])
    return L["beliefs"][name]


def _pop_p(jm: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Same steps as the Sc branch of Phase 4's Layer 1: unguarded centred Mode P, W = 7 d."""
    cfg, l1 = R.G["cfg"], R.G["cfg"]["layer1"]
    pop = P.scenario_population(jm, R.G["fr"], R.G["theta"], cfg)
    pop = P.add_centred(pop, l1["centre_window_hours"], l1["centre_min_n"])
    return pop, P.mode_p_on(pop, "score_c", l1["window_hours"], l1["min_ref"])


def layer2(sid: str):
    """Rn(1) snapshots of one scenario (D83)."""
    meta, changes, truth, jm = R._scenario_inputs(sid)
    if meta["kind"] == "null" and L["null"] is not None:
        pop, p = L["null"]
    else:
        pop, p = _pop_p(jm)
        if meta["kind"] == "null":
            L["null"] = (pop, p)
    placement = P.load_placement(meta["job_nodes"])
    pairs = A.make_pairs(pop, placement)
    lc = R.G["cfg"]["layer2"]
    eval_ns = R.G["eval_ns"]
    part = pd.factorize(pop["partition"].astype(str))[0]
    grid = A.block_grid(eval_ns, int(pop["end_ns"].min()))
    ev = A.node_evidence_relative(
        pairs, p, part, R.G["bets"]["Sc"], lc["jumper_J"], 1.0, grid, "node"
    )
    snap = A.snapshots(ev, eval_ns, lc["m_fixed"])
    extra = None
    if meta["kind"] == "node":
        aff = set(changes["row_id"].tolist())
        deg = set(truth["node_id"].astype(int).tolist())
        main_rows = set(pop["row_id"]) & aff
        main_nodes = set(placement.loc[placement["row_id"].isin(main_rows), "node_id"]) & deg
        coalloc = set(placement.loc[placement["row_id"].isin(aff), "node_id"].tolist()) - deg
        ok = np.isfinite(p[pairs["k"].to_numpy()])
        nd, en = pairs["node_id"].to_numpy()[ok], pairs["end_ns"].to_numpy()[ok]
        jobs_end = {int(n): en[nd == n] for n in deg}
        extra = (main_nodes, coalloc, jobs_end)
    return meta, truth, snap, extra


def world_arrays(meta: dict, truth: pd.DataFrame, n_nodes: int) -> tuple:
    cause = np.zeros(n_nodes, int)
    onset = np.full(n_nodes, 10**9)
    rows = []
    if meta["kind"] == "node":
        ns = R.G["eval_ns"]
        for n, o, c in zip(
            truth["node_id"].astype(int), truth["onset_time"], truth["cause"], strict=True
        ):
            cause[n] = HYPOTHESES.index(c)
            onset[n] = int(np.searchsorted(ns, pd.Timestamp(o).value, side="left"))
            rows.append((int(n), int(cause[n]), int(onset[n])))
    return cause, onset, rows


def run_one(sid, meta, truth, snap, extra, variant, base_policy, B) -> dict:
    cfg = L["cfg"]
    lc = R.G["cfg"]["layer2"]
    eval_ns, m = R.G["eval_ns"], lc["m_fixed"]
    t0 = time.time()
    pol = SIM.Policy(f"{base_policy}_B{B}", float(B), **POLICIES[base_policy])
    cause, onset, truth_rows = world_arrays(meta, truth, snap.shape[1])
    s = float(meta["s"]) if meta["kind"] == "node" else 1.0
    bel = _belief(variant)
    wld = ST.make_world(cfg, ST.VARIANTS[variant], s, cause, onset, key(meta["base"]), key(sid))
    rng = np.random.default_rng([cfg["seed"], key(sid), key(variant), key(pol.name)])
    excl = int(cfg["policy"]["exclude_days"] * SIM.BLOCKS_PER_DAY)
    logE, inv = SIM.run_policy(snap, bel, wld, pol, rng, lc["alpha"], m, excl)
    t_sim = time.time() - t0
    flags = SEL.ebh_over_time(logE, lc["alpha"], m)
    out = MT.null_metrics(flags, eval_ns)
    mc = R.G["cfg"]["metrics"]
    if extra is not None:
        out.update(
            MT.node_metrics(
                flags,
                eval_ns,
                truth,
                extra[0],
                extra[1],
                extra[2],
                mc["recall_horizon_days"],
                mc["fdp_offsets_days"],
            )
        )
    days = (eval_ns[-1] - eval_ns[0]) / MT.DAY_NS
    healthy = [r for r in inv if r["state"] == 0]
    sec = float(sum(r["seconds"] for r in inv))
    diag = SIM.diagnosis_table(inv, truth_rows)
    if extra is not None:
        ever = flags.any(axis=0)
        first = {d["node_id"]: d["detected"] for d in out["detections"]}
        for d in diag:
            d["flagged"] = bool(first.get(d["node_id"], False))
            d["flagged_any_time"] = bool(ever[d["node_id"]])
    seqs: dict[str, int] = {}
    for r in inv:
        k = str(len(r["seq"]))
        seqs[k] = seqs.get(k, 0) + 1
    return {
        "scenario_id": sid,
        "kind": meta["kind"],
        "base": meta["base"],
        "s": meta.get("s"),
        "K_pct": meta.get("K_pct"),
        "seed": meta.get("seed"),
        "variant": variant,
        "policy": base_policy,
        "B": B,
        "metrics": out,
        "diagnosis": diag,
        "n_investigations": len(inv),
        "seconds_total": sec,
        "seconds_per_day": sec / days,
        "mean_seconds_per_investigation": sec / len(inv) if inv else None,
        "share_cut": float(np.mean([r["cut"] for r in inv])) if inv else None,
        "healthy_investigations_per_day": len(healthy) / days,
        "healthy_nodes_probed_per_day": len({(r["node"], r["t"] // 4) for r in healthy}) / days,
        "healthy_seconds_per_day": float(sum(r["seconds"] for r in healthy)) / days,
        "probes_per_investigation": seqs,
        "max_logE_final": float(logE[-1].max()),
        "sim_s": t_sim,
        "total_s": time.time() - t0,
    }


def _process(task: tuple[str, list]) -> list[dict]:
    sid, runs = task
    t0 = time.time()
    meta, truth, snap, extra = layer2(sid)
    t_l2 = time.time() - t0
    out = [run_one(sid, meta, truth, snap, extra, *r) for r in runs]
    for r in out:
        r["layer2_s"] = t_l2
    return out


# ----------------------------------------------------------------------------- task lists


def tasks_full(cfg: dict, idx: pd.DataFrame) -> dict[str, list]:
    g = cfg["grid"]
    t: dict[str, list] = {}

    def add(sids, runs):
        for s in sids:
            t.setdefault(s, []).extend(runs)

    rand = idx[idx["base"] != "real"]
    node = rand[rand["kind"] == "node"]
    k1 = node[node["K_pct"] == 1]["scenario_id"].tolist()
    other = rand[rand["kind"].isin(g["validity_kinds"])]["scenario_id"].tolist()
    ours = [("main", "ours", b) for b in [0, *g["budgets_ours"]]]
    add(node["scenario_id"].tolist() + other, ours)
    add(k1, [("main", p, b) for p in BASELINES for b in g["budgets_baselines"]])
    b = g["sens_budget"]
    add(k1, [(v, "ours", b) for v in g["sens_variants"]])
    add(k1, [("main", p, b) for p in g["sens_rules"]])
    add(other, [(v, "ours", b) for v in g["sens_validity_variants"]])
    real = idx[(idx["base"] == "real") & (idx["kind"] == "node") & (idx["K_pct"] == 1)]
    add(real["scenario_id"].tolist(), [("main", "ours", 0), ("main", "ours", b)])
    add(
        idx[(idx["base"] == "real") & (idx["kind"] == "null")]["scenario_id"].tolist(),
        [("main", "ours", 0), ("main", "ours", b)],
    )
    return t


def tasks_pilot(cfg: dict, idx: pd.DataFrame) -> dict[str, list]:
    g = cfg["grid"]
    t = {}
    runs = [("main", "ours", b) for b in [0, *g["budgets_ours"]]]
    runs += [("main", p, 300) for p in BASELINES]
    runs += [("main", p, 300) for p in g["sens_rules"]]
    for k in range(2):
        t[f"node__rand{k:02d}__K1__s1.3__seed{k:02d}"] = runs
        t[f"null__rand{k:02d}"] = runs
    return t


def ordered(t: dict[str, list]) -> list[tuple[str, list]]:
    return sorted(t.items(), key=lambda x: (not x[0].startswith("null__"), x[0]))


# ----------------------------------------------------------------------------- driver


def _run(run: Run, config_path: str, mode: str, workers: int) -> int:
    cfg, log = run.config, run.log
    problems = verify(resolve("data/raw"), resolve("data/raw/CHECKSUMS.txt"))
    if problems:
        raise RuntimeError(f"raw checksum problems: {problems}")
    idx = pd.read_parquet(resolve("data/processed/scenarios/index.parquet"))
    t = tasks_pilot(cfg, idx) if mode == "pilot" else tasks_full(cfg, idx)
    check_sealed(list(t))
    tasks = ordered(t)
    n_runs = sum(len(v) for _, v in tasks)
    log.info("%d scenarios, %d runs, %d workers", len(tasks), n_runs, workers)
    t0 = time.time()
    done = 0
    ctx = mp.get_context("spawn")
    with (
        ctx.Pool(workers, initializer=_init, initargs=(config_path,)) as pool,
        open(run.dir / "results.jsonl", "w") as fh,
    ):
        for rows in pool.imap_unordered(_process, tasks):
            for r in rows:
                fh.write(json.dumps(r, default=R._json) + "\n")
            fh.flush()
            done += len(rows)
            log.info(
                "%s: %d runs, layer2 %.1fs, sim %.1fs; %d/%d at %.0fs",
                rows[0]["scenario_id"],
                len(rows),
                rows[0]["layer2_s"],
                sum(r["total_s"] for r in rows),
                done,
                n_runs,
                time.time() - t0,
            )
    wall = time.time() - t0
    run.write_json("metrics.json", {"mode": mode, "n_runs": n_runs, "wall_s": wall})
    run.finish("ok", {"n_runs": n_runs, "wall_s": round(wall, 1)})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["pilot", "full"], required=True)
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    return run_guarded(
        start_run("phase5", a.config, tag=a.mode), lambda r: _run(r, a.config, a.mode, a.workers)
    )


if __name__ == "__main__":
    sys.exit(main())
