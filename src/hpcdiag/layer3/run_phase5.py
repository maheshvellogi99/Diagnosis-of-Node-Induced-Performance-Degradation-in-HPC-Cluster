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

BATTERY = {"probes": "battery", "exclusion": "cleared"}
D86 = {"exclusion": "cleared"}
POLICIES: dict[str, dict] = {
    # Phase 5 names (D86 primary then), kept so the Phase 5 task lists mean what they meant
    "ours": dict(D86),
    "ours_design_rule": {"exclusion": "all"},
    "ours_skipdone": {"exclusion": "cleared_skipdone"},
    "P-all": dict(BATTERY),
    "P-IG": {"probes": "ig_unit", **D86},
    "P-random": {"probes": "random", **D86},
    "P-randnode": {"nodes": "random", **D86},
    "P-sweep": {"nodes": "sweep", "exclusion": "none", "probes": "battery"},
    # Phase 5b (D95 rule primary)
    "ig": {"exclusion": "cleared_skipdone"},  # IG per second only
    "stage": {"exclusion": "cleared_skipdone", "probes": "stage"},  # D97 with D103
    "d86": dict(D86),  # the architect's failed rule, reported row (D95)
    "design_rule": {"exclusion": "all"},
    # Phase 5c (D109): the primary, probe evidence as a restart mixture with J from the config
    "ig_d109": {"exclusion": "cleared_skipdone", "probe_J": "config"},
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


def growth_summary(v: list[float]) -> dict:
    """D102 (ii): fixed-pool healthy growth of the bets used on healthy nodes in one run."""
    if not v:
        return {"n": 0}
    a = np.asarray(v)
    return {
        "n": len(a),
        "max": float(a.max()),
        "q99": float(np.quantile(a, 0.99)),
        "mean": float(a.mean()),
        "n_gt_0": int((a > 0).sum()),
    }


def bel_names(cfg: dict) -> list[str]:
    return list(cfg["probes"]["names"])


def run_one(sid, meta, truth, snap, extra, variant, base_policy, B, k_pre=0) -> dict:
    cfg = L["cfg"]
    lc = R.G["cfg"]["layer2"]
    eval_ns, m = R.G["eval_ns"], lc["m_fixed"]
    t0 = time.time()
    kw = dict(POLICIES[base_policy])
    if kw.get("probe_J") == "config":
        kw["probe_J"] = float(cfg["evalue"]["probe_J"])
    pol = SIM.Policy(f"{base_policy}_B{B}", float(B), **kw)
    cause, onset, truth_rows = world_arrays(meta, truth, snap.shape[1])
    s = float(meta["s"]) if meta["kind"] == "node" else 1.0
    bel = _belief(variant)
    wld = ST.make_world(cfg, ST.VARIANTS[variant], s, cause, onset, key(meta["base"]), key(sid))
    rng = np.random.default_rng([cfg["seed"], key(sid), key(variant), key(pol.name)])
    excl = int(cfg["policy"]["exclude_days"] * SIM.BLOCKS_PER_DAY)
    pre = {n: int(k_pre) for n, _, _ in truth_rows} if k_pre else None  # D110 stress test
    logE, inv = SIM.run_policy(
        snap, bel, wld, pol, rng, lc["alpha"], m, excl, log_growth=True, pre=pre
    )
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
    names = [*bel_names(cfg), "tel"]
    used = dict.fromkeys(names, 0)
    second = dict.fromkeys(names, 0)  # probe chosen right after the first one
    for r in inv:
        k = str(len(r["seq"]))
        seqs[k] = seqs.get(k, 0) + 1
        for a in r["seq"]:
            used[names[a]] += 1
        if len(r["seq"]) > 1:
            second[names[r["seq"][1]]] += 1
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
        "probe_J": pol.probe_J,
        "k_pre": int(k_pre),
        "metrics": out,
        "diagnosis": diag,
        "n_investigations": len(inv),
        "seconds_total": sec,
        "seconds_per_day": sec / days,
        "mean_seconds_per_investigation": sec / len(inv) if inv else None,
        "share_cut": float(np.mean([r["cut"] for r in inv])) if inv else None,
        "n_uncut": sum(not r["cut"] for r in inv),
        "seconds_uncut": float(sum(r["seconds"] for r in inv if not r["cut"])),
        "n_cleared": sum(r["pi_h0"] >= bel.tau for r in inv),
        "healthy_investigations_per_day": len(healthy) / days,
        "healthy_nodes_probed_per_day": len({(r["node"], r["t"] // 4) for r in healthy}) / days,
        "healthy_seconds_per_day": float(sum(r["seconds"] for r in healthy)) / days,
        "probes_per_investigation": seqs,
        "probe_counts": used,
        "second_probe": second,
        "max_logE_final": float(logE[-1].max()),
        "healthy_growth": growth_summary([g for r in inv for g in r["healthy_growth"]]),
        "investigations_unflagged": sum(not r["flagged"] for r in inv),
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


def tasks_addon(cfg: dict, idx: pd.DataFrame) -> dict[str, list]:
    """Null bases (lost to an unquoted YAML null in the first full run), Gate 5a for the two other
    investigation rules, and telemetry sensitivities on a rule that is not starved of budget."""
    g = cfg["grid"]
    t: dict[str, list] = {}

    def add(sids, runs):
        for s in sids:
            t.setdefault(s, []).extend(runs)

    rand = idx[idx["base"] != "real"]
    nulls = rand[rand["kind"] == "null"]["scenario_id"].tolist()
    decoys = rand[rand["kind"].isin(g["validity_kinds"]) & (rand["kind"] != "null")]
    node = rand[rand["kind"] == "node"]
    k1 = node[node["K_pct"] == 1]["scenario_id"].tolist()
    b = g["sens_budget"]
    rules = [("main", p, b) for p in g["sens_rules"]]
    add(nulls, [("main", "ours", x) for x in [0, *g["budgets_ours"]]])
    add(nulls, [(v, "ours", b) for v in g["sens_validity_variants"]])
    add(nulls + decoys["scenario_id"].tolist() + node["scenario_id"].tolist(), rules)
    add(k1, [("main", "ours_skipdone", x) for x in g["budgets_baselines"] + [7200] if x != b])
    tel = ("tel_crossblock", "tel_design", "tel_transfer")
    add(k1, [(v, "ours_skipdone", b) for v in tel] + [(v, "ours", 7200) for v in tel])
    add(k1, [("main", "ours", b), ("main", "ours", 7200)])
    return t


def tasks_5b(cfg: dict, idx: pd.DataFrame) -> dict[str, list]:
    """Phase 5b development grid (D95-D98, D103). Reads index.parquet only; seeds 0-19."""
    g = cfg["grid5b"]
    t: dict[str, list] = {}

    def add(sids, runs):
        for s in sids:
            t.setdefault(s, []).extend(runs)

    rand = idx[idx["base"] != "real"]
    node = rand[rand["kind"] == "node"]
    nodes = node["scenario_id"].tolist()
    k1 = node[node["K_pct"] == 1]["scenario_id"].tolist()
    valid = rand[rand["kind"].isin(g["validity_kinds"])]["scenario_id"].tolist()
    rules, Bs = g["probe_rules"], g["budgets"]
    pos = [b for b in Bs if b > 0]
    add(nodes + valid, [("main", "ig", 0)] + [("main", r, b) for r in rules for b in pos])
    add(valid, [("pool_redraw", r, b) for r in rules for b in pos])
    add(k1, [("main", "d86", b) for b in g["d86_budgets"]])
    add(k1, [("main", "design_rule", b) for b in g["design_rule_budgets"]])
    add(k1, [(v, r, b) for v in g["k1_variants"] for r in rules for b in g["k1_variant_budgets"]])
    add(valid + k1, [(v, r, b) for v in g["misspec_variants"] for r in rules
                     for b in g["misspec_budgets"]])  # fmt: skip
    real = idx[
        (idx["base"] == "real")
        & (((idx["kind"] == "node") & (idx["K_pct"] == 1)) | (idx["kind"] == "null"))
    ]
    add(
        real["scenario_id"].tolist(),
        [("main", "ig", 0)] + [("main", r, b) for r in rules for b in g["real_budgets"]],
    )
    return t


def tasks_5c(cfg: dict, idx: pd.DataFrame) -> dict[str, list]:
    """Phase 5c development evaluation (D109-D111). Reads index.parquet only; seeds 0-19.
    The old-formula reference rows are the dev5b grid's "ig" rows; a re-run of part of them here
    checks that they reproduce."""
    g = cfg["grid5c"]
    t: dict[str, list] = {}

    def add(sids, runs):
        for s in sids:
            t.setdefault(s, []).extend(runs)

    rand = idx[idx["base"] != "real"]
    node = rand[rand["kind"] == "node"]
    nodes = node["scenario_id"].tolist()
    k1 = node[node["K_pct"] == 1]
    k1_ids = k1["scenario_id"].tolist()
    k1s13 = k1[k1["s"] == 1.3]["scenario_id"].tolist()
    valid = rand[rand["kind"].isin(g["validity_kinds"])]["scenario_id"].tolist()
    nulls = rand[rand["kind"] == "null"]["scenario_id"].tolist()
    pos = [b for b in g["budgets"] if b > 0]
    add(nodes + valid, [("main", "ig", 0)] + [("main", "ig_d109", b) for b in pos])
    add(valid, [("pool_redraw", "ig_d109", b) for b in pos])
    add(valid + k1_ids, [(v, "ig_d109", b) for v in g["misspec_variants"]
                         for b in g["misspec_budgets"]])  # fmt: skip
    add(k1s13 + nulls, [("main", "ig", g["repro_budget"])])
    add(k1s13, [("main", p, g["stress_budget"], k) for p in ("ig", "ig_d109")
                for k in g["stress_k_pre"]])  # fmt: skip
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
    t = {
        "pilot": tasks_pilot,
        "full": tasks_full,
        "addon": tasks_addon,
        "dev5b": tasks_5b,
        "dev5c": tasks_5c,
    }[mode](cfg, idx)
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
    ap.add_argument("--mode", choices=["pilot", "full", "addon", "dev5b", "dev5c"], required=True)
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    return run_guarded(
        start_run("phase5", a.config, tag=a.mode), lambda r: _run(r, a.config, a.mode, a.workers)
    )


if __name__ == "__main__":
    sys.exit(main())
