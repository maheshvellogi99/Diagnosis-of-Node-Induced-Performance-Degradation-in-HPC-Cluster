"""Phase 4 runner: Layer 2 on Phase 3 scenarios (PLAN.md §9 Phase 4; D38–D58).

    uv run python -m hpcdiag.layer2.run_layer2 --config configs/phase4_layer2.yaml --mode pilot \
        [--workers 7]

Variant sets (D58):
- "ablation": jumper (primary) / plain product / jumper + decoy weight, each with guard none, (a),
  (b); B6 sensitivity (Mode P W = 1 d, 365 d, Mode E as-run; jumper, no guard); baselines B1-B5.
- "primary": jumper with guard none, (a), (b); baselines B1-B5.
Every variant uses e-BH with fixed m (D50) at alpha (and the alpha curve) every 6 h.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time

import numpy as np
import pandas as pd

from hpcdiag.inject.store import materialize
from hpcdiag.layer2 import accumulate as A
from hpcdiag.layer2 import metrics as MT
from hpcdiag.layer2 import prep as P
from hpcdiag.layer2 import select as SEL
from hpcdiag.utils.checksums import verify
from hpcdiag.utils.run import Run, load_config, resolve, run_guarded, start_run

H_NS = 3600 * 10**9
DAY_NS = 24 * H_NS
G: dict = {}  # per-process globals (filled by _init)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["pilot", "pilot2"], required=True)
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    run = start_run("phase4", a.config, tag=a.mode)
    return run_guarded(run, lambda r: _run(r, a.config, a.mode, a.workers))


# ----------------------------------------------------------------------------- worker side


def _init(config_path: str) -> None:
    cfg = load_config(config_path)
    jobs = pd.read_parquet(resolve(cfg["data"]["jobs"]))
    fr = P.frozen_predictions(cfg, jobs)
    theta = P.theta_by_row(len(jobs), cfg["theta_seed"])
    elig = jobs["state_raw"].isin(cfg["population"]["states"])
    trace_end = jobs.loc[elig, "end_time"].max()
    eval_start = fr.freeze + pd.Timedelta(days=7)
    step = int(cfg["layer2"]["eval_every_hours"] * H_NS)
    eval_ns = np.arange(eval_start.value, trace_end.value + 1, step, dtype=np.int64)
    day_ns = np.arange(eval_start.value, trace_end.value + 1, DAY_NS, dtype=np.int64)
    G.update(
        cfg=cfg,
        jobs=jobs,
        fr=fr,
        theta=theta,
        eval_ns=eval_ns,
        day_ns=day_ns,
        null_cache={},
        decoy_cache={},
    )
    G["bets"] = _build_bets() if "shift_bets" in cfg else {}


def _null_pop() -> pd.DataFrame:
    """Population of the unmodified trace with centred scores (shared by every null)."""
    l1 = G["cfg"]["layer1"]
    pop = P.scenario_population(G["jobs"], G["fr"], G["theta"], G["cfg"])
    return P.add_centred(pop, l1["centre_window_hours"], l1["centre_min_n"])


def _build_bets() -> dict:
    """D61: shift-LR histogram bets from the warm-up week [freeze, freeze + 7 d] of the
    unmodified trace (identical in every scenario: no injected job ends before freeze + 7 d)."""
    from hpcdiag.layer2.bets import build_shift_bets

    l1, sb = G["cfg"]["layer1"], G["cfg"]["shift_bets"]
    pop = _null_pop()
    fz = G["fr"].freeze.value
    kw = dict(
        window_h=l1["window_hours"], min_ref=l1["min_ref"], bins=sb["bins"], mode_p_on=P.mode_p_on
    )
    return {
        "S": build_shift_bets(pop, "score", fz, sb["s0"], **kw),
        "Sc": build_shift_bets(pop, "score_c", fz, sb["s0"], **kw),
        "Sc_a": build_shift_bets(
            pop, "score_c", fz, sb["s0"], detrend_hours=l1["guard_a_hours"], **kw
        ),
    }


def _layer1(jobs_mat: pd.DataFrame, variant_set: str) -> tuple[pd.DataFrame, dict, dict]:
    cfg, l1 = G["cfg"], G["cfg"]["layer1"]
    t = {}
    t0 = time.time()
    pop = P.scenario_population(jobs_mat, G["fr"], G["theta"], cfg)
    t["population"] = time.time() - t0
    ps = {}
    t0 = time.time()
    ps["P7"] = P.mode_p(pop, l1["window_hours"], l1["min_ref"])
    t["modeP7"] = time.time() - t0
    t0 = time.time()
    ps["P7_a"] = P.mode_p(pop, l1["window_hours"], l1["min_ref"], l1["guard_a_hours"])
    t["modeP7_guard_a"] = time.time() - t0
    if variant_set == "pilot2":
        t0 = time.time()
        pop = P.add_centred(pop, l1["centre_window_hours"], l1["centre_min_n"])
        ps["Sc"] = P.mode_p_on(pop, "score_c", l1["window_hours"], l1["min_ref"])
        ps["Sc_a"] = P.mode_p_on(
            pop, "score_c", l1["window_hours"], l1["min_ref"], l1["guard_a_hours"]
        )
        t["centred"] = time.time() - t0
    if variant_set == "ablation":
        t0 = time.time()
        for w in l1["sensitivity_windows_hours"]:
            ps[f"P{w}h"] = P.mode_p(pop, w, l1["min_ref"])
        ps["E"] = P.mode_e_asrun(pop)
        t["sensitivity_l1"] = time.time() - t0
    return pop, ps, t


def _variants(variant_set: str) -> list[tuple[str, str, str, float | None, bool, str]]:
    """(name, p key, guard, J, decoy weight, bet: 'kappa' or a key of G['bets'])."""
    J = G["cfg"]["layer2"]["jumper_J"]
    if variant_set == "pilot2":  # D60/D61 primary + ablations
        return [
            ("lr_Sc_none", "Sc", "none", J, False, "Sc"),
            ("lr_Sc_a", "Sc_a", "a", J, False, "Sc_a"),
            ("lr_Sc_b", "Sc", "b", J, False, "Sc"),
            ("kappa_Sc_none", "Sc", "none", J, False, "kappa"),
            ("lr_S_none", "P7", "none", J, False, "S"),
            ("kappa_S_none", "P7", "none", J, False, "kappa"),
        ]
    out = []
    for g in ("none", "a", "b"):
        pk = "P7_a" if g == "a" else "P7"
        out.append((f"jumper_{g}", pk, g, J, False, "kappa"))
        if variant_set == "ablation":
            out.append((f"plain_{g}", pk, g, None, False, "kappa"))
            out.append((f"jumper_decoyw_{g}", pk, g, J, True, "kappa"))
    if variant_set == "ablation":
        for w in G["cfg"]["layer1"]["sensitivity_windows_hours"]:
            out.append((f"jumper_none_P{w}h", f"P{w}h", "none", J, False, "kappa"))
        out.append(("jumper_none_modeE", "E", "none", J, False, "kappa"))
    return out


def _scenario_inputs(sid: str):
    cfg = G["cfg"]
    d = resolve(cfg["data"]["scenarios"]) / sid
    meta = json.loads((d / "scenario.json").read_text())
    changes = pd.read_parquet(d / "changes.parquet")
    truth = pd.read_parquet(d / "truth.parquet")
    jobs_mat = materialize(G["jobs"], changes, validate=False)
    return meta, changes, truth, jobs_mat


def _process(task: tuple[str, str]) -> dict:
    """task = (scenario_id, variant_set)."""
    sid, vset = task
    cfg, lc = G["cfg"], G["cfg"]["layer2"]
    T: dict = {}
    t_all = time.time()
    meta, changes, truth, jobs_mat = _scenario_inputs(sid)
    kind = meta["kind"]
    # Layer 1 (shared for nulls: p-values do not depend on placement, D37)
    t0 = time.time()
    if kind == "null" and vset in G["null_cache"]:
        pop, ps, tl1 = G["null_cache"][vset]
        tl1 = {"cached": 0.0}
    else:
        pop, ps, tl1 = _layer1(jobs_mat, vset)
        if kind == "null":
            G["null_cache"][vset] = (pop, ps, tl1)
    T["layer1"] = time.time() - t0
    T["layer1_parts"] = tl1
    # placement, pairs, weights
    t0 = time.time()
    placement = P.load_placement(meta["job_nodes"])
    pairs = A.make_pairs(pop, placement)
    w_base = A.weights(pairs)
    w_decoy = None
    if vset == "ablation":
        key = meta["job_nodes"]
        if key not in G["decoy_cache"]:
            G["decoy_cache"] = {key: A.decoy_counts(placement, G["jobs"], lc["decoy_weight_hours"])}
        w_decoy = A.weights(pairs, G["decoy_cache"][key])
    end_ns = pop["end_ns"].to_numpy()
    frozen = {}
    T["pairs"] = time.time() - t0
    eval_ns, m = G["eval_ns"], lc["m_fixed"]
    if placement["node_id"].max() >= m:
        raise RuntimeError("node id outside the fixed m")
    res: dict = {
        "scenario_id": sid,
        "kind": kind,
        "base": meta["base"],
        "s": meta.get("s"),
        "K_pct": meta.get("K_pct"),
        "seed": meta.get("seed"),
        "n_pop": int(len(pop)),
        "variants": {},
    }
    # truth helpers for node kinds
    node_kind = kind in ("node", "node_partial")
    if node_kind:
        aff = set(changes["row_id"].tolist())
        aff_pairs = placement[placement["row_id"].isin(aff)]
        deg = set(truth["node_id"].astype(int).tolist())
        aff_main_rows = set(pop["row_id"]) & aff
        aff_main_nodes = (
            set(placement.loc[placement["row_id"].isin(aff_main_rows), "node_id"]) & deg
        )
        coalloc = set(aff_pairs["node_id"].tolist()) - deg
    T["variants"] = {}
    for name, pkey, guard, J, dw, bet in _variants(vset):
        t0 = time.time()
        p = ps[pkey]
        fz = None
        if guard == "b":
            if pkey not in frozen:
                gb = lc["guard_b"]
                frozen[pkey] = A.guard_b_frozen(p, end_ns, gb["J"], gb["ratio"])
            fz = frozen[pkey]
        bets = lc["kappas"] if bet == "kappa" else G["bets"][bet]
        ev = A.node_evidence(pairs, p, w_decoy if dw else w_base, bets, J, fz)
        snap = A.snapshots(ev, eval_ns, m)
        out = _evaluate(
            snap,
            eval_ns,
            truth,
            node_kind,
            pairs,
            p,
            extra=((aff_main_nodes, coalloc) if node_kind else None),
        )
        if guard == "b":
            out["frozen_share"] = float(fz.mean())
        res["variants"][name] = out
        T["variants"][name] = time.time() - t0
    # baselines (unguarded Mode P W = 7 d; on S_c with the LR primary in pilot2)
    t0 = time.time()
    p = ps["Sc"] if vset == "pilot2" else ps["P7"]
    b5_bets = G["bets"]["Sc"] if vset == "pilot2" else lc["kappas"]
    cs = SEL.cumulative_node_stats(pairs, p, lc["kappas"])
    fis = A.snapshots(cs, eval_ns, m, "fis", 0.0)
    cnt = A.snapshots(cs, eval_ns, m, "cnt", 0.0)
    sf = A.snapshots(cs, eval_ns, m, "f", 0.0)
    ev_primary = A.node_evidence(pairs, p, w_base, b5_bets, lc["jumper_J"])
    lm = A.snapshots(ev_primary, eval_ns, m)
    rel = (pop["run_time_s"].to_numpy() - np.maximum(pop["that"].to_numpy(), 1.0)) / np.maximum(
        pop["that"].to_numpy(), 1.0
    )
    rb = SEL.residual_baselines(pairs, rel, pop["user"].astype(str).to_numpy(), G["day_ns"], m, cfg)
    alpha = lc["alpha"]
    base_flags = {
        "B3_fisher_bh": (SEL.fisher_bh_flags(fis, cnt, alpha, m), eval_ns),
        "B4_averaged_e": (SEL.averaged_e_flags(sf, cnt, alpha, m), eval_ns),
        "B5_ville": (SEL.ville_flags(lm, alpha), eval_ns),
        "B1_guide": (rb["B1"], G["day_ns"]),
        "B2_top5": (rb["B2"], G["day_ns"]),
    }
    for name, (fl, grid) in base_flags.items():
        res["variants"][name] = _metrics_from_flags(
            fl, grid, truth, node_kind, pairs, p, (aff_main_nodes, coalloc) if node_kind else None
        )
    T["baselines"] = time.time() - t0
    T["total"] = time.time() - t_all
    res["timing_s"] = T
    return res


def _metrics_from_flags(flags, grid, truth, node_kind, pairs, p, extra) -> dict:
    out = MT.null_metrics(flags, grid)
    if node_kind:
        aff_main_nodes, coalloc = extra
        pk = pairs["k"].to_numpy()
        ok = np.isfinite(p[pk])
        nd, en = pairs["node_id"].to_numpy()[ok], pairs["end_ns"].to_numpy()[ok]
        jobs_end = {int(n): en[nd == n] for n in truth["node_id"].astype(int)}
        mc = G["cfg"]["metrics"]
        out.update(
            MT.node_metrics(
                flags,
                grid,
                truth,
                aff_main_nodes,
                coalloc,
                jobs_end,
                mc["recall_horizon_days"],
                mc["fdp_offsets_days"],
            )
        )
    return out


def _evaluate(snap, eval_ns, truth, node_kind, pairs, p, extra) -> dict:
    lc = G["cfg"]["layer2"]
    m = lc["m_fixed"]
    out = _metrics_from_flags(
        SEL.ebh_over_time(snap, lc["alpha"], m), eval_ns, truth, node_kind, pairs, p, extra
    )
    curve = {}
    for a in lc["alphas_curve"]:
        if a == lc["alpha"]:
            continue
        fl = SEL.ebh_over_time(snap, a, m)
        c = MT.null_metrics(fl, eval_ns)
        if node_kind:
            nm = _metrics_from_flags(fl, eval_ns, truth, True, pairs, p, extra)
            c.update({k: nm[k] for k in ("recall_a", "recall_b", "fdp_mean_over_time")})
        curve[str(a)] = c
    out["alpha_curve"] = curve
    out["max_logM_final"] = float(snap[-1].max())
    return out


# ----------------------------------------------------------------------------- driver


def _tasks_pilot(cfg: dict) -> list[tuple[str, str]]:
    pc = cfg["pilot"]
    t = [(f"null__rand{k:02d}", "ablation") for k in range(pc["null_bases"])]
    nd = pc["node"]
    t += [
        (f"{nd['kind']}__rand{s:02d}__K{nd['K_pct']}__s{nd['s']}__seed{s:02d}", "ablation")
        for s in range(nd["seeds"])
    ]
    return t


def _tasks_pilot2(cfg: dict) -> list[tuple[str, str]]:
    pc = cfg["pilot2"]
    t = [(f"null__rand{k:02d}", "pilot2") for k in range(pc["null_bases"])]
    t += [
        (f"node__rand{k:02d}__K{pc['K_pct']}__s{s}__seed{k:02d}", "pilot2")
        for s in pc["s"]
        for k in pc["seeds"]
    ]
    return t


def _pilot2_diagnostics(cfg: dict, log) -> dict:
    """D59 paired check (raw S and S_c) and D62 detectability table, in the main process."""
    from hpcdiag.layer2.power import detectability, paired_check

    l1, pc = cfg["layer1"], cfg["pilot2"]
    null = _null_pop()
    pn = {
        "S": pd.Series(
            P.mode_p_on(null, "score", l1["window_hours"], l1["min_ref"]), index=null["row_id"]
        ),
        "Sc": pd.Series(
            P.mode_p_on(null, "score_c", l1["window_hours"], l1["min_ref"]), index=null["row_id"]
        ),
    }
    d59 = []
    for s in pc["s"]:
        for k in pc["seeds"]:
            sid = f"node__rand{k:02d}__K{pc['K_pct']}__s{s}__seed{k:02d}"
            _, ch, _, jm = _scenario_inputs(sid)
            pop = P.add_centred(
                P.scenario_population(jm, G["fr"], G["theta"], cfg),
                l1["centre_window_hours"],
                l1["centre_min_n"],
            )
            for lab, col in (("S", "score"), ("Sc", "score_c")):
                ps = pd.Series(
                    P.mode_p_on(pop, col, l1["window_hours"], l1["min_ref"]), index=pop["row_id"]
                )
                r = paired_check(ps, pn[lab], ch["row_id"].to_numpy())
                d59.append({"scenario_id": sid, "s": s, "seed": k, "score": lab, **r})
    log.info("D59 done")
    pl = P.load_placement(cfg["d62"]["placement"])
    d62 = detectability(null, pl, cfg, G["bets"], P.mode_p_on)
    log.info("D62 done")
    centre_share = null["centre_level"].value_counts(normalize=True).sort_index()
    return {
        "d59": d59,
        "d62": d62,
        "centre_level_share": {int(k): float(v) for k, v in centre_share.items()},
        "bets": {
            k: [{"label": b.label, "n": b.n, "density": b.density.tolist()} for b in v]
            for k, v in G["bets"].items()
        },
    }


def _run(run: Run, config_path: str, mode: str, workers: int) -> int:
    cfg, log = run.config, run.log
    problems = verify(resolve(cfg["raw_dir"]), resolve(cfg["checksums_file"]))
    if problems:
        raise RuntimeError(f"raw checksum problems: {problems}")
    t0 = time.time()
    _init(config_path)  # builds/validates the frozen cache
    log.info("frozen predictions: %s (%.0fs)", G["fr"].check, time.time() - t0)
    tasks = _tasks_pilot(cfg) if mode == "pilot" else _tasks_pilot2(cfg)
    diag = _pilot2_diagnostics(cfg, log) if mode == "pilot2" else None
    log.info("%d tasks, %d workers", len(tasks), workers)
    out_path = run.dir / "results.jsonl"
    t0 = time.time()
    results = []
    ctx = mp.get_context("spawn")
    with (
        ctx.Pool(workers, initializer=_init, initargs=(config_path,)) as pool,
        open(out_path, "w") as fh,
    ):
        for r in pool.imap_unordered(_process, tasks):
            fh.write(json.dumps(r, default=_json) + "\n")
            fh.flush()
            results.append(r)
            log.info("%s done in %.1fs", r["scenario_id"], r["timing_s"]["total"])
    wall = time.time() - t0
    summ = {
        "mode": mode,
        "n_tasks": len(tasks),
        "workers": workers,
        "wall_s": wall,
        "eval_times": int(len(G["eval_ns"])),
        "eval_start": str(pd.Timestamp(int(G["eval_ns"][0]))),
        "eval_end": str(pd.Timestamp(int(G["eval_ns"][-1]))),
        "m_fixed": cfg["layer2"]["m_fixed"],
        "frozen_check": G["fr"].check,
    }
    if diag is not None:
        summ.update(diag)
    run.write_json("metrics.json", summ)
    run.finish("ok", {"n_tasks": len(tasks), "wall_s": round(wall, 1)})
    return 0


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


if __name__ == "__main__":
    sys.exit(main())
