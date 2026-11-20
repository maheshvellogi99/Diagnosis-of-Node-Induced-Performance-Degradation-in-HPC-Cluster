"""D122: Layers 1-3 end to end on ONE development scenario -> deliverables/demo/.

    uv run python scripts/demo_pipeline.py --config configs/phase5.yaml \
        --scenario node__rand00__K1__s1.3__seed00 --budget 7200 --ref 20261113-223241-dev5c \
        --out deliverables/demo

Layer 1: frozen LightGBM q90 scores, centred and turned into Mode P p-values (the input Layer 2
consumes). Layer 2: Rn(1) relative e-processes per node, e-BH every 6 h (alpha 0.1, m 980).
Layer 3: the frozen primary ig_d109 (D95, D96, D98, D109) at B node-seconds per day. Same code path
and seeds as the dev5c grid; the script checks that its metrics equal the dev5c row. Development
seeds only (raises on a sealed id). Writes job_pvalues.csv, node_flags.csv, investigations.csv,
summary.txt and a run dir runs/phaseR/<id>/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hpcdiag.layer2 import run_layer2 as R
from hpcdiag.layer2 import select as SEL
from hpcdiag.layer3 import run_phase5 as RP
from hpcdiag.layer3 import setup as ST
from hpcdiag.layer3 import simulate as SIM
from hpcdiag.layer3.probes import HYPOTHESES
from hpcdiag.utils.run import Run, resolve, run_guarded, start_run

TIMING = {"layer2_s", "sim_s", "total_s"}


def _run(run: Run, config_path: str, sid: str, B: int, ref: str, out: Path) -> int:
    t_start = time.time()
    RP.check_sealed([sid])
    RP._init(config_path)
    cfg, lc = RP.L["cfg"], R.G["cfg"]["layer2"]
    eval_ns = R.G["eval_ns"]
    # Layer 1 -> p-values that Layer 2 consumes
    meta, changes, truth, jm = R._scenario_inputs(sid)
    pop, p = RP._pop_p(jm)
    # Layer 2
    meta, truth, snap, extra = RP.layer2(sid)
    # Layer 3: same steps and seeds as run_phase5.run_one
    res = RP.run_one(sid, meta, truth, snap, extra, "main", "ig_d109", B)
    pol = RP.make_policy("ig_d109", B, cfg)
    cause, onset, truth_rows = RP.world_arrays(meta, truth, snap.shape[1])
    wld = ST.make_world(cfg, ST.VARIANTS["main"], float(meta["s"]), cause, onset,
                        RP.key(meta["base"]), RP.key(sid))  # fmt: skip
    rng = np.random.default_rng([cfg["seed"], RP.key(sid), RP.key("main"), RP.key(pol.name)])
    excl = int(cfg["policy"]["exclude_days"] * SIM.BLOCKS_PER_DAY)
    logE, inv = SIM.run_policy(snap, RP._belief("main"), wld, pol, rng, lc["alpha"], lc["m_fixed"],
                               excl, log_growth=True)  # fmt: skip
    flags = SEL.ebh_over_time(logE, lc["alpha"], lc["m_fixed"])
    assert len(inv) == res["n_investigations"], "replicated Layer 3 differs from run_one"
    with open(resolve(f"runs/phase5/{ref}/results.jsonl")) as fh:
        dev = [json.loads(x) for x in fh]
    dev = [r for r in dev if r["scenario_id"] == sid and r["policy"] == "ig_d109" and r["B"] == B
           and r["variant"] == "main" and r.get("k_pre", 0) == 0]  # fmt: skip

    def canon(r):
        return json.dumps(
            {k: v for k, v in r.items() if k not in TIMING}, sort_keys=True, default=str
        )

    same = bool(dev) and canon(json.loads(json.dumps(res, default=str))) == canon(dev[0])

    out.mkdir(parents=True, exist_ok=True)
    injected = set(changes["row_id"].tolist())
    cols = [c for c in ("row_id", "job_id", "end_time", "user", "partition", "nodes_req", "run_time_s",
                        "score", "score_c") if c in pop.columns]  # fmt: skip
    jp = pop[cols].copy()
    jp["p_value"] = p
    jp["injected_slow"] = jp["row_id"].isin(injected)
    jp.to_csv(out / "job_pvalues.csv", index=False)
    ts = lambda i: str(pd.Timestamp(int(eval_ns[i])))  # noqa: E731
    deg = {n: (HYPOTHESES[c], o) for n, c, o in truth_rows}
    ever = np.flatnonzero(flags.any(axis=0))
    rows = []
    for n in sorted(set(ever) | set(deg)):
        f = np.flatnonzero(flags[:, n])
        c, o = deg.get(n, ("h0", None))
        after = f[f >= o] if o is not None else np.array([], int)
        rows.append({"node_id": int(n), "injected_cause": c, "onset": ts(o) if o is not None and o < len(eval_ns) else None,
                     "first_flag": ts(f[0]) if len(f) else None,
                     "first_flag_after_onset": ts(after[0]) if len(after) else None,
                     "delay_days": float((eval_ns[after[0]] - eval_ns[o]) / 86400e9) if len(after) else None,
                     "evaluations_flagged": int(len(f)), "final_logE": float(logE[-1, n])})  # fmt: skip
    nf = pd.DataFrame(rows)
    nf.to_csv(out / "node_flags.csv", index=False)
    names = [*cfg["probes"]["names"], "tel"]

    def true_state(r):  # the node's true state at the investigation: its cause after onset, else h0
        c, o = deg.get(r["node"], ("h0", None))
        return c if o is not None and r["t"] >= o else "h0"

    ir = [{"time": ts(r["t"]), "node_id": r["node"], "true_state": true_state(r),
           "probes": " ".join(names[a] for a in r["seq"]), "seconds": r["seconds"], "log_probe_evidence": r["dlog"],
           "pi_h0": r["pi_h0"], "diagnosis": HYPOTHESES[r["cause"]], "pi_max": r["pi_max"], "cut_by_budget": r["cut"]}
          for r in inv]  # fmt: skip
    pd.DataFrame(ir).to_csv(out / "investigations.csv", index=False)

    m = res["metrics"]
    dg = res["diagnosis"]
    flagged_deg = [d for d in dg if d.get("flagged")]
    wall = time.time() - t_start
    lines = [
        f"Demo, development scenario {sid} (development seeds; sealed evaluation pending)",
        f"Layer 1: {len(jp):,} jobs scored; {int(jp['injected_slow'].sum()):,} injected slow job-rows; "
        f"share p <= 0.10: {float((jp['p_value'] <= 0.10).mean()):.4f}",
        f"Layer 2+3: primary ig_d109 at B = {B} node-s/day; {len(eval_ns)} evaluations every 6 h",
        f"Injected degraded nodes: {m['n_degraded']} (cause per node in node_flags.csv), s = {meta['s']}",
        f"Flagged after onset (ever): {sum(d['detected'] for d in m['detections'])} of {m['n_degraded']}; "
        f"within 14 days: {sum(bool(d['within_horizon']) for d in m['detections'])}",
        f"Median delay: {m['median_delay_h'] / 24:.2f} days",
        f"Healthy nodes ever flagged (false): {m['false_nodes_ever']}",
        f"Investigations: {len(inv):,}; probe seconds: {res['seconds_total']:,.0f}",
        f"Diagnosis among flagged degraded nodes: {sum(d['correct'] for d in flagged_deg)} of {len(flagged_deg)} correct",
        f"Identical to dev5c run {ref} row: {same}",
        f"Wall time: {wall:.1f} s",
    ]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    run.write_json(
        "metrics.json", {"scenario": sid, "B": B, "identical_to_dev5c": same, "wall_s": wall}
    )
    run.finish("ok" if same else "differs", {"identical": same, "wall_s": round(wall, 1)})
    return 0 if same else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    return run_guarded(start_run("phaseR", a.config, tag="demo"),
                       lambda r: _run(r, a.config, a.scenario, a.budget, a.ref, resolve(a.out)))  # fmt: skip


if __name__ == "__main__":
    sys.exit(main())
