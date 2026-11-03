"""D105/D106 diagnosis: why recall falls at s = 1.1, B = 7200 (IG-only, D95), and why the design
rule has F(14 d) 0.220 at B = 7200. Measurement only; the design is not changed.

    uv run python scripts/phase5_d105_diag.py --config configs/phase5.yaml

Re-runs development scenarios (K = 1%, random base, seeds 0-19) exactly as the dev5b grid does
(same seeds, same code path) and checks that the number of investigations and seconds match the
grid row. Per degraded node it records every investigation (time, before/after onset, log E before
and after, dlog = sum of log f, pi(h0), cleared), the probe evidence sum before onset ("debt"),
the blocks within 14 d after onset during which the node was excluded, and the first flag.

Two diagnostic counterfactuals, not design proposals:
  (a) post hoc, same run: probe evidence from before onset removed for degraded nodes
      (log E = log M + probe sum since onset), e-BH recomputed;
  (b) re-run with the 7-day exclusion after clearing set to 0 (exclude_blocks = 0).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time

import numpy as np
import pandas as pd

from hpcdiag.layer2 import run_layer2 as R
from hpcdiag.layer2 import select as SEL
from hpcdiag.layer3 import run_phase5 as RP
from hpcdiag.layer3 import setup as ST
from hpcdiag.layer3 import simulate as SIM
from hpcdiag.utils.run import Run, resolve, run_guarded, start_run

H14 = 14 * SIM.BLOCKS_PER_DAY
# (label, s, policy, B, exclude_days override or None)
CASES = [
    ("ig", 1.1, "ig", 1800, None),
    ("ig", 1.1, "ig", 7200, None),
    ("ig_excl0", 1.1, "ig", 1800, 0),
    ("ig_excl0", 1.1, "ig", 7200, 0),
    ("ig", 1.3, "ig", 300, None),
    ("ig", 1.3, "ig", 7200, None),
    ("design_rule", 1.3, "design_rule", 300, None),
    ("design_rule", 1.3, "design_rule", 7200, None),
    ("ig", 1.1, "ig", 0, None),
]


def one(sid, meta, truth, snap, label, policy, B, excl_days) -> tuple[dict, list[dict], list[dict]]:
    cfg, lc = RP.L["cfg"], R.G["cfg"]["layer2"]
    alpha, m = lc["alpha"], lc["m_fixed"]
    pol = SIM.Policy(f"{policy}_B{B}", float(B), **RP.POLICIES[policy])
    cause, onset, rows = RP.world_arrays(meta, truth, snap.shape[1])
    bel = RP._belief("main")
    wld = ST.make_world(cfg, ST.VARIANTS["main"], float(meta["s"]), cause, onset,
                        RP.key(meta["base"]), RP.key(sid))  # fmt: skip
    rng = np.random.default_rng([cfg["seed"], RP.key(sid), RP.key("main"), RP.key(pol.name)])
    days = cfg["policy"]["exclude_days"] if excl_days is None else excl_days
    excl = int(days * SIM.BLOCKS_PER_DAY)
    logE, inv = SIM.run_policy(snap, bel, wld, pol, rng, alpha, m, excl, log_growth=True)
    flags = SEL.ebh_over_time(logE, alpha, m)
    lp = logE - snap  # probe evidence sum known at each evaluation
    # (a) post hoc: drop each degraded node's probe evidence from before onset
    cf = logE.copy()
    for n, _, o in rows:
        if o < len(cf):
            cf[o:, n] -= lp[o, n]
    flags_cf = SEL.ebh_over_time(cf, alpha, m)
    T = snap.shape[0]
    per_node, per_inv = [], []
    for n, _c, o in rows:
        mine = [r for r in inv if r["node"] == n]
        excluded = np.zeros(T, bool)
        for r in mine:
            if r["pi_h0"] >= bel.tau or pol.exclusion == "all":
                excluded[r["t"] + 1 : r["t"] + excl] = True
        win = slice(o, min(o + H14, T))
        f_after = np.flatnonzero(flags[o:, n]) if o < T else []
        f_cf = np.flatnonzero(flags_cf[o:, n]) if o < T else []
        pre = [r for r in mine if r["t"] < o]
        post = [r for r in mine if r["t"] >= o]
        per_node.append({
            "scenario_id": sid, "case": label, "s": meta["s"], "B": B, "node": n, "onset": o,
            "n_inv_pre": len(pre), "n_inv_post": len(post),
            "n_inv_post_14d": sum(r["t"] < o + H14 for r in post),
            "n_cleared_pre": sum(r["pi_h0"] >= bel.tau for r in pre),
            "n_cleared_post": sum(r["pi_h0"] >= bel.tau for r in post),
            "n_diagnosed_post": sum(r["pi_max"] >= bel.tau and r["pi_h0"] < bel.tau for r in post),
            "debt_at_onset": float(lp[o, n]) if o < T else np.nan,
            "probe_sum_post_14d": float(lp[min(o + H14, T - 1), n] - lp[o, n]) if o < T else np.nan,
            "logM_at_onset": float(snap[o, n]) if o < T else np.nan,
            "logM_max_14d": float(snap[win, n].max()) if o < T else np.nan,
            "logE_max_14d": float(logE[win, n].max()) if o < T else np.nan,
            "blocks_excluded_14d": int(excluded[win].sum()),
            "excluded_at_onset": bool(o < T and excluded[o]),
            "flag_delay_blocks": int(f_after[0]) if len(f_after) else None,
            "flag14": bool(len(f_after) and f_after[0] < H14),
            "flag_ever": bool(len(f_after)),
            "cf_no_debt_flag14": bool(len(f_cf) and f_cf[0] < H14),
            "cf_no_debt_flag_ever": bool(len(f_cf)),
        })  # fmt: skip
        for r in mine:
            per_inv.append({
                "scenario_id": sid, "case": label, "B": B, "node": n, "t": r["t"],
                "after_onset": r["t"] >= o, "logE_before": float(logE[r["t"], n]),
                "logE_after": float(logE[r["t"], n] + r["dlog"]), "dlog": float(r["dlog"]),
                "n_probes": len(r["seq"]), "pi_h0": float(r["pi_h0"]),
                "cleared": r["pi_h0"] >= bel.tau, "cut": bool(r["cut"]),
            })  # fmt: skip
    sec = float(sum(r["seconds"] for r in inv))
    summ = {"scenario_id": sid, "case": label, "s": meta["s"], "B": B, "policy": policy,
            "n_investigations": len(inv), "seconds_total": sec}  # fmt: skip
    return summ, per_node, per_inv


def _process(task):
    sid, cases = task
    meta, truth, snap, _ = RP.layer2(sid)
    out = []
    for label, _s, policy, B, ex in cases:
        out.append(one(sid, meta, truth, snap, label, policy, B, ex))
    return out


def _run(run: Run, config_path: str, workers: int) -> int:
    log = run.log
    idx = pd.read_parquet(resolve("data/processed/scenarios/index.parquet"))
    k1 = idx[(idx["base"] != "real") & (idx["kind"] == "node") & (idx["K_pct"] == 1)]
    tasks: dict[str, list] = {}
    for c in CASES:
        for sid in k1[k1["s"] == c[1]]["scenario_id"]:
            tasks.setdefault(sid, []).append(c)
    RP.check_sealed(list(tasks))
    log.info("%d scenarios, %d runs", len(tasks), sum(len(v) for v in tasks.values()))
    summ, nodes, invs = [], [], []
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers, RP._init, (config_path,)) as pool:
        for res in pool.imap_unordered(_process, list(tasks.items())):
            for s, pn, pi in res:
                summ.append(s)
                nodes += pn
                invs += pi
            log.info("%s done at %.0fs", res[0][0]["scenario_id"], time.time() - t0)
    pd.DataFrame(nodes).to_parquet(run.dir / "per_node.parquet")
    pd.DataFrame(invs).to_parquet(run.dir / "per_investigation.parquet")
    pd.DataFrame(summ).to_csv(run.dir / "runs.csv", index=False)
    run.write_json("metrics.json", {"n_runs": len(summ), "wall_s": time.time() - t0})
    run.finish("ok", {"n_runs": len(summ)})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    run = start_run("phase5", a.config, tag="d105-diag")
    return run_guarded(run, lambda r: _run(r, a.config, a.workers))


if __name__ == "__main__":
    sys.exit(main())
