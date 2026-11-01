"""Planning numbers for Phase 5: how much evidence can probes add? Development seeds only.

    uv run python scripts/phase5_budget_check.py --config configs/phase5.yaml

(a) bet g_a per probe (design §3) and E[log f] per cause and s;
(b) spread of E[f(p) | pool] over pools under the healthy model (design §3 fixes one pool);
(c) Rn(1) log E and rank of degraded nodes at onset + 7 d, + 14 d.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

from hpcdiag.layer2 import accumulate as A
from hpcdiag.layer2 import run_layer2 as R
from hpcdiag.utils.run import Run, run_guarded, start_run


def pvals(x, pool, rng):
    """Design §3 p-value, continuous case (no ties)."""
    s = np.sort(pool)
    r = len(s) - np.searchsorted(s, x, side="right")
    return (r + rng.uniform(size=len(x))) / (len(s) + 1)


def delta(pc, cause, probe, s):
    if cause == "mem" and probe == "mem":
        return pc["mem_fixed_shift"]
    return pc["k"][cause][probe] * np.log(s)


def build_bet(pc, ec, probe, rng):
    """g_a averaged over independent pools, so g does not depend on the pool that is scored."""
    n_pools = ec["bet_pools"]
    n = ec["bet_draws"] // len(ec["bet_causes"]) // n_pools
    ps = []
    for _ in range(n_pools):
        pool = rng.normal(0, pc["sigma"], ec["n_pool"])
        for c in ec["bet_causes"]:
            x = rng.normal(delta(pc, c, probe, ec["bet_s0"]), pc["sigma"], n)
            ps.append(pvals(x, pool, rng))
    h = np.histogram(np.concatenate(ps), bins=ec["bins"], range=(0, 1))[0].astype(float)
    return h / h.mean()


def f_of(p, g, beta):
    b = len(g)
    return (1 - beta) + beta * g[np.minimum((p * b).astype(int), b - 1)]


def cond_mean_f(g, beta, pool, sigma):
    """Exact E[f(p) | pool] under the healthy model: r = number of pool values above x."""
    n = len(pool)
    s = np.sort(pool)[::-1]
    cdf = norm.cdf(s / sigma)
    prob = -np.diff(np.r_[1.0, cdf, 0.0])  # P(r = 0), ..., P(r = n)
    grid = (np.arange(n + 1)[:, None] + (np.arange(2000)[None, :] + 0.5) / 2000) / (n + 1)
    return float(np.sum(prob * f_of(grid, g, beta).mean(axis=1)))


def layer2_part(cfg):
    bc = cfg["budget_check"]
    rows = []
    for s in bc["layer2_s"]:
        for k in bc["layer2_seeds"]:
            sid = f"node__rand{k:02d}__K{bc['layer2_K_pct']}__s{s}__seed{k:02d}"
            meta, _, truth, jm = R._scenario_inputs(sid)
            pop, ps, _ = R._layer1(jm, "compare4c")
            pairs = A.make_pairs(pop, R.P.load_placement(meta["job_nodes"]))
            lc = R.G["cfg"]["layer2"]
            eval_ns = R.G["eval_ns"]
            grid = A.block_grid(eval_ns, int(pop["end_ns"].min()))
            part = pd.factorize(pop["partition"].astype(str))[0]
            ev = A.node_evidence_relative(
                pairs, ps["Sc"], part, R.G["bets"]["Sc"], lc["jumper_J"], 1.0, grid, "node"
            )
            snap = A.snapshots(ev, eval_ns, lc["m_fixed"])
            for n, o in zip(truth["node_id"].astype(int), truth["onset_time"], strict=True):
                on = pd.Timestamp(o).value
                for d in bc["offsets_days"]:
                    t = on + d * 24 * 3600 * 10**9
                    if t > eval_ns[-1]:
                        continue
                    i = int(np.searchsorted(eval_ns, t, side="right")) - 1
                    v = snap[i, n]
                    rows.append(
                        {"s": s, "days": d, "logE": float(v), "rank": int((snap[i] > v).sum()) + 1}
                    )
    return pd.DataFrame(rows)


def _run(run: Run, config_path: str) -> int:
    cfg = run.config
    pc, ec, bc = cfg["probes"], cfg["evalue"], cfg["budget_check"]
    rng = np.random.default_rng(cfg["seed"])
    sig, n_pool, beta = pc["sigma"], ec["n_pool"], ec["beta"]
    pools = {a: rng.normal(0, sig, n_pool) for a in pc["names"]}
    bets = {a: build_bet(pc, ec, a, rng) for a in pc["names"]}
    rows = []
    for a in pc["names"]:
        g = bets[a]
        h = pvals(rng.normal(0, sig, bc["mc_draws"]), pools[a], rng)
        row = {
            "probe": a,
            "g_bin1": g[0],
            "f_max": (1 - beta) + beta * g.max(),
            "healthy": np.log(f_of(h, g, beta)).mean(),
        }
        for c in ec["bet_causes"]:
            for s in bc["s"]:
                x = rng.normal(delta(pc, c, a, s), sig, bc["mc_draws"])
                row[f"{c}_{s}"] = np.log(f_of(pvals(x, pools[a], rng), g, beta)).mean()
        rows.append(row)
    t = pd.DataFrame(rows)
    pool_rows = []
    for a in pc["names"]:
        v = np.array(
            [
                cond_mean_f(bets[a], beta, rng.normal(0, sig, n_pool), sig)
                for _ in range(bc["pool_draws"])
            ]
        )
        pool_rows.append(
            {
                "probe": a,
                "mean": v.mean(),
                "sd": v.std(),
                "q05": np.quantile(v, 0.05),
                "q95": np.quantile(v, 0.95),
                "max": v.max(),
                "actual_design_pool": cond_mean_f(bets[a], beta, pools[a], sig),
            }
        )
    tp = pd.DataFrame(pool_rows)
    R._init(cfg["layer2_config"])
    l2 = layer2_part(cfg)
    thresholds = {"one_flag": float(np.log(980 / 0.1)), "ten_flags": float(np.log(980 / 10 / 0.1))}
    summ = []
    for (s, d), g in l2.groupby(["s", "days"]):
        row = {
            "s": s,
            "days": d,
            "nodes": len(g),
            "median_logE": g["logE"].median(),
            "q25": g["logE"].quantile(0.25),
            "q75": g["logE"].quantile(0.75),
            "median_rank": g["rank"].median(),
            "rank_le_10": (g["rank"] <= 10).mean(),
            "rank_le_28": (g["rank"] <= 28).mean(),
        }
        for c in (0, 2, 4, 6):
            row[f"ge_9.19_with_{c}"] = (g["logE"] + c >= thresholds["one_flag"]).mean()
            row[f"ge_6.91_with_{c}"] = (g["logE"] + c >= thresholds["ten_flags"]).mean()
        summ.append(row)
    ts = pd.DataFrame(summ)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)
    print(t.round(3).T.to_string(header=False))
    print(tp.round(3).to_string(index=False))
    print(thresholds)
    print(ts.round(3).T.to_string(header=False))
    run.write_json(
        "metrics.json",
        {
            "probe_table": t.to_dict("records"),
            "pool": tp.to_dict("records"),
            "thresholds": thresholds,
            "layer2": ts.to_dict("records"),
            "bets": {k: v.tolist() for k, v in bets.items()},
        },
    )
    run.finish("ok", {})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    return run_guarded(
        start_run("phase5", a.config, tag="budget-check"), lambda r: _run(r, a.config)
    )


if __name__ == "__main__":
    sys.exit(main())
