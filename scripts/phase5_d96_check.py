"""D96 check before the Phase 5b grid: probe bets over (cause, s0) states.

    uv run python scripts/phase5_d96_check.py --config configs/phase5.yaml

Healthy mean log f per probe on the fixed pool: at the prior, after the mem probe has ruled out
cause mem, at every single state, and the MAXIMUM over all bet weights on the simplex (concave
objective, so exact). Every posterior the policy can reach gives weights on that simplex, so the
maximum bounds them all. Also: the share of 500 fresh pools whose maximum is above 0, growth per
cause and s, and E[f | pool] over 2,000 pools. STOP rule (D96): any healthy growth > 0.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from hpcdiag.layer3 import d87 as D
from hpcdiag.layer3 import evalue as EV
from hpcdiag.layer3 import probes as PR
from hpcdiag.layer3 import setup as ST
from hpcdiag.utils.run import Run, run_guarded, start_run


def one(cfg: dict, n_pool: int, rng) -> dict:
    ec, po = cfg["evalue"], cfg["policy"]
    m = PR.from_config(cfg["probes"])
    bets = ST.make_belief(cfg, ST.Variant(n_pool=n_pool)).bets
    pools = ST.fixed_pools(m.names, m.sigma, n_pool, [ec["pool_seed"], n_pool])
    means, hyp = PR.state_table(m, po["s_grid"])
    labels = [f"{c}@{s}" for c in PR.CAUSES for s in po["s_grid"]]  # state_table order, no h0
    assert len(labels) == int((hyp > 0).sum()) and means.shape[0] == len(labels) + 1
    K = bets.G.shape[1]
    w_prior = np.full(K, 1 / K)
    no_mem = np.array([0.0 if lab.startswith("mem@") else 1.0 for lab in labels])
    w_nomem = no_mem / no_mem.sum()
    edges = bets.edges
    healthy, cause_rows, pool_rows = [], [], []
    for a, name in enumerate(m.names):
        P = EV.bin_probs(np.array([0.0]), m.sigma, pools[a], edges)[0]
        G = bets.G[a]
        vert = [float(P @ bets.log_f_bins(a, np.eye(K)[k])) for k in range(K)]
        vmax, wmax = D.max_growth_simplex(P, G, bets.beta)
        top = np.argsort(-wmax)[:3]
        healthy.append(
            {
                "probe": name,
                "prior": float(P @ bets.log_f_bins(a, w_prior)),
                "no_mem": float(P @ bets.log_f_bins(a, w_nomem)),
                "best_vertex": max(vert),
                "best_vertex_state": labels[int(np.argmax(vert))],
                "max_over_simplex": vmax,
                "argmax_weights": {labels[i]: round(float(wmax[i]), 3) for i in top},
            }
        )
        # fresh pools: is the maximum above 0 for some pools? (marginal context, D100)
        fresh = rng.normal(0, m.sigma, (500, n_pool))
        ov = EV.bin_overlap(n_pool, edges)
        Pf = EV.rank_probs(0.0, m.sigma, fresh) @ ov
        mx = np.array([D.max_growth_simplex(p, G, bets.beta)[0] for p in Pf])
        healthy[-1]["fresh_pools_share_max_gt_0"] = float((mx > 0).mean())
        healthy[-1]["fresh_pools_max"] = float(mx.max())
        # E[f | pool], 2,000 pools: healthy bin probabilities do not depend on the bet
        many = rng.normal(0, m.sigma, (2000, n_pool))
        Pm = EV.rank_probs(0.0, m.sigma, many) @ ov
        for lab, w in (("prior", w_prior), ("no_mem", w_nomem)):
            fb = np.exp(bets.log_f_bins(a, w))
            v = Pm @ fb
            pool_rows.append(
                {
                    "probe": name,
                    "bet": lab,
                    "mean": v.mean(),
                    "sd": v.std(),
                    "q95": np.quantile(v, 0.95),
                    "fixed_pool": float(P @ fb),
                }
            )
        # E[f | pool] is linear in w: its maximum over the simplex is at a vertex
        fv = np.exp(np.stack([bets.log_f_bins(a, np.eye(K)[k]) for k in range(K)]))  # (K, bins)
        Ef_fixed = fv @ P
        Ef_many = Pm @ fv.T  # (pools, K)
        pool_rows.append(
            {
                "probe": name,
                "bet": "worst single state",
                "mean": float(Ef_many.max(axis=1).mean()),
                "sd": float(Ef_many.max(axis=1).std()),
                "q95": float(np.quantile(Ef_many.max(axis=1), 0.95)),
                "fixed_pool": float(Ef_fixed.max()),
            }
        )
        for c in PR.CAUSES:
            for s in po["s_grid"]:
                Pc = EV.bin_probs(np.array([m.delta(c, name, s)]), m.sigma, pools[a], edges)[0]
                j = labels.index(f"{c}@{s}")
                j13 = labels.index(f"{c}@1.3")
                cause_rows.append(
                    {
                        "probe": name,
                        "cause": c,
                        "s": s,
                        "prior": float(Pc @ bets.log_f_bins(a, w_prior)),
                        "no_mem": float(Pc @ bets.log_f_bins(a, w_nomem)),
                        "true_state": float(Pc @ bets.log_f_bins(a, np.eye(K)[j])),
                        "state_s0_1.3": float(Pc @ bets.log_f_bins(a, np.eye(K)[j13])),
                    }
                )
    return {
        "healthy": healthy,
        "cause": cause_rows,
        "pool": pool_rows,
    }


def _run(run: Run) -> int:
    cfg = run.config
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 500)
    out, bad = {}, []
    for n in (cfg["evalue"]["n_pool"], 200):
        r = one(cfg, n, np.random.default_rng([cfg["seed"], n, 96]))
        h = pd.DataFrame(r["healthy"])
        print(f"\nn_pool = {n}: healthy mean log f on the fixed pool")
        print(h.round(4).to_string(index=False))
        c = pd.DataFrame(r["cause"])
        print(c[c["cause"] != "mem"].round(3).to_string(index=False))
        print(pd.DataFrame(r["pool"]).round(4).to_string(index=False))
        for x in r["healthy"]:
            for k in ("prior", "no_mem", "best_vertex", "max_over_simplex"):
                if x[k] > 0:
                    bad.append({"n_pool": n, "probe": x["probe"], "bet": k, "growth": x[k]})
        out[str(n)] = r
    print("\nhealthy growth > 0:", bad if bad else "none")
    run.write_json("metrics.json", {**out, "healthy_growth_positive": bad})
    run.finish("ok", {"healthy_growth_positive": len(bad)})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    return run_guarded(start_run("phase5", a.config, tag="d96-check"), _run)


if __name__ == "__main__":
    sys.exit(main())
