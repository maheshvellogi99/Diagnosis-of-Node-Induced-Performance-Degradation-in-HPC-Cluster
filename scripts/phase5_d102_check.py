"""D102 marginal bet check before the Phase 5b grid.

    uv run python scripts/phase5_d102_check.py --config configs/phase5.yaml

(i) Mean healthy log-growth averaged over 2,000 independent pools (fresh, not those that built the
bets) for every bet the policy can use: at the prior, after mem rules out cause mem, at every single
(cause, s0) state, and the maximum over all weights on the simplex (concave, so exact). Also the
exact value, where the pool-averaged bin probability is the bin width. Mean over pools of
E[f | pool]. (ii) Healthy growth on the fixed pool at the bets actually used, from 20,000
investigations of a healthy node per probe rule. Stop rule (D108): pass = exact value <= 1e-12
AND the 2,000-pool estimate within max(twice its standard error over pools, 1e-12). The exact
computation is the criterion; Monte-Carlo is a cross-check.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from hpcdiag.layer3 import bayes as BY
from hpcdiag.layer3 import d87 as D
from hpcdiag.layer3 import evalue as EV
from hpcdiag.layer3 import probes as PR
from hpcdiag.layer3 import setup as ST
from hpcdiag.layer3 import simulate as SIM
from hpcdiag.utils.run import Run, run_guarded, start_run

N_INV = 20_000


def marginal(cfg: dict, n_pool: int, rng) -> list[dict]:
    ec, po = cfg["evalue"], cfg["policy"]
    m = PR.from_config(cfg["probes"])
    bets = ST.make_belief(cfg, ST.Variant(n_pool=n_pool)).bets
    labels = [f"{c}@{s}" for c in PR.CAUSES for s in po["s_grid"]]
    K = len(labels)
    w_prior = np.full(K, 1 / K)
    keep = np.array([0.0 if x.startswith("mem@") else 1.0 for x in labels])
    w_nomem = keep / keep.sum()
    ov = EV.bin_overlap(n_pool, bets.edges)
    width = np.diff(bets.edges)
    rows = []
    for a, name in enumerate(m.names):
        pools = rng.normal(0, m.sigma, (ec["bet_pools"], n_pool))
        Pp = EV.rank_probs(0.0, m.sigma, pools) @ ov  # (pools, bins)
        Pbar = Pp.mean(axis=0)
        G = bets.G[a]
        for lab, w in (("prior", w_prior), ("no_mem", w_nomem)):
            lf = bets.log_f_bins(a, w)
            rows.append(
                {
                    "n_pool": n_pool, "probe": name, "bet": lab,
                    "mean_log_f": float(Pbar @ lf), "exact_log_f": float(width @ lf),
                    "mean_Ef_given_pool": float((Pp @ np.exp(lf)).mean()),
                }
            )  # fmt: skip
        vert = np.array([Pbar @ bets.log_f_bins(a, np.eye(K)[k]) for k in range(K)])
        k = int(np.argmax(vert))
        lf = bets.log_f_bins(a, np.eye(K)[k])
        rows.append(
            {
                "n_pool": n_pool, "probe": name, "bet": f"best single state ({labels[k]})",
                "mean_log_f": float(vert[k]), "exact_log_f": float(width @ lf),
                "mean_Ef_given_pool": float((Pp @ np.exp(lf)).mean()),
            }
        )  # fmt: skip
        vmax, wmax = D.max_growth_simplex(Pbar, G, bets.beta)
        vex, _ = D.max_growth_simplex(width, G, bets.beta)
        lf = bets.log_f_bins(a, wmax)
        rows.append(
            {
                "n_pool": n_pool, "probe": name, "bet": "max over all weights",
                "mean_log_f": float(vmax), "exact_log_f": float(vex),
                "mean_Ef_given_pool": float((Pp @ np.exp(lf)).mean()),
            }
        )  # fmt: skip
        per_pool = Pp @ lf
        rows[-1]["se_over_pools"] = float(per_pool.std() / np.sqrt(len(per_pool)))
    for r in rows:
        r.setdefault("se_over_pools", 0.0)
    return rows


def used(cfg: dict, pol: SIM.Policy, flagged: bool, seed: int) -> dict:
    bel = ST.make_belief(cfg, ST.VARIANTS["main"])
    wld = ST.make_world(cfg, ST.VARIANTS["main"], 1.3, np.zeros(1, int), np.full(1, 10**9), 0, 0)
    rng = np.random.default_rng(seed)
    na = bel.means.shape[1]
    v = []
    for _ in range(N_INV):
        pi, unused = bel.pi0, list(range(len(bel.cost)))
        while unused:
            a = SIM.next_probe(pi, unused, bel, pol, rng, wld, flagged)
            unused.remove(a)
            if a < na:
                v.append(SIM.fixed_pool_healthy_growth(a, bel.bets.weights(pi, bel.hyp), bel, wld))
                pi = BY.update_gauss(
                    pi, bel.means[:, a], bel.sigma, float(rng.normal(0, wld.sigma))
                )
            else:
                o = int(rng.choice(len(wld.conf), p=wld.conf[0]))
                pi = BY.update_tel(pi, bel.hyp, bel.conf, o)
            if BY.marginal(pi, bel.hyp).max() >= bel.tau:
                break
    v = np.array(v)
    return {
        "probes": len(v),
        "max": float(v.max()),
        "q99": float(np.quantile(v, 0.99)),
        "mean": float(v.mean()),
        "share_gt_0": float((v > 0).mean()),
    }


def _run(run: Run) -> int:
    cfg = run.config
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 200)
    rows = []
    for n in (cfg["evalue"]["n_pool"], 200):
        rows += marginal(cfg, n, np.random.default_rng([cfg["seed"], n, 102]))
    t = pd.DataFrame(rows)
    print(t.round(5).to_string(index=False))
    tol = np.maximum(2 * t["se_over_pools"], 1e-12)  # 1e-12: floating-point rounding of log 1
    bad = t[(t["exact_log_f"] > 1e-12) | (t["mean_log_f"] > tol)]
    bad = bad.to_dict("records")
    u = {}
    for lab, pol, fl in (
        ("ig", SIM.Policy("ig", 300), False),
        ("stage_unflagged", SIM.Policy("s", 300, probes="stage"), False),
        ("stage_flagged", SIM.Policy("s", 300, probes="stage"), True),
    ):
        u[lab] = used(cfg, pol, fl, 11)
        print(lab, u[lab])
    print("\nmarginal healthy growth > 0:", bad if bad else "none")
    run.write_json("metrics.json", {"marginal": rows, "used_fixed_pool": u, "positive": bad})
    run.finish("ok", {"marginal_positive": len(bad)})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    return run_guarded(start_run("phase5", a.config, tag="d102-check"), _run)


if __name__ == "__main__":
    sys.exit(main())
