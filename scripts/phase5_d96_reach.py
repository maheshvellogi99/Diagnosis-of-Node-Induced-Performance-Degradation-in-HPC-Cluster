"""D96 follow-up: healthy growth at the bets the policy actually uses on healthy nodes.

    uv run python scripts/phase5_d96_reach.py --config configs/phase5.yaml

For IG-only and the D97 stage rule (unflagged and flagged), 20,000 investigations each on a healthy
node with the main beliefs and the fixed pools; at every probe, the exact mean log f of the bet in
use under the healthy model on that fixed pool. Also the simplex maximum for the Phase 5 bets
(per cause at s0 = 1.3, D87), which the D87 check did not search.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from hpcdiag.layer3 import bayes as BY
from hpcdiag.layer3 import d87 as D
from hpcdiag.layer3 import evalue as EV
from hpcdiag.layer3 import setup as ST
from hpcdiag.layer3 import simulate as SIM
from hpcdiag.utils.run import Run, run_guarded, start_run

N_INV = 20_000


def healthy_P(bel, wld):
    return [
        EV.bin_probs(np.array([0.0]), wld.sigma, wld.pools[a], bel.bets.edges)[0]
        for a in range(bel.means.shape[1])
    ]


def reach(cfg, pol, flagged, seed):
    bel = ST.make_belief(cfg, ST.VARIANTS["main"])
    wld = ST.make_world(cfg, ST.VARIANTS["main"], 1.3, np.zeros(1, int), np.full(1, 10**9), 0, 0)
    P = healthy_P(bel, wld)
    rng = np.random.default_rng(seed)
    na = bel.means.shape[1]
    vals = {a: [] for a in range(na)}
    for _ in range(N_INV):
        pi, unused = bel.pi0, list(range(len(bel.cost)))
        while unused:
            a = SIM.next_probe(pi, unused, bel, pol, rng, wld, flagged)
            unused.remove(a)
            if a < na:
                w = bel.bets.weights(pi, bel.hyp)
                vals[a].append(float(P[a] @ bel.bets.log_f_bins(a, w)))
                y = float(rng.normal(0.0, wld.sigma))
                pi = BY.update_gauss(pi, bel.means[:, a], bel.sigma, y)
            else:
                o = int(rng.choice(len(wld.conf), p=wld.conf[0]))
                pi = BY.update_tel(pi, bel.hyp, bel.conf, o)
            if BY.marginal(pi, bel.hyp).max() >= bel.tau:
                break
    names = [*cfg["probes"]["names"]]
    out = {}
    for a, v in vals.items():
        v = np.array(v)
        out[names[a]] = {
            "probes": len(v),
            "share_gt_0": float((v > 0).mean()) if len(v) else None,
            "max": float(v.max()) if len(v) else None,
            "mean": float(v.mean()) if len(v) else None,
        }
    return out


def _run(run: Run) -> int:
    cfg = run.config
    out = {}
    for lab, pol, fl in (
        ("ig", SIM.Policy("ig", 300), False),
        ("stage_unflagged", SIM.Policy("s", 300, probes="stage"), False),
        ("stage_flagged", SIM.Policy("s", 300, probes="stage"), True),
    ):
        out[lab] = reach(cfg, pol, fl, 11)
        print(lab, out[lab])
    old = ST.make_belief(cfg, ST.VARIANTS["bets_cause13"])
    wld = ST.make_world(cfg, ST.VARIANTS["main"], 1.3, np.zeros(1, int), np.full(1, 10**9), 0, 0)
    P = healthy_P(old, wld)
    out["phase5_bets_simplex_max"] = {
        n: D.max_growth_simplex(P[a], old.bets.G[a], old.bets.beta)[0]
        for a, n in enumerate(cfg["probes"]["names"])
    }
    print("Phase 5 per-cause bets, simplex max:", out["phase5_bets_simplex_max"])
    run.write_json("metrics.json", out)
    run.finish("ok", {})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    return run_guarded(start_run("phase5", a.config, tag="d96-reach"), _run)


if __name__ == "__main__":
    sys.exit(main())
