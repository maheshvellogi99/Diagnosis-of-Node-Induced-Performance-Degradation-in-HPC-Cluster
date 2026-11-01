"""D87 check, before the grid: healthy and per-cause log-growth per probe, and E[f | pool].

    uv run python scripts/phase5_d87_check.py --config configs/phase5.yaml
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from hpcdiag.layer3 import d87 as D
from hpcdiag.layer3 import probes as PR
from hpcdiag.layer3 import setup as ST
from hpcdiag.utils.run import Run, run_guarded, start_run


def one(cfg: dict, n_pool: int) -> dict:
    ec = cfg["evalue"]
    m = PR.from_config(cfg["probes"])
    bets = ST.make_belief(cfg, ST.Variant(n_pool=n_pool)).bets
    fp = ST.fixed_pools(m.names, m.sigma, n_pool, [ec["pool_seed"], n_pool])
    pools = dict(zip(m.names, fp, strict=True))
    g = D.growth_table(m, bets, pools, cfg["policy"]["s_grid"])
    p = D.pool_table(m, bets, pools, n_pool, ec["bet_pools"], np.random.default_rng(n_pool + 1))
    return {"growth": g, "pool": p, "G_bin1": bets.G[:, :, 0].tolist()}


def _run(run: Run) -> int:
    cfg = run.config
    pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 500)
    out, bad = {}, []
    for n in (cfg["evalue"]["n_pool"], 200):
        r = one(cfg, n)
        g, p = r["growth"], r["pool"]
        h = g[g["state"] == "healthy"].pivot(index="probe", columns="bet", values="growth")
        print(f"\nn_pool = {n}: healthy mean log f on the fixed pool\n{h.round(4)}")
        c = g[(g["state"] != "healthy") & (g["bet"].isin(["prior", "no_mem"]))]
        t = c.pivot_table(index=["bet", "state", "s"], columns="probe", values="growth")
        print(f"\nper-cause mean log f\n{t.round(3)}")
        print(f"\nE[f | pool] over {cfg['evalue']['bet_pools']} pools\n{p.round(4).to_string()}")
        pos = g[(g["state"] == "healthy") & (g["growth"] > 0)]
        bad += [{"n_pool": n, **x} for x in pos.to_dict("records")]
        out[str(n)] = {
            "growth": g.to_dict("records"),
            "pool": p.to_dict("records"),
            "G_bin1": r["G_bin1"],
        }
    print("\nhealthy growth > 0:", bad if bad else "none")
    run.write_json("metrics.json", {**out, "healthy_growth_positive": bad})
    run.finish("ok", {"healthy_growth_positive": len(bad)})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    return run_guarded(start_run("phase5", a.config, tag="d87-check"), _run)


if __name__ == "__main__":
    sys.exit(main())
