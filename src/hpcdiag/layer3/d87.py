"""D87 checks, reported before the grid: log-growth per probe and E[f | pool], exact given a pool."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hpcdiag.layer3 import evalue as EV
from hpcdiag.layer3.probes import CAUSES, ProbeModel


def bet_weights(m: ProbeModel) -> dict[str, np.ndarray]:
    """Cause mixtures a probe can meet: start of an investigation, after the mem probe has ruled out
    cause mem, and each single cause (the extreme a posterior can reach)."""
    n = len(CAUSES)
    out = {"prior": np.full(n, 1 / n), "no_mem": np.array([1 / 3, 1 / 3, 1 / 3, 0.0])}
    for i, c in enumerate(CAUSES):
        out[f"only_{c}"] = np.eye(n)[i]
    return out


def growth_table(m: ProbeModel, bets: EV.Bets, pools: dict, s_grid) -> pd.DataFrame:
    """Mean log f per probe, bet mixture and true state, on the fixed pool of each probe."""
    rows = []
    for ia, a in enumerate(m.names):
        pool = pools[a][None, :]
        for label, w in bet_weights(m).items():
            g = bets.mixture(ia, w)
            lf = np.log((1 - bets.beta) + bets.beta * g)
            hl = float(EV.conditional_mean(lf, 0.0, m.sigma, pool, bets.edges)[0])
            rows.append({"probe": a, "bet": label, "state": "healthy", "s": None, "growth": hl})
            for c in CAUSES:
                for s in s_grid:
                    v = EV.conditional_mean(lf, m.delta(c, a, s), m.sigma, pool, bets.edges)[0]
                    rows.append(
                        {"probe": a, "bet": label, "state": c, "s": s, "growth": float(v)}
                    )
    return pd.DataFrame(rows)


def pool_table(
    m: ProbeModel, bets: EV.Bets, pools: dict, n_pool: int, n_pools: int, rng
) -> pd.DataFrame:
    """E[f | pool] under the healthy model over independent pools, and on the fixed pool."""
    rows = []
    for ia, a in enumerate(m.names):
        fresh = rng.normal(0, m.sigma, (n_pools, n_pool))
        for label, w in bet_weights(m).items():
            f = (1 - bets.beta) + bets.beta * bets.mixture(ia, w)
            v = EV.conditional_mean(f, 0.0, m.sigma, fresh, bets.edges)
            fx = EV.conditional_mean(f, 0.0, m.sigma, pools[a][None, :], bets.edges)[0]
            lv = EV.conditional_mean(np.log(f), 0.0, m.sigma, fresh, bets.edges)
            rows.append(
                {
                    "probe": a,
                    "bet": label,
                    "mean": v.mean(),
                    "sd": v.std(),
                    "q05": np.quantile(v, 0.05),
                    "q95": np.quantile(v, 0.95),
                    "max": v.max(),
                    "fixed_pool": fx,
                    "share_pools_growth_gt_0": float((lv > 0).mean()),
                }
            )
    return pd.DataFrame(rows)
