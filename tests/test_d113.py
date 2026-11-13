"""D113 plumbing: per-probe sigma and the empirical healthy pool (written before the code).

With equal per-probe sigmas, and with an empirical pool holding the Gaussian pool's values, every
output must equal the scalar-sigma / Gaussian-pool output bit for bit.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from hpcdiag.layer3 import probes as PR
from hpcdiag.layer3 import run_phase5 as RP
from hpcdiag.layer3 import setup as ST
from hpcdiag.layer3 import simulate as SIM
from hpcdiag.utils.run import load_config


@pytest.fixture(scope="module")
def cfg():
    c = load_config("configs/phase5.yaml")
    c["evalue"]["bet_pools"] = 40  # small, for speed; same for every variant compared
    c["evalue"]["n_pool"] = 120
    return c


def with_sigma(cfg, sigma):
    c = copy.deepcopy(cfg)
    c["probes"]["sigma"] = sigma
    return c


def world(cfg, v="main", s=1.3, n=6):
    cause = np.array([1, 2, 3, 0, 0, 0])[:n]
    onset = np.array([5, 5, 5, 10**9, 10**9, 10**9])[:n]
    return ST.make_world(cfg, ST.VARIANTS[v], s, cause, onset, 11, 22)


def run(cfg, B=7200, T=50):
    bel = ST.make_belief(cfg, ST.VARIANTS["main"])
    wld = world(cfg)
    logM = np.random.default_rng(0).normal(0, 2, (T, 6))
    pol = RP.make_policy("ig_d109", B, cfg)
    return SIM.run_policy(logM, bel, wld, pol, np.random.default_rng(3), 0.1, 6, 28)


def test_from_config_sigma_dict(cfg):
    m = PR.from_config(
        with_sigma(cfg, {"cpu": 0.03, "mbw": 0.05, "cache": 0.08, "mem": 0.02})["probes"]
    )
    assert m.sigma == (0.03, 0.05, 0.08, 0.02)  # in the order of probes.names
    assert PR.from_config(cfg["probes"]).sigma == 0.05
    with pytest.raises(ValueError):
        PR.from_config(with_sigma(cfg, {"cpu": 0.03})["probes"])  # every probe needs a sigma


def test_equal_per_probe_sigma_is_bit_identical(cfg):
    c2 = with_sigma(cfg, dict.fromkeys(cfg["probes"]["names"], 0.05))
    b1, b2 = ST.make_belief(cfg, ST.VARIANTS["main"]), ST.make_belief(c2, ST.VARIANTS["main"])
    assert np.array_equal(b1.bets.G, b2.bets.G) and np.array_equal(b1.bets.P, b2.bets.P)
    w1, w2 = world(cfg), world(c2)
    assert all(np.array_equal(a, b) for a, b in zip(w1.pools, w2.pools, strict=True))
    (e1, i1), (e2, i2) = run(cfg), run(c2)
    assert np.array_equal(e1, e2) and i1 == i2 and len(i1) > 0


def test_each_probe_uses_its_own_sigma(cfg):
    sig = {"cpu": 0.03, "mbw": 0.05, "cache": 0.08, "mem": 0.02}
    per = with_sigma(cfg, sig)
    bp, wp = ST.make_belief(per, ST.VARIANTS["main"]), world(per)
    for i, a in enumerate(cfg["probes"]["names"]):
        sc = with_sigma(cfg, sig[a])
        bs, ws = ST.make_belief(sc, ST.VARIANTS["main"]), world(sc)
        assert np.allclose(bp.bets.G[i], bs.bets.G[i], rtol=1e-12, atol=0)
        assert np.allclose(bp.bets.P[i], bs.bets.P[i], rtol=1e-12, atol=0)
        assert np.array_equal(wp.pools[i], ws.pools[i])
    assert np.allclose(bp.sigma, list(sig.values())) and np.allclose(wp.sigma, list(sig.values()))
    e, inv = run(per)
    assert np.isfinite(e).all() and len(inv) > 0


def _pool_csv(tmp_path, cfg, extra=True):
    w = world(cfg)
    rows = []
    rng = np.random.default_rng(5)
    for a, pool in zip(cfg["probes"]["names"], w.pools, strict=True):
        vals = rng.permutation(pool)  # time order differs from sorted order
        rows += [{"probe": a, "log_x": float(x), "t_start": 1000.0 + j} for j, x in enumerate(vals)]
        if extra:  # later runs must not enter the pool (first n_pool in time order)
            rows += [{"probe": a, "log_x": 99.0, "t_start": 10**6 + j} for j in range(5)]
    p = tmp_path / "pool.csv"
    pd.DataFrame(rows).sample(frac=1, random_state=1).to_csv(p, index=False)
    return p, w


def test_empirical_pool_with_gaussian_values_is_bit_identical(cfg, tmp_path):
    p, w = _pool_csv(tmp_path, cfg)
    c2 = copy.deepcopy(cfg)
    c2["probes"]["healthy_pool"] = str(p)
    w2 = world(c2)
    assert all(np.array_equal(a, b) for a, b in zip(w.pools, w2.pools, strict=True))
    (e1, i1), (e2, i2) = run(cfg), run(c2)
    assert np.array_equal(e1, e2) and i1 == i2


def test_empirical_pool_errors(cfg, tmp_path):
    p, _ = _pool_csv(tmp_path, cfg, extra=False)
    c2 = copy.deepcopy(cfg)
    c2["probes"]["healthy_pool"] = str(p)
    for v in ("pool_redraw", "misspec"):  # not defined for a measured pool (open question)
        with pytest.raises(ValueError):
            world(c2, v)
    short = copy.deepcopy(c2)
    short["evalue"]["n_pool"] = 121  # more than the measured runs
    with pytest.raises(ValueError):
        world(short)
    df = pd.read_csv(p)
    df[df["probe"] != "mem"].to_csv(tmp_path / "nomem.csv", index=False)
    c3 = copy.deepcopy(cfg)
    c3["probes"]["healthy_pool"] = str(tmp_path / "nomem.csv")
    with pytest.raises(ValueError):
        world(c3)
