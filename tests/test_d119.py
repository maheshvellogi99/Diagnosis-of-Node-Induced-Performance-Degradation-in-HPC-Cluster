"""D119(b): Phase 5F draws with measured tables (written before the code).

Measured-runs CSV: probe, log_x, t_start, host, split (fit | test). Pool = first n_pool FIT rows per
probe in time order; healthy outcomes = resampled TEST rows (never in the pool); faulty outcomes =
such a draw + the model mean (k log s, or the mem shift). Pool per base = fit rows resampled with
replacement; cross-machine = pool from one host's fit rows, outcomes from another host's test rows.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from hpcdiag.layer3 import evalue as EV
from hpcdiag.layer3 import run_phase5 as RP
from hpcdiag.layer3 import setup as ST
from hpcdiag.layer3 import simulate as SIM
from hpcdiag.utils.run import load_config

NAMES = ["cpu", "mbw", "cache", "mem"]


@pytest.fixture(scope="module")
def cfg():
    c = load_config("configs/phase5.yaml")
    c["evalue"]["bet_pools"] = 40
    c["evalue"]["n_pool"] = 100
    return c


def measured_csv(tmp_path, n_fit=150, n_test=60, hosts=("hA", "hB")):
    """Fit rows: values in (-1, 1) offset by host; test rows: values near 5 (+0.5 for hB), so every
    draw can be traced to its split and host."""
    rng = np.random.default_rng(0)
    rows = []
    for hi, h in enumerate(hosts):
        for a in NAMES:
            for j in range(n_fit):
                x = float(rng.uniform(-1, 1) + 10 * hi)
                rows.append({"probe": a, "log_x": x, "t_start": 1e6 + j * 10 + hi,
                             "host": h, "split": "fit"})  # fmt: skip
            for j in range(n_test):
                rows.append({"probe": a, "log_x": float(5 + 0.5 * hi + rng.uniform(0, 0.1)),
                             "t_start": 2e6 + j * 10 + hi, "host": h, "split": "test"})  # fmt: skip
    df = pd.DataFrame(rows).sample(frac=1, random_state=2)
    p = tmp_path / "measured.csv"
    df.to_csv(p, index=False)
    return p, df


def with_pool(cfg, path):
    c = copy.deepcopy(cfg)
    c["probes"]["healthy_pool"] = str(path)
    return c


def world(cfg, v="main", s=1.3, base=11):
    cause = np.array([1, 0, 0])
    onset = np.array([0, 10**9, 10**9])
    return ST.make_world(cfg, ST.VARIANTS[v], s, cause, onset, base, 22)


def test_gaussian_world_draw_is_the_old_call(cfg):
    w = world(cfg)
    assert w.noise is None
    r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
    for state, a in [(0, 0), (1, 2), (0, 3)]:
        assert w.draw(state, a, r1) == float(r2.normal(w.means[state, a], w.sigma))


def test_pool_is_first_fit_rows_and_outcomes_are_test_rows(cfg, tmp_path):
    p, df = measured_csv(tmp_path)
    w = world(with_pool(cfg, p))
    for i, a in enumerate(NAMES):
        fit = df[(df["probe"] == a) & (df["split"] == "fit")].sort_values("t_start", kind="stable")
        assert np.array_equal(w.pools[i], np.sort(fit["log_x"].to_numpy()[:100]))
        test = df[(df["probe"] == a) & (df["split"] == "test")]["log_x"].to_numpy()
        assert np.array_equal(np.sort(w.noise[i]), np.sort(test))
        assert not set(w.noise[i]) & set(w.pools[i])  # outcomes never in the pool
    rng = np.random.default_rng(0)
    for i in range(4):
        for _ in range(50):
            y0 = w.draw(0, i, rng)  # healthy: a test-day value
            assert y0 in set(w.noise[i])
            y1 = w.draw(1, i, rng)  # node 0's cause cpu: mean + a test-day value
            assert (y1 - w.means[1, i]) == pytest.approx(
                min(w.noise[i], key=lambda v: abs(v - (y1 - w.means[1, i])))
            )


def test_pool_per_base_resamples_fit_rows(cfg, tmp_path):
    p, df = measured_csv(tmp_path)
    c = with_pool(cfg, p)
    w1, w1b, w2 = (
        world(c, "pool_redraw", base=1),
        world(c, "pool_redraw", base=1),
        world(c, "pool_redraw", base=2),
    )
    for i, a in enumerate(NAMES):
        fit = set(df[(df["probe"] == a) & (df["split"] == "fit")]["log_x"])
        assert len(w1.pools[i]) == 100 and set(w1.pools[i]) <= fit
        assert np.array_equal(w1.pools[i], w1b.pools[i])  # fixed per base
        assert not np.array_equal(w1.pools[i], w2.pools[i])  # differs between bases
        assert len(np.unique(w1.pools[i])) < 100  # with replacement (150 values, 100 draws)
    assert w1.noise is not None


def test_cross_machine(cfg, tmp_path):
    p, df = measured_csv(tmp_path)
    w = world(with_pool(cfg, p), "cross_machine")
    for i, a in enumerate(NAMES):
        fa = set(df[(df["probe"] == a) & (df["split"] == "fit") & (df["host"] == "hA")]["log_x"])
        tb = set(df[(df["probe"] == a) & (df["split"] == "test") & (df["host"] == "hB")]["log_x"])
        assert set(w.pools[i]) <= fa and len(w.pools[i]) == 100
        assert set(w.noise[i]) == tb


def test_cross_machine_needs_two_hosts_and_measured_rows(cfg, tmp_path):
    p, _ = measured_csv(tmp_path, hosts=("hA",))
    with pytest.raises(ValueError):
        world(with_pool(cfg, p), "cross_machine")
    with pytest.raises(ValueError):
        world(cfg, "cross_machine")  # Gaussian path: not defined


def test_misspec_refused_with_measured_tables(cfg, tmp_path):
    p, _ = measured_csv(tmp_path)
    with pytest.raises(ValueError):
        world(with_pool(cfg, p), "misspec")


def test_healthy_growth_is_exact_for_empirical_outcomes(cfg, tmp_path):
    p, _ = measured_csv(tmp_path)
    c = with_pool(cfg, p)
    bel, w = ST.make_belief(c, ST.VARIANTS["main"]), world(c)
    wt = bel.bets.weights(bel.pi0, bel.hyp)
    for a in range(4):
        got = SIM.fixed_pool_healthy_growth(a, wt, bel, w)
        # brute force: for each test value the conformal p is uniform on [r, r+1)/(n+1)
        n = len(w.pools[a])
        ov = EV.bin_overlap(n, bel.bets.edges)
        lf = bel.bets.log_f_bins(a, wt)
        r = n - np.searchsorted(w.pools[a], w.noise[a], side="right")
        assert got == pytest.approx(float(np.mean(ov[r] @ lf)), abs=1e-12)


def test_policy_runs_with_measured_tables(cfg, tmp_path):
    p, _ = measured_csv(tmp_path)
    c = with_pool(cfg, p)
    bel, w = ST.make_belief(c, ST.VARIANTS["main"]), world(c)
    logM = np.random.default_rng(0).normal(0, 2, (30, 3))
    pol = RP.make_policy("ig_d109", 7200, c)
    logE, inv = SIM.run_policy(
        logM, bel, w, pol, np.random.default_rng(1), 0.1, 3, 28, log_growth=True
    )
    assert np.isfinite(logE).all() and len(inv) > 0


def test_csv_without_split_keeps_gaussian_outcomes(cfg, tmp_path):
    """The D113 behaviour (pool only, Gaussian outcomes) when the CSV has no split column."""
    p, df = measured_csv(tmp_path)
    q = tmp_path / "plain.csv"
    df[df["split"] == "fit"][["probe", "log_x", "t_start"]].to_csv(q, index=False)
    w = world(with_pool(cfg, q))
    assert w.noise is None
