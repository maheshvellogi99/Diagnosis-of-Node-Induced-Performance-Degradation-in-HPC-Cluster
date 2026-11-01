"""Layer 3 math (docs/PHASE5_DESIGN.md §3-§5, D86-D90). Tests first (CLAUDE.md rule 7)."""

from __future__ import annotations

import numpy as np
import pytest

from hpcdiag.layer3 import bayes as BY
from hpcdiag.layer3 import evalue as EV
from hpcdiag.layer3 import probes as PR
from hpcdiag.layer3 import simulate as SIM
from hpcdiag.stats.evalues import jumper_init, jumper_step, log_total

EDGES = [0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
S_GRID = [1.1, 1.2, 1.3, 1.5, 2.0, 3.0]
PC = {
    "names": ["cpu", "mbw", "cache", "mem"],
    "cost_s": {"cpu": 20, "mbw": 20, "cache": 20, "mem": 1},
    "sigma": 0.05,
    "k": {
        "cpu": {"cpu": 1.0, "mbw": 0.1, "cache": 0.3, "mem": 0.0},
        "cache": {"cpu": 0.3, "mbw": 0.0, "cache": 1.0, "mem": 0.0},
        "mbw": {"cpu": 0.1, "mbw": 1.0, "cache": 0.3, "mem": 0.0},
        "mem": {"cpu": 0.0, "mbw": 0.1, "cache": 0.0, "mem": 0.0},
    },
    "mem_fixed_shift": 0.5,
}
EC = {"n_pool": 200, "beta": 0.9, "bet_s0": 1.3, "edges": EDGES, "bet_pools": 200}


@pytest.fixture(scope="module")
def model():
    return PR.from_config(PC)


@pytest.fixture(scope="module")
def bets(model):
    return EV.build_bets(model, EC, np.random.default_rng(0))


def make_belief(model, bets, pi0_h0=0.5, conf=None):
    means, hyp = PR.state_table(model, S_GRID)
    z, w = np.polynomial.hermite_e.hermegauss(11)
    return SIM.Belief(
        means=means,
        hyp=hyp,
        sigma=model.sigma,
        conf=PR.design_confusion(0.8, 0.08, 0.96) if conf is None else conf,
        bets=bets,
        pi0=BY.prior(hyp, pi0_h0),
        tau=0.95,
        cost=np.array([*model.cost, 60.0]),
        z=z,
        w=w / w.sum(),
    )


def make_world(model, n_nodes=6, cause=None, onset=None, s=1.3, n_pool=200, seed=1, sigma=None):
    rng = np.random.default_rng(seed)
    sg = model.sigma if sigma is None else sigma
    means = np.array([[model.delta(h, a, s) for a in model.names] for h in PR.HYPOTHESES])
    return SIM.World(
        pools=[np.sort(rng.normal(0, sg, n_pool)) for _ in model.names],
        means=means,
        sigma=sg,
        conf=PR.design_confusion(0.8, 0.08, 0.96),
        cause=np.zeros(n_nodes, int) if cause is None else np.asarray(cause),
        onset_idx=np.full(n_nodes, 10**9) if onset is None else np.asarray(onset),
    )


# ------------------------------------------------------------------ model


def test_delta_by_hand(model):
    assert model.delta("cpu", "cpu", 2.0) == pytest.approx(np.log(2))
    assert model.delta("mbw", "cache", 3.0) == pytest.approx(0.3 * np.log(3))
    assert model.delta("mem", "mem", 3.0) == 0.5
    assert model.delta("h0", "cpu", 3.0) == 0.0
    assert model.delta("cache", "mbw", 3.0) == 0.0


def test_state_table_shape(model):
    means, hyp = PR.state_table(model, S_GRID)
    assert means.shape == (1 + 4 * 6, 4)
    assert (means[0] == 0).all() and hyp[0] == 0
    assert (np.bincount(hyp) == [1, 6, 6, 6, 6]).all()


def test_primary_only_changes_diagonal(model):
    m2 = PR.with_primary(model, 2.0)
    assert m2.k["cpu"]["cpu"] == 2.0 and m2.k["cpu"]["cache"] == 0.3
    assert m2.k["mem"]["mbw"] == 0.1 and model.k["cpu"]["cpu"] == 1.0


def test_perturbed_within_spread(model):
    m2 = PR.perturbed(model, np.random.default_rng(3))
    for c, row in model.k.items():
        for a, v in row.items():
            assert 0.5 * v - 1e-12 <= m2.k[c][a] <= 1.5 * v + 1e-12


def test_design_confusion_rows():
    c = PR.design_confusion(0.8, 0.08, 0.96)
    assert c.shape == (5, 5)
    assert np.allclose(c.sum(axis=1), 1) and (c > 0).all()
    assert c[1, 1] == 0.8 and c[1, 2] == 0.08 and c[0, 0] == 0.96


# ------------------------------------------------------------------ p-values


def test_p_by_hand_ties():
    pool = np.array([1.0, 2.0, 2.0, 3.0])
    assert EV.conformal_p([2.0], pool, np.array([0.5]))[0] == pytest.approx((1 + 0.5 * 3) / 5)
    assert EV.conformal_p([9.0], pool, np.array([0.5]))[0] == pytest.approx(0.1)
    assert EV.conformal_p([0.0], pool, np.array([0.5]))[0] == pytest.approx((4 + 0.5) / 5)


def test_p_exactly_uniform_over_ranks_with_ties():
    # x exchangeable with a pool on {0, 1}: enumerate every assignment of 5 draws
    from itertools import product

    n, hits, total = 4, 0.0, 0
    for v in product([0.0, 1.0], repeat=n + 1):
        pool, x = np.sort(v[:-1]), v[-1]
        for th in np.linspace(0.005, 0.995, 100):
            hits += EV.conformal_p([x], pool, np.array([th]))[0] <= 0.3
            total += 1
    assert hits / total == pytest.approx(0.3, abs=0.01)


def test_rank_probs_match_draws():
    rng = np.random.default_rng(0)
    pool = rng.normal(0, 1, (1, 20))
    pr = EV.rank_probs(0.7, 1.0, pool)[0]
    assert pr.sum() == pytest.approx(1)
    x = rng.normal(0.7, 1, 400_000)
    r = (x[:, None] < pool).sum(axis=1)
    emp = np.bincount(r, minlength=21) / len(x)
    assert np.abs(emp - pr).max() < 0.004


def test_bin_overlap_rows_sum_to_one():
    edges = np.array(EDGES)
    for n in (200, 1000):
        ov = EV.bin_overlap(n, edges)
        assert np.allclose(ov.sum(axis=1), 1)
        # all ranks equally likely: the mass of each bin is its width
        assert np.allclose(ov.mean(axis=0), np.diff(edges) * (n + 1) / (n + 1))


def test_density_integrates_to_one_and_uniform_under_h0():
    rng = np.random.default_rng(0)
    edges = np.array(EDGES)
    pools = rng.normal(0, 0.05, (2000, 200))
    for mu in (0.0, 0.03, 0.26):
        g = EV.density(mu, 0.05, pools, edges)
        assert g @ np.diff(edges) == pytest.approx(1)
        assert (g >= 0).all()
    assert np.abs(EV.density(0.0, 0.05, pools, edges) - 1).max() < 0.1


def test_bets_integrate_and_floor(bets):
    w = np.diff(bets.edges)
    assert np.allclose(bets.G @ w, 1)
    for a in range(4):
        g = bets.mixture(a, np.array([0.1, 0.2, 0.3, 0.4]))
        assert g @ w == pytest.approx(1)
        f = (1 - bets.beta) + bets.beta * g
        assert f @ w == pytest.approx(1)
        assert f.min() >= 1 - bets.beta


def test_zero_mean_cause_gives_flat_bet(bets, model):
    a = model.names.index("mem")
    for j in range(3):  # cpu, cache, mbw do not move the mem probe
        assert (bets.G[a, j] == 1).all()


def test_bet_does_not_use_scored_pool(model):
    # bets depend on the rng stream of independent pools only: scored pool plays no part
    b1 = EV.build_bets(model, EC, np.random.default_rng(5))
    b2 = EV.build_bets(model, EC, np.random.default_rng(5))
    assert (b1.G == b2.G).all()


def test_f_lookup_bins(bets):
    g = np.arange(1.0, 11.0)
    f = bets.f(np.array([0.0, 0.0005, 0.001, 0.0999, 0.1, 0.999, 1.0]), g)
    want = 0.1 + 0.9 * np.array([1, 1, 2, 7, 8, 10, 10.0])
    assert np.allclose(f, want)


def test_marginal_mean_of_f_is_one_over_pools(bets, model):
    # E[f] = 1 over pools (x and pool drawn fresh), for the mixture actually used first
    rng = np.random.default_rng(2)
    a = model.names.index("cpu")
    g = bets.mixture(a, np.full(4, 0.25))
    x = rng.normal(0, 0.05, (200_000, 201))
    p = (x[:, 1:] > x[:, :1]).sum(axis=1) + rng.uniform(size=len(x))
    f = bets.f(p / 201, g)
    assert f.mean() == pytest.approx(1, abs=0.15)
    # exact version over pools: the mean over ranks of the bin mass is the bin width
    ov = EV.bin_overlap(200, bets.edges)
    assert ov.mean(axis=0) @ ((1 - bets.beta) + bets.beta * g) == pytest.approx(1)


def test_conditional_mean_matches_draws(bets):
    rng = np.random.default_rng(4)
    pool = np.sort(rng.normal(0, 0.05, 200))
    g = bets.mixture(0, np.full(4, 0.25))
    f_bins = (1 - bets.beta) + bets.beta * g
    exact = EV.conditional_mean(f_bins, 0.0, 0.05, pool[None, :], bets.edges)[0]
    x = rng.normal(0, 0.05, 2_000_000)
    sim = bets.f(EV.conformal_p(x, pool, rng.uniform(size=len(x))), g).mean()
    assert exact == pytest.approx(sim, abs=0.05)


# ------------------------------------------------------------------ posterior and gain


def test_prior(model):
    _, hyp = PR.state_table(model, S_GRID)
    pi = BY.prior(hyp, 0.5)
    assert pi.sum() == pytest.approx(1)
    assert np.allclose(BY.marginal(pi, hyp), [0.5, 0.125, 0.125, 0.125, 0.125])


def test_gauss_update_by_hand():
    pi = np.array([0.5, 0.5])
    post = BY.update_gauss(pi, np.array([0.0, 1.0]), 1.0, 0.0)
    assert post[1] / post[0] == pytest.approx(np.exp(-0.5))
    assert post.sum() == pytest.approx(1)


def test_tel_update_by_hand():
    pi = np.array([0.5, 0.5])
    hyp = np.array([0, 1])
    conf = np.zeros((5, 5))
    conf[0, 0], conf[1, 0] = 0.9, 0.1
    assert BY.update_tel(pi, hyp, conf, 0)[0] == pytest.approx(0.9)


def test_posterior_finds_cause_from_strong_probe(model, bets):
    bel = make_belief(model, bets)
    a = model.names.index("cpu")
    y = model.delta("cpu", "cpu", 2.0)
    h = BY.marginal(BY.update_gauss(bel.pi0, bel.means[:, a], bel.sigma, y), bel.hyp)
    assert h.argmax() == 1 and h[1] > 0.9


def _brute_force_entropy(pi, means, sigma, hyp, rng, n=40_000):
    st = rng.choice(len(pi), n, p=pi)
    x = rng.normal(means[st], sigma)
    ll = -0.5 * ((x[:, None] - means[None, :]) / sigma) ** 2
    post = pi * np.exp(ll - ll.max(axis=1, keepdims=True))
    post /= post.sum(axis=1, keepdims=True)
    return float(BY.entropy(BY.marginal(post, hyp)).mean())


@pytest.mark.parametrize("a", [0, 1, 2, 3])
def test_expected_entropy_matches_brute_force(model, bets, a):
    bel = make_belief(model, bets)
    rng = np.random.default_rng(a)
    z, w = np.polynomial.hermite_e.hermegauss(31)
    e = BY.expected_entropy_gauss(
        bel.pi0, bel.means[:, a], bel.sigma, bel.hyp, z, w / w.sum()
    )
    assert e == pytest.approx(
        _brute_force_entropy(bel.pi0, bel.means[:, a], bel.sigma, bel.hyp, rng), abs=0.02
    )


def test_identical_rows_have_no_gain(model, bets):
    bel = make_belief(model, bets)
    h = float(BY.entropy(BY.marginal(bel.pi0, bel.hyp)))
    e = BY.expected_entropy_gauss(
        bel.pi0, np.zeros(len(bel.pi0)), bel.sigma, bel.hyp, bel.z, bel.w
    )
    assert e == pytest.approx(h)
    flat = np.full((5, 5), 0.2)
    assert BY.expected_entropy_tel(bel.pi0, bel.hyp, flat) == pytest.approx(h)


def test_tel_entropy_by_enumeration(model, bets):
    bel = make_belief(model, bets)
    c = bel.conf
    want = 0.0
    for o in range(5):
        pi = BY.update_tel(bel.pi0, bel.hyp, c, o)
        po = (BY.marginal(bel.pi0, bel.hyp) * c[:, o]).sum()
        want += po * float(BY.entropy(BY.marginal(pi, bel.hyp)))
    assert BY.expected_entropy_tel(bel.pi0, bel.hyp, c) == pytest.approx(want)


def test_ig_per_cost_picks_cheap_probe_toy(model, bets):
    # mem separates h_mem completely at 1 s; per second it beats any 20 s probe, unit cost does not
    bel = make_belief(model, bets)
    cheap = SIM.gain_order(bel.pi0, [0, 1, 2, 3, 4], bel, True)
    unit = SIM.gain_order(bel.pi0, [0, 1, 2, 3, 4], bel, False)
    assert model.names[cheap] == "mem"
    assert unit != cheap


# ------------------------------------------------------------------ stopping, budget, exclusion


def test_investigation_budget_and_cut(model, bets):
    bel, wld = make_belief(model, bets), make_world(model)
    pol = SIM.Policy("p", 300)
    r = SIM.investigate(0, 0, 0.5, bel, wld, pol, np.random.default_rng(0))
    assert r["seconds"] == 0 and r["cut"] and r["dlog"] == 0
    r = SIM.investigate(0, 0, 10, bel, wld, pol, np.random.default_rng(0))
    assert r["seconds"] == 1 and r["cut"] and r["seq"] == (model.names.index("mem"),)


def test_stop_at_tau_on_strong_signal(model, bets):
    wld = make_world(model, cause=[1, 0, 0, 0, 0, 0], onset=[0, 10**9, 9, 9, 9, 9], s=2.0)
    bel = make_belief(model, bets)
    r = SIM.investigate(0, 5, 1000, bel, wld, SIM.Policy("p", 300), np.random.default_rng(0))
    assert r["pi_max"] >= 0.95 and r["cause"] == 1
    assert r["seconds"] < 121 and not r["cut"]
    assert r["dlog"] > 2


def test_battery_runs_everything(model, bets):
    bel, wld = make_belief(model, bets), make_world(model)
    pol = SIM.Policy("all", 300, probes="battery")
    r = SIM.investigate(0, 0, 1000, bel, wld, pol, np.random.default_rng(0))
    assert r["seconds"] == 121 and len(r["seq"]) == 5


def test_state_before_onset_is_healthy(model, bets):
    bel = make_belief(model, bets)
    wld = make_world(model, cause=[1, 0, 0, 0, 0, 0], onset=[50, 10**9, 9, 9, 9, 9], s=3.0)
    r = SIM.investigate(0, 49, 1000, bel, wld, SIM.Policy("p", 1), np.random.default_rng(0))
    assert r["state"] == 0
    r = SIM.investigate(0, 50, 1000, bel, wld, SIM.Policy("p", 1), np.random.default_rng(0))
    assert r["state"] == 1


def _run(model, bets, pol, T=40, n=6, **kw):
    bel, wld = make_belief(model, bets), make_world(model, n_nodes=n, **kw)
    logM = np.zeros((T, n))
    logM[:, 0] = 5.0
    return SIM.run_policy(logM, bel, wld, pol, np.random.default_rng(1), 0.1, n, 28)


def test_budget_never_exceeded_with_carry(model, bets):
    for B in (60, 300, 1800):
        _, inv = _run(model, bets, SIM.Policy("p", B))
        used = np.zeros(40)
        for r in inv:
            used[r["t"]] += r["seconds"]
        left = 0.0
        for t in range(40):
            avail = min(B, left + B / 4)
            assert used[t] <= avail + 1e-9
            left = avail - used[t]


def test_carry_lets_20s_probe_run_at_b60(model, bets):
    _, inv = _run(model, bets, SIM.Policy("p", 60))
    assert any(len(r["seq"]) >= 2 for r in inv)


def test_carry_capped_at_one_day(model, bets):
    bel, wld = make_belief(model, bets), make_world(model, n_nodes=2)
    bel.cost = np.array([500.0, 500.0, 500.0, 500.0, 500.0])  # nothing fits below the cap
    logM = np.zeros((40, 2))
    _, inv = SIM.run_policy(
        logM, bel, wld, SIM.Policy("p", 400), np.random.default_rng(1), 0.1, 2, 28
    )
    assert len(inv) == 0 or all(r["seconds"] <= 400 for r in inv)
    bel.cost = np.full(5, 401.0)
    _, inv = SIM.run_policy(
        logM, bel, wld, SIM.Policy("p", 400), np.random.default_rng(1), 0.1, 2, 28
    )
    assert inv == []


def test_cleared_nodes_excluded_seven_days(model, bets):
    _, inv = _run(model, bets, SIM.Policy("p", 7200), T=60)
    by = {}
    for r in inv:
        by.setdefault(r["node"], []).append(r)
    for rows in by.values():
        for a, b in zip(rows, rows[1:], strict=False):
            if a["pi_h0"] >= 0.95:
                assert b["t"] - a["t"] >= 28


def test_uncleared_node_stays_eligible(model, bets):
    # node 0 degraded from the start: D86 investigates it again in the next block
    pol = SIM.Policy("p", 300)
    _, inv = _run(model, bets, pol, cause=[1, 0, 0, 0, 0, 0], onset=[0, 9, 9, 9, 9, 9], s=2.0)
    ts = [r["t"] for r in inv if r["node"] == 0]
    assert ts[:3] == [0, 1, 2]


def test_design_rule_excludes_every_investigated_node(model, bets):
    pol = SIM.Policy("p", 300, exclusion="all")
    _, inv = _run(model, bets, pol, cause=[1, 0, 0, 0, 0, 0], onset=[0, 9, 9, 9, 9, 9], s=2.0)
    ts = [r["t"] for r in inv if r["node"] == 0]
    assert all(b - a >= 28 for a, b in zip(ts, ts[1:], strict=False))


def test_skipdone_skips_flagged_diagnosed_node(model, bets):
    pol = SIM.Policy("p", 300, exclusion="cleared_skipdone")
    bel = make_belief(model, bets)
    wld = make_world(model, cause=[1, 0, 0, 0, 0, 0], onset=[0, 9, 9, 9, 9, 9], s=2.0)
    logM = np.zeros((10, 6))
    logM[:, 0] = 30.0  # flagged at every evaluation
    _, inv = SIM.run_policy(logM, bel, wld, pol, np.random.default_rng(1), 0.1, 6, 28)
    assert [r["t"] for r in inv if r["node"] == 0] == [0]


def test_random_nodes_ignore_rank(model, bets):
    pol = SIM.Policy("p", 7200, nodes="random")
    _, inv = _run(model, bets, pol, T=4)
    assert len({r["node"] for r in inv}) > 1


def test_sweep_cycles(model, bets):
    pol = SIM.Policy("p", 1800, nodes="sweep", probes="battery", exclusion="none")
    _, inv = _run(model, bets, pol, T=8, n=6)
    assert [r["node"] for r in inv][:7] == [0, 1, 2, 3, 4, 5, 0]


# ------------------------------------------------------------------ evidence


def test_jumper_scale_equivariance():
    """log E = log M + sum log f, exactly, against the jumper with every state scaled."""
    rng = np.random.default_rng(0)
    T, S, J = 200, 6, 0.01
    a = jumper_init(3, S)
    b = jumper_init(3, S)
    cum = np.zeros(3)
    for _ in range(T):
        lg = np.c_[np.zeros(3), rng.normal(-0.1, 0.5, (3, S - 1))]
        a = jumper_step(a, lg, J)
        b = jumper_step(b, lg, J)
        if rng.uniform() < 0.3:
            lf = rng.normal(0, 2, 3)
            b = b + lf[:, None]  # probe factor on every state of each node
            cum += lf
    assert np.allclose(log_total(b), log_total(a) + cum, atol=1e-9)


def test_b0_equals_layer2_exactly(model, bets):
    bel, wld = make_belief(model, bets), make_world(model)
    logM = np.random.default_rng(0).normal(0, 3, (20, 6))
    logE, inv = SIM.run_policy(
        logM, bel, wld, SIM.Policy("p", 0), np.random.default_rng(1), 0.1, 6, 28
    )
    assert (logE == logM).all() and inv == []


def test_probe_counts_from_next_evaluation(model, bets):
    bel, wld = make_belief(model, bets), make_world(model)
    logM = np.zeros((5, 6))
    logE, inv = SIM.run_policy(
        logM, bel, wld, SIM.Policy("p", 300), np.random.default_rng(1), 0.1, 6, 28
    )
    assert (logE[0] == 0).all()
    r = inv[0]
    assert r["t"] == 0
    t1 = sum(x["dlog"] for x in inv if x["node"] == r["node"] and x["t"] == 0)
    assert logE[1, r["node"]] == pytest.approx(t1)


def test_future_layer2_does_not_change_past_choices(model, bets):
    bel, wld = make_belief(model, bets), make_world(model)
    a = np.random.default_rng(0).normal(0, 3, (30, 6))
    b = a.copy()
    b[15:] += 40 * np.random.default_rng(1).normal(size=(15, 6))

    def run(m):
        return SIM.run_policy(
            m, bel, wld, SIM.Policy("p", 300), np.random.default_rng(2), 0.1, 6, 28
        )

    ea, ia = run(a)
    eb, ib = run(b)
    assert (ea[:15] == eb[:15]).all()
    assert [r["node"] for r in ia if r["t"] < 15] == [r["node"] for r in ib if r["t"] < 15]


def test_null_combined_process_mean_one_with_fresh_pools(model):
    """Null: repeated adaptive bets with a fresh pool per run have E[exp(sum log f)] = 1."""
    ec = {**EC, "n_pool": 50, "edges": [0, 0.02, 0.05, 0.1, 0.2, 0.5, 1], "bet_pools": 400}
    bets = EV.build_bets(model, ec, np.random.default_rng(1))
    bel = make_belief(model, bets)
    rng = np.random.default_rng(9)
    pol = SIM.Policy("p", 300)
    tot, n_sim = [], 6000
    for _ in range(n_sim):
        wld = make_world(
            model, n_nodes=1, n_pool=50, seed=int(rng.integers(1 << 31))
        )
        r = SIM.investigate(0, 0, 1000, bel, wld, pol, rng)
        tot.append(r["dlog"])
    e = np.exp(np.array(tot))
    assert e.mean() == pytest.approx(1, abs=4 * e.std() / np.sqrt(n_sim))


def test_diagnosis_table_rules():
    invs = [
        {"node": 3, "t": 2, "cause": 2, "pi_max": 0.5},
        {"node": 3, "t": 9, "cause": 1, "pi_max": 0.99},
        {"node": 3, "t": 12, "cause": 3, "pi_max": 0.6},
        {"node": 4, "t": 1, "cause": 1, "pi_max": 0.9},
    ]
    t = SIM.diagnosis_table(invs, [(3, 3, 5), (4, 1, 5), (5, 1, 5)])
    assert t[0]["correct"] and t[0]["cause_out"] == "mbw"  # latest after onset
    assert not t[1]["investigated"] and not t[1]["correct"]  # before onset only: wrong (D90)
    assert not t[2]["investigated"]


# ------------------------------------------------------------------ runner


def test_sealed_seeds_raise():
    from hpcdiag.layer3.run_phase5 import check_sealed

    check_sealed(["null__rand199", "node__rand19__K1__s1.3__seed19", "null__real"])
    check_sealed(["user_decoy__rand03__s2.0__seed19"])
    with pytest.raises(RuntimeError):
        check_sealed(["node__rand20__K1__s1.3__seed20"])
    with pytest.raises(RuntimeError):
        check_sealed(["null__rand200"])


def test_runner_does_not_read_the_fresh_index():
    from pathlib import Path

    import hpcdiag.layer3.run_phase5 as RP

    assert "index_fresh" not in Path(RP.__file__).read_text()


def test_misspec_pool_uses_true_sigma():
    from hpcdiag.layer3 import setup as ST
    from hpcdiag.utils.run import load_config

    cfg = load_config("configs/phase5.yaml")
    v = ST.VARIANTS["misspec"]
    w = ST.make_world(cfg, v, 1.3, np.zeros(3, int), np.zeros(3, int), 1, 2)
    assert w.sigma == 0.1
    assert 0.09 < w.pools[0].std() < 0.11
    bel = ST.make_belief(cfg, v)
    assert bel.sigma == 0.05


def test_telemetry_rows_smoothed():
    from hpcdiag.layer3 import setup as ST

    c = np.diag([8, 8, 8, 8, 8]).tolist()
    r = ST.telemetry_rows({"telemetry": {"counts": c}})
    assert np.allclose(r.sum(axis=1), 1) and (r > 0).all()
    assert r[0, 0] == pytest.approx(9 / 13)


def test_telemetry_variants_use_their_matrices():
    from hpcdiag.layer3 import setup as ST
    from hpcdiag.utils.run import load_config

    cfg = load_config("configs/phase5.yaml")
    oof = ST.telemetry_rows(cfg, "oof")
    cross = ST.telemetry_rows(cfg, "crossblock")
    design = ST.telemetry_rows(cfg, "design")
    assert oof[1, 1] > 0.99 and cross[0, 0] < 0.9 and design[1, 1] == 0.8
    v = ST.VARIANTS["tel_transfer"]
    assert (ST.make_belief(cfg, v).conf == oof).all()
    w = ST.make_world(cfg, v, 1.3, np.zeros(2, int), np.zeros(2, int), 1, 2)
    assert (w.conf == cross).all()


def test_config_keeps_null_kind():
    from hpcdiag.utils.run import load_config

    assert load_config("configs/phase5.yaml")["grid"]["validity_kinds"][0] == "null"
