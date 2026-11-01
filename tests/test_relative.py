"""Relative (cross-sectional) e-values (D78, D80).

The enumeration tests below are exact. They show that the pair-weighted denominator of D78 gives
mean 1 for equal job sizes but not for mixed sizes, and that the node-weighted denominator gives
mean 1 for any sizes.
"""

from __future__ import annotations

from itertools import combinations, product

import numpy as np
import pandas as pd
import pytest

from hpcdiag.layer2.accumulate import node_evidence_relative, snapshots
from hpcdiag.stats.ebh import ebh_log
from hpcdiag.stats.evalues import node_logM_by_rank, relative_evalues, relative_logM


def _enumerate(sizes, f, Q, lam=1.0, denom="pair"):
    """Mean update factor of each node over ALL placements (each job: uniform |N_j|-subset of Q
    nodes, independently, as in inject.placement). No pairs in the block: factor 1 (no update).
    Returns (mean factor per node, mean e per node given at least one pair)."""
    tot, cond, active, n = np.zeros(Q), np.zeros(Q), np.zeros(Q), 0
    for pl in product(*[list(combinations(range(Q), s)) for s in sizes]):
        node = np.array([x for nodes in pl for x in nodes])
        job = np.array([j for j, nodes in enumerate(pl) for _ in nodes])
        fv = np.asarray(f, float)[job][:, None]
        _, nd, e = relative_evalues(
            np.zeros(len(node), int), np.zeros(len(node), int), node, fv, Q, denom
        )
        fac = np.ones(Q)
        fac[nd] = 1 + lam * (e[:, 0] - 1)
        tot += fac
        cond[nd] += e[:, 0]
        active[nd] += 1
        n += 1
    return tot / n, cond / active


@pytest.mark.parametrize("denom", ["pair", "node"])
@pytest.mark.parametrize("lam", [1.0, 0.25])
def test_enumeration_equal_sizes_mean_one(denom, lam):
    for sizes, f in [((1, 1, 1), (3.0, 0.5, 0.0)), ((2, 2, 2), (4.0, 1.0, 0.25))]:
        fac, cond = _enumerate(sizes, f, 4, lam, denom)
        np.testing.assert_allclose(fac, 1.0)
        np.testing.assert_allclose(cond, 1.0)


def test_d78_pair_weighted_is_not_fair_for_mixed_sizes_hand_case():
    """Counterexample to the D78 proof. Job A size 1, job B size 2, Q = 3 nodes, 9 placements. For
    node 0: both jobs 2, A only 1, B only 4, none 2. With f_A = 3, f_B = 0 the pair-weighted mean
    is (3 + 0 + 0) / 3 = 1, so e = 1.5, 3, 0: mean factor (2*1.5 + 3 + 0 + 2) / 9 = 8/9, and mean
    e given at least one pair = 6/7. With f_A = 0, f_B = 3: fbar = 2, mean factor 19/18 > 1."""
    fac, cond = _enumerate((1, 2), (3.0, 0.0), 3)
    np.testing.assert_allclose(fac, 8 / 9)
    np.testing.assert_allclose(cond, 6 / 7)
    fac, _ = _enumerate((1, 2), (0.0, 3.0), 3)
    np.testing.assert_allclose(fac, 19 / 18)


@pytest.mark.parametrize("lam", [1.0, 0.25])
@pytest.mark.parametrize(
    "sizes,f,Q",
    [
        ((1, 2), (3.0, 0.0), 3),
        ((1, 2), (0.0, 3.0), 3),
        ((1, 3, 2), (5.0, 0.2, 1.0), 4),
        ((2, 1, 1, 3), (0.0, 7.0, 2.0, 0.5), 4),
    ],
)
def test_node_weighted_denominator_is_exact_for_mixed_sizes(sizes, f, Q, lam):
    """Node-weighted block mean: e averages 1 over the active nodes of every placement, and node
    labels are exchangeable, so the mean is exactly 1 for any sizes."""
    fac, cond = _enumerate(sizes, f, Q, lam, "node")
    np.testing.assert_allclose(fac, 1.0)
    np.testing.assert_allclose(cond, 1.0)


def test_block_all_zero_bets_gives_one():
    fv = np.zeros((3, 1))
    _, _, e = relative_evalues(
        np.zeros(3, int), np.zeros(3, int), np.array([0, 1, 2]), fv, 3, "pair"
    )
    np.testing.assert_array_equal(e, 1.0)


def _two_blocks():
    block = np.array([0, 0, 0, 0, 1, 1])
    part = np.array([0, 0, 1, 1, 0, 0])
    node = np.array([0, 1, 2, 3, 0, 1])
    f = np.array([[3.0], [1.0], [8.0], [2.0], [0.5], [1.5]])
    return block, part, node, f


@pytest.mark.parametrize("denom", ["pair", "node"])
def test_means_are_per_partition_and_per_block(denom):
    block, part, node, f = _two_blocks()
    b, nd, e = relative_evalues(block, part, node, f, 4, denom)
    assert b.tolist() == [0, 0, 0, 0, 1, 1] and nd.tolist() == [0, 1, 2, 3, 0, 1]
    np.testing.assert_allclose(e[:, 0], [1.5, 0.5, 1.6, 0.4, 0.5, 1.5])
    g = f.copy()
    g[2:4] = 100.0  # other partition, same block
    g[4:] = [[9.0], [1.0]]  # same partition, later block
    _, _, e2 = relative_evalues(block, part, node, g, 4, denom)
    np.testing.assert_allclose(e2[:2, 0], e[:2, 0])


def test_pair_mean_counts_a_job_once_per_node_and_node_mean_averages_nodes():
    """Block with nodes 0, 0, 1 and f = 4, 2, 0: pair mean 2; node means 3 and 0 average to 1.5."""
    block, part = np.zeros(3, int), np.zeros(3, int)
    node = np.array([0, 0, 1])
    f = np.array([[4.0], [2.0], [0.0]])
    np.testing.assert_allclose(relative_evalues(block, part, node, f, 2, "pair")[2][:, 0], [1.5, 0])
    np.testing.assert_allclose(relative_evalues(block, part, node, f, 2, "node")[2][:, 0], [2.0, 0])


@pytest.mark.parametrize("lam", [0.0, -0.1, 1.5, np.array([0.5, 0.5])])
def test_lambda_must_be_one_constant_in_range(lam):
    with pytest.raises(ValueError):
        relative_logM(np.array([0, 1]), np.ones((2, 1)), lam, 0.01)


def test_update_formula_with_jumper_at_zero_switching():
    """J = 0, states {no bet, bet}: M = 1/2 + 1/2 (1 + lam (e - 1)) per step."""
    lm = relative_logM(np.array([0, 0]), np.array([[3.0], [0.5]]), 0.25, 0.0)
    a, b = 1 + 0.25 * 2, 1 + 0.25 * (-0.5)
    np.testing.assert_allclose(np.exp(lm), [0.5 + 0.5 * a, 0.5 + 0.5 * a * b])


def _pairs():
    return pd.DataFrame(
        {
            "k": [0, 1, 1, 2, 3],
            "node_id": [0, 0, 1, 1, 3],
            "n_nodes": [1, 2, 2, 1, 1],
            "end_ns": [5, 6, 6, 15, 16],
        }
    )


def test_node_evidence_relative_blocks_and_nodes_without_pairs():
    """Blocks (0, 10] and (10, 20]; job 0 has no p-value. Node 2 has no pairs: never updated. Job 3
    is alone in its partition, so e = 1 for it. Every update is dated at its block end."""
    p = np.array([np.nan, 0.1, 0.8, 0.4])
    part = np.array([0, 0, 0, 1])
    bets = [lambda x: 2 * (1 - x)]
    ev = node_evidence_relative(pairs := _pairs(), p, part, bets, 0.01, 1.0, np.array([10, 20]))
    assert ev["end_ns"].tolist() == [10, 10, 20, 20]
    assert ev["node_id"].tolist() == [0, 1, 1, 3]
    s = snapshots(ev, np.array([10, 20]), 4)
    assert s[1, 2] == 0.0
    assert abs(s[1, 3]) < 1e-12
    assert s[0, 0] == pytest.approx(0.0, abs=1e-9)  # one job on both nodes: e = 1 for each
    with pytest.raises(ValueError):
        node_evidence_relative(
            pairs.assign(node_id=[0, 0, 1, 3, 3]), p, part, bets, 0.01, 1.0, np.array([10, 20])
        )


def _sim_bursts(seed, m=100, blocks=120, jobs=40):
    """Global null with strong bursts, random placement. A two-state block regime; in a burst all
    p-values are small. Jobs sizes are mixed. Returns e-BH rejection flags per method."""
    rng = np.random.default_rng(seed)
    burst, states = False, []
    for _ in range(blocks):
        burst = rng.random() < (0.8 if burst else 0.05)
        states.append(burst)
    rows = []
    for b in range(blocks):
        for _ in range(jobs):
            s = int(rng.choice([1, 1, 2, 4, 8]))
            p = rng.beta(0.3, 1.0) if states[b] else rng.uniform()
            rows += [(b, n, p) for n in rng.choice(m, s, replace=False)]
    x = np.array(rows)
    block, node, p = x[:, 0].astype(int), x[:, 1].astype(int), x[:, 2]
    f = np.column_stack([2 * (1 - p), np.where(p < 0.5, 1.5, 0.5)])
    out = {}
    for lab, denom, lam in (("R1", "pair", 1.0), ("R025", "pair", 0.25), ("Rn1", "node", 1.0)):
        b, nd, e = relative_evalues(block, np.zeros(len(p), int), node, f, m, denom)
        out[lab] = _ever(b, nd, relative_logM(nd, e, lam, 0.01), m, blocks)
    out["w1"] = _ever(block, node, node_logM_by_rank(node, np.log(f), 0.01), m, blocks)
    return out


def _ever(block, node, lm, m, blocks):
    cur = np.zeros(m)
    ends = np.searchsorted(block, np.arange(blocks), side="right")
    a = 0
    for b in ends:
        for i in range(a, b):
            cur[node[i]] = lm[i]
        a = b
        if ebh_log(cur, 0.1, m).any():
            return True
    return False


def test_bursts_null_relative_valid_and_full_bet_fails():
    n = 200
    r = pd.DataFrame([_sim_bursts(s) for s in range(n)]).mean()
    se = np.sqrt(0.09 / n)
    assert r["R1"] <= 0.1 + 3 * se
    assert r["R025"] <= 0.1 + 3 * se
    assert r["Rn1"] <= 0.1 + 3 * se
    assert r["w1"] > 0.5
