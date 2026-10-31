"""Predictable blame allocation (D71): hand cases, predictability, fairness, D49 under the null."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import integrate

from hpcdiag.layer2.accumulate import node_evidence, node_evidence_blame, snapshots
from hpcdiag.stats.calibrators import power_calibrator
from hpcdiag.stats.ebh import ebh_log
from hpcdiag.stats.evalues import blame_logM, blame_reference, node_logM_by_rank


def _layout(jobs: list[list[int]], blocks: list[int] | None = None):
    """Pair arrays (job, block, node) from a list of node sets, one tie block per job by default."""
    blocks = list(range(len(jobs))) if blocks is None else blocks
    job = np.concatenate([[j] * len(n) for j, n in enumerate(jobs)])
    block = np.concatenate([[blocks[j]] * len(n) for j, n in enumerate(jobs)])
    node = np.concatenate(jobs)
    return job, block, node


def _one_state(g_per_job, job):
    """One betting state: g (n_pairs, 1) from one value per job."""
    return np.asarray(g_per_job, float)[job][:, None]


def test_first_job_splits_equally():
    job, block, node = _layout([[0, 1, 2, 3]])
    _, lam = blame_reference(job, block, node, _one_state([3.0], job), 0.0, 4)
    np.testing.assert_allclose(lam, 0.25)


def test_single_node_jobs_get_lambda_one_and_match_full_weight():
    rng = np.random.default_rng(0)
    node_list = rng.integers(0, 5, 200)
    job, block, node = _layout([[int(n)] for n in node_list])
    g = rng.uniform(0.2, 3.0, (200, 3))
    lm, lam = blame_reference(job, block, node, g, 0.01, 5)
    np.testing.assert_allclose(lam, 1.0)
    np.testing.assert_allclose(lm, node_logM_by_rank(node, np.log(g), 0.01))


def test_equal_capital_reduces_to_inverse_size():
    """All jobs on the same node set: every M stays equal, so lambda = 1/|N| at every step."""
    rng = np.random.default_rng(1)
    job, block, node = _layout([[0, 1, 2]] * 40)
    g = rng.uniform(0.2, 3.0, 40)[job][:, None]
    lm, lam = blame_reference(job, block, node, g, 0.01, 3)
    np.testing.assert_allclose(lam, 1 / 3)
    np.testing.assert_allclose(lm, node_logM_by_rank(node, np.log1p((g - 1) / 3), 0.01))


def test_hand_explaining_away():
    """J = 0, states {no bet, bet}, start capital 1/2 each. Job 0 on node 0 alone with g = 7:
    M_0 = 1/2 + 7/2 = 4, M_1 = 1. Job 1 on {0, 1}: lambda = 4/5 and 1/5. With g = 2:
    M_0 = 1/2 + 7/2 (1 + 4/5) = 6.8; M_1 = 1/2 + 1/2 (1 + 1/5) = 1.1."""
    job, block, node = _layout([[0], [0, 1]])
    lm, lam = blame_reference(job, block, node, _one_state([7.0, 2.0], job), 0.0, 2)
    np.testing.assert_allclose(lam, [1.0, 0.8, 0.2])
    np.testing.assert_allclose(np.exp(lm), [4.0, 6.8, 1.1])


def test_tie_block_uses_state_before_the_block():
    """Same three updates as the hand case, but jobs 0 and 1 tie: job 1 cannot see job 0, so its
    lambda is 1/2 on both nodes. Node 0 then has both updates (bet state 7 x 1.5 = 10.5, /2)."""
    job, block, node = _layout([[0], [0, 1]], blocks=[0, 0])
    lm, lam = blame_reference(job, block, node, _one_state([7.0, 2.0], job), 0.0, 2)
    np.testing.assert_allclose(lam, [1.0, 0.5, 0.5])
    np.testing.assert_allclose(np.exp(lm), [4.0, 0.5 + 10.5 / 2, 0.5 + 1.5 / 2])


def _random_layout(seed, n_jobs=400, n_nodes=30):
    rng = np.random.default_rng(seed)
    sizes = rng.choice([1, 1, 1, 2, 3, 8], n_jobs)
    jobs = [sorted(rng.choice(n_nodes, s, replace=False).tolist()) for s in sizes]
    blocks = np.cumsum(rng.random(n_jobs) < 0.6).tolist()  # many ties
    job, block, node = _layout(jobs, blocks)
    g = rng.uniform(0.0, 4.0, (n_jobs, 3))[job]  # g = 0 reachable: log capital -inf in a state
    return job, block, node, g, n_nodes, rng


@pytest.mark.parametrize("seed", range(5))
def test_lambda_range_and_sum_to_one_per_job(seed):
    job, block, node, g, m, _ = _random_layout(seed)
    _, lam = blame_reference(job, block, node, g, 0.01, m)
    assert np.all((lam > 0) & (lam <= 1))
    np.testing.assert_allclose(np.bincount(job, lam), 1.0)


@pytest.mark.parametrize("seed", range(5))
def test_lambda_uses_only_earlier_blocks(seed):
    """Changing the bets of block b and later leaves lambda unchanged up to and including block b,
    and a change in an earlier block does reach some later lambda."""
    job, block, node, g, m, rng = _random_layout(seed)
    _, lam0 = blame_reference(job, block, node, g, 0.01, m)
    b = int(np.median(block))
    g2 = g.copy()
    later = block >= b
    g2[later] = rng.uniform(0.0, 4.0, g2[later].shape)
    _, lam1 = blame_reference(job, block, node, g2, 0.01, m)
    np.testing.assert_array_equal(lam0[block <= b], lam1[block <= b])
    g3 = g.copy()
    early = block < b
    g3[early] = rng.uniform(0.0, 4.0, g3[early].shape)
    _, lam2 = blame_reference(job, block, node, g3, 0.01, m)
    assert not np.allclose(lam0[block >= b], lam2[block >= b])


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("J", [0.0, 0.01])
def test_vectorised_matches_reference(seed, J):
    job, block, node, g, m, _ = _random_layout(seed)
    lm0, lam0 = blame_reference(job, block, node, g, J, m)
    lm1, lam1 = blame_logM(job, block, node, g, J, m)
    np.testing.assert_allclose(lam1, lam0, rtol=1e-12)
    np.testing.assert_allclose(lm1, lm0, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("lam", [0.05, 0.3, 0.8, 1.0])
@pytest.mark.parametrize("kappa", [0.25, 0.5])
def test_one_step_fair_by_quadrature_with_lambda_fixed(lam, kappa):
    """For lambda fixed before p is seen: E[1 + lambda (f(p) - 1)] = 1 under p ~ U(0, 1), for a
    calibrator and a histogram bet, and E[M after] = M before for the two-state jumper."""
    fint = integrate.quad(lambda x: kappa, 0, 1, weight="alg", wvar=(kappa - 1, 0))[0]
    assert abs(1 + lam * (fint - 1) - 1) < 1e-6
    h = np.array([3.0, 1.5, 0.4, 0.1])
    assert h.mean() == pytest.approx(1.25)
    h = h / h.mean()
    assert np.mean(1 + lam * (h - 1)) == pytest.approx(1.0)
    # jumper: capital (1, 3) with J = 0.01: after mixing the total is 4; the bet state keeps its
    # mixed capital in expectation.
    J, C = 0.01, np.array([1.0, 3.0])
    mixed = (1 - J) * C + J / 2 * C.sum()
    assert np.mean(mixed[0] + mixed[1] * (1 + lam * (h - 1))) == pytest.approx(C.sum())
    assert mixed[0] + mixed[1] * (1 + lam * (fint - 1)) == pytest.approx(C.sum(), abs=1e-5)


def test_one_step_fair_through_the_implementation():
    """E over p of M_n after one blame update equals M_n before, for every node, computed by exact
    averaging over the bins of a histogram bet on the real code path, with unequal capital."""
    h = np.array([3.0, 1.5, 0.4, 0.1])
    h = h / h.mean()
    job, block, node = _layout([[0], [1], [0, 1, 2]])
    gs = [[5.0, 0.5, v] for v in h]
    after = []
    for row in gs:
        lm, lam = blame_reference(job, block, node, _one_state(row, job), 0.01, 3)
        after.append(np.exp(lm[-3:]))
    before = np.exp(blame_reference(job, block, node, _one_state([5.0, 0.5, 1.0], job), 0.01, 3)[0])
    np.testing.assert_allclose(np.mean(after, axis=0), before[-3:])


def _sim_null(seed, m=20, n_jobs=300, J=0.01):
    """Global null on a small cluster: uniform p, mixed job sizes, bets g(p) = 2 (1 - p)
    (integral 1) and g(p) = 1.5 for p < 1/2 else 0.5. Returns the logM matrix at every 10th job."""
    rng = np.random.default_rng(seed)
    sizes = rng.choice([1, 1, 2, 4], n_jobs)
    jobs = [rng.choice(m, s, replace=False).tolist() for s in sizes]
    job, block, node = _layout(jobs)
    p = rng.uniform(size=n_jobs)
    g = np.column_stack([2 * (1 - p), np.where(p < 0.5, 1.5, 0.5)])[job]
    lm, _ = blame_logM(job, block, node, g, J, m)
    cur = np.zeros(m)
    snaps = []
    stops = set(np.flatnonzero(np.diff(block // 10)).tolist())
    for i in range(len(node)):
        cur[node[i]] = lm[i]
        if i in stops:
            snaps.append(cur.copy())
    return np.array(snaps)


def test_null_mean_at_stopping_time_and_ever_flagged():
    """D49 by simulation: each stopped M_n has mean about 1 and P(e-BH ever rejects) stays below
    alpha. The bets are small, so the rejection rate is far below alpha; the mean check matters."""
    alpha, stopped, ever = 0.5, [], 0
    for seed in range(600):
        S = _sim_null(seed)
        t_hit = S.shape[0] - 1
        for t in range(S.shape[0]):
            if ebh_log(S[t], alpha, S.shape[1]).any():
                t_hit, ever = t, ever + 1
                break
        stopped.append(np.exp(S[t_hit]))
    v = np.array(stopped)
    se = v.mean(axis=1).std() / np.sqrt(len(v))
    assert v.mean() <= 1 + 3 * se
    assert ever / len(v) <= alpha


def _pairs():
    pop = pd.DataFrame({"k": [0, 1, 2], "end_ns": [10, 20, 20]})
    return pd.DataFrame(
        {
            "k": [0, 1, 1, 2],
            "node_id": [0, 0, 1, 2],
            "n_nodes": [1, 2, 2, 1],
            "end_ns": pop["end_ns"].to_numpy()[[0, 1, 1, 2]],
        }
    )


def test_node_evidence_blame_equal_to_v1_on_disjoint_equal_capital_and_snapshots():
    """NaN p-value (warm-up): no update and no lambda. Blame output feeds snapshots unchanged."""
    p = np.array([np.nan, 0.2, 0.7])
    bets = [lambda x: 2 * (1 - x)]
    pairs = _pairs()
    ev = node_evidence_blame(pairs, p, bets, 0.01)
    v1 = node_evidence(pairs, p, 1.0 / pairs["n_nodes"].to_numpy(float), bets, 0.01)
    assert len(ev) == 3 and set(ev.columns) >= {"node_id", "end_ns", "logM", "lam"}
    np.testing.assert_allclose(ev["lam"], [0.5, 0.5, 1.0])
    np.testing.assert_allclose(ev["logM"], v1["logM"])  # first update of each node: M all equal
    s = snapshots(ev, np.array([5, 25]), 3)
    assert s[0].tolist() == [0.0, 0.0, 0.0] and np.allclose(s[1], ev["logM"])


def test_node_evidence_blame_accepts_calibrators():
    pairs = _pairs()
    p = np.array([0.3, 0.2, 0.7])
    ev = node_evidence_blame(pairs, p, (0.25, 0.5), 0.01)
    bets = [lambda x, k=k: power_calibrator(x, k) for k in (0.25, 0.5)]
    pd.testing.assert_frame_equal(ev, node_evidence_blame(pairs, p, bets, 0.01))
