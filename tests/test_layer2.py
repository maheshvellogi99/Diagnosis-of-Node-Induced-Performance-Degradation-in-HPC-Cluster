"""Layer 2 building blocks: guards, decoy weight, snapshots, selection, metrics (Phase 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hpcdiag.layer2.accumulate import (
    decoy_counts,
    guard_b_frozen,
    make_pairs,
    node_evidence,
    snapshots,
    weights,
)
from hpcdiag.layer2.metrics import node_metrics, null_metrics
from hpcdiag.layer2.prep import sliding_median_prev
from hpcdiag.layer2.select import ebh_over_time, residual_baselines
from hpcdiag.stats.calibrators import KAPPAS

H = 3600 * 10**9
DAY = 24 * H
T0 = pd.Timestamp("2020-01-01")


def test_sliding_median_prev():
    s = np.array([5.0, 1.0, 3.0, 10.0])
    e = np.array([0, 1, 2, 10]) * H
    # j0: none -> 0; j1: [5] -> 5; j2: [5,1] -> 3; j3 (6 h window): ends 0..2 h < 4 h -> none
    np.testing.assert_allclose(sliding_median_prev(s, e, 6 * H), [0, 5, 3, 0])


def test_decoy_counts_strictly_before_same_user_node_24h():
    jobs = pd.DataFrame({"row_id": [0, 1, 2, 3, 4], "user": ["u", "u", "u", "v", "u"],
                         "start_time": [T0 + pd.Timedelta(hours=h) for h in (0, 2, 2, 3, 25)]})
    pl = pd.DataFrame({"row_id": [0, 1, 2, 3, 4], "node_id": [7, 7, 7, 7, 7], "n_nodes": 1})
    c = decoy_counts(pl, jobs, 24).set_index("row_id")["c"]
    # job 1 and 2 start together: neither counts the other (not strictly before); both count job 0
    # job 4: jobs 1 and 2 are within 24 h before it; job 0 is not
    assert c.to_dict() == {0: 0, 1: 1, 2: 1, 3: 0, 4: 2}
    pairs = pl.assign(user="u", k=range(5), start_time=jobs["start_time"], end_ns=0)
    w = weights(pairs, decoy_counts(pl, jobs, 24))
    np.testing.assert_allclose(w, [1, 1, 1, 1, 0.5])


def test_guard_b_predictable():
    rng = np.random.default_rng(0)
    n = 400
    e = np.sort(rng.integers(0, 50, n)) * H              # many equal end times
    p = rng.uniform(size=n)
    p[150:200] = 1e-4                                    # a burst of small p-values
    fr = guard_b_frozen(p, e, 0.01, 100.0)
    assert fr[200:].any()                                # the burst triggers the signal later on
    for j in (120, 180, 250):
        q = p.copy()
        same_or_later = e >= e[j]
        q[same_or_later] = rng.uniform(size=same_or_later.sum())
        assert guard_b_frozen(q, e, 0.01, 100.0)[j] == fr[j]


def _pop_pairs():
    pop = pd.DataFrame({"row_id": [10, 11, 12, 13], "user": ["a", "b", "a", "c"],
                        "start_time": [T0] * 4, "end_ns": np.array([1, 2, 3, 4]) * H})
    pl = pd.DataFrame({"row_id": [10, 11, 11, 12, 13], "node_id": [0, 0, 1, 1, 2],
                       "n_nodes": [1, 2, 2, 1, 1]})
    return pop, make_pairs(pop, pl)


def test_pairs_weights_snapshots():
    pop, pairs = _pop_pairs()
    assert pairs["k"].tolist() == [0, 1, 1, 2, 3] and pairs["node_id"].tolist() == [0, 0, 1, 1, 2]
    np.testing.assert_allclose(weights(pairs), [1, 0.5, 0.5, 1, 1])
    p = np.array([0.5, np.nan, 0.01, 0.2])               # job 11 in warm-up: no update
    ev = node_evidence(pairs, p, weights(pairs), KAPPAS, None)
    assert ev["node_id"].tolist() == [0, 1, 2]
    snap = snapshots(ev, np.array([0, 1, 3, 10]) * H, 3)
    assert snap[0].tolist() == [0, 0, 0]                 # nothing ended yet
    assert snap[1, 0] == ev["logM"].iloc[0] and snap[1, 1] == 0
    np.testing.assert_allclose(snap[3], ev["logM"].to_numpy())


def test_ebh_over_time_fixed_m():
    logM = np.log(np.array([[1.0, 1.0, 1.0], [40.0, 1.0, 1.0], [40.0, 25.0, 1.0]]))
    f = ebh_over_time(logM, 0.1, m=3)
    # row 1: 1 * 40 / 3 >= 10 -> flag node 0; row 2: k=2: 2 * 25 / 3 = 16.7 >= 10 -> both
    assert f.tolist() == [[False] * 3, [True, False, False], [True, True, False]]
    assert not ebh_over_time(logM, 0.1, m=10)[1].any()  # 40 / 10 < 10


def test_null_and_node_metrics_hand():
    ev = np.array([0, 1, 2, 3, 4]) * DAY
    flags = np.zeros((5, 4), bool)
    flags[1, 3] = True                                  # node 3 flagged at day 1 (false)
    flags[3, 0] = flags[4, 0] = True                    # node 0 flagged from day 3
    nm = null_metrics(flags, ev)
    assert nm["ever_flagged"] and nm["first_flag_eval"] == 1 and nm["node_evals_flagged"] == 3
    truth = pd.DataFrame({"node_id": [0, 1], "onset_time": [T0 + pd.Timedelta(days=2)] * 2})
    truth["onset_time"] = truth["onset_time"]
    ev_abs = ev + T0.value
    out = node_metrics(flags, ev_abs, truth, affected_main_nodes={0}, coalloc_nodes={3},
                       node_jobs_end={0: T0.value + np.array([1, 2.5, 2.7, 3.5]) * DAY},
                       horizon_days=14, offsets_days=[1])
    assert out["recall_a"] == 0.5 and out["recall_b"] == 1.0
    assert out["median_delay_h"] == pytest.approx(24.0)
    assert out["median_delay_jobs"] == 2                # jobs ending in (onset, detection]
    # FDP per eval: [0, 1 (node 3 false), 0, 0, 0] -> mean 0.2; onset+1 d = day 3 -> FDP 0
    assert out["fdp_mean_over_time"] == pytest.approx(0.2)
    assert out["fdp_at_onset_plus_1d"] == 0.0
    assert out["false_nodes_coallocated"] == 1 and out["false_nodes_unrelated"] == 0


def test_b1_persistence():
    rng = np.random.default_rng(1)
    n = 400
    days = np.arange(n) % 20
    users = rng.choice(["a", "b", "c", "d"], n)
    end = days * DAY + rng.integers(0, DAY - 1, n)
    pairs = pd.DataFrame({"k": np.arange(n), "node_id": np.where(days >= 10, 1, 0),
                          "end_ns": end})
    r = np.where(days >= 10, 0.5, 0.0)                    # node 1 late from day 10 on
    cfg = {"baselines": {"window_days": 7, "b2_top_frac": 0.5,
                         "b1": {"median_rel_resid": 0.2, "min_jobs": 10, "min_users": 3,
                                "persist": 3}}}
    day_ns = (np.arange(1, 21) * DAY) - 1
    out = residual_baselines(pairs, r, users, day_ns, 2, cfg)
    first = int(np.argmax(out["B1"][:, 1]))
    assert out["B1"][:, 1].any() and not out["B1"][:, 0].any()
    assert first >= 12                                   # condition from day ~10, persists 3 days


def test_paired_check_and_before_after():
    from hpcdiag.layer2.power import before_after, paired_check
    ps = pd.Series([0.1, 0.2, np.nan, 0.9], index=[1, 2, 3, 4])
    pn = pd.Series([0.3, 0.2, 0.5, 0.95], index=[1, 2, 3, 4])
    r = paired_check(ps, pn, np.array([1, 2, 3, 4, 99]))
    assert r["n_pairs"] == 3 and r["median_diff"] == pytest.approx(-0.05)
    assert r["share_negative"] == pytest.approx(2 / 3) and r["share_zero"] == pytest.approx(1 / 3)
    ba = before_after(pd.Series([0.1, 0.9, 0.5]), pd.Series([1, 5, 9]), 4)
    assert ba == {"n_before": 1, "n_after": 2, "median_before": 0.1, "median_after": 0.7}


def _brute_centre(s, e, u, nr, wt, W, min_n=3):
    out, lev = s.copy(), np.full(len(s), 2)
    for j in range(len(s)):
        prev = [i for i in range(len(s)) if e[i] < e[j] and e[i] >= e[j] - W]
        a = [s[i] for i in prev if u[i] == u[j] and nr[i] == nr[j] and wt[i] == wt[j]]
        b = [s[i] for i in prev if u[i] == u[j]]
        if len(a) >= min_n:
            out[j], lev[j] = s[j] - np.median(a), 0
        elif len(b) >= min_n:
            out[j], lev[j] = s[j] - np.median(b), 1
    return out, lev


def test_centre_scores_matches_brute_force():
    from hpcdiag.layer2.centred import centre_scores
    rng = np.random.default_rng(3)
    n = 600
    e = np.sort(rng.integers(0, 3000, n))               # many ties in end time
    s = rng.standard_normal(n)
    u = rng.choice(["a", "b", "c"], n)
    nr = rng.choice([1, 2], n)
    wt = rng.choice([600.0, 3600.0], n)
    got, lev = centre_scores(s, e, u, nr, wt, 500)
    exp, elev = _brute_centre(s, e, u, nr, wt, 500)
    np.testing.assert_allclose(got, exp)
    np.testing.assert_array_equal(lev, elev)
    assert set(np.unique(lev)) == {0, 1, 2}


def test_centre_scores_excludes_equal_end_times():
    from hpcdiag.layer2.centred import centre_scores
    s = np.array([1.0, 2.0, 3.0, 10.0, 10.0])
    e = np.array([0, 1, 2, 5, 5])
    u = np.array(["a"] * 5)
    sc, lev = centre_scores(s, e, u, np.ones(5), np.ones(5), 100)
    # jobs 3 and 4 end together: each sees only jobs 0-2 (median 2), not each other
    np.testing.assert_allclose(sc[3:], [8.0, 8.0])
    assert lev.tolist() == [2, 2, 2, 0, 0]
