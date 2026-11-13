"""Gate 3: injection follows the formulas exactly; unaffected rows identical; truth exact."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hpcdiag.inject.placement import random_placement
from hpcdiag.inject.scenarios import (
    Window,
    apply_multiplier,
    decoy_draw,
    decoy_scenario,
    draw_degraded_nodes,
    node_scenario,
    shock_scenario,
)
from hpcdiag.inject.store import materialize
from hpcdiag.inject.verify import check_changes
from hpcdiag.io.contract import JOBS_SCHEMA

T0 = pd.Timestamp("2020-01-01")
H = pd.Timedelta(hours=1)
ELIG = ["COMPLETED", "TIMEOUT"]


def _jobs():
    rows = [  # user, start (h), T (s), limit (s), state, nodes
        ("a", 0, 3600, 7200, "COMPLETED", [1]),
        ("a", 10, 3600, 3900, "COMPLETED", [1, 2]),
        ("b", 10, 1000, 9999, "FAILED", [1]),
        ("b", 5, 36000, 99999, "COMPLETED", [2]),
        ("c", 20, 0, 600, "COMPLETED", [3]),
        ("a", 12, 100, 100, "TIMEOUT", [3]),
    ]
    start = [T0 + h * H for _, h, *_ in rows]
    T = [float(t) for _, _, t, *_ in rows]
    df = pd.DataFrame(
        {
            "row_id": np.arange(len(rows)),
            "job_id": np.arange(100, 100 + len(rows)),
            "system": "pm100",
            "user": [r[0] for r in rows],
            "partition": "P",
            "qos": "1",
            "account": pd.Series([None] * len(rows), dtype="str"),
            "name": pd.Series([None] * len(rows), dtype="str"),
            "submit_time": start,
            "start_time": start,
            "end_time": [s + pd.Timedelta(seconds=t) for s, t in zip(start, T, strict=True)],
            "run_time_s": T,
            "walltime_req_s": [float(r[3]) for r in rows],
            "nodes_req": [len(r[5]) for r in rows],
            "procs_req": 1,
            "gpus_req": 0,
            "mem_req": 0.0,
            "state_raw": [r[4] for r in rows],
        }
    )
    df["state"] = df["state_raw"].where(df["state_raw"].isin(ELIG), "other")
    df["censored"] = df["state"] == "TIMEOUT"
    for c in ("submit_time", "start_time", "end_time"):
        df[c] = df[c].astype("datetime64[us]")
    jn = pd.DataFrame(
        [(i, n) for i, r in enumerate(rows) for n in r[5]], columns=["row_id", "node_id"]
    )
    return df[JOBS_SCHEMA.names], jn


def _deg():
    return pd.DataFrame(
        {"node_id": [1, 2], "onset_time": [T0 + 9 * H, T0 + 10 * H], "cause": ["cpu", "mbw"]}
    ).astype({"onset_time": "datetime64[us]"})


def _elig(df):
    return df["state_raw"].isin(ELIG).to_numpy()


END = T0 + 100 * H


def test_node_kind_formula_and_censoring():
    jobs, jn = _jobs()
    ch, truth = node_scenario(jobs, jn, _deg(), 1.5, False, _elig(jobs), END, "x")
    assert ch["row_id"].tolist() == [1]  # only row 1 starts after an onset
    r = ch.iloc[0]
    assert r["factor_applied"] == 1.5
    assert r["run_time_s"] == 3900.0  # 5400 > limit 3900 -> max(T, limit)
    assert r["censored"] and r["censored_by_injection"] and r["state"] == "TIMEOUT"
    assert r["end_time"] == T0 + 10 * H + pd.Timedelta(seconds=3900)
    assert truth["node_id"].tolist() == [1, 2] and set(truth["kind"]) == {"node"}
    assert check_changes(jobs, jn, truth, ch, ELIG) == []


def test_node_partial_phi():
    jobs, jn = _jobs()
    ch, truth = node_scenario(jobs, jn, _deg(), 1.5, True, _elig(jobs), END, "x")
    got = dict(zip(ch["row_id"], ch["factor_applied"], strict=True))
    # row 1: earliest onset 9h <= start 10h -> phi 1; row 3: onset 10h, runs 5h-15h -> phi 0.5
    assert got == {1: pytest.approx(1.5), 3: pytest.approx(1.25)}
    assert ch.set_index("row_id").loc[3, "run_time_s"] == pytest.approx(36000 * 1.25)
    assert check_changes(jobs, jn, truth, ch, ELIG) == []


def test_decoy_and_shock():
    jobs, jn = _jobs()
    ch, truth = decoy_scenario(jobs, _elig(jobs), T0 + 10 * H, "a", 1.2, END, "d")
    assert ch["row_id"].tolist() == [1, 5]
    r5 = ch.set_index("row_id").loc[5]  # already TIMEOUT at limit: unchanged
    assert r5["run_time_s"] == 100.0 and r5["censored"] and not r5["censored_by_injection"]
    assert truth["user"].tolist() == ["a"] and truth["node_id"].isna().all()
    assert check_changes(jobs, jn, truth, ch, ELIG) == []
    ch2, truth2 = shock_scenario(jobs, _elig(jobs), T0 + 10.5 * H, 6, 1.3, "s")
    assert sorted(ch2["row_id"].tolist()) == [1, 3, 5]  # row 2 (FAILED) never affected
    assert check_changes(jobs, jn, truth2, ch2, ELIG) == []


def test_unaffected_rows_identical_and_contract_valid():
    jobs, jn = _jobs()
    ch, _ = node_scenario(jobs, jn, _deg(), 1.3, True, _elig(jobs), END, "x")
    out = materialize(jobs, ch)  # validates the contract
    keep = ~jobs["row_id"].isin(ch["row_id"])
    pd.testing.assert_frame_equal(out[keep], jobs[keep])
    assert (
        out.loc[~keep, "run_time_s"].to_numpy() >= jobs.loc[~keep, "run_time_s"].to_numpy()
    ).all()


def test_censoring_never_lowers_runtime():
    jobs, _ = _jobs()
    jobs = jobs.copy()
    jobs.loc[0, "walltime_req_s"] = 1000.0  # T = 3600 already above the limit
    ch = apply_multiplier(jobs, np.array([0]), np.array([1.5]))
    assert ch["run_time_s"].iloc[0] == 3600.0 and ch["censored"].iloc[0]


def test_window_draws_avoid_gaps_and_stay_in_range():
    w = Window(T0, T0 + 10 * H, [(T0 + 2 * H, T0 + 8 * H)])
    rng = np.random.default_rng(0)
    ts = [w.draw(rng, 100)[0] for _ in range(500)]
    assert all(T0 <= t <= T0 + 10 * H for t in ts)
    assert not any(T0 + 2 * H < t < T0 + 8 * H for t in ts)


def test_degraded_draw_deterministic_and_balanced():
    w = Window(T0, T0 + 10 * H)
    causes = ["cpu", "cache", "mbw"]
    a, _ = draw_degraded_nodes(np.random.default_rng(3), np.arange(100), 10, w, causes, 10)
    b, _ = draw_degraded_nodes(np.random.default_rng(3), np.arange(100), 10, w, causes, 10)
    pd.testing.assert_frame_equal(a, b)
    assert a["node_id"].is_unique and sorted(a["cause"].value_counts().tolist()) == [3, 3, 4]
    assert a["onset_time"].nunique() > 1  # independent onset per node


def test_decoy_draw_user_qualifies():
    jobs, _ = _jobs()
    main = _elig(jobs) & (jobs["run_time_s"] > 30).to_numpy()
    w = Window(T0, T0 + 9 * H)
    t0, user, _ = decoy_draw(np.random.default_rng(1), jobs, main, w, 2, 50)
    n = ((jobs["user"] == user) & (jobs["start_time"] >= t0) & main).sum()
    assert n >= 2


def test_random_placement():
    rng = np.random.default_rng(0)
    n_jobs = 300
    parts = rng.choice(["P", "Q"], n_jobs)
    jobs = pd.DataFrame({"row_id": np.arange(n_jobs), "partition": parts})
    pool = {"P": np.arange(0, 50), "Q": np.arange(50, 60)}
    rows = []
    for i, p in enumerate(parts):
        k = int(rng.integers(1, 5))
        rows += [(i, n) for n in rng.choice(pool[p], k, replace=False)]
    jn = pd.DataFrame(rows, columns=["row_id", "node_id"])
    a = random_placement(jn, jobs, 7)
    b = random_placement(jn, jobs, 7)
    c = random_placement(jn, jobs, 8)
    pd.testing.assert_frame_equal(a, b)
    assert not a["node_id"].equals(c["node_id"])
    assert (a.groupby("row_id").size() == jn.groupby("row_id").size()).all()
    assert (a.groupby("row_id")["node_id"].nunique() == a.groupby("row_id").size()).all()
    m = a.merge(jobs, on="row_id")
    assert m.apply(lambda r: r["node_id"] in pool[r["partition"]], axis=1).all()


def test_heavy_decoy_draw_top_users():
    from hpcdiag.inject.scenarios import heavy_decoy_draw

    rng = np.random.default_rng(0)
    n = 2000
    probs = np.r_[np.full(5, 0.1), np.full(25, 0.02)]
    users = rng.choice([f"u{i:02d}" for i in range(30)], n, p=probs)
    start = T0 + pd.to_timedelta(rng.integers(0, 100 * 3600, n), unit="s")
    jobs = pd.DataFrame({"user": users, "start_time": start})
    main = np.ones(n, bool)
    w = Window(T0, T0 + 10 * H)
    for seed in range(20):
        t0, user, rank, n_user, _ = heavy_decoy_draw(
            np.random.default_rng(seed), jobs, main, w, 3, 10
        )
        counts = jobs.loc[jobs["start_time"] >= t0, "user"].value_counts()
        top3 = counts.sort_values(ascending=False).iloc[:3]
        assert n_user == counts[user] and n_user >= top3.min() and 1 <= rank <= 3
