"""Independent re-derivation of the injected set from the truth table alone (Gate 3).

Written as plain per-row loops, deliberately different from the vectorised generators, so that a
mistake in one is unlikely to be repeated in the other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def expected_multipliers(jobs: pd.DataFrame, job_nodes: pd.DataFrame, truth: pd.DataFrame,
                         eligible_states: list[str]) -> dict[int, float]:
    """{row_id: multiplier before censoring} implied by `truth` for the eligible jobs."""
    if truth.empty:
        return {}
    kind = truth["kind"].iloc[0]
    s = float(truth["factor"].iloc[0])
    j = jobs[jobs["state_raw"].isin(eligible_states)].set_index("row_id")
    out: dict[int, float] = {}
    if kind in ("node", "node_partial"):
        onset = dict(zip(truth["node_id"].astype(int), truth["onset_time"], strict=True))
        nodes_of: dict[int, list[int]] = {}
        for rid, nid in zip(job_nodes["row_id"], job_nodes["node_id"], strict=True):
            if int(nid) in onset and rid in j.index:
                nodes_of.setdefault(int(rid), []).append(int(nid))
        for rid, nodes in nodes_of.items():
            r = j.loc[rid]
            if kind == "node":
                if any(r["start_time"] >= onset[n] for n in nodes):
                    out[rid] = s
            else:
                ts = [onset[n] for n in nodes if onset[n] < r["end_time"]]
                if not ts:
                    continue
                t = min(ts)
                T = float(r["run_time_s"])
                if T > 0:
                    phi = min(1.0, max(0.0, (r["end_time"] - t).total_seconds() / T))
                else:
                    phi = 1.0 if r["start_time"] >= t else 0.0
                out[rid] = 1.0 + (s - 1.0) * phi
    elif kind in ("user_decoy", "user_decoy_heavy"):
        u = truth["user"].iloc[0]
        t0 = np.datetime64(truth["onset_time"].iloc[0], "us")
        for rid, usr, st in zip(j.index, j["user"].astype(str).tolist(),
                                j["start_time"].to_numpy().astype("datetime64[us]"), strict=True):
            if usr == u and st >= t0:
                out[int(rid)] = s
    elif kind == "system_shock":
        t1 = np.datetime64(truth["onset_time"].iloc[0], "us")
        t2 = np.datetime64(truth["end_time"].iloc[0], "us")
        for rid, en in zip(j.index, j["end_time"].to_numpy().astype("datetime64[us]"), strict=True):
            if t1 <= en <= t2:
                out[int(rid)] = s
    else:
        raise ValueError(kind)
    return out


def check_changes(jobs: pd.DataFrame, job_nodes: pd.DataFrame, truth: pd.DataFrame,
                  changes: pd.DataFrame, eligible_states: list[str]) -> list[str]:
    """Problems (empty list = exact match) between `changes` and what `truth` implies."""
    exp = expected_multipliers(jobs, job_nodes, truth, eligible_states)
    got = dict(zip(changes["row_id"].astype(int), changes["factor_applied"], strict=True))
    problems = []
    if set(exp) != set(got):
        problems.append(f"affected sets differ: {len(set(exp) - set(got))} missing, "
                        f"{len(set(got) - set(exp))} extra")
    common = sorted(set(exp) & set(got))
    if common:
        d = np.abs(np.array([exp[r] for r in common]) - np.array([got[r] for r in common]))
        if d.max() > 1e-12:
            problems.append(f"multipliers differ (max {d.max():.3g})")
    return problems
