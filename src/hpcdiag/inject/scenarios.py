"""Scenario generators (PLAN.md §9 Phase 3; D29–D35). Pure functions: inputs are never modified.

Every generator returns (changes, truth, info):
- changes: one row per affected job — row_id, run_time_s, end_time, state, censored,
  factor_applied (the multiplier applied before censoring), censored_by_injection;
- truth: the §3.1 truth table (extended: cause, user, scenario_id; node_id null for non-node kinds);
- info: draws and redraw counts.
Only rows whose state_raw is eligible (COMPLETED/TIMEOUT) can be affected (D34).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRUTH_COLUMNS = [
    "node_id",
    "onset_time",
    "end_time",
    "factor",
    "kind",
    "cause",
    "user",
    "scenario_id",
]
CHANGE_COLUMNS = [
    "row_id",
    "run_time_s",
    "end_time",
    "state",
    "censored",
    "factor_applied",
    "censored_by_injection",
]


@dataclass
class Window:
    """Allowed onset range [start, end] minus open submission gaps (a, b)."""

    start: pd.Timestamp
    end: pd.Timestamp
    gaps: list[tuple[pd.Timestamp, pd.Timestamp]] = field(default_factory=list)

    def in_gap(self, t: pd.Timestamp) -> bool:
        return any(a < t < b for a, b in self.gaps)

    def draw(self, rng: np.random.Generator, max_redraws: int) -> tuple[pd.Timestamp, int]:
        """Uniform time in [start, end], redrawn while inside a gap. Returns (t, redraws)."""
        lo, hi = self.start.value, self.end.value
        for r in range(max_redraws + 1):
            t = pd.Timestamp(int(rng.integers(lo, hi + 1)))
            if not self.in_gap(t):
                return t, r
        raise RuntimeError("no onset outside the gaps after max_redraws")


def empty_changes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            c: pd.Series(dtype=t)
            for c, t in zip(
                CHANGE_COLUMNS,
                ["int64", "float64", "datetime64[us]", "str", "bool", "float64", "bool"],
                strict=True,
            )
        }
    )


def empty_truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            c: pd.Series(dtype=t)
            for c, t in zip(
                TRUTH_COLUMNS,
                [
                    "Int64",
                    "datetime64[us]",
                    "datetime64[us]",
                    "float64",
                    "str",
                    "str",
                    "str",
                    "str",
                ],
                strict=True,
            )
        }
    )


def apply_multiplier(jobs: pd.DataFrame, row_ids: np.ndarray, mult: np.ndarray) -> pd.DataFrame:
    """Changes for the given rows: T' = mult * T; censoring at the time limit (D34):
    if mult * T > limit then T' = max(T, limit), censored = TRUE, state = TIMEOUT. Start fixed."""
    j = jobs.set_index("row_id").loc[row_ids]
    T = j["run_time_s"].to_numpy(float)
    limit = j["walltime_req_s"].to_numpy(float)
    raw = T * np.asarray(mult, float)
    cens = raw > limit
    Tn = np.where(cens, np.maximum(T, limit), raw)
    start = j["start_time"].to_numpy().astype("datetime64[us]")
    end = start + np.round(Tn * 1e6).astype("int64").astype("timedelta64[us]")
    return pd.DataFrame(
        {
            "row_id": np.asarray(row_ids, dtype="int64"),
            "run_time_s": Tn,
            "end_time": end,
            "state": np.where(cens, "TIMEOUT", j["state"].to_numpy(dtype=object)).astype(str),
            "censored": cens | j["censored"].to_numpy(bool),
            "factor_applied": np.asarray(mult, float),
            "censored_by_injection": cens & ~j["censored"].to_numpy(bool),
        }
    )


def balanced_causes(rng: np.random.Generator, causes: list[str], k: int) -> np.ndarray:
    """Equal mix (counts differ by at most 1), randomly assigned (D30)."""
    return rng.permutation(np.resize(np.asarray(causes, dtype=object), k)).astype(str)


def draw_degraded_nodes(
    rng: np.random.Generator,
    active_nodes: np.ndarray,
    k: int,
    window: Window,
    causes: list[str],
    max_redraws: int,
) -> tuple[pd.DataFrame, int]:
    """K distinct nodes, an independent onset per node (D34), balanced causes."""
    nodes = np.sort(rng.choice(np.asarray(active_nodes), size=k, replace=False))
    onsets, redraws = [], 0
    for _ in range(k):
        t, r = window.draw(rng, max_redraws)
        onsets.append(t)
        redraws += r
    return pd.DataFrame(
        {
            "node_id": nodes.astype("int64"),
            "onset_time": pd.to_datetime(onsets).astype("datetime64[us]"),
            "cause": balanced_causes(rng, causes, k),
        }
    ), redraws


def node_scenario(
    jobs: pd.DataFrame,
    job_nodes: pd.DataFrame,
    degraded: pd.DataFrame,
    s: float,
    partial: bool,
    eligible: np.ndarray,
    trace_end: pd.Timestamp,
    scenario_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """node: jobs on a degraded node starting at/after that node's onset get T' = s T.
    node_partial: T' = T (1 + (s - 1) phi), phi = fraction of the job's time after t*, where t* is
    the earliest onset among the job's degraded nodes that falls before the job's end."""
    ej = jobs.loc[eligible, ["row_id", "start_time", "end_time", "run_time_s"]]
    x = job_nodes[["row_id", "node_id"]].merge(degraded[["node_id", "onset_time"]], on="node_id")
    x = x.merge(ej, on="row_id")
    if partial:
        x = x[x["onset_time"] < x["end_time"]]
        g = x.groupby("row_id").agg(
            t=("onset_time", "min"),
            start=("start_time", "first"),
            end=("end_time", "first"),
            T=("run_time_s", "first"),
        )
        after = (g["end"] - g["t"]).dt.total_seconds().to_numpy()
        T = g["T"].to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            phi = np.where(
                T > 0, np.clip(after / T, 0.0, 1.0), (g["start"] >= g["t"]).to_numpy().astype(float)
            )
        mult = 1.0 + (s - 1.0) * phi
        rows = g.index.to_numpy()
    else:
        rows = np.sort(x.loc[x["start_time"] >= x["onset_time"], "row_id"].unique())
        mult = np.full(len(rows), s)
    changes = apply_multiplier(jobs, rows, mult) if len(rows) else empty_changes()
    truth = pd.DataFrame(
        {
            "node_id": degraded["node_id"].astype("Int64"),
            "onset_time": degraded["onset_time"],
            "end_time": pd.Series(trace_end, index=degraded.index).astype("datetime64[us]"),
            "factor": float(s),
            "kind": "node_partial" if partial else "node",
            "cause": degraded["cause"].astype(str),
            "user": pd.Series(None, index=degraded.index, dtype="str"),
            "scenario_id": scenario_id,
        }
    )[TRUTH_COLUMNS]
    return changes, truth


def decoy_draw(
    rng: np.random.Generator,
    jobs: pd.DataFrame,
    main_pop: np.ndarray,
    window: Window,
    min_jobs: int,
    max_redraws: int,
) -> tuple[pd.Timestamp, str, int]:
    """t0 from the window, then a user with >= min_jobs main-population jobs starting after t0;
    t0 is redrawn if no user qualifies (D34). Returns (t0, user, t0 redraws incl. gap redraws)."""
    mp = jobs.loc[main_pop, ["user", "start_time"]]
    redraws = 0
    for _ in range(max_redraws + 1):
        t0, r = window.draw(rng, max_redraws)
        redraws += r
        counts = mp.loc[mp["start_time"] >= t0, "user"].value_counts()
        ok = np.sort(counts[counts >= min_jobs].index.astype(str).to_numpy())
        if len(ok):
            return t0, str(rng.choice(ok)), redraws
        redraws += 1
    raise RuntimeError("no decoy user qualifies")


def heavy_decoy_draw(
    rng: np.random.Generator,
    jobs: pd.DataFrame,
    main_pop: np.ndarray,
    window: Window,
    top_n: int,
    max_redraws: int,
) -> tuple[pd.Timestamp, str, int, int, int]:
    """D36: t0 from the window, then a user drawn uniformly from the top_n users by number of
    main-population jobs starting at/after t0 (ties broken by user id). Returns
    (t0, user, rank 1..top_n, that user's main-population jobs after t0, gap redraws)."""
    mp = jobs.loc[main_pop, ["user", "start_time"]]
    t0, red = window.draw(rng, max_redraws)
    counts = mp.loc[mp["start_time"] >= t0, "user"].astype(str).value_counts()
    top = (
        counts.rename("n")
        .reset_index()
        .rename(columns={"index": "user"})
        .sort_values(["n", "user"], ascending=[False, True])
        .head(top_n)
    )
    if top.empty:
        raise RuntimeError("no main-population jobs after t0")
    i = int(rng.integers(len(top)))
    return t0, str(top["user"].iloc[i]), i + 1, int(top["n"].iloc[i]), red


def decoy_scenario(
    jobs: pd.DataFrame,
    eligible: np.ndarray,
    t0: pd.Timestamp,
    user: str,
    s: float,
    trace_end: pd.Timestamp,
    scenario_id: str,
    kind: str = "user_decoy",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = (
        eligible
        & (jobs["user"].astype(str).to_numpy() == user)
        & (jobs["start_time"] >= t0).to_numpy()
    )
    rows = jobs.loc[m, "row_id"].to_numpy()
    changes = apply_multiplier(jobs, rows, np.full(len(rows), s)) if len(rows) else empty_changes()
    truth = pd.DataFrame(
        {
            "node_id": pd.array([None], dtype="Int64"),
            "onset_time": pd.to_datetime([t0]).astype("datetime64[us]"),
            "end_time": pd.to_datetime([trace_end]).astype("datetime64[us]"),
            "factor": [float(s)],
            "kind": [kind],
            "cause": pd.array([None], dtype="str"),
            "user": [user],
            "scenario_id": [scenario_id],
        }
    )[TRUTH_COLUMNS]
    return changes, truth


def shock_scenario(
    jobs: pd.DataFrame,
    eligible: np.ndarray,
    t1: pd.Timestamp,
    hours: float,
    s: float,
    scenario_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Jobs whose ORIGINAL end_time is in [t1, t1 + hours] get T' = s T."""
    t2 = t1 + pd.Timedelta(hours=hours)
    e = jobs["end_time"]
    m = eligible & ((e >= t1) & (e <= t2)).to_numpy()
    rows = jobs.loc[m, "row_id"].to_numpy()
    changes = apply_multiplier(jobs, rows, np.full(len(rows), s)) if len(rows) else empty_changes()
    truth = pd.DataFrame(
        {
            "node_id": pd.array([None], dtype="Int64"),
            "onset_time": pd.to_datetime([t1]).astype("datetime64[us]"),
            "end_time": pd.to_datetime([t2]).astype("datetime64[us]"),
            "factor": [float(s)],
            "kind": ["system_shock"],
            "cause": pd.array([None], dtype="str"),
            "user": pd.array([None], dtype="str"),
            "scenario_id": [scenario_id],
        }
    )[TRUTH_COLUMNS]
    return changes, truth
