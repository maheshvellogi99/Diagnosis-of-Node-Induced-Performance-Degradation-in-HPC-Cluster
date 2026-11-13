"""Phase 6 analysis: campaign logs (cluster/kit_harness.py JSONL) -> D113 inputs and Gate 6 numbers.

Procedure, fixed before any measurement (docs/PHASE6_PROTOCOL.md §5):
1. Days = UTC dates of probe/benchmark start times; at least 3. The first ceil(2D/3) days are
   the FIT days, the rest the TEST days. Nothing from a test day enters any fitted value.
2. log x = log(value / b), b = median healthy value of that probe on that host on fit days.
3. sigma per probe = SD (ddof 1) of healthy fit-day log x over all hosts; 95% CI by percentile
   bootstrap over runs (2000 resamples, seed 0).
4. cost_s per probe = median duration of healthy fit-day runs.
5. s per anomaly x intensity = median over hosts of (median benchmark time with the anomaly / median
   healthy benchmark time), fit days.
6. k_{cause,probe} = sum(L y) / sum(L^2) over fit-day anomaly runs of that cause on that probe,
   L = log s of the run's anomaly x intensity; not identifiable (None, frozen value kept)
   if every |L| < 0.01.
   memleak on the mem probe: mem_fixed_shift = mean fit-day log x (fixed, not scaled by s).
7. Healthy pool: healthy fit-day log x per probe with start times (D113 empirical pool).
8. Held-out (test days): each anomaly repetition's probe outcomes update the frozen-prior posterior
   over (cause, s) with the fitted tables (frozen s_grid, pi0_h0); top-1 hypothesis vs the true
   cause.
   Accuracy with a Wilson 95% interval; healthy test batteries: share called h0.
telemetry.counts is not produced: no telemetry classifier is fixed for Phase 6 (UNKNOWN — needs a
decision).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from hpcdiag.layer3 import bayes as BY
from hpcdiag.layer3 import probes as PR

CAUSE_OF = {"cpuoccupy": "cpu", "cachecopy": "cache", "membw": "mbw", "memleak": "mem"}
MEASURED = ("cpu", "mbw", "cache", "mem")  # probes with an x value; tel has none
L_MIN = 0.01


def load_logs(paths) -> pd.DataFrame:
    rows = []
    for p in paths:
        with open(p) as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    df = pd.DataFrame(rows)
    df["day"] = pd.to_datetime(df["t_start"], unit="s", utc=True).dt.date
    return df


def split_days(df: pd.DataFrame) -> tuple[list, list]:
    days = sorted(df["day"].dropna().unique())
    if len(days) < 3:
        raise ValueError(f"need at least 3 campaign days, got {len(days)}")
    n_fit = math.ceil(2 * len(days) / 3)
    return days[:n_fit], days[n_fit:]


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


@dataclass
class Fit:
    fit_days: list
    test_days: list
    sigma: dict
    sigma_ci: dict
    n_healthy: dict
    cost_s: dict
    s: dict  # (anomaly, intensity) -> s
    k: dict  # cause -> probe -> value or None
    mem_fixed_shift: float | None
    pool: pd.DataFrame
    baseline: dict = field(default_factory=dict)


def fit(df: pd.DataFrame, seed: int = 0, n_boot: int = 2000) -> Fit:
    fit_days, test_days = split_days(df)
    f = df[df["day"].isin(fit_days)]
    pr = f[(f["event"] == "probe")]
    healthy = pr[pr["condition"] == "healthy"]
    base = {
        (h, a): float(g["value"].median())
        for (h, a), g in healthy[healthy["probe"].isin(MEASURED)].groupby(["host", "probe"])
    }
    m = pr[pr["probe"].isin(MEASURED)].copy()
    m["log_x"] = [
        math.log(v / base[(h, a)])
        for v, h, a in zip(m["value"], m["host"], m["probe"], strict=True)
    ]
    hm = m[m["condition"] == "healthy"]
    rng = np.random.default_rng(seed)
    sigma, ci, n_h = {}, {}, {}
    for a in MEASURED:
        y = hm.loc[hm["probe"] == a, "log_x"].to_numpy()
        sigma[a] = float(np.std(y, ddof=1))
        bs = [np.std(rng.choice(y, len(y)), ddof=1) for _ in range(n_boot)]
        ci[a] = (float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)))
        n_h[a] = int(len(y))
    dur = healthy.assign(d=healthy["t_stop"] - healthy["t_start"])
    cost = {a: float(g["d"].median()) for a, g in dur.groupby("probe")}
    bench = f[f["event"] == "benchmark"]
    s = {}
    for (an, i), g in bench[bench["condition"] == "anomaly"].groupby(["anomaly", "intensity"]):
        ratios = []
        for h, gh in g.groupby("host"):
            ref = bench[(bench["condition"] == "healthy") & (bench["host"] == h)]["seconds"]
            ratios.append(float(gh["seconds"].median() / ref.median()))
        s[(an, i)] = float(np.median(ratios))
    an = m[m["condition"] == "anomaly"].copy()
    an["L"] = [math.log(s[(x, i)]) for x, i in zip(an["anomaly"], an["intensity"], strict=True)]
    k: dict = {c: {} for c in PR.CAUSES}
    shift = None
    for anomaly, cause in CAUSE_OF.items():
        g = an[an["anomaly"] == anomaly]
        for a in MEASURED:
            ga = g[g["probe"] == a]
            if cause == "mem" and a == "mem":
                shift = float(ga["log_x"].mean()) if len(ga) else None
                k[cause][a] = None
                continue
            L, y = ga["L"].to_numpy(), ga["log_x"].to_numpy()
            k[cause][a] = (
                None if (len(L) == 0 or np.all(np.abs(L) < L_MIN)) else float((L @ y) / (L @ L))
            )
    pool = (
        hm[["probe", "log_x", "t_start"]].sort_values(["probe", "t_start"]).reset_index(drop=True)
    )
    return Fit(fit_days, test_days, sigma, ci, n_h, cost, s, k, shift, pool, base)


def model_from_fit(ft: Fit, frozen_pc: dict) -> tuple[PR.ProbeModel, dict]:
    """Fitted ProbeModel; entries that could not be estimated keep the frozen value (listed)."""
    kept = {}
    k = {c: dict(v) for c, v in frozen_pc["k"].items()}
    for c, row in ft.k.items():
        for a, v in row.items():
            if v is None:
                if not (c == "mem" and a == "mem"):
                    kept[f"{c}.{a}"] = k[c][a]
            else:
                k[c][a] = v
    shift = (
        ft.mem_fixed_shift
        if ft.mem_fixed_shift is not None
        else float(frozen_pc["mem_fixed_shift"])
    )
    names = tuple(frozen_pc["names"])
    m = PR.ProbeModel(
        names=names,
        cost=tuple(ft.cost_s.get(a, float(frozen_pc["cost_s"][a])) for a in names),
        sigma=tuple(ft.sigma[a] for a in names),
        k=k,
        mem_shift=shift,
    )
    return m, kept


def held_out(df: pd.DataFrame, ft: Fit, m: PR.ProbeModel, s_grid, pi0_h0: float) -> dict:
    t = df[
        df["day"].isin(ft.test_days) & (df["event"] == "probe") & df["probe"].isin(MEASURED)
    ].copy()
    t["log_x"] = [
        math.log(v / ft.baseline[(h, a)])
        for v, h, a in zip(t["value"], t["host"], t["probe"], strict=True)
    ]
    means, hyp = PR.state_table(m, s_grid)
    idx = {a: i for i, a in enumerate(m.names)}
    keys = ["host", "session", "condition", "anomaly", "intensity", "rep", "block"]
    t = t.fillna({"anomaly": "", "intensity": "", "rep": -1, "block": -1})
    res = {"anomaly": [0, 0], "healthy": [0, 0]}
    per_cause: dict = {}
    for kv, g in t.groupby(keys, dropna=False):
        pi = BY.prior(hyp, pi0_h0)
        for a, y in zip(g["probe"], g["log_x"], strict=True):
            i = idx[a]
            pi = BY.update_gauss(pi, means[:, i], PR.sig(m.sigma, i), float(y))
        top = PR.HYPOTHESES[int(np.argmax(BY.marginal(pi, hyp)))]
        cond, anomaly = kv[2], kv[3]
        truth = "h0" if cond == "healthy" else CAUSE_OF[anomaly]
        r = res["healthy" if cond == "healthy" else "anomaly"]
        r[0] += int(top == truth)
        r[1] += 1
        if cond != "healthy":
            pc = per_cause.setdefault(truth, [0, 0])
            pc[0] += int(top == truth)
            pc[1] += 1
    out = {}
    for lab, (kk, n) in res.items():
        out[lab] = {
            "correct": kk,
            "n": n,
            "share": kk / n if n else float("nan"),
            "ci95": wilson(kk, n),
        }
    out["per_cause"] = {c: {"correct": a, "n": b} for c, (a, b) in per_cause.items()}
    return out


def gate6(
    df: pd.DataFrame, ft: Fit, ho: dict, s_range: tuple[float, float], reps_min: int = 10
) -> dict:
    hosts = sorted(df["host"].unique())
    reps = (
        df[(df["event"] == "anomaly_start")].groupby(["anomaly", "intensity"]).size().to_dict()
        if "anomaly_start" in set(df["event"])
        else {}
    )
    return {
        "days": [str(d) for d in sorted(df["day"].dropna().unique())],
        "fit_days": [str(d) for d in ft.fit_days],
        "test_days": [str(d) for d in ft.test_days],
        "hosts": hosts,
        "sigma": ft.sigma,
        "sigma_ci95": ft.sigma_ci,
        "n_healthy_per_probe": ft.n_healthy,
        "held_out": ho,
        "s": {f"{a}/{i}": v for (a, i), v in ft.s.items()},
        "s_phase3_range": list(s_range),
        "s_inside_phase3_range": {
            f"{a}/{i}": s_range[0] <= v <= s_range[1] for (a, i), v in ft.s.items()
        },
        "completeness": {
            "healthy_runs_per_probe_ge_1000": all(n >= 1000 for n in ft.n_healthy.values()),
            "days_ge_3": len(df["day"].dropna().unique()) >= 3,
            "hosts_ge_2": len(hosts) >= 2,
            "reps_per_anomaly_intensity_ge_10": bool(reps)
            and all(v >= reps_min for v in reps.values()),
        },
        "telemetry_counts": (
            "not produced: UNKNOWN — needs a decision on the Phase 6 telemetry classifier"
        ),
    }


def d113_inputs(ft: Fit, m: PR.ProbeModel, kept: dict, pool_path: str) -> dict:
    return {
        "probes": {
            "k": m.k,
            "sigma": dict(zip(m.names, m.sigma, strict=True)),
            "cost_s": dict(zip(m.names, m.cost, strict=True)),
            "mem_fixed_shift": m.mem_shift,
            "healthy_pool": pool_path,
        },
        "_kept_frozen_k": kept,
    }


def write_pool(ft: Fit, path: Path) -> None:
    ft.pool.to_csv(path, index=False, float_format="%.17g")
