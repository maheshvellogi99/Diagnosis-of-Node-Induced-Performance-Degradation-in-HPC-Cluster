"""Phase 6 analysis on SYNTHETIC campaign logs with known parameters (D115). No real measurement."""

from __future__ import annotations

import copy
import json
import math

import numpy as np
import pandas as pd
import pytest

from hpcdiag.phase6 import analysis as AN
from hpcdiag.utils.run import load_config

CFG = load_config("configs/phase5.yaml")
SIG = {"cpu": 0.02, "mbw": 0.04, "cache": 0.06, "mem": 0.01}
S_TRUE = {("cpuoccupy", "low"): 1.3, ("cpuoccupy", "high"): 2.0, ("cachecopy", "low"): 1.2,
          ("cachecopy", "high"): 1.5, ("membw", "low"): 1.5, ("membw", "high"): 3.0,
          ("memleak", "low"): 1.0, ("memleak", "high"): 1.0}  # fmt: skip
BASE = {"hA": {"cpu": 20.0, "mbw": 1e-4, "cache": 80.0, "mem": 1.5},
        "hB": {"cpu": 22.0, "mbw": 1.2e-4, "cache": 90.0, "mem": 1.4}}  # fmt: skip
COST = {"cpu": 20.0, "mbw": 21.0, "cache": 19.0, "mem": 1.0, "tel": 60.0}
DAY0 = 1_790_000_000.0  # an epoch second; days are 86400 s apart


def mean_log_x(anomaly, intensity, probe):
    c = AN.CAUSE_OF[anomaly]
    if c == "mem" and probe == "mem":
        return float(CFG["probes"]["mem_fixed_shift"])
    return CFG["probes"]["k"][c][probe] * math.log(S_TRUE[(anomaly, intensity)])


def synth(seed=0, n_healthy=200, reps=4, days=3, test_shift=0.0):
    """Synthetic JSONL rows; test_shift adds a large offset to every test-day value (leak check)."""
    rng = np.random.default_rng(seed)
    rows = []
    n_fit = math.ceil(2 * days / 3)
    for d in range(days):
        off = test_shift if d >= n_fit else 0.0
        for h in BASE:
            t = DAY0 + d * 86400 + (1000 if h == "hB" else 0)
            for b in range(n_healthy):
                for a in ("cpu", "mbw", "cache", "mem", "tel"):
                    v = None if a == "tel" else BASE[h][a] * math.exp(rng.normal(0, SIG[a]) + off)
                    rows.append({"event": "probe", "session": f"s{d}", "host": h, "condition": "healthy",
                                 "block": b, "probe": a, "value": v, "t_start": t,
                                 "t_stop": t + COST[a] * (1 + 0.01 * rng.normal())})  # fmt: skip
                    t += COST[a]
            for (an, i), s in S_TRUE.items():
                for r in range(reps):
                    rows.append({"event": "anomaly_start", "session": f"s{d}", "host": h, "anomaly": an,
                                 "intensity": i, "rep": r, "t_start": t})  # fmt: skip
                    for a in ("cpu", "mbw", "cache", "mem"):
                        mu = mean_log_x(an, i, a)
                        v = BASE[h][a] * math.exp(mu + rng.normal(0, SIG[a]) + off)
                        rows.append({"event": "probe", "session": f"s{d}", "host": h, "condition": "anomaly",
                                     "anomaly": an, "intensity": i, "rep": r, "probe": a, "value": v,
                                     "t_start": t, "t_stop": t + COST[a]})  # fmt: skip
                        t += COST[a]
                    for cond, ss in (("anomaly", s), ("healthy", 1.0)):
                        sec = 60.0 * ss * math.exp(rng.normal(0, 0.01))
                        rows.append({"event": "benchmark", "session": f"s{d}", "host": h, "condition": cond,
                                     "anomaly": an if cond == "anomaly" else None, "intensity": i, "rep": r,
                                     "t_start": t, "t_stop": t + sec, "seconds": sec})  # fmt: skip
                        t += sec
    return rows


def frame(rows, tmp_path, name="log.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return AN.load_logs([p])


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    df = frame(synth(), tmp_path_factory.mktemp("a"))
    return df, AN.fit(df, n_boot=300)


def test_split_is_by_day(fitted):
    df, ft = fitted
    assert len(ft.fit_days) == 2 and len(ft.test_days) == 1
    assert max(ft.fit_days) < min(ft.test_days)


def test_sigma_recovered_with_ci(fitted):
    _, ft = fitted
    for a, s in SIG.items():
        assert ft.sigma[a] == pytest.approx(s, rel=0.1)
        lo, hi = ft.sigma_ci[a]
        assert lo < ft.sigma[a] < hi
        assert ft.n_healthy[a] == 2 * 2 * 200  # 2 hosts x 2 fit days


def test_costs_and_s_recovered(fitted):
    _, ft = fitted
    for a in ("cpu", "mbw", "cache", "mem", "tel"):
        assert ft.cost_s[a] == pytest.approx(COST[a], rel=0.02)
    for key, s in S_TRUE.items():
        assert ft.s[key] == pytest.approx(s, rel=0.02)


def test_k_and_mem_shift_recovered(fitted):
    _, ft = fitted
    k = CFG["probes"]["k"]
    for c in ("cpu", "cache", "mbw"):
        for a in ("cpu", "mbw", "cache", "mem"):
            assert ft.k[c][a] == pytest.approx(k[c][a], abs=0.05)
    assert ft.mem_fixed_shift == pytest.approx(CFG["probes"]["mem_fixed_shift"], abs=0.02)
    assert all(v is None for v in ft.k["mem"].values())  # memleak: s = 1, not identifiable


def test_test_days_never_enter_the_fit(tmp_path):
    a = AN.fit(frame(synth(seed=1), tmp_path, "a.jsonl"), n_boot=50)
    b = AN.fit(frame(synth(seed=1, test_shift=5.0), tmp_path, "b.jsonl"), n_boot=50)
    assert a.sigma == b.sigma and a.k == b.k and a.s == b.s and a.cost_s == b.cost_s
    assert a.pool.equals(b.pool)


def test_held_out_accuracy_and_gate6(fitted):
    df, ft = fitted
    m, kept = AN.model_from_fit(ft, CFG["probes"])
    assert set(kept) == {"mem.cpu", "mem.mbw", "mem.cache"}
    ho = AN.held_out(df, ft, m, CFG["policy"]["s_grid"], CFG["policy"]["pi0_h0"])
    assert (
        ho["anomaly"]["n"] == 2 * 8 * 4
    )  # 2 hosts x 8 anomaly-intensity cells x 4 reps, one test day
    assert ho["anomaly"]["share"] > 0.9
    assert ho["healthy"]["n"] == 2 * 200 and ho["healthy"]["share"] > 0.9
    g = AN.gate6(df, ft, ho, (1.1, 3.0))
    assert g["s_inside_phase3_range"]["memleak/low"] is False
    assert g["s_inside_phase3_range"]["membw/high"] is True
    assert g["completeness"]["hosts_ge_2"] and g["completeness"]["days_ge_3"]
    assert g["completeness"]["healthy_runs_per_probe_ge_1000"] is False  # 800 < 1000
    assert g["completeness"]["reps_per_anomaly_intensity_ge_10"] is True  # 2 x 3 x 4 = 24
    assert "UNKNOWN" in g["telemetry_counts"]


def test_outputs_feed_the_d113_plumbing(fitted, tmp_path):
    from hpcdiag.freeze import allowed_change, changed_keys
    from hpcdiag.layer3 import setup as ST

    _, ft = fitted
    m, kept = AN.model_from_fit(ft, CFG["probes"])
    AN.write_pool(ft, tmp_path / "pool.csv")
    d = AN.d113_inputs(ft, m, kept, str(tmp_path / "pool.csv"))
    cfg = copy.deepcopy(CFG)
    cfg["probes"].update(d["probes"])
    keys = changed_keys(CFG, cfg)
    assert keys and all(allowed_change(k) for k in keys)  # only D113 fields change
    cfg["evalue"]["n_pool"] = 300
    cfg["evalue"]["bet_pools"] = 20
    w = ST.make_world(cfg, ST.VARIANTS["main"], 1.3, np.zeros(3, int), np.full(3, 10**9), 1, 2)
    first = ft.pool[ft.pool["probe"] == "cpu"].sort_values("t_start")["log_x"].to_numpy()[:300]
    assert np.array_equal(w.pools[0], np.sort(first))
    assert np.allclose(w.sigma, [ft.sigma[a] for a in cfg["probes"]["names"]])


def test_fewer_than_three_days_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        AN.fit(frame(synth(days=2, n_healthy=20, reps=1), tmp_path), n_boot=10)


def test_wilson():
    lo, hi = AN.wilson(9, 10)
    assert 0.55 < lo < 0.6 and 0.98 < hi < 0.99
    assert all(np.isnan(AN.wilson(0, 0)))


def test_pool_csv_round_trip(fitted, tmp_path):
    _, ft = fitted
    AN.write_pool(ft, tmp_path / "p.csv")
    back = pd.read_csv(tmp_path / "p.csv", float_precision="round_trip")
    assert np.array_equal(back["log_x"].to_numpy(), ft.pool["log_x"].to_numpy())


def test_cli_on_synthetic_logs(tmp_path):
    import subprocess

    from hpcdiag.utils.run import PROJECT_ROOT

    log = tmp_path / "synthetic.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in synth(n_healthy=30, reps=2)) + "\n")
    r = subprocess.run(["uv", "run", "python", "scripts/phase6_analyze.py", "--config", "configs/phase5.yaml",
                        "--logs", str(log), "--out", str(tmp_path / "out")],
                       cwd=PROJECT_ROOT, capture_output=True, text=True)  # fmt: skip
    assert r.returncode == 0, r.stderr
    d = __import__("yaml").safe_load((tmp_path / "out" / "d113_inputs.yaml").read_text())
    assert set(d["probes"]) == {"k", "sigma", "cost_s", "mem_fixed_shift", "healthy_pool"}
    g = json.loads((tmp_path / "out" / "gate6.json").read_text())
    assert g["s_phase3_range"] == [1.1, 3.0]
