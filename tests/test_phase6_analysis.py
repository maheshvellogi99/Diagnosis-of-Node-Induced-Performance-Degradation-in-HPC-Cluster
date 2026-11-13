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


# D117 synthetic telemetry: per-second rates per class; hA also has a per-core line (dropped by intersection)
TEL_RATES = {
    "h0": {"user": 100, "ctxt": 2000, "avail": 0, "pgfault": 1000, "numa": 1e5},
    "cpu": {"user": 400, "ctxt": 2000, "avail": 0, "pgfault": 1000, "numa": 1e5},
    "cache": {"user": 150, "ctxt": 5000, "avail": 0, "pgfault": 1000, "numa": 1e5},
    "mbw": {"user": 150, "ctxt": 2000, "avail": 0, "pgfault": 1000, "numa": 1e6},
    "mem": {"user": 100, "ctxt": 2000, "avail": -1e5, "pgfault": 2e4, "numa": 1e5},
}


def tel_snaps(rng, cls, host, shift=0.0, dt=60.0):
    r = {k: v * (1 + 0.05 * rng.normal()) * (1 + shift) for k, v in TEL_RATES[cls].items()}
    a = {"user": 1e7, "ctxt": 5e8, "avail": 3e7, "pgfault": 1e9, "numa": 1e10}
    b = {k: a[k] + r[k] * dt for k in a}

    def snap(v):
        stat = f"cpu  {int(v['user'])} 0 500 900000\nctxt {int(v['ctxt'])}\n"
        if host == "hA":
            stat += f"cpu0 {int(v['user'])} 0 1 2\n"
        return {"stat": stat, "meminfo": f"MemTotal: 65536000 kB\nMemAvailable: {int(v['avail'])} kB\n",
                "vmstat": f"pgfault {int(v['pgfault'])}\nnuma_hit {int(v['numa'])}\n"}  # fmt: skip

    return [snap(a), snap(b)]


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
                    if a == "tel":
                        rows[-1]["snapshots"] = tel_snaps(rng, "h0", h, off)
                        rows[-1]["snapshot_t"] = [t, t + 60.0]
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
                    rows.append({"event": "probe", "session": f"s{d}", "host": h, "condition": "anomaly",
                                 "anomaly": an, "intensity": i, "rep": r, "probe": "tel", "value": None,
                                 "t_start": t, "t_stop": t + 60.0, "snapshot_t": [t, t + 60.0],
                                 "snapshots": tel_snaps(rng, AN.CAUSE_OF[an], h, off)})  # fmt: skip
                    t += 60.0
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
    assert g["completeness"]["fit_healthy_runs_per_probe_ge_1000"] is False  # 800 < 1000 (D119a)
    assert g["completeness"]["test_healthy_runs_per_probe_ge_300"] is True  # 400 >= 300
    assert g["completeness"]["reps_per_anomaly_intensity_ge_10"] is True  # 2 x 3 x 4 = 24
    assert g["telemetry"] is None  # not passed here; D117 tests below
    tel = AN.telemetry(df, ft, RF, seed=0)
    assert AN.gate6(df, ft, ho, (1.1, 3.0), tel=tel)["telemetry"]["replace"] is True


def test_outputs_feed_the_d113_plumbing(fitted, tmp_path):
    from hpcdiag.freeze import allowed_change, changed_keys
    from hpcdiag.layer3 import setup as ST

    df, ft = fitted
    m, kept = AN.model_from_fit(ft, CFG["probes"])
    AN.write_measured(df, ft, tmp_path / "pool.csv")
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
    mr = AN.measured_runs(df, ft)
    test_cpu = mr[(mr["probe"] == "cpu") & (mr["split"] == "test")].sort_values(
        "t_start", kind="stable"
    )
    assert np.array_equal(w.noise[0], test_cpu["log_x"].to_numpy())  # D119: outcomes from test days


def test_fewer_than_three_days_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        AN.fit(frame(synth(days=2, n_healthy=20, reps=1), tmp_path), n_boot=10)


def test_wilson():
    lo, hi = AN.wilson(9, 10)
    assert 0.55 < lo < 0.6 and 0.98 < hi < 0.99
    assert all(np.isnan(AN.wilson(0, 0)))


def test_pool_csv_round_trip(fitted, tmp_path):
    df, ft = fitted
    AN.write_measured(df, ft, tmp_path / "p.csv")
    back = pd.read_csv(tmp_path / "p.csv", float_precision="round_trip")
    assert np.array_equal(back["log_x"].to_numpy(), AN.measured_runs(df, ft)["log_x"].to_numpy())


def test_measured_runs_split_and_normalisation(fitted):
    """D119: fit rows = the healthy fit-day pool; test rows = test-day healthy runs, normalised by the
    FIT-day per-host median (no test-day information in the normalisation)."""
    df, ft = fitted
    mr = AN.measured_runs(df, ft)
    assert set(mr.columns) == {"probe", "log_x", "t_start", "host", "split"}
    for a in ("cpu", "mbw", "cache", "mem"):
        assert ((mr["probe"] == a) & (mr["split"] == "fit")).sum() == 800
        assert ((mr["probe"] == a) & (mr["split"] == "test")).sum() == 400
    raw = df[(df["event"] == "probe") & (df["condition"] == "healthy") & df["day"].isin(ft.test_days)
             & (df["probe"] == "cpu") & (df["host"] == "hB")].sort_values("t_start")  # fmt: skip
    got = mr[(mr["probe"] == "cpu") & (mr["split"] == "test") & (mr["host"] == "hB")].sort_values(
        "t_start"
    )
    assert np.allclose(
        got["log_x"].to_numpy(), np.log(raw["value"].to_numpy() / ft.baseline[("hB", "cpu")])
    )


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


# ------------------------------------------------------------------ D117 telemetry classifier

RF = CFG["antarex"]["rf"]


def test_tel_features_by_hand():
    a = {
        "stat": "cpu  100 0 50 1000\nctxt 600\n",
        "meminfo": "MemAvailable: 9000 kB\n",
        "vmstat": "pgfault 10\n",
    }
    b = {
        "stat": "cpu  160 0 50 1060\nctxt 1200\n",
        "meminfo": "MemAvailable: 8400 kB\n",
        "vmstat": "pgfault 70\n",
    }
    f = AN.tel_features([a, b], 60.0)
    assert f == {"stat.cpu.0": 1.0, "stat.cpu.1": 0.0, "stat.cpu.2": 0.0, "stat.cpu.3": 1.0,
                 "stat.ctxt.0": 10.0, "meminfo.MemAvailable": -10.0, "vmstat.pgfault": 1.0}  # fmt: skip


def test_tel_confusion_replaces_when_every_class_has_30(fitted):
    df, ft = fitted
    tel = AN.telemetry(df, ft, RF, seed=0)
    assert tel["labels"] == ["h0", "cpu", "cache", "mbw", "mem"]
    n = tel["n_per_class"]
    assert n == {"h0": 800, "cpu": 32, "cache": 32, "mbw": 32, "mem": 32}  # fit days only
    m = np.array(tel["matrix"])
    assert m.shape == (5, 5) and m.sum() == sum(n.values())
    assert np.trace(m) / m.sum() > 0.9
    assert tel["replace"] is True
    assert not any(f.startswith("stat.cpu0") for f in tel["features"])  # only on one host


def test_tel_test_days_never_used(tmp_path):
    a = AN.telemetry(*_df_fit(tmp_path, "a.jsonl", 0.0), RF, seed=0)
    b = AN.telemetry(*_df_fit(tmp_path, "b.jsonl", 3.0), RF, seed=0)
    assert a["matrix"] == b["matrix"] and a["n_per_class"] == b["n_per_class"]


def _df_fit(tmp_path, name, shift):
    df = frame(synth(seed=2, n_healthy=40, reps=4, test_shift=shift), tmp_path, name)
    return df, AN.fit(df, n_boot=20)


def test_tel_keeps_frozen_when_a_class_is_small(tmp_path):
    df = frame(synth(seed=3, n_healthy=40, reps=1), tmp_path)
    ft = AN.fit(df, n_boot=20)
    tel = AN.telemetry(df, ft, RF, seed=0)
    assert tel["replace"] is False and "fewer than 30" in tel["reason"]
    assert np.array(tel["matrix"]).sum() == sum(tel["n_per_class"].values())  # still reported


def test_d113_inputs_carry_telemetry_only_when_replaced(fitted, tmp_path):
    from hpcdiag.freeze import allowed_change, changed_keys

    df, ft = fitted
    m, kept = AN.model_from_fit(ft, CFG["probes"])
    tel = AN.telemetry(df, ft, RF, seed=0)
    d = AN.d113_inputs(ft, m, kept, "p.csv", tel)
    assert d["telemetry"]["crossblock_counts"] == tel["matrix"]
    new = copy.deepcopy(CFG)
    new["probes"].update(d["probes"])
    new["telemetry"].update(d["telemetry"])
    assert all(allowed_change(k) for k in changed_keys(CFG, new))
    assert "telemetry" not in AN.d113_inputs(ft, m, kept, "p.csv", {**tel, "replace": False})
