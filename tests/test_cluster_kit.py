"""D115 measurement kit (cluster/). Safe on the development Mac: no HPAS, no stress program, no probe
binary is ever run; the only process started is a benign sleep that the watchdog must kill."""

from __future__ import annotations

import json
import subprocess
import sys
import time

import pytest

from hpcdiag.utils.run import PROJECT_ROOT

KIT = PROJECT_ROOT / "cluster"
sys.path.insert(0, str(KIT))

import kit_campaign as KC  # noqa: E402
import kit_harness as KH  # noqa: E402
import kit_probes as KP  # noqa: E402
import kit_safety as KS  # noqa: E402

CFG = json.loads((KIT / "campaign.json").read_text())
LINUX = {"system": "Linux", "machine": "x86_64", "hostname": "node7.lab.example"}

# ------------------------------------------------------------------ safety gate


def test_approved_hosts_is_empty_in_git():
    assert (KIT / "approved_hosts.txt").read_text() == ""
    assert KS.approved_hosts(KIT / "approved_hosts.txt") == set()


def test_approved_hosts_parsing(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("# lab machines\nnode7\n\n  node8  # second\n")
    assert KS.approved_hosts(p) == {"node7", "node8"}
    assert KS.approved_hosts(tmp_path / "missing.txt") == set()


def test_this_mac_is_refused():
    with pytest.raises(KS.Refused):
        KS.check_host(KS.host_facts(), {KS.host_facts()["hostname"]})  # even if listed


@pytest.mark.parametrize(
    "facts",
    [
        {"system": "Darwin", "machine": "arm64", "hostname": "node7"},
        {"system": "Linux", "machine": "aarch64", "hostname": "node7"},
        {"system": "Darwin", "machine": "x86_64", "hostname": "node7"},
        {"system": "Linux", "machine": "x86_64", "hostname": "node9"},
    ],
)
def test_host_refusals(facts):
    with pytest.raises(KS.Refused):
        KS.check_host(facts, {"node7"})


def test_host_accepted_by_full_or_short_name():
    KS.check_host(LINUX, {"node7"})
    KS.check_host(LINUX, {"node7.lab.example"})


@pytest.mark.parametrize("a", ["iometadata", "iobandwidth", "netoccupy", "rm_rf", ""])
def test_forbidden_and_unknown_anomalies(a):
    with pytest.raises(KS.Refused):
        KS.check_anomaly(a)


@pytest.mark.parametrize("a", ["cpuoccupy", "cachecopy", "membw", "memleak"])
def test_allowed_anomalies(a):
    KS.check_anomaly(a)


def test_unknown_template_refused():
    with pytest.raises(KS.Refused):
        KS.check_template("UNKNOWN - from the README")
    KS.check_template("/opt/hpas/cpuoccupy -d 60")


def test_memleak_cap():
    total = 64 * 2**30
    assert KS.check_memleak(int(0.1 * total), total, 0.20) == int(0.20 * total)
    for frac in (0.25, 0.3, 0.0, -1):
        with pytest.raises(KS.Refused):
            KS.check_memleak(1, total, frac)
    with pytest.raises(KS.Refused):
        KS.check_memleak(int(0.21 * total), total, 0.20)


def test_config_respects_the_caps():
    assert 0 < CFG["memleak_cap_fraction"] < KS.MEMLEAK_MAX_FRACTION
    assert all(
        v <= CFG["memleak_cap_fraction"] for v in CFG["anomalies"]["memleak"]["fractions"].values()
    )
    assert set(CFG["anomaly_names"]) == set(KS.ALLOWED_ANOMALIES)
    assert CFG["anomaly_duration_s"] > CFG["warmup_s"]


def test_mem_total():
    assert KS.mem_total_bytes("MemTotal:  1000 kB\nMemFree: 5 kB\n") == 1000 * 1024


# ------------------------------------------------------------------ probes (parsers, injected I/O)

STREAM_OUT = """-------------------------------------------------------------
Function    Best Rate MB/s  Avg time     Min time     Max time
Copy:           12000.5     0.013500     0.013333     0.013900
Scale:          11800.0     0.013700     0.013559     0.014000
Add:            13000.2     0.018700     0.018461     0.019000
Triad:          12950.0     0.018800     0.018533     0.019100
-------------------------------------------------------------"""
MEMINFO = "MemTotal:       65536000 kB\nMemFree:  1000 kB\nMemAvailable:   32768000 kB\n"


def test_parsers():
    assert KP.parse_stream_triad(STREAM_OUT) == 12950.0
    assert KP.parse_last_number("checksum 12 0.5\nseconds 20.125\n") == 20.125
    assert KP.mem_value(MEMINFO) == 2.0
    with pytest.raises(ValueError):
        KP.parse_stream_triad("no triad here")


def test_run_probe_with_injected_io():
    cfg = {**CFG, "probe_params": {**CFG["probe_params"], "cpu_work_iterations": 10,
                                   "pchase_bytes": 2**26, "pchase_steps": 10}}  # fmt: skip
    outs = {"cpu_work": "seconds 20.5", "pchase": "ns_per_access 81.25", "stream": STREAM_OUT}
    calls = []

    def run_cmd(c):
        calls.append(c)
        return outs[c[0].rsplit("/", 1)[1]]

    read = lambda p: MEMINFO if p.endswith("meminfo") else "x 1\n"  # noqa: E731
    clock = iter(range(100)).__next__
    assert KP.run_probe("cpu", cfg, run_cmd, read, clock=clock)["value"] == 20.5
    assert KP.run_probe("cache", cfg, run_cmd, read, clock=clock)["value"] == 81.25
    assert KP.run_probe("mbw", cfg, run_cmd, read, clock=clock)["value"] == pytest.approx(
        1 / 12950.0
    )
    assert KP.run_probe("mem", cfg, run_cmd, read, clock=clock)["value"] == 2.0
    slept = []
    tel = KP.run_probe("tel", cfg, run_cmd, read, sleep=slept.append, clock=clock)
    assert tel["value"] is None and len(tel["snapshots"]) == 2 and slept == [60.0]
    assert tel["snapshot_t"][1] > tel["snapshot_t"][0]
    assert tel["t_stop"] > tel["t_start"]
    assert calls[0] == ["cluster/bin/cpu_work", "10"]


# ------------------------------------------------------------------ plans


def test_injection_plan_counts_and_randomised():
    p = KC.injection_plan(CFG["anomaly_names"], CFG["intensities"], 4, CFG["probes"], seed=1)
    assert len(p) == 4 * 2 * 4
    for a in CFG["anomaly_names"]:
        for i in CFG["intensities"]:
            assert sum(s["anomaly"] == a and s["intensity"] == i for s in p) == 4
    assert all(sorted(s["probes"]) == sorted(CFG["probes"]) for s in p)
    assert p == KC.injection_plan(
        CFG["anomaly_names"], CFG["intensities"], 4, CFG["probes"], seed=1
    )
    keyed = [(s["anomaly"], s["intensity"], s["rep"]) for s in p]
    assert keyed != sorted(keyed)  # not in factorial order


def test_benchmark_plan_pairs():
    p = KC.benchmark_plan(CFG["anomaly_names"], CFG["intensities"], 3, seed=2)
    assert len(p) == 2 * 4 * 2 * 3
    assert sum(s["anomaly"] is None for s in p) == len(p) // 2


def test_healthy_plan():
    p = KC.healthy_plan(5, CFG["probes"], seed=3)
    assert len(p) == 5 and all(s["condition"] == "healthy" for s in p)


# ------------------------------------------------------------------ harness


def _no_exec(*a, **k):
    raise AssertionError("nothing may be executed here")


def test_dry_run_executes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", _no_exec)
    monkeypatch.setattr(subprocess, "run", _no_exec)
    lines = []
    plan = KC.injection_plan(CFG["anomaly_names"], CFG["intensities"], 1, CFG["probes"], 1)
    plan += KC.healthy_plan(1, CFG["probes"], 2) + KC.benchmark_plan(["membw"], ["low"], 1, 3)
    h = KH.Harness(CFG, str(tmp_path / "log.jsonl"), dry_run=True, run_cmd=_no_exec, read=_no_exec,
                   sleep=_no_exec, out=lines.append)  # fmt: skip
    h.run_plan(plan)
    text = "\n".join(lines)
    assert "REFUSED here" in text and "anomaly memleak" in text and "benchmark" in text
    assert not (tmp_path / "log.jsonl").exists()


def test_real_mode_refuses_on_this_mac(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", _no_exec)
    h = KH.Harness(CFG, str(tmp_path / "log.jsonl"), approved=set())
    step = {"type": "anomaly", "anomaly": "membw", "intensity": "low", "rep": 0, "probes": []}
    with pytest.raises(KS.Refused):
        h.start_anomaly(step)


def test_real_mode_refuses_unknown_template_on_approved_linux(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", _no_exec)
    h = KH.Harness(
        CFG, str(tmp_path / "log.jsonl"), facts=LINUX, approved={"node7"}, read=lambda p: MEMINFO
    )
    for a in CFG["anomaly_names"]:
        with pytest.raises(KS.Refused):
            h.start_anomaly({"anomaly": a, "intensity": "high", "rep": 0})


def test_real_mode_refuses_forbidden_anomaly_even_if_configured(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", _no_exec)
    cfg = {
        **CFG,
        "anomalies": {
            **CFG["anomalies"],
            "iometadata": {"template": "echo x", "intensity_args": {"low": ""}},
        },
    }
    h = KH.Harness(cfg, str(tmp_path / "log.jsonl"), facts=LINUX, approved={"node7"})
    with pytest.raises(KS.Refused):
        h.start_anomaly({"anomaly": "iometadata", "intensity": "low", "rep": 0})


def _sleeper():
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )


def test_watchdog_kills_at_hard_end():
    p = _sleeper()
    t0 = time.monotonic()
    wd = KH.Watchdog(p, time.monotonic() + 0.5, poll_s=0.05, rss=lambda pid: 0)
    wd.start()
    wd.join(10)
    assert p.poll() is not None and wd.reason == "hard_end" and time.monotonic() - t0 < 8


def test_watchdog_kills_on_memory_cap():
    p = _sleeper()
    wd = KH.Watchdog(p, time.monotonic() + 30, cap_bytes=100, poll_s=0.05, rss=lambda pid: 101)
    wd.start()
    wd.join(10)
    assert p.poll() is not None and wd.reason == "rss_cap"


def test_cli_refuses_session_on_this_mac(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(KC.healthy_plan(1, ["mem"], 1)))
    log = tmp_path / "log.jsonl"
    r = subprocess.run([sys.executable, str(KIT / "kit_harness.py"), "--config", str(KIT / "campaign.json"),
                        "--plan", str(plan), "--log", str(log)], capture_output=True, text=True)  # fmt: skip
    assert r.returncode != 0 and "Refused" in r.stderr and not log.exists()


def test_cli_dry_run(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(KC.injection_plan(["cpuoccupy"], ["low"], 1, ["cpu"], 1)))
    r = subprocess.run([sys.executable, str(KIT / "kit_harness.py"), "--config", str(KIT / "campaign.json"),
                        "--plan", str(plan), "--dry-run"], capture_output=True, text=True)  # fmt: skip
    assert r.returncode == 0 and "cpuoccupy" in r.stdout and "REFUSED here" in r.stdout


# ------------------------------------------------------------------ D119(a): campaign size before the campaign


@pytest.mark.parametrize(
    "m, d, b, ok, fit, test",
    [
        (2, 3, 250, True, 1000, 500),
        (2, 3, 170, False, 680, 340),  # the D114 plan: too few fit-day runs
        (2, 3, 249, False, 996, 498),
        (3, 3, 167, True, 1002, 501),
        (4, 3, 125, True, 1000, 500),
        (2, 6, 125, True, 1000, 500),
        (4, 3, 74, False, 592, 296),
    ],
)
def test_campaign_size(m, d, b, ok, fit, test):
    assert KS.campaign_size(m, d, b) == {"fit_runs_per_probe": fit, "test_runs_per_probe": test}
    cfg = {**CFG, "planned_machines": m, "planned_days": d, "healthy_batteries_per_day": b}
    if ok:
        assert KS.check_campaign_size(cfg)["fit_runs_per_probe"] == fit
    else:
        with pytest.raises(KS.Refused):
            KS.check_campaign_size(cfg)


@pytest.mark.parametrize("m, d", [(1, 3), (2, 2)])
def test_campaign_needs_two_machines_and_three_days(m, d):
    cfg = {**CFG, "planned_machines": m, "planned_days": d, "healthy_batteries_per_day": 10_000}
    with pytest.raises(KS.Refused):
        KS.check_campaign_size(cfg)


@pytest.mark.parametrize(
    "m, d, b", [(2, 3, 250), (3, 3, 167), (4, 3, 125), (2, 6, 125), (3, 6, 84), (4, 6, 63)]
)
def test_min_batteries_per_day(m, d, b):
    assert KS.min_batteries_per_day(m, d) == b
    assert KS.campaign_size(m, d, b)["fit_runs_per_probe"] >= 1000
    assert (
        KS.campaign_size(m, d, b - 1)["fit_runs_per_probe"] < 1000
        or KS.campaign_size(m, d, b - 1)["test_runs_per_probe"] < 300
    )


def test_shipped_config_passes_the_size_check():
    sz = KS.check_campaign_size(CFG)
    assert sz["fit_runs_per_probe"] >= 1000 and sz["test_runs_per_probe"] >= 300


def test_dry_run_prints_the_size_check(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(KC.healthy_plan(1, ["mem"], 1)))
    r = subprocess.run([sys.executable, str(KIT / "kit_harness.py"), "--config", str(KIT / "campaign.json"),
                        "--plan", str(plan), "--dry-run"], capture_output=True, text=True)  # fmt: skip
    assert r.returncode == 0 and "campaign size" in r.stdout and "fit-day" in r.stdout


def test_cli_refuses_an_undersized_campaign_before_anything(tmp_path, monkeypatch):
    cfg = {**CFG, "healthy_batteries_per_day": 170}
    h = KH.Harness(cfg, str(tmp_path / "log.jsonl"), facts=LINUX, approved={"node7"})
    with pytest.raises(KS.Refused):
        h.preflight()
    assert not (tmp_path / "log.jsonl").exists()
