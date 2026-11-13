"""Measurement harness (D115). Standard library only. Runs a plan (from kit_campaign.py) and logs JSONL.

    python3 cluster/kit_harness.py --config cluster/campaign.json --plan plan.json --log logs/x.jsonl
    python3 cluster/kit_harness.py --config cluster/campaign.json --plan plan.json --dry-run

--dry-run prints every command and every safety verdict and executes nothing. Without it, every anomaly
passes the kit_safety gate first (x86-64 Linux, hostname in approved_hosts.txt, allowed anomaly, no
UNKNOWN placeholder, memleak cap < 25% of MemTotal); it runs in its own process group with a
watchdog that kills the group at its hard end time and, for memleak, when the group's resident memory
passes the cap. Every log line has UTC start/stop timestamps.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kit_probes as KP  # noqa: E402
import kit_safety as KS  # noqa: E402

HERE = Path(__file__).resolve().parent


def utc(t: float) -> str:
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat(timespec="milliseconds")  # noqa: UP017 (py<3.11 targets)


def group_rss_bytes(pgid: int) -> int:
    """Resident memory of every process in process group pgid (Linux /proc)."""
    total = 0
    for d in Path("/proc").glob("[0-9]*"):
        try:
            stat = (d / "stat").read_text()
            if int(stat.rsplit(")", 1)[1].split()[2]) != pgid:  # field 5: pgrp
                continue
            for line in (d / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            continue
    return total


class Watchdog(threading.Thread):
    """Kills the process group of proc at hard_end (time.monotonic) or when its RSS passes cap."""

    def __init__(self, proc, hard_end: float, cap_bytes: int | None = None, poll_s: float = 0.2,
                 grace_s: float = 3.0, rss=group_rss_bytes):  # fmt: skip
        super().__init__(daemon=True)
        self.proc, self.hard_end, self.cap, self.poll, self.grace = (
            proc,
            hard_end,
            cap_bytes,
            poll_s,
            grace_s,
        )
        self.rss = rss
        self.reason: str | None = None
        self._halt = threading.Event()

    def kill(self, reason: str) -> None:
        if self.reason is None:
            self.reason = reason
        stop_group(self.proc, self.grace)

    def run(self) -> None:
        while not self._halt.is_set() and self.proc.poll() is None:
            if time.monotonic() >= self.hard_end:
                self.kill("hard_end")
                return
            if self.cap is not None and self.rss(self.proc.pid) > self.cap:
                self.kill("rss_cap")
                return
            self._halt.wait(self.poll)

    def stop(self) -> None:
        self._halt.set()


def stop_group(proc, grace_s: float = 3.0) -> None:
    """SIGTERM the process group, then SIGKILL after grace_s."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    except ProcessLookupError:
        pass


class Harness:
    def __init__(self, cfg: dict, log_path: str | None, dry_run: bool = False, facts=None,
                 approved=None, run_cmd=None, read=KP._read, sleep=time.sleep, out=print):  # fmt: skip
        self.cfg, self.dry, self.out = cfg, dry_run, out
        self.facts = facts or KS.host_facts()
        self.approved = (
            approved if approved is not None else KS.approved_hosts(HERE / "approved_hosts.txt")
        )
        self.run_cmd, self.read, self.sleep = run_cmd, read, sleep
        self.log_path = Path(log_path) if log_path else None
        self.session = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")  # noqa: UP017

    # ------------------------------------------------------------------ logging
    def log(self, ev: dict) -> None:
        ev = {"session": self.session, "host": self.facts["hostname"], **ev}
        for k in ("t_start", "t_stop"):
            if ev.get(k) is not None:
                ev[f"{k}_utc"] = utc(ev[k])
        if self.dry:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as fh:
            fh.write(json.dumps(ev) + "\n")

    # ------------------------------------------------------------------ commands
    def anomaly_command(self, a: str, intensity: str, duration_s: float) -> tuple[str, int | None]:
        spec = self.cfg["anomalies"][a]
        req = None
        fmt = {"hpas_dir": self.cfg["hpas_dir"], "duration_s": int(duration_s),
               "intensity_arg": spec["intensity_args"][intensity]}  # fmt: skip
        if a == "memleak":
            mem_total = KS.mem_total_bytes(self.read("/proc/meminfo")) if not self.dry else 0
            req = int(float(spec["fractions"][intensity]) * mem_total)
            fmt["bytes"] = req
        return spec["template"].format(**fmt), req

    def check_probes(self, probes) -> None:
        for name in probes:
            c = KP.probe_command(name, self.cfg)
            if c is not None:
                KS.check_template(" ".join(c))

    # ------------------------------------------------------------------ steps
    def battery(self, probes, context: dict) -> None:
        for name in probes:
            if self.dry:
                c = KP.probe_command(name, self.cfg)
                self.out(f"  probe {name}: {' '.join(c) if c else 'read /proc (in process)'}")
                continue
            rec = KP.run_probe(
                name, self.cfg, run_cmd=self.run_cmd, read=self.read, sleep=self.sleep
            )
            self.log({"event": "probe", **context, **rec})

    def start_anomaly(self, step: dict):
        a, i = step["anomaly"], step["intensity"]
        cfg = self.cfg
        dur = float(cfg["anomaly_duration_s"])
        cmd, req = self.anomaly_command(a, i, dur)
        if self.dry:
            verdict = "allowed"
            try:
                KS.check_anomaly(a)
                KS.check_host(self.facts, self.approved)
                KS.check_template(cmd)
            except KS.Refused as e:
                verdict = f"REFUSED here: {e}"
            self.out(
                f"  anomaly {a}/{i} for {dur:.0f} s (hard end +{cfg['grace_s']} s): {cmd}  [{verdict}]"
            )
            return None, None
        KS.check_anomaly(a)
        KS.check_host(self.facts, self.approved)
        KS.check_template(cmd)
        cap = None
        if a == "memleak":
            cap = KS.check_memleak(req, KS.mem_total_bytes(self.read("/proc/meminfo")),
                                   float(cfg["memleak_cap_fraction"]))  # fmt: skip
        t0 = time.time()
        proc = subprocess.Popen(shlex.split(cmd), start_new_session=True,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # fmt: skip
        wd = Watchdog(proc, time.monotonic() + dur + float(cfg["grace_s"]), cap)
        wd.start()
        self.log({"event": "anomaly_start", "anomaly": a, "intensity": i, "rep": step.get("rep"),
                  "command": cmd, "pid": proc.pid, "t_start": t0, "cap_bytes": cap,
                  "hard_end_s": dur + float(cfg["grace_s"])})  # fmt: skip
        return proc, wd

    def stop_anomaly(self, step: dict, proc, wd) -> None:
        if proc is None:
            return
        wd.stop()
        stop_group(proc)
        self.log({"event": "anomaly_stop", "anomaly": step["anomaly"], "intensity": step["intensity"],
                  "rep": step.get("rep"), "t_stop": time.time(), "reason": wd.reason or "stopped",
                  "returncode": proc.returncode})  # fmt: skip

    def step(self, s: dict) -> None:
        cfg = self.cfg
        ctx = {k: s.get(k) for k in ("condition", "anomaly", "intensity", "rep", "block")}
        if s["type"] == "battery":
            self.out(f"battery ({s.get('condition')}), probes {s['probes']}") if self.dry else None
            self.battery(s["probes"], ctx)
        elif s["type"] == "anomaly":
            self.out(
                f"anomaly step {s['anomaly']}/{s['intensity']} rep {s.get('rep')}"
            ) if self.dry else None
            proc, wd = self.start_anomaly(s)
            try:
                self.sleep(float(cfg["warmup_s"])) if not self.dry else None
                self.battery(s["probes"], ctx)
            finally:
                self.stop_anomaly(s, proc, wd)
            self.sleep(float(cfg["cooldown_s"])) if not self.dry else None
        elif s["type"] == "benchmark":
            KS.check_template(cfg["benchmark_cmd"]) if not self.dry else None
            proc = wd = None
            if s.get("anomaly"):
                proc, wd = self.start_anomaly(s)
            try:
                if self.dry:
                    self.out(
                        f"  benchmark ({s.get('anomaly') or 'healthy'}): {cfg['benchmark_cmd']}"
                    )
                else:
                    self.sleep(float(cfg["warmup_s"])) if proc else None
                    t0 = time.time()
                    subprocess.run(
                        shlex.split(cfg["benchmark_cmd"]), check=True, capture_output=True
                    )
                    self.log({"event": "benchmark", **ctx, "t_start": t0, "t_stop": time.time(),
                              "seconds": time.time() - t0})  # fmt: skip
            finally:
                if s.get("anomaly"):
                    self.stop_anomaly(s, proc, wd)
            self.sleep(float(cfg["cooldown_s"])) if not self.dry and s.get("anomaly") else None
        else:
            raise ValueError(f"unknown step type {s['type']!r}")

    def run_plan(self, plan: list[dict]) -> None:
        if not self.dry:
            for s in plan:
                self.check_probes(s.get("probes", []))
        for s in plan:
            self.step(s)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--log")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.dry_run and not a.log:
        ap.error("--log is required unless --dry-run")
    cfg = json.loads(Path(a.config).read_text())
    plan = json.loads(Path(a.plan).read_text())
    h = Harness(cfg, a.log, dry_run=a.dry_run)
    if not a.dry_run:
        KS.check_host(h.facts, h.approved)  # refuse the whole session on an unapproved host
    h.run_plan(plan)
    return 0


if __name__ == "__main__":
    sys.exit(main())
