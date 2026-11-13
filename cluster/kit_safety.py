"""Safety gate of the Phase 6 measurement kit (D115; docs/PHASE6_PROTOCOL.md §2). Standard library only.

An anomaly runs only if ALL hold: the host is x86-64 Linux; its hostname is listed in
cluster/approved_hosts.txt (empty in git); the anomaly is one of cpuoccupy, cachecopy, membw, memleak
(iometadata, iobandwidth and netoccupy are always refused); its command template has no UNKNOWN
placeholder; memleak asks for at most cap_fraction of MemTotal, with cap_fraction < 0.25.
"""

from __future__ import annotations

import math
import platform
import socket
from pathlib import Path

ALLOWED_ANOMALIES = ("cpuoccupy", "cachecopy", "membw", "memleak")
FORBIDDEN_ANOMALIES = ("iometadata", "iobandwidth", "netoccupy")
MEMLEAK_MAX_FRACTION = 0.25  # strict upper bound on the configured cap (D115)
X86_64 = ("x86_64", "amd64")
FIT_MIN, TEST_MIN = 1000, 300  # D119(a): healthy runs per probe on fit days / test days
MIN_MACHINES, MIN_DAYS = 2, 3  # D114


class Refused(RuntimeError):
    """A safety rule forbids this action."""


def host_facts() -> dict:
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
    }


def approved_hosts(path: str | Path) -> set[str]:
    """Hostnames, one per line; blank lines and '#' comments ignored. Missing file: empty set."""
    p = Path(path)
    if not p.exists():
        return set()
    out = set()
    for line in p.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.add(line)
    return out


def check_host(facts: dict, approved: set[str]) -> None:
    if facts["system"] != "Linux":
        raise Refused(f"host OS is {facts['system']!r}; anomalies run on Linux only")
    if facts["machine"].lower() not in X86_64:
        raise Refused(f"host architecture is {facts['machine']!r}; anomalies need x86-64")
    names = {facts["hostname"], facts["hostname"].split(".")[0]}
    if not names & approved:
        raise Refused(f"host {facts['hostname']!r} is not in cluster/approved_hosts.txt")


def check_anomaly(name: str) -> None:
    if name in FORBIDDEN_ANOMALIES:
        raise Refused(f"{name} is never run (safety rules: filesystem/network outages)")
    if name not in ALLOWED_ANOMALIES:
        raise Refused(f"{name} is not an allowed anomaly {ALLOWED_ANOMALIES}")


def check_template(cmd: str) -> None:
    if "UNKNOWN" in cmd:
        raise Refused(f"command still has an UNKNOWN placeholder: {cmd!r}")


def campaign_size(machines: int, days: int, batteries_per_day: int) -> dict:
    """Healthy runs per probe on fit days (the first ceil(2D/3)) and on test days (D119a)."""
    n_fit = math.ceil(2 * days / 3)
    return {
        "fit_runs_per_probe": machines * n_fit * batteries_per_day,
        "test_runs_per_probe": machines * (days - n_fit) * batteries_per_day,
    }


def min_batteries_per_day(machines: int, days: int) -> int:
    n_fit = math.ceil(2 * days / 3)
    return max(
        math.ceil(FIT_MIN / (machines * n_fit)), math.ceil(TEST_MIN / (machines * (days - n_fit)))
    )


def check_campaign_size(cfg: dict) -> dict:
    """Refuse a campaign plan that cannot give >= 1000 fit-day and >= 300 test-day healthy runs per
    probe on >= 2 machines over >= 3 days (D114, D119a). Checked before the campaign starts."""
    m, d, b = (
        int(cfg["planned_machines"]),
        int(cfg["planned_days"]),
        int(cfg["healthy_batteries_per_day"]),
    )
    if m < MIN_MACHINES or d < MIN_DAYS:
        raise Refused(
            f"campaign needs >= {MIN_MACHINES} machines and >= {MIN_DAYS} days, plan has {m} and {d}"
        )
    sz = campaign_size(m, d, b)
    if sz["fit_runs_per_probe"] < FIT_MIN or sz["test_runs_per_probe"] < TEST_MIN:
        raise Refused(
            f"plan gives {sz['fit_runs_per_probe']} fit-day and {sz['test_runs_per_probe']} test-day healthy runs per "
            f"probe; needs >= {FIT_MIN} and >= {TEST_MIN} (at least {min_batteries_per_day(m, d)} batteries per day)"
        )
    return sz


def mem_total_bytes(meminfo_text: str) -> int:
    for line in meminfo_text.splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            return int(parts[1]) * 1024  # kB
    raise ValueError("MemTotal not found")


def check_memleak(requested_bytes: int, mem_total: int, cap_fraction: float) -> int:
    """Returns the cap in bytes (used by the RSS watchdog)."""
    if not 0 < cap_fraction < MEMLEAK_MAX_FRACTION:
        raise Refused(f"memleak cap fraction {cap_fraction} must be in (0, {MEMLEAK_MAX_FRACTION})")
    cap = int(cap_fraction * mem_total)
    if requested_bytes > cap:
        raise Refused(
            f"memleak asks for {requested_bytes} bytes, cap is {cap} ({cap_fraction:.0%})"
        )
    return cap
