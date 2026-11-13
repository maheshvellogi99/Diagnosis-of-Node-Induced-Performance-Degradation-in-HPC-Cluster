"""Safety gate of the Phase 6 measurement kit (D115; docs/PHASE6_PROTOCOL.md §2). Standard library only.

An anomaly runs only if ALL hold: the host is x86-64 Linux; its hostname is listed in
cluster/approved_hosts.txt (empty in git); the anomaly is one of cpuoccupy, cachecopy, membw, memleak
(iometadata, iobandwidth and netoccupy are always refused); its command template has no UNKNOWN
placeholder; memleak asks for at most cap_fraction of MemTotal, with cap_fraction < 0.25.
"""

from __future__ import annotations

import platform
import socket
from pathlib import Path

ALLOWED_ANOMALIES = ("cpuoccupy", "cachecopy", "membw", "memleak")
FORBIDDEN_ANOMALIES = ("iometadata", "iobandwidth", "netoccupy")
MEMLEAK_MAX_FRACTION = 0.25  # strict upper bound on the configured cap (D115)
X86_64 = ("x86_64", "amd64")


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
