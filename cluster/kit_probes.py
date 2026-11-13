"""Probes of the measurement kit (D115). Standard library only. Larger x = worse for every probe.

cpu    seconds of a fixed amount of work (bin/cpu_work <iterations>)
mbw    seconds per MB at STREAM's Triad best rate (bin/stream), i.e. 1 / (MB/s)
cache  nanoseconds per access of a random pointer chase (bin/pchase <bytes> <steps>)
mem    MemTotal / MemAvailable from /proc/meminfo
tel    /proc/stat, /proc/meminfo and /proc/vmstat snapshots at the start and after tel_seconds
       (diagnosis only; no x value)
"""

from __future__ import annotations

import re
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

PROBES = ("cpu", "mbw", "cache", "mem", "tel")
_FLOAT = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def parse_last_number(text: str) -> float:
    nums = _FLOAT.findall(text)
    if not nums:
        raise ValueError(f"no number in probe output: {text!r}")
    return float(nums[-1])


def parse_stream_triad(text: str) -> float:
    """Best rate in MB/s from the 'Triad:' line of STREAM's output (first number on that line).
    Check against the real output of the pinned STREAM source before the healthy campaign."""
    for line in text.splitlines():
        if line.strip().startswith("Triad:"):
            return float(_FLOAT.findall(line)[0])
    raise ValueError("no Triad: line in STREAM output")


def parse_meminfo(text: str) -> dict[str, int]:
    out = {}
    for line in text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            nums = v.split()
            if nums and nums[0].isdigit():
                out[k.strip()] = int(nums[0])
    return out


def mem_value(meminfo_text: str) -> float:
    m = parse_meminfo(meminfo_text)
    return m["MemTotal"] / m["MemAvailable"]


def probe_command(name: str, cfg: dict) -> list[str] | None:
    """Command line of an external probe, or None for an in-process probe (mem, tel)."""
    b = cfg["bin_dir"]
    p = cfg["probe_params"]
    if name == "cpu":
        return [f"{b}/cpu_work", str(p["cpu_work_iterations"])]
    if name == "mbw":
        return [f"{b}/stream"]
    if name == "cache":
        return [f"{b}/pchase", str(p["pchase_bytes"]), str(p["pchase_steps"])]
    if name in ("mem", "tel"):
        return None
    raise ValueError(f"unknown probe {name!r}")


def _read(path: str) -> str:
    return Path(path).read_text()


def run_probe(
    name: str,
    cfg: dict,
    run_cmd: Callable[[list[str]], str] | None = None,
    read: Callable[[str], str] = _read,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.time,
) -> dict:
    """Run one probe; returns {probe, value, t_start, t_stop[, snapshots]} (epoch seconds, UTC)."""
    run_cmd = run_cmd or (
        lambda c: subprocess.run(c, check=True, capture_output=True, text=True).stdout
    )
    t0 = clock()
    rec: dict = {"probe": name}
    if name == "cpu" or name == "cache":
        rec["value"] = parse_last_number(run_cmd(probe_command(name, cfg)))
    elif name == "mbw":
        rec["value"] = 1.0 / parse_stream_triad(run_cmd(probe_command(name, cfg)))
    elif name == "mem":
        rec["value"] = mem_value(read("/proc/meminfo"))
    elif name == "tel":
        snap = lambda: {f: read(f"/proc/{f}") for f in ("stat", "meminfo", "vmstat")}  # noqa: E731
        first = snap()
        sleep(float(cfg["probe_params"]["tel_seconds"]))
        rec["value"] = None
        rec["snapshots"] = [first, snap()]
    else:
        raise ValueError(f"unknown probe {name!r}")
    rec["t_start"], rec["t_stop"] = t0, clock()
    return rec
