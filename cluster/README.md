# cluster/ — Phase 6 measurement kit (D114, D115)

Measures the D113 inputs (probe outcome tables, σ per probe, probe costs, the memory shift and a
healthy pool) on real x86-64 Linux machines. The protocol, the three tiers and Gate 6 are in
`docs/PHASE6_PROTOCOL.md`. This folder holds only what runs on the target machine. The analysis
(`scripts/phase6_analyze.py`) runs back on the development machine, on copied logs.

**Status (14 Nov 2026): nothing has run on any machine.** There is no approved machine yet. The
development Mac is arm64, and HPAS membw needs x86-64. On that Mac, only the unit tests
(`uv run pytest tests/test_cluster_kit.py`) and `--dry-run` may run. **Never run an HPAS anomaly or a
stress program on the development Mac.**

## Safety rules (copied verbatim from docs/LAYER3_DESIGN.md §5)

> **Safety rules (non-negotiable).** Never run iometadata or iobandwidth (filesystem outages, F23; runs terminated by admins, F25).
> Do not run netoccupy (shared network; SHMEM; effect only in two-node runs, F23/F25). memleak capped well below node memory, with a
> watchdog that kills it. Every anomaly has an explicit end time. Only on the approved nodes.

The kit enforces them in code (`kit_safety.py`, `kit_harness.py`). An anomaly is refused unless every
one of these holds:
- The host is **x86-64 Linux**.
- The **hostname is listed in `approved_hosts.txt`**. That file is **empty in git**; add a host only
  after the written approval (tier A, `docs/admin_approval.md`) or the tier-B machine list exists.
- The anomaly is one of **cpuoccupy, cachecopy, membw, memleak**. iometadata, iobandwidth and
  netoccupy are always refused.
- Its command has **no `UNKNOWN` placeholder**.
- For memleak, the request is at most `memleak_cap_fraction` of MemTotal, and that fraction must be
  **below 0.25**. It is 0.20 in `campaign.json`.

Each anomaly runs in its own process group. A watchdog kills the group at its **hard end time**
(`anomaly_duration_s + grace_s`). For memleak, it also kills the group when the group's resident memory
passes the cap. The harness refuses the whole session, even healthy probes, on a host that is not
approved.

## Files

| File | What |
|---|---|
| `kit_safety.py` | the safety gate (host, anomaly list, template, memleak cap) |
| `kit_probes.py` | probes: cpu (fixed work, s), mbw (1 / STREAM Triad MB/s), cache (pointer chase, ns per access), mem (MemTotal / MemAvailable), tel (/proc/stat, /proc/meminfo, /proc/vmstat snapshots 60 s apart); larger x = worse |
| `kit_harness.py` | runs a plan, logs JSONL (UTC start/stop timestamps on every line), watchdog, `--dry-run` |
| `kit_campaign.py` | randomised plans: healthy batteries, injection campaign, benchmark with/without anomaly |
| `campaign.json` | settings: [D] values fixed now, `UNKNOWN` values filled on the target before the healthy campaign |
| `src/cpu_work.c`, `src/pchase.c` | the cpu and cache probes, built on the target |
| `setup_target.sh` | target-only setup: host check, HPAS clone at a pinned commit, STREAM checksum, builds |
| `approved_hosts.txt` | approved hostnames, one per line; empty in git |

`bin/`, `third_party/` and `logs/` are git-ignored. Copy the logs back for analysis.

## Steps on an approved target (tier A or B)

1. Add the hostname to `approved_hosts.txt` and commit, citing the approval.
2. Place McCalpin's `stream.c` at `third_party/stream.c`. Then run:
   `HPAS_COMMIT=<40-hex> STREAM_SHA256=<sha256> STREAM_ARRAY_SIZE=<n> bash cluster/setup_target.sh`.
   - The `git clone` of https://github.com/peaclab/HPAS inside this script is the kit's only network
     call.
   - The HPAS build commands are **UNKNOWN — needs check**. Build HPAS by hand following its README at
     the pinned commit, and record the exact commands in `cluster/HPAS_BUILD.md`.
3. Fill every `UNKNOWN` in `campaign.json`, then commit, all **before** any healthy run:
   - **Command templates and intensity arguments.** Take these from the HPAS README at the pinned
     commit. The HPAS command-line options are not known here and must not be guessed.
   - **Probe work sizes.** Calibrate them so that a healthy run takes about `configs/phase5.yaml`
     `probes.cost_s`.
   - **Benchmark command.** Choose the fixed single-node benchmark.
4. Dry run: `python3 cluster/kit_harness.py --config cluster/campaign.json --plan plan.json --dry-run`.
5. Generate the plans with `kit_campaign.py` (seed = day × 100 + machine), then run them with `--log`.
   The schedule and run counts are in `docs/PHASE6_PROTOCOL.md`.
6. Copy `logs/*.jsonl` back and run `scripts/phase6_analyze.py` (D113 inputs and Gate 6 numbers).

## Python

The kit uses the standard library only. It is written for Python 3.8+, but has been tested only with
Python 3.11 here.
