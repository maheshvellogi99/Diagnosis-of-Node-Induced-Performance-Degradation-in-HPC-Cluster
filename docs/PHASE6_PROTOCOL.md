# PHASE6_PROTOCOL.md — measuring Layer 3 on real hardware (D112–D115)

Written 14 Nov 2026, before any real measurement. It is hardware-agnostic and supersedes the
provisional Phase 6 text in PLAN §9 and LAYER3_DESIGN §5 wherever they differ (see §9 for the
differences that need your confirmation).

**Status:** no approved machine exists. There is no `docs/admin_approval.md`, and the development Mac is
arm64; HPAS membw needs x86-64 (F23). Nothing in this protocol has run on any machine. **Never run an
HPAS anomaly or a stress program on the development Mac.**

## 1. Tiers

| Tier | Hardware | Entry condition | What it yields |
|---|---|---|---|
| **A** | Institute cluster, 2–4 dedicated nodes | written admin approval in `docs/admin_approval.md` (PROMPT 6 as written, with the §9 changes) | measured D113 inputs; multi-node MPI straggler check; end-to-end demonstration |
| **B** | 2–4 identical dedicated x86-64 Linux machines we control (lab PCs or rented instances), running no one else's jobs | written list of the machines, their owner and their exclusivity, in `docs/tierB_machines.md`; hostnames in `cluster/approved_hosts.txt` | measured D113 inputs; MPI straggler check only with ≥ 2 machines |
| **C** | none | — | Phase 5F on the design tables (FROZEN_CONFIG); the paper states that Layer 3 is simulation-only |

In tiers A and B, the result is the frozen primary with only the D113 inputs replaced by
measurements, tagged `v1.1-measured`. Phase 5F then runs once on the sealed seeds, after the D112
guard passes (§3).

## 2. Safety rules

Copied verbatim from docs/LAYER3_DESIGN.md §5:

> **Safety rules (non-negotiable).** Never run iometadata or iobandwidth (filesystem outages, F23; runs terminated by admins, F25).
> Do not run netoccupy (shared network; SHMEM; effect only in two-node runs, F23/F25). memleak capped well below node memory, with a
> watchdog that kills it. Every anomaly has an explicit end time. Only on the approved nodes.

Copied verbatim from docs/PHASE_PROMPTS_5_6.md, PROMPT 6, Step 0 (it applies in tier B too, with "approved
nodes" read as the machines in `docs/tierB_machines.md`):

> Step 0 - safety check, before any code runs on the cluster
> - List the approved nodes, CPU architecture (must be x86-64 for membw), memory per node, scheduler.
> - Confirm in writing in reports/phase_6_safety.md: no iometadata, no iobandwidth, no netoccupy; memleak capped at a
>   config value below 25% of node memory with a watchdog; every anomaly has an end time; only approved nodes.
> - Show me the exact commands you will run. Wait for my "go".

The kit enforces these rules in code (`cluster/kit_safety.py`, `cluster/kit_harness.py`, tested in
`tests/test_cluster_kit.py`):
- An anomaly is refused unless the host is x86-64 Linux and its hostname is in
  `cluster/approved_hosts.txt`, which is empty in git.
- iometadata, iobandwidth and netoccupy are always refused.
- memleak is capped at 0.20 of MemTotal. The cap must stay below 0.25, and a resident-memory watchdog
  enforces it.
- Every anomaly has a hard end time (`anomaly_duration_s + grace_s` = 250 s), at which its process group
  is killed.
- Any command still containing `UNKNOWN` is refused.

## 3. What may change (D113) and the freeze guard (D112)

**Phase 6 may replace only these measured inputs:**
- `probes.k`
- `probes.sigma`, which may become one σ per probe
- `probes.cost_s`
- `probes.mem_fixed_shift`
- the healthy pool, as `probes.healthy_pool`: an empirical pool of measured runs that replaces the
  Gaussian pool
- `telemetry.counts`

**Everything else stays frozen:** bin edges, β, bet_s0, s_grid, J, π0(h0), τ, exclude_days, the policy,
and Layers 1 and 2.

`scripts/check_frozen.py` (D112) enforces this. It exits non-zero unless three checks pass:
- **(a)** the tag `v1.0-dev-frozen` exists and is an ancestor of HEAD;
- **(b)** every file in the tagged `docs/FROZEN_CONFIG.md` has its listed SHA-256, or, in
  `configs/phase5.yaml` only, every difference from the tagged file is one of the fields above;
- **(c)** the policy is `ig_d109` with probe_J = 0.01, built as the runner builds it. A hand-built
  `Policy` with the default probe_J = 1.0 fails.

The plumbing exists and has been checked:
- **Per-probe σ:** `probes.sigma` may be a mapping from probe to σ.
- **Empirical pool:** per probe, the first n_pool measured healthy runs in time order.
- **Reproduction:** with scalar σ and the Gaussian pool, it reproduces the dev5c runs bit for bit
  (reports/phase_6prep.md §3).

## 4. Measurement plan (tiers A and B)

**4.0 Before the first healthy run (pre-registration, committed):**
- HPAS commit hash, its build commands (`cluster/HPAS_BUILD.md`), and the STREAM source checksum.
- Every `UNKNOWN` in `cluster/campaign.json` filled in:
  - the HPAS command templates and the two intensity arguments per anomaly;
  - the probe work sizes, calibrated so that a healthy run takes about `probes.cost_s`;
  - the fixed single-node benchmark (about 60 s healthy).
- The tier's machine list, and a passing `--dry-run` of every plan type.
- The analysis procedure is already fixed (§5); it is not changed after data arrive.

**4.1 Healthy baselines.** At least **1000 healthy runs of every probe type**, over **at least 3 days**
and **at least 2 machines**:
- Each battery runs every probe once, in a random order.
- With `healthy_batteries_per_day` = 170 per machine, 2 machines and 3 days give 1020 runs per probe.

**4.2 Injection campaign.** {cpuoccupy, cachecopy, membw, memleak} × 2 intensities × at least 10
repetitions × all probes:
- Order is randomised by `cluster/kit_campaign.py`, seed = day × 100 + machine.
- Each repetition has a 30 s warm-up, then a battery of every probe under the anomaly, then the anomaly
  is stopped and a 60 s cool-down follows.
- Start and stop are logged in UTC to the millisecond.
- At 4 repetitions per machine and day, 2 machines and 3 days give 24 per cell.

**4.3 Slowdown s.** s per anomaly × intensity is measured with a fixed single-node benchmark:
- With and without the anomaly, in random order, paired per repetition.
- s = median over hosts of (median time with the anomaly / median healthy time), on fit days.

**4.4 Multi-node MPI straggler check.** Only with ≥ 2 machines, after 4.1–4.3:
- 2–4-node MPI jobs, with one anomaly on one node.
- The job's slowdown is compared with the single-node s.
- The MPI benchmark is **UNKNOWN — needs a choice** and is pinned before it runs.

**4.5 Time split.** Tables are fitted on the first ⌈2D/3⌉ campaign days and checked on the remaining
days. There is no random split.

**Not part of this protocol:** the end-to-end Layer 1–2 demonstration of PROMPT 6 step 5 (pipeline only,
tier A). As LAYER3_DESIGN §5 notes, e-BH over 4 nodes makes no statistical claim.

## 5. Analysis (fixed now; `src/hpcdiag/phase6/analysis.py`, `scripts/phase6_analyze.py`)

1. Campaign days are the UTC dates of start times; at least 3 are needed. Fit days are the first
   ⌈2D/3⌉ and test days the rest. No test-day value enters any fitted number; a test checks this.
2. log x = log(value / b), where b is the median healthy value of that probe on that host on fit days.
3. **σ per probe** = SD of healthy fit-day log x. Its 95% interval comes from a percentile bootstrap
   over runs (2000 resamples, seed 0).
4. **cost_s per probe** = median duration of healthy fit-day runs.
5. **k_{cause,probe}** = Σ L·y / Σ L² over fit-day anomaly runs, with L = log s of the run's anomaly ×
   intensity.
   - If every |L| < 0.01, the entry is not identifiable and keeps its frozen value. That is listed.
     memleak is expected to be such a case, since memleak and memeater "do not visibly affect
     performance", F23.
   - memleak on the mem probe gives **mem_fixed_shift** = mean fit-day log x.
6. **Healthy pool** = healthy fit-day log x with start times (D113 empirical pool, first n_pool per
   probe in time order).
7. **Held-out diagnosis.** On test days, each anomaly repetition's outcomes update the frozen-prior
   posterior over (cause, s), using the fitted tables with the frozen s_grid and π0(h0). Accuracy is
   top-1 hypothesis against the true cause, with a Wilson 95% interval. For healthy test batteries, the
   share called h0 is reported.
8. **telemetry.counts** is not produced. No Phase 6 telemetry classifier is fixed: **UNKNOWN — needs a
   decision** (§9).

## 6. Gate 6 (defined now; reported numbers, no tuning on them)

Gate 6 reports the following. None of them is a pass/fail threshold on a measured value, and nothing
is tuned on any of them:
- **σ per probe**, with a 95% bootstrap interval and the number of healthy runs.
- **Held-out diagnosis accuracy** on the test days, with a Wilson 95% interval and per cause, plus the
  healthy-battery share called h0.
- **Measured s per anomaly × intensity** against the Phase 3 range [1.1, 3.0] (configs/phase3_inject.yaml,
  s and s_extra), flagged inside or outside.
- **Probe costs** and **fitted k**, listing the entries kept frozen.

Gate 6 is **complete** only if all of these hold:
- at least 1000 healthy runs per probe;
- at least 3 days and at least 2 machines;
- at least 10 repetitions per anomaly × intensity;
- every number traced to a log file;
- the safety checklist signed;
- in tier A, the admin sign-off recorded.

`gate6.json` holds every field.

## 7. Wall-clock estimate per tier

Run 20261114-002601-estimate:
`uv run python scripts/phase6_estimate.py --config configs/phase5.yaml --kit cluster/campaign.json`.

The estimate uses the planned probe costs from configs/phase5.yaml: one battery of every probe = 121 s.
Kit settings: 30 s warm-up, 60 s cool-down, 60 s benchmark, 170 healthy batteries and 4 repetitions per
machine and day, 3 days.

| Machines (tier A or B) | per machine and day: healthy / injection / benchmark / total (h) | machine-hours, 3 days | healthy runs per probe | reps per anomaly × intensity |
|---|---|---|---|---|
| 2 | 5.71 / 1.88 / 1.87 / 9.46 | 56.74 | 1020 | 24 |
| 3 | 5.71 / 1.88 / 1.87 / 9.46 | 85.11 | 1530 | 36 |
| 4 | 5.71 / 1.88 / 1.87 / 9.46 | 113.47 | 2040 | 48 |

- **Tier C:** 0 machine-hours.
- **Healthy floor:** the 1000-run floor alone is 33.61 machine-hours.
- **Not included:** setup, calibration and the MPI straggler check. The real probe costs replace the
  planned 121 s once they are measured.

## 8. Tier C

Phase 5F runs on the design tables with the D112 guard. The paper states that Layer 3 is
simulation-only: the probe model, σ, k, costs and pools are design values [A], and no hardware number
exists.

## 9. Differences from earlier text, and open items (need your confirmation)

1. **Bin edges.** PROMPT 6 step 2 says "fix outcome bin edges from these quantiles". D113 freezes the
   bin edges, so this protocol does **not** refit them.
2. **Healthy runs.** PROMPT 6 asks for ≥ 30 runs per node over ≥ 3 days; D114 asks for ≥ 1000 per probe
   type over ≥ 3 days and ≥ 2 machines. This protocol uses D114, which is stricter.
3. **Telemetry key.** D113 lists `telemetry.counts`, but the primary reads
   `telemetry.crossblock_counts` (D98; variant main uses `tel_belief = "crossblock"`). A measured matrix
   written into `telemetry.counts` would therefore **not** be used by the primary. The guard allows only
   `telemetry.counts`, as D113 says. **UNKNOWN — needs check:** should Phase 6 write the measured matrix
   into `crossblock_counts`, or should the primary read `counts` once measured?
4. **No telemetry classifier for Phase 6.** No classifier is fixed (model, features, split), so
   `telemetry.counts` cannot be produced yet. **UNKNOWN — needs a decision.**
5. **Pool per base and misspecified rows** are not defined for an empirical pool. The code refuses them.
   Resampling the measured runs per base would be one option. **UNKNOWN — needs a decision** before 5F
   uses a measured pool.
6. **More measured runs than n_pool.** If there are more measured healthy runs than n_pool = 1000, the
   first 1000 in time order form the pool and the rest are unused. Changing n_pool is not a D113 field.
7. **HPAS and STREAM.**
   - The HPAS command-line options, the build steps and the commit to pin are **UNKNOWN** until the
     repository is read on the target.
   - STREAM is not downloaded by the kit: PROMPT 6 allows only the documented git clone, so `stream.c`
     is placed by hand and checksum-verified.
   - The `Triad:` output parser is tested on a sample written here, not on real STREAM output.
