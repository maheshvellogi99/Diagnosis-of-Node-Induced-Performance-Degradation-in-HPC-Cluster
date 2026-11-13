# PHASE6_PROTOCOL.md — measuring Layer 3 on real hardware (D112–D120)

Written 14 Nov 2026, before any real measurement. It is hardware-agnostic and supersedes the
provisional Phase 6 text in PLAN §9 and LAYER3_DESIGN §5. **It stands where it differs from PROMPT 6
(D118):** the bin edges stay frozen and there are ≥ 1000 healthy runs per probe type. PROMPT 6 steps 2–3
are marked superseded in docs/PHASE_PROMPTS_5_6.md.

**Status:** no approved machine exists. There is no `docs/admin_approval.md`, `docs/tierB_machines.md`
lists no machine, and the development Mac is arm64; HPAS membw needs x86-64 (F23). Nothing in this
protocol has run on any machine. **Never run an HPAS anomaly or a stress program on the development
Mac.** No Phase 6 hardware work happens until `docs/admin_approval.md` (tier A) or a filled
`docs/tierB_machines.md` (tier B) exists (D115, accepted in D120).

## 1. Tiers

| Tier | Hardware | Entry condition | What it yields |
|---|---|---|---|
| **A** | Institute cluster, 2–4 dedicated nodes | written admin approval in `docs/admin_approval.md` (PROMPT 6 steps 0, 1, 4, 5, 6; steps 2–3 are replaced by §4 here) | measured D113 inputs; multi-node MPI straggler check; end-to-end demonstration |
| **B** | 2–4 identical dedicated x86-64 Linux machines we control (lab PCs or rented instances), running no one else's jobs | `docs/tierB_machines.md` filled in (machines, owner, exclusivity); hostnames in `cluster/approved_hosts.txt` | measured D113 inputs; MPI straggler check only with ≥ 2 machines |
| **C** | none | — | Phase 5F on the design tables (FROZEN_CONFIG); the paper states that Layer 3 is simulation-only |

In tiers A and B, the result is the frozen primary with only the D113 inputs replaced by
measurements, tagged `v1.1-measured`. Phase 5F then runs once on the sealed seeds, after the D112
guard passes (§3), with the measured tables drawn as in §8.

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
- Before the campaign, the harness refuses a plan that cannot give the run counts of §4.1 (D119a).

## 3. What may change (D113, D117) and the freeze guard (D112, D116)

**Phase 6 may replace only these measured inputs:**
- `probes.k`
- `probes.sigma`, which may become one σ per probe
- `probes.cost_s`
- `probes.mem_fixed_shift`
- the healthy pool and the measured outcome draws, as `probes.healthy_pool` (§8)
- **`telemetry.crossblock_counts`**: the matrix the primary reads (D117). `telemetry.counts` is no
  longer a D113 field; listing it in D113 was the architect's error.

**Everything else stays frozen:** bin edges, β, bet_s0, s_grid, J, π0(h0), τ, exclude_days, the policy,
and Layers 1 and 2.

`scripts/check_frozen.py` exits non-zero unless all four checks pass:
- **(a)** the tag `v1.0-dev-frozen` exists and is an ancestor of HEAD;
- **(b)** every file in the tagged `docs/FROZEN_CONFIG.md` has its listed SHA-256, or, in
  `configs/phase5.yaml` only, every difference from the tagged file is a D113 field;
- **(c)** the policy is `ig_d109` with probe_J = 0.01, built as the runner builds it;
- **(d) (D116)** `src/` differs from the tag only in `src/hpcdiag/layer3/` (D113/D119 plumbing),
  `src/hpcdiag/freeze.py` and `src/hpcdiag/phase6/`. This covers committed and uncommitted changes and
  untracked files.

**Process rule (D116):** run the formatter only on the files you edited.

## 4. Measurement plan (tiers A and B)

**4.0 Before the first healthy run (pre-registration, committed):**
- HPAS commit hash, its build commands (`cluster/HPAS_BUILD.md`), and the STREAM source checksum.
  STREAM is placed by hand (D120).
- **The `Triad:` parser of `cluster/kit_probes.py` is verified on real STREAM output on the target**,
  and that output is committed as a test fixture (D120).
- Every `UNKNOWN` in `cluster/campaign.json` filled in:
  - the HPAS command templates and the two intensity arguments per anomaly;
  - the probe work sizes, calibrated so that a healthy run takes about `probes.cost_s`;
  - the fixed single-node benchmark (about 60 s healthy);
  - `planned_machines`, `planned_days` and `healthy_batteries_per_day` for the real campaign.
- `docs/tierB_machines.md` or `docs/admin_approval.md`.
- A passing `--dry-run` of every plan type, including the campaign-size check.
- The analysis procedure is already fixed (§5); it is not changed after data arrive.

**4.1 Healthy baselines (D119a).** The campaign must give **≥ 1000 fit-day and ≥ 300 test-day healthy
runs of every probe type**, on **≥ 2 machines** over **≥ 3 days**:
- Fit days are the first ⌈2D/3⌉ campaign days.
- The pool is the first 1000 fit-day runs per probe (n_pool is not changed); test-day runs are never in
  the pool.
- `kit_safety.check_campaign_size` computes both counts from the plan (machines, days, batteries per
  machine-day); the harness refuses an undersized plan before anything runs.
- The shipped plan is 2 machines × 3 days × 250 batteries per day: 1000 fit-day and 500 test-day runs
  per probe.
- Each battery runs every probe once, in a random order.

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
   Test-day runs are normalised with the same fit-day b.
3. **σ per probe** = SD of healthy fit-day log x. Its 95% interval comes from a percentile bootstrap
   over runs (2000 resamples, seed 0).
4. **cost_s per probe** = median duration of healthy fit-day runs.
5. **k_{cause,probe}** = Σ L·y / Σ L² over fit-day anomaly runs, with L = log s.
   - If every |L| < 0.01, the entry is not identifiable and keeps its frozen value. That is listed.
     memleak is expected to be such a case (F23).
   - memleak on the mem probe gives **mem_fixed_shift** = mean fit-day log x.
6. **Measured tables** (`healthy_pool` CSV: probe, log_x, t_start, host, split):
   - fit rows: every healthy fit-day run; the pool is the first 1000 per probe in time order;
   - test rows: every healthy test-day run, used only as outcome draws (§8).
7. **Held-out diagnosis.** On test days, each anomaly repetition's outcomes update the frozen-prior
   posterior over (cause, s), using the fitted tables with the frozen s_grid and π0(h0). Accuracy is
   top-1 hypothesis against the true cause, with a Wilson 95% interval. For healthy test batteries, the
   share called h0 is reported.
8. **Telemetry (D117).**
   - **Features:** per-second rates of every numeric field between the two tel snapshots (/proc/stat,
     /proc/meminfo, /proc/vmstat), over the fields present in every sample. The kit records the time of
     each snapshot.
   - **Classifier:** the D94 random forest (`configs/phase5.yaml` `antarex.rf`: 300 trees, balanced
     class weights, seed = config seed).
   - **Classes:** h0, cpu, cache, mbw, mem (healthy, cpuoccupy, cachecopy, membw, memleak).
   - **Confusion matrix:** leave-one-fit-day-out, summed, fit days only.
   - **Replacement rule:** it replaces `telemetry.crossblock_counts` **only if every class has ≥ 30
     fit-day samples**. Otherwise the frozen Antarex matrix stays and the measured one is reported as
     secondary.
   - The counts-to-likelihood code (add-one smoothing) is unchanged.

## 6. Gate 6 (defined now; reported numbers, no tuning on them)

Gate 6 reports the following. None of them is a pass/fail threshold on a measured value, and nothing
is tuned on any of them:
- **σ per probe**, with a 95% bootstrap interval and the number of fit-day and test-day healthy runs.
- **Held-out diagnosis accuracy** on the test days, with a Wilson 95% interval and per cause, plus the
  healthy-battery share called h0.
- **Measured s per anomaly × intensity** against the Phase 3 range [1.1, 3.0] (configs/phase3_inject.yaml,
  s and s_extra), flagged inside or outside.
- **Probe costs** and **fitted k**, listing the entries kept frozen.
- **The telemetry matrix**, whether it replaced the frozen one, and why.

Gate 6 is **complete** only if all of these hold:
- ≥ 1000 fit-day and ≥ 300 test-day healthy runs per probe;
- ≥ 3 days and ≥ 2 machines;
- ≥ 10 repetitions per anomaly × intensity;
- every number traced to a log file;
- the safety checklist signed and, in tier A, the admin sign-off recorded.

## 7. Wall-clock estimate per tier (D119a)

Run 20261114-010029-estimate:
`uv run python scripts/phase6_estimate.py --config configs/phase5.yaml --kit cluster/campaign.json`.

The estimate uses the planned probe costs: one battery of every probe = 121 s. Healthy batteries per
machine-day are the minimum that gives ≥ 1000 fit-day and ≥ 300 test-day runs per probe. Other settings:
30 s warm-up, 60 s cool-down, 60 s benchmark, 4 anomaly repetitions per cell per machine-day.

| Machines | Days | Healthy batteries per machine-day | Fit / test healthy runs per probe | h per machine-day: healthy / injection / benchmark / total | Machine-h total | Reps per cell |
|---|---|---|---|---|---|---|
| 2 | 3 | 250 | 1000 / 500 | 8.40 / 1.88 / 1.87 / 12.14 | 72.87 | 24 |
| 2 | 6 | 125 | 1000 / 500 | 4.20 / 1.88 / 1.87 / 7.94 | 95.32 | 48 |
| 3 | 3 | 167 | 1002 / 501 | 5.61 / 1.88 / 1.87 / 9.36 | 84.20 | 36 |
| 3 | 6 | 84 | 1008 / 504 | 2.82 / 1.88 / 1.87 / 6.57 | 118.18 | 72 |
| 4 | 3 | 125 | 1000 / 500 | 4.20 / 1.88 / 1.87 / 7.94 | 95.32 | 48 |
| 4 | 6 | 63 | 1008 / 504 | 2.12 / 1.88 / 1.87 / 5.86 | 140.63 | 96 |

- **Tier C:** 0 machine-hours.
- **Not included:** setup, calibration and the MPI straggler check. The real probe costs replace the
  planned 121 s once they are measured.

## 8. Phase 5F with measured tables (D119b)

Implemented in `src/hpcdiag/layer3/setup.py` and `simulate.py` (`World.draw`) and tested in
`tests/test_d119.py`. The Gaussian path is unchanged bit for bit (25/25 plumbing check).

**Draws:**
- **Healthy outcomes:** a resampled test-day healthy log x, never in the pool.
- **Faulty outcomes:** such a draw plus the model mean, k·log s (or the mem shift for memleak on the mem
  probe).

**Validity rows:**
- **Fixed pool:** the first 1000 fit-day runs per probe.
- **Pool per base:** 1000 fit-day runs per probe resampled with replacement, seeded by the base.
- **Cross-machine (≥ 2 machines):** the pool is 1000 runs resampled with replacement from the fit-day
  runs of the first host in sorted order; outcomes are drawn from the second host's test-day runs. One
  direction only. With replacement because one machine may have fewer than 1000 fit-day runs.
- **The misspecified row is not run** with measured tables; the code refuses it.

**Healthy-growth diagnostic:** with measured outcomes it is computed exactly from the test-day draws
against the pool.

## 9. Tier C

Phase 5F runs on the design tables with the D112 guard. The paper states that Layer 3 is
simulation-only: the probe model, σ, k, costs and pools are design values [A], and no hardware number
exists.

## 10. Still open

1. **Unknown until the repository is read on the target:**
   - the HPAS command-line options, build steps and commit to pin;
   - the single-node benchmark;
   - the MPI benchmark.

   All are `UNKNOWN` placeholders that the harness refuses.
2. **Cross-machine row:** one direction only (the first host's pool, the second host's outcomes), and a
   resampled pool. Both are my choices and need your confirmation.
3. **Rare ties:** ties between a measured outcome and pool values are not split by the healthy-growth
   diagnostic. That diagnostic is reported only and is never used by the policy; continuous
   measurements make ties rare.
