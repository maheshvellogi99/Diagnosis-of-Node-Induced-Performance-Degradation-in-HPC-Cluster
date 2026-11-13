# Phase 6-prep report — everything that does not need the target machine (D112–D115)

14 Nov 2026. Development machine only (arm64 Mac).
- **Not done:** no hardware work, no HPAS or STREAM, no stress program, no anomaly, no package install,
  no network call, no clone. Phase 5F was not run.
- **Sealed scenarios:** seeds 20–39 and nulls 200–399 were not touched. The only scenario runs were the
  25 development runs of §3.
- **Before this phase:** nothing had been done for Phase 6 (no `cluster/` folder, no installs, no clones,
  no Phase 6 commits), so there was nothing to list or undo.
- **One side effect:** editing `pyproject.toml` (a ruff per-file setting) made `uv` rebuild the local
  `hpcdiag` package. No dependency changed.

## 1. Summary

| Decision | Delivered | Check | Result |
|---|---|---|---|
| D112 freeze guard | `src/hpcdiag/freeze.py`, `scripts/check_frozen.py`; PLAN 5F now requires it | 38 tests (`tests/test_freeze.py`); guard on this repo | `PASS: 6 frozen files checked, policy 'ig_d109'` (exit 0); policy `ig` exits 1 |
| D113 substitution rule + plumbing | per-probe σ (`probes.sigma` as a mapping), empirical pool (`probes.healthy_pool`) | 5 tests (`tests/test_d113.py`); plumbing check run 20261114-001524-plumbing-check | **25 of 25 dev5c runs identical**, every field except wall-clock timings |
| D114 protocol | `docs/PHASE6_PROTOCOL.md`: tiers A/B/C, verbatim safety rules, analysis, Gate 6, estimate | estimate run 20261114-002601-estimate | 9.46 h per machine and day; 56.74 machine-h for 2 machines × 3 days |
| D115 measurement kit | `cluster/` (own README), standard library only | 34 tests (`tests/test_cluster_kit.py`), `--dry-run` | the harness refuses on this Mac; dry run prints commands, executes nothing |
| D115 analysis | `src/hpcdiag/phase6/analysis.py`, `scripts/phase6_analyze.py` | 11 tests on **synthetic** logs (`tests/test_phase6_analysis.py`) | σ, k, s, costs and mem shift recovered; test days never enter the fit |

All tests: 416 passed (`uv run pytest -q`), 88 of them new. `ruff check` is clean.

## 2. D112 freeze guard

`scripts/check_frozen.py --config configs/phase5.yaml --policy <name>` exits non-zero unless all three
checks pass:
- **(a) Tag.** `v1.0-dev-frozen` exists and is an ancestor of HEAD.
- **(b) Frozen files.**
  - It reads `docs/FROZEN_CONFIG.md` from the **tag**, not the working tree, and the working-tree copy
    must equal it.
  - Every listed file must have its SHA-256. These are the Layer 1, 2 and 3 configs, the scenario
    config, the raw-data checksums and the scenario index.
  - In `configs/phase5.yaml` only, a difference is allowed if every changed key is a D113 field. The
    tagged file must itself match the document.
- **(c) Policy.** The policy must be `ig_d109`, built by `run_phase5.make_policy` (now shared with the
  runner), with probe_J = 0.01 exactly and rank / ig_cost / cleared_skipdone.
  - The trap is caught: `Policy("ig_d109_B7200", 7200, exclusion="cleared_skipdone")` has the default
    probe_J = 1.0 and fails `check_policy_object` (tested).

Tests cover:
- the D113 field list: 8 allowed keys and 12 refused, including `telemetry.crossblock_counts`,
  `evalue.edges`, `policy.tau` and a new top-level key;
- allowed versus refused changes, and changes in a non-phase5 config;
- binary files, and the tag and the document disagreeing;
- the policy trap;
- a temporary git repo without and with the tag;
- the guard and the CLI exit codes on this repo.

## 3. D113 plumbing

**Config shape.**
- `probes.sigma` may be a number, as now, or a mapping {cpu, mbw, cache, mem} → σ that must name exactly
  the probes.
- `probes.healthy_pool` may name a CSV (probe, log_x, t_start). Per probe, the **first n_pool runs in
  time order** form the pool.

**Code changes.**
- **Per-probe σ.** The D96 state bets draw one standard-normal pool set and scale it by each probe's σ.
  The scalar path keeps its old draw, so it is unchanged. The fixed pools, the outcome draws, the Bayes
  update and the IG all take each probe's own σ.
- **Empirical pool.** Pool-per-base and misspecified rows are refused with an empirical pool (open item,
  §7).
- **Exact CSV floats.** The CSV is read with `float_precision="round_trip"`. Without it, pandas'
  default parser changed the last bit of some values; the test caught this.

**Tests** (`tests/test_d113.py`, written before the code; 4 failed, then all pass):
- equal per-probe σ equals scalar σ bit for bit (bets G and P, pools, a whole policy run);
- with different σ, each probe's bets and pool equal those of a scalar model at that probe's σ;
- an empirical pool holding the Gaussian pool's values (shuffled in time, with later outliers) equals the
  Gaussian run bit for bit;
- errors for pool per base, misspecified rows, too few runs, and a missing probe.

**Plumbing check (D113).** `scripts/phase6prep_plumbing_check.py`, run 20261114-001524-plumbing-check at
commit 4199cad:
- **What was re-run:** dev5c's ig_d109 runs at B = 7200, K = 1%, s = 1.3, seeds 0–4, plus the first 20
  nulls.
- **Result: 25 of 25 identical** to run 20261113-223241-dev5c in every field except `layer2_s`, `sim_s`
  and `total_s`.
- **Still valid:** no file in `src/hpcdiag/layer3` or `src/hpcdiag/stats` changed after that commit.

**Disclosure: whole-tree reformatting.** In the D113 step I ran `ruff format` on all of `src/` and
`tests/` instead of only my files, and committed the result in 4199cad. `scripts/phase6prep_format_check.py`
compares each changed file with the tag:
- **39 files are formatting only:** their syntax trees are identical to the tag. They are 26 source files
  (all `layer1/`, `inject/`, `eval/`, `features/encoding.py`,
  `stats/{aci,calibrators,conformal}.py`, `utils/run.py`, and four `io/` files) and 13 test files.
- **`io/build.py`** differs only in the indentation of one usage line in its docstring.
- **The 5 `layer3/` files** carry the intended D113 changes.
- **Effect:** no behaviour of Layers 1–2 changed. But frozen code no longer has the tag's bytes.
- **Your call:** I can restore those 40 files byte for byte from the tag. I have not done so.

## 4. D114 protocol and estimate

`docs/PHASE6_PROTOCOL.md` contains:
- **§1** the tiers;
- **§2** the safety rules, copied verbatim from LAYER3_DESIGN §5 and PROMPT 6 Step 0, with how the kit
  enforces them;
- **§3** the D113 rule and the guard;
- **§4** the measurement plan;
- **§5** the analysis procedure, fixed now;
- **§6** Gate 6;
- **§7** the estimate;
- **§8** tier C;
- **§9** the differences from earlier text, for your confirmation.

**Estimate** (`scripts/phase6_estimate.py`, run 20261114-002601-estimate). It uses the planned probe
costs, so one battery is 121 s:

| Machines | per machine and day: healthy / injection / benchmark / total (h) | machine-h over 3 days | healthy runs per probe | reps per cell |
|---|---|---|---|---|
| 2 | 5.71 / 1.88 / 1.87 / 9.46 | 56.74 | 1020 | 24 |
| 3 | 5.71 / 1.88 / 1.87 / 9.46 | 85.11 | 1530 | 36 |
| 4 | 5.71 / 1.88 / 1.87 / 9.46 | 113.47 | 2040 | 48 |

The 1000-run healthy floor alone is 33.61 machine-hours. Tier C needs none.

An earlier estimate run, 20261114-002550-estimate, printed the same numbers. It ran from the script
before lint passed and before it was committed: I chained with `;` instead of `&&`. Run 20261114-002601
is the one from the committed script.

**Gate 6** (defined in the protocol, §6) reports these numbers, with no thresholds on them and no tuning:
- σ per probe with a 95% bootstrap interval;
- held-out diagnosis accuracy on the last third of campaign days, with a Wilson 95% interval and per
  cause, plus the healthy share called h0;
- measured s per anomaly × intensity against the Phase 3 range [1.1, 3.0];
- costs and k.

It is complete only with:
- ≥ 1000 healthy runs per probe;
- ≥ 3 days and ≥ 2 machines;
- ≥ 10 repetitions per cell;
- every number traced to a log;
- the safety checklist signed and, in tier A, the admin sign-off.

## 5. D115 measurement kit (`cluster/`)

| File | What |
|---|---|
| `kit_safety.py` | refuses unless x86-64 Linux and the hostname is in `approved_hosts.txt` (0 bytes in git); iometadata, iobandwidth and netoccupy always refused; `UNKNOWN` placeholders refused; memleak cap must be < 0.25 of MemTotal (set to 0.20) |
| `kit_probes.py` | cpu (fixed work, s), mbw (1 / STREAM Triad MB/s), cache (pointer chase, ns per access), mem (MemTotal / MemAvailable), tel (/proc/stat, /proc/meminfo, /proc/vmstat, 60 s apart) |
| `kit_harness.py` | JSONL log with UTC start/stop on every line; each anomaly in its own process group; watchdog kills at the hard end time (250 s) and on the memleak resident-memory cap; whole session refused on an unapproved host; `--dry-run` executes nothing |
| `kit_campaign.py` | randomised plans: healthy batteries, injection campaign, paired benchmarks |
| `setup_target.sh` | target only: host check, HPAS clone at a required 40-hex `HPAS_COMMIT` (the only network call), STREAM checksum, builds. **Not run.** |
| `src/cpu_work.c`, `src/pchase.c` | the cpu and cache probes. **Not compiled here.** |

**Tests on this Mac** (34):
- the Mac is refused, even when its hostname is listed;
- refusals for linux/aarch64, darwin/x86_64 and an unlisted host;
- forbidden and unknown anomalies;
- the memleak cap;
- the parsers and probes with injected I/O;
- the plan counts and randomisation;
- the dry run executes nothing: `subprocess.Popen` and `subprocess.run` raise if called;
- real mode refuses on this Mac and on an approved Linux host while templates are `UNKNOWN`;
- the watchdog kills a benign `sleep` at its hard end and on the memory cap;
- the CLI refuses the session and writes no log, and the CLI dry run.

**Still `UNKNOWN`** (`cluster/campaign.json`), to be filled on the target from the pinned HPAS README
and a calibration run, then committed before any healthy run:
- the HPAS command templates and intensity arguments;
- the probe work sizes;
- the single-node benchmark;
- the HPAS commit and build commands.

## 6. Analysis (tested on synthetic logs only)

The procedure is fixed in the module docstring and in protocol §5:
- day-wise split: the first ⌈2D/3⌉ days are fit days;
- log x relative to each host's healthy fit-day median;
- σ with a bootstrap interval;
- k by least squares on log s, kept frozen where it is not identifiable (memleak);
- the memory shift;
- the empirical pool;
- held-out top-1 accuracy with Wilson intervals;
- the Gate 6 fields.

**Synthetic test campaign:** 2 hosts, 3 days, the known σ, k, s and costs. Everything is recovered:
- σ within 10%;
- k within 0.05;
- s and costs within 2%;
- the memory shift within 0.02;
- held-out accuracy above 0.9.

**Other checks:**
- Adding +5 to every test-day value leaves every fitted number unchanged.
- The output passes the D112 field check and loads through the D113 plumbing.

No real log exists.

## 7. Open items (need your decision; also in protocol §9)

1. **Telemetry key.** D113 names `telemetry.counts`, but the primary reads `telemetry.crossblock_counts`
   (D98). A measured matrix in `counts` would not reach the primary. The guard allows only
   `telemetry.counts`, as written. **UNKNOWN — needs check:** which key should Phase 6 replace?
2. **No Phase 6 telemetry classifier** is defined, so `telemetry.counts` cannot be produced yet.
3. **Bin edges.** PROMPT 6 step 2 refits the bin edges from healthy quantiles; D113 freezes them. The
   protocol follows D113.
4. **Healthy run count.** PROMPT 6 asks for ≥ 30 runs per node; D114 asks for ≥ 1000 per probe type. The
   protocol follows D114.
5. **Empirical pool and n_pool.** Pool per base and misspecified rows are undefined for an empirical pool.
   When there are more measured runs than n_pool = 1000, only the first 1000 in time order are used.
6. **Things I added beyond your D-text:**
   - the tier-B entry file `docs/tierB_machines.md`;
   - the clause in the D115 row "no Phase 6 hardware work until docs/admin_approval.md (tier A) or a
     tier-B machine list exists";
   - the kit settings marked [D] in `campaign.json` (30 s warm-up, 60 s cool-down, 240 s anomaly
     duration plus 10 s grace, memleak 0.05/0.15 of MemTotal with cap 0.20, 4 repetitions and 170
     healthy batteries per machine and day).
7. **STREAM** is placed by hand and checksum-verified, because the kit's only network call is the HPAS
   clone. The `Triad:` parser is tested on a sample I wrote, not on real STREAM output.
8. **Formatting.** Should the 40 formatting-only files (§3) be restored from the tag?

## 8. Commands

```
uv run pytest -q
uv run ruff check .
uv run python scripts/check_frozen.py --config configs/phase5.yaml --policy ig_d109
uv run python scripts/phase6prep_plumbing_check.py --config configs/phase5.yaml --ref 20261113-223241-dev5c   # 20261114-001524-plumbing-check
uv run python scripts/phase6_estimate.py --config configs/phase5.yaml --kit cluster/campaign.json             # 20261114-002601-estimate
uv run python scripts/phase6prep_format_check.py
python3 cluster/kit_harness.py --config cluster/campaign.json --plan <plan.json> --dry-run
```
