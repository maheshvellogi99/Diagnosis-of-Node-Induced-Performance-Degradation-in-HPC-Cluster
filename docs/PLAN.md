# MASTER PLAN — From Job Evidence to Root Cause

**Statistically Guaranteed and Cost-Aware Diagnosis of Node-Induced Performance Degradation in HPC Clusters**

Owner: Mahesh Vellogi (with Vishwesh M, Awhan Ray) · Guide: Dr. K. Nagaraju · IIITDM Kurnool
Version 1.0 · 9 Oct 2026 · Status: Layers 1–2 specified in full; Layer 3 provisional (papers pending)

---

## 0. How to use this document

- This is the single source of truth for design. Claude Code implements it phase by phase.
- Each claim carries a tag:
  - **[V]** verified — from a paper we have, or computed from the actual data (source given).
  - **[D]** design decision — our choice; can be changed, but only by editing this file and the Decision Log (§16).
  - **[A]** assumption — must be tested; the phase that tests it is named.
- Each phase has a **gate**: a measurable pass condition. We do not move on until it passes, or we
  record why it failed and what we changed.
- If code and this document disagree, the document wins until it is updated on purpose.

---

## 1. Project summary

### 1.1 Problem
A compute node can be **slow without being broken** (thermal throttling, memory-bandwidth loss,
leftover processes, failing parts). In a tightly coupled job all nodes wait for the slowest one,
so one slow node slows the whole job. A slow job may be the **workload's** fault (bigger input,
heavier code) or a **node's** fault. Telling these apart today needs expensive always-on telemetry
on every node.

### 1.2 Mapping of the guide's objectives (from `hpc_project.pdf`) to our layers [V]

| Guide's objective | Our component | Output |
|---|---|---|
| O1 Predict expected wall time; compute residuals | **Layer 1** — conformal job evidence | p-value per finished job |
| O2 Separate workload variation from node degradation | **Layer 2** — node e-processes + e-BH | set of suspect nodes, FDR ≤ α |
| O3 Trigger diagnostics only for suspects | **Layer 3** — cost-aware active probing | root cause + confidence + cost |
| O4 Unified dataset and validation (HPAS) | **Phases 3 and 6** — semi-synthetic + real injection | ground truth for every layer |

### 1.3 Gaps we fill (each backed by a source we hold)
1. **No calibrated definition of "late".** UARP lists conformal prediction as future work:
   "future work could employ conformal prediction (CP), which provides formal guarantees on
   coverage probabilities" [V: UARP, §6].
2. **Node flagging uses hand-picked thresholds.** The guide's notes use "median residual > 20%,
   ≥ 10 jobs, ≥ 3 users, ≥ 3 applications, 3 windows" [V: hpc_project.pdf §10.5]. No error-rate
   guarantee across hundreds of nodes.
3. **Shared blame in multi-node jobs.** RUAD treats nodes as independent and leaves
   inter-node dependencies to future work [V: RUAD §4.1].
4. **Diagnosis cost not optimised.** Telemetry diagnosis assumes data from every node
   (Tuncer 2019; Aksar 2024). *(Layer 3 papers pending — finalise in Phase 5.)*
5. **Integration missing** — the guide's own SLR: "little work integrates uncertainty-aware
   runtime residuals, multi-job node consensus, and on-demand high-resolution diagnosis" [V: hpc_prisma_slr.docx].

### 1.4 Expected contributions (what the paper will claim, if results hold)
- C1. Conformal p-values that make "this job was abnormally late" a statement with a bounded false-alarm rate.
- C2. Anytime-valid per-node evidence (product e-process) plus e-BH selection: the expected fraction
  of healthy nodes among flagged nodes is ≤ α, under any dependence between nodes.
- C3. Allocation-size weighting that shares blame across a job's nodes while keeping C2 valid.
- C4. Information-gain-per-cost probe selection for root cause, triggered only by C2.
- C5. An evaluation protocol with exact ground truth (semi-synthetic injection on real job traces
  + HPAS on a real cluster), released as open data and code.

---

## 2. What we know for certain, and what is still open

### 2.1 Verified facts used by this plan

| # | Fact | Source |
|---|---|---|
| F1 | Eagle file: 11,030,377 rows; submit times 2018-11-14 to 2023-02-22; 936 users; 42 partitions | computed on `eagle_data.parquet`, 9 Oct 2026 |
| F2 | Eagle columns: job_id, user, account, partition, qos, wallclock_req, nodes_req, processors_req, gpus_req, mem_req, submit_time, start_time, end_time, run_time, state, name, work_dir, submit_line. **No node list.** | same |
| F3 | Eagle states: COMPLETED 7,390,586; CANCELLED 1,813,244; FAILED 975,478; TIMEOUT 803,002; OUT_OF_MEMORY 20,400; PENDING 14,646; NODE_FAIL 12,088; RUNNING 933 | same |
| F4 | Eagle `run_time` is seconds (= end − start for 99.97% of completed jobs). `wallclock_req` is consistent with seconds (run_time > wallclock_req for only 0.016% of completed jobs). 107 rows have no wallclock_req. | same |
| F5 | Eagle COMPLETED+TIMEOUT: 8,193,588 jobs, ≈5,252/day; median run_time 992 s (≈16.5 min, matching the paper's "approximately 16 minutes"); 91.3% of completed jobs request 1 node; 10.1% run < 30 s | same |
| F6 | Menear et al. (PEARC'23) protocol: drop cancelled and failed jobs, keep timeouts; train on jobs **ending** before split, test on jobs **submitted** on the split day; training window 100 days; test window 1 day (retrain daily); label encoding; best categorical set = user + partition; best numerical sets include processors and memory (wallclock always included); XGBoost 1.7.1 `XGBRegressor` defaults; 244 split times every 6 days; RMSE 8.14 h for their per-split feature selection (8.45 h for previous-split selection) | Mastering HPC Runtime Prediction §3.2–3.6, App. A–B |
| F7 | PM100 (Marconi100): Zenodo description says 231,116 jobs; the md5-verified file has 231,238 rows, May 5 – Oct 13 2020 (162 days). **We use the file** (D11). Only jobs that ran exclusively on their nodes (Zenodo); one 287 MB parquet; Zenodo record 10127767 | Zenodo page; file computed in Phase 0 (`reports/phase_0.md`) |
| F8 | PM100 fields include `nodes` ("List of nodes allocated to job"), `num_nodes_alloc`, `user_id`, `partition`, `qos`, `submit_time`, `start_time`, `end_time`, `run_time` (seconds), `time_limit` (minutes or INFINITE), `job_state`, `num_cores_req/alloc`, `num_gpus_req/alloc`, `mem_req/alloc` | PM100 repo `documentation/job_features.md` |
| F9 | M100 node identifiers, user ids, job ids are anonymised to random integers; Nagios-flagged nodes are put **offline** and cannot run jobs until fixed | M100 ExaData paper |
| F10 | UARP: LightGBM quantile models τ ∈ {0.50, 0.99}; residual model σ(x)=sqrt(E[(y−Q0.50)²|x]); margin = max(0.2·Q0.99, 0.5·σ); 200 iterations, learning rate 0.05, 50 leaves; 70/30 temporal split; 6 features; MinMax scaling | UARP §4, §5 |
| F11 | CQR one-sided upper bound: P{Y ≤ q̂hi(X) + Q1−α(Ehi)} ≥ 1−α with Ehi = Y − q̂hi(X); Q = (1−α)(1+1/|I2|) empirical quantile | CQR eq. (11), (16) |
| F12 | Smoothed conformal p-value p_n = (|{i ≤ n: α_i > α_n}| + θ_n |{i ≤ n: α_i = α_n}|)/n, θ_n ~ U[0,1]; if observations are exchangeable, p_1, p_2, … are IID uniform | Vovk, *Testing Randomness Online*, eq. (5), Prop. 3.1, Remark 3.2 |
| F13 | Product of betting functions F = Π f_n(p_n) with ∫₀¹ f = 1 is a martingale; Ville: P(∃n: S_n ≥ c) ≤ 1/c | Vovk 2021 (STS); Vovk et al. 2021 (Retrain) |
| F14 | Simple Jumper betting martingale, f_ε(p) = 1 + ε(p − 0.5), ε ∈ {−1, 0, 1}, switching rate J = 0.01 | Vovk et al. 2021 (Retrain), Alg. 1 |
| F15 | Calibrator κp^(κ−1), κ ∈ (0,1) turns a p-value into an e-value; any decreasing f with ∫₀¹ f ≤ 1 is a calibrator | Ramdas & Wang monograph §2.3, eq. (2.1), Prop. 2.5 |
| F16 | M_t = Π((1−λ_s) + λ_s E_s) with predictable λ_s ∈ [0,1] over sequential e-values (E[E_t | F_{t−1}] ≤ 1) is an e-process; the arithmetic mean is the special case "bet a fixed amount 1/K" | monograph Def. 7.19, Prop. 7.20, Def. 7.21, Def. 8.10 |
| F17 | Base e-BH rejects the k* largest e-values, k* = max{k: k·e_[k]/K ≥ 1/α}; controls FDR at α under arbitrary dependence | Wang & Ramdas, eq. (5), Thm. 2 |
| F18 | If e-values come only from calibrating valid p-values, e-BH "will typically not be more powerful than BH" | Wang & Ramdas, Remark 1 |
| F19 | ACI update α_{t+1} = α_t + γ(α − err_t); long-run miscoverage → α for any data sequence; γ = 0.005 in their experiments | Gibbs & Candès eq. (2), Prop. 4.1, §2.1 |
| F20 | TARE metrics: a_i = min(ŷ_i/y_i, y_i/ŷ_i); A_geo = Π a_i^(c_i/Σc) with c_i = nodes_req × runtime; UnderRate = mean 1[ŷ < y]; keep runtime > 30 s | TARE eq. (2)–(6) |
| F21 | NodeSentry was evaluated on two private datasets (1,294 and 30 nodes), not public data | NodeSentry §4.1 |

### 2.2 Open questions (each is closed by a named phase)

| Q | Question | Closed in |
|---|---|---|
| Q1 | Exact PM100 column names/types, format of `nodes`, number of distinct nodes, jobs per node per day | Phase 0 |
| Q2 | Do PM100 jobs ever overlap in time on the same node? (They should not — exclusive jobs) | Phase 0 |
| Q3 | PM100 licence (check Zenodo page) | Phase 0 |
| Q4 | Can we reproduce Menear's error level with a fixed feature set? | Phase 1 |
| Q5 | Are our conformal p-values uniform on real, unmodified data (per user group)? | Phase 2 |
| Q6 | Does the node e-process stay ≤ 1/α on clean data (no false flags), despite system-wide slowdowns? | Phase 4 |
| Q7 | How many jobs (days) does it take to flag a node slowed by 10/20/30%? | Phase 4 |
| Q8 | Institute cluster: size, scheduler, permission for fault injection | before Phase 6 |
| Q9 | PM100 training-window / warm-up length (PM100 spans only 162 days, so Eagle's 100-day windows do not transfer; D15) | Phase 2 |

---

## 3. System architecture

```
                  ┌──────────────────────────── offline training (daily) ───────────────────────────┐
                  │  history (jobs ended before t) ──► quantile model q̂_0.9(x) on log runtime       │
                  └──────────────────────────────────────────────────────────────────────────────────┘
 Slurm job record (finished)
   x_j = submit features, T_j = run_time, N_j = node list, u_j = user
        │
        ▼
 ┌──────────────┐  S_j = log T_j − q̂(x_j)      ┌──────────────────────┐
 │  LAYER 1     │ ───────────────────────────►  │ Mondrian (per user)  │──► p_j  (smoothed conformal p-value)
 │  job evidence│                               │ online conformal     │──► alarm_j = 1[p_j ≤ α_job]  (ACI variant)
 └──────────────┘                               └──────────────────────┘
        │ p_j, N_j
        ▼
 ┌──────────────┐  for each node n in N_j:  M_n ← M_n · (1 + w_j (f(p_j) − 1)),  w_j = 1/|N_j|
 │  LAYER 2     │  every evaluation time:   e-BH over {M_n} at level α  ──► suspect set S
 │  node screen │
 └──────────────┘
        │ S  (only suspects)
        ▼
 ┌──────────────┐  belief π over causes; pick probe a* = argmax info-gain(a)/cost(a); update; stop at τ
 │  LAYER 3     │  probes: CPU loop · STREAM · cache test · network ping-pong · 60-s telemetry → classifier
 │  diagnosis   │  ──► report: node, cause, confidence, node-seconds used, recommended action (human approves)
 └──────────────┘
```

### 3.1 Data contracts (tables passed between stages)
All tables are parquet in `data/processed/`. Column names are fixed here; Phase 0 maps raw names onto them.

**`jobs_<system>.parquet`** (one row per job, both systems)
| column | type | notes |
|---|---|---|
| row_id | int64 | position in the raw file (0-based); **unique key within system** (D10). Downstream tables are keyed by (system, row_id) |
| job_id | int64 | as given in the source; **not unique in Eagle** (Phase 0: 6,751 ids repeat), unique in PM100 |
| system | str | `eagle` or `pm100` |
| user | str | anonymised id |
| partition, qos | str | |
| account | str or null | Eagle only |
| name | str or null | job name (Eagle has it; PM100: Phase 0 decides) |
| submit_time, start_time, end_time | timestamp | |
| run_time_s | float64 | seconds, ≥ 0 or null (loaders are lossless, D8; `> 0` and other filters are applied per phase) |
| walltime_req_s | float64 | requested limit in seconds (PM100: time_limit × 60; INFINITE → null) |
| nodes_req, procs_req, gpus_req | int64 | |
| mem_req | float64 | units UNKNOWN for both systems; used as-is, never converted; only used by tree models (scale-invariant) (D12) |
| state | str | normalised: COMPLETED, TIMEOUT, other |
| state_raw | str | state exactly as in the source file (D9) |
| censored | bool | TRUE if state == TIMEOUT (true runtime ≥ observed) |

**`job_nodes_pm100.parquet`** (one row per job × node; PM100 only)
| row_id | job_id | node_id (int64) | n_nodes (int64) | start_time | end_time |

**`layer1_scores_<system>_<run_id>.parquet`**
| system | row_id | job_id | end_time | group_key | q_hat | score | p_value | theta | alarm_fixed | alarm_aci | model_version |

**`layer2_state_<run_id>.parquet`** (one row per node per evaluation time)
| eval_time | node_id | log_M | n_jobs | n_users | flagged_ebh | flagged_<baseline> |

**`truth_<scenario_id>.parquet`** (Phase 3)
| node_id | onset_time | end_time | factor | kind (`node`, `user_decoy`, `system_shock`) |

Phase 3 extension (D30, D34): `node_id` nullable (null for user_decoy and system_shock), plus
`cause` (node kinds), `user` (user_decoy), `scenario_id`; kind also `node_partial`; the `null` kind has an empty truth table.

---

## 4. Datasets

| Dataset | Role | Get it | Size | Status |
|---|---|---|---|---|
| NREL Eagle jobs | Layer 1 development and reproduction | https://data.openei.org/submissions/5860 → `eagle_data.parquet` | 241 MB | **You have it** [V]. Licence CC-BY 4.0 [V]. Cite dataset + Menear et al. |
| PM100 (Marconi100) | Layers 1 + 2 (has node lists) | https://zenodo.org/records/10127767 → `job_table.parquet`; code https://github.com/francescoantici/PM100-data | 287 MB | Downloading. Licence: check Zenodo page (Q3) |
| M100 full (Nagios) | Weak real labels for Layer 2 (exploratory) | parent DOI 10.5281/zenodo.7588815; tools https://gitlab.com/ecs-lab/exadata | 30–45 GB per month | **Do not download yet.** Phase 4b only, 1–2 months, Nagios plugin only |
| Prodigy (Eclipse) | Layer 3 telemetry classifier | 10.5281/zenodo.8079388 | small | You have it (earlier project) |
| HPAS | Fault injection (Phase 6) | https://github.com/peaclab/HPAS (BSD-3) | code | Phase 6 |
| F-DATA (Fugaku) | — | — | — | **Not used**: no node list |

Rules [D]: raw files live in `data/raw/<dataset>/` and are never edited. `scripts/download_data.py`
records SHA-256 of every raw file in `data/raw/CHECKSUMS.txt`; every run checks them.

---

## 5. Repository layout [D]

```
HPC/                                  ← project root (open Claude Code here)
├── CLAUDE.md                         ← rules for Claude Code
├── README.md
├── pyproject.toml                    ← uv-managed; pinned versions
├── .gitignore                        ← data/, runs/, .venv/, *.parquet
├── docs/
│   ├── PLAN.md                       ← this file
│   ├── PHASE_PROMPTS.md
│   ├── data_cards/eagle.md, pm100.md ← schema, units, filters, counts (Phase 0)
│   └── decisions.md                  ← decision log (copy of §16, kept current)
├── configs/
│   ├── phase1_menear.yaml
│   ├── phase2_conformal.yaml
│   ├── phase3_inject.yaml
│   └── phase4_layer2.yaml
├── data/
│   ├── raw/eagle/eagle_data.parquet
│   ├── raw/pm100/job_table.parquet
│   ├── raw/CHECKSUMS.txt
│   ├── interim/
│   └── processed/
├── src/hpcdiag/
│   ├── io/            (loaders, schema mapping, validation)
│   ├── features/      (encoders, feature sets)
│   ├── layer1/        (runtime models, splits, conformal, aci)
│   ├── stats/         (conformal p-values, calibrators, betting, e-process, ebh, bh)
│   ├── layer2/        (node accumulation, baselines, evaluation)
│   ├── inject/        (semi-synthetic scenarios)
│   ├── layer3/        (classifier, probes, bayes, selection)   ← provisional
│   ├── eval/          (metrics, plots, tables)
│   └── utils/         (config, logging, run ids, seeds)
├── scripts/           (download_data.py, make_figures.py)
├── tests/             (pytest; one file per module)
├── notebooks/         (exploration only; never the source of reported numbers)
├── runs/<phase>/<run_id>/   (outputs; git-ignored)
└── reports/phase_<N>.md     (what was done, numbers, figures, gate result)
```

---

## 6. Environment [D]

- Machine: MacBook Air, Apple Silicon (arm64) [V: device info]. Keep the project in
  `~/Downloads/Final Year Project/HPC/` (not Desktop/Documents, which sync to iCloud and would upload GBs).
- Python 3.11 via `uv`. Core packages: pandas, pyarrow, duckdb, numpy, scipy, scikit-learn,
  xgboost, lightgbm, matplotlib, pyyaml, tqdm, pytest, ruff, hypothesis (property tests).
  PyTorch only in Phase 5.
- Memory discipline: read only needed columns; use duckdb for filtering 11M rows; use
  `category`/int codes for user/partition.
- Speed budget: a 100-day Eagle training set is ≈ 525k jobs (F5 × 100). One XGBoost fit on ~7
  features should take well under a minute; 244 splits ≈ 1–3 h. Always run a **fast mode**
  (12–24 splits) first.

---

## 7. Engineering rules [D]

1. Config-driven: no magic numbers in code; every constant is in `configs/*.yaml` with a comment citing this plan.
2. Every run writes `runs/<phase>/<run_id>/` with `config_used.yaml`, `git_commit.txt`,
   `metrics.json`, `log.txt`, figures, and the exact command.
3. Seeds: numpy, random, xgboost, lightgbm, torch. The conformal tie-breaker θ uses its own seeded generator.
4. Leakage tests (must exist and pass):
   - model for split t trained only on jobs with `end_time < t`;
   - conformal calibration for job j uses only jobs with `end_time < end_time_j`;
   - Layer 2 state at time t uses only jobs ended ≤ t.
5. Statistical unit tests (must exist and pass before real data):
   - conformal p-values are uniform on IID simulated scores (KS test p > 0.01 over 10⁴ draws, several seeds);
   - calibrator integrates to 1 numerically (|∫ − 1| < 1e−6);
   - e-process stays a martingale under the null: one-step fairness E[M_t | past] = M_{t−1} checked exactly by quadrature through the implemented bets (the Monte-Carlo mean test is used only on a finite-variance bet: for κ ≤ 0.5 the calibrator has infinite variance; confirmed by Mahesh 10 Oct 2026), and P(max M ≥ 1/α) ≤ α + tolerance;
   - e-BH matches a hand-computed example and the brute-force definition in eq. (F17);
   - e-BH on simulated null e-values gives empirical FDR ≤ α.
6. Reports are generated from `metrics.json` files by a script, not typed by hand.

---

## 8. Security and safety rules [D]

- No secrets in the repo. No credentials needed for public datasets.
- Data are anonymised by their publishers [V: F9]; do not try to de-anonymise; do not join with other sources.
- Project code makes no network calls except the download script (HTTPS, checksum-verified).
- Downloaded archives are untrusted: extract into their own empty folder; never run code from them.
- The system **recommends**; it never drains, reboots or modifies a real node. Any cluster action
  (Phase 6) needs written admin approval, runs only on dedicated nodes, and uses the HPAS
  generators with time limits.
- Pin dependency versions; record `uv.lock`.
- Licences: cite Eagle (CC-BY 4.0), PM100/M100 (per Zenodo), HPAS (BSD-3), Prodigy (BSD-3).

---

## 9. Phases

Estimated calendar (adjust after Phase 1): Phase 0–1 Oct · Phase 2 Nov · Phase 3–4 Nov–Dec ·
Phase 5 Jan · Phase 6 Jan–Feb (cluster permitting) · Phase 7 Feb–Mar.

---

### Phase 0 — Setup and data check

**Goal.** Working repo, environment, verified schemas for both datasets, closed questions Q1–Q3.

**Tasks.**
1. Create the layout in §5; `git init`; `.gitignore`; `uv init`; pin packages (§6).
2. Move `eagle_data.parquet` → `data/raw/eagle/` (ask the human first); put PM100 in `data/raw/pm100/`.
3. `scripts/download_data.py`: computes and stores SHA-256 for files already present (no download needed now).
4. `src/hpcdiag/io/`: loaders that map raw columns to the contract in §3.1. Units documented.
5. Data cards `docs/data_cards/{eagle,pm100}.md` with: schema; row counts; date range; state counts;
   units; missing values; runtime quantiles; per-user job counts.
6. PM100 specifics: parse `nodes` into `job_nodes_pm100.parquet`; count distinct nodes; jobs per
   node per day (median, 10th, 90th percentile); check overlaps of jobs on the same node (Q2);
   fraction of multi-node jobs; `time_limit` conversion; record licence (Q3).

**Gate 0.** Both loaders pass schema tests; data cards exist; Eagle numbers match F1–F5 exactly;
Q1–Q3 answered in `reports/phase_0.md`.

---

### Phase 1 — Layer 1 baseline: reproduce Menear et al. on Eagle

**Goal.** A trustworthy runtime model and evaluation harness, checked against a published number.

**Method (from F6).**
- Filter: keep COMPLETED and TIMEOUT; drop all other states [V: F6 says cancelled and failed are
  dropped and timeouts kept; OUT_OF_MEMORY, NODE_FAIL, PENDING, RUNNING are also dropped — **D**].
  Drop rows with missing wallclock_req [D].
- Target: `run_time` in seconds. Report RMSE in **hours** (to compare with 8.14 h).
- Features (fixed set, **D**): numerical = wallclock_req, processors_req, mem_req, nodes_req, gpus_req;
  categorical = user, partition (label-encoded; codes fitted on the training window; unseen → −1).
- Splits: split times every 6 days across the data (first split ≥ 100 days after start);
  train = jobs with `end_time` in [t − 100 d, t); test = jobs with `submit_time` in [t, t + 1 d).
- Model: `xgboost.XGBRegressor()` defaults (record the version; the paper used 1.7.1). Save the full
  XGBoost configuration actually used (`model.get_booster().save_config()`) in the run folder.
- Duplicate rows (D10): the main run keeps all rows, including Eagle rows on repeated `job_id`.
  A sensitivity variant drops exact duplicate rows (identical in all raw columns) before the
  filters; both are reported.
- Also run: the "similar jobs" and "recent jobs" heuristics only if time permits (optional).

**Outputs.** Per-split metrics (RMSE h, MAE h, TARE A_geo, UnderRate), error by actual-runtime
bins (0–1 h, 1–6 h, 6–24 h, 1–2 d, > 2 d), plot of predicted vs actual by bin.

**Gate 1.** Overall RMSE over all splits is within ±20% of 8.14 h (i.e. 6.5–9.8 h) **and** the
qualitative pattern holds: underestimation grows for jobs > 2 days [V: Menear §3.6]. If outside the
range: report it, check filters/units, do not tune on test splits. If Gate 1 fails, the first
sensitivity check is a rerun with `tree_method="approx"` (xgboost 3.2.0 vs the paper's 1.7.1 may
differ in the default tree method), before changing anything else.

---

### Phase 2 — Layer 1: quantile model + conformal p-values + ACI

**Goal.** For every finished job, a p-value that is uniform when the job is "normal", and a job
alarm whose long-run false-alarm rate is α.

**2.1 Quantile model [D].** LightGBM with `objective="quantile"`, `alpha=0.9`, on y = log(max(run_time,1)).
Same features and split protocol as Phase 1. Also fit τ = 0.5 (median) for reporting.
Hyperparameters (D20): UARP settings — 200 iterations, learning rate 0.05, 50 leaves [V: F10].
Scoring population (D17): jobs with run_time_s > 30 (TARE convention) for the main results; all
jobs as a reported variant. The population filter applies to training, calibration and test rows.

**2.2 Nonconformity score (one-sided CQR, F11).** S_j = log T_j − q̂_0.9(x_j). Large S = late.
TIMEOUT jobs are right-censored: their S is a lower bound, which can only make their p-value larger
(conservative) [A — check the fraction of TIMEOUT jobs among alarms].

**2.3 Two p-value modes (both implemented, compared) [D].**
- **Mode E (exact).** Train the quantile model **once** on a 100-day prefix (PM100: the chosen
  prefix L, 2.7), then freeze it. Jobs are processed in (end_time, row_id) order (D14 convention;
  jobs with equal end_time are ordered by row_id).
  For each group g (user), process that user's later jobs in end-time order and compute the
  smoothed online conformal p-value (F12):
  p_n = ( |{i ≤ n : S_i > S_n}| + θ_n · |{i ≤ n : S_i = S_n}| ) / n,  i over the group's jobs so far including n.
  Under exchangeability within the group these are IID uniform [V: F12] — this is what Layer 2's
  product needs.
  Mode E grouping (D19): every job updates three streams (its user, its partition, global). Default
  = fallback: the reported p-value comes from the user stream if the user has ≥ n_min = 30 earlier
  post-freeze jobs, else the partition stream if it has ≥ 30, else the global stream. Each stream
  alone is exactly valid (F12); the sequential independence of the reported p-values across a
  switch between streams is an assumption **[A — tested in Gate 4a]**. Variant = pure per-user
  streams (no fallback), exactly IID under within-user exchangeability.
- **Mode A (adaptive).** Retrain at each split time t (Eagle: the 244 Phase 1 split times, evaluated
  on the same non-empty test days; PM100: daily). Exact windows (addendum B, 9 Oct 2026):
  **train on end_time ∈ [t − 114 d, t − 14 d), calibrate on end_time ∈ [t − 14 d, t), test on
  submit_time ∈ [t, t + 1 d)** (PM100: train on [t − L − 14 d, t − 14 d), L from 2.7). Training and
  calibration rows are disjoint; every calibration job ends before every test job is submitted.
  Calibration uses the group's jobs in the calibration window, re-scored with the current model:
  p_j = (1 + |{i ∈ C_g : S_i ≥ S_j}|) / (|C_g| + 1). Valid marginally under exchangeability with C_g;
  dependence across jobs is not exact [A].
- Group fallback [D]: if a user has fewer than n_min = 30 calibration jobs, use partition; then global.
  Record which level was used (`group_key`).

**2.4 Job alarm and ACI (F19).** Fixed rule: alarm if p_j ≤ α_job (α_job = 0.05).
ACI rule (per group): alarm if p_j ≤ α_t; after each job, α_{t+1} = α_t + γ(α_job − err_t),
err_t = alarm indicator; γ = 0.005 [V: Gibbs & Candès' choice] plus a sensitivity check γ ∈ {0.001, 0.01}.

**2.5 Baselines.**
- B-guide: relative residual r = (T − T̂)/T̂ from Phase 1 XGBoost; alarm if r > 0.20 [V: hpc_project.pdf].
- B-UARP: alarm if T > Q0.99 + max(0.2·Q0.99, 0.5·σ) [V: F10].
- B-user: alarm if T > wallclock_req (trivially rare; reported for context).

**2.6 Metrics.** On held-out, unmodified data (assumed mostly healthy):
false-alarm rate vs α (overall, per runtime bin, per top-decile resource use);
p-value histogram + KS distance to uniform (overall and per group);
coverage and width of the upper bound; TARE A_geo and UnderRate for the τ=0.5 model.
"Top-decile resource use" key = nodes_req × walltime_req_s (known at submission); this differs
from TARE's weight c_i = nodes_req × actual run time. Also: share of alarms that are TIMEOUT
(censored) jobs, and false-alarm rates in the 1–2 d and > 2 d bins reported separately.
Mode E is reported on (i) all post-freeze jobs and (ii) the Mode A test days.

**2.8 Mode P — pooled sliding-window conformal (D22, PROMPT 2B).** Scores = the Mode E frozen-model
scores S_j (no refit). Reference R_j = all jobs of the same population that come before j in D14
order (end_time, then row_id) with end_time ≥ end_time_j − W; no user/partition grouping.
p_j = (#{i ∈ R_j : S_i > S_j} + θ_j (1 + #{i ∈ R_j : S_i = S_j})) / (|R_j| + 1), θ_j from the
theta_seed generator; |R_j| < 30 → NaN (warm-up), excluded from evaluation and from Layer 2.
Main W = 7 d; W ∈ {6 h, 1 d, 365 d} and a q̂-decile Mondrian variant are reported (disclosure: the
architect chose W = 7 d after looking at these on PM100 fast Mode E scores; Eagle not used).
Validity [A]: exchangeability of S within the window plus job j, checked by Gate 2-L2.
Known cost of pooling: a user whose jobs are all slow gives small p-values on every node that user
runs on — handled in Layer 2 (PROMPT 4 addendum).
Mode A uses the smoothed split p-value p_j = (#{S_i > S_j} + θ_j (1 + #{S_i = S_j})) / (n + 1) (D23).
Tail checks for every mode (D24): P(p ≤ 1e-3), P(p ≤ 1e-4), and mean f(p) with f = average over
κ ∈ {0.05, 0.1, 0.25, 0.5} of κ p^(κ−1), compared with its null from 300 simulated U(0,1) samples of
the same n. ACI for gating = one global stream in D14 order; per-user ACI descriptive only (D25).

**Gate 2-L2 (D26; input to Layer 2).** PM100 main, Mode P, W = 7 d: KS ≤ 0.02; FA within α ± 0.01
at α ∈ {0.01, 0.05, 0.10}; P(p ≤ 1e-3) ≤ 0.002; mean f(p) ≤ simulated-null 95th percentile;
global-stream ACI rate within α ± 0.005. Also on PM100 test days ≥ 90 (D21); the same table is
reported for Eagle main (Eagle does not feed Layer 2). Decile tables (q̂, walltime_req_s) reported,
not gated. If PM100 Mode P fails Gate 2-L2: STOP and report; no further windows.
**Gate 2-L2 result (D27, recorded exactly as measured, not relaxed):** PM100 main PASS; D21 subset
(test days ≥ 90) FAIL on KS (0.0232 > 0.02), all other D26 checks pass on that subset; Eagle FAIL
(extreme tail). Mode P (pooled, W = 7 d, main population) is the Layer-2 input (D28).

**Gate 2 (original; kept and reported, D26).** Checked separately on Eagle and PM100, main population (D17). Mode E (fallback
grouping, D19) on all post-freeze jobs: KS distance of pooled p-values to uniform ≤ 0.02, and
empirical false-alarm rate within α ± 0.01 at α ∈ {0.01, 0.05, 0.10}. ACI on Mode A p-values,
γ = 0.005, one global stream in D14 order over the test-day jobs (D25; per-user ACI is descriptive): long-run alarm rate within
α ± 0.005 over the test period. PM100 sensitivity (D21): Gate 2 is also reported restricted to
test days ≥ 90 (days 74–89 were used for window selection, D18); L = 45 is not changed by it. If Mode E fails, report per-group drift and switch the
default to Mode A only with written reasons (Decision Log).

**Note.** Eagle has no node list, so Layer 1 is first built and validated on Eagle,
then re-run unchanged on PM100 to feed Layer 2 (Gate 2 is checked on both).

**2.7 PM100 windows (D15, closes Q9).** PM100 spans only 162 days, so Eagle's 100-day windows do
not transfer. The PM100 training prefix / warm-up is chosen by comparing training prefixes of
30, 45 and 60 days, evaluated on the **first 90 days only**; the rest of PM100 is not looked at
for this choice. The chosen value is recorded in the Decision Log.
Rule (D18): all three L share the same test days, days 74–89 after the earliest submit time
(test windows [start + 74 d + k, start + 75 d + k), k = 0..15); each L uses the Mode A protocol
retrained daily (train on [t − L − 14 d, t − 14 d)); choose the L with the lowest pooled pinball
loss at τ = 0.9 on log runtime; if the pinball losses are within 1% of each other, choose the
shorter window. The rule never looks at p-values or at anything after day 90.

---

### Phase 3 — Ground truth: semi-synthetic injection on PM100

**Goal.** Exact, known labels on a real 980-node workload, so Layer 2 can be measured.

**Scenarios [D].**
- **Node degradation.** Pick K nodes (K = 1%, 2%, 5% of active nodes), onset time t₀ (random, after
  the warm-up period), factor s ∈ {1.1, 1.2, 1.3, 1.5}. Any job using a degraded node and starting
  after t₀ gets T' = s · T [A: straggler model — a tightly coupled job runs at its slowest node's
  speed]. Variant "partial": T' = T · (1 + (s−1)·φ), φ = fraction of the job's time after t₀.
- **User decoy (workload variation).** Pick a user; all their jobs after t₀ get T' = s · T on every
  node. Correct behaviour: **no node is flagged** (this directly tests the guide's Objective 2).
- **System shock.** All jobs ending in a time window [t₁, t₁ + 6 h] get T' = s · T. Correct
  behaviour: no node is flagged.
- **Censoring.** If T' exceeds the job's time limit, set T' = limit and `censored = TRUE`.
- Each scenario has ≥ 20 random seeds.
- Onsets t₀ (and shock windows t₁) must fall after the PM100 warm-up chosen in Phase 2 (D15).
- Phase 3 rules (D29–D35, 10 Oct 2026): two bases — random placement (primary, exact truth) and real
  placement (secondary); per-node `cause` ∈ {cpu, cache, mbw}; onsets uniform on [Mode E freeze + 7 d,
  trace end − 14 d], independent per degraded node, redrawn if inside a listed submission gap;
  degradation lasts to the trace end; only COMPLETED/TIMEOUT rows can be affected; the run_time > 30 s
  main-population filter is applied AFTER injection (Phase 4 re-filters); a "null" kind (no injection)
  exists for every base; scenario seed k uses random-placement base k.

**Gate 3.** Unit tests show: unaffected jobs are byte-identical; affected jobs follow the formula;
truth tables match the injected set exactly.

---

### Phase 4 — Layer 2: node e-processes and e-BH

**4.1 Per-node evidence (F15, F16).** Each node n starts with M_n = 1. When job j ends, for every node
n ∈ N_j:

  M_n ← M_n · (1 + w_j · (f(p_j) − 1)),  w_j = 1 / |N_j|.

- f is a betting function with ∫₀¹ f = 1, decreasing (we only bet on lateness). Default [D]:
  mixture over κ ∈ {0.05, 0.1, 0.25, 0.5} of the power calibrator f_κ(p) = κ p^(κ−1), implemented as
  an average of the four e-processes (an average of e-processes is an e-process).
- Re-adaptation [D]: a one-sided Simple-Jumper-style mixture (F14) that switches between "bet" and
  "no bet" with J = 0.01, so a node that was healthy for months can still be flagged quickly
  after it degrades. Compare against the plain product (ablation).
- w_j is known when the job starts (allocation size), so it is predictable [V: F16 requires λ_s
  predictable].
- Validity condition [A, tested in Gate 4a]: under "node n is healthy", the p-values of jobs on n
  are sequentially valid, E[f(p_j) | past] ≤ 1. This holds if p-values are uniform given the past
  (Mode E) and node placement does not depend on lateness.
- **Honest validity statement (D72, D76).** The guarantee rests on p-values that are valid given
  the past. Two violations are known:
  (1) Straggler contamination (D72): the per-node null "p-values of jobs on n are super-uniform given
  the past" is violated for a healthy node that shares jobs with a degraded node, because the shared
  job is late on account of the other node. With degraded nodes present, FDR is controlled
  empirically, not proved.
  (2) Temporal clustering (D76, a result of Phase 4b): real Mode P p-values are marginally but not
  conditionally calibrated. Late jobs come in bursts (autocorrelation of 1{p < 0.1} = 0.29 at
  lag 1), so even under the global null the condition E[f(p_j) | past] ≤ 1 fails on the real trace.
  The bound of 4.3 therefore holds for p-values that meet the condition, not as a theorem about real
  Mode P p-values: on the real trace Gate 4a is an empirical result. v1 passes Gate 4a partly because
  w_j = 1/|N_j| dilutes bursts. See reports/phase_4b.md.
- **Predictable blame allocation, "explaining away" (D71) — REJECTED (D75).** Tested as v2 in Phase
  4b: replace w_j = 1/|N_j| by λ_{j,n} = M_n⁻ / Σ_{k ∈ N_j} M_k⁻, with M_k⁻ node k's total jumper
  capital just before job j (D14 order; jobs with the same end time use the state before that tie
  block). It is predictable, and valid on iid p-values (null P(ever) 0.035), but it raises null
  P(ever) to 0.300 ± 0.032 on the real trace. v1 (lr_Sc_a, w_j = 1/|N_j|) remains the Layer-2
  configuration. No λ caps, batching or other tuning of D71 (D75).
- Work in log space: log M (avoid overflow).
- Per-node job order (D14): jobs are applied to M_n in `end_time` order, ties broken by `row_id`.
  Same-node overlaps ≤ 60 s (Phase 0: max 19 s) are treated as non-overlapping. The PM100
  `shared` column has unknown semantics and is not used.

**4.2 System-shock guard [D].** A filesystem slowdown makes all jobs late at once, which would raise
every node's M together. Two options, both implemented and compared:
(a) detrend: S'_j = S_j − median(S over all jobs that ended in the 6 h before end_time_j);
(b) global monitor: a conformal martingale on all jobs; while it signals a system-wide shift,
freeze Layer 2 updates. Default chosen by Gate 4a.

**4.3 Selection (F17).** At each evaluation time (every 6 h, **D**): among m active nodes, base
e-BH at level α = 0.1: flag the k* nodes with largest M, k* = max{k : k·M_[k]/m ≥ 1/α}.
After a node is flagged and handed to Layer 3, its M is reset to 1 once it is cleared or repaired.
m is FIXED at the number of nodes (PM100: all 980 nodes) at every evaluation; nodes with no scored
job yet have M = 1 (D50 — a data-dependent m breaks the proof below).

**Why P(any node ever flagged) ≤ α under the global null (D49).** Let τ be the first evaluation
time at which e-BH rejects at least one node (τ = ∞ if never) and T the last evaluation. τ ∧ T is a
stopping time for the filtration generated by the jobs ended up to each evaluation. Each M_n is an
e-process for H_n, so the stopped value M_n(τ ∧ T) is an e-value (F16; optional stopping). e-BH
applied to the e-values M_1(τ ∧ T), …, M_m(τ ∧ T) controls FDR ≤ α under arbitrary dependence
(F17, Wang & Ramdas Thm. 2). Under the global null every rejection is false, so FDR equals
P(e-BH at τ ∧ T rejects anything). If τ ≤ T, e-BH at τ rejects by definition; if τ = ∞, it rejects
nothing at T. Hence P(τ ≤ T) = FDR(τ ∧ T) ≤ α. This holds for any dependence between nodes and any
number of evaluation times, provided m is fixed (D50) and every update is predictable (guard (b)
freezes are decided from jobs that ended before j, D51). The premise "each M_n is an e-process"
needs the validity condition of 4.1: straggler contamination (D72) and temporal clustering of real
Mode P p-values (D76) both break it, so on real data the bound is checked empirically by Gate 4a.

**4.4 Baselines.**
- B1 guide heuristic: median relative residual > 20%, ≥ 10 jobs, ≥ 3 users, ≥ 3 distinct job
  names, persistent for 3 consecutive windows [V: hpc_project.pdf §10.5]. **On PM100** (no job
  name, D13): median relative residual > 20%, ≥ 10 jobs, ≥ 3 distinct users, persistent for 3
  consecutive windows; the "≥ 3 applications" condition is dropped (limitation, §12).
- B2 top-5%: flag the top 5% of nodes by mean residual each window.
- B3 BH on per-node Fisher-combined p-values (assumes independence).
- B4 averaged e-values (our earlier design; expected to be nearly powerless — included to show why).
- B5 per-node Ville threshold M_n ≥ 1/α without e-BH (no multiplicity control).

**4.5 Metrics.** Over seeds: empirical FDR (mean of false flags / max(flags, 1)), recall
(degraded nodes flagged), detection delay (hours and jobs after onset), false flags in decoy and
shock scenarios, number of flags per day on clean data.

**Gate 4a (validity, clean data) — binding (D27).** If Gate 4a fails on the random-placement base
with Mode P p-values, we stop and return to Layer 1. On unmodified PM100 and on decoy/shock scenarios:
empirical FDR ≤ α (+ 2 standard errors) and mean false flags per day reported; B2 must show
its fixed false-flag rate (sanity check).
**Gate 4b (power).** At s = 1.3 and K = 1%: recall reported with detection delay; target ≥ 0.8
recall within 14 days [D — adjust only with a written reason]. Report the s at which recall drops
below 0.5.

**4.6 Exploratory (optional) — real labels.** Download Nagios for 1–2 months of M100. Because
flagged nodes are taken offline [V: F9], test whether M_n rises in the days **before** a Nagios
critical event. Report as exploratory; not a primary result.

---

### Phase 5 — Layer 3 offline (provisional — finalise after Layer 3 papers)

- 5.1 Telemetry classifier: Prodigy data, three classes (healthy / cpuoccupy / memleak);
  Random Forest (default), Triplet and gated-Triplet as comparisons; grouped CV by job (as in the
  existing project). Output: class probabilities = likelihood for the "telemetry" probe.
- 5.2 Probe-selection engine: belief π over causes H; probes A with cost c_a; outcome model
  P(o | h, a) (discretised outcomes, e.g. benchmark score ratio vs healthy baseline in bins);
  choose a* = argmax_a [H(π) − E_o H(π | o, a)] / c_a; Bayes update; stop when max π ≥ τ or budget spent.
  Tests on toy tables where the right first probe is known by hand.

**Gate 5.** Classifier results reproduced with grouped CV; selection engine passes the toy tests;
on simulated likelihood tables, the engine reaches the same accuracy as "run all probes" with fewer
probes (report the number).

---

### Phase 6 — Real cluster and HPAS (needs admin approval)

Provisional. Tasks: install HPAS on 2–4 dedicated nodes; write probes (fixed-work CPU loop, STREAM,
cache test, OSU ping-pong against a healthy node, 60-s telemetry capture); run healthy baselines;
run a factorial campaign (anomaly type × intensity × repetitions) with exact start/stop logs;
estimate P(o | h, a); end-to-end run: submit benchmark jobs, inject at random times, measure
time-to-flag, diagnosis accuracy and node-seconds spent vs a fixed battery.

**Gate 6.** End-to-end run completes; every number traced to logs; admin sign-off recorded.

---

### Phase 7 — Paper, release, final report

- Ablations: no weighting (w=1); no conformal (raw residual + threshold); averaging vs product;
  no shock guard; Mode A vs Mode E; jumper vs plain product.
- Figures: p-value histograms; false-alarm vs α; FDR vs α; recall and delay vs s; probe cost.
- Release: code (open licence), the semi-synthetic scenario generator, scenario seeds, and (if
  allowed) cluster injection logs.
- Venue choice with the guide (see §13).

---

## 10. Metric definitions (central)

| Metric | Definition |
|---|---|
| RMSE (h) | sqrt(mean((ŷ − y)²)) / 3600 |
| A_geo | Π a_i^(c_i/Σ_j c_j), a_i = min(ŷ_i/y_i, y_i/ŷ_i), c_i = nodes_req × run_time [V: F20] |
| UnderRate | mean 1[ŷ_i < y_i] [V: F20] |
| False-alarm rate (job) | fraction of jobs with alarm, on data with no injected change |
| KS distance | sup_u |F̂_p(u) − u| over pooled p-values |
| FDP / FDR | false flags / max(flags,1); FDR = its mean over seeds and evaluation times |
| Recall | degraded nodes flagged at least once after onset / degraded nodes |
| Detection delay | time (h) and jobs from onset to first flag |
| Probe cost | node-seconds spent by Layer 3 per diagnosis |

---

## 11. Experiment tracking

Every run appends one line to `runs/registry.csv`: run_id, phase, config hash, git commit, start,
end, status, key metrics. Reports cite run ids.

---

## 12. Threats to validity and mitigations

| Threat | Mitigation |
|---|---|
| Workload drift breaks exchangeability | Mode A, ACI, short calibration windows; report measured (not just theoretical) error rates |
| System-wide slowdowns inflate all nodes | §4.2 guard; shock scenarios in Phase 3 |
| Node placement correlated with lateness (e.g. certain users on certain racks) | Mondrian by user; check per-node p-value uniformity on clean data (Gate 4a) |
| Straggler contamination: a healthy node sharing jobs with a degraded node sees late jobs, so its per-node null is violated (D72) | Guarantee not claimed with degraded nodes present; co-allocated false flags measured separately. D71 blame allocation did not reduce them on the development pilot and is rejected (D75) |
| Real Mode P p-values are marginally but not conditionally calibrated: temporal clustering (autocorrelation of 1{p < 0.1} = 0.29, 0.20, 0.06 at lags 1, 10, 100) breaks the condition behind the e-process guarantee, even under the global null (D76) | Gate 4a on the real p-value trace is the evidence, stated as empirical; v1 passes it partly because w_j = 1/\|N_j\| dilutes bursts, so designs that concentrate bets must be re-checked on real nulls (D71: 0.300 ± 0.032 vs 0.035 on iid p-values). Details in reports/phase_4b.md |
| Injection model too simple (straggler assumption) | "partial" variant; Phase 6 real injections |
| PM100 contains only exclusive jobs | state as limitation; shared-node jobs future work |
| PM100 has no job name, so baseline B1 drops the "≥ 3 applications" condition (D13) | state as limitation; B1 on PM100 is weaker than the guide's rule as written |
| Eagle `job_id` repeats (rows on 6,751 ids) | `row_id` key (D10); Phase 1 sensitivity run without exact duplicate rows |
| Mode P window W = 7 d was chosen on PM100 itself and the pass is narrow (W = 1 d and W = 365 d fail Gate 2-L2) (D28) | Gate 4a binding (D27); Gate 4a also run with W = 1 d, 365 d and Mode E as sensitivity/ablation (PROMPT 4 addendum B6) |
| Mode P calibration is marginal only: decile tables show poor conditional calibration (PM100 Mode P W = 7 d: q̂ decile 10 FA 0.004, walltime decile 10 FA 0.123 at α = 0.05) — a risk under real node placement (D28) | Report conditional tables; random-placement base is primary for Gate 4a; real-placement flags labelled "unknown truth" |
| Eagle's extreme tail fails Gate 2-L2 (D28) | No claim of generality across systems |
| Nagios labels mark offline nodes, not slow nodes | used only as exploratory precursor analysis |
| e-BH conservative (F18) | report power honestly; compare with BH (B3) |
| Small cluster for Phase 6 | primary results from PM100 semi-synthetic; Phase 6 as demonstration |

---

## 13. Paper plan

Working title: *Anytime-Valid Detection of Degraded Compute Nodes from Scheduler Logs with
Conformal E-Processes* (Layers 1–2), with Layer 3 as an extension if Phase 6 succeeds.
Target order: journal (J. Supercomputing / FGCS) or an IPDPS/SC workshop or HiPC; Euro-Par/ICPP
only with full three-layer results. Decide with the guide after Gate 4b.

---

## 14. Glossary (plain English)

- **Quantile model**: predicts "90% of jobs like this finish within X".
- **Conformal p-value**: "what fraction of this user's normal jobs were at least this late".
- **Exchangeable**: the order of past jobs carries no information (like shuffled cards).
- **E-value**: a bet against "nothing is wrong"; worth at most 1 on average when nothing is wrong.
- **E-process**: the running total of such bets over time; you can check it whenever you like.
- **FDR**: among the nodes we flag, the average fraction that are actually healthy.
- **e-BH**: the rule that picks which nodes to flag so that FDR ≤ α.
- **ACI**: a thermostat that nudges the alarm threshold so the long-run alarm rate stays at α.

---

## 15. References (we hold all of these)

1. Menear et al., Mastering HPC Runtime Prediction, PEARC'23, 10.1145/3569951.3593598
2. Choi & Oh, UARP, J. Supercomputing 2026, 10.1007/s11227-026-08422-8
3. Xiao et al., TARE, ICPP'26, arXiv 2607.04935
4. Romano, Patterson, Candès, Conformalized Quantile Regression, NeurIPS 2019, arXiv 1905.03222
5. Angelopoulos & Bates, Conformal Prediction: A Gentle Introduction, arXiv 2107.07511
6. Gibbs & Candès, Adaptive Conformal Inference Under Distribution Shift, NeurIPS 2021, arXiv 2106.00170
7. Fan, Rich, Allcock, Papka, Lan, Trade-off between Prediction Accuracy and Underestimation Rate in Job Runtime Estimates, IEEE CLUSTER 2017 (DOI not printed in our copy — confirm on IEEE Xplore before citing)
8. Wang & Ramdas, False Discovery Rate Control with E-values, JRSSB 2022, 10.1111/rssb.12489
9. Ramdas & Wang, Hypothesis Testing with E-values, 2025, 10.1561/3600000002
10. Vovk, Testing Randomness Online, Statistical Science 2021, 10.1214/20-STS817
11. Vovk et al., Retrain or not retrain: conformal test martingales, COPA 2021, arXiv 2102.10439
12. Borghesi et al., M100 ExaData, Scientific Data 2023, 10.1038/s41597-023-02174-3
13. Antici et al., PM100, SC-W 2023, 10.1145/3624062.3624263
14. Molan et al., RUAD, arXiv 2208.13169
15. Xia et al., NodeSentry, SC'25, 10.1145/3712285.3759794
16. Duplyakin & Menear, NREL Eagle supercomputer jobs (dataset), OEDI 5860, 2023
17. Layer 3 (pending): Tuncer TPDS 2019 (10.1109/TPDS.2018.2870403); Ates HPAS ICPP 2019
    (10.1145/3337821.3337907); Rish et al. TNN 2005 (10.1109/TNN.2005.853423); Aksar TPDS 2024
    (10.1109/TPDS.2024.3365462); Prodigy SC'23 (10.1145/3581784.3607076)

---

## 16. Decision log (initial)

| # | Date | Decision | Reason |
|---|---|---|---|
| D1 | 9 Oct 2026 | Layer 1 developed on Eagle; Layer 2 on PM100 | Eagle has no node list (F2); PM100 has `nodes` (F8) |
| D2 | 9 Oct 2026 | Per-node evidence is a **product** e-process, not an average | Average is a fixed-amount bet (F16) and cannot reach e-BH thresholds with ~1000 nodes |
| D3 | 9 Oct 2026 | Two conformal modes (E exact, A adaptive) | Exact sequential validity (F12) vs drift robustness |
| D4 | 9 Oct 2026 | Nagios used only for exploratory precursor analysis | Flagged nodes are offline (F9) |
| D5 | 9 Oct 2026 | Keep COMPLETED + TIMEOUT; drop other states | Follows F6 for cancelled/failed; others not runtime-meaningful |
| D6 | 9 Oct 2026 | Fixed Phase 1 feature set (no per-split search) | Cost; the paper's search needed 50,752 fits (F6) |
| D7 | 9 Oct 2026 | Mondrian group = user, fallback partition → global, n_min = 30 | User is the most predictive feature (F6); groups need enough history |
| D8 | 9 Oct 2026 | Loaders are lossless; filtering happens per phase | Every raw row is kept in `data/processed/` (null/0 allowed, negatives rejected); each phase applies and documents its own filters |
| D9 | 9 Oct 2026 | `state_raw` kept alongside `state` | Normalised `state` (COMPLETED/TIMEOUT/other) loses the original state; data cards and later filters need it |
| D10 | 9 Oct 2026 | Contract column `row_id` (int64, position in raw file) is the unique key for both systems; `job_id` kept as given; Layer-1/Layer-2 tables keyed by (system, row_id). Phase 1 main run keeps duplicate rows; a sensitivity variant drops exact duplicate rows; both reported | Eagle `job_id` repeats: 6,751 ids on 371,160 rows, 105,510 exact copies (Phase 0) |
| D11 | 9 Oct 2026 | F7 corrected: Zenodo description says 231,116 jobs; the md5-verified file has 231,238 rows, May 5 – Oct 13 2020 (162 days). We use the file | Phase 0 row count; file MD5 equals the published one |
| D12 | 9 Oct 2026 | `mem_req` units unknown for both systems: values used as-is, never converted; only used by tree models (scale-invariant) | Units not stated in sources we hold (Phase 0) |
| D13 | 9 Oct 2026 | Baseline B1 on PM100 = median relative residual > 20%, ≥ 10 jobs, ≥ 3 distinct users, persistent for 3 consecutive windows; "≥ 3 applications" dropped and recorded as a limitation (§12) | PM100 has no job-name field; partition/qos fallback has too few values (Phase 0) |
| D14 | 9 Oct 2026 | PM100 same-node overlaps ≤ 60 s treated as non-overlapping; per-node job order = `end_time`, ties broken by `row_id`; `shared` column semantics UNKNOWN, not used | Phase 0: 2,763 overlaps, max 19 s, none > 60 s |
| D15 | 9 Oct 2026 | PM100 windows chosen in Phase 2 by comparing training prefixes of 30/45/60 days on the first 90 days only; Phase 3 onsets after the chosen warm-up (open question Q9) | PM100 spans only 162 days; Eagle's 100-day windows do not transfer |
| D16 | 9 Oct 2026 | Not used — the Phase 1 pre-registered fallback ("harness validated, exact reproduction not achieved") was not triggered: Gate 1 passed (pooled RMSE 7.380 h) | `reports/phase_1.md` |
| D17 | 9 Oct 2026 | Layer-1 conformal scoring population = jobs with run_time_s > 30 (TARE convention) for main results; all jobs reported as a variant; the filter applies to training, calibration and test rows | TARE convention: keep runtime > 30 s (F20); addendum A to PROMPT 2 |
| D18 | 9 Oct 2026 | PM100 window selection rule (Phase 2.7): shared test days 74–89, Mode A daily, lowest pooled pinball loss at τ = 0.9 on log runtime, ties within 1% → shorter window; never looks at p-values or anything after day 90 | PM100 spans 162 days (D15); rule fixed before running |
| D19 | 9 Oct 2026 | Mode E grouping: fallback user → partition → global (n_min = 30) as default; pure per-user streams as a variant; fallback cross-switch independence is [A], tested in Gate 4a | Power for new users vs exact IID validity |
| D20 | 9 Oct 2026 | Quantile models use UARP settings (200 iterations, lr 0.05, 50 leaves); B-UARP on every split | Cited settings (F10); approved compute (~80 min Eagle) |
| D21 | 9 Oct 2026 | PM100 Gate 2 also reported restricted to test days ≥ 90 (Mode E: jobs submitted on the Mode A test days, all ≥ day 90; Mode A test days are ≥ 90 by construction): sensitivity check, L = 45 unchanged | Days 74–89 were used for window selection (D18) |
| D22 | 10 Oct 2026 | Mode P (pooled sliding-window conformal on Mode E scores, W = 7 d main; 6 h / 1 d / 365 d and q̂-decile reported) is the candidate Layer-1 input to Layer 2; warm-up |R_j| < 30 → NaN | Fast-run evidence that per-user streams cause most over-alarming; W = 7 d chosen by the architect on PM100 fast scores (disclosed) |
| D23 | 10 Oct 2026 | Mode A uses the smoothed (tie-broken) split-conformal p-value | Unsmoothed p-values with ties are non-uniform (PM100 Mode A "all" fast KS 0.30) |
| D24 | 10 Oct 2026 | Tail checks for every mode: P(p ≤ 1e-3), P(p ≤ 1e-4), mean mixture f(p) vs a 300-sample simulated null | Layer 2 bets with κ down to 0.05; FA at α ≥ 0.01 cannot certify the tail |
| D25 | 10 Oct 2026 | ACI for the gate = one global stream (D14 order); per-user ACI descriptive only | The per-stream bound (max(α₁,1−α₁)+γ)/(γT) gives no guarantee for short per-user streams (bound as recalled by the architect — paper not in project, needs check) |
| D26 | 10 Oct 2026 | Gate 2 revised: original Gate 2 kept and reported; Gate 2-L2 (Mode P, W = 7 d, PM100 main; also test days ≥ 90; Eagle reported) gates the Layer-1 input to Layer 2 | Approved by Mahesh in PROMPT 2B |
| D27 | 10 Oct 2026 | Gate 2-L2 recorded exactly as measured: "PM100 main PASS; D21 subset (test days ≥ 90) FAIL on KS (0.0232 > 0.02), all other D26 checks pass on that subset; Eagle FAIL (extreme tail)". The gate is NOT relaxed after the fact. We proceed because the gate's purpose (a valid input to Layer 2) is measured directly by Gate 4a, which becomes binding: if Gate 4a fails on the random-placement base with Mode P p-values, we stop and return to Layer 1 | Decision by Mahesh after Phase 2 review (`reports/phase_2.md`) |
| D28 | 10 Oct 2026 | Mode P (pooled, W = 7 d, main population) is the Layer-2 input. Threats added to §12: W chosen on PM100 itself and narrow pass (W = 1 d, 365 d fail); marginal-only calibration (q̂ decile 10 FA 0.004, walltime decile 10 FA 0.123); Eagle extreme tail fails, no generality claim | Decision by Mahesh after Phase 2 review |
| D29 | 10 Oct 2026 | Two base traces per scenario: random placement (PRIMARY; each job's node set replaced by |N_j| distinct nodes drawn uniformly from its partition's nodes, seeded, one draw per base seed) and real placement (SECONDARY; flags on non-injected nodes are "unknown truth") | PROMPT 3 addendum A1: exact truth needs no real slow node and no user-node link |
| D30 | 10 Oct 2026 | Truth table has a `cause` per degraded node from {cpu, cache, mbw} (equal mix, seeded); memory-pressure causes are not injected as slowdowns | Addendum A2; HPAS memleak/memeater did not visibly slow applications (LAYER3_DESIGN F23) |
| D31 | 10 Oct 2026 | Onset t₀ ≥ PM100 Mode E freeze time + 7 d (no injected job in the frozen model's training rows or Mode P's warm-up) and never inside a submission gap listed in the data card | Addendum A3 (updated after Phase 2 review) |
| D32 | 10 Oct 2026 | s ∈ {1.1, 1.2, 1.3, 1.5} | Addendum A4; compared with Phase 6.3 measurements later |
| D33 | 10 Oct 2026 | Report per scenario: affected jobs, affected nodes, censored jobs; for random placement the per-node job counts; the full scenario grid with counts per kind and base | Addendum A5 and Mahesh's answer 8 |
| D34 | 10 Oct 2026 | Injection rules: only COMPLETED/TIMEOUT rows can be affected (others copied unchanged); main-population filter (run_time > 30 s) applied after injection, jobs crossing 30 s reported; independent onset per degraded node, uniform on [freeze + 7 d, trace end − 14 d], redrawn if in a listed gap (same range for shock t₁); degradation lasts to the trace end; decoy user drawn from users with ≥ 50 main-population jobs starting after that scenario's t₀ (t₀/user redrawn if none; redraws recorded); censoring T' = max(T, limit) if s·T > limit; start times fixed | Mahesh's answers 1–5 to the Phase 3 plan |
| D35 | 10 Oct 2026 | A "null" kind (no injection) for every base: 20 random-placement bases + the real base; scenario seed k uses random-placement base k so every scenario has a matching null on the same placement | Exact null for Gate 4a (Mahesh's answers 6–7) |
| D36 | 10 Oct 2026 | Heavy decoys: kind `user_decoy_heavy`, user drawn uniformly from the 10 users with the most main-population jobs starting after t₀ (ties by user id); s ∈ {1.1, 1.2, 1.3, 1.5}; 20 seeds; both bases (160 scenarios); existing scenarios untouched | Mahesh, Phase 3 review (P1) |
| D37 | 10 Oct 2026 | 200 random-placement null bases (seeds 0–199; 0–19 are the existing ones); bases 20–199 have placement tables and an empty null scenario only (Layer-1 p-values do not depend on placement) | Mahesh, Phase 3 review (P2); Gate 4a Monte-Carlo precision |
| D38 | 10 Oct 2026 | Layer-1 input to Layer 2 = Mode P, W = 7 d (Gate 2-L2 as recorded in D27; proceed per D27/D28) | PROMPT 4 addendum B1 |
| D39 | 10 Oct 2026 | Gate 4a measured on the random-placement base (exact truth); real-placement flags on non-injected nodes labelled "unknown truth" | Addendum B2 |
| D40 | 10 Oct 2026 | User-decoy defence (ablation variant): w_j = 1 / (|N_j| · max(1, c_{u,n})), c_{u,n} = jobs of the same user that STARTED on node n in the 24 h before job j started (predictable, F16); tested to use only jobs started before j | Addendum B3 |
| D41 | 10 Oct 2026 | Shock guard (a) detrends S before Mode P: S'_j = S_j − median(S of jobs that ended in the 6 h before end_time_j), then Mode P on S'; guard (b) unchanged | Addendum B4 |
| D42 | 10 Oct 2026 | E-process tail test: on simulated null streams with Mode P p-values from IID scores, P(max M ≥ 1/α) ≤ α + Monte-Carlo error | Addendum B5 |
| D43 | 10 Oct 2026 | Gate 4a also run with Mode P W = 1 d and 365 d and with Mode E as-run p-values (sensitivity/ablation); Gate 4a decided on W = 7 d only | Addendum B6 |
| D44 | 10 Oct 2026 | Metrics over time: null scenarios — P(any node ever flagged at any 6-h evaluation over the trace) is the Gate 4a quantity, compared with α (+ 2 Monte-Carlo SE over the 200 nulls); node scenarios — FDP at every 6-h evaluation, reported as its mean over time and at onset + 7 d and + 14 d; FDP of the cumulative flagged set descriptive only; detection time = first evaluation at which the node is in the e-BH set; no reset in simulation (reset logic unit-tested only) | Mahesh, Phase 4 addition C1 |
| D45 | 10 Oct 2026 | Recall over (a) all degraded nodes and (b) degraded nodes with ≥ 1 affected main-population job | C2 |
| D46 | 10 Oct 2026 | False flags split into "co-allocated with a degraded node after its onset" vs "unrelated" | C3 |
| D47 | 10 Oct 2026 | Decoy and shock results reported against decoy size (affected main jobs) and separately for user_decoy_heavy; the 4 degenerate shock scenarios reported apart | C4 |
| D48 | 10 Oct 2026 | Before any full Phase 4 run: measure per-scenario runtime (Mode P recompute + Layer 2) and give the total; pilot (null: 20 bases; node: K = 1%, s = 1.3, 5 seeds, random base) shown to Mahesh first | C5 |
| D49 | 10 Oct 2026 | Gate 4a null bound proved via the stopping time τ = first evaluation at which e-BH rejects anything: stopped e-processes are e-values, e-BH at τ controls FDR ≤ α (F17), and under the global null FDR = P(τ < ∞) ≤ α (proof in §9 Phase 4.3 and the Phase 4 report). Replaces my incomplete Markov argument | Mahesh, correction K1 |
| D50 | 10 Oct 2026 | e-BH denominator m fixed at 980 = all PM100 nodes, at every evaluation; nodes with no scored job have M = 1. (Mahesh's K2 said 972 = "all nodes"; 972 is the count with a post-freeze main-population job in the REAL placement only — every random-placement base checked (00, 01, 07, 19, 150) uses all 980 nodes, so m = 972 would be smaller than the number of hypotheses there. 980 is fixed and data-independent; slightly conservative on the real base. Confirmed by Mahesh, 10 Oct 2026.) | K2: a data-dependent m breaks the e-BH proof |
| D51 | 10 Oct 2026 | Guard (b) is predictable: the freeze decision for job j uses only p-values of jobs that ended strictly before j | K3 |
| D52 | 10 Oct 2026 | The 200 random nulls share one real p-value realization (only placement varies); the Monte-Carlo SE is conditional on this trace | K4 |
| D53 | 10 Oct 2026 | Default shock guard = the first of (a) detrend, (b) global monitor, none that passes Gate 4a null + shock; all three reported | Answer 1 (superseded by D65) |
| D54 | 10 Oct 2026 | Decoy/shock part of Gate 4a: P(any node ever flagged) ≤ α + 2 SE, pooled over the four s values, for user_decoy and system_shock on the random base (80 scenarios each). Heavy decoys: same metric, reported, NOT binding. Real-base decoys: excess flags over the real-base null reported | Answer 2 |
| D55 | 10 Oct 2026 | Adaptation of baselines B1/B2: trailing 7-day windows evaluated daily; B1 "persistent" = condition holds on 3 consecutive daily evaluations (the guide's ≥ 10 jobs per window is impossible with 6-h windows; PM100 median ≈ 5 jobs per node per day) | Answer 3 |
| D56 | 10 Oct 2026 | Guard (b): global Simple-Jumper martingale on pooled Mode P p-values betting on small p (f(p) = 1 + ε(p − 0.5), ε ∈ {−1, 0}, J = 0.01); signal while S_t / min_{s≤t} S_s ≥ 100; Layer-2 updates frozen for jobs ending during the signal (decided from jobs that ended before, D51); running minimum reset when the signal ends | Answer 4 |
| D57 | 10 Oct 2026 | FDP at onset + 7 d / + 14 d = scenario-level FDP at the first evaluation at or after each degraded node's own onset + 7 d (+ 14 d), averaged over the degraded nodes | Answer 5 |
| D58 | 10 Oct 2026 | Variant allocation: primary + baselines on all scenarios; ablations (plain product, averaging/B4, guards none/a/b, decoy-defence weight) and B6 sensitivity (Mode P W = 1 d, 365 d, Mode E) on the Gate 4a set (nulls, decoys, heavy decoys, shocks) and the Gate 4b slice (node, random base, K = 1%, all s); real base: primary + baselines only | Answer 6 |
| D59 | 10 Oct 2026 | Sanity check before any new method: for the pilot node scenarios, each affected job's p-value is compared with the SAME job's p-value in the matching null (paired); the median paired difference must be negative (else STOP: bug); the before/after-onset median p difference is explained | Architect's power analysis after the first pilot (zero recall) |
| D60 | 10 Oct 2026 | Centred score S_c(j) = S(j) − median(S of the same user's jobs with the same nodes_req and walltime_req_s that ended in the 7 d before end_time(j), if ≥ 3); else the same user's jobs in that window (≥ 3); else S(j). Only jobs that ended strictly before j. Mode P (pooled, W = 7 d) on S_c; level used recorded. Gate 2-L2 recomputed on PM100 main for S_c before Layer 2 uses it; STOP if it fails | Architect: centring removes user/shape heterogeneity (≈ 90% of jobs at the same-shape level, to be measured) |
| D61 | 10 Oct 2026 | Shift-likelihood-ratio bets: g_s0(p) = 20-bin histogram density of Mode P p-values obtained by adding log(s0) to the scores of jobs that ended in the warm-up week [freeze, freeze + 7 d] (before every onset), renormalised to integrate to 1; s0 ∈ {1.2, 1.3, 1.5, 2.0}. Primary bet = jumper over {no bet, g_1.2, g_1.3, g_1.5, g_2.0}, J = 0.01; κ power-calibrator mixture kept as an ablation. Tests: ∫g = 1 (quadrature), g ≥ 0, no data after freeze + 7 d | Architect: power-calibrator bets waste capital on tiny p; an LR bet matched to a runtime shift is near-optimal |
| D62 | 10 Oct 2026 | Detectability floor (reported, not a gate): for s ∈ {1.1, 1.2, 1.3, 1.5, 2.0, 3.0}, expected log-growth per job-node pair (w = 1/|N_j|) of the κ mixture, the shift-LR jumper and the oracle bet, and the implied days to reach log(980/0.1) at the median node's job-node-pair rate; raw S and S_c | Architect |
| D63 | 10 Oct 2026 | Add node scenarios with s ∈ {2.0, 3.0} (all K, both bases, 20 seeds; same degraded nodes and onsets as the existing seeds); only the new ones are generated | Tuncer TPDS 2019 §4.3 ("increases the running time ... at most by 3X, which is in line with the performance variation observed in production systems"); Prodigy SC'23 §1 ("over 100%") — quotes as given by Mahesh |
| D64 | 10 Oct 2026 | Seeds 0–4 are DEVELOPMENT seeds (used to choose D60/D61); all Gate 4b numbers come from seeds 5–19 only; development seeds reported separately. Gate 4a/4b criteria unchanged; if Gate 4b fails at s = 1.3 it is reported as failed with the s at which recall crosses 0.5 and 0.8 | Avoid selecting the method on the evaluation seeds |
| D65 | 10 Oct 2026 | Guard (b) as specified (D56) is degenerate on this trace: it freezes 99.4% of updates on the nulls (pilot2). Removed from the guard rule and reported as a failed design with its freeze fraction. Guard rule (replaces D53): guard (a) if it passes Gate 4a (null + decoy + shock), else none. All variants reported | Pilot2 (run 20261010-092242-pilot2) |
| D66 | 10 Oct 2026 | Shift-LR bet grid = {1.2, 1.3, 1.5, 2.0, 3.0}: g_3.0 added so the grid covers the scenario range of D63 (Tuncer §4.3, up to 3×). Disclosed: added AFTER seeing development seeds 0–4. J stays 0.01 (F14). No other tuning | Mahesh, after pilot2 |
| D67 | 10 Oct 2026 | Gate 4b unchanged and reported as measured. Evidence budget added: nats needed = log(980/0.1) + log(number of jumper states / J); measured mean log-growth per job-node pair after onset on degraded nodes (seeds 5–19); node pair rate → predicted delay; predicted vs observed per s | Mahesh |
| D68 | 10 Oct 2026 | Detection is right-censored at the trace end: Kaplan–Meier curves of P(detected within d days of onset) per s and K, with KM median delay, in addition to recall at trace end; delay vs the node's job-node-pair rate | Mahesh |
| D69 | 10 Oct 2026 | Price of the guarantee: B5 (per-node Ville, no multiplicity control) and e-BH at α ∈ {0.05, 0.1, 0.2} shown together as delay vs FDP | Mahesh |
| D70 | 31 Oct 2026 | Fresh evaluation seeds. Seeds 5–19 have now been looked at, so every NEW Layer-2 design is judged only on new seeds 20–39: node scenarios (both bases, K ∈ {1, 2, 5}%, s ∈ {1.1, 1.2, 1.3, 1.5, 2.0, 3.0}), user_decoy, user_decoy_heavy, system_shock, plus random-placement nulls with base seeds 200–399. Phase 3 code and config unchanged (the same keyed draws, through a separate driver that imports them). Seeds 0–19 become development seeds | Mahesh |
| D71 | 31 Oct 2026 | Predictable blame allocation ("explaining away", v2): replace w_j = 1/\|N_j\| by λ_{j,n} = M_n⁻ / Σ_{k ∈ N_j} M_k⁻, M_k⁻ = node k's total jumper capital just before job j (D14 order; jobs with the same end time all use the state before that tie block). λ ∈ (0, 1], predictable, reduces to 1/\|N_j\| when all M are equal, = 1 for single-node jobs. No other parameter. The D49 stopping-time proof still applies | Mahesh |
| D72 | 31 Oct 2026 | Honest validity statement in PLAN Phase 4.1: the per-node null is violated for a healthy node that shares jobs with a degraded node (straggler contamination). The guarantee is exact under the global null; with degraded nodes present, FDR is controlled empirically. Measure whether D71 reduces co-allocated false flags | Mahesh |
| D73 | 31 Oct 2026 | Development pilot: seeds 0–4, K = 1%, s ∈ {1.3, 2.0, 3.0}, random base, plus the 20 original nulls. Compare v1 (lr_Sc_a) with v2 (lr_Sc_a + D71): KM median delay, share found within 14 d, recall at trace end, mean FDP, co-allocated false flags, mean λ on degraded nodes after onset, null P(ever). Show the numbers and STOP | Mahesh |
| D74 | 31 Oct 2026 | Only after Mahesh's go: v1 and v2 on fresh seeds 20–39 and nulls 200–399, Gate 4a and Gate 4b definitions unchanged; reports/phase_4b.md | Mahesh |
| D75 | 31 Oct 2026 | D71 (predictable blame allocation) is rejected. On the real trace it raises null P(ever) to 0.300 ± 0.032, while on iid p-values it is 0.035: Mode P p-values are marginally but not conditionally calibrated (bursts; lag-1 autocorrelation of 1{p < 0.1} = 0.29). v1 (lr_Sc_a) remains the Layer-2 configuration. No λ caps, batching or other tuning of D71 | Mahesh |
| D76 | 31 Oct 2026 | Recorded as a result, not a footnote: (i) the per-node e-process guarantee needs p-values that are valid given the past, and real Mode P p-values violate this through temporal clustering; (ii) v1 passes Gate 4a empirically partly because w = 1/\|N_j\| dilutes bursts; (iii) report the diagnostic table (real / iid / shuffled / shuffled within job size) for v1 and v2, the autocorrelation of 1{p < 0.1} at lags 1, 10, 100, and the dates of the null flags. Added to PLAN Phase 4.1 and §12 | Mahesh |
| D77 | 31 Oct 2026 | Fresh seeds 20–39 and nulls 200–399 stay sealed: nothing runs on them until the final system (Layer 2 v1 + Phase 5 active probing) is frozen. Replaces the D74 run. Disclosed: bases 20–39 were already used as null bases in Phase 4 | Mahesh |
