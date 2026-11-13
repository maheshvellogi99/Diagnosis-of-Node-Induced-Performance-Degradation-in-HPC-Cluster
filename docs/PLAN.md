# MASTER PLAN — From Job Evidence to Root Cause

**Statistically Guaranteed and Cost-Aware Diagnosis of Node-Induced Performance Degradation in HPC Clusters**

Owner: Mahesh Vellogi (with Vishwesh M, Awhan Ray) · Guide: Dr. K. Nagaraju · IIITDM Kurnool
Version 1.0 · 9 Oct 2026 · Status: Layers 1–2 specified in full; Layer 3 per docs/PHASE5_DESIGN.md (Phase 5 merged, D85–D94)

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
4. **Diagnosis cost not optimised.** Existing diagnosis methods assume 1 Hz telemetry from every node
   plus labelled anomaly data (F22, F25, F26), and they still confuse CPU, cache and memory-bandwidth
   causes (F23, F26). Probe-based diagnosis (F24) needs periodic detection probes on all nodes and
   treats every probe as equally costly. No method we hold starts from scheduler logs only, probes
   only nodes that a statistically controlled detector ranks as suspects, and chooses probes by
   information per unit of node time. (Rish uses unit cost; dividing by cost is ours [D]. "Always-on
   telemetry is expensive per node" is not supported: LDMS uses < 0.1% of a core, F22.)
5. **Integration missing** — the guide's own SLR: "little work integrates uncertainty-aware
   runtime residuals, multi-job node consensus, and on-demand high-resolution diagnosis" [V: hpc_prisma_slr.docx].

### 1.4 Expected contributions (what the paper will claim, if results hold)
- C1. Conformal p-values that make "this job was abnormally late" a statement with a bounded false-alarm rate.
- C2. Anytime-valid per-node evidence (product e-process) plus e-BH selection: the expected fraction
  of healthy nodes among flagged nodes is ≤ α, under any dependence between nodes.
- C3. Allocation-size weighting that shares blame across a job's nodes while keeping C2 valid.
- C4. Cost-weighted active probing (information gain ÷ node-seconds) on suspects ranked by Layer 2,
  where probe results are also e-values that feed the node evidence, with a telemetry probe whose
  likelihood is the measured confusion matrix of a telemetry classifier, and micro-benchmark probes
  that separate the classes telemetry confuses.
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
| F22 | Tuncer et al.: LDMS collects 721 metrics/s per node on Volta (52 nodes); features computed per node "and do not account for the interaction and correlation between multiple nodes"; window W = 45 s, alarm only after C = 5 consistent windows; random forest finds 98% of injected anomalies with 0.08% false alarms; LDMS uses "less than 0.1 percent of a core"; evaluation = 5-fold stratified CV over application runs; in 4-node runs one synthetic anomaly is injected on **one of the four nodes** | Tuncer TPDS 2019, §3.1, §3.4, §4.1, §4.6, §5.1, §5.3, §5.4, §5.11 |
| F23 | HPAS: 8 user-space anomaly generators (cpuoccupy, cachecopy, membw, memeater, memleak, netoccupy, iometadata, iobandwidth) with intensity knobs; membw uses x86 non-temporal store instructions (MOVNT*); netoccupy uses the SHMEM API; 256 instances of the I/O anomalies caused outages of Voltrino's Lustre filesystem; membw "significantly reduces memory bandwidth" measured by STREAM while cachecopy "has no significant impact on memory bandwidth"; cachecopy, cpuoccupy and membw slow applications most; memleak and memeater "do not visibly affect performance" (no swap; too large → crash); RF diagnosis: "the cachecopy, cpuoccupy, and membw anomalies are sometimes mistaken for each other. This could be due to the lack of metrics representing memory bandwidth"; code at github.com/peaclab/HPAS | Ates et al. ICPP 2019, Table 1, §3.3.3, §3.4, §4 (Figs. 4, 8), §5 (Figs. 9–10) |
| F24 | Rish et al.: active probing picks the next probe that maximises information gain I(X; T \| previous outcomes) (Fig. 6a, eq. 3); probes are counted, i.e. **unit cost**; detection relies on a small probe set that covers all nodes and runs on a schedule; single-fault assumption and noisy-OR likelihoods in experiments; optimal probe-set selection is NP-hard; active probing used 60–75% fewer probes than pre-planned probing on real applications; dynamics handled with DBNs | Rish et al. TNN 2005, §II–§VII, §VIII, §XI |
| F25 | Prodigy: VAE trained only on healthy samples; LDMS samples all Eclipse compute nodes at 1 Hz, ≈10 TB/day; HPAS cpuoccupy, cachecopy, membw, memleak used; I/O anomaly runs were terminated by administrators; the network anomaly caused contention only with two-node runs; macro-F1 0.95 (Eclipse) / 0.88 (Volta); anomaly threshold chosen as "the threshold that results in the highest F1-score in the **test** dataset"; output is binary (healthy/anomalous), not the anomaly type | Aksar et al. SC'23, §3, §4.1, §5.1–5.2, §5.4.4, §6.1 |
| F26 | Aksar et al. (ALBADross runtime): 60-s windows with 15-s skip; 156 node-level metrics after dropping per-core metrics; 6 applications × 5 labels (healthy + 4 HPAS anomalies); the anomaly runs on **every** node of the job; margin-sampling active learning reaches the supervised F1 with far fewer labels (0.78 F1 with 16× fewer samples); cachecopy and membw are the most-queried (most confused) classes; LDMS runs on all 1,488 Eclipse nodes; feature extraction for 1,488 windows × 2,000 features ≈ 28.5 s on one node | Aksar et al. TPDS 2024, Abstract, §IV-B, §IV-C, §IV-E, §V-B (Fig. 5), §V-D |

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
| Antarex HPC Fault Dataset | Layer 3 telemetry classifier, Gate 5d (D94) | 10.5281/zenodo.2553224 → `Antarex.zip`, `Readme_Antarex_Dataset.pdf` | about 1.65 GB with Prodigy, download | CC-BY 4.0. Fetched by `scripts/download_data.py` after Mahesh's OK (D93); only CPU/memory blocks extracted |
| Prodigy artifact (Eclipse) | Secondary binary check only, reported (D94, D99). Labels: healthy, cpuoccupy, memleak (D94's "healthy vs memleak only" was wrong, D99); read with h5py 3.16.0 | 10.5281/zenodo.8079388 → `eclipse_small_prod_dataset.tar`, `fe_eclipse_tsfresh_raw_CHI_2000.json` | see above | CC-BY 4.0. The earlier project's copy is lost; not used (D93) |
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
- **Relative (cross-sectional) e-values (D78, Phase 4c, development only).** Blocks are the 6-h
  e-BH evaluation intervals; a job belongs to the block in which it ends. For block b, partition q
  and each bet f (each jumper state's g_s0):

    f̄_{b,q} = average of f(p_j) over all pairs (j, n') with j ending in b and n' in q;
    e_{n,b} = (mean of f(p_j) over node n's pairs in b) / f̄_{b,q}.

  Each jumper state updates M ← M (1 + λ (e_{n,b} − 1)), with λ a constant fixed in advance; nodes
  with no pair in b are not updated. Jumper (J = 0.01, same states and bets) and e-BH (α = 0.1,
  m = 980) are unchanged. Scores: S_c, Mode P W = 7 d, no guard (a). Variants R(1), R(0.25) (D79).
- **Rn(1) is the Layer-2 configuration for Phase 5 (D83).** The pair-weighted form R of D78 is
  retired. v1 (lr_Sc_a) stays in every table as "faster, empirical validity only".
- **The architect's D78 proof was wrong (D83).** It argued that node n's pairs are a uniform draw
  from the block's pairs in its partition, so E[e_{n,b} | …] = 1 with the pair-weighted f̄_{b,q}.
  That is true when all jobs in the block have the same |N_j|. It is false for mixed sizes: e divides
  by the node's own pair count K_n, and K_n depends on which jobs landed on the node, while the
  pair-weighted mean weights each job by its size. Exact counterexample, found by the D80(1)
  enumeration: 3 nodes, job A size 1, job B size 2, each drawn as in the random base. f_A = 3,
  f_B = 0 gives mean update 8/9; f_A = 0, f_B = 3 gives 19/18 > 1. So R(λ) is not an e-process under
  the global null. Its measured drift on this trace was small (reports/phase_4c.md).
- **Adopted denominator, Rn (D83).** Replace f̄_{b,q} by a_{b,q} = the mean of a_{n,b} over the nodes
  of q with a pair in b, where a_{n,b} = mean of f(p_j) over n's pairs, so
  e_{n,b} = a_{n,b} / a_{b,q}. It equals the pair-weighted form when every active node has one pair.
  Proof. Assume the global null and placement independent of p-values, with each job's node set
  uniform within its partition, independently across jobs (true by construction on the random base).
  Fix all p-values and sizes of the whole trace, all placements before b, and block b's placement
  up to a relabelling of the nodes of q. A uniform relabelling leaves the law of the placement
  unchanged, so given that n has a pair in b, n is uniform over the A active positions. The values
  a_{n,b} / a_{b,q} sum to A over those positions, so E[e_{n,b} | …] = 1 for every f, whatever the
  temporal dependence (if a_{b,q} = 0 then all f = 0 and e = 1). Each jumper state's update has mean
  1 and the mixing step conserves capital, so M_n is an e-process over blocks and the stopping-time
  argument of 4.3 applies. On the real placement this exchangeability is an assumption [A] (§12).
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

### Phase 5 — Layer 3 v2: active probing as evidence + cause diagnosis (simulation)

`docs/PHASE5_DESIGN.md` is the specification (D85), as amended by D86–D104 below. It supersedes
PROMPT 5 in `docs/PHASE_PROMPTS_5_6.md` and LAYER3_DESIGN.md §3–§4. Everything is labelled
"simulation, design probe model [A]"; real numbers come from Phase 6.

- 5.0 Layer 2 frozen as Rn(1) (D83). v1 (`lr_Sc_a`) stays in every table as "faster, empirical
  validity only".
- 5.1 Probe model: a ∈ {cpu, mbw, cache, mem}; healthy log x ~ N(0, σ²), cause h on a node degraded
  with slowdown s: log x ~ N(k_{h,a} log s, σ²), k table in design §2, σ = 0.05, costs 20/20/20/1 s;
  telemetry a_tel 60 s, diagnosis only, never in e-values.
- 5.2 Probe e-values: conformal p against a healthy pool of n_pool = 1000 runs (200 as sensitivity,
  D87). f_a(p) = (1 − β) + β g_a(p), β = 0.9, on bin edges {0, 0.001, 0.002, 0.005, 0.01, 0.02,
  0.05, 0.1, 0.2, 0.5, 1}. Since D96: g_a = Σ_{(h, s0)} π(h, s0 | not h0) g_{a,h,s0}, the CURRENT
  joint posterior of the investigation (predictable: finished probes only) over cause and s0 ∈
  {1.1, 1.2, 1.3, 1.5, 2.0, 3.0}, each g_{a,h,s0} built over 2,000 independent pools and integrating
  to exactly 1. (Phase 5 used per-cause bets at s0 = 1.3 only, D87; they lowered recall at s = 1.1.)
- 5.3 Combined evidence (D109, replaces the PHASE5_DESIGN §3 formula). For node n, number its
  investigations k = 1, 2, ... in time order and let F_i = product of f over the probes of
  investigation i (telemetry contributes no factor). With N = investigations finished by t,
  P_n(t) = Σ_{k ≤ N} w_k Π_{i = k..N} F_i + (1 − J)^N,  w_k = J (1 − J)^(k−1),  J = 0.01 (the
  Layer 2 jumper J), P_n(0) = 1. Recursion per finished investigation: A ← F (A + J R),
  R ← (1 − J) R, P = A + R, with A_0 = 0, R_0 = 1. Combined evidence
  log E_n = log M_n^{Rn}(t) + log P_n(t); e-BH α = 0.1, m = 980 on E_n.
  **Validity (D109).** Write P_n = Σ_{k ≥ 1} w_k T_k with T_k = 1 before investigation k finishes and
  T_k = Π_{i = k..N} F_i after; Σ_{k > N} w_k = (1 − J)^N gives the formula above, and Σ_k w_k = 1.
  Under H0_n each probe factor f is conditionally valid given everything before it (§5.2), so
  E[F_i | past] ≤ 1 by the tower rule over the probes of investigation i; each T_k is therefore a
  test supermartingale (equal to 1 before it starts), and a mixture with fixed nonnegative weights
  summing to 1 is a test supermartingale (monotone convergence for the infinite sum). M_n^{Rn} is a
  test supermartingale over job ends. Probe completions and job ends are distinct events in one
  time order, so at every event only one factor of E_n = M_n P_n changes, with conditional mean
  ≤ 1 given the joint past; E_n is a test supermartingale in the joint filtration and the D49
  stopping-time argument applies unchanged. J = 1 gives w_1 = 1 and P = Π F_i, the old formula.
  **Design error (D109).** PHASE5_DESIGN §3 applied every probe factor to every jumper state, so
  log E = log M + Σ log f over all probes ever run. It is valid, but probe evidence never restarts:
  probes of a healthy node accumulate debt without limit (about −0.22 nats per healthy day at
  B = 7200, reports/phase_5b.md §10), which costs power after a late onset. This was the
  architect's design error; the old formula is kept only as a reference row (J = 1).
  **Validity statement (D100):** probe e-values are valid marginally over the draw of the healthy
  pool, as in split conformal prediction: E[f] ≤ 1 when the pool and the new run are exchangeable
  healthy runs, while E[f | pool] for one fixed pool can exceed 1 (Phase 5: up to 4.0 for the mbw
  pool on one bet). Assumption [A] (D88): the measured pool is exchangeable with future healthy runs
  of the same node type. Every validity table reports the fixed pool and a pool drawn per base side by
  side. Phase 6 target: at least 1000 healthy runs per probe type and a measured σ per probe.
  One fixed pool cannot make every bet fair conditionally: 39–76% of fresh 1000-run pools have a
  positive maximum of healthy log-growth over bet weights; on the fixed pools the measured worst
  case is +0.0035 nats per probe, about 2,600 probes to reach 9.19 nats (D102). Bet check (D102):
  (i) mean healthy log-growth averaged over 2,000 independent pools ≤ 0 for every bet the policy can
  use; (ii) on the fixed pool, healthy growth at the bets actually used is reported (max, q99,
  mean); (iii) validity is judged by null, decoy and shock P(ever), fixed pool and pool per base
  side by side.
- 5.4 Diagnosis: joint posterior over (cause, s), uniform over the 6 scenario s values (D89), summed
  to the 5 hypotheses; prior π0(h0) = 0.5, causes uniform, restarted for each investigation (D90).
  Next probe = argmax [H(π) − E_x H(π | x, a)] / c_a. Stop at max π ≥ 0.95, all probes used, or
  budget cut (the investigation ends with what it has, D90).
  D97 variant (stage-wise): while the node is not flagged by e-BH at the current evaluation, the
  next probe maximises the expected log f per second under the current posterior (evidence); once
  it is flagged, information gain per second (diagnosis). Expected log f is computed from bin
  probabilities averaged over independent pools, the construction of the bets, never from the
  fixed pool (D103). Selection rule fixed before results: D97 is
  kept if its KM F(14 d) at K = 1%, s = 1.3, B = 1800 is at least 0.02 above IG-only AND its null
  P(ever) ≤ α + 2 SE; otherwise IG-only is kept.
- 5.5 Schedule: every 6 h, budget B/4 plus carry-over, available budget capped at one day's budget
  (answer 2). Nodes in rank order of log E_n. **Primary rule (D95):** cleared nodes (π(h0) ≥ 0.95)
  are excluded for 7 d; a node flagged by e-BH at the current evaluation whose latest investigation
  ended with max cause posterior ≥ 0.95 is not investigated (handed to the admin; if it leaves the
  e-BH set it becomes eligible again); all others stay eligible. Adopted after the Phase 5
  development grid; it had been committed as a sensitivity row before any result. D86 (only
  cleared nodes excluded) is the architect's failed rule, kept as a reported row: it starves the
  budget (reports/phase_5.md §1). The design rule (every investigated node excluded 7 d) is also a row.
- 5.6 Baselines P-all (all five probes, 121 s), P-IG, P-random, P-randnode, P-sweep; B ∈ {60, 300,
  1800} node-s/day, and B = 0 is Layer 2 alone. Our policy also at B = 7200 (D91).
- 5.7 Telemetry likelihood (D98): the cross-block confusion matrix of the Antarex Random Forest
  (train on one block, test on the other, both directions summed), add-one smoothed; the
  out-of-fold matrix of D94 is a sensitivity row. The Gate 5d score of 1.0 is a fault-program
  fingerprint: cross-configuration macro-F1 is 0.69 and 0.85. Transfer to PM100 nodes is an
  assumption [A].

**Gate 5** (pre-registered; design §7, 5d replaced by D94).
- 5a validity (binding): B = 300, null P(ever) ≤ α + 2 SE on the 200 old null bases; decoy and shock
  P(ever) ≤ α + 2 SE (80 each, pooled over s).
- 5b speed: K = 1%, s = 1.3, B = 300: KM P(flag ≤ 14 d) ≥ 0.8 [D target; report honestly if missed].
- 5c diagnosis: top-1 cause accuracy ≥ 0.9 among flagged degraded nodes (design table, σ = 0.05),
  with mean node-seconds per investigation ≤ 72.6 s (0.6 × P-all's 121 s).
- 5d telemetry classifier (D94): RF on Tuncer-style statistical features over 60-s windows from the
  Antarex CPU/memory blocks, classes healthy, dial → cpu, ddot → cache, memeater → mbw, leak → mem,
  split by contiguous time blocks or runs, never random windows: macro-F1 ≥ 0.90 [D].

**Gate 5 results on development seeds (reports/phase_5.md):** 5a passed; **5b FAILED** (0.245 at
B = 300; recorded as failed, D101); 5c failed under D86; 5d 1.0000, a fingerprint (D98).

### Phase 5F — final sealed evaluation (pre-registered, D101)

Runs once, after the Phase 6 tables (or with the design tables if Phase 6 is impossible), on the
sealed seeds 20–39 (node, user_decoy, user_decoy_heavy, system_shock) and nulls 200–399, with the
system frozen as at the end of Phase 5b. Budget grid {0, 300, 1800, 7200} node-s/day; headline
budget B = 7200.

**Gate 5F.** (i) Validity: null, decoy and shock P(ever) ≤ α + 2 SE at every B, fixed pool and pool
per base reported side by side (D100). (ii) KM P(flag ≤ 14 d) ≥ 0.8 at K = 1%, s = 1.3, B = 7200.
(iii) Top-1 diagnosis accuracy ≥ 0.9 among flagged degraded nodes at B = 7200. No change after the run. Before anything runs, Phase 5F calls the D112 freeze guard
(`scripts/check_frozen.py`) and stops if it fails.

---

### Phase 6 — Real cluster and HPAS (needs admin approval)

Provisional. Tasks: install HPAS on 2–4 dedicated nodes; write probes (fixed-work CPU loop, STREAM,
cache test, OSU ping-pong against a healthy node, 60-s telemetry capture); run healthy baselines;
run a factorial campaign (anomaly type × intensity × repetitions) with exact start/stop logs;
estimate P(o | h, a); end-to-end run: submit benchmark jobs, inject at random times, measure
time-to-flag, diagnosis accuracy and node-seconds spent vs a fixed battery.

**Gate 6.** End-to-end run completes; every number traced to logs; admin sign-off recorded.

**Phase 6-prep (D112–D115, 14 Nov 2026).** No approved machine yet, and the development Mac is
arm64 (HPAS membw needs x86-64), so only the machine-independent parts are done: the freeze guard
(D112), the measurement-substitution rule and its plumbing (D113), the tiered protocol
docs/PHASE6_PROTOCOL.md with Gate 6 (D114, which supersedes the provisional text above), and the
measurement kit in cluster/ (D115). Report: reports/phase_6prep.md.

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
| Relative e-values (D78) rely on placement being exchangeable within a partition and block: true by construction on the random base, an assumption [A] on the real placement | Validity results are on the random base; on the real base flags are reported as unknown truth, with placement concentration per flagged node (D84) |
| Probe e-values use one fixed healthy pool, so E[f \| pool] ≠ 1 and repeated probes share it: the e-value property holds marginally over pools only, as in split conformal (D88, D100). Phase 5: null P(ever) 9/200 with the fixed pool against 1/200 with a pool per base at B = 300, 8 of the 9 created by probes | Assumption [A]: the pool is exchangeable with future healthy probe runs. Every validity table shows fixed pool and pool per base side by side; Phase 6 target ≥ 1000 healthy runs per probe type and a measured σ per probe |
| Nodes 460, 480, 481, 500 are flagged by Rn(1) on real placement (D84) and are user-dominated: top-user share above q90 of all nodes, 7–12 users. Likely placement effects that break exchangeability [A] (D92) | They are the motivating case for Layer 3: a probe measures the node, not the user. No simulated claim about them, because the simulation has no model of what these nodes are |
| Antarex telemetry classifier transferred to PM100 nodes (D94). Macro-F1 1.0000 within a block is a fingerprint of the fault programs: train on one block and test on the other gives 0.69 and 0.85 on the same machine (reports/phase_5.md §6) | Assumption [A]: different hardware, applications and metrics. The confusion matrix is a design input for a_tel, never used in e-values; cross-block and transfer sensitivity rows reported; Phase 6 replaces it |
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
| D78 | 1 Nov 2026 | Relative e-values: blocks = 6-h e-BH intervals, jobs assigned to the block in which they end; f̄_{b,q} = pair-weighted mean of f(p_j) in block b and partition q; e_{n,b} = mean of f over node n's pairs / f̄_{b,q}; nodes with no pairs not updated; M ← M (1 + λ (e_{n,b} − 1)), λ fixed in advance; jumper J = 0.01, same states and bets; e-BH α = 0.1, m = 980; S_c, Mode P W = 7 d, no guard (a). Note: the validity proof as specified fails for mixed job sizes (D80(1) enumeration, PLAN Phase 4.1); corrected denominator proposed, not adopted | Mahesh |
| D79 | 1 Nov 2026 | Variants R(λ = 1) and R(λ = 0.25). References: v1 (lr_Sc_a) and v1 with w = 1 from Phase 4b. No other parameters | Mahesh |
| D80 | 1 Nov 2026 | Tests first: exact enumeration of block placements; simulation with bursts under random placement; means per partition using block-b jobs only; λ fixed in advance, nodes with no pairs not updated | Mahesh |
| D81 | 1 Nov 2026 | Development evaluation only: (a) null diagnostic on 200 old bases for R(1), R(0.25), v1, v1 w = 1; (b) dev pilot seeds 0–4, node, K = 1%, s ∈ {1.3, 2.0, 3.0}, random base; (c) user_decoy, user_decoy_heavy, system_shock, random base, seeds 0–19: P(ever) | Mahesh |
| D82 | 1 Nov 2026 | Sealing unchanged (D77). reports/phase_4c.md | Mahesh |
| D83 | 1 Nov 2026 | Adopt the Rn denominator (mean of node means over active nodes; exact for any job sizes). Layer 2 for Phase 5 = Rn(1). v1 (lr_Sc_a) stays in every table as \"faster, empirical validity only\". R (pair-weighted) retired. The architect's D78 proof was wrong: the uniform-draw argument fails for mixed job sizes (PLAN Phase 4.1) | Mahesh |
| D84 | 1 Nov 2026 | Real-placement check of Rn(1), development only: real-base null P(ever), flagged nodes with first-flag dates, overlap with the Phase 2 exploratory list (840, 287, 841, 80, 309, 81, 441, 42, 679, 40) and with Phase 4's 9 real-null flags under v1, and per flagged node the share of its jobs from its single top user. Exploratory, unknown truth | Mahesh |
| D85 | 1 Nov 2026 | Commit docs/PHASE5_DESIGN.md (docs only). It supersedes PROMPT 5 in docs/PHASE_PROMPTS_5_6.md and LAYER3_DESIGN.md §3–§4; LAYER3_DESIGN §1–§2 and §5 still stand | Mahesh |
| D86 | 2 Nov 2026 | PRIMARY investigation rule: only cleared nodes (π(h0) ≥ 0.95) are excluded for 7 d; nodes not cleared stay eligible in the next block. The rule as written in the design (every investigated node excluded 7 d) becomes a sensitivity row. Gate 5b unchanged. Decided before any result | Mahesh |
| D87 | 2 Nov 2026 | Probe bets with resolution near 0: bin edges {0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1}; β = 0.9 (was 0.5); n_pool = 1000 main, 200 sensitivity. Per-cause bets g_{a,h} built over 2,000 independent pools at s0 = 1.3, each integrating to exactly 1. Bet for a probe = Σ_h π(h \| not h0) g_{a,h} with the investigation's current cause posterior (predictable). Before the grid report healthy and per-cause log-growth per probe for s ∈ {1.1, 1.2, 1.3, 1.5, 2.0, 3.0} and E[f \| pool] over 2,000 pools (sd, q95, value for the fixed pool). STOP if healthy growth > 0 for any probe | Mahesh |
| D88 | 2 Nov 2026 | Fixed pool is main, redraw-per-base is a sensitivity; assumption [A] stated in PLAN Phase 5 and §12 | Mahesh |
| D89 | 2 Nov 2026 | Diagnosis posterior is joint over (cause, s), uniform over the 6 scenario s values | Mahesh |
| D90 | 2 Nov 2026 | Each investigation starts from π0; a node's output is its latest investigation; a flagged degraded node never investigated counts as wrong and is also reported separately; budget-cut investigations end with what they have; misspecification pools use the true σ | Mahesh |
| D91 | 2 Nov 2026 | B = 7200 node-s/day added to our policy's grid (baselines stay at {60, 300, 1800}). Report the smallest B in the grid at which KM P(flag ≤ 14 d) at K = 1%, s = 1.3 reaches 0.8, if any | Mahesh |
| D92 | 2 Nov 2026 | D84 recorded in §12: nodes 460, 480, 481, 500 are user-dominated, likely placement effects that break exchangeability [A]; motivating case for Layer 3; no simulated claim about them | Mahesh |
| D93 | 2 Nov 2026 | Datasets: extend scripts/download_data.py to fetch and md5-verify Antarex (10.5281/zenodo.2553224, CC-BY 4.0) and the Prodigy artifact (10.5281/zenodo.8079388, CC-BY 4.0) into data/raw/antarex/, data/raw/prodigy/, after Mahesh's OK; SHA-256 in CHECKSUMS.txt; extract only the CPU/memory blocks needed; data cards antarex.md and prodigy.md; STOP if a checksum fails. The earlier project's Prodigy data is lost and not used | Mahesh |
| D94 | 2 Nov 2026 | Gate 5d redefined before any result: RF on Tuncer-style statistical features over 60-s windows (F22, F26) from Antarex CPU/memory blocks, classes healthy, dial → cpu, ddot → cache, memeater → mbw, leak → mem (cpufreq, pagefail, ioerr, copy reported but excluded from the H mapping); split by contiguous time blocks or runs, never random windows; macro-F1 ≥ 0.90 [D], reported next to Netti et al.'s published RF F-score 0.98 (arXiv 2007.14241, figure given by Mahesh, not checked against the paper here) as context. Out-of-fold confusion matrix = a_tel likelihood (diagnosis only, never in e-values); transfer to PM100 is an assumption [A]. Prodigy artifact = secondary binary check, reported only | Mahesh |
| D95 | 3 Nov 2026 | Investigation rule: the skip rule is PRIMARY. Cleared nodes (π(h0) ≥ 0.95) excluded for 7 d; flagged nodes that already have a diagnosis (max cause posterior ≥ 0.95) are not investigated again (handed to the admin); all others stay eligible. Disclosed: adopted after the development grid; it was committed as a sensitivity row before any result. D86 is kept as a reported row and recorded as the architect's failed rule | Mahesh |
| D96 | 3 Nov 2026 | Bets over all slowdown sizes: bet for probe a = Σ_{(h, s0)} π(h, s0 \| not h0) g_{a,h,s0} with the CURRENT joint posterior (D89), s0 ∈ {1.1, 1.2, 1.3, 1.5, 2.0, 3.0}, each g built over 2,000 independent pools and integrating to exactly 1. Re-run the D87 check (healthy growth per probe ≤ 0 for every reachable posterior state; growth per cause and s); STOP if any healthy growth > 0 | Mahesh |
| D97 | 3 Nov 2026 | Stage-wise probe choice (variant): before the node is flagged, the probe maximising expected log f per second under the current posterior; after, information gain per second. Compared with IG-only on development seeds. Kept if its KM F(14 d) at K = 1%, s = 1.3, B = 1800 is ≥ 0.02 above IG-only AND its null P(ever) ≤ α + 2 SE; otherwise IG-only | Mahesh |
| D98 | 3 Nov 2026 | Telemetry likelihood = cross-block matrix (both directions) as main; OOF matrix as sensitivity. The 5d score of 1.0 is a fault-program fingerprint (cross-configuration 0.69 / 0.85) | Mahesh |
| D99 | 3 Nov 2026 | Install h5py (pinned h5py==3.16.0, chosen by Mahesh 3 Nov 2026); run the Prodigy secondary check (reported only). Correction: Prodigy labels include cpuoccupy; the architect's D94 description (healthy vs memleak only) was wrong | Mahesh |
| D100 | 3 Nov 2026 | Probe e-values are valid marginally over the healthy-pool draw (as in split conformal); fixed-pool and pool-per-base results side by side in every validity table; Phase 6 pool target ≥ 1000 runs per probe type plus a measured σ per probe | Mahesh |
| D101 | 3 Nov 2026 | Pre-registered final sealed evaluation (Phase 5F), after the Phase 6 tables or with design tables if Phase 6 is impossible: budget grid {0, 300, 1800, 7200}; headline B = 7200 node-s/day. Gate 5F: (i) null, decoy and shock P(ever) ≤ α + 2 SE at every B; (ii) KM F(14 d) ≥ 0.8 at K = 1%, s = 1.3, B = 7200; (iii) diagnosis accuracy ≥ 0.9 at B = 7200. Gate 5b (B = 300) stays recorded as FAILED on development seeds | Mahesh |
| D102 | 3 Nov 2026 | Replace the strict D96 check by the marginal check: (i) mean healthy log-growth averaged over 2,000 independent pools ≤ 0 for every bet the policy can use (report the mean over pools of E[f \| pool] and of log f); (ii) on the fixed pool, healthy growth at the bets the policy actually used (max, q99, mean); (iii) validity judged by null, decoy and shock P(ever), fixed pool and pool per base side by side (D100). Probe e-values are valid marginally over the healthy-pool draw, as in split-conformal calibration; one fixed pool cannot make every bet fair conditionally (39–76% of fresh 1000-run pools have a positive maximum); measured worst case +0.0035 nats per probe, about 2,600 probes to reach 9.19 nats | Mahesh |
| D103 | 3 Nov 2026 | D97 ranks probes by expected log f computed from bin probabilities averaged over independent pools (the construction of the bets), never from the fixed pool | Mahesh |
| D104 | 3 Nov 2026 | Disclose in the Phase 5b report that Phase 5's D87 check tested only a few bets; under the full search the Phase 5 bets also reach +0.0034 (mbw) and +0.0012 (mem) nats per probe | Mahesh |
| D105 | 3 Nov 2026 | Diagnosis only of the s = 1.1 recall drop at B = 7200 against B = 1800: per degraded node, number of investigations, how many ended cleared (π(h0) ≥ 0.95) while degraded, log E before and after each investigation, and whether the 7-day exclusion after a wrong clearing explains the drop. Measure, do not change the design; if a fix seems needed, propose it and STOP | Mahesh |
| D106 | 3 Nov 2026 | Diagnose the row "IG + design rule, B = 7200, F(14 d) 0.220" (lower than at B = 300): if it is a bug in code the primary configuration also uses, STOP and report before anything else | Mahesh |
| D107 | 3 Nov 2026 | Disclose in reports/phase_5b.md the four D102 check runs (20261103-151450, -151526, -151555, -151624; marginal_positive 10, 10, 2, 0) and what changed in the code between them, so it is clear the check was corrected, not re-run until it passed. If no fix is needed after D105–D106, tag the primary configuration (Rn(1) Layer 2 + IG-only + D95 + D96 + D98) as v1.0-dev-frozen | Mahesh |
| D108 | 13 Nov 2026 | D102 stop rule confirmed: pass = exact value ≤ 1e-12 AND Monte-Carlo within max(2 SE, 1e-12); the exact computation is the criterion, Monte-Carlo a cross-check. Process rule: after editing a check script, verify the edit applied before re-running (run 20261103-151526 should not have run) | Mahesh |
| D109 | 13 Nov 2026 | Probe evidence is a restart mixture over investigations (P1 of reports/phase_5b.md §10.3), decided before any result: P_n = Σ_{k ≤ N} w_k Π_{i = k..N} F_i + (1 − J)^N, w_k = J (1 − J)^(k−1), J = 0.01; recursion A ← F (A + J R), R ← (1 − J) R; log E_n = log M_n + log P_n. Validity proof in §9 Phase 5, 5.3. The old formula (probe factors on every jumper state, PHASE5_DESIGN §3) was the architect's design error; it is J = 1 and kept as a reference row. Tests: P(0) = 1, one-step fairness, recursion = explicit sum, J = 1 = old formula | Mahesh |
| D110 | 13 Nov 2026 | Development evaluation of the primary (Rn(1) + IG-only + D95 + D96 + D98 + D109), old formula as reference: B ∈ {0, 300, 1800, 7200}, K ∈ {1, 2, 5}%, all s; validity (nulls 0–199, decoy, heavy, shock; fixed pool and pool per base); misspecified rows at B = 300 and 7200. Debt stress test (diagnostic, not a gate): k_pre ∈ {0, 10, 25} extra healthy investigations of each degraded node before onset, old formula vs D109, F(14 d) at K = 1%, s = 1.3, B = 7200 | Mahesh |
| D111 | 13 Nov 2026 | Freeze rule fixed before results: if every validity row passes α + 2 SE, tag the primary as git tag v1.0-dev-frozen and write docs/FROZEN_CONFIG.md (every parameter, config hash, commit); if any validity row fails, STOP, no tag | Mahesh |
| D112 | 14 Nov 2026 | Freeze guard: scripts/check_frozen.py exits non-zero unless (a) tag v1.0-dev-frozen exists, (b) the SHA-256 of every file listed in docs/FROZEN_CONFIG.md matches, or every difference is a D113 field, (c) the policy to be run is ig_d109 with probe_J = 0.01 (Policy's default probe_J = 1.0 is a trap the guard must catch). Phase 5F calls this guard before anything runs. Tests | Mahesh |
| D113 | 14 Nov 2026 | Measurement-substitution rule, pre-registered before any real measurement. Phase 6 may replace ONLY measured inputs: probes.k, probes.sigma (may become one σ per probe), probes.cost_s, probes.mem_fixed_shift, the healthy pool (an empirical pool of measured runs replaces the Gaussian pool), telemetry.counts. Everything else stays frozen: bin edges, β, bet_s0, s_grid, J, π0(h0), τ, exclude_days, the policy, Layers 1 and 2. Result tagged v1.1-measured. Plumbing (per-probe σ, empirical-pool option) implemented now; with scalar σ and the Gaussian pool, dev5c runs (K 1%, s 1.3, B 7200, seeds 0–4, 20 nulls) must reproduce bit for bit, else STOP | Mahesh |
| D114 | 14 Nov 2026 | docs/PHASE6_PROTOCOL.md, hardware-agnostic, three tiers: A institute cluster with written admin approval (PROMPT 6); B 2–4 identical dedicated x86-64 Linux machines we control, no one else's jobs; C no hardware: Phase 5F on the design tables and the paper states Layer 3 is simulation-only. For A/B: ≥ 1000 healthy runs per probe type over ≥ 3 days and ≥ 2 machines; {cpuoccupy, cachecopy, membw, memleak} × 2 intensities × ≥ 10 reps × all probes, randomised order; tables fitted on the first 2/3 of campaign days, checked on the last 1/3; s per anomaly × intensity from a fixed single-node benchmark with and without the anomaly (multi-node MPI straggler check only with ≥ 2 machines). Wall-clock hours per tier from probe costs. Safety rules copied verbatim. Gate 6 defined now (σ per probe with CI, held-out diagnosis accuracy, s vs the Phase 3 range): reported numbers, no tuning on them | Mahesh |
| D115 | 14 Nov 2026 | Measurement kit in cluster/ with its own README: probes (cpu fixed-work timing, mbw = STREAM, cache pointer-chase, mem from /proc/meminfo, tel = /proc/stat, /proc/meminfo, /proc/vmstat snapshots); a harness logging JSONL with start/stop timestamps, a watchdog and hard end time for every anomaly, memleak capped below 25% of node memory, refusing any anomaly unless the host is x86-64 Linux AND its hostname is in cluster/approved_hosts.txt (empty in git). HPAS is cloned on the target machine only, at a pinned commit, by a documented script; not cloned now. Analysis script: campaign logs → D113 inputs, tested on synthetic logs. On the development Mac (arm64): unit tests and --dry-run only; never run an HPAS anomaly or stress program on it. No Phase 6 hardware work until docs/admin_approval.md (tier A) or a tier-B machine list exists | Mahesh |
