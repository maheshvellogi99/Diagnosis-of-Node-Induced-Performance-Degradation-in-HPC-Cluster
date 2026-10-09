# Phase prompts for Claude Code

How to use:
1. Open a terminal in the project folder:
   `cd ~/Downloads/"Final Year Project"/HPC && claude`
2. Paste **one** prompt below. Let Claude Code finish and write `reports/phase_<N>.md`.
3. Send that report (and any figures) to your senior architect chat for review.
4. Only after review, paste the next prompt.

Every prompt assumes `CLAUDE.md` (root) and `docs/PLAN.md` are present.

---

## PROMPT 0 — Setup and data check

```
Read CLAUDE.md and docs/PLAN.md fully before doing anything. We are doing Phase 0 only.

Goal: Phase 0 in docs/PLAN.md §9 (Setup and data check). Gate 0 must pass.

Steps:
1. Show me a short plan of what you will create and run. Wait for my "go".
2. Create the repository layout from PLAN.md §5. git init, .gitignore (data/, runs/, .venv/,
   *.parquet), uv project with Python 3.11 and the packages in PLAN.md §6 (no PyTorch yet).
   Ask me before installing anything not listed.
3. Ask me before moving files. Then move eagle_data.parquet to data/raw/eagle/ and the PM100
   file to data/raw/pm100/. Write scripts/download_data.py that records SHA-256 of all raw files
   into data/raw/CHECKSUMS.txt (it must NOT download anything yet).
4. Implement src/hpcdiag/io/ loaders that map both datasets to the contracts in PLAN.md §3.1.
   Use only needed columns. Units: seconds. PM100 time_limit is minutes (INFINITE -> null).
5. Write tests in tests/test_io.py: schema, dtypes, no negative run times, PM100 nodes parsed to
   lists of ints, job_nodes table row count == sum of nodes per job.
6. Produce docs/data_cards/eagle.md and docs/data_cards/pm100.md (schema, counts, date range,
   state counts, units, missing values, runtime quantiles, per-user job counts).
   For PM100 also: number of distinct nodes; jobs per node per day (median, p10, p90);
   fraction of multi-node jobs; whether any two jobs overlap in time on the same node;
   whether a job-name-like field exists; the dataset licence from its Zenodo page if available
   locally (otherwise write UNKNOWN and ask me).
7. Check that the Eagle numbers match PLAN.md §2.1 facts F1–F5 exactly. Report any mismatch.
8. Write reports/phase_0.md: what was done, exact commands, all numbers, Gate 0 pass/fail,
   answers to Q1–Q3. Commit. STOP and wait for me.

Do not start Phase 1.
```

---

## PROMPT 1 — Reproduce Menear et al. (Layer 1 baseline on Eagle)

```
Read CLAUDE.md and docs/PLAN.md §9 Phase 1 and §10. We are doing Phase 1 only.

1. Show me a plan (modules, config keys, tests, expected runtime). Wait for "go".
2. configs/phase1_menear.yaml with every constant from PLAN.md Phase 1 (filters, features,
   window = 100 days, test window = 1 day, split every 6 days, fast_mode with 12 splits).
3. src/hpcdiag/layer1/splits.py: train = jobs with end_time in [t-100d, t);
   test = jobs with submit_time in [t, t+1d). Tests that prove no training job ends at/after t.
4. src/hpcdiag/features/: label encoding of user and partition fitted on the training window
   only; unseen categories -> -1. Tests.
5. src/hpcdiag/layer1/baseline.py: xgboost.XGBRegressor() defaults, target run_time seconds.
   Record the xgboost version.
6. src/hpcdiag/eval/metrics.py: RMSE (hours), MAE (hours), TARE A_geo and UnderRate exactly as
   PLAN.md §10, error by runtime bins (0-1h, 1-6h, 6-24h, 1-2d, >2d). Unit tests with tiny
   hand-computed examples.
7. Run fast mode first, show me the numbers, then (after my go) the full run.
8. reports/phase_1.md: commands, run ids, overall RMSE vs 8.14 h, per-bin table, plot of
   predicted vs actual by bin, Gate 1 pass/fail. Do not tune anything on test splits.
   Commit. STOP.
```

---

## PROMPT 2 — Quantile model, conformal p-values, ACI

```
Read CLAUDE.md and docs/PLAN.md §9 Phase 2 and §2.1 facts F10-F12, F19. Phase 2 only.

1. Plan first; wait for "go".
2. src/hpcdiag/stats/conformal.py:
   - smoothed online conformal p-value (PLAN Phase 2.3 Mode E formula, F12), with a seeded
     generator for theta;
   - split-conformal p-value for Mode A: (1 + #{S_i >= S_j}) / (n + 1).
   Tests: on 10,000 IID simulated scores the p-values are uniform (KS p > 0.01, 5 seeds);
   with ties, smoothing still gives uniform p-values; order/leakage tests.
3. src/hpcdiag/layer1/quantile.py: LightGBM quantile alpha=0.9 (and 0.5) on log(max(run_time,1)),
   same features and splits as Phase 1.
4. src/hpcdiag/layer1/scores.py: S = log T - q_hat; group key user -> partition -> global
   (n_min = 30); Mode E (model frozen after the first 100 days) and Mode A (daily retrain,
   14-day calibration window not used for training).
5. src/hpcdiag/layer1/aci.py: alpha_{t+1} = alpha_t + gamma (alpha - err_t), per group,
   gamma in {0.001, 0.005, 0.01}. Unit test of the long-run rate on a simulated drifting stream.
6. Baselines B-guide (r > 0.20) and B-UARP (margin formula F10).
7. Run on Eagle (fast mode first), then on PM100 with the same code.
8. reports/phase_2.md: p-value histograms (pooled and 5 largest users), KS distances,
   false-alarm rate vs alpha table for Mode E, Mode A, ACI and both baselines, TIMEOUT share among
   alarms, Gate 2 pass/fail on both datasets. Commit. STOP.
```

---

## PROMPT 3 — Semi-synthetic ground truth on PM100

```
Read CLAUDE.md and docs/PLAN.md §9 Phase 3. Phase 3 only.

1. Plan first; wait for "go".
2. configs/phase3_inject.yaml: K in {1%,2%,5%}, s in {1.1,1.2,1.3,1.5}, scenarios
   node / node_partial / user_decoy / system_shock, 20 seeds each, warm-up period
   (onset never in the first 30 days), censoring at the time limit.
3. src/hpcdiag/inject/: pure functions that take jobs + job_nodes + scenario and return
   modified jobs + truth table (PLAN §3.1). Never modify data/raw.
4. Tests: unaffected jobs identical; affected jobs follow the formula exactly; censoring
   applied; truth tables exact; same seed -> same output.
5. Generate all scenarios to data/processed/scenarios/<scenario_id>/.
6. reports/phase_3.md: list of scenarios, counts of affected jobs/nodes per scenario,
   Gate 3 pass/fail. Commit. STOP.
```

---

## PROMPT 4 — Layer 2: e-processes and e-BH

```
Read CLAUDE.md and docs/PLAN.md §9 Phase 4 and facts F13-F18. Phase 4 only.

1. Plan first; wait for "go".
2. src/hpcdiag/stats/evalues.py: power calibrator kappa*p^(kappa-1); betting update
   1 + w (f(p) - 1); mixture over kappa; one-sided jumper mixture (J = 0.01); all in log space.
   Tests: each betting function integrates to 1 (numerical); simulated null streams have
   mean M_T ~ 1 and P(max M >= 1/alpha) <= alpha (+ Monte-Carlo error); w = 1 gives the
   plain product.
3. src/hpcdiag/stats/ebh.py: base e-BH (F17) and BH. Tests vs a hand example and a brute-force
   check; simulated null e-values give FDR <= alpha.
4. src/hpcdiag/layer2/: per-node accumulation from layer1 scores + job_nodes; evaluation every
   6 h; reset after a node is cleared; system-shock guard options (a) detrend and (b) global
   monitor; baselines B1-B5 exactly as PLAN §9 Phase 4.4.
5. Run Gate 4a on clean PM100 and on user_decoy / system_shock scenarios first. Show me.
6. After my go, run Gate 4b on node scenarios: FDR, recall, delay (hours and jobs) vs s and K.
7. reports/phase_4.md: tables and plots (FDR vs alpha, recall vs s, delay vs s), all
   baselines, ablations (product vs average, jumper vs plain, guard a vs b), gates.
   Commit. STOP.
```

---

## PROMPT 5 and 6

Will be written after the Layer 3 papers are read (PLAN.md §9 Phase 5–6 are provisional).
