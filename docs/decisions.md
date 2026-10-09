# Decision log

Copy of PLAN.md §16, kept current. PLAN.md is the source of truth.

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
