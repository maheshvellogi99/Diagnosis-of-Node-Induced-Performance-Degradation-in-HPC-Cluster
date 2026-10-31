# Phase 4b report — predictable blame allocation (D70–D77)

**Status: D71 rejected (D75). v1 (`lr_Sc_a`) remains the Layer-2 configuration.** The Phase 4 gate results stand as reported: Gate 4a PASS, Gate 4b FAIL. No Layer-2 design was evaluated on the fresh seeds, which stay sealed (D77).

## 1. Result

- v2 (D71) detects faster on development seeds, but flags **0.300 ± 0.032** of the 200 old null bases on the real p-value trace, above the Gate 4a limit of 0.165 (α + 2 SE). v1 flags 0.040 ± 0.014.
- On iid uniform p-values v2 flags 0.035 ± 0.013, so the code and the proof hold. The failure comes from the p-values.
- **The per-node e-process guarantee needs p-values that are valid given the past. Real Mode P p-values violate this through temporal clustering (D76 i).** They are marginally calibrated, but 1{p < 0.1} has autocorrelation 0.29 at lag 1.
- **v1 passes Gate 4a empirically partly because w = 1/|N_j| dilutes bursts (D76 ii).** With w = 1 the same process flags 0.985 of real nulls but only 0.045 of iid nulls.
- Gate 4a on real data is therefore an empirical result, not a consequence of the theorem. The theorem applies to p-values that meet its condition.

## 2. What was tested (D71, D72)

v2 replaced w_j = 1/|N_j| with λ_{j,n} = M_n⁻ / Σ_{k ∈ N_j} M_k⁻, where M_k⁻ is node k's total jumper capital before the job's tie block. `blame_reference` (step by step) and `blame_logM` (vectorised) are in `src/hpcdiag/stats/evalues.py`. `tests/test_blame.py` checks hand cases, equal capital, single-node jobs, predictability across tie blocks, one-step fairness with λ fixed, and the D49 stopped mean under a simulated global null. The code stays in the repository to reproduce this report.

## 3. Development pilot (D73)

Run `20261031-224342-pilot4b` (commit `bc20d67`, 35 scenarios). Seeds 0–4, K = 1%, random base, plus the 20 original nulls. `KM` = Kaplan–Meier; delays in days.

| s | version | scenarios | degraded nodes | KM median delay | found within 14 d | KM P(≤ 14 d) | recall at trace end | mean FDP | co-allocated false nodes per scenario | unrelated false nodes per scenario | mean λ after onset | mean λ, multi-node jobs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.3 | v1 | 5 | 50 | 28.35 | 0.16 | 0.16 | 0.84 | 0.000 | 0.0 | 0.0 | 0.229 | 0.074 |
| 1.3 | v2 | 5 | 50 | 20.25 | 0.36 | 0.36 | 0.90 | 0.000 | 0.4 | 0.0 | 0.781 | 0.737 |
| 2.0 | v1 | 5 | 50 | 19.45 | 0.26 | 0.26 | 0.90 | 0.000 | 0.4 | 0.0 | 0.229 | 0.074 |
| 2.0 | v2 | 5 | 50 | 15.45 | 0.46 | 0.46 | 0.92 | 0.002 | 0.4 | 0.0 | 0.803 | 0.764 |
| 3.0 | v1 | 5 | 50 | 17.20 | 0.32 | 0.32 | 0.88 | 0.000 | 0.2 | 0.0 | 0.228 | 0.074 |
| 3.0 | v2 | 5 | 50 | 14.45 | 0.48 | 0.48 | 0.92 | 0.002 | 0.4 | 0.0 | 0.802 | 0.762 |

Multi-node jobs give 83% of the updates on degraded nodes after onset. For v1 the λ columns show w on the same updates.

| version | null P(ever), 20 original nulls | flagged bases |
|---|---|---|
| v1 | 0.050 ± 0.049 (n = 20) | rand19 |
| v2 | 0.400 ± 0.110 (n = 20) | rand02, rand03, rand04, rand09, rand10, rand15, rand16, rand18 |

- **Power:** the v2 gains cannot be separated from the invalid null behaviour, so they are not claimed as real speed. Even so, the found-within-14-d share at s = 1.3 is 0.36, below the 0.8 target.
- **Co-allocated false flags (D72):** D71 did not reduce them on this sample of 5 scenarios per s.

## 4. Null diagnostic (D76 iii)

Run `20261031-230740-diag-d71` (commit `ef8e80d`), on null bases 0–199, not on bases 200–399. Each stream keeps the warm-up mask of the real p-values. Random streams use seed 10000 + base.

| p-value stream | v1 P(ever) | v2 P(ever) | v1 median max log M | v2 median max log M |
|---|---|---|---|---|
| real | 0.040 ± 0.014 | 0.300 ± 0.032 | 6.17 | 7.59 |
| iid uniform | 0.085 ± 0.020 | 0.035 ± 0.013 | 6.65 | 6.46 |
| shuffled across jobs | 0.055 ± 0.016 | 0.030 ± 0.012 | 6.55 | 6.21 |
| shuffled within job size | 0.025 ± 0.011 | 0.005 ± 0.005 | 6.32 | 5.42 |

n = 200 per cell, α = 0.1.

- **iid:** code and proof hold for v2.
- **Shuffled across jobs:** the real marginal distribution does not cause the failure.
- **Shuffled within job size:** the link between p-value and job size does not cause it either.
- **Real:** only the time-ordered stream inflates v2.

### Direct check of D76 (ii): no dilution

Added beyond the requested table: v1 with w = 1 instead of 1/|N_j|, so a job bets at full weight on every node.

| p-value stream | v1, w = 1/\|N_j\| | v2, λ | v1 with w = 1 |
|---|---|---|---|
| real | 0.040 ± 0.014 | 0.300 ± 0.032 | 0.985 ± 0.009 |
| iid uniform | 0.085 ± 0.020 | 0.035 ± 0.013 | 0.045 ± 0.015 |

The less a job's bet is diluted, the higher the false-flag rate on real p-values, while all three stay below α on iid p-values.

## 5. Autocorrelation of 1{p < 0.1}

Real guard (a) S_c Mode P p-values in processing order (D14), 102,090 scored jobs. Lag is in jobs.

| lag | 1 | 10 | 100 | 1000 |
|---|---|---|---|---|
| autocorrelation | 0.289 | 0.202 | 0.061 | 0.009 |

For independent indicators the standard error would be about 1/√n = 0.0031. Which source produces the bursts (for example same-user campaigns) was not tested.

### Calibration by job size

| job size (nodes) | jobs | job-node pairs | mean p | share p < 0.1 |
|---|---|---|---|---|
| 1 | 72,725 | 72,725 | 0.492 | 0.105 |
| 2 | 7,691 | 15,382 | 0.497 | 0.121 |
| 3-4 | 6,814 | 24,984 | 0.533 | 0.084 |
| 5-16 | 11,701 | 144,418 | 0.519 | 0.104 |
| ≥ 17 | 3,159 | 182,109 | 0.515 | 0.066 |

These are marginal shares within each size class. They do not capture time order.

## 6. Dates of the null flags

Date of the first flag per base, real stream. All bases use the same p-value trace (D52), so a date recurring across bases points to the trace, not the placement. e-BH is evaluated every 6 h. v2 has 60 flagged bases, v1 has 8, v1 with w = 1 has 197.

**By week (Monday start):**

| week of | v1 | v2 | v1 with w = 1 |
|---|---|---|---|
| 2020-06-29 | 0 | 5 | 4 |
| 2020-07-06 | 1 | 7 | 40 |
| 2020-07-13 | 0 | 12 | 40 |
| 2020-07-20 | 1 | 0 | 0 |
| 2020-07-27 | 0 | 8 | 81 |
| 2020-08-10 | 0 | 1 | 3 |
| 2020-08-17 | 1 | 7 | 8 |
| 2020-08-24 | 3 | 2 | 5 |
| 2020-08-31 | 1 | 11 | 7 |
| 2020-09-07 | 1 | 7 | 9 |

The five busiest weeks hold 45 of the 60 v2 flags. For v1 with w = 1, 54 flags fall on 2020-08-01 and 27 on 2020-07-31.

**Every v1 null flag:**

| base | first flag (UTC) | node | jobs on node so far | median job size, 20 largest gains |
|---|---|---|---|---|
| rand19 | 2020-07-07 21:55:59 | 759 | 68 | 1.5 |
| rand64 | 2020-07-24 21:55:59 | 753 | 140 | 1.0 |
| rand94 | 2020-08-23 15:55:59 | 727 | 249 | 1.0 |
| rand155 | 2020-08-27 09:55:59 | 814 | 213 | 1.0 |
| rand28 | 2020-08-28 15:55:59 | 906 | 246 | 1.0 |
| rand137 | 2020-08-30 09:55:59 | 447 | 270 | 1.0 |
| rand164 | 2020-08-31 15:55:59 | 285 | 247 | 1.0 |
| rand27 | 2020-09-13 21:55:59 | 613 | 280 | 1.0 |

<details><summary>Flags per day</summary>

| day | v1 | v2 | v1 with w = 1 |
|---|---|---|---|
| 2020-07-04 | 0 | 1 | 1 |
| 2020-07-05 | 0 | 4 | 3 |
| 2020-07-06 | 0 | 3 | 7 |
| 2020-07-07 | 1 | 2 | 14 |
| 2020-07-08 | 0 | 0 | 4 |
| 2020-07-09 | 0 | 0 | 5 |
| 2020-07-10 | 0 | 2 | 3 |
| 2020-07-11 | 0 | 0 | 3 |
| 2020-07-12 | 0 | 0 | 4 |
| 2020-07-13 | 0 | 5 | 17 |
| 2020-07-14 | 0 | 2 | 11 |
| 2020-07-15 | 0 | 4 | 9 |
| 2020-07-17 | 0 | 1 | 3 |
| 2020-07-24 | 1 | 0 | 0 |
| 2020-07-31 | 0 | 6 | 27 |
| 2020-08-01 | 0 | 2 | 54 |
| 2020-08-13 | 0 | 0 | 2 |
| 2020-08-16 | 0 | 1 | 1 |
| 2020-08-17 | 0 | 1 | 0 |
| 2020-08-18 | 0 | 1 | 3 |
| 2020-08-19 | 0 | 4 | 4 |
| 2020-08-20 | 0 | 1 | 0 |
| 2020-08-21 | 0 | 0 | 1 |
| 2020-08-23 | 1 | 0 | 0 |
| 2020-08-24 | 0 | 2 | 3 |
| 2020-08-26 | 0 | 0 | 1 |
| 2020-08-27 | 1 | 0 | 1 |
| 2020-08-28 | 1 | 0 | 0 |
| 2020-08-30 | 1 | 0 | 0 |
| 2020-08-31 | 1 | 0 | 0 |
| 2020-09-03 | 0 | 1 | 0 |
| 2020-09-04 | 0 | 5 | 3 |
| 2020-09-05 | 0 | 4 | 4 |
| 2020-09-06 | 0 | 1 | 0 |
| 2020-09-07 | 0 | 5 | 6 |
| 2020-09-08 | 0 | 1 | 2 |
| 2020-09-11 | 0 | 1 | 0 |
| 2020-09-13 | 1 | 0 | 1 |

</details>

## 7. Fresh scenarios: generated and sealed (D70, D77)

Run `20261031-224212-fresh` (commit `0142a48`) wrote 1,400 scenarios in 574 s, listed in `data/processed/scenarios/index_fresh.parquet`.

| kind | random base | real base |
|---|---|---|
| node | 360 | 360 |
| user_decoy | 80 | 80 |
| user_decoy_heavy | 80 | 80 |
| system_shock | 80 | 80 |
| null | 200 | 0 |

- **Sealed:** nothing runs on seeds 20–39 or nulls 200–399 until the final system (Layer 2 v1 + Phase 5 active probing) is frozen. D77 replaces the D74 run. No Layer-2 code reads `index_fresh.parquet`.
- **Generation:** `src/hpcdiag/inject/run_fresh.py` reuses the Phase 3 keyed draws and checks. `run_inject.py`, `configs/phase3_inject.yaml` and `index.parquet` are unchanged.
- **Disclosure:** random bases 20–39 were already used as null bases in Phase 4, and again in the null diagnostic above. Their placements are not new. Their degraded-node draws are new and have not been evaluated. Null bases 200–399 have not been used in any evaluation.

## 8. Decisions

| # | Decision |
|---|---|
| D70 | Fresh evaluation seeds 20–39 and nulls 200–399 generated; seeds 0–19 become development seeds |
| D71 | Predictable blame allocation tested as v2 |
| D72 | Honest validity statement added to PLAN Phase 4.1 |
| D73 | Development pilot run |
| D74 | Not run; replaced by D77 |
| D75 | D71 rejected. v1 stays. No λ caps, batching or other tuning of D71 |
| D76 | Temporal clustering recorded as a result, in PLAN Phase 4.1 and §12 |
| D77 | Fresh seeds sealed until the final system is frozen |

## 9. Limitations

- The 200 real-stream nulls share one p-value trace (D52). Their SEs are conditional on this trace, and this is one trace. The iid and shuffled rows draw new randomness per base.
- The D73 pilot has 5 scenarios per s, on development seeds. The 0.300 null rate is on old bases, so it is not a formal Gate 4a result.
- The check of D76 (ii) changes bet size on every job, so it shows that dilution matters, not how much of v1's pass is due to dilution alone.
- v1 is also exposed to this clustering. Its pass does not show the p-values are valid, and a new trace with stronger bursts could behave differently.

## 10. Reproduce

```
uv run python -m hpcdiag.inject.run_fresh --config configs/phase4b_fresh.yaml
uv run python -m hpcdiag.layer2.run_layer2 --config configs/phase4_layer2.yaml --mode pilot4b --workers 7
uv run python scripts/diag_d71_null.py --config configs/phase4_layer2.yaml --workers 7
uv run python scripts/make_phase4b_report.py --pilot 20261031-224342-pilot4b --diag 20261031-230740-diag-d71 --fresh 20261031-224212-fresh
```

`uv run pytest -q`: 213 passed in 18.20s.
