# Final results (D123)

**Development seeds; sealed evaluation pending.** Every number below comes from a run in this repo, and its
run id is given. The one exception is the Prodigy binary score, which comes from the mid-review report and
is marked as such. Development used scenario seeds 0–19 and null bases 0–199. Seeds 20–39 and null bases
200–399 are sealed and have not been run. Layer 3 results use the **assumed** probe model (`configs/phase5.yaml`
`probes`), not measured probes (Phase 6 has not run).

Phase R runs (20 Nov 2026, this document):

| Run | Command | What |
|---|---|---|
| 20261120-191921-layer1-export | `uv run python scripts/export_layer1.py --config configs/phase2_conformal.yaml --ref 20261010-003334-pm100-E-main-full --out deliverables/models/layer1` | Re-fit of the frozen Layer 1 models; max \|diff\| to the frozen q_hat and q50_log 0.0 on 102,120 jobs (also after reloading the text files) |
| 20261120-192023-components | `uv run python scripts/export_components.py --config configs/phase5.yaml --out deliverables/models` | Layer 2/3 tables and the telemetry random forest |
| 20261120-192256-demo | `uv run python scripts/demo_pipeline.py --config configs/phase5.yaml --scenario node__rand00__K1__s1.3__seed00 --budget 7200 --ref 20261113-223241-dev5c --out deliverables/demo` | End-to-end demo; identical to the dev5c row |
| 20261120-192518-final-l1 | `uv run python scripts/final_layer1_metrics.py --config configs/phase2_conformal.yaml --ref 20261010-003334-pm100-E-main-full --l2-config configs/phase4_layer2.yaml` | Layer 1 accuracy and calibration in section 1 |

## 1. Layer 1

### 1.1 Prediction accuracy, PM100, 102,120 jobs after the freeze (2020-06-19 15:55:59)

Run 20261120-192518-final-l1, on `runs/phase2/20261010-003334-pm100-E-main-full/scores_eval.parquet`.
Definitions:
- LightGBM median = exp(q50_log).
- Constant median = median run time of the 27,659 training jobs (end before the freeze) = 641 s.
- Within 2x = 0.5 ≤ prediction / max(run time, 1 s) ≤ 2.
- R² and MAE on log(max(·, 1)).

| Metric | LightGBM median | Constant median | Requested walltime |
|---|---|---|---|
| Share within 2x | 0.372 | 0.252 | 0.186 |
| RMSE (h) | 5.551 | 6.284 | 11.281 |
| MAE (h) | 2.777 | 2.555 | 7.183 |
| R² on log runtime | 0.062 | −0.079 | −1.072 |
| MAE on log runtime | 1.535 | 1.820 | 2.477 |
| Spearman (prediction, run time) | 0.489 | not defined (constant) | 0.530 |

q90 model (τ 0.9): coverage P(log runtime ≤ q̂) = 0.874 (target 0.90); pinball loss (log runtime) = 0.243.

Requested walltime ranks jobs slightly better than the LightGBM median (Spearman 0.530 against 0.489). It is
still worse on every error measure.

### 1.2 Eagle (Phase 1)

Pooled RMSE 7.380 h over 1,358,770 test jobs of 239 non-empty splits (gate band 6.512–9.768 h, pass). The
median per-split RMSE is 6.437 h, outside the band. Run 20261009-221851-main-full (reports/phase_1.md).

### 1.3 Calibration on the unmodified PM100 trace

Mode P is pooled over all jobs that ended in the previous 7 days, with at least 30 references.
- Plain score S: the Gate 2-L2 score.
- Centred score S_c: S minus the median of the same user's same-shape jobs in the previous 7 days. Layer 2
  consumes S_c.

| p-values | Run | n | KS | p ≤ 0.01 | p ≤ 0.05 | p ≤ 0.10 | P(p ≤ 1e-3) |
|---|---|---|---|---|---|---|---|
| Mode E per user, user → partition → global (mid-review plan) | 20261010-003334-pm100-E-main-full | 102,120 | 0.0907 | 0.0229 | 0.0855 | 0.1545 | not in this run |
| Mode P, plain score S (Gate 2-L2, main set) | 20261010-013303-post-full | 102,090 | 0.0131 | 0.0125 | 0.0523 | 0.0997 | 0.00124 |
| Mode P, plain score S, Layer 2 population (`layer2.prep`) | 20261120-192518-final-l1 | 102,090 | 0.0131 | 0.0125 | 0.0524 | 0.0997 | 0.00124 |
| Mode P, centred score S_c (Layer 2 input) | 20261120-192518-final-l1 | 102,090 | 0.0091 | 0.0100 | 0.0513 | 0.1061 | 0.00091 |
| Mode P, plain S, test days ≥ 90 (D21 subset, Gate 2-L2 **FAIL** on KS) | 20261010-013303-post-full (reports/phase_2.md) | — | 0.0232 | 0.0119 | 0.0534 | 0.0985 | 0.00131 |

The two plain-score rows come from different code paths: `layer1.post_scores` for the first and
`layer2.prep` for the second. They differ slightly: unrounded p ≤ 0.05 is 0.052346 against 0.052375, and KS
is 0.013136 against 0.013118. The cause was not traced. The first row is the gate value.

Gate 2 (original, PLAN): Mode E fails (KS 0.0907 > 0.02). Gate 2-L2 (D26): main set pass, D21 subset fail on
KS.

S_c at p ≤ 0.10 is 0.1061. That is within the ±0.01 tolerance of Gate 2-L2, but S_c was never a gate
quantity.

## 2. Layer 2 (logs only, B = 0)

### 2.1 Baselines (Phase 4, run 20261010-094328-full, reports/phase_4.md §7)

Random base. Null and decoy P(ever) at α = 0.1. Node rows: seeds 5–19.

| Method | Null P(ever) | Decoy P(ever) | Shock P(ever) | s = 1.3: within 14 d / recall at end / FDP |
|---|---|---|---|---|
| Ours, Phase 4 version (`lr_Sc_a`) | 0.040 | 0.050 | 0.050 | 0.100 / 0.807 / 0.013 |
| B1 guide heuristic (median residual > 20%, ≥ 10 jobs, ≥ 3 users; D55) | 1.000 | 1.000 | 1.000 | 0.200 / 0.873 / 0.484 |
| B2 top 5% | 1.000 | 1.000 | 1.000 | 0.167 / 0.527 / 0.994 |
| B3 Fisher + BH | 1.000 | 1.000 | 1.000 | 0.333 / 0.680 / 0.538 |
| B4 averaged e-values + e-BH | 0.000 | 0.000 | 0.000 | 0.000 / 0.000 / 0.000 |
| B5 per-node Ville (no multiplicity control) | 1.000 | 1.000 | 1.000 | 0.393 / 0.913 / 0.413 |

### 2.2 Real-stream bursts (Phases 4b, 4c)

On the real Mode P stream, 1{p < 0.1} has lag-1 autocorrelation 0.29 (reports/phase_4b.md). Null P(ever) on
200 real-stream nulls (run 20261101-215416-diag-4c, reports/phase_4c.md §4):

| Rule | P(ever) |
|---|---|
| v1 (`lr_Sc_a`) | 0.040 ± 0.014 |
| v1 with w = 1 (mid-review weighted form) | 0.985 ± 0.009 |
| R(1) relative | 0.070 ± 0.018 |
| Rn(1) relative (adopted) | 0.070 ± 0.018 |

Decoys and shocks under R(1): P(ever) 0.000 on all three kinds, 240 scenarios.

### 2.3 Logs-only speed: Rn(1), B = 0, K = 1%

Run 20261113-223241-dev5c (reports/phase_5c.md §4). KM P(flag ≤ 14 d) and recall at trace end.

| s | 1.1 | 1.2 | 1.3 | 1.5 | 2.0 | 3.0 |
|---|---|---|---|---|---|---|
| F(14 d) | 0.040 | 0.085 | 0.115 | 0.115 | 0.175 | 0.200 |
| Recall at end | 0.335 | 0.585 | 0.725 | 0.825 | 0.870 | 0.870 |

Gate 4b (F(14 d) ≥ 0.8 at K 1%, s 1.3) **FAILS**: 0.115 here; 0.100 for `lr_Sc_a` in Phase 4.

## 3. Layer 3 (frozen primary `ig_d109`, run 20261113-223241-dev5c, reports/phase_5c.md)

### 3.1 Validity, null P(ever), fixed pool / pool per base

| B | null (200) | user decoy (80) | heavy decoy (80) | shock (80) |
|---|---|---|---|---|
| 0 (Rn(1) alone) | 0.070 ± 0.018 | 0.000 | 0.000 | 0.000 |
| 300 | 0.065 ± 0.017 / 0.065 ± 0.017 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| 1800 | 0.065 ± 0.017 / 0.065 ± 0.017 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| 7200 | 0.065 ± 0.017 / 0.060 ± 0.017 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |

All cells are ≤ α + 2 SE, so they pass.

### 3.2 Speed: KM F(14 d), random base, 20 seeds per cell

| K | B | s 1.1 | 1.2 | 1.3 | 1.5 | 2.0 | 3.0 |
|---|---|---|---|---|---|---|---|
| 1% | 300 | 0.070 | 0.475 | 0.505 | 0.570 | 0.595 | 0.635 |
| 1% | 1800 | 0.135 | 0.775 | 0.830 | 0.860 | 0.880 | 0.895 |
| 1% | 7200 | 0.155 | 0.975 | 0.985 | 0.985 | 0.995 | 1.000 |
| 2% | 300 | 0.075 | 0.412 | 0.462 | 0.537 | 0.597 | 0.597 |
| 2% | 1800 | 0.125 | 0.732 | 0.790 | 0.830 | 0.850 | 0.850 |
| 2% | 7200 | 0.180 | 0.965 | 0.980 | 0.995 | 0.995 | 0.992 |
| 5% | 300 | 0.026 | 0.333 | 0.384 | 0.413 | 0.447 | 0.469 |
| 5% | 1800 | 0.091 | 0.666 | 0.731 | 0.757 | 0.771 | 0.805 |
| 5% | 7200 | 0.179 | 0.943 | 0.977 | 0.988 | 0.994 | 0.995 |

Recall at trace end, K 1%:

| B | s 1.1 | 1.2 | 1.3 | 1.5 | 2.0 | 3.0 |
|---|---|---|---|---|---|---|
| 300 | 0.500 | 0.925 | 0.945 | 0.960 | 0.965 | 0.960 |
| 1800 | 0.655 | 0.995 | 0.995 | 0.995 | 1.000 | 1.000 |
| 7200 | 0.730 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

At K 1%, s 1.3, B 7200, F(14 d) is 0.985, with a 95% interval of 0.97–1.00 (§6).

### 3.3 Diagnosis (all K and s)

| B | flagged degraded nodes | accuracy | s per investigation | share cut by budget | healthy investigations per day |
|---|---|---|---|---|---|
| 300 | 7,057 | 0.990 | 41.1 | 0.542 | 6.4 |
| 1800 | 8,415 | 0.991 | 64.9 | 0.142 | 26.6 |
| 7200 | 8,960 | 0.991 | 71.0 | 0.039 | 100.2 |

Cluster time at budget B is B / (980 × 86,400 s): 0.0004% at 300, 0.0021% at 1800, 0.0085% at 7200.

### 3.4 Layer 2 ranking against random node and sweep

Source: first Layer 3 version (D86 policy, not the frozen primary). Runs 20261102-005008-full and
20261102-010753-addon (reports/phase_5.md §4). K 1%, s 1.3. Cells give KM F(14 d) / accuracy.

| Policy | B = 60 | B = 300 | B = 1800 |
|---|---|---|---|
| Ours (Layer 2 ranking) | 0.180 / 0.125 | 0.245 / 0.365 | 0.600 / 0.856 |
| P-randnode (random node) | 0.100 / 0.107 | 0.140 / 0.297 | 0.135 / 0.803 |
| P-sweep (all nodes in turn) | 0.120 / 0.130 | 0.130 / 0.395 | 0.120 / 0.929 |
| P-random (random probe) | 0.190 / 0.159 | 0.255 / 0.364 | 0.610 / 0.894 |

This comparison was not repeated with the frozen primary.

### 3.5 Debt stress test (D110, diagnostic)

Each degraded node gets k_pre extra healthy investigations before onset. K 1%, s 1.3, B 7200.

| k_pre | old product: F(14 d) | old: recall | D109 restart: F(14 d) | D109: recall |
|---|---|---|---|---|
| 0 | 0.965 (0.95–0.98) | 1.000 | 0.985 (0.97–1.00) | 1.000 |
| 10 | 0.245 (0.19–0.31) | 0.970 | 0.975 (0.95–0.99) | 1.000 |
| 25 | 0.000 (0.00–0.00) | 0.070 | 0.930 (0.89–0.97) | 1.000 |

Before D109, each healthy investigation cost a mean probe log-sum of −2.11 nats (reports/phase_5b.md).

### 3.6 Misspecified outcomes (D90)

Every k entry is multiplied by 1 + U(−0.5, 0.5), and the true σ is 0.1 against the assumed 0.05.

| Outcomes | B 300: F(14 d) s 1.3 / recall s 1.1 / accuracy | B 7200 |
|---|---|---|
| main | 0.505 / 0.500 / 0.990 | 0.985 / 0.730 / 0.991 |
| misspecified | 0.295 / 0.325 / 0.874 | 0.615 / 0.345 / 0.889 |

Misspecified validity, null / user decoy / heavy decoy / shock:
- B 300: 0.060 / 0.000 / 0.013 / 0.000.
- B 7200: 0.065 / 0.000 / 0.000 / 0.000.

All pass.

### 3.7 Gates on development seeds

| Gate | Criterion | Measured | Result |
|---|---|---|---|
| 5a | B 300 validity | 0.065 / 0.065 | pass |
| 5b | K 1%, s 1.3, B 300: F(14 d) ≥ 0.8 | 0.505 | **FAIL** (D101) |
| 5c | B 300: accuracy ≥ 0.9, ≤ 72.6 s per investigation | 0.990, 41.1 s | pass |
| 5F preview (i) | validity at every B | worst 0.070 | pass |
| 5F preview (ii) | B 7200: F(14 d) ≥ 0.8 | 0.985 | pass |
| 5F preview (iii) | B 7200: accuracy ≥ 0.9 | 0.991 | pass |

### 3.8 Demo (D122)

Run 20261120-192256-demo, scenario `node__rand00__K1__s1.3__seed00`, B 7200:
- 10 of 10 degraded nodes flagged within 14 days; median delay 4.83 d.
- 0 healthy nodes ever flagged.
- 10 of 10 causes correct.
- 11,001 investigations; wall time 9.2 s.
- Metrics identical to the dev5c row.

## 4. Telemetry classifier (random forest)

| Setting | Classes | Macro-F1 | Source |
|---|---|---|---|
| Prodigy, grouped 5-fold CV, 3 seeds (mid-review prototype) | healthy, anomalous | 0.970 ± 0.015 | **mid-review report (Table 2), Sep 2026; not re-run in this repo** |
| Prodigy, grouped by job | healthy, CPU occupation, memory leak | 0.883 (recall: none 0.7125, cpuoccupy 1.0, memleak 1.0) | 20261103-151709-prodigy-check |
| Prodigy, leave one application out | same three | 0.25 (exa), 0.40 (lammps), 1.00 (sw4), 0.92 (sw4lite) | 20261103-151709-prodigy-check |
| Antarex, out-of-fold (5 contiguous time folds per block, 1-h purge) | healthy, cpu, cache, mbw, mem | 1.0000 (26,359 windows) | 20261102-003936-classifier |
| Antarex, out-of-fold, 7 classes (+ cpufreq, pagefail) | 7 | 0.9962 | 20261102-003936-classifier |
| Antarex, forward (train early, test late) | 5 | 1.0000 | 20261102-003936-classifier |
| Antarex, cross-block: train mono → test multi / train multi → test mono | 5 | 0.690 / 0.854 | 20261102-004249-d94-diag |

The out-of-fold score of 1.0000 reflects fingerprints of the fault programs (D94 diagnosis), not general
skill. The cross-block matrix is the telemetry likelihood the primary uses (D98).

Per-class cross-block F1 (mono → multi): healthy 0.669, cpu 0.232, cache 0.559, mbw 0.996, mem 0.996.

## 5. Weaknesses

1. **Gate 5b failed:** at B 300, F(14 d) is 0.505 against 0.8. It stays failed (D101).
2. **Gate 4b failed:** logs alone flag 0.115 within 14 d at s 1.3.
3. **s = 1.1 is barely detectable within 14 days:** at most 0.180 at any K and B, although recall at the end
   reaches 0.730 at B 7200.
4. **Layer 3 is simulation only.** Probe effects (k), noise (σ), costs and healthy pools are assumed. Under
   misspecification, F(14 d) at B 7200 falls from 0.985 to 0.615 and accuracy from 0.991 to 0.889. Phase 6
   (hardware) has not run.
5. **The telemetry classifier learns fingerprints.** Out-of-fold scores are perfect, but cross-block scores
   are 0.690 and 0.854, and cpu falls to 0.232 F1. Prodigy does not transfer across applications (0.25–1.00).
6. **Exchangeability is assumed.** The exactness of Rn(1) assumes random placement within a partition. On the
   real placement, nodes 460, 480, 481 and 500 are flagged and user-dominated (D84, D92); no claim is made
   about them.
7. **One trace.** Layers 1–2 are evaluated on PM100 only. All 200 real-stream nulls share one p-value
   stream (D52).
8. **Layer 1 accuracy is low.** Only 0.372 of median predictions are within 2x. The guarantee does not depend
   on accuracy, but power does.
9. **D21 subset:** KS 0.0232 on test days ≥ 90, which fails Gate 2-L2 on KS only.
10. **The Layer 2 ranking comparison (§3.4)** comes from the first Layer 3 version, not the frozen primary.
