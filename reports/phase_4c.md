# Phase 4c report — relative (cross-sectional) e-values (D78–D82)

**Status: development evaluation only. No design decision is made here.** v1 (`lr_Sc_a`) remains the Layer-2 configuration. No scenario with seeds 20–39 and no null base 200–399 was run (D77, D82). The 200 old null bases of D81 (a) include placements 20–39, as in Phases 4 and 4b.

## 1. Result

- **Validity:** R(1) flags 0.070 ± 0.018 of 200 real-stream nulls, where v1 with w = 1 flags 0.985 ± 0.009. Relative evidence removes the burst failure of Phase 4b.
- **Decoys and shocks:** R(1) has P(ever) = 0.000 on all three kinds (240 scenarios, bases 0–19).
- **Speed:** R(1) is slower than v1 at every s tested (section 5). At s = 1.3, K = 1%, KM median delay is 43.25 d vs 28.35 d and recall at trace end 0.72 vs 0.84. Found within 14 d is 0.16 for both, far below the 0.8 target of Gate 4b. At s = 2.0 and 3.0 its recall at trace end is slightly higher than v1, on 50 nodes.
- **R(0.25) is worse than R(1) at every s** on delay, within-14-d share and recall at trace end.
- **The D78 proof does not hold as specified.** D80(1) exact enumeration gives mean update 8/9 and 19/18 for mixed job sizes (section 2). A node-weighted denominator (Rn, added by me, not adopted) is exact and gives almost the same numbers.

## 2. Problem in the D78 validity proof

D78 argues that node n's pairs are a uniform draw from the block's pairs in its partition, so E[e_{n,b} | …] = 1. The D80(1) enumeration shows this holds for equal job sizes but not for mixed ones.

Hand case, confirmed by `tests/test_relative.py`: 3 nodes; job A has size 1, job B has size 2; 9 equally likely placements, each job drawn uniformly as in `inject/placement.py`. Node 0 holds both jobs in 2 placements, A only in 1, B only in 4, neither in 2.

| f_A | f_B | f̄ (pair-weighted) | exact mean update | mean e given a pair | D78 claim |
|---|---|---|---|---|---|
| 3 | 0 | 1 | 8/9 | 6/7 | 1 |
| 0 | 3 | 2 | 19/18 | — | 1 |

The reason is that K_n, the number of pairs on the node, depends on which jobs are there, and the mean inside e divides by K_n. So R(λ) with the pair-weighted f̄ is not an e-process under the global null. Mean 1 does hold exactly when all jobs in the block have the same size.

**Candidate correction (Rn), not adopted because it changes a D78 formula.** Use as denominator the mean of the node means a_{n,b} over the nodes of q with a pair in b. The ratios then sum to the number of active nodes in every placement. Node labels are exchangeable, so the mean is exactly 1 for any sizes and any temporal dependence. The full proof is in PLAN Phase 4.1. Rn equals R when every active node has one pair.

**Practical size of the error (section 7):** over the whole trace the average drift of R is at most 1.42 nats in size, against an evidence budget of 15.59 nats. The error is real but small here, and R and Rn give close results in sections 4–6.

## 3. Tests (D80)

| D80 item | Result |
|---|---|
| (1) Exact enumeration, mean 1 | Equal sizes: passes for R and Rn. Mixed sizes: **fails for R as specified** (the test pins 8/9 and 19/18); exact for Rn |
| (2) Burst simulation, n = 200, 100 nodes, 120 blocks, mixed sizes | P(ever): R(1) 0.010, R(0.25) 0.000, Rn(1) 0.010, full bet with w = 1 0.790. α = 0.1 |
| (3) Means per partition, block-b jobs only | Passes |
| (4) λ one constant fixed in advance; nodes with no pairs not updated | Passes |

`src/hpcdiag/stats/evalues.py`: `relative_evalues`, `relative_logM`. `src/hpcdiag/layer2/accumulate.py`: `node_evidence_relative`, `block_grid`.

## 4. Null diagnostic (D81 a)

Run `20261101-215416-diag-4c` (commit `50035bc`), null bases 0–199, n = 200 per cell, α = 0.1. v1 and v1 with w = 1 use the guard (a) p-values, as in Phase 4b. R and Rn use unguarded S_c p-values (D78). The v1 and v1 with w = 1 cells on the real and iid streams, and the v1 cells on the shuffled streams, match run `20261031-230740-diag-d71` exactly (checked by this script).

| p-value stream | v1 | v1, w = 1 | R(1) | R(0.25) | Rn(1) | Rn(0.25) |
|---|---|---|---|---|---|---|
| real | 0.040 ± 0.014 | 0.985 ± 0.009 | 0.070 ± 0.018 | 0.000 ± 0.000 | 0.070 ± 0.018 | 0.000 ± 0.000 |
| iid uniform | 0.085 ± 0.020 | 0.045 ± 0.015 | 0.045 ± 0.015 | 0.015 ± 0.009 | 0.045 ± 0.015 | 0.015 ± 0.009 |
| shuffled across jobs | 0.055 ± 0.016 | 0.025 ± 0.011 | 0.080 ± 0.019 | 0.000 ± 0.000 | 0.085 ± 0.020 | 0.000 ± 0.000 |
| shuffled within job size | 0.025 ± 0.011 | 0.010 ± 0.007 | 0.040 ± 0.014 | 0.005 ± 0.005 | 0.045 ± 0.015 | 0.005 ± 0.005 |

Flagged real-stream bases: R(1) 14, v1 8, v1 with w = 1 197. The most bases flagged on one day: R(1) 3, v1 with w = 1 54. The R(1) first-flag dates do not pile up the way the w = 1 dates do.

## 5. Development pilot (D81 b)

Run `20261101-214937-pilot4c` (commit `50035bc`). Seeds 0–4, node, K = 1%, random base: 5 scenarios and 50 degraded nodes per s. Delays in days.

| s | variant | KM median delay | found within 14 d | KM P(≤ 14 d) | recall at trace end | mean FDP | co-allocated false nodes per scenario | unrelated false nodes per scenario |
|---|---|---|---|---|---|---|---|---|
| 1.3 | v1 | 28.35 | 0.16 | 0.16 | 0.84 | 0.000 | 0.0 | 0.0 |
| 1.3 | v1, w = 1 | 10.60 | 0.54 | 0.54 | 0.92 | 0.092 | 20.6 | 0.2 |
| 1.3 | R(1) | 43.25 | 0.16 | 0.16 | 0.72 | 0.000 | 0.0 | 0.0 |
| 1.3 | R(0.25) | 54.35 | 0.00 | 0.00 | 0.62 | 0.001 | 0.2 | 0.0 |
| 1.3 | Rn(1) | 43.25 | 0.14 | 0.14 | 0.72 | 0.000 | 0.0 | 0.0 |
| 1.3 | Rn(0.25) | 54.25 | 0.00 | 0.00 | 0.62 | 0.001 | 0.2 | 0.0 |
| 2.0 | v1 | 19.45 | 0.26 | 0.26 | 0.90 | 0.000 | 0.4 | 0.0 |
| 2.0 | v1, w = 1 | 10.50 | 0.62 | 0.62 | 0.92 | 0.105 | 22.6 | 0.0 |
| 2.0 | R(1) | 24.30 | 0.20 | 0.20 | 0.92 | 0.000 | 0.0 | 0.0 |
| 2.0 | R(0.25) | 34.15 | 0.06 | 0.06 | 0.84 | 0.006 | 0.2 | 0.0 |
| 2.0 | Rn(1) | 24.30 | 0.20 | 0.20 | 0.92 | 0.000 | 0.0 | 0.0 |
| 2.0 | Rn(0.25) | 34.15 | 0.06 | 0.06 | 0.84 | 0.007 | 0.2 | 0.0 |
| 3.0 | v1 | 17.20 | 0.32 | 0.32 | 0.88 | 0.000 | 0.2 | 0.0 |
| 3.0 | v1, w = 1 | 9.80 | 0.62 | 0.62 | 0.92 | 0.084 | 18.8 | 0.0 |
| 3.0 | R(1) | 23.90 | 0.24 | 0.24 | 0.92 | 0.000 | 0.0 | 0.0 |
| 3.0 | R(0.25) | 35.35 | 0.02 | 0.02 | 0.84 | 0.003 | 0.2 | 0.0 |
| 3.0 | Rn(1) | 23.90 | 0.24 | 0.24 | 0.90 | 0.000 | 0.0 | 0.0 |
| 3.0 | Rn(0.25) | 35.35 | 0.02 | 0.02 | 0.84 | 0.003 | 0.2 | 0.0 |

v1 with w = 1 is fast, but it fails every null check, so its delays are not a usable result. It is shown as the reference D79 asks for.

## 6. Decoys and shocks (D81 c)

Random base, seeds 0–19, 4 values of s, 80 scenarios per kind. P(ever) ± SE.

| kind | v1 | v1, w = 1 | R(1) | R(0.25) | Rn(1) | Rn(0.25) |
|---|---|---|---|---|---|---|
| user_decoy | 0.050 ± 0.024 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| user_decoy_heavy | 0.050 ± 0.024 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| system_shock | 0.050 ± 0.024 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |

v1's flags are all on base rand19, whose null also flags, as in Phase 4. The 7% null rate of R(1) comes from bases outside 0–19, so these 240 scenarios sit on 20 bases with no R(1) null flag. A zero here does not mean R(1) can never flag on a decoy scenario.

## 7. Drift of the pair-weighted form

Sum over blocks of log(mean of e over active nodes), R denominator, averaged over 200 bases. This is a population average, not any single node. A fair update gives 0. Compare with the budget of 15.59 nats.

| stream | g_1.2 | g_1.3 | g_1.5 | g_2.0 | g_3.0 |
|---|---|---|---|---|---|
| real | 0.08 | 0.09 | 0.03 | -0.41 | -0.71 |
| iid uniform | -0.57 | -0.74 | -0.92 | -1.10 | -1.42 |

## 8. Choices made while implementing

- **Warm-up jobs:** the 6-h grid is extended backwards from the first evaluation, so jobs before it fall in 6-h blocks. D78 does not cover these jobs.
- **Bets:** the existing `Sc` set, built on unguarded S_c, because D78 uses no guard. Bets do not depend on placement, so the proof is not affected.
- **All f = 0 in a block:** e = 1.
- **Blocks:** a block is (t_{b−1}, t_b]. Jobs ending after the last evaluation are not used, since no evaluation could see them.
- **Jumper:** it mixes once per block and active node, so it takes fewer steps than v1, which mixes once per pair.
- **Rn variants:** added beyond D79, before any R result was seen, because of section 2.

## 9. Decisions

| # | Decision |
|---|---|
| D78 | Relative e-values implemented as specified; validity proof fails for mixed sizes (section 2) |
| D79 | R(1), R(0.25), v1 and v1 with w = 1 evaluated |
| D80 | Tests written first; item (1) found the proof failure |
| D81 | Development evaluation done |
| D82 | Sealing unchanged |

**Open, for Mahesh:** whether to adopt the Rn denominator, and whether R(1) or v1 is Layer 2.

## 10. Limitations

- Evaluated on the random base only. On the real placement, exchangeability within partition and block is an assumption [A] and has not been tested. Phase 4 found 9 real-placement null flags under v1.
- The pilot has 5 scenarios per s, on development seeds. The speed differences are not precise.
- All real-stream nulls share one trace (D52). For Rn this matches the proof, which fixes the p-values and randomises placement. For v1 it does not.
- With degraded nodes present, straggler contamination still applies (D72): co-allocated jobs are late, and they enter e_{n,b} for healthy nodes.
- Relative evidence cannot see a slowdown shared by all active nodes of a partition. K up to 5% was not tested in this phase.

## 11. Reproduce

```
uv run python -m hpcdiag.layer2.run_layer2 --config configs/phase4_layer2.yaml --mode pilot4c --workers 7
uv run python scripts/diag_4c_null.py --config configs/phase4_layer2.yaml --workers 7
uv run python scripts/phase4c_tables.py --run 20261101-214937-pilot4c --diag 20261101-215416-diag-4c
uv run python scripts/make_phase4c_report.py --pilot 20261101-214937-pilot4c --diag 20261101-215416-diag-4c --diag4b 20261031-230740-diag-d71
```

`uv run pytest -q`: 237 passed in 31.96s. Scenarios in the pilot: 255.
