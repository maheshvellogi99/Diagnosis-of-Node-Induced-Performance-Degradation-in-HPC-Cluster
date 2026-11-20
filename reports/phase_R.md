# Phase R report — final review package (D121–D125)

20 Nov 2026. Development seeds only. Sealed seeds 20–39 and null bases 200–399 were not touched. Constraints
held:
- No change to `src/`: `git diff --stat 9c5a1c1 HEAD -- src` is empty.
- The guard passes: `PASS: 6 frozen files checked, policy 'ig_d109'`.
- No installs and no network calls, except `git push`.
- All 460 tests pass (`uv run pytest -q`), and `uv run ruff check .` is clean.

| Decision | Done | Evidence |
|---|---|---|
| D121(a) | Layer 1 τ 0.9 and τ 0.5 models re-fitted with the frozen config (seed 0, 45-day window, freeze 2020-06-19 15:55:59, 27,659 training jobs). Saved as LightGBM text, with the feature encoding. | Run 20261120-191921-layer1-export. Max \|diff\| to q_hat and q50_log of run 20261010-003334-pm100-E-main-full on all 102,120 jobs: **0.0**, before and after reloading the text files (tolerance 1e-9). The loading snippet in `models/README.md` also gives 0.0 against q_hat (checked interactively). |
| D121(b,c) | Layer 2: g_s0 bets on S_c and the settings. Layer 3: settings, state table, D96 state bets, H0 bin probabilities, fixed healthy pool, telemetry counts and likelihoods. Antarex random forest (joblib, all 26,359 five-class windows), with the two metric files. | Run 20261120-193043-components; SHA-256 of every file in its `metrics.json` |
| D121(d) | `deliverables/models/README.md`: what each file is and how to load it. Layers 2–3 are procedures (code + settings), not trained models, except the telemetry random forest. The exported forest was not scored on held-out data; the reported scores come from runs 20261102-003936-classifier and 20261102-004249-d94-diag. | text in `scripts/build_deliverables.py` |
| D122 | `scripts/demo_pipeline.py` on `node__rand00__K1__s1.3__seed00`, B 7200 → `deliverables/demo/`. 10/10 degraded nodes flagged within 14 d; median delay 4.83 d; 0 false nodes; 10/10 causes correct; 11,001 investigations. | Run 20261120-192256-demo: **9.2 s**; result identical to the dev5c row of run 20261113-223241-dev5c (all fields except timings) |
| D123 | `reports/FINAL_RESULTS.md`, every number with its run id, labelled "development seeds; sealed evaluation pending". New Layer 1 numbers (baselines, S_c calibration) come from run 20261120-192518-final-l1. The Prodigy binary score 0.970 ± 0.015 is sourced "mid-review report (Table 2), Sep 2026; not re-run in this repo". | — |
| D124 | 8 number fixes in `G34_Final_Report.tex` (table below). No number changes in `architecture_fig.tex`. Claims I think are wrong but are not numbers were left unchanged and are listed for you. **PDF not rebuilt.** | Commits c5dab7e (as received) and d69895c (fixes): `git diff c5dab7e d69895c` |
| D125 | `deliverables/` built by `scripts/build_deliverables.py`, git-ignored: **63 files, 23M** (`du -sh deliverables`). No raw data and no parquet file. | — |

## D124 — changes to `Claude outputs/report/G34_Final_Report.tex`

| # | Where | Old → new | Source |
|---|---|---|---|
| 1 | Abstract, false-alarm range of the full system | 0.065–0.070 → **0.060–0.070** | phase_5c.md §3, run 20261113-223241-dev5c: B 0 gives 0.070; B 300/1800 give 0.065; B 7200 gives 0.065 (fixed pool) and 0.060 (pool per base) |
| 2 | Table l1pred, constant median, within 2× | 19.5% → **25.2%** | run 20261120-192518-final-l1. Constant = median of the training jobs, 641 s. |
| 3 | Table l1pred, constant median, RMSE | 6.24 → **6.28** | same |
| 4 | Table l1pred, constant median, R² on log runtime | 0.00 → **−0.08** | same |
| 5 | Table l1pred, requested walltime, Spearman | -- → **0.53** | same. Walltime varies, so Spearman is defined. |
| 6 | Table l1cal, per user, p ≤ 0.05 | 0.086 → **0.085** | run 20261010-003334-pm100-E-main-full: 0.08546 |
| 7 | Quality controls, decision count | 120 → **125** | PLAN §16 has D1–D125 |
| 8 | Gate table, row 4a / 5a, measured | 0.065–0.070 → **0.040–0.065** | Gate 4a, run 20261010-094328-full: null 0.040, decoy and shock 0.050. Gate 5a, run 20261113-223241-dev5c: 0.065 |

I could not find in any run the source of the old constant-median values (19.5%, 6.24, 0.00).

**Verified, no change:**
- Dataset facts (data cards): 231,238 jobs, 980 nodes, 3 partitions, May–Oct 2020 (submit 2020-05-05 to
  2020-10-13), Eagle 11,030,377 rows, Zenodo/OEDI ids, CC-BY 4.0.
- 3,281 scenarios: `ls data/processed/scenarios`, excluding `_bases` and the two index files, gives node
  1,440, node_partial 480, null 401, and 320 each of user_decoy, user_decoy_heavy and system_shock. This
  count includes the sealed scenarios; their names were only listed, never read.
- 460 tests.
- Layer 1 settings (200 trees, 0.05, 50 leaves, τ 0.9, features, 45 days chosen from 30/45/60, freeze
  19 June 2020, W 7 d, ≥ 30 references).
- Table l1pred, LightGBM column (37.2%, 5.55, 2.78, 0.06, 0.49, 0.874, 0.243) and walltime 18.6%, 11.28,
  7.18, −1.07; constant MAE 2.55.
- Table l1cal, other cells; 0.124%; D21 KS 0.023.
- Eagle 7.38 h / 1,358,770; 8.14 h; gate band ±20%.
- Layer 2 settings (s0 set, J 0.01, α 0.1, m 980, 6 h).
- Table l2base (phase_4.md §7, K 1%); lag-1 autocorrelation 0.29; 98.5% → 7.0%.
- Table l2speed; tables l3budget, l3grid and debt (phase_5c.md §3–6), including cluster-time shares.
- Layer 2 ranking 60.0% / 13.5% / 12.0% (phase_5.md §4).
- 2.1 nats (phase_5b.md: −2.11); misspecification ±50% and σ doubled (`probes.perturbed` spread 0.5,
  true σ 0.1); 0.985 → 0.615 and 0.991 → 0.889.
- Probe costs and σ, pool 1000, log(1/J) = 4.6, τ 0.95, 7-day clearing.
- Telemetry table (0.883, 1.000, 0.69/0.85; Prodigy 0.970 kept, from the mid-review report).
- "Recall 0 at s = 1.3 without centring in the first pilot": `kappa_S_none` (first-pilot primary) is
  0.000 / 0.000 at s 1.3 (phase_4.md §7).
- Phase gate results; s = 1.1 at most 18% and recall 0.73; Phase 6 plan (2 × 3 days, 1000 healthy runs, 4
  anomalies × 2 intensities × ≥ 10 repetitions).

**Claims I think are wrong or loose, left unchanged (D124 allows only number fixes). Please decide:**
1. §Results L1: "The median model predicts runtime roughly twice as well as simple baselines". Measured:
   within 2× is 0.372 against 0.252 and 0.186. RMSE is 5.55 against 6.28 and 11.28. MAE is 2.78, worse
   than the constant median's 2.55. Walltime has the higher Spearman (0.53 against 0.49). "Twice" holds
   only against requested walltime.
2. §Discussion: "No configuration exceeded the false-alarm bound". This is false for the mid-review
   weighted form, which flags 0.985 of real-stream nulls (Phase 4c), as the report itself says in §5. It
   is also false for Phase 4 variant `kappa_none_modeE` (1.000). It holds for the final design.
3. §Discussion: the four real-placement nodes are "each dominated by one user". D92 says user-dominated:
   top-user share above the 90th percentile of all nodes, 7–12 users per node. Not one user.
4. Literature numbers (98%, F1 0.95 and > 0.876, Menear 8.14 h) and bibliography entries cannot be checked
   from this repo. 8.14 h agrees with PLAN (F6).

**`architecture_fig.tex`:** the settings in the boxes agree with the frozen config: τ 0.9, last 7 days,
6-h block, α 0.1, 980 nodes, the probe list, budget B and the 0.95 report threshold. No number was changed.
One loose point is left unchanged: the Layer 1 box shows S_j = log T_j − q̂ feeding Layer 2, but Layer 2
consumes the centred score S_c (D60). The report text states this in its changes table.

**PDF:** not rebuilt. `pdflatex` and `latexmk` are not installed. `tectonic` is installed but has no local
cache and would download its TeX bundle, which the no-network rule forbids. `deliverables/report/Project
Report.pdf` is the PDF as received, so it still shows the 8 old numbers. The deliverables README says so.

## Disclosures

- **Superseded run.** Components run 20261120-192023-components stored 244 metric names as the forest's
  `feature_names`. The model has 2,684 inputs (244 metrics × 11 statistics). I fixed the export (commit
  01455a6) and re-ran it as 20261120-193043-components. The forest itself was not changed by the fix: same
  data, settings and seed.
- **Two plain-score calibration rows.** Phase 2 (`post_scores`) and the Layer 2 population (`layer2.prep`)
  differ at the 5th decimal: p ≤ 0.05 is 0.052346 against 0.052375. I did not trace why. Both rows are in
  FINAL_RESULTS §1.3; the gate value is Phase 2's.
- **Tag already on origin.** `git ls-remote` showed `v1.0-dev-frozen` (→ 0b9f5d8) already on origin, so
  pushing the tag changes nothing.
- **Report sources committed.** `G34_Final_Report.tex`, `architecture_fig.tex`, `architecture_standalone.tex`
  and the logo were committed as received, then fixed in a separate commit. The PDFs, PNG/SVG and
  `paper_v0.tex` under `Claude outputs/` stay untracked.
- **Not included.** `architecture_ieee.png` does not exist. As you chose, `Claude outputs/report/architecture.png`
  and `architecture_standalone.pdf` are used. The older top-level `Claude outputs/architecture.png/svg` are
  not included.
- **Comparison from an older layer.** The Layer 2 ranking comparison (FINAL_RESULTS §3.4) is from the first
  Layer 3 version (D86), not the frozen primary.

## Commands

```
uv run python scripts/export_layer1.py --config configs/phase2_conformal.yaml --ref 20261010-003334-pm100-E-main-full --out deliverables/models/layer1   # 20261120-191921-layer1-export
uv run python scripts/export_components.py --config configs/phase5.yaml --out deliverables/models                                                     # 20261120-193043-components
uv run python scripts/demo_pipeline.py --config configs/phase5.yaml --scenario node__rand00__K1__s1.3__seed00 --budget 7200 --ref 20261113-223241-dev5c --out deliverables/demo   # 20261120-192256-demo
uv run python scripts/final_layer1_metrics.py --config configs/phase2_conformal.yaml --ref 20261010-003334-pm100-E-main-full --l2-config configs/phase4_layer2.yaml   # 20261120-192518-final-l1
uv run python scripts/build_deliverables.py
uv run python scripts/check_frozen.py --config configs/phase5.yaml --policy ig_d109
uv run pytest -q
```

## Addendum (D126, 20 Nov 2026)

The architect revised the report, and the revision is accepted (D126):
- **Wording.** `G34_Final_Report.tex` was reworded for D124 items 1–3: the baseline comparison now states
  the MAE and Spearman exceptions; the false-alarm claim is limited to the final design and mentions the
  98.5% failure of the mid-review form; the four real-placement nodes are described as in D92.
- **Centred score.** The centred score is now described in the Layer 1 section and in the
  `architecture_fig.tex` score box.
- **Rebuilt files.** `Project Report.pdf`, `architecture.png` and `architecture_standalone.pdf` were rebuilt
  by the architect, so the PDF now matches the .tex. I did not rebuild or inspect the PDF here.
- **Numbers checked.** I checked the numbers added in the new wording (37.2%, 25.2%, 18.6%, 2.55 h,
  2.78 h, 0.53, 0.49, 98.5%, 7–12 users, 90th percentile, at least 3 jobs). All match FINAL_RESULTS, D92
  and `configs/phase4_layer2.yaml`.
- **One small omission (not a number, left unchanged).** "at least 3 jobs; otherwise the same user's jobs"
  does not say that the user level also needs at least 3 jobs. Below that, the score is left uncentred
  (`src/hpcdiag/layer2/centred.py`, level 2).
- **Deliverables.** `deliverables/` was rebuilt, and its README now says the PDF matches the .tex.
