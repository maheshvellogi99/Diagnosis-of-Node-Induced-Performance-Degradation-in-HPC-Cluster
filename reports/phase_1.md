# Phase 1 report — Layer 1 baseline: reproduce Menear et al. on Eagle

<!-- PREREGISTRATION:BEGIN -->
## 0. Pre-committed decision rules (recorded before any full-run result was seen)

Given by Mahesh on 9 Oct 2026 after the fast-mode review; committed to git before the full
runs started. The report generator copies this section verbatim; it is never regenerated.

1. Gate 1 stays on **pooled RMSE (main variant)**, as agreed. The gate metric is not changed.
2. If pooled RMSE misses 6.512–9.768 h, Phase 1 is recorded as **"harness validated, exact
   reproduction not achieved"** if and only if:
   (a) the long-job rule passes (median r = log(max(ŷ,1)/y) in the > 2 d bin, jobs with
   y > 30 s, is negative and the lowest of the five bins);
   (b) the **median per-split RMSE** of the main run is inside 6.512–9.768 h;
   (c) the report lists the known setup gaps: xgboost 3.2.0 default tree method and estimated
   base_score vs 1.7.1; fixed feature set vs the paper's per-split search; unknown aggregation.
   In that case decision D16 is added to PLAN.md §16 and docs/decisions.md saying exactly this,
   dated 9 Oct 2026. Otherwise Gate 1 is recorded as FAIL.
3. Sensitivity runs (approved): `tree_method="approx"` and `base_score=0.5`, both on the main
   variant, full 244 splits. They are **SENSITIVITY ONLY** and cannot change the Gate 1 outcome.
4. The report also includes: per-split RMSE over time (one line plot); the 5 worst splits with
   their dates and the number of test jobs > 2 d in each; the per-bin table for all variants.
<!-- PREREGISTRATION:END -->

_Results pending (full runs not yet started)._
