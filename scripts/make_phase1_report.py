# ruff: noqa: E501  (markdown report template: table rows are long by nature)
"""Render reports/phase_1.md from the Phase 1 run outputs (PLAN.md §7.6).

Numbers come from runs/phase1/<run_id>/{metrics.json, per_split.csv, per_bin.csv,
predictions.parquet}. The pre-registration section (§0) of the existing report is copied
verbatim. Figures are written to reports/figures/.

    uv run python scripts/make_phase1_report.py --runs main=<id> v1_rt30=<id> v2_nodup=<id> \
        sens_approx=<id> sens_base05=<id>
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys

import duckdb
import numpy as np
import pandas as pd

from hpcdiag.eval.plots import rmse_over_time
from hpcdiag.utils.run import PROJECT_ROOT, load_config, resolve

ORDER = ["main", "v1_rt30", "v2_nodup", "sens_approx", "sens_base05"]
DESC = {
    "main": "COMPLETED+TIMEOUT, walltime present; XGBoost defaults",
    "v1_rt30": "main + run_time_s > 30",
    "v2_nodup": "main − exact duplicate rows (D10)",
    "sens_approx": "SENSITIVITY ONLY: main, tree_method=approx",
    "sens_base05": "SENSITIVITY ONLY: main, base_score=0.5",
}
PRE_RE = re.compile(r"<!-- PREREGISTRATION:BEGIN -->.*?<!-- PREREGISTRATION:END -->", re.S)


def sh(cmd: list[str]) -> str:
    r = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def f(v, d=3) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        return "n/a" if not np.isfinite(v) else f"{v:,.{d}f}"
    return str(v)


def tree_method(cfg: dict) -> str:
    """tree_method recorded in a saved booster config (searched, since the layout varies)."""
    stack = [cfg]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            if "tree_method" in x:
                return str(x["tree_method"])
            stack.extend(x.values())
    return "not recorded in config"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="variant=run_id ...")
    ap.add_argument("--config", default="configs/phase1_menear.yaml")
    a = ap.parse_args()
    runs = dict(x.split("=", 1) for x in a.runs)
    missing = [v for v in ORDER if v not in runs]
    if missing:
        raise SystemExit(f"missing runs for {missing}")
    cfg = load_config(a.config)
    g = cfg["gate1"]

    report = resolve("reports/phase_1.md")
    pre = PRE_RE.search(report.read_text())
    if not pre:
        raise SystemExit("pre-registration section not found in reports/phase_1.md")

    M, PS, PB, CM, TM = {}, {}, {}, {}, {}
    for v in ORDER:
        d = resolve(f"runs/phase1/{runs[v]}")
        M[v] = json.loads((d / "metrics.json").read_text())
        if M[v]["variant"] != v or M[v]["fast"]:
            raise SystemExit(f"run {runs[v]} is not a full run of {v}")
        PS[v] = pd.read_csv(d / "per_split.csv", parse_dates=["split_t", "train_end_max"])
        PB[v] = pd.read_csv(d / "per_bin.csv")
        CM[v] = (d / "git_commit.txt").read_text().strip()
        TM[v] = tree_method(json.loads((d / "xgb_config.json").read_text()))
    m, ps = M["main"], PS["main"]
    lo, hi = m["gate1"]["band_h"]

    # ---- Gate 1 outcome (pre-registered rules 1-2)
    pooled = m["rmse_h_pooled"]
    med = m["rmse_h_median_of_splits"]
    in_band = lo <= pooled <= hi
    qual = m["qualitative"]["pass"]
    med_in_band = lo <= med <= hi
    if in_band and qual:
        outcome = "PASS"
    elif (not in_band) and qual and med_in_band:
        outcome = "FAIL on pooled RMSE — recorded as \"harness validated, exact reproduction not achieved\" (rule 2, D16)"
    else:
        outcome = "FAIL"

    # ---- figures
    figs = resolve("reports/figures")
    figs.mkdir(parents=True, exist_ok=True)
    rmse_over_time(ps["split_t"], ps["rmse_h"], (lo, hi), g["reference_rmse_h"],
                   f"Eagle, main variant: RMSE per split ({len(ps)} splits, test day after each split time)",
                   figs / "phase1_rmse_per_split.png")
    shutil.copy(resolve(f"runs/phase1/{runs['main']}/figures/pred_vs_actual_by_bin.png"),
                figs / "phase1_pred_vs_actual_by_bin_main.png")

    # ---- 5 worst splits (main)
    pr = pd.read_parquet(resolve(f"runs/phase1/{runs['main']}/predictions.parquet"))
    long_n = pr.assign(long=pr["y"] >= 48 * 3600).groupby("split_t")["long"].sum()
    sse = pr.assign(se=(pr["yhat"] - pr["y"]) ** 2).groupby("split_t")["se"].sum()
    worst = ps.nlargest(5, "rmse_h").copy()
    worst["n_gt_2d"] = worst["split_t"].map(long_n).astype(int)
    worst["sse_share"] = worst["split_t"].map(sse) / sse.sum()

    # ---- skipped splits (logged by the runner), with window counts recomputed from the data
    log_txt = resolve(f"runs/phase1/{runs['main']}/log.txt").read_text()
    skipped = [pd.Timestamp(x) for x in re.findall(r"split (\S+ \S+) skipped", log_txt)]
    con = duckdb.connect()
    con.sql("SET TimeZone='UTC'")
    jobs = str(resolve(cfg["data"]["jobs"])).replace("'", "''")
    states = ", ".join(f"'{x}'" for x in cfg["filters"]["states"])
    base = f"from '{jobs}' where state_raw in ({states}) and walltime_req_s is not null"
    sk_rows = []
    for t in skipped:
        tr = con.sql(f"select count(*) {base} and end_time >= TIMESTAMP '{t - pd.Timedelta(days=100)}' and end_time < TIMESTAMP '{t}'").fetchone()[0]
        te = con.sql(f"select count(*) {base} and submit_time >= TIMESTAMP '{t}' and submit_time < TIMESTAMP '{t + pd.Timedelta(days=1)}'").fetchone()[0]
        sk_rows.append((t, tr, te))
    if len(skipped) + len(ps) != m["n_splits_full_list"]:
        raise SystemExit("skipped + used splits != full list")

    pytest_lines = sh(["uv", "run", "pytest", "-q"]).splitlines()
    pytest_summary = [ln for ln in pytest_lines if re.search(r"\d+ passed", ln)][-1]
    ruff = sh(["uv", "run", "ruff", "check", "."]).splitlines()[-1]

    s = "# Phase 1 report — Layer 1 baseline: reproduce Menear et al. on Eagle\n\n"
    s += pre.group(0) + "\n\n"
    s += f"""_Sections 1–9 generated by `scripts/make_phase1_report.py` from the run folders listed in §3.
Do not edit by hand; re-run the script._

## 1. Result

**Gate 1: {outcome}**

| condition | value | result |
|---|---|---|
| pooled RMSE (main, all {f(m["test_jobs"])} test jobs of {m["n_splits_used"]} non-empty splits; 244 split times) in [{lo:.3f}, {hi:.3f}] h | {pooled:.3f} h | {"pass" if in_band else "**fail**"} |
| long-job rule: median r in > 2 d bin negative and lowest of the five bins | {m["qualitative"]["median_r_long_bin"]:+.3f} (negative: {f(m["qualitative"]["negative"])}, lowest: {f(m["qualitative"]["lowest"])}) | {"pass" if qual else "**fail**"} |
| rule 2(b): median per-split RMSE (main) in band | {med:.3f} h | {"inside" if med_in_band else "outside"} |
| rule 2(c): known setup gaps listed | §2 | listed |

Reference: Menear et al. report RMSE 8.14 h for per-split feature selection (F6). The paper does
**not** say whether this is pooled over all test jobs or averaged over splits; this report gives
both (§4). Gate band = 8.14 h ± 20% = [{lo:.3f}, {hi:.3f}] h (PLAN.md rounds it to 6.5–9.8 h).
Sensitivity runs (§6) are labelled SENSITIVITY ONLY and do not change this outcome.

## 2. Known setup gaps vs the paper (rule 2c)

1. **xgboost version**: {m["xgboost_version"]} here vs 1.7.1 in the paper. Two defaults may differ:
   the default tree method (saved booster config of the main run records
   `tree_method = {TM["main"]}`; `sens_approx` records `{TM["sens_approx"]}`) and `base_score`, which 3.2.0 estimates from each training set
   (observed: per-split base_score ranges {ps["base_score"].min():,.0f}–{ps["base_score"].max():,.0f} s in the main run),
   while 1.7.1 is believed to use a fixed 0.5 (xgboost 2.0 release notes; not verified in this
   repo; checked by the `sens_base05` run).
2. **Feature set**: fixed (numerical walltime_req_s, procs_req, mem_req, nodes_req, gpus_req;
   categorical user, partition) per D6, vs the paper's per-split feature-set search
   (8.14 h is the per-split-selection result; 8.45 h with previous-split selection, F6).
3. **Aggregation**: unknown in the paper (pooled vs mean of splits).
4. Row filter: COMPLETED + TIMEOUT with non-null walltime (D5, addendum A). The paper states
   cancelled/failed dropped and timeouts kept; it does not mention OUT_OF_MEMORY, NODE_FAIL,
   PENDING, RUNNING or null walltime handling.

## 3. Runs and commands

```bash
"""
    for v in ORDER:
        s += f"uv run python -m hpcdiag.layer1.run_menear --config configs/phase1_menear.yaml --variant {v}   # {runs[v]}\n"
    s += f"uv run python scripts/make_phase1_report.py --runs {' '.join(f'{v}={runs[v]}' for v in ORDER)}\n```\n\n"
    s += "Fast mode (12 splits) was run first for main, v1_rt30, v2_nodup and reviewed before the full runs.\n\n"
    dirty = [v for v in ORDER if CM[v].endswith("-dirty")]
    if dirty:
        s += (f"Provenance note: runs {', '.join(dirty)} recorded a `-dirty` commit. While they ran, the only "
              "uncommitted changes were `src/hpcdiag/eval/plots.py` (a new function `rmse_over_time` appended; "
              "not called by the runner), the untracked `scripts/make_phase1_report.py`, and an addendum to "
              "`reports/phase_0.md` — none on the training/prediction path. These were committed right after "
              "the runs. The gated main run is on a clean commit.\n\n")
    s += "| variant | run id | git commit | rows after filter | splits | test jobs | seconds (fit+predict total) |\n|---|---|---|---|---|---|---|\n"
    for v in ORDER:
        x = M[v]
        s += f"| {v} | `{runs[v]}` | `{CM[v][:10]}` | {f(x['rows_after_variant'])} | {x['n_splits_used']} | {f(x['test_jobs'])} | {x['timing_s']['splits_total']:.0f} |\n"
    s += f"""
- xgboost {m["xgboost_version"]}; model params: main `{m["model_params"] or "defaults"}`; full booster config in each run's `xgb_config.json` (identical across splits apart from base_score).
- Splits: {m["n_splits_full_list"]} in the full list, first {m["first_split"]}, last {m["last_split"]}; all variants use the same split times (from the main-filtered rows).
- Splits used: {len(ps)} of {m["n_splits_full_list"]}. {len(sk_rows)} split(s) were skipped because the train or test window was empty (main filter):

| skipped split time | training jobs | test jobs |
|---|---|---|
""" + "".join(f"| {t} | {f(int(a))} | {f(int(b))} |\n" for t, a, b in sk_rows) + f"""
  A skipped split contributes no test jobs, so pooled metrics are unaffected; mean/median of
  per-split RMSE are over the {len(ps)} non-empty splits. Whether the paper's 244 splits include
  empty test days is unknown.
- Leakage check: max(train end_time − split time) over splits = {m["leakage_check"]["max_train_end_minus_split_s"]:.0f} s (must be < 0).
- Tests: `{pytest_summary}`; lint: `{ruff}`.

## 4. Overall metrics

RMSE/MAE use unclipped predictions on all test jobs. TARE A_geo / UnderRate use predictions
clipped to ≥ 1 s on test jobs with run_time > 30 s (addendum D).

| variant | RMSE pooled (h) | RMSE mean of splits (h) | RMSE median of splits (h) | MAE pooled (h) | MAE mean of splits (h) | A_geo | UnderRate | predictions < 1 s (of which < 0) |
|---|---|---|---|---|---|---|---|---|
"""
    for v in ORDER:
        x = M[v]
        s += (f"| {v} | {x['rmse_h_pooled']:.3f} | {x['rmse_h_mean_of_splits']:.3f} | {x['rmse_h_median_of_splits']:.3f} | "
              f"{x['mae_h_pooled']:.3f} | {x['mae_h_mean_of_splits']:.3f} | {x['tare_pooled']['a_geo']:.4f} | "
              f"{x['tare_pooled']['under_rate']:.4f} | {f(x['predictions']['n_pred_lt_1s'])} ({f(x['predictions']['n_pred_lt_0'])}) |\n")
    s += "\n" + "".join(f"- {v}: {DESC[v]}\n" for v in ORDER)

    s += """
## 5. Error by actual-runtime bin (all variants)

`median r` = median of log(max(ŷ,1 s)/y) and `UnderRate` = share with ŷ < y, both on jobs with
y > 30 s (`n_r`). RMSE, MAE and mean signed error (ŷ − y, hours; context only, not the gate)
use all jobs in the bin.

"""
    for v in ORDER:
        t = PB[v]
        s += f"**{v}**{' — SENSITIVITY ONLY' if v.startswith('sens') else ''}\n\n"
        s += "| bin | n | RMSE (h) | MAE (h) | mean signed error (h) | mean actual (h) | mean predicted (h) | n_r | median r | UnderRate |\n|---|---|---|---|---|---|---|---|---|---|\n"
        for _, r in t.iterrows():
            s += (f"| {r['bin']} | {f(int(r['n']))} | {f(r['rmse_h'])} | {f(r['mae_h'])} | {r['mean_signed_error_h']:+.3f} | "
                  f"{f(r['mean_actual_h'])} | {f(r['mean_pred_h'])} | {f(int(r['n_r']))} | {r['median_r']:+.3f} | {f(r['under_rate_r'])} |\n")
        q = M[v]["qualitative"]
        s += f"\nLong-job rule: median r (> 2 d) = {q['median_r_long_bin']:+.3f}; {'pass' if q['pass'] else 'fail'}.\n\n"

    s += """## 6. Sensitivity runs (SENSITIVITY ONLY — cannot change Gate 1)

| run | pooled RMSE (h) | Δ vs main (h) | median per-split RMSE (h) | long-job rule |
|---|---|---|---|---|
"""
    for v in ["main", "sens_approx", "sens_base05"]:
        x = M[v]
        s += (f"| {v} | {x['rmse_h_pooled']:.3f} | {x['rmse_h_pooled'] - pooled:+.3f} | {x['rmse_h_median_of_splits']:.3f} | "
              f"{'pass' if x['qualitative']['pass'] else 'fail'} |\n")
    s += "\n| run | tree_method in saved config | fit+predict seconds (all splits) | max abs prediction difference vs main (s) | predictions identical to main |\n|---|---|---|---|---|\n"
    for v in ["main", "sens_approx", "sens_base05"]:
        pv = pd.read_parquet(resolve(f"runs/phase1/{runs[v]}/predictions.parquet"), columns=["row_id", "split_t", "yhat"])
        if not (pv["row_id"].to_numpy() == pr["row_id"].to_numpy()).all():
            raise SystemExit(f"{v}: prediction rows not aligned with main")
        diff = float(np.abs(pv["yhat"].to_numpy() - pr["yhat"].to_numpy()).max())
        s += f"| {v} | `{TM[v]}` | {M[v]['timing_s']['splits_total']:.0f} | {diff:,.1f} | {f(diff == 0.0)} |\n"

    s += f"""
## 7. Per-split RMSE over time (main)

![RMSE per split](figures/phase1_rmse_per_split.png)

Per-split RMSE (main): min {ps["rmse_h"].min():.3f} h, median {med:.3f} h, mean {ps["rmse_h"].mean():.3f} h,
max {ps["rmse_h"].max():.3f} h; {int(((ps["rmse_h"] >= lo) & (ps["rmse_h"] <= hi)).sum())} of {len(ps)} splits inside the band,
{int((ps["rmse_h"] > hi).sum())} above, {int((ps["rmse_h"] < lo).sum())} below.

**5 worst splits (main)**

| split time | RMSE (h) | test jobs | test jobs with actual run time ≥ 2 d | share of all squared error (pooled) |
|---|---|---|---|---|
"""
    for _, r in worst.iterrows():
        s += f"| {r['split_t']} | {r['rmse_h']:.3f} | {f(int(r['test_rows']))} | {f(int(r['n_gt_2d']))} | {r['sse_share']:.3f} |\n"
    s += f"""
Across all {len(ps)} splits the main run has {f(int(long_n.sum()))} test jobs ≥ 2 d
({100 * long_n.sum() / len(pr):.2f}% of test jobs).

## 8. Predicted vs actual by bin (main)

![Predicted vs actual](figures/phase1_pred_vs_actual_by_bin_main.png)

Left: all main test jobs, log-log; predictions and actuals below 1 s (including negative
predictions) are drawn at 1 s, which is the flat row at the bottom. Right: median r per bin.

## 9. Other checks

- Test jobs with a user unseen in that split's training window (main): {f(int(ps["unseen_user_test"].sum()))}
  ({100 * ps["unseen_user_test"].sum() / ps["test_rows"].sum():.2f}%); unseen partition: {f(int(ps["unseen_partition_test"].sum()))}.
- Training rows per split (main): min {f(int(ps["train_rows"].min()))}, median {f(int(ps["train_rows"].median()))}, max {f(int(ps["train_rows"].max()))}.
- Test rows per split (main): min {f(int(ps["test_rows"].min()))}, median {f(int(ps["test_rows"].median()))}, max {f(int(ps["test_rows"].max()))}.
- Nothing was tuned on test splits. Gate 1 uses only the main variant.
"""
    report.write_text(s)
    print(f"wrote {report}; Gate 1: {outcome}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
