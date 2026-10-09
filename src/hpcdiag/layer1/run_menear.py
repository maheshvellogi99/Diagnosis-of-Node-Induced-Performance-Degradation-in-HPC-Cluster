"""Phase 1 runner: reproduce Menear et al. on Eagle (PLAN.md §9 Phase 1).

    uv run python -m hpcdiag.layer1.run_menear --config configs/phase1_menear.yaml \
        --variant main [--fast]

Outputs in runs/phase1/<run_id>/: metrics.json, per_split.csv, per_bin.csv,
predictions.parquet, xgb_config.json, figures/pred_vs_actual_by_bin.png (+ run files).
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost

from hpcdiag.eval import metrics as M
from hpcdiag.eval.plots import pred_vs_actual_by_bin
from hpcdiag.layer1 import data as D
from hpcdiag.layer1.baseline import config_fingerprint, run_split, summarise_predictions
from hpcdiag.layer1.splits import TimeIndex, fast_subset, split_times
from hpcdiag.utils.checksums import verify
from hpcdiag.utils.run import Run, resolve, run_guarded, start_run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--fast", action="store_true", help="fast mode (fast_mode.n_splits splits)")
    a = ap.parse_args()
    tag = f"{a.variant}-{'fast' if a.fast else 'full'}"
    run = start_run("phase1", a.config, tag=tag)
    return run_guarded(run, lambda r: _run(r, a.variant, a.fast))


def _run(run: Run, variant_name: str, fast: bool) -> int:
    cfg, log = run.config, run.log
    if variant_name not in cfg["variants"]:
        raise ValueError(f"unknown variant {variant_name}")
    variant = cfg["variants"][variant_name] or {}
    dcfg, fcfg, scfg, ecfg = cfg["data"], cfg["features"], cfg["splits"], cfg["evaluation"]

    problems = verify(resolve(dcfg["raw_dir"]), resolve(dcfg["checksums_file"]))
    if problems:
        raise RuntimeError(f"raw checksum problems: {problems}")

    feats = fcfg["numeric"] + fcfg["categorical"]
    t0 = time.time()
    df = D.load_jobs(resolve(dcfg["jobs"]), feats, cfg["filters"]["states"],
                     cfg["filters"]["require_walltime"])
    log.info("filtered jobs (main filter): %d rows", len(df))

    # Split times come from the main-filtered rows, so every variant uses the same splits.
    times_all = split_times(df["submit_time"], scfg["first_split_offset_days"],
                            scfg["split_every_days"], scfg["test_days"])
    times = fast_subset(times_all, cfg["fast_mode"]["n_splits"]) if fast else times_all
    log.info("split times: %d in full list, %d used (%s .. %s)", len(times_all), len(times),
             times[0], times[-1])

    dups = None
    if variant.get("drop_exact_duplicates"):
        p = resolve(dcfg["exact_dup_row_ids"])
        if not p.exists():
            n = D.build_dup_table(resolve(dcfg["raw"]), p)
            log.info("built %s (%d rows)", p, n)
        dups = pq.read_table(p)["row_id"].to_numpy()
        if len(dups) != dcfg["expected_exact_dups"]:
            raise RuntimeError(f"exact duplicates {len(dups)} != expected "
                               f"{dcfg['expected_exact_dups']}")
    n_before = len(df)
    df = D.apply_variant(df, variant, dups)
    log.info("variant %s: %d rows (%d removed)", variant_name, len(df), n_before - len(df))
    index = TimeIndex.build(df["end_time"], df["submit_time"])
    load_s = time.time() - t0

    per_split, preds, cfg_ref, fit_times = [], [], None, []
    for i, t in enumerate(times):
        ts = time.time()
        res = run_split(df, index, t, scfg["train_days"], scfg["test_days"], fcfg["numeric"],
                        fcfg["categorical"], cfg["target"], fcfg["unseen_code"],
                        cfg["model"]["params"])
        dt_s = time.time() - ts
        if res is None:
            log.warning("split %s skipped: empty train or test", t)
            continue
        fp = config_fingerprint(res.config)
        if cfg_ref is None:
            cfg_ref = res.config
            run.write_json("xgb_config.json", cfg_ref)
        elif fp != config_fingerprint(cfg_ref):
            raise RuntimeError(f"xgboost config changed at split {t}")
        p = res.predictions.assign(split_t=t)
        y, yh = p["y"].to_numpy(), p["yhat"].to_numpy()
        tare = M.tare_metrics(y, yh, p["nodes_req"].to_numpy(),
                              ecfg["tare_min_run_time_s_exclusive"], ecfg["tare_clip_pred_min_s"])
        per_split.append({
            "split_t": t, "train_rows": res.train_rows, "test_rows": res.test_rows,
            "train_end_max": res.train_end_max, "rmse_h": M.rmse_h(y, yh), "mae_h": M.mae_h(y, yh),
            "a_geo": tare["a_geo"], "under_rate": tare["under_rate"], "n_tare": tare["n"],
            "n_pred_lt_1s": int((yh < 1).sum()),
            "users_train": res.n_classes["user"], "partitions_train": res.n_classes["partition"],
            "unseen_user_test": res.unseen_test["user"],
            "unseen_partition_test": res.unseen_test["partition"], "seconds": dt_s,
        })
        preds.append(p)
        fit_times.append(dt_s)
        log.info("split %d/%d %s train=%d test=%d rmse_h=%.3f (%.1fs)", i + 1, len(times),
                 t.date(), res.train_rows, res.test_rows, per_split[-1]["rmse_h"], dt_s)

    ps = pd.DataFrame(per_split)
    pr = pd.concat(preds, ignore_index=True)
    y, yh = pr["y"].to_numpy(), pr["yhat"].to_numpy()
    tare = M.tare_metrics(y, yh, pr["nodes_req"].to_numpy(),
                          ecfg["tare_min_run_time_s_exclusive"], ecfg["tare_clip_pred_min_s"])
    table = M.per_bin_table(y, yh, ecfg["bins_hours"], ecfg["bin_labels"],
                            ecfg["qualitative_min_run_time_s_exclusive"])
    qual = M.qualitative_gate(table, ecfg["bin_labels"][-1])
    g = cfg["gate1"]
    ref, band = g["reference_rmse_h"], g["band_rel"]
    lo, hi = ref * (1 - band), ref * (1 + band)
    pooled = M.rmse_h(y, yh)
    rmse_ok = bool(lo <= pooled <= hi)

    m = {
        "variant": variant_name, "fast": fast, "xgboost_version": xgboost.__version__,
        "model_params": cfg["model"]["params"],
        "n_splits_full_list": len(times_all), "n_splits_used": len(ps),
        "first_split": times_all[0], "last_split": times_all[-1],
        "split_times_used": [str(t) for t in ps["split_t"]],
        "rows_after_main_filter": n_before, "rows_after_variant": len(df),
        "test_jobs": len(pr),
        "rmse_h_pooled": pooled, "rmse_h_mean_of_splits": float(ps["rmse_h"].mean()),
        "rmse_h_median_of_splits": float(ps["rmse_h"].median()),
        "mae_h_pooled": M.mae_h(y, yh), "mae_h_mean_of_splits": float(ps["mae_h"].mean()),
        "tare_pooled": tare,
        "a_geo_mean_of_splits": float(ps["a_geo"].mean()),
        "under_rate_mean_of_splits": float(ps["under_rate"].mean()),
        "predictions": summarise_predictions(y, yh),
        "qualitative": qual,
        "gate1": {"band_h": [lo, hi], "rmse_h_pooled_in_band": rmse_ok,
                  "qualitative_pass": qual["pass"], "pass": bool(rmse_ok and qual["pass"]),
                  "applies": variant_name == g["gate_variant"] and not fast},
        "timing_s": {"load_and_index": load_s, "splits_total": float(np.sum(fit_times)),
                     "per_split_mean": float(np.mean(fit_times)),
                     "per_split_max": float(np.max(fit_times))},
        "leakage_check": {"max_train_end_minus_split_s":
                          float((ps["train_end_max"] - ps["split_t"]).dt.total_seconds().max())},
    }
    ps.to_csv(run.dir / "per_split.csv", index=False)
    table.to_csv(run.dir / "per_bin.csv", index=False)
    pr.to_parquet(run.dir / "predictions.parquet", index=False)
    run.write_json("metrics.json", m)
    title = (f"Eagle, XGBoost defaults — variant {variant_name}, "
             f"{'fast' if fast else 'full'} ({len(ps)} splits, {len(pr):,} test jobs)")
    pred_vs_actual_by_bin(y, yh, table, ecfg["bins_hours"], title,
                          run.dir / "figures" / "pred_vs_actual_by_bin.png")
    run.finish("ok", {"variant": variant_name, "fast": fast, "n_splits": len(ps),
                      "rmse_h_pooled": round(pooled, 4),
                      "rmse_h_mean_of_splits": round(m["rmse_h_mean_of_splits"], 4),
                      "qualitative_pass": qual["pass"]})
    log.info("pooled RMSE %.3f h (mean of splits %.3f h); qualitative %s",
             pooled, m["rmse_h_mean_of_splits"], qual)
    return 0


if __name__ == "__main__":
    sys.exit(main())
