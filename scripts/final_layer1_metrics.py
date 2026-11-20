"""D123: Layer 1 numbers for reports/FINAL_RESULTS.md that no earlier run computed.

    uv run python scripts/final_layer1_metrics.py --config configs/phase2_conformal.yaml \
        --ref 20261010-003334-pm100-E-main-full --l2-config configs/phase4_layer2.yaml

(1) Prediction accuracy on the post-freeze PM100 jobs of the frozen Mode E run
    (scores_eval.parquet): LightGBM median exp(q50_log), constant median (median run_time_s of
    the 45-day training rows, end_time < freeze) and requested walltime. Share within 2x
    (0.5 <= pred / actual <= 2, actual clipped to >= 1 s), RMSE and MAE in hours, R2 and MAE on
    log(max(., 1)), Spearman; q90 coverage P(log runtime <= q_hat) and pinball loss (tau 0.9,
    log runtime).
(2) Calibration of the Mode P p-values on the unmodified trace (the population Layer 2 builds,
    layer2.prep), for the plain score and for the user/shape-centred score score_c (the input of the
    primary, run_phase5._pop_p): n, KS, share p <= 0.01 / 0.05 / 0.10, P(p <= 1e-3).
Writes runs/phaseR/<id>/metrics.json.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

from hpcdiag.eval.conformal_metrics import ks_distance
from hpcdiag.layer1 import run_conformal as RC
from hpcdiag.layer1.quantile import log_runtime, pinball_loss
from hpcdiag.layer2 import prep as P
from hpcdiag.layer2 import run_layer2 as R
from hpcdiag.utils.run import Run, resolve, run_guarded, start_run


def accuracy(pred_s: np.ndarray, rt_s: np.ndarray) -> dict:
    a = np.maximum(rt_s, 1.0)
    p = np.asarray(pred_s, float)
    ly, lp = np.log(a), np.log(np.maximum(p, 1.0))
    r = p / a
    sst = float(np.sum((ly - ly.mean()) ** 2))
    const = np.all(p == p[0])
    return {
        "within_2x": float(np.mean((r >= 0.5) & (r <= 2.0))),
        "rmse_h": float(np.sqrt(np.mean((p - rt_s) ** 2)) / 3600),
        "mae_h": float(np.mean(np.abs(p - rt_s)) / 3600),
        "r2_log": float(1 - np.sum((ly - lp) ** 2) / sst),
        "mae_log": float(np.mean(np.abs(ly - lp))),
        "spearman": None if const else float(stats.spearmanr(p, rt_s).statistic),
    }


def calib(p: np.ndarray) -> dict:
    p = np.asarray(p, float)
    p = p[np.isfinite(p)]
    share = {f"p_le_{a:g}": float(np.mean(p <= a)) for a in (0.01, 0.05, 0.10, 0.001)}
    return {"n": int(len(p)), "ks": ks_distance(p), **share}


def _run(run: Run, ref: str, l2_config: str) -> int:
    cfg, log = run.config, run.log
    df = pd.read_parquet(resolve(f"runs/phase2/{ref}/scores_eval.parquet"),
                         columns=["run_time_s", "walltime_req_s", "q_hat", "q50_log"])  # fmt: skip
    base, pop = RC._load(cfg, "pm100", "main", log)
    t0 = base["submit_time"].min() + pd.Timedelta(days=RC._train_days(cfg, "pm100"))
    med = float(pop.loc[pop["end_time"] < t0, "run_time_s"].median())
    rt = df["run_time_s"].to_numpy(float)
    wall = df["walltime_req_s"].to_numpy(float)
    y = log_runtime(rt)
    out = {
        "ref": ref,
        "n_jobs": int(len(df)),
        "constant_median_s": med,
        "lightgbm_median": accuracy(np.exp(df["q50_log"].to_numpy()), rt),
        "constant_median": accuracy(np.full(len(rt), med), rt),
        "requested_walltime": accuracy(wall, rt),
        "q90_coverage": float(np.mean(y <= df["q_hat"].to_numpy())),
        "q90_pinball_log": pinball_loss(y, df["q_hat"].to_numpy(), 0.9),
    }
    R._init(l2_config)
    l1 = R.G["cfg"]["layer1"]
    npop = R._null_pop()
    out["calibration_mode_p_W7d"] = {
        "plain_score": calib(P.mode_p_on(npop, "score", l1["window_hours"], l1["min_ref"])),
        "centred_score_c": calib(P.mode_p_on(npop, "score_c", l1["window_hours"], l1["min_ref"])),
    }
    run.write_json("metrics.json", out)
    print(json.dumps(out, indent=1))
    run.finish("ok", {"n_jobs": out["n_jobs"]})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--l2-config", required=True)
    a = ap.parse_args()
    return run_guarded(start_run("phaseR", a.config, tag="final-l1"),
                       lambda r: _run(r, a.ref, a.l2_config))  # fmt: skip


if __name__ == "__main__":
    sys.exit(main())
