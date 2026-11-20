"""D121(a): re-fit the frozen Layer 1 LightGBM models and export them.

    uv run python scripts/export_layer1.py --config configs/phase2_conformal.yaml \
        --ref 20261010-003334-pm100-E-main-full --out deliverables/models/layer1

Repeats Mode E of src/hpcdiag/layer1/run_conformal.py (PM100, population main, 45-day freeze
window, seed 0, taus 0.9 and 0.5). The predictions on the post-freeze rows must equal q_hat and
q50_log of the reference run to 1e-9, before and after a save/load round trip of the text models;
otherwise the script exits 1 and writes nothing to --out. Writes runs/phaseR/<id>/metrics.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from hpcdiag.features.encoding import LabelEncoder
from hpcdiag.layer1 import run_conformal as RC
from hpcdiag.layer1.quantile import fit_quantile, log_runtime
from hpcdiag.utils.checksums import verify
from hpcdiag.utils.run import Run, resolve, run_guarded, start_run

TOL = 1e-9


def _run(run: Run, ref: str, out: Path) -> int:
    cfg, log = run.config, run.log
    problems = verify(resolve(cfg["raw_dir"]), resolve(cfg["checksums_file"]))
    if problems:
        raise RuntimeError(f"raw checksum problems: {problems}")
    base, pop = RC._load(cfg, "pm100", "main", log)
    L = RC._train_days(cfg, "pm100")
    t0 = base["submit_time"].min() + pd.Timedelta(days=L)
    tr = pop[pop["end_time"] < t0]
    post = pop[pop["end_time"] >= t0].sort_values(["end_time", "row_id"], kind="stable")
    post = post.reset_index(drop=True)
    xtr, (xpost,) = RC._features(cfg, tr, post)
    ytr = log_runtime(tr["run_time_s"])
    qp, seed, taus = cfg["quantile"]["lgbm_params"], cfg["seed"], cfg["quantile"]["taus"]
    m90 = fit_quantile(xtr, ytr, taus["upper"], qp, seed)
    m50 = fit_quantile(xtr, ytr, taus["median"], qp, seed)
    refdf = pd.read_parquet(resolve(f"runs/phase2/{ref}/scores_eval.parquet"),
                            columns=["row_id", "q_hat", "q50_log"])  # fmt: skip
    refdf = refdf.set_index("row_id").loc[post["row_id"].to_numpy()]
    d90 = float(np.max(np.abs(m90.predict(xpost) - refdf["q_hat"].to_numpy())))
    d50 = float(np.max(np.abs(m50.predict(xpost) - refdf["q50_log"].to_numpy())))
    tmp = run.dir
    m90.booster_.save_model(str(tmp / "layer1_q90.txt"))
    m50.booster_.save_model(str(tmp / "layer1_q50.txt"))
    r90 = float(np.max(np.abs(lgb.Booster(model_file=str(tmp / "layer1_q90.txt")).predict(xpost)
                              - refdf["q_hat"].to_numpy())))  # fmt: skip
    r50 = float(np.max(np.abs(lgb.Booster(model_file=str(tmp / "layer1_q50.txt")).predict(xpost)
                              - refdf["q50_log"].to_numpy())))  # fmt: skip
    f = cfg["features"]
    enc = {
        "numeric": f["numeric"],
        "categorical": f["categorical"],
        "column_order": f["numeric"] + f["categorical"],
        "unseen_code": f["unseen_code"],
        "classes": {
            c: LabelEncoder(f["unseen_code"]).fit(tr[c]).classes_.tolist() for c in f["categorical"]
        },
        "target": "log(max(run_time_s, 1))",
        "train_window": {"freeze_time": str(t0), "train_days": L, "train_rows": int(len(tr))},
    }
    m = {"ref": ref, "rows": int(len(post)), "max_abs_diff_q90": d90, "max_abs_diff_q50": d50,
         "max_abs_diff_q90_reloaded": r90, "max_abs_diff_q50_reloaded": r50, "tol": TOL,
         "freeze_time": str(t0), "train_rows": int(len(tr))}  # fmt: skip
    ok = max(d90, d50, r90, r50) <= TOL
    run.write_json("metrics.json", {**m, "ok": ok})
    print(json.dumps(m))
    if not ok:
        print("STOP: re-fit does not reproduce the frozen predictions")
        run.finish("mismatch", m)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    for name in ("layer1_q90.txt", "layer1_q50.txt"):
        (out / name).write_bytes((tmp / name).read_bytes())
    (out / "layer1_feature_encoding.json").write_text(json.dumps(enc, indent=1))
    (out / "layer1_export_check.json").write_text(json.dumps({**m, "run": run.dir.name}, indent=1))
    run.finish("ok", m)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    return run_guarded(start_run("phaseR", a.config, tag="layer1-export"),
                       lambda r: _run(r, a.ref, resolve(a.out)))  # fmt: skip


if __name__ == "__main__":
    sys.exit(main())
