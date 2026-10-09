"""Diagnostic (after Phase 1): does tree_method change XGBoost predictions on one Eagle split?

    uv run python scripts/diag_xgb_tree_method.py

Fits XGBRegressor with several tree methods on split index 90 of the Phase 1 main protocol,
outside the runner, and compares predictions and dumped trees against tree_method="hist".
"""

from __future__ import annotations

import json
import time

import numpy as np
import xgboost

from hpcdiag.features.encoding import encode_frame
from hpcdiag.layer1 import data as D
from hpcdiag.layer1.splits import TimeIndex, split_times
from hpcdiag.utils.run import load_config, resolve

SPLIT_INDEX = 90
VARIANTS = {
    "default": {},
    "hist": {"tree_method": "hist"},
    "approx": {"tree_method": "approx"},
    "exact": {"tree_method": "exact"},
    "hist_maxbin64": {"tree_method": "hist", "max_bin": 64},
    "approx_maxbin64": {"tree_method": "approx", "max_bin": 64},
}


def main() -> None:
    cfg = load_config("configs/phase1_menear.yaml")
    f = cfg["features"]
    df = D.load_jobs(resolve(cfg["data"]["jobs"]), f["numeric"] + f["categorical"],
                     cfg["filters"]["states"], True)
    idx = TimeIndex.build(df["end_time"], df["submit_time"])
    t = split_times(df["submit_time"], 100, 6, 1)[SPLIT_INDEX]
    tr, te = df.iloc[idx.train_positions(t, 100)], df.iloc[idx.test_positions(t, 1)]
    xtr, xte, _ = encode_frame(tr, te, f["numeric"], f["categorical"])
    y = tr["run_time_s"].to_numpy()
    print("xgboost", xgboost.__version__, "split", t, "train", len(tr), "test", len(te))

    res = {}
    for name, kw in VARIANTS.items():
        start = time.time()
        m = xgboost.XGBRegressor(**kw).fit(xtr, y)
        cfg_saved = json.loads(m.get_booster().save_config())
        tm = cfg_saved["learner"]["gradient_booster"]["gbtree_train_param"]["tree_method"]
        res[name] = (m.predict(xte), tm, time.time() - start, m.get_booster().get_dump())
    ref_pred, ref_trees = res["hist"][0], res["hist"][3]
    for name, (p, tm, sec, trees) in res.items():
        print(f"{name:16s} tree_method={tm:7s} {sec:5.1f}s  "
              f"max|diff vs hist|={np.abs(p - ref_pred).max():.6g}  "
              f"identical_trees_vs_hist={trees == ref_trees}")


if __name__ == "__main__":
    main()
