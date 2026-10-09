"""Menear et al. baseline: xgboost.XGBRegressor() with defaults, target run_time seconds (F6)."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost

from hpcdiag.features.encoding import encode_frame
from hpcdiag.layer1.splits import TimeIndex


def make_model(params: dict | None = None) -> xgboost.XGBRegressor:
    """Defaults only (F6). `params` must be empty for the main protocol; it exists so a
    documented sensitivity check (e.g. tree_method='approx', PLAN Phase 1) can be configured."""
    return xgboost.XGBRegressor(**(params or {}))


def booster_config(model: xgboost.XGBRegressor) -> dict:
    """Full configuration actually used (PLAN Phase 1: save_config of the booster)."""
    return json.loads(model.get_booster().save_config())


@dataclass
class SplitResult:
    t: pd.Timestamp
    train_rows: int
    test_rows: int
    train_end_max: pd.Timestamp | None
    n_classes: dict
    unseen_test: dict
    predictions: pd.DataFrame          # row_id, y, yhat, nodes_req
    config: dict | None


def run_split(df: pd.DataFrame, index: TimeIndex, t: pd.Timestamp, train_days: float,
              test_days: float, numeric: list[str], categorical: list[str], target: str,
              unseen: int = -1, params: dict | None = None) -> SplitResult | None:
    """Fit on train(t), predict test(t). Returns None if either side is empty."""
    tr_pos = index.train_positions(t, train_days)
    te_pos = index.test_positions(t, test_days)
    if len(tr_pos) == 0 or len(te_pos) == 0:
        return None
    train, test = df.iloc[tr_pos], df.iloc[te_pos]
    # leakage guard (CLAUDE.md rule 8): every training job ended strictly before t
    end_max = train["end_time"].max()
    if not end_max < t:
        raise AssertionError(f"training job ends at/after split time {t}: {end_max}")
    xtr, xte, n_classes = encode_frame(train, test, numeric, categorical, unseen)
    model = make_model(params)
    model.fit(xtr, train[target].to_numpy(dtype="float64"))
    yhat = model.predict(xte).astype("float64")
    preds = pd.DataFrame({
        "row_id": test["row_id"].to_numpy(),
        "y": test[target].to_numpy(dtype="float64"),
        "yhat": yhat,
        "nodes_req": test["nodes_req"].to_numpy(dtype="float64"),
    })
    unseen_test = {c: int((xte[c] == unseen).sum()) for c in categorical}
    return SplitResult(t, len(tr_pos), len(te_pos), end_max, n_classes, unseen_test, preds,
                       booster_config(model))


def config_fingerprint(cfg: dict) -> str:
    return json.dumps(cfg, sort_keys=True)


def summarise_predictions(y: np.ndarray, yhat: np.ndarray) -> dict:
    return {"n_pred_lt_1s": int((yhat < 1).sum()), "n_pred_lt_0": int((yhat < 0).sum()),
            "n": int(len(y))}
