"""Label encoding fitted on the training window only (PLAN.md §9 Phase 1; F6)."""

from __future__ import annotations

import numpy as np
import pandas as pd


class LabelEncoder:
    """Maps each training category to 0..K-1 (sorted order); anything unseen -> `unseen`."""

    def __init__(self, unseen: int = -1) -> None:
        self.unseen = unseen
        self.classes_: np.ndarray | None = None

    def fit(self, values: pd.Series) -> LabelEncoder:
        v = values.astype("str")
        if v.isna().any():
            raise ValueError("null categories in training data")
        self.classes_ = np.unique(v.to_numpy(dtype=object).astype(str))
        return self

    def transform(self, values: pd.Series) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("fit first")
        v = values.astype("str").to_numpy(dtype=object).astype(str)
        if len(self.classes_) == 0:
            return np.full(len(v), self.unseen, dtype="int64")
        pos = np.searchsorted(self.classes_, v)
        pos_c = np.minimum(pos, len(self.classes_) - 1)
        hit = self.classes_[pos_c] == v
        return np.where(hit, pos_c, self.unseen).astype("int64")


def encode_frame(
    train: pd.DataFrame,
    test: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    unseen: int = -1,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Feature matrices for one split. Encoders see only `train`. Returns (X_train, X_test,
    number of training classes per categorical column)."""
    xtr = train[numeric].astype("float64").copy()
    xte = test[numeric].astype("float64").copy()
    n_classes = {}
    for c in categorical:
        enc = LabelEncoder(unseen).fit(train[c])
        xtr[c] = enc.transform(train[c])
        xte[c] = enc.transform(test[c])
        n_classes[c] = len(enc.classes_)
    cols = numeric + categorical
    return xtr[cols], xte[cols], n_classes
