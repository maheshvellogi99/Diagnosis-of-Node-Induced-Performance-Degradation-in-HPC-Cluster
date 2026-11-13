"""Label encoding fitted on the training window only (PLAN.md §9 Phase 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hpcdiag.features.encoding import LabelEncoder, encode_frame


def test_codes_sorted_and_unseen_minus_one():
    enc = LabelEncoder().fit(pd.Series(["b", "a", "c", "a"]))
    assert enc.classes_.tolist() == ["a", "b", "c"]
    got = enc.transform(pd.Series(["c", "a", "zzz", "b", "0"]))
    assert got.tolist() == [2, 0, -1, 1, -1]
    assert got.dtype == np.int64


def test_test_rows_do_not_change_codes():
    train = pd.DataFrame({"user": ["u2", "u1"], "partition": ["p", "p"], "x": [1.0, 2.0]})
    test1 = pd.DataFrame({"user": ["u1"], "partition": ["p"], "x": [3.0]})
    test2 = pd.DataFrame({"user": ["u0", "u1", "u9"], "partition": ["q", "p", "p"],
                          "x": [3.0, 4.0, 5.0]})
    _, a, _ = encode_frame(train, test1, ["x"], ["user", "partition"])
    xtr, b, n = encode_frame(train, test2, ["x"], ["user", "partition"])
    assert a["user"].tolist() == [0]
    assert b["user"].tolist() == [-1, 0, -1]          # u0 sorts before u1 but is unseen
    assert b["partition"].tolist() == [-1, 0, 0]
    assert xtr["user"].tolist() == [1, 0]
    assert n == {"user": 2, "partition": 1}
    assert list(xtr.columns) == ["x", "user", "partition"]


def test_custom_unseen_and_empty_train():
    enc = LabelEncoder(unseen=-7).fit(pd.Series(["a"]))
    assert enc.transform(pd.Series(["b"])).tolist() == [-7]
    empty = LabelEncoder().fit(pd.Series([], dtype="str"))
    assert empty.transform(pd.Series(["a"])).tolist() == [-1]


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        LabelEncoder().transform(pd.Series(["a"]))
