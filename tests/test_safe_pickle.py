"""Restricted unpickler (D99): numpy object arrays load, anything else is refused."""

from __future__ import annotations

import os
import pickle

import numpy as np
import pytest

from hpcdiag.io.safe_pickle import loads


def test_object_array_round_trip():
    a = np.array(["15715588", "601382", None], dtype=object)
    assert list(loads(pickle.dumps(a, protocol=4))) == list(a)


class Evil:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


def test_other_globals_refused():
    with pytest.raises(pickle.UnpicklingError):
        loads(pickle.dumps(Evil()))
