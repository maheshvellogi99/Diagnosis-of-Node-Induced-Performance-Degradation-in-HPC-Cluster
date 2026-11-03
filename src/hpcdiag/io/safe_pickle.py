"""Restricted unpickling for downloaded data (pandas fixed-format HDF object blocks).

Only the globals needed to rebuild numpy arrays of plain Python objects are allowed; any other
global raises, so a crafted file cannot run code (Python docs, pickle, "Restricting Globals").
"""

from __future__ import annotations

import io
import pickle

ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "scalar"),
}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) in ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"global not allowed: {module}.{name}")


def loads(data: bytes):
    return SafeUnpickler(io.BytesIO(data)).load()
