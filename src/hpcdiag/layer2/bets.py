"""Shift-likelihood-ratio bets (D61).

g_s0(p) = 20-bin histogram density of the Mode P p-values obtained by adding log(s0) to the scores
of jobs that ended in the warm-up week [freeze, freeze + 7 d]; each p is computed against the
job's UNSHIFTED reference set. Piecewise constant, >= 0, integrates to exactly 1 on [0, 1].
Built once, before any evaluation time, so the bet is predictable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class HistBet:
    density: np.ndarray  # (bins,), mean = 1
    label: str
    n: int

    def __call__(self, p: np.ndarray) -> np.ndarray:
        b = len(self.density)
        idx = np.minimum((np.asarray(p, float) * b).astype(int), b - 1)
        return self.density[idx]


def hist_bet(p: np.ndarray, bins: int, label: str) -> HistBet:
    p = np.asarray(p, float)
    p = p[np.isfinite(p)]
    counts = np.bincount(np.minimum((p * bins).astype(int), bins - 1), minlength=bins)
    dens = counts / counts.sum() * bins  # sum(dens) / bins = 1 exactly
    return HistBet(dens.astype(float), label, int(len(p)))


def warmup_mask(pop: pd.DataFrame, freeze_ns: int, days: float = 7) -> np.ndarray:
    e = pop["end_ns"].to_numpy()
    return (e >= freeze_ns) & (e <= freeze_ns + int(days * 24 * 3600 * 10**9))


def build_shift_bets(
    pop: pd.DataFrame,
    col: str,
    freeze_ns: int,
    s0_list,
    window_h: float,
    min_ref: int,
    bins: int = 20,
    detrend_hours: float | None = None,
    mode_p_on=None,
) -> list[HistBet]:
    """One HistBet per s0, from warm-up-week jobs only (pop restricted to end <= freeze + 7 d)."""
    wu = pop.loc[warmup_mask(pop, freeze_ns)].reset_index(drop=True)
    out = []
    for s0 in s0_list:
        p = mode_p_on(wu, col, window_h, min_ref, detrend_hours, query_shift=float(np.log(s0)))
        out.append(hist_bet(p, bins, f"g_{s0}"))
    return out
