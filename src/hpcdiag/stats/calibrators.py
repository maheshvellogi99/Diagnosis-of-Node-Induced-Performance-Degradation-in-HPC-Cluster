"""p-to-e calibrators (F15) and the Layer-1 tail check that Layer 2 needs (D24)."""

from __future__ import annotations

import numpy as np

KAPPAS = (0.05, 0.1, 0.25, 0.5)  # PLAN §9 Phase 4.1 mixture


def power_calibrator(p: np.ndarray, kappa: float) -> np.ndarray:
    """f_kappa(p) = kappa * p^(kappa - 1); integrates to 1 on [0, 1] for kappa in (0, 1)."""
    if not 0 < kappa < 1:
        raise ValueError("kappa must be in (0, 1)")
    return kappa * np.power(np.asarray(p, dtype="float64"), kappa - 1.0)


def mixture_calibrator(p: np.ndarray, kappas=KAPPAS) -> np.ndarray:
    """Average of power calibrators over kappas (still integrates to 1)."""
    p = np.asarray(p, dtype="float64")
    return np.mean([power_calibrator(p, k) for k in kappas], axis=0)


def null_mean_f(
    n: int, n_sims: int, seed: int, kappas=KAPPAS, chunk: int = 1_000_000
) -> np.ndarray:
    """Sample means of the mixture calibrator over n IID U(0,1) p-values, n_sims times."""
    rng = np.random.default_rng(seed)
    out = np.empty(n_sims)
    for b in range(n_sims):
        tot, left = 0.0, n
        while left:
            m = min(chunk, left)
            tot += float(mixture_calibrator(rng.uniform(size=m), kappas).sum())
            left -= m
        out[b] = tot / n
    return out


def tail_checks(
    p: np.ndarray, n_sims: int, seed: int, thresholds=(1e-3, 1e-4), kappas=KAPPAS
) -> dict:
    """D24: P(p <= t) for small t, and mean mixture f(p) against its simulated null."""
    p = np.asarray(p, dtype="float64")
    p = p[np.isfinite(p)]
    null = null_mean_f(len(p), n_sims, seed, kappas)
    mf = float(mixture_calibrator(p, kappas).mean())
    return {
        "n": int(len(p)),
        "p_le": {f"{t:g}": float(np.mean(p <= t)) for t in thresholds},
        "mean_f": mf,
        "null_mean_f": {
            "p50": float(np.percentile(null, 50)),
            "p95": float(np.percentile(null, 95)),
            "p99": float(np.percentile(null, 99)),
            "n_sims": n_sims,
            "seed": seed,
        },
        "mean_f_le_null_p95": bool(mf <= np.percentile(null, 95)),
        "kappas": list(kappas),
    }
