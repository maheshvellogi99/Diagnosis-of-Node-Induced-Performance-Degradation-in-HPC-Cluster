"""Simulated probe model (docs/PHASE5_DESIGN.md §2) [A]: larger x = worse.

Healthy: log x ~ N(0, sigma^2). Cause h with slowdown s: log x ~ N(k_{h,a} log s, sigma^2); cause
mem on the mem probe is a fixed shift. Hypotheses: h0 plus the causes; states of the diagnosis
posterior are (cause, s) pairs plus one h0 state (D89).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

HYPOTHESES = ("h0", "cpu", "cache", "mbw", "mem")
CAUSES = HYPOTHESES[1:]


def sig(sigma, a: int):
    """sigma of probe a: the scalar itself (unchanged object), or entry a of a per-probe sigma
    (D113)."""
    return sigma if np.ndim(sigma) == 0 else float(sigma[a])


@dataclass(frozen=True)
class ProbeModel:
    names: tuple[str, ...]
    cost: tuple[float, ...]
    sigma: float | tuple[float, ...]  # one sigma, or one per probe in the order of names (D113)
    k: dict  # k[cause][probe]
    mem_shift: float

    def delta(self, cause: str, probe: str, s: float) -> float:
        """Mean of log x under cause on a node with slowdown s; 0 for h0."""
        if cause == "h0":
            return 0.0
        if cause == "mem" and probe == "mem":
            return float(self.mem_shift)
        return float(self.k[cause][probe] * np.log(s))


def from_config(pc: dict) -> ProbeModel:
    s = pc["sigma"]
    if isinstance(s, dict):  # D113: one sigma per probe
        missing = [a for a in pc["names"] if a not in s]
        if missing or set(s) - set(pc["names"]):
            raise ValueError(f"probes.sigma per probe must name exactly {pc['names']}")
        sigma = tuple(float(s[a]) for a in pc["names"])
    else:
        sigma = float(s)
    return ProbeModel(
        names=tuple(pc["names"]),
        cost=tuple(float(pc["cost_s"][a]) for a in pc["names"]),
        sigma=sigma,
        k={c: dict(v) for c, v in pc["k"].items()},
        mem_shift=float(pc["mem_fixed_shift"]),
    )


def with_primary(m: ProbeModel, value: float) -> ProbeModel:
    """Sensitivity: diagonal entries k_{a,a} = value (the mem shift is not scaled by k)."""
    k = {c: dict(v) for c, v in m.k.items()}
    for c in ("cpu", "cache", "mbw"):
        k[c][c] = float(value)
    return replace(m, k=k)


def perturbed(m: ProbeModel, rng: np.random.Generator, spread: float = 0.5) -> ProbeModel:
    """Misspecified outcomes: every k entry times 1 + U(-spread, spread) (design §6)."""
    k = {
        c: {a: float(v * (1 + rng.uniform(-spread, spread))) for a, v in row.items()}
        for c, row in m.k.items()
    }
    return replace(m, k=k)


def state_table(m: ProbeModel, s_grid) -> tuple[np.ndarray, np.ndarray]:
    """(means, hyp): means (n_states, n_probes) of log x; hyp[i] = index into HYPOTHESES.
    State 0 is h0; then every cause x s."""
    rows, hyp = [[0.0] * len(m.names)], [0]
    for i, c in enumerate(CAUSES, start=1):
        for s in s_grid:
            rows.append([m.delta(c, a, s) for a in m.names])
            hyp.append(i)
    return np.array(rows), np.array(hyp)


def design_confusion(correct: float, confused: float, healthy: float) -> np.ndarray:
    """Design §2 telemetry confusion [A]: rows and columns follow HYPOTHESES. The remainder of each
    row, which the design leaves open, is spread evenly over the other classes [D]."""
    n = len(HYPOTHESES)
    out = np.zeros((n, n))
    trio = [HYPOTHESES.index(c) for c in ("cpu", "cache", "mbw")]
    out[0] = (1 - healthy) / (n - 1)
    out[0, 0] = healthy
    for i in trio:
        others = [j for j in range(n) if j != i and j not in trio]
        out[i, others] = (1 - correct - 2 * confused) / len(others)
        out[i, [j for j in trio if j != i]] = confused
        out[i, i] = correct
    m = HYPOTHESES.index("mem")
    out[m] = (1 - correct) / (n - 1)
    out[m, m] = correct
    return out
