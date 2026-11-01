"""Probe e-values (docs/PHASE5_DESIGN.md §3, D87).

p = (#{y in pool: y > x} + theta (1 + #{y in pool: y = x})) / (n_pool + 1), theta ~ U(0, 1).
f(p) = (1 - beta) + beta g(p), with g a histogram density on fixed edges. Each per-cause density is
averaged over independent pools, so g does not depend on the pool that is scored, and the bet for a
probe is the mixture of per-cause densities under the current cause posterior.

For a normal x and a given pool the rank r = #{y > x} has an exact distribution and p | r is
uniform on [r, r + 1) / (n + 1), so densities and conditional means are computed exactly, without
draws of x.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from hpcdiag.layer3.probes import CAUSES, ProbeModel


def conformal_p(x: np.ndarray, pool: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Design §3 p-value, ties included. pool must be sorted ascending."""
    x = np.asarray(x, float)
    above = len(pool) - np.searchsorted(pool, x, side="right")
    ties = np.searchsorted(pool, x, side="right") - np.searchsorted(pool, x, side="left")
    return (above + theta * (1 + ties)) / (len(pool) + 1)


def rank_probs(mu: float, sigma: float, pools: np.ndarray) -> np.ndarray:
    """P(r = 0..n | pool), x ~ N(mu, sigma^2). pools: (n_pools, n). Returns (n_pools, n + 1)."""
    s = -np.sort(-pools, axis=1)  # descending: r = j means x lies between s[j-1] and s[j]
    sf = norm.sf((s - mu) / sigma)  # P(x > s_j), increasing in j
    z = np.zeros((len(s), 1))
    return np.diff(np.concatenate([z, sf, z + 1.0], axis=1), axis=1)


def bin_overlap(n: int, edges: np.ndarray) -> np.ndarray:
    """(n + 1, n_bins): share of the uniform law on [r, r + 1) / (n + 1) that falls in each bin."""
    lo = np.arange(n + 1)[:, None] / (n + 1)
    hi = (np.arange(n + 1)[:, None] + 1) / (n + 1)
    a = np.maximum(lo, edges[:-1][None, :])
    b = np.minimum(hi, edges[1:][None, :])
    return np.maximum(b - a, 0.0) * (n + 1)


def density(mu: float, sigma: float, pools: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Histogram density of p on edges, averaged over pools; integrates to exactly 1."""
    mass = rank_probs(mu, sigma, pools).mean(axis=0) @ bin_overlap(pools.shape[1], edges)
    mass = mass / mass.sum()
    return mass / np.diff(edges)


@dataclass(frozen=True)
class Bets:
    """G[a, c, b]: density of p for probe a, cause c (order of CAUSES), bin b."""

    G: np.ndarray
    edges: np.ndarray
    beta: float

    def mixture(self, a: int, w: np.ndarray) -> np.ndarray:
        """g_a = sum_c w_c g_{a,c}; w sums to 1."""
        return w @ self.G[a]

    def f(self, p: np.ndarray, g: np.ndarray) -> np.ndarray:
        """Bet value at p for density g."""
        b = np.clip(np.searchsorted(self.edges, p, side="right") - 1, 0, len(g) - 1)
        return (1 - self.beta) + self.beta * g[b]


def build_bets(m: ProbeModel, ec: dict, rng: np.random.Generator) -> Bets:
    """D87. A probe with mean 0 under a cause has uniform p, so its density is 1 exactly."""
    edges = np.asarray(ec["edges"], float)
    n, n_pools = ec["n_pool"], ec["bet_pools"]
    pools = rng.normal(0, m.sigma, (n_pools, n))
    G = np.ones((len(m.names), len(CAUSES), len(edges) - 1))
    for i, a in enumerate(m.names):
        for j, c in enumerate(CAUSES):
            mu = m.delta(c, a, ec["bet_s0"])
            if mu != 0:
                G[i, j] = density(mu, m.sigma, pools, edges)
    return Bets(G, edges, float(ec["beta"]))


def conditional_mean(
    h_of_f: np.ndarray, mu: float, sigma: float, pools: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    """E[h(f(p)) | pool] for x ~ N(mu, sigma^2). h_of_f: value per bin. One value per pool."""
    return rank_probs(mu, sigma, pools) @ (bin_overlap(pools.shape[1], edges) @ h_of_f)
