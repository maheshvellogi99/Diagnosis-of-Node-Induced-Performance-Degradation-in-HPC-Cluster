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

from hpcdiag.layer3.probes import CAUSES, ProbeModel, state_table


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


def bin_probs(means: np.ndarray, sigma: float, pool: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """P(p in bin | pool) for x ~ N(mean, sigma^2), one row per mean. pool: one pool (n,)."""
    ov = bin_overlap(len(pool), edges)
    return np.vstack([rank_probs(float(mu), sigma, pool[None, :])[0] @ ov for mu in means])


@dataclass(frozen=True)
class Bets:
    """G[a, c, b]: density of p for probe a, component c, bin b. Components are the causes (order of
    CAUSES; D87) or, with per_state, the (cause, s0) states of the posterior without h0 (D96)."""

    G: np.ndarray
    edges: np.ndarray
    beta: float
    per_state: bool = False

    def weights(self, pi: np.ndarray, hyp: np.ndarray) -> np.ndarray:
        """Bet weights from the current posterior over states: pi(component | not h0)."""
        if self.per_state:
            w = pi[hyp > 0]
        else:
            w = np.array([pi[hyp == i].sum() for i in range(1, len(CAUSES) + 1)])
        return w / w.sum()

    def log_f_bins(self, a: int, w: np.ndarray) -> np.ndarray:
        """log f on each bin for probe a under weights w."""
        return np.log((1 - self.beta) + self.beta * self.mixture(a, w))

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


def build_state_bets(m: ProbeModel, ec: dict, s_grid, rng: np.random.Generator) -> Bets:
    """D96: one density per probe and (cause, s0) state, in the order of state_table without h0,
    each over the same independent pools and integrating to exactly 1. Zero shift: density 1."""
    edges = np.asarray(ec["edges"], float)
    n, n_pools = ec["n_pool"], ec["bet_pools"]
    pools = rng.normal(0, m.sigma, (n_pools, n))
    means, hyp = state_table(m, s_grid)
    means = means[hyp > 0]
    G = np.ones((len(m.names), len(means), len(edges) - 1))
    s = -np.sort(-pools, axis=1)
    ov = bin_overlap(n, edges)
    z = np.zeros((n_pools, 1))
    cache: dict[float, np.ndarray] = {}
    for i in range(len(m.names)):
        for j, mu in enumerate(means[:, i]):
            if mu == 0:
                continue
            if mu not in cache:
                sf = norm.sf((s - mu) / m.sigma)
                pr = np.diff(np.concatenate([z, sf, z + 1.0], axis=1), axis=1)
                mass = pr.mean(axis=0) @ ov
                cache[mu] = mass / mass.sum() / np.diff(edges)
            G[i, j] = cache[mu]
    return Bets(G, edges, float(ec["beta"]), per_state=True)


def conditional_mean(
    h_of_f: np.ndarray, mu: float, sigma: float, pools: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    """E[h(f(p)) | pool] for x ~ N(mu, sigma^2). h_of_f: value per bin. One value per pool."""
    return rank_probs(mu, sigma, pools) @ (bin_overlap(pools.shape[1], edges) @ h_of_f)
