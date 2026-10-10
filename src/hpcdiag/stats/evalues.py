"""Per-node e-processes (PLAN.md §9 Phase 4.1; F13–F16). All in log space.

Bet on one p-value with weight w in [0, 1]:   g = 1 + w (f(p) - 1),  f a calibrator (∫f = 1).
- plain mixture: M = mean over kappa of prod_j g_kappa(p_j)   (average of e-processes);
- jumper (F14, one-sided): states {no bet} ∪ {bet_kappa}; before each update every state receives
  J/S of the total capital and keeps (1 - J) of its own: C_e <- (1-J) C_e + (J/S) sum(C); then
  C_e <- C_e g_e(p) (g = 1 for "no bet"). Total capital is a nonnegative supermartingale under the
  null, so it is an e-process.
"""

from __future__ import annotations

import numpy as np

from hpcdiag.stats.calibrators import KAPPAS, power_calibrator


def log_bet(p: np.ndarray, w: np.ndarray, kappas=KAPPAS) -> np.ndarray:
    """log(1 + w (f_kappa(p) - 1)) for each p (rows) and kappa (columns)."""
    p = np.asarray(p, float)[:, None]
    w = np.broadcast_to(np.asarray(w, float), (p.shape[0],))[:, None]
    if np.any((w < 0) | (w > 1)):
        raise ValueError("weights must be in [0, 1]")
    f = np.column_stack([power_calibrator(p[:, 0], k) for k in kappas])
    return np.log1p(w * (f - 1.0))


def plain_mixture_logM(lb: np.ndarray) -> np.ndarray:
    """Running log M after each step of ONE stream: log mean_kappa exp(cumsum log g_kappa)."""
    c = np.cumsum(lb, axis=0)
    mx = c.max(axis=1, keepdims=True)
    return (mx[:, 0] + np.log(np.mean(np.exp(c - mx), axis=1)))


def jumper_init(n_streams: int, n_states: int) -> np.ndarray:
    """log capital per (stream, state): each state starts with 1/S (total 1)."""
    return np.full((n_streams, n_states), -np.log(n_states))


def jumper_step(logC: np.ndarray, log_g: np.ndarray, J: float) -> np.ndarray:
    """One jumper update for several streams at once. logC: (n, S); log_g: (n, S) with column 0 =
    'no bet' (0). Returns the new log capital."""
    if J == 0:
        return logC + log_g
    S = logC.shape[1]
    mx = logC.max(axis=1, keepdims=True)
    log_total = mx + np.log(np.exp(logC - mx).sum(axis=1, keepdims=True))
    mixed = np.logaddexp(np.log1p(-J) + logC, np.log(J / S) + log_total)
    return mixed + log_g


def log_total(logC: np.ndarray) -> np.ndarray:
    mx = logC.max(axis=1, keepdims=True)
    return (mx + np.log(np.exp(logC - mx).sum(axis=1, keepdims=True)))[:, 0]


def jumper_logM(lb: np.ndarray, J: float) -> np.ndarray:
    """Running log M after each step of ONE stream (reference implementation for tests)."""
    logC = jumper_init(1, lb.shape[1] + 1)
    out = np.empty(lb.shape[0])
    for t in range(lb.shape[0]):
        g = np.concatenate([[0.0], lb[t]])[None, :]
        logC = jumper_step(logC, g, J)
        out[t] = log_total(logC)[0]
    return out


def node_logM_by_rank(node: np.ndarray, lb: np.ndarray, J: float | None) -> np.ndarray:
    """log M of each node after each of its updates, for many nodes at once.

    `node` (n,) and `lb` (n, K) must be in processing order (D14). Updates of the same node are
    applied in order; different nodes are independent, so step r processes the r-th update of every
    node together (vectorised). J=None -> plain mixture; else jumper with switching rate J."""
    node = np.asarray(node)
    n, K = lb.shape
    order = np.argsort(node, kind="stable")              # group by node, keep processing order
    nd = node[order]
    starts = np.r_[0, np.flatnonzero(np.diff(nd)) + 1]
    rank = np.arange(n) - np.repeat(starts, np.diff(np.r_[starts, n]))
    out = np.empty(n)
    if J is None:
        c = np.empty((n, K))
        lbo = lb[order]
        for k in range(K):                                  # cumulative sums within node groups
            cs = np.cumsum(lbo[:, k])
            base = np.repeat(np.r_[0.0, cs[starts[1:] - 1]], np.diff(np.r_[starts, n]))
            c[:, k] = cs - base
        mx = c.max(axis=1, keepdims=True)
        out[order] = mx[:, 0] + np.log(np.mean(np.exp(c - mx), axis=1))
        return out
    grp = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, n]))
    logC = jumper_init(len(starts), K + 1)
    lbo = lb[order]
    for r in range(int(rank.max()) + 1):
        idx = np.flatnonzero(rank == r)
        g = grp[idx]
        lg = np.column_stack([np.zeros(len(idx)), lbo[idx]])
        logC[g] = jumper_step(logC[g], lg, J)
        out[order[idx]] = log_total(logC[g])
    return out
