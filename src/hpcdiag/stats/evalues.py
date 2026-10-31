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
    return mx[:, 0] + np.log(np.mean(np.exp(c - mx), axis=1))


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
    order = np.argsort(node, kind="stable")  # group by node, keep processing order
    nd = node[order]
    starts = np.r_[0, np.flatnonzero(np.diff(nd)) + 1]
    rank = np.arange(n) - np.repeat(starts, np.diff(np.r_[starts, n]))
    out = np.empty(n)
    if J is None:
        c = np.empty((n, K))
        lbo = lb[order]
        for k in range(K):  # cumulative sums within node groups
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


def log_bet_funcs(p: np.ndarray, w: np.ndarray, funcs) -> np.ndarray:
    """log(1 + w (g(p) - 1)) for arbitrary bet densities g (each with integral 1) as columns."""
    p = np.asarray(p, float)
    w = np.broadcast_to(np.asarray(w, float), p.shape)
    if np.any((w < 0) | (w > 1)):
        raise ValueError("weights must be in [0, 1]")
    with np.errstate(divide="ignore"):
        return np.column_stack([np.log1p(w * (g(p) - 1.0)) for g in funcs])


def _lambda_from_log(lm: np.ndarray) -> np.ndarray:
    """lambda = M / sum(M) for one job, from log M (D71)."""
    r = np.exp(lm - lm.max())
    return r / r.sum()


def blame_reference(
    job: np.ndarray, block: np.ndarray, node: np.ndarray, g: np.ndarray, J: float, m: int
) -> tuple[np.ndarray, np.ndarray]:
    """Jumper with blame allocation (D71), written out step by step (reference for tests).

    Pairs (job, node) in processing order (D14); `block` = tie-block id (same end time). `g` (n, K)
    = bet densities at the job's p-value, one column per betting state; state 0 'no bet' is added
    here. All jobs of a block take lambda_{j,n} = M_n / sum_{k in N_j} M_k from the state BEFORE
    the block; the updates are then applied in order. Returns (log M after each pair, lambda)."""
    logC = jumper_init(m, g.shape[1] + 1)
    out, lam = np.empty(len(node)), np.empty(len(node))
    starts = np.r_[0, np.flatnonzero(np.diff(block)) + 1, len(block)]
    for a, b in zip(starts[:-1], starts[1:], strict=True):
        before = log_total(logC)
        for j in np.unique(job[a:b]):
            i = a + np.flatnonzero(job[a:b] == j)
            lam[i] = _lambda_from_log(before[node[i]])
        for i in range(a, b):
            with np.errstate(divide="ignore"):
                lb = np.log1p(lam[i] * (g[i] - 1.0))
            n = node[i : i + 1]
            logC[n] = jumper_step(logC[n], np.r_[0.0, lb][None, :], J)
            out[i] = log_total(logC[n])[0]
    return out, lam


def _block_levels(block: np.ndarray, node: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Schedule tie blocks: level(b) = 1 + the latest level of any of its nodes, so blocks on one
    level share no node and can be updated together. Returns (level per pair, level has a node
    twice inside one block)."""
    starts = np.r_[0, np.flatnonzero(np.diff(block)) + 1, len(block)]
    nodes = node.tolist()
    last = [0] * m
    lev, repeat = [], {}
    for a, b in zip(starts[:-1].tolist(), starts[1:].tolist(), strict=True):
        seg = nodes[a:b]
        lv = 1 + max([last[x] for x in seg])
        for x in seg:
            last[x] = lv
        lev.append(lv)
        if len(set(seg)) < len(seg):
            repeat[lv] = True
    rep = np.zeros(max(lev) + 1, bool)
    rep[list(repeat)] = True
    return np.repeat(np.array(lev), np.diff(starts)), rep


def blame_logM(
    job: np.ndarray, block: np.ndarray, node: np.ndarray, g: np.ndarray, J: float, m: int
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised blame_reference (same arguments, same result; D71)."""
    n = len(node)
    out, lam = np.empty(n), np.empty(n)
    if n == 0:
        return out, lam
    level, rep = _block_levels(block, node, m)
    order = np.argsort(level, kind="stable")
    cuts = np.r_[0, np.flatnonzero(np.diff(level[order])) + 1, n]
    logC = jumper_init(m, g.shape[1] + 1)
    total = np.zeros(m)
    for a, b in zip(cuts[:-1].tolist(), cuts[1:].tolist(), strict=True):
        idx = order[a:b]
        nd, jb = node[idx], job[idx]
        st = np.r_[0, np.flatnonzero(jb[1:] != jb[:-1]) + 1]
        if len(st) == len(idx):
            la = np.ones(len(idx))
        else:
            sizes = np.diff(np.r_[st, len(idx)])
            lm = total[nd]
            r = np.exp(lm - np.repeat(np.maximum.reduceat(lm, st), sizes))
            la = r / np.repeat(np.add.reduceat(r, st), sizes)
        lam[idx] = la
        with np.errstate(divide="ignore"):
            lg = np.log1p(la[:, None] * (g[idx] - 1.0))
        lg = np.column_stack([np.zeros(len(idx)), lg])
        if not rep[level[idx[0]]]:
            logC[nd] = jumper_step(logC[nd], lg, J)
            total[nd] = log_total(logC[nd])
            out[idx] = total[nd]
            continue
        o = np.argsort(nd, kind="stable")  # a node twice in one block: apply its updates in order
        first = np.r_[True, nd[o][1:] != nd[o][:-1]]
        grp = np.cumsum(first) - 1
        rank = np.empty(len(o), int)
        rank[o] = np.arange(len(o)) - np.flatnonzero(first)[grp]
        for r_i in range(int(rank.max()) + 1):
            s = np.flatnonzero(rank == r_i)
            logC[nd[s]] = jumper_step(logC[nd[s]], lg[s], J)
            total[nd[s]] = log_total(logC[nd[s]])
            out[idx[s]] = total[nd[s]]
    return out, lam
