"""Probing policy on a fixed Layer-2 trajectory (docs/PHASE5_DESIGN.md §3-§5, D86-D91).

Multiplying every jumper state by f leaves the state proportions unchanged and the jumper update is
linear homogeneous, so log E_n(t) = log M_n(t) + sum of log f over probes finished by t. Layer 2 is
computed once; a policy only adds a running log-probe vector. Probes launched after evaluation t
finish inside the next block, so they count at t + 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hpcdiag.layer3 import bayes as BY
from hpcdiag.layer3.evalue import Bets, conformal_p
from hpcdiag.layer3.probes import CAUSES, HYPOTHESES
from hpcdiag.stats.ebh import ebh_log

BLOCKS_PER_DAY = 4


@dataclass(frozen=True)
class Policy:
    name: str
    B: float  # node-seconds per day
    nodes: str = "rank"  # rank | random | sweep
    probes: str = "ig_cost"  # ig_cost | ig_unit | random | battery
    exclusion: str = "cleared"  # cleared (D86) | all (design) | cleared_skipdone | none


@dataclass
class Belief:
    """What the policy assumes: state means, sigma, telemetry confusion, bets, prior."""

    means: np.ndarray  # (n_states, n_probes)
    hyp: np.ndarray
    sigma: float
    conf: np.ndarray
    bets: Bets
    pi0: np.ndarray
    tau: float
    cost: np.ndarray  # micro-benchmarks, then telemetry
    z: np.ndarray
    w: np.ndarray
    cache: dict = field(default_factory=dict)


@dataclass
class World:
    """What really happens. cause[n] = index into HYPOTHESES (0 = never degraded)."""

    pools: list  # sorted pool per probe
    means: np.ndarray  # true mean (len(HYPOTHESES), n_probes) at this scenario's s
    sigma: float
    conf: np.ndarray
    cause: np.ndarray
    onset_idx: np.ndarray  # first evaluation index at or after onset; large = never


def gain_order(pi: np.ndarray, unused: list[int], bel: Belief, by_cost: bool) -> int:
    """argmax over unused probes of expected information gain, per second or per probe."""
    key = (by_cost, tuple(unused), np.round(pi, 7).tobytes())
    if key in bel.cache:
        return bel.cache[key]
    h_now = float(BY.entropy(BY.marginal(pi, bel.hyp)))
    na = bel.means.shape[1]
    best, best_v = unused[0], -np.inf
    for a in unused:
        if a < na:
            e = BY.expected_entropy_gauss(pi, bel.means[:, a], bel.sigma, bel.hyp, bel.z, bel.w)
        else:
            e = BY.expected_entropy_tel(pi, bel.hyp, bel.conf)
        v = (h_now - e) / (bel.cost[a] if by_cost else 1.0)
        if v > best_v + 1e-12:
            best, best_v = a, v
    if len(bel.cache) < 200_000:
        bel.cache[key] = best
    return best


def next_probe(pi, unused, bel: Belief, pol: Policy, rng) -> int:
    if pol.probes == "ig_cost":
        return gain_order(pi, unused, bel, True)
    if pol.probes == "ig_unit":
        return gain_order(pi, unused, bel, False)
    if pol.probes == "random":
        return int(unused[rng.integers(len(unused))])
    return unused[0]  # battery: pre-planned order


def investigate(n, t, budget, bel: Belief, wld: World, pol: Policy, rng) -> dict:
    """One investigation from pi0 (D90). Returns its outcome and the log-evidence it adds."""
    pi = bel.pi0
    h = BY.marginal(pi, bel.hyp)
    na = bel.means.shape[1]
    unused = list(range(len(bel.cost)))
    state = int(wld.cause[n]) if t >= wld.onset_idx[n] else 0
    spent, dlog, seq, cut = 0.0, 0.0, [], False
    while unused:
        a = next_probe(pi, unused, bel, pol, rng)
        c = float(bel.cost[a])
        if c > budget - spent:
            cut = True
            break
        spent += c
        seq.append(a)
        unused.remove(a)
        if a < na:
            y = float(rng.normal(wld.means[state, a], wld.sigma))
            p = float(conformal_p(np.array([y]), wld.pools[a], rng.uniform(size=1))[0])
            cm = h[1:] / h[1:].sum()
            dlog += float(np.log(bel.bets.f(np.array([p]), bel.bets.mixture(a, cm))[0]))
            pi = BY.update_gauss(pi, bel.means[:, a], bel.sigma, y)
        else:
            o = int(rng.choice(len(HYPOTHESES), p=wld.conf[state]))
            pi = BY.update_tel(pi, bel.hyp, bel.conf, o)
        h = BY.marginal(pi, bel.hyp)
        if pol.probes != "battery" and h.max() >= bel.tau:
            break
    return {
        "node": n,
        "t": t,
        "seconds": spent,
        "seq": tuple(seq),
        "dlog": dlog,
        "cut": cut,
        "pi_h0": float(h[0]),
        "cause": int(np.argmax(h[1:])) + 1,
        "pi_max": float(h.max()),
        "state": state,
    }


def run_policy(
    logM: np.ndarray,
    bel: Belief,
    wld: World,
    pol: Policy,
    rng: np.random.Generator,
    alpha: float,
    m: int,
    exclude_blocks: int,
) -> tuple[np.ndarray, list[dict]]:
    """Returns log E (n_evals, n_nodes) and the list of investigations with probes run."""
    T, n_nodes = logM.shape
    logE = logM.copy()
    if pol.B <= 0:
        return logE, []
    lp = np.zeros(n_nodes)
    until = np.full(n_nodes, -1)
    done = np.zeros(n_nodes, bool)  # latest investigation ended with a cause at max pi >= tau
    per_block = pol.B / BLOCKS_PER_DAY
    left, sweep, out = 0.0, 0, []
    for t in range(T):
        logE[t] = logM[t] + lp
        budget = min(pol.B, left + per_block)
        if pol.nodes == "rank":
            order = np.argsort(-logE[t], kind="stable")
        elif pol.nodes == "random":
            order = rng.permutation(n_nodes)
        else:
            order = (sweep + np.arange(n_nodes)) % n_nodes
        skip = until > t
        if pol.exclusion == "cleared_skipdone":
            skip = skip | (ebh_log(logE[t], alpha, m) & done)
        for n in order:
            n = int(n)
            if skip[n]:
                continue
            r = investigate(n, t, budget, bel, wld, pol, rng)
            budget -= r["seconds"]
            if r["seconds"] > 0:
                out.append(r)
                lp[n] += r["dlog"]
                cleared = r["pi_h0"] >= bel.tau
                done[n] = (not cleared) and r["pi_max"] >= bel.tau
                if (cleared and pol.exclusion != "none") or pol.exclusion == "all":
                    until[n] = t + exclude_blocks
                if pol.nodes == "sweep":
                    sweep = (n + 1) % n_nodes
            if r["cut"]:
                break
        left = budget
    return logE, out


def diagnosis_table(invs: list[dict], truth: list[tuple[int, int, int]]) -> list[dict]:
    """truth rows: (node, cause index, onset evaluation index). The output of a node is its latest
    investigation at or after onset (D90). None = never investigated after onset."""
    rows = []
    for n, c, on in truth:
        last = None
        for r in invs:
            if r["node"] == n and r["t"] >= on:
                last = r
        rows.append(
            {
                "node_id": n,
                "cause_true": HYPOTHESES[c],
                "investigated": last is not None,
                "cause_out": CAUSES[last["cause"] - 1] if last else None,
                "pi_out": last["pi_max"] if last else None,
                "correct": bool(last and last["cause"] == c),
            }
        )
    return rows
