"""Build the policy's beliefs and the simulated world from configs/phase5.yaml (Phase 5)."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from hpcdiag.layer3 import bayes as BY
from hpcdiag.layer3 import evalue as EV
from hpcdiag.layer3 import probes as PR
from hpcdiag.layer3.simulate import Belief, World
from hpcdiag.utils.run import resolve


@dataclass(frozen=True)
class Variant:
    """One row of the sensitivity table (design §2, §4, §6; D87, D88)."""

    name: str = "main"
    n_pool: int | None = None  # None: config value
    sigma: float | None = None  # belief and outcomes
    k_primary: float | None = None  # belief and outcomes
    pi0_h0: float | None = None
    redraw_pool: bool = False  # D88
    misspec: bool = False  # outcomes from a perturbed table and true_sigma; pool at true_sigma
    true_sigma: float = 0.1
    tel_belief: str = "crossblock"  # crossblock (D98) | oof | design: matrix the posterior uses
    tel_world: str = "crossblock"  # matrix outcomes are drawn from
    bets: str = "state"  # state: over (cause, s0) (D96) | cause13: per cause at s0 = 1.3 (D87)
    cross_machine: bool = False  # D119: pool from one host's fit runs, outcomes from another's


VARIANTS = {
    "main": Variant(),
    "n_pool_200": Variant("n_pool_200", n_pool=200),
    "pool_redraw": Variant("pool_redraw", redraw_pool=True),
    "sigma_0.1": Variant("sigma_0.1", sigma=0.1),
    "k_primary_2": Variant("k_primary_2", k_primary=2.0),
    "pi0_0.1": Variant("pi0_0.1", pi0_h0=0.1),
    "pi0_0.9": Variant("pi0_0.9", pi0_h0=0.9),
    "misspec": Variant("misspec", misspec=True),
    "tel_oof": Variant("tel_oof", tel_belief="oof", tel_world="oof"),
    "tel_design": Variant("tel_design", tel_belief="design", tel_world="design"),
    "tel_transfer": Variant("tel_transfer", tel_belief="oof", tel_world="crossblock"),
    "bets_cause13": Variant("bets_cause13", bets="cause13"),
    "cross_machine": Variant("cross_machine", cross_machine=True),  # measured tables only (D119)
}


def policy_model(pc: dict, v: Variant) -> PR.ProbeModel:
    """The table and sigma the policy and the bets use."""
    m = PR.from_config(pc)
    if v.sigma is not None:
        m = replace(m, sigma=v.sigma)
    if v.k_primary is not None:
        m = PR.with_primary(m, v.k_primary)
    return m


def fixed_pools(names, sigma, n_pool: int, seed_key: list[int]) -> list[np.ndarray]:
    """One sorted healthy pool per probe, in the order of names. sigma: scalar or per probe."""
    rng = np.random.default_rng(seed_key)
    return [np.sort(rng.normal(0, PR.sig(sigma, i), n_pool)) for i in range(len(names))]


def read_measured(path: str) -> pd.DataFrame:
    """Measured healthy runs; floats parsed exactly (round_trip): values read back bit for bit."""
    return pd.read_csv(resolve(path), float_precision="round_trip")


def is_measured_tables(df: pd.DataFrame) -> bool:
    """D119: a CSV with a split column holding test rows also defines the outcome draws."""
    return "split" in df.columns and bool((df["split"] == "test").any())


def empirical_pools(path: str, names, n_pool: int) -> list[np.ndarray]:
    """D113/D119: measured healthy pool. CSV with columns probe, log_x, t_start (and optionally
    host, split); per probe the FIRST n_pool runs in time order (time order is sacred) among the
    fit rows if a split column exists, sorted. Too few runs or a missing probe: error."""
    df = read_measured(path)
    if "split" in df.columns:
        df = df[df["split"] == "fit"]
    out = []
    for a in names:
        d = df[df["probe"] == a].sort_values("t_start", kind="stable")
        if len(d) < n_pool:
            raise ValueError(
                f"empirical pool {path}: probe {a} has {len(d)} runs, n_pool is {n_pool}"
            )
        out.append(np.sort(d["log_x"].to_numpy(float)[:n_pool]))
    return out


def resampled_pools(fit: pd.DataFrame, names, n_pool: int, seed_key: list[int]) -> list[np.ndarray]:
    """D119: n_pool fit-day runs per probe drawn with replacement (time order fixes the sample)."""
    rng = np.random.default_rng(seed_key)
    out = []
    for a in names:
        v = fit[fit["probe"] == a].sort_values("t_start", kind="stable")["log_x"].to_numpy(float)
        if len(v) == 0:
            raise ValueError(f"no fit-day runs for probe {a}")
        out.append(np.sort(rng.choice(v, n_pool, replace=True)))
    return out


def outcome_noise(test: pd.DataFrame, names) -> list[np.ndarray]:
    """D119: test-day healthy log x per probe, in time order; outcome = model mean + a draw."""
    out = []
    for a in names:
        v = test[test["probe"] == a].sort_values("t_start", kind="stable")["log_x"].to_numpy(float)
        if len(v) == 0:
            raise ValueError(f"no test-day runs for probe {a}")
        out.append(v)
    return out


def telemetry_rows(cfg: dict, which: str = "oof") -> np.ndarray:
    """P(class | hypothesis), rows and columns in the order of HYPOTHESES. oof: the measured
    out-of-fold counts (D94); crossblock: train on one Antarex block, test on the other. Add-one
    smoothing [D] so that no outcome has probability 0. design, or no counts configured: the design
    confusion [A]."""
    t = cfg.get("telemetry", {})
    key = {"oof": "counts", "crossblock": "crossblock_counts"}.get(which)
    if key is not None and t.get(key) is not None:
        c = np.asarray(t[key], float) + 1.0
        return c / c.sum(axis=1, keepdims=True)
    p = cfg["policy"]
    return PR.design_confusion(p["tel_correct"], p["tel_confused"], p["tel_healthy"])


def make_belief(cfg: dict, v: Variant) -> Belief:
    pc, ec, po = cfg["probes"], cfg["evalue"], cfg["policy"]
    m = policy_model(pc, v)
    n_pool = v.n_pool or ec["n_pool"]
    rng = np.random.default_rng([cfg["seed"], n_pool])
    if v.bets == "state":
        bets = EV.build_state_bets(m, {**ec, "n_pool": n_pool}, po["s_grid"], rng)
    else:
        bets = EV.build_bets(m, {**ec, "n_pool": n_pool}, rng)
    means, hyp = PR.state_table(m, po["s_grid"])
    z, w = np.polynomial.hermite_e.hermegauss(po["gh_nodes"])
    return Belief(
        means=means,
        hyp=hyp,
        sigma=np.asarray(m.sigma, float) if np.ndim(m.sigma) else m.sigma,
        conf=telemetry_rows(cfg, v.tel_belief),
        bets=bets,
        pi0=BY.prior(hyp, po["pi0_h0"] if v.pi0_h0 is None else v.pi0_h0),
        tau=float(po["tau"]),
        cost=np.array([*m.cost, float(po["tel_cost_s"])]),
        z=z,
        w=w / w.sum(),
    )


def make_world(
    cfg: dict,
    v: Variant,
    s: float,
    cause: np.ndarray,
    onset_idx: np.ndarray,
    base_key: int,
    scenario_key: int,
) -> World:
    """Outcome side for one scenario. Pools: fixed (main), or redrawn per base (D88); misspecified
    runs draw outcomes from a perturbed table at the true sigma, and the pool too (D90)."""
    pc, ec = cfg["probes"], cfg["evalue"]
    m = policy_model(pc, v)
    n_pool = v.n_pool or ec["n_pool"]
    sigma = m.sigma
    if v.misspec:
        m = PR.perturbed(m, np.random.default_rng([cfg["seed"], scenario_key, 77]))
        sigma = v.true_sigma
    key = [ec["pool_seed"], n_pool]
    if v.redraw_pool:
        key = [ec["pool_seed"], n_pool, base_key]
    elif v.misspec:
        key = [ec["pool_seed"], n_pool, 1]
    means = np.array([[m.delta(h, a, s) for a in m.names] for h in PR.HYPOTHESES])
    noise = None
    if pc.get("healthy_pool"):  # D113: measured pool replaces the Gaussian pool
        df = read_measured(pc["healthy_pool"])
        if v.misspec:
            raise ValueError("the misspecified row is not run with measured tables (D119)")
        if not is_measured_tables(df):  # pool only, Gaussian outcomes (D113)
            if v.redraw_pool or v.cross_machine:
                raise ValueError(
                    "pool per base / cross-machine need measured tables (split column)"
                )
            pools = empirical_pools(pc["healthy_pool"], m.names, n_pool)
        else:  # D119: outcomes = model mean + resampled test-day healthy log x
            fit, test = df[df["split"] == "fit"], df[df["split"] == "test"]
            if v.cross_machine:
                hosts = sorted(set(fit["host"]) & set(test["host"]))
                if len(hosts) < 2:
                    raise ValueError("cross-machine row needs fit and test runs from >= 2 hosts")
                pools = resampled_pools(
                    fit[fit["host"] == hosts[0]], m.names, n_pool, [ec["pool_seed"], n_pool, 2]
                )
                noise = outcome_noise(test[test["host"] == hosts[1]], m.names)
            else:
                if v.redraw_pool:
                    pools = resampled_pools(fit, m.names, n_pool, key)
                else:
                    pools = empirical_pools(pc["healthy_pool"], m.names, n_pool)
                noise = outcome_noise(test, m.names)
    elif v.cross_machine:
        raise ValueError("the cross-machine row needs measured tables (D119)")
    else:
        pools = fixed_pools(m.names, sigma, n_pool, key)
    return World(
        pools=pools,
        means=means,
        sigma=np.asarray(sigma, float) if np.ndim(sigma) else sigma,
        conf=telemetry_rows(cfg, v.tel_world),
        cause=cause,
        onset_idx=onset_idx,
        pool_key=hash(tuple(key)) if not v.misspec else hash((*key, sigma)),
        noise=noise,
    )
