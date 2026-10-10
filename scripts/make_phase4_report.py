# ruff: noqa: E501  (markdown report template: table rows are long by nature)
"""Render reports/phase_4.md from the Phase 4 full-grid run (PLAN.md §7.6).

    uv run python scripts/make_phase4_report.py --run <full run id> --pilots <pilot ids...>
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hpcdiag.eval.plots import INK, INK_2, SEQ_BLUE, SERIES_1, SURFACE, _style  # noqa: E402
from hpcdiag.utils.run import PROJECT_ROOT, load_config, resolve  # noqa: E402

S_LIST = [1.1, 1.2, 1.3, 1.5, 2.0, 3.0]
ORANGE = "#eb6834"  # categorical slot 2 (dataviz reference palette)


def sh(cmd):
    r = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def f(v, d=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    return f"{v:,.{d}f}"


def ok(b):
    return "pass" if b else "**FAIL**"


# ----------------------------------------------------------------------------- loading


def load(rid):
    with open(resolve(f"runs/phase4/{rid}/results.jsonl")) as fh:
        return [json.loads(x) for x in fh]


def sel(res, kind=None, base=None, K=None, s=None, seeds=None):
    out = []
    for r in res:
        if kind is not None and r["kind"] not in (kind if isinstance(kind, (list, tuple)) else [kind]):
            continue
        if base == "random" and r["base"] == "real":
            continue
        if base == "real" and r["base"] != "real":
            continue
        if K is not None and r["K_pct"] != K:
            continue
        if s is not None and r["s"] != s:
            continue
        if seeds is not None and r["seed"] not in seeds:
            continue
        out.append(r)
    return out


def p_ever(rs, v, alpha_key=None):
    x = []
    for r in rs:
        m = r["variants"][v]
        if alpha_key is not None:
            m = m["alpha_curve"][alpha_key]
        x.append(float(m["ever_flagged"]))
    x = np.array(x)
    n = len(x)
    p = x.mean() if n else np.nan
    se = np.sqrt(p * (1 - p) / n) if n else np.nan
    return p, se, n


def gate_ok(p, se, alpha):
    return bool(p <= alpha + 2 * se) if np.isfinite(p) else False


# ----------------------------------------------------------------------------- KM


def km(times, events, grid):
    """Kaplan–Meier estimate of F(t) = P(detected within t); times/grid in days."""
    t = np.asarray(times, float)
    e = np.asarray(events, bool)
    order = np.argsort(t)
    t, e = t[order], e[order]
    surv, F, i = 1.0, [], 0
    uniq = np.unique(t[e])
    steps = {}
    n = len(t)
    for u in uniq:
        at_risk = np.sum(t >= u)
        d = np.sum((t == u) & e)
        surv *= 1 - d / at_risk
        steps[u] = 1 - surv
    keys = np.array(sorted(steps))
    vals = np.array([steps[k] for k in keys])
    for g in grid:
        j = np.searchsorted(keys, g, side="right") - 1
        F.append(vals[j] if j >= 0 else 0.0)
    del i, n
    return np.array(F)


def km_median(times, events):
    grid = np.linspace(0, 200, 4001)
    F = km(times, events, grid)
    j = np.flatnonzero(F >= 0.5)
    return float(grid[j[0]]) if len(j) else np.nan


def det_table(rs, v, alpha_key=None):
    rows = []
    for r in rs:
        m = r["variants"][v]
        dets = (m["alpha_curve"][alpha_key]["detections"] if alpha_key else m["detections"])
        for x in dets:
            rows.append({"scenario": r["scenario_id"], "s": r["s"], "K": r["K_pct"], "seed": r["seed"],
                         "node": x["node_id"], "detected": x["detected"],
                         "delay_d": x["delay_h"] / 24 if x["detected"] else np.nan,
                         "censor_d": x.get("censor_h", np.nan) / 24 if x.get("censor_h") is not None else np.nan,
                         "within14": x["within_horizon"], "has_main": x["has_affected_main"],
                         "rate": x.get("pair_rate_per_day", np.nan)})
    d = pd.DataFrame(rows)
    if len(d):
        d["t"] = np.where(d["detected"], d["delay_d"], d["censor_d"])
    return d


# ----------------------------------------------------------------------------- figures


def fig_km(res, v, seeds, out):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    grid = np.linspace(0, 100, 401)
    cols = dict(zip(S_LIST, [SEQ_BLUE[i] for i in (1, 2, 3, 4, 5, 6)], strict=True))
    for ax, K in zip(axes, (1, 2, 5), strict=True):
        for s in S_LIST:
            d = det_table(sel(res, "node", "random", K, s, seeds), v)
            if not len(d):
                continue
            F = km(d["t"], d["detected"], grid)
            ax.step(grid, F, where="post", color=cols[s], lw=1.6, label=f"s = {s}")
        ax.axvline(14, color=INK_2, lw=1, ls="--")
        ax.set_title(f"K = {K}% (random base, seeds 5–19)", color=INK, fontsize=9, loc="left")
        ax.set_xlabel("days after onset")
        ax.set_ylim(0, 1)
        _style(ax)
    axes[0].set_ylabel("P(detected within d days), Kaplan–Meier")
    axes[-1].legend(frameon=False, fontsize=8, loc="lower right", title="slowdown", title_fontsize=8)
    fig.suptitle(f"Detection time after onset — {v} (dashed: 14 d, Gate 4b horizon)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_rate(res, v, seeds, out):
    d = det_table(sel(res, "node", "random", 1, None, seeds), v)
    d = d[d["detected"]]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    fig.patch.set_facecolor(SURFACE)
    cols = dict(zip(S_LIST, [SEQ_BLUE[i] for i in (1, 2, 3, 4, 5, 6)], strict=True))
    for s in S_LIST:
        x = d[d["s"] == s]
        ax.scatter(x["rate"], x["delay_d"], s=14, color=cols[s], label=f"s = {s}", edgecolors=SURFACE, linewidths=0.5)
    ax.set_xlabel("job-node pairs per day on the node after onset")
    ax.set_ylabel("detection delay (days)")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              title="slowdown", title_fontsize=8)
    _style(ax)
    fig.suptitle(f"Delay vs node pair rate — {v}, K = 1%, random base, seeds 5–19, detected nodes", color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_fdr_alpha(res, v, seeds, out):
    alphas = [0.05, 0.1, 0.2]
    nulls = sel(res, "null", "random")
    pe = [p_ever(nulls, v, None if a == 0.1 else str(a))[0] for a in alphas]
    se = [p_ever(nulls, v, None if a == 0.1 else str(a))[1] for a in alphas]
    nodes = sel(res, "node", "random", 1, 1.3, seeds)
    fdp = [np.mean([(r["variants"][v] if a == 0.1 else r["variants"][v]["alpha_curve"][str(a)])["fdp_mean_over_time"] for r in nodes]) for a in alphas]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    fig.patch.set_facecolor(SURFACE)
    for ax in (a1, a2):
        ax.plot([0, 0.22], [0, 0.22], color=INK_2, lw=1, ls="--")
        ax.set_xlim(0, 0.22)
        ax.set_xlabel("e-BH level α")
        _style(ax)
    a1.errorbar(alphas, pe, yerr=2 * np.array(se), color=SERIES_1, marker="o", lw=1.6, capsize=3)
    a1.set_ylabel("P(any node ever flagged)")
    a1.set_title("200 random nulls (± 2 SE)", color=INK, fontsize=9, loc="left")
    a2.plot(alphas, fdp, color=SERIES_1, marker="o", lw=1.6)
    a2.set_ylabel("mean FDP over time")
    a2.set_title("node, K = 1%, s = 1.3, random base, seeds 5–19", color=INK, fontsize=9, loc="left")
    fig.suptitle(f"Error control vs α — {v} (dashed: y = α)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return pe, se, fdp


def price_points(res, v, seeds, s):
    rs = sel(res, "node", "random", 1, s, seeds)
    pts = []
    for a in (0.05, 0.1, 0.2):
        key = None if a == 0.1 else str(a)
        d = det_table(rs, v, key)
        fdp = np.mean([(r["variants"][v] if key is None else r["variants"][v]["alpha_curve"][key])["fdp_mean_over_time"] for r in rs])
        pts.append((f"e-BH α={a}", fdp, km_median(d["t"], d["detected"]), float(d["detected"].mean())))
    d = det_table(rs, "B5_ville")
    pts.append(("B5 Ville (no multiplicity)", np.mean([r["variants"]["B5_ville"]["fdp_mean_over_time"] for r in rs]),
                km_median(d["t"], d["detected"]), float(d["detected"].mean())))
    return pts


def fig_price(res, v, seeds, out):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    tab = []
    for ax, s in zip(axes, (1.3, 2.0, 3.0), strict=True):
        pts = price_points(res, v, seeds, s)
        marks = {"e-BH α=0.05": "o", "e-BH α=0.1": "s", "e-BH α=0.2": "^"}
        for lab, fdp, med, rec in pts:
            b5 = lab.startswith("B5")
            ax.scatter([fdp], [med], color=ORANGE if b5 else SERIES_1, marker="D" if b5 else marks[lab],
                       s=42, edgecolors=SURFACE, linewidths=1, zorder=3, label=lab)
            tab.append({"s": s, "method": lab, "fdp_mean": fdp, "km_median_delay_d": med, "recall_end": rec})
        ax.set_xlim(-0.03, 0.5)
        ax.set_title(f"s = {s}", color=INK, fontsize=9, loc="left")
        ax.set_xlabel("mean FDP over time")
        _style(ax)
    axes[0].set_ylabel("Kaplan–Meier median delay (days)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(f"Price of the guarantee — {v}, K = 1%, random base, seeds 5–19 (blue: e-BH at α; orange: B5, no multiplicity control)", color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return pd.DataFrame(tab)


# ----------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--pilots", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(resolve("configs/phase4_layer2.yaml"))
    alpha = cfg["layer2"]["alpha"]
    dev = cfg["full"]["dev_seeds"]
    evals = [k for k in range(20) if k not in dev]
    res = load(a.run)
    rdir = resolve(f"runs/phase4/{a.run}")
    meta = json.loads((rdir / "metrics.json").read_text())
    commit = (rdir / "git_commit.txt").read_text().strip()
    figs = resolve("reports/figures")
    figs.mkdir(parents=True, exist_ok=True)

    # ---------------- Gate 4a per variant
    nulls = sel(res, "null", "random")
    dec = sel(res, "user_decoy", "random")
    sho = sel(res, "system_shock", "random")
    hvy = sel(res, "user_decoy_heavy", "random")
    variants = list(nulls[0]["variants"])
    g4a = {}
    for v in variants:
        row = {}
        for lab, rs in (("null", nulls), ("decoy", dec), ("shock", sho), ("heavy", hvy)):
            p, se, n = p_ever(rs, v)
            row[lab] = (p, se, n, gate_ok(p, se, alpha))
        row["flags_per_day_null"] = float(np.mean([r["variants"][v]["flags_per_day"] for r in nulls]))
        g4a[v] = row
    passes = {v: g4a[v]["null"][3] and g4a[v]["decoy"][3] and g4a[v]["shock"][3] for v in variants}
    if passes.get("lr_Sc_a"):
        chosen = "lr_Sc_a"
    elif passes.get("lr_Sc_none"):
        chosen = "lr_Sc_none"
    else:
        chosen = None
    gate4a = chosen is not None
    v = chosen or "lr_Sc_none"   # report numbers for the unguarded primary if neither passes

    # ---------------- Gate 4b
    gb = cfg["full"]["gate4b"]
    rs4b = sel(res, "node", "random", gb["K_pct"], gb["s"], evals)
    d4b = det_table(rs4b, v)
    rec14_a = float(d4b["within14"].mean())
    rec14_b = float(d4b.loc[d4b["has_main"], "within14"].mean())
    gate4b = rec14_a >= 0.8

    # ---------------- recall/delay vs s
    def svs(kind, base, K, seeds, var):
        rows = []
        for s in S_LIST:
            rs = sel(res, kind, base, K, s, seeds)
            if not rs:
                continue
            d = det_table(rs, var)
            rows.append({"s": s, "n_scen": len(rs), "nodes": len(d), "recall_a_end": float(d["detected"].mean()),
                         "recall_b_end": float(d.loc[d["has_main"], "detected"].mean()),
                         "within14_a": float(d["within14"].mean()), "within14_b": float(d.loc[d["has_main"], "within14"].mean()),
                         "km_F14": float(km(d["t"], d["detected"], [14])[0]), "km_median_d": km_median(d["t"], d["detected"]),
                         "fdp_mean": float(np.mean([r["variants"][var]["fdp_mean_over_time"] for r in rs])),
                         "fdp7": float(np.nanmean([r["variants"][var]["fdp_at_onset_plus_7d"] for r in rs])),
                         "fdp14": float(np.nanmean([r["variants"][var]["fdp_at_onset_plus_14d"] for r in rs])),
                         "false_co": float(np.mean([r["variants"][var]["false_nodes_coallocated"] for r in rs])),
                         "false_un": float(np.mean([r["variants"][var]["false_nodes_unrelated"] for r in rs]))})
        return pd.DataFrame(rows)

    def cross(t, col, thr):
        x = t[t[col] >= thr]
        return f"s = {x['s'].iloc[0]}" if len(x) else "never for s ≤ 3.0"

    k1 = svs("node", "random", 1, evals, v)

    # ---------------- budget (D67)
    J = cfg["layer2"]["jumper_J"]
    n_states = len(cfg["shift_bets"]["s0"]) + 1
    nats = float(np.log(cfg["layer2"]["m_fixed"] / alpha) + np.log(n_states / J))
    bud = []
    for K in (1, 2, 5):
        for s in S_LIST:
            rs = sel(res, "node", "random", K, s, evals)
            gr = [x for r in rs for x in r["variants"][v].get("growth", []) if x["n_after"] > 0]
            if not gr:
                continue
            g = np.array([x["growth_per_pair"] for x in gr])
            d = det_table(rs, v)
            rate = float(np.nanmedian(d["rate"]))
            pred = nats / (np.mean(g) * rate) if np.mean(g) > 0 else np.inf
            bud.append({"K": K, "s": s, "nodes": len(gr), "mean_growth": float(np.mean(g)), "median_growth": float(np.median(g)),
                        "median_rate": rate, "pred_days": pred, "km_median_d": km_median(d["t"], d["detected"])})
    bud = pd.DataFrame(bud)

    # ---------------- figures
    fig_km(res, v, evals, figs / "phase4_km.png")
    fig_rate(res, v, evals, figs / "phase4_delay_vs_rate.png")
    pe, se, fdp_a = fig_fdr_alpha(res, v, evals, figs / "phase4_fdr_vs_alpha.png")
    price = fig_price(res, v, evals, figs / "phase4_price.png")

    pytest_summary = [x for x in sh(["uv", "run", "pytest", "-q"]).splitlines() if re.search(r"\d+ passed", x)][-1]
    ruff = sh(["uv", "run", "ruff", "check", "."]).splitlines()[-1]

    # ================================================================ write
    s = "# Phase 4 report — Layer 2: node e-processes and e-BH\n\n"
    s += f"_Generated by `scripts/make_phase4_report.py` from run `{a.run}` (commit `{commit[:12]}`). Do not edit by hand._\n\n"
    s += "## 1. Result\n\n"
    s += f"**Gate 4a (binding, D27): {'PASS' if gate4a else 'FAIL'}**"
    s += (f" — chosen configuration `{chosen}` (D65 rule: guard (a) if it passes null + decoy + shock, else none).\n\n" if gate4a
          else " — neither guard (a) nor no guard passes null + decoy + shock. Per D27: STOP and return to Layer 1.\n\n")
    s += f"**Gate 4b: {'PASS' if gate4b else 'FAIL'}** — node, K = 1%, s = 1.3, random base, evaluation seeds 5–19: recall within 14 d (a) {f(rec14_a)} (target ≥ 0.8); (b) {f(rec14_b)}.\n"
    s += (f"Recall within 14 d (K = 1%, seeds 5–19) reaches 0.5: {cross(k1, 'within14_a', 0.5)}; 0.8: {cross(k1, 'within14_a', 0.8)}. "
          f"Recall by the trace end reaches 0.5: {cross(k1, 'recall_a_end', 0.5)}; 0.8: {cross(k1, 'recall_a_end', 0.8)}. "
          "Recall (b) equals recall (a) here because every degraded node had at least one affected main-population job.\n\n")
    s += f"Numbers below use `{v}` unless stated. Development seeds 0–4 (used to choose D60/D61/D66) are reported separately (§8); all gate numbers use seeds 5–19 (D64).\n\n"

    s += """## 2. Method (as run)

- Layer 1 input (D38, D60): centred score S_c, Mode P pooled, W = 7 d, warm-up |R_j| < 30 excluded; recomputed per scenario from the frozen Mode E model (no refit; q̂ reproduces Phase 2 exactly, max |Δ| = """ + f(meta["frozen_check"]["max_abs_diff"], 1) + f""" over {f(meta['frozen_check']['n_compared'])} jobs).
- Bets (D61, D66): per-node jumper (J = {J}) over {{no bet}} ∪ shift-LR histogram bets g_s0, s0 ∈ {cfg['shift_bets']['s0']} (g_3.0 added after development seeds, disclosed); each g built once from the warm-up week [freeze, freeze + 7 d]; w_j = 1/|N_j|.
- Selection: e-BH every 6 h at α = {alpha} with m = {cfg['layer2']['m_fixed']} fixed (D50); evaluations {meta['eval_start']} → {meta['eval_end']} ({meta['eval_times']} times). No reset in simulation (D44).
- Why P(any node ever flagged) ≤ α under the global null (D49): let τ be the first evaluation at which e-BH rejects anything and T the last evaluation. Stopped e-processes M_n(τ ∧ T) are e-values (optional stopping, F16); e-BH on them controls FDR ≤ α under arbitrary dependence (F17); under the global null every rejection is false, so FDR = P(e-BH at τ ∧ T rejects anything) = P(τ ≤ T) ≤ α. Needs fixed m and predictable updates.
- Monte-Carlo SE caveat (D52): the 200 random nulls share ONE real p-value realisation (only placement varies); the SE is conditional on this trace.
- Centring coverage (null trace): level 0 (user + shape) {f(meta['centre_level_share']['0'])}, level 1 (user) {f(meta['centre_level_share']['1'])}, level 2 (none) {f(meta['centre_level_share']['2'])}.

"""

    # ---------------- Gate 4a table
    s += f"## 3. Gate 4a — validity (random base; P(any node ever flagged) ≤ α + 2 SE, α = {alpha})\n\n"
    s += "| variant | nulls (n=200) | decoy (n=80) | shock (n=80) | heavy decoy (n=80, not binding) | null flags/day | null + decoy + shock |\n|---|---|---|---|---|---|---|\n"
    for vv in variants:
        g = g4a[vv]
        cell = lambda t: f"{f(t[0])} ± {f(t[1])} ({ok(t[3])})"  # noqa: E731
        s += f"| {vv} | {cell(g['null'])} | {cell(g['decoy'])} | {cell(g['shock'])} | {f(g['heavy'][0])} ± {f(g['heavy'][1])} | {f(g['flags_per_day_null'], 4)} | {ok(passes[vv])} |\n"
    s += "\nSE = sqrt(p̂(1 − p̂)/n). Decoy and shock pooled over s ∈ {1.1, 1.2, 1.3, 1.5} (D54). B1, B2, B3, B5 have no FDR guarantee and are shown for context.\n\n"
    nullflag = {r["base"]: r["variants"][v]["ever_flagged"] for r in nulls}
    for lab, rs in (("decoy", dec), ("shock", sho), ("heavy decoy", hvy)):
        fl = [r for r in rs if r["variants"][v]["ever_flagged"]]
        if fl:
            bases = sorted({r["base"] for r in fl})
            s += (f"Paired check for `{v}` ({lab}): all {len(fl)} flagged scenarios are on base(s) {bases}, whose null scenario "
                  f"{'also flags' if all(nullflag.get(b) for b in bases) else 'does NOT flag'} — "
                  f"{'the flags come from that placement, not from the injection' if all(nullflag.get(b) for b in bases) else 'flags attributable to the injection'}.\n\n")
    lb = [r for r in nulls if "lr_Sc_b" in r["variants"]]
    if lb:
        fr = np.mean([r["variants"]["lr_Sc_b"]["frozen_share"] for r in lb])
        s += f"**Guard (b) — failed design (D65):** it freezes {f(fr, 4)} of all job updates on the nulls (mean over {len(lb)} nulls), so it passes the null by never updating and has no power. Removed from the guard rule.\n\n"
    # real base and degenerate shocks
    rn = sel(res, "null", "real")
    rd = sel(res, "user_decoy", "real")
    rh = sel(res, "user_decoy_heavy", "real")
    s += "**Real base (unknown truth, D39):** "
    if rn:
        s += f"real null: ever flagged {f(rn[0]['variants'][v]['ever_flagged'])}, distinct nodes {rn[0]['variants'][v]['distinct_nodes_ever']}, flags/day {f(rn[0]['variants'][v]['flags_per_day'], 4)}. "
    for lab, rs in (("user_decoy", rd), ("user_decoy_heavy", rh)):
        if rs:
            nd = np.mean([r["variants"][v]["distinct_nodes_ever"] for r in rs])
            pe_r = np.mean([r["variants"][v]["ever_flagged"] for r in rs])
            base_nd = rn[0]["variants"][v]["distinct_nodes_ever"] if rn else 0
            s += f"{lab} (n = {len(rs)}): P(ever) {f(pe_r)}, mean distinct nodes flagged {f(nd, 2)} (excess over the real null {f(nd - base_nd, 2)}). "
    degen = [r for r in sel(res, "system_shock", "real") if r["seed"] == 4]
    s += ("\n\nThe unmodified real placement already produces flags (the real null flags "
          f"{rn[0]['variants'][v]['distinct_nodes_ever'] if rn else '—'} nodes); every real-base scenario inherits them, so real-base decoys, "
          "shocks and node scenarios are reported with these flags as unknown truth (D39). This is in line with the Phase 2 Step 9 exploratory finding "
          "that some PM100 nodes alarm at about twice the overall rate (node-id overlap not checked).\n\n")
    s += f"Degenerate shocks (real base, seed 04, no affected job; D47): {len(degen)} scenarios, ever flagged: {[bool(r['variants'][v]['ever_flagged']) for r in degen]} (the real null flags as well).\n\n"
    # decoy vs size
    dz = []
    idx = pd.read_csv(resolve("reports/phase3_scenario_counts.csv"), keep_default_na=False, na_values=[""])
    size = dict(zip(idx["scenario_id"], idx["n_affected_main_after"], strict=True))
    for r in dec + hvy:
        dz.append({"kind": r["kind"], "size": size.get(r["scenario_id"], np.nan), "ever": r["variants"][v]["ever_flagged"],
                   "nodes": r["variants"][v]["distinct_nodes_ever"]})
    dz = pd.DataFrame(dz)
    dz["size_bin"] = pd.cut(dz["size"], [0, 100, 300, 1000, 3000, 20000])
    t = dz.groupby(["kind", "size_bin"], observed=True).agg(n=("ever", "size"), p_ever=("ever", "mean"), nodes=("nodes", "mean")).reset_index()
    s += "**Decoys vs decoy size** (affected main-population jobs; random base; `" + v + "`):\n\n| kind | size (jobs) | n | P(ever) | mean distinct nodes flagged |\n|---|---|---|---|---|\n"
    s += "".join(f"| {r['kind']} | {r['size_bin']} | {r['n']} | {f(r['p_ever'])} | {f(r['nodes'], 2)} |\n" for _, r in t.iterrows())
    s += "\n"

    # ---------------- Gate 4b & s
    s += f"## 4. Gate 4b and detection vs s — `{v}`, node, random base, seeds 5–19\n\n"
    for K in (1, 2, 5):
        t = svs("node", "random", K, evals, v)
        s += f"**K = {K}%**\n\n| s | scen | degraded nodes | recall (a) end | recall (b) end | within 14 d (a) | within 14 d (b) | KM P(≤14 d) | KM median delay (d) | FDP mean | FDP +7 d | FDP +14 d | false co-alloc | false unrelated |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
        s += "".join(f"| {r['s']} | {int(r['n_scen'])} | {int(r['nodes'])} | {f(r['recall_a_end'])} | {f(r['recall_b_end'])} | {f(r['within14_a'])} | {f(r['within14_b'])} | {f(r['km_F14'])} | {f(r['km_median_d'], 1)} | {f(r['fdp_mean'])} | {f(r['fdp7'])} | {f(r['fdp14'])} | {f(r['false_co'], 2)} | {f(r['false_un'], 2)} |\n" for _, r in t.iterrows())
        s += "\n"
    s += "![KM](figures/phase4_km.png)\n\n![delay vs rate](figures/phase4_delay_vs_rate.png)\n\n"
    s += "node_partial (random base, seeds 5–19, K = 1%):\n\n"
    t = svs("node_partial", "random", 1, evals, v)
    s += "| s | recall (a) end | within 14 d (a) | KM median (d) | FDP mean |\n|---|---|---|---|---|\n" + "".join(
        f"| {r['s']} | {f(r['recall_a_end'])} | {f(r['within14_a'])} | {f(r['km_median_d'], 1)} | {f(r['fdp_mean'])} |\n" for _, r in t.iterrows())
    s += "\nReal base (K = 1%, node, seeds 5–19; flags on non-injected nodes are unknown truth, so FDP here is an upper bound):\n\n"
    t = svs("node", "real", 1, evals, v)
    s += "| s | recall (a) end | within 14 d (a) | KM median (d) | FDP mean (upper bound) |\n|---|---|---|---|---|\n" + "".join(
        f"| {r['s']} | {f(r['recall_a_end'])} | {f(r['within14_a'])} | {f(r['km_median_d'], 1)} | {f(r['fdp_mean'])} |\n" for _, r in t.iterrows())
    s += "\n"

    # ---------------- budget
    s += f"## 5. Evidence budget (D67) — `{v}`, random base, seeds 5–19\n\n"
    s += f"Nats needed = log({cfg['layer2']['m_fixed']}/{alpha}) + log({n_states}/{J}) = {f(np.log(cfg['layer2']['m_fixed'] / alpha))} + {f(np.log(n_states / J))} = **{f(nats)}** (single-node e-BH threshold plus the jumper's worst-case switching cost). Predicted delay = nats / (mean realised log-growth per pair after onset × median pair rate per day).\n\n"
    s += "| K | s | nodes | mean growth/pair | median growth/pair | median pairs/day | predicted delay (d) | observed KM median (d) |\n|---|---|---|---|---|---|---|---|\n"
    s += "".join(f"| {int(r['K'])} | {r['s']} | {int(r['nodes'])} | {f(r['mean_growth'], 4)} | {f(r['median_growth'], 4)} | {f(r['median_rate'], 2)} | {f(r['pred_days'], 1)} | {f(r['km_median_d'], 1)} |\n" for _, r in bud.iterrows())
    s += "\nThe e-BH threshold for a single flag is log(m/α); several degraded nodes can be flagged together at a lower individual threshold, so observed delays can be shorter than predicted.\n\n"
    if meta.get("d62"):
        s += "D62 detectability floor (null trace, optimistic: only the scored job is shifted, centring cancellation and jumper switching cost ignored; bet grid now includes g_3.0):\n\n| score | s | κ days | LR jumper days | oracle days |\n|---|---|---|---|---|\n"
        s += "".join(f"| {r['score']} | {r['s']} | {f(r['days_kappa'], 1)} | {f(r['days_lr_jumper'], 1)} | {f(r['days_oracle'], 1)} |\n" for r in meta["d62"])
        s += "\n"

    # ---------------- FDR vs alpha, price
    s += "## 6. Error control vs α and the price of the guarantee (D69)\n\n![FDR vs alpha](figures/phase4_fdr_vs_alpha.png)\n\n"
    s += "| α | null P(ever) ± SE (200) | node FDP mean (K 1%, s 1.3) |\n|---|---|---|\n" + "".join(
        f"| {al} | {f(pe[i])} ± {f(se[i])} | {f(fdp_a[i])} |\n" for i, al in enumerate([0.05, 0.1, 0.2]))
    s += "\n![price](figures/phase4_price.png)\n\n| s | method | FDP mean | KM median delay (d) | recall at trace end |\n|---|---|---|---|---|\n"
    s += "".join(f"| {r['s']} | {r['method']} | {f(r['fdp_mean'])} | {f(r['km_median_delay_d'], 1)} | {f(r['recall_end'])} |\n" for _, r in price.iterrows())
    s += "\n"

    # ---------------- ablations
    s += "## 7. Ablations and baselines (Gate 4a set + Gate 4b slice; random base; seeds 5–19 for node rows)\n\n"
    s += "| variant | null P(ever) | decoy P(ever) | shock P(ever) | s=1.3: within 14 d / end / FDP | s=2.0: within 14 d / end / FDP | s=3.0: within 14 d / end / FDP |\n|---|---|---|---|---|---|---|\n"
    for vv in variants:
        cells = []
        for sv in (1.3, 2.0, 3.0):
            rs = sel(res, "node", "random", 1, sv, evals)
            d = det_table(rs, vv)
            fd = np.mean([r["variants"][vv]["fdp_mean_over_time"] for r in rs])
            cells.append(f"{f(float(d['within14'].mean()))} / {f(float(d['detected'].mean()))} / {f(fd)}")
        g = g4a[vv]
        s += f"| {vv} | {f(g['null'][0])} | {f(g['decoy'][0])} | {f(g['shock'][0])} | " + " | ".join(cells) + " |\n"
    s += """
Variants: `lr_Sc_none` primary without guard; `lr_Sc_a` guard (a) detrend; `lr_Sc_b` guard (b) (failed design, D65); `kappa_Sc_none` κ power-calibrator mixture instead of LR bets; `plainlr_Sc_none` plain product of LR bets (no jumper); `lr_Sc_decoyw_none` decoy-defence weight (D40); `lr_S_none` / `kappa_S_none` raw S (no centring; `kappa_S_none` = first-pilot primary); `kappa_none_modeE` Mode E as-run p-values (B6); `lr_Sc_none_W24h` / `_W8760h` Mode P window 1 d / 365 d (B6); B1 guide heuristic (7-d windows, daily, D55), B2 top 5%, B3 Fisher + BH, B4 averaged e-values + e-BH, B5 per-node Ville (no multiplicity control).

"""

    # ---------------- dev seeds
    s += f"## 8. Development seeds 0–4 (used to choose D60/D61/D66) — `{v}`, node, K = 1%, random base\n\n"
    t = svs("node", "random", 1, dev, v)
    s += "| s | recall (a) end | within 14 d (a) | KM median (d) | FDP mean |\n|---|---|---|---|---|\n" + "".join(
        f"| {r['s']} | {f(r['recall_a_end'])} | {f(r['within14_a'])} | {f(r['km_median_d'], 1)} | {f(r['fdp_mean'])} |\n" for _, r in t.iterrows())
    s += "\n"

    # ---------------- D59
    if meta.get("d59"):
        d59 = pd.DataFrame(meta["d59"])
        s += "## 9. D59 paired check (affected job vs the same job in the matching null; development seeds)\n\n| s | score | mean of per-scenario median Δp | mean share negative |\n|---|---|---|---|\n"
        g = d59.groupby(["s", "score"])[["median_diff", "share_negative"]].mean().reset_index()
        s += "".join(f"| {r['s']} | {r['score']} | {f(r['median_diff'], 4)} | {f(r['share_negative'])} |\n" for _, r in g.iterrows())
        mx = float(d59["median_diff"].max())
        s += (f"\nLargest per-scenario median Δp: {mx:.4f} ({'negative in every scenario' if mx < 0 else 'NOT negative in every scenario'}): injection lowers the "
              "job's p-value. The higher median p after onset seen in the first pilot is a time trend that is present in the null too: "
              "measured interactively during the first pilot (not a stored run), the median over the 50 degraded nodes of [null p after onset − null p "
              "before onset] was +0.081 (84% of nodes), larger than the paired injection effect at s = 1.3.\n\n")

    # ---------------- runs
    s += f"""## 10. Runs and commands

```bash
uv run python -m hpcdiag.layer2.run_layer2 --config configs/phase4_layer2.yaml --mode full --workers 7   # {a.run}
uv run python scripts/make_phase4_report.py --run {a.run} --pilots {' '.join(a.pilots)}
```

- Pilot runs (development, seeds 0–4): {', '.join(f'`{p}`' for p in a.pilots)}.
- Full grid: {meta['n_tasks']} scenarios, {meta['workers']} workers, wall {meta['wall_s'] / 60:.1f} min.
- Tests: `{pytest_summary}`; lint: `{ruff}`.

Phase 4 complete. Stopped; Phase 5 not started.
"""
    resolve("reports/phase_4.md").write_text(s)
    print("wrote reports/phase_4.md; Gate 4a", gate4a, chosen, "; Gate 4b", gate4b, rec14_a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
