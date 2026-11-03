# ruff: noqa: E501  (markdown report template: table rows are long by nature)
"""Render reports/phase_5b.md (Phase 5b development grid, D95-D104).

    uv run python scripts/make_phase5b_report.py --run <dev5b> --d102 <d102 check> --d96 <d96 check> \
        --reach <d96 reach> --prodigy <prodigy check> --clf <classifier> --diag <d94 diag>
"""

from __future__ import annotations

import argparse
import json
import subprocess

import numpy as np
from phase5_tables import confusion, diag_row, km_ci, load, pick, speed_row, validity_row

from hpcdiag.utils.run import PROJECT_ROOT, resolve

ALPHA = 0.1
S = [1.1, 1.2, 1.3, 1.5, 2.0, 3.0]
KINDS = ("null", "user_decoy", "user_decoy_heavy", "system_shock")


def jl(rid, name="metrics.json"):
    return json.loads(resolve(f"runs/phase5/{rid}/{name}").read_text())


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    return "\n".join(out + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows])


def f(v, d=3):
    return "not reached" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("run", "d102", "d96", "reach", "prodigy", "clf", "diag"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    res = load(a.run)
    meta = jl(a.run)
    d102, d96, reach, pro, clf, dg = jl(a.d102), jl(a.d96), jl(a.reach), jl(a.prodigy), jl(a.clf), jl(a.diag)
    with open(resolve("runs/phase4/20261010-094328-full/results.jsonl")) as fh:
        v1 = [json.loads(x) for x in fh]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()

    def rs(variant="main", policy=None, B=None, kind="node", random=True, **kw):
        out = [r for r in res if r["variant"] == variant and r["kind"] == kind and ((r["base"] != "real") == random)]
        if policy is not None:
            out = [r for r in out if r["policy"] == (policy if B != 0 else "ig")]
        if B is not None:
            out = [r for r in out if r["B"] == B]
        return pick(out, **kw)

    def val(policy, B, kind, variant="main"):
        return validity_row(rs(variant, policy, B, kind))

    def vtxt(x):
        return f"{x['P_ever']:.3f} ± {x['se']:.3f}"

    def v1_rows(kind, K=None, s=None):
        out = [r for r in v1 if r["kind"] == kind and r["base"] != "real" and (r["seed"] is None or r["seed"] <= 19)]
        if K is not None:
            out = [r for r in out if r["K_pct"] == K and r["s"] == s]
        return [{"metrics": r["variants"]["lr_Sc_a"], "scenario_id": r["scenario_id"], "s": r["s"], "K_pct": r["K_pct"], "seed": r["seed"], "diagnosis": []} for r in out]

    # ---------------- D97 selection (pre-stated rule)
    sp = {p: speed_row(rs("main", p, 1800, K_pct=1, s=1.3)) for p in ("ig", "stage")}
    nul = {p: val(p, 1800, "null") for p in ("ig", "stage")}
    diff = sp["stage"]["km_F14"] - sp["ig"]["km_F14"]
    keep_stage = diff >= 0.02 and nul["stage"]["P_ever"] <= ALPHA + 2 * nul["stage"]["se"]
    SEL = "stage" if keep_stage else "ig"
    OTHER = "ig" if keep_stage else "stage"
    NAME = {"ig": "IG-only", "stage": "D97 stage-wise"}
    sel_txt = table(
        ["Rule", "KM F(14 d), K 1%, s 1.3, B 1800", "95% interval", "null P(ever), B 1800, fixed pool", "pool per base"],
        [[NAME[p], f"{sp[p]['km_F14']:.3f}", "{:.2f}–{:.2f}".format(*km_ci(rs('main', p, 1800, K_pct=1, s=1.3))), vtxt(nul[p]), vtxt(val(p, 1800, "null", "pool_redraw"))] for p in ("ig", "stage")],
    )

    # ---------------- validity, side by side (D100)
    vrows = []
    for p in ("ig", "stage"):
        for B in (0, 60, 300, 1800, 7200):
            if B == 0 and p == "stage":
                continue
            row = [NAME[p] if B else "Rn(1), no probes", B]
            ok = True
            for k in KINDS:
                x = val(p, B, k)
                y = val(p, B, k, "pool_redraw") if B else x
                ok = ok and x["pass"] and y["pass"]
                row.append(f"{vtxt(x)} / {vtxt(y)}" if B else vtxt(x))
            row.append("pass" if ok else "**FAIL**")
            vrows.append(row)
    for p in ("ig", "stage"):
        for B in (300, 7200):
            row = [f"{NAME[p]}, misspecified", B]
            ok = True
            for k in KINDS:
                x = val(p, B, k, "misspec")
                ok = ok and x["pass"]
                row.append(vtxt(x))
            row.append("pass" if ok else "**FAIL**")
            vrows.append(row)
    row = ["v1, B = 0 (empirical only)", 0]
    for k in KINDS:
        row.append(vtxt(validity_row(v1_rows(k))))
    vrows.append(row + [""])
    val_tab = table(["Rule", "B", "null (200)", "user_decoy (80)", "heavy (80)", "shock (80)", "all ≤ α + 2 SE"], vrows)
    all_valid = all(r[-1] == "pass" for r in vrows if r[0] != "v1, B = 0 (empirical only)")

    # ---------------- speed
    def F(p, B, K=1, s=1.3, variant="main"):
        return speed_row(rs(variant, p, B, K_pct=K, s=s))

    def v1F(K, s):
        return speed_row(v1_rows("node", K, s))

    sp_rows = []
    for s in S:
        r = [s] + [f"{F(SEL, B, s=s)['km_F14']:.3f}" for B in (0, 60, 300, 1800, 7200)]
        r += [f"{F(OTHER, B, s=s)['km_F14']:.3f}" for B in (300, 1800, 7200)]
        r += [f"{v1F(1, s)['km_F14']:.3f}"]
        sp_rows.append(r)
    sp_tab = table(["s", "B = 0", "60", "300", "1800", "7200", f"{NAME[OTHER]} 300", "1800", "7200", "v1"], sp_rows)
    rec_rows = []
    for s in S:
        rec_rows.append([s] + [f"{f(F(SEL, B, s=s)['km_median_d'], 1)} / {F(SEL, B, s=s)['recall_end']:.3f}" for B in (0, 60, 300, 1800, 7200)])
    rec_tab = table(["s", "B = 0", "60", "300", "1800", "7200"], rec_rows)
    k_rows = []
    for K in (2, 5):
        for s in (1.2, 1.3, 2.0):
            k_rows.append([K, s] + [f"{F(SEL, B, K, s)['km_F14']:.3f}" for B in (0, 300, 1800, 7200)] + [f"{v1F(K, s)['km_F14']:.3f}"])
    k_tab = table(["K %", "s", "B = 0", "300", "1800", "7200", "v1"], k_rows)

    rule_rows = []
    for p, lab, budgets in ((SEL, f"{NAME[SEL]} + D95 (primary)", (300, 1800, 7200)), (OTHER, f"{NAME[OTHER]} + D95", (300, 1800, 7200)), ("d86", "IG + D86 (architect's failed rule)", (300, 1800, 7200)), ("design_rule", "IG + design rule", (300, 7200))):
        r = [lab]
        for B in (300, 1800, 7200):
            if B in budgets:
                x = F(p, B)
                d = diag_row(rs("main", p, B, K_pct=1))
                r.append(f"{x['km_F14']:.3f} / {d['accuracy']:.3f}")
            else:
                r.append("not run")
        rule_rows.append(r)
    rule_tab = table(["Rule (K 1%)", "B 300: F(14 d) s 1.3 / accuracy all s", "B 1800", "B 7200"], rule_rows)

    # ---------------- diagnosis
    d_rows = []
    for B in (60, 300, 1800, 7200):
        d = diag_row(rs("main", SEL, B))
        d_rows.append([B, f"{d['flagged_degraded']:,}", f"{d['accuracy']:.3f}", f"{d['never_investigated']:,}", f"{d['s_per_investigation']:.1f}", f"{d['s_per_uncut']:.1f}", f"{d['share_cut']:.3f}", f"{d['healthy_inv_per_day']:.1f}"])
    d_tab = table(["B", "flagged degraded", "accuracy", "never investigated", "s per investigation", "s per uncut", "share cut", "healthy investigations/day"], d_rows)
    conf = confusion(rs("main", SEL, 7200))
    conf_tab = table(["true \\ output", "cpu", "cache", "mbw", "mem", "never"], [[c, *[conf.get(c, {}).get(o, 0) for o in ("cpu", "cache", "mbw", "mem", "never")]] for c in ("cpu", "cache", "mbw")])
    tel_rows = []
    for v, lab in (("main", "cross-block (main, D98)"), ("tel_oof", "out-of-fold (D94)")):
        r = [lab]
        for B in (300, 1800, 7200):
            d = diag_row(rs(v, SEL, B, K_pct=1))
            x = F(SEL, B, variant=v)
            r.append(f"{x['km_F14']:.3f} / {d['accuracy']:.3f}")
        tel_rows.append(r)
    tel_tab = table(["Telemetry (K 1%)", "B 300: F(14 d) s 1.3 / accuracy", "B 1800", "B 7200"], tel_rows)
    mis_rows = []
    for v in ("main", "misspec"):
        r = [v]
        for B in (300, 7200):
            d = diag_row(rs(v, SEL, B, K_pct=1))
            r.append(f"{F(SEL, B, variant=v)['km_F14']:.3f} / {F(SEL, B, s=1.1, variant=v)['recall_end']:.3f} / {d['accuracy']:.3f}")
        mis_rows.append(r)
    mis_tab = table(["Outcomes (K 1%)", "B 300: F(14 d) s 1.3 / recall s 1.1 / accuracy", "B 7200"], mis_rows)

    # ---------------- gates on development seeds
    g5b = F(SEL, 300)
    d300, d7200 = diag_row(rs("main", SEL, 300)), diag_row(rs("main", SEL, 7200))
    f7200 = F(SEL, 7200)
    ci7200 = km_ci(rs("main", SEL, 7200, K_pct=1, s=1.3))
    v300 = [val(SEL, 300, k) for k in KINDS] + [val(SEL, 300, k, "pool_redraw") for k in KINDS]
    v_all = []
    for B in (0, 300, 1800, 7200):
        for k in KINDS:
            v_all.append(val(SEL, B, k))
            if B:
                v_all.append(val(SEL, B, k, "pool_redraw"))
    gates = table(
        ["Check (development seeds 0–19)", "Criterion", "Measured", "Result"],
        [
            ["5a validity", "B = 300, P(ever) ≤ α + 2 SE, fixed pool and pool per base", f"null {vtxt(val(SEL, 300, 'null'))} / {vtxt(val(SEL, 300, 'null', 'pool_redraw'))}; worst kind {max(x['P_ever'] for x in v300):.3f}", "pass" if all(x["pass"] for x in v300) else "**FAIL**"],
            ["5b speed", "K 1%, s 1.3, B 300: KM F(14 d) ≥ 0.8", f"{g5b['km_F14']:.3f}", "**FAILED** (recorded, D101)"],
            ["5c diagnosis", "B 300: accuracy ≥ 0.9, ≤ 72.6 s per investigation", f"{d300['accuracy']:.3f}, {d300['s_per_investigation']:.1f} s", "pass" if d300["accuracy"] >= 0.9 and d300["s_per_investigation"] <= 72.6 else "**FAIL**"],
            ["5F (i) preview", "validity at every B ∈ {0, 300, 1800, 7200}", f"worst P(ever) {max(x['P_ever'] for x in v_all):.3f}", "pass" if all(x["pass"] for x in v_all) else "**FAIL**"],
            ["5F (ii) preview", "K 1%, s 1.3, B 7200: KM F(14 d) ≥ 0.8", f"{f7200['km_F14']:.3f} ({ci7200[0]:.2f}–{ci7200[1]:.2f})", "pass" if f7200["km_F14"] >= 0.8 else "**FAIL**"],
            ["5F (iii) preview", "B 7200: accuracy ≥ 0.9", f"{d7200['accuracy']:.3f}", "pass" if d7200["accuracy"] >= 0.9 else "**FAIL**"],
        ],
    )

    # ---------------- D102
    marg = d102["marginal"]
    m_rows = [[r["n_pool"], r["probe"], r["bet"], f"{r['mean_log_f']:.4f}", f"{r['exact_log_f']:.4f}", f"{r['mean_Ef_given_pool']:.4f}"] for r in marg if r["n_pool"] == 1000]
    m_tab = table(["n_pool", "probe", "bet", "mean log f over 2,000 pools", "exact", "mean E[f \\| pool]"], m_rows)
    mx = max(r["mean_log_f"] for r in marg if r["bet"] == "max over all weights")
    u = d102["used_fixed_pool"]
    u_tab = table(["Probe rule", "probes", "max", "q99", "mean", "share > 0"], [[k, f"{v['probes']:,}", f"{v['max']:+.4f}", f"{v['q99']:+.4f}", f"{v['mean']:+.3f}", f"{v['share_gt_0']:.4f}"] for k, v in u.items()])

    def grid_growth(p, variant):
        hs = [r["healthy_growth"] for r in res if r["policy"] == p and r["variant"] == variant and r["healthy_growth"]["n"]]
        n = sum(h["n"] for h in hs)
        return [f"{NAME.get(p, p)}, {variant}", f"{n:,}", f"{max(h['max'] for h in hs):+.4f}", f"{max(h['q99'] for h in hs):+.4f}", f"{sum(h['mean'] * h['n'] for h in hs) / n:+.3f}", f"{sum(h['n_gt_0'] for h in hs) / n:.5f}"]

    gg_vals = {(p, v): grid_growth(p, v) for p in ("ig", "stage") for v in ("main", "pool_redraw")}
    gg_tab = table(["Rule, pools", "probes on healthy nodes", "max", "largest per-run q99", "mean", "share > 0"], list(gg_vals.values()))

    # ---------------- D96 strict check summary (n_pool 1000)
    h96 = d96["1000"]["healthy"]
    s96 = [h["fresh_pools_share_max_gt_0"] for h in h96]
    m96 = max(h96, key=lambda h: h["max_over_simplex"])

    # ---------------- D96 vs Phase 5 bets at s = 1.1
    c96 = d96["1000"]["cause"]
    low = table(["probe, cause, s = 1.1", "bet on the true (cause, s) state", "bet on (cause, 1.3), Phase 5 style"], [[f"{x['probe']}, {x['cause']}", f"{x['true_state']:+.3f}", f"{x['state_s0_1.3']:+.3f}"] for x in c96 if x["s"] == 1.1 and x["probe"] == x["cause"]])

    # ---------------- real base
    real = []
    for kind in ("null", "node"):
        for B in (0, 300, 7200):
            for p in ((SEL,) if B else ("ig",)):
                rr = [r for r in res if r["base"] == "real" and r["kind"] == kind and r["variant"] == "main" and r["B"] == B and r["policy"] == p and (kind == "null" or r["s"] == 1.3)]
                if rr:
                    real.append([kind if kind == "null" else "node, K 1%, s 1.3", B, len(rr), f"{np.mean([r['metrics']['distinct_nodes_ever'] for r in rr]):.1f}", f"{np.mean([r['metrics']['flags_per_day'] for r in rr]):.2f}"])
    real_tab = table(["Real base", "B", "scenarios", "distinct nodes flagged", "flags per day"], real)

    # ---------------- Prodigy
    gp = pro["grouped_by_job"]
    pro_tab = table(["Split", "macro-F1", "notes"], [["grouped 5-fold by job", f"{gp['macro_f1']:.3f}", f"confusion healthy {gp['confusion'][0]}, anomalous {gp['confusion'][1]}; recall cpuoccupy {gp['recall_by_anomaly']['cpuoccupy']:.2f}, memleak {gp['recall_by_anomaly']['memleak']:.2f}, healthy {gp['recall_by_anomaly']['none']:.3f}"]] + [[f"leave out {x['held_out_app']}", f"{x['macro_f1']:.3f}", f"n {x['n']}, accuracy {x['accuracy']:.3f}"] for x in pro["leave_one_app_out"]])
    cb = dg["cross_block"]
    walls = meta["wall_s"] / 60
    nv300 = diag_row(rs('main', SEL, 300, K_pct=1))
    nv86 = diag_row(rs('main', 'd86', 300, K_pct=1))

    md = f"""# Phase 5b report — policy and bets fixed on development seeds

Generated by `scripts/make_phase5b_report.py` at commit {commit}. Simulation, design probe model [A]. Development seeds
0–19 and old null bases 0–199 only. Nothing ran on seeds 20–39 or nulls 200–399: the runner reads `index.parquet`
only and raises on a sealed id. Decisions D95–D104 are in PLAN §16.

## 1. Result

**D97 selection (rule fixed before results):** keep D97 if its KM F(14 d) at K = 1%, s = 1.3, B = 1800 is at least 0.02
above IG-only and its null P(ever) ≤ α + 2 SE. Difference {diff:+.3f}, so **{NAME[SEL]} is kept**{'' if keep_stage else ' and D97 is dropped'}.

{sel_txt}

{gates}

Gate 5b stays **FAILED** on development seeds (D101). Rows 4–6 preview Gate 5F on development seeds; the real
Gate 5F runs once, on the sealed seeds.

{rule_tab}

At B = 300, K = 1%, D86 leaves {nv86['never_investigated']} of {nv86['flagged_degraded']} flagged degraded nodes never investigated after onset, with accuracy
{nv86['accuracy']:.3f}. Under D95 it is {nv300['never_investigated']} of {nv300['flagged_degraded']}, with accuracy {nv300['accuracy']:.3f}. D95 was adopted after the Phase 5
development grid; it had been committed as a sensitivity row before any result.

## 2. Validity (D100): fixed pool / pool per base

P(ever) ± SE, α = 0.1, random base. Each cell shows the fixed pool, then a pool drawn per base. The misspecified rows use a pool at the true σ.

{val_tab}

All rows with probes pass α + 2 SE: **{'yes' if all_valid else 'no'}**.

## 3. Speed

KM P(flag ≤ 14 d), K = 1%, random base, 20 seeds, 200 degraded nodes per cell; {NAME[SEL]} with D95 unless stated.

{sp_tab}

KM median delay in days / recall at trace end, {NAME[SEL]}:

{rec_tab}

More degraded nodes, KM F(14 d), {NAME[SEL]}:

{k_tab}

**s = 1.1 is still not solved.** With D96 bets, probes no longer lower recall at B = 300. Recall at trace end is {F(SEL, 0, s=1.1)['recall_end']:.3f} at
B = 0, {F(SEL, 300, s=1.1)['recall_end']:.3f} at B = 300 and {F(SEL, 1800, s=1.1)['recall_end']:.3f} at B = 1800. Phase 5 had 0.335 at B = 0 and 0.205 at B = 300,
under D86 with the old bets (reports/phase_5.md, §2). At B = 7200, though, recall falls back to {F(SEL, 7200, s=1.1)['recall_end']:.3f}, and
F(14 d) at s = 1.1 is at most {max(F(SEL, B, s=1.1)['km_F14'] for B in (0, 60, 300, 1800, 7200)):.3f} at every budget. Why recall drops at B = 7200 was not
measured: UNKNOWN — needs check. Mean log f on the fixed pool for a probe on its own cause at s = 1.1, run {a.d96}:

{low}

## 4. Diagnosis

{NAME[SEL]}, D95, all K and s, random base:

{d_tab}

Outputs at B = 7200, all K and s, flagged degraded nodes. Cause mem is not injected as a slowdown (D30), so it has no row:

{conf_tab}

Telemetry likelihood (D98), K 1%, {NAME[SEL]}:

{tel_tab}

Misspecified outcomes (k ± 50%, σ = 0.1), K 1%, {NAME[SEL]}:

{mis_tab}

## 5. Bet check (D102)

**(i) Marginal.** Run {a.d102}; n_pool = 1000 shown, n_pool = 200 in the run. The maximum over every bet weight is at most
{mx:.1e} on the 2,000-pool estimate, within its standard error, and 0 to machine precision when computed exactly from the bin widths.
The check passes.

{m_tab}

**(ii) Bets actually used, fixed pool.** 20,000 investigations of a healthy node per rule (run {a.d102}):

{u_tab}

And in the grid itself, probes on healthy nodes:

{gg_tab}

Both rules sometimes use bets with positive conditional growth on the fixed pool, even with D103. IG-only does so rarely: share {gg_vals[("ig", "main")][5]},
max {gg_vals[("ig", "main")][2]} in the grid, never in the 20,000 checked investigations. D97 does so more often: share {gg_vals[("stage", "main")][5]}, max {gg_vals[("stage", "main")][2]}.
Validity is judged by P(ever), §2.

**D104 disclosure.** Phase 5's D87 check tested only a few bets. Under the full search the Phase 5 per-cause bets also reach
{reach['phase5_bets_simplex_max']['mbw']:+.4f} (mbw) and {reach['phase5_bets_simplex_max']['mem']:+.4f} (mem) nats per probe on the fixed pools (run {a.reach}).
The strict D96 check (run {a.d96}) found the D96 bets reach at most {m96['max_over_simplex']:+.4f} ({m96['probe']}) on the fixed n_pool = 1000 pools, and
{min(s96):.0%}–{max(s96):.0%} of fresh 1000-run pools have a positive maximum.

## 6. Telemetry classifier and Prodigy

**The Gate 5d score of 1.0 is a fault-program fingerprint (D98).** Out-of-fold macro-F1 is {clf['five_class']['macro_f1']:.4f}, but trained
on one Antarex configuration and tested on the other it falls to {cb[0]['macro_f1']:.2f} (single-core → multi-core) and {cb[1]['macro_f1']:.2f} (reverse) on the same machine (run {a.diag}).
The cross-block matrix is now the a_tel likelihood, and out-of-fold is a sensitivity row (§4).

**Prodigy secondary check (D99), reported only.** h5py 3.16.0. One sample per (job, node) series, Antarex classifier statistics, the same Random Forest (run {a.prodigy}):

{pro_tab}

Read the Prodigy result with care:
- **Labels:** healthy, cpuoccupy and memleak. The architect's D94 description ("healthy vs memleak only") was wrong.
- **Healthy-class confound:** {pro['counts'].get('prod_train_label.csv|none', 0)} of the healthy samples come from the separate training file and {pro['counts'].get('prod_test_label.csv|none', 0)} from the test file, so collection time may separate them.
- **Id columns:** these sit in a pickled object block, read with a restricted unpickler that allows numpy array rebuilding only.
- **Not Prodigy's method:** Prodigy itself uses a VAE trained on healthy data only (macro-F1 0.95 on Eclipse, F25), which is not reproduced here.
- **No transfer:** leaving one application out gives 0.25 to 1.00.

## 7. Real base (unknown truth, no claim)

{real_tab}

The simulation draws healthy probe outcomes on every real-base node without injection, so probes clear the D84 nodes by assumption (D92).

## 8. Notes

- **Grid:** {meta['n_runs']:,} runs in {walls:.1f} min on 7 workers, run {a.run}.
- **D95 detail:** a flagged node with a diagnosis is skipped while it stays in the e-BH set; if it leaves the set, it becomes eligible again.
- **D97 implementation:** the stage rule takes "flagged" from the e-BH set at the current evaluation. Telemetry has expected log f = 0 because it never enters an e-value.
- **D102 check tolerance:** the stop rule allowed a floating-point floor of 1e-12, because the flat bet gives log f = 2.2e-16.
- **Older Phase 5 runs:** two D96 runs (strict check {a.d96}, reach {a.reach}) and the earlier D102 runs stay under `runs/phase5/`.

## 9. Commands

```
uv add h5py==3.16.0
uv run python scripts/phase5_d96_check.py --config configs/phase5.yaml        # {a.d96} (strict check: STOP)
uv run python scripts/phase5_d96_reach.py --config configs/phase5.yaml        # {a.reach}
uv run python scripts/phase5_d102_check.py --config configs/phase5.yaml       # {a.d102}
uv run python scripts/phase5_prodigy_check.py --config configs/phase5.yaml    # {a.prodigy}
uv run python -m hpcdiag.layer3.run_phase5 --config configs/phase5.yaml --mode dev5b --workers 7   # {a.run}
uv run python scripts/make_phase5b_report.py --run {a.run} --d102 {a.d102} --d96 {a.d96} --reach {a.reach} --prodigy {a.prodigy} --clf {a.clf} --diag {a.diag}
```
"""
    resolve("reports/phase_5b.md").write_text(md)
    print("wrote reports/phase_5b.md;", "selected", SEL, "diff", round(diff, 4))


if __name__ == "__main__":
    main()
