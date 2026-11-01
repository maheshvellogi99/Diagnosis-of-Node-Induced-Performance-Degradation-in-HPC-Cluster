# ruff: noqa: E501  (markdown report template: table rows are long by nature)
"""Render reports/phase_5.md from the Phase 5 runs (PLAN.md §7.6).

    uv run python scripts/make_phase5_report.py --run <full> --addon <addon> --clf <classifier> \
        --diag <d94 diag> --d87 <d87 check>
"""

from __future__ import annotations

import argparse
import json
import subprocess

import numpy as np

from hpcdiag.utils.run import PROJECT_ROOT, resolve


def jl(rid, name="metrics.json"):
    return json.loads(resolve(f"runs/phase5/{rid}/{name}").read_text())


def f(v, d=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "not reached"
    return f"{v:.{d}f}"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def rows_of(rid):
    with open(resolve(f"runs/phase5/{rid}/results.jsonl")) as fh:
        return [json.loads(x) for x in fh]


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("run", "addon", "clf", "diag", "d87"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    t = jl(a.run, "tables.json")
    clf, dg, d87 = jl(a.clf), jl(a.diag), jl(a.d87)
    full, add = jl(a.run), jl(a.addon)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()

    val = t["validity"]

    def v(kind, row, B=300):
        return next(r for r in val if r["kind"] == kind and r["row"] == row and r["B"] == B)

    def vtxt(kind, row, B=300):
        r = v(kind, row, B)
        return f"{r['P_ever']:.3f} ± {r['se']:.3f}"

    def sp(K, s, row, B):
        return next(r for r in t["speed"] if r["K"] == K and r["s"] == s and r["row"] == row and r["B"] == B)

    def sens(row, s):
        return next(r for r in t["sensitivity"] if r["row"] == row and r["s"] == s)

    def dg_row(s, B):
        return next(r for r in t["diagnosis"] if r["s"] == s and r["B"] == B)

    S = [1.1, 1.2, 1.3, 1.5, 2.0, 3.0]
    g5b = sp(1, 1.3, "ours", 300)
    ci = t["gate5b_ci"]
    alt = {r["policy"]: r for r in t["gate5b_rows"]}
    rules = {r["rule"]: r for r in t["gate5c_rules"]}
    d300, d7200 = dg_row("all", 300), dg_row("all", 7200)
    pass5a = all(
        v(k, "ours")["pass"] for k in ("null", "user_decoy", "user_decoy_heavy", "system_shock")
    )
    f1 = clf["five_class"]
    cb = dg["cross_block"]
    bm = {r["policy"] + str(r["B"]): r for r in t["d91"]}

    gates = table(
        ["Gate", "Criterion", "Measured", "Result"],
        [
            ["5a validity (binding)", "B = 300, P(ever) ≤ α + 2 SE", f"null {vtxt('null', 'ours')} (n 200); user_decoy {vtxt('user_decoy', 'ours')}; heavy {vtxt('user_decoy_heavy', 'ours')}; shock {vtxt('system_shock', 'ours')} (n 80 each)", "**PASS**" if pass5a else "**FAIL**"],
            ["5b speed", "K = 1%, s = 1.3, B = 300: KM P(flag ≤ 14 d) ≥ 0.8", f"{g5b['km_F14']:.3f} (95% bootstrap over scenarios {ci[0]:.2f}–{ci[1]:.2f})", "**FAIL**"],
            ["5c diagnosis", "accuracy ≥ 0.9 and mean s per investigation ≤ 72.6", f"accuracy {d300['accuracy']:.3f}; {d300['s_per_investigation']:.1f} s (B = 300, all K and s)", "**FAIL** on accuracy"],
            ["5d classifier (D94)", "RF macro-F1 ≥ 0.90", f"{f1['macro_f1']:.4f} on {f1['n']:,} windows; see the warning in §6", "**PASS, not to be relied on**"],
        ],
    )

    budget = []
    for s in S:
        row = [s]
        for B in (0, 60, 300, 1800, 7200):
            r = sp(1, s, "ours", B)
            row.append(f"{r['km_F14']:.3f}")
        row.append(f"{sp(1, s, 'v1', 0)['km_F14']:.3f}")
        budget.append(row)
    budget = table(["s", "B = 0 (Rn(1))", "60", "300", "1800", "7200", "v1, B = 0"], budget)

    recall = []
    for s in S:
        row = [s]
        for B in (0, 60, 300, 1800, 7200):
            r = sp(1, s, "ours", B)
            row.append(f"{f(r['km_median_d'], 1)} / {r['recall_end']:.3f}")
        x = sp(1, s, "v1", 0)
        row.append(f"{f(x['km_median_d'], 1)} / {x['recall_end']:.3f}")
        recall.append(row)
    recall = table(["s", "B = 0", "60", "300", "1800", "7200", "v1, B = 0"], recall)

    ks = []
    for K in (2, 5):
        for s in (1.3, 2.0):
            row = [K, s]
            for B in (0, 300, 1800, 7200):
                row.append(f"{sp(K, s, 'ours', B)['km_F14']:.3f}")
            row.append(f"{sp(K, s, 'v1', 0)['km_F14']:.3f}")
            ks.append(row)
    ks = table(["K %", "s", "B = 0", "300", "1800", "7200", "v1"], ks)

    rule_rows = []
    for key, lab in (("main/ours/B300", "D86, primary"), ("main/ours_design_rule/B300", "Design: every investigated node out 7 d"), ("main/ours_skipdone/B300", "D86 + skip flagged nodes already diagnosed")):
        r13 = sens(key, 1.3)
        name = key.split("/")[1]
        rc = rules[name]
        rule_rows.append([lab, f"{r13['km_F14']:.3f}", f(r13["km_median_d"], 1), f"{r13['recall_end']:.3f}", f"{rc['accuracy']:.3f}", f"{rc['never_investigated']:,} of {rc['flagged_degraded']:,}", f"{rc['s_per_investigation']:.1f}", f"{rc['healthy_inv_per_day']:.1f}", vtxt("null", "ours" if name == "ours" else name)])
    rule_tab = table(["Rule, B = 300", "KM F(14 d), K 1%, s 1.3", "KM median d", "recall end", "accuracy, all K, s", "flagged, never investigated", "s per investigation", "healthy investigations/day", "null P(ever)"], rule_rows)

    val_rows = []
    for row, B, lab in (("ours", 0, "Rn(1), B = 0"), ("ours", 60, "ours, B = 60"), ("ours", 300, "ours, B = 300 (gate)"), ("ours", 1800, "ours, B = 1800"), ("ours", 7200, "ours, B = 7200"), ("ours_design_rule", 300, "design rule, B = 300"), ("ours_skipdone", 300, "skip-diagnosed, B = 300"), ("n_pool_200", 300, "n_pool 200, B = 300"), ("pool_redraw", 300, "pool per base, B = 300"), ("misspec", 300, "misspecified, B = 300"), ("v1", 0, "v1, B = 0 (empirical only)")):
        r = [lab]
        for k in ("null", "user_decoy", "user_decoy_heavy", "system_shock"):
            x = v(k, row, B)
            r.append(f"{x['P_ever']:.3f} ± {x['se']:.3f}")
        val_rows.append(r)
    val_tab = table(["Row", "null (200)", "user_decoy (80)", "heavy (80)", "shock (80)"], val_rows)

    base_rows = []
    for p in ("ours", "P-all", "P-IG", "P-random", "P-randnode", "P-sweep"):
        row = [p]
        for B in (60, 300, 1800):
            r = next(x for x in t["baselines"] if x["s"] == 1.3 and x["row"] == p and x["B"] == B)
            row.append(f"{r['km_F14']:.3f} / {r['accuracy']:.3f}")
        base_rows.append(row)
    base_tab = table(["Policy", "B = 60", "B = 300", "B = 1800"], base_rows)

    diag_rows = []
    for B in (60, 300, 1800, 7200):
        r = dg_row("all", B)
        diag_rows.append([B, f"{r['flagged_degraded']:,}", f"{r['accuracy']:.3f}", f"{r['never_investigated']:,}", f"{r['accuracy_investigated']:.3f}", f"{r['s_per_investigation']:.1f}", f"{r['s_per_uncut']:.1f}", f"{r['share_cut']:.3f}", f"{r['s_per_correct']:,.0f}", f"{r['healthy_inv_per_day']:.1f}"])
    diag_tab = table(["B", "flagged degraded", "accuracy", "never investigated", "accuracy, investigated", "s per investigation", "s per uncut", "share cut", "s per correct", "healthy investigations/day"], diag_rows)

    sens_rows = []
    labels = [("n_pool_200/ours/B300", "n_pool 200"), ("pool_redraw/ours/B300", "pool redrawn per base"), ("sigma_0.1/ours/B300", "σ = 0.1"), ("k_primary_2/ours/B300", "k_primary = 2"), ("pi0_0.1/ours/B300", "π0(h0) = 0.1"), ("pi0_0.9/ours/B300", "π0(h0) = 0.9"), ("misspec/ours/B300", "misspecified: k ± 50%, σ = 0.1"), ("tel_crossblock/ours/B300", "telemetry cross-block, both sides"), ("tel_design/ours/B300", "telemetry design matrix, both sides"), ("tel_transfer/ours/B300", "belief OOF, world cross-block")]
    for key, lab in [("main/ours/B300", "main")] + labels:
        a13, a20 = sens(key, 1.3), sens(key, 2.0)
        sens_rows.append([lab, f"{a13['km_F14']:.3f}", f"{a13['accuracy']:.3f}", f"{a20['km_F14']:.3f}", f"{a20['accuracy']:.3f}"])
    sens_tab = table(["Row (D86, B = 300, K 1%)", "F(14 d) s 1.3", "accuracy s 1.3", "F(14 d) s 2", "accuracy s 2"], sens_rows)

    tel_rows = []
    for lab, a1, a2 in (("OOF (main)", "main/ours_skipdone/B300", "main/ours/B7200"), ("cross-block, both sides", "tel_crossblock/ours_skipdone/B300", "tel_crossblock/ours/B7200"), ("design matrix, both sides", "tel_design/ours_skipdone/B300", "tel_design/ours/B7200"), ("belief OOF, world cross-block", "tel_transfer/ours_skipdone/B300", "tel_transfer/ours/B7200")):
        r = [lab]
        for key in (a1, a2):
            for s in (1.2, 1.3):
                x = sens(key, s)
                r.append(f"{x['accuracy']:.3f}")
        tel_rows.append(r)
    tel_tab = table(["Telemetry", "skip rule, B 300, s 1.2", "s 1.3", "D86, B 7200, s 1.2", "s 1.3"], tel_rows)

    n1000 = d87[str(1000)]
    gr = n1000["growth"]

    def growth(probe, bet, state, s=None):
        return next(x["growth"] for x in gr if x["probe"] == probe and x["bet"] == bet and x["state"] == state and (s is None or x["s"] == s))

    PROBES = ("cpu", "mbw", "cache", "mem")
    d87a = table(["Bet \\ probe", *PROBES], [[b, *[f"{growth(a, b, 'healthy'):.3f}" for a in PROBES]] for b in ("prior", "no_mem", "only_cpu", "only_cache", "only_mbw", "only_mem")])
    d87b = []
    for c in ("cpu", "cache", "mbw"):
        for s in S:
            d87b.append([c, s, *[f"{growth(a, 'no_mem', c, s):.2f}" for a in PROBES]])
    d87b = table(["Cause", "s", *PROBES], d87b)
    pool = n1000["pool"]
    d87c = table(["Probe, bet", "sd", "q95", "fixed pool"], [[f"{x['probe']}, {x['bet']}", f"{x['sd']:.3f}", f"{x['q95']:.3f}", f"{x['fixed_pool']:.3f}"] for x in pool if x["bet"] in ("prior", "no_mem") and x["sd"] > 0])

    conf_oof = table(["true \\ predicted", "h0", "cpu", "cache", "mbw", "mem"], [[n, *r] for n, r in zip(("h0", "cpu", "cache", "mbw", "mem"), f1["confusion"], strict=True)])
    cross = np.array(cb[0]["confusion"]) + np.array(cb[1]["confusion"])
    conf_cross = table(["true \\ predicted", "h0", "cpu", "cache", "mbw", "mem"], [[n, *r] for n, r in zip(("h0", "cpu", "cache", "mbw", "mem"), cross.tolist(), strict=True)])
    cb_tab = table(["Train → test", "macro-F1", "healthy", "cpu", "cache", "mbw", "mem"], [[f"{x['train']} → {x['test']}", f"{x['macro_f1']:.3f}", *[f"{x['per_class'][c]:.3f}" for c in ("healthy", "cpu", "cache", "mbw", "mem")]] for x in cb])
    fam = table(["Metrics used", "macro-F1"], [[x["family"], f"{x['macro_f1']:.3f}"] for x in dg["families"]] + [[f"all but the top {x['dropped_top']}", f"{x['macro_f1']:.3f}"] for x in dg["drop_top"]] + [["all, permuted labels", f"{dg['permuted_labels']['macro_f1']:.3f}"]])

    real = t["real"]
    rn = {(r["kind"], r["s"], r["B"]): r for r in real}
    mix = {f"{r['policy']}{r['B']}": r for r in t["probe_mix"]}
    node_share = 7200 / (980 * 86400)
    raw_f, raw_a = rows_of(a.run), rows_of(a.addon)
    key = lambda r: (r["scenario_id"], r["variant"], r["policy"], r["B"])  # noqa: E731
    fm = {key(r): r for r in raw_f}
    rep = [abs(r["max_logE_final"] - fm[key(r)]["max_logE_final"]) for r in raw_a if key(r) in fm]
    low = [
        d
        for r in raw_f
        if r["policy"] == "ours" and r["B"] == 300 and r["variant"] == "main" and r["kind"] == "node" and r["base"] != "real"
        for d in r["diagnosis"]
        if d.get("flagged") and d["investigated"] and d["pi_out"] < 0.95
    ]
    mem_only = sum(round(d["pi_out"], 3) == 0.571 for d in low)
    low_cpu = sum(d["cause_out"] == "cpu" for d in low)

    def null_flags(v, B):
        return {r["scenario_id"] for r in raw_a if r["kind"] == "null" and r["variant"] == v and r["policy"] == "ours" and r["B"] == B and r["metrics"]["ever_flagged"]}

    fx, rd, b0 = null_flags("main", 300), null_flags("pool_redraw", 300), null_flags("main", 0)

    md = f"""# Phase 5 report — Layer 3 v2: active probing as evidence and diagnosis

Generated by `scripts/make_phase5_report.py` at commit {commit}. Simulation, design probe model [A], development
seeds only. Seeds 20–39 and nulls 200–399 were not touched: the runner reads `index.parquet` only and raises on a
sealed id. Real numbers come from Phase 6.

## 1. Result

{gates}

The FDR guarantee holds with probes on. Gate 5b misses at {g5b['km_F14']:.3f} against 0.8, inside the 0.2–0.4 predicted
before the run. Gate 5c misses on accuracy, and most of that miss has one cause, below.

**The primary rule D86 starves the budget.** A node that is diagnosed but not cleared stays eligible, keeps its rank
at the top and is investigated again in every block. A few nodes use the whole budget and the rest are never reached:
{rules['ours']['never_investigated']:,} of {rules['ours']['flagged_degraded']:,} flagged degraded nodes were never
investigated after onset. Worse, a good diagnosis is overwritten: the node's output is its latest investigation (D90),
and that is often cut by budget after the 1-s mem probe, leaving the causes tied. {rules['ours']['share_output_pi_below_tau']:.0%}
of outputs are below 0.95, and those are right {rules['ours']['accuracy_output_pi_below_tau']:.1%} of the time, against
{rules['ours']['accuracy_output_pi_at_tau']:.1%} for outputs at or above 0.95. Of those {len(low):,} low outputs, {mem_only:,} sit at exactly π = 0.571, the
posterior after the mem probe alone, and {low_cpu:,} name cpu, the first cause in a tie.

{rule_tab}

The third row is not in the design or D86. I added it and committed it before any result, because the starvation
follows from the rule by logic; it is a sensitivity row and every gate above uses D86. Gate 5b at B = 300 is still
missed by all three rules: {alt['ours_design_rule']['km_F14']:.3f} (design rule, 95% {alt['ours_design_rule']['ci'][0]:.2f}–{alt['ours_design_rule']['ci'][1]:.2f}) and
{alt['ours_skipdone']['km_F14']:.3f} (skip rule, 95% {alt['ours_skipdone']['ci'][0]:.2f}–{alt['ours_skipdone']['ci'][1]:.2f}). The skip rule passes 5c at B = 300
(accuracy {rules['ours_skipdone']['accuracy']:.3f}, {rules['ours_skipdone']['s_per_investigation']:.1f} s).

**D91:** the smallest B in the grid that reaches 0.8 is **{t['d91_smallest_B']} node-s/day** (F(14 d) = {bm['ours7200']['km_F14']:.3f}).
Under D86 at that budget Gate 5c also passes ({d7200['accuracy']:.3f}, {d7200['s_per_investigation']:.1f} s), and null P(ever) is
{vtxt('null', 'ours', 7200)}. 7,200 node-s/day is {node_share:.4%} of the node time of 980 nodes.

## 2. Speed against budget

KM P(flag ≤ 14 d), K = 1%, random base, 20 seeds, 200 degraded nodes per cell, policy D86. v1 values come from Phase 4
run 20261010-094328-full and are "faster, empirical validity only".

{budget}

KM median delay in days / recall at trace end:

{recall}

Skip rule at s = 1.3: B = 60 {bm['ours_skipdone60']['km_F14']:.3f}, 300 {bm['ours_skipdone300']['km_F14']:.3f}, 1800 {bm['ours_skipdone1800']['km_F14']:.3f}, 7200 {bm['ours_skipdone7200']['km_F14']:.3f}.

More degraded nodes, KM F(14 d):

{ks}

Three things to note.
- **Probes can hurt at s = 1.1.** Recall at trace end falls from {sp(1, 1.1, 'ours', 0)['recall_end']:.3f} at B = 0 to {sp(1, 1.1, 'ours', 300)['recall_end']:.3f} at B = 300. The bets are built at s0 = 1.3, so p-values
  from a smaller shift land outside the bet and f falls towards 0.1. See the D87 table: cpu probe, cause cache at s = 1.1.
- **Rn(1) needs probes more as K grows.** At K = 5%, s = 1.3, B = 0 gives {sp(5, 1.3, 'ours', 0)['km_F14']:.3f} and recall {sp(5, 1.3, 'ours', 0)['recall_end']:.3f}, against v1's {sp(5, 1.3, 'v1', 0)['km_F14']:.3f} and {sp(5, 1.3, 'v1', 0)['recall_end']:.3f}.
  The likely reason is that degraded nodes raise their partition's denominator, though I did not test that. B = 7200 lifts it to {sp(5, 1.3, 'ours', 7200)['km_F14']:.3f}.
- **FDP stayed low everywhere.** Mean FDP at K = 1%, s = 1.3: {sp(1, 1.3, 'ours', 300)['fdp_mean']:.3f} at B = 300 and {sp(1, 1.3, 'ours', 7200)['fdp_mean']:.3f} at B = 7200. Co-allocated false nodes per
  scenario: {sp(1, 1.3, 'ours', 0)['false_coalloc']:.2f} at B = 0, {sp(1, 1.3, 'ours', 300)['false_coalloc']:.2f} at B = 300, {sp(1, 1.3, 'ours', 7200)['false_coalloc']:.2f} at B = 7200, all from small counts.

## 3. Validity

P(ever) ± SE, random base, seeds 0–19 and old null bases 0–199, α = 0.1. Every row passes α + 2 SE.

{val_tab}

The identity log E = log M + Σ log f holds exactly against the step-by-step jumper (`test_jumper_scale_equivariance`), and
B = 0 reproduces Phase 4c's Rn(1) on {t['b0_matches_phase4c_Rn1']['scenarios']} shared scenarios with a maximum absolute difference in final log M of
{t['b0_matches_phase4c_Rn1']['max_abs_diff']}.

**Fixed-pool risk (D88).** The fixed-pool null is {vtxt('null', 'ours')}, the pool-per-base null {vtxt('null', 'pool_redraw')}: {len(fx)} against {len(rd)} of 200. Both pass. But B = 0 flags {len(b0)} bases, and {len(fx - b0)} of the
{len(fx)} fixed-pool flags at B = 300 are new, so probes created them. That fits the fixed mbw pool being on the unlucky side of §5, though I did not trace the flags to a probe.
One pool is a real assumption, not a formality: Phase 6 must measure pools with care, and the sealed run should report both.

## 4. Baselines and diagnosis

K = 1%, s = 1.3: KM F(14 d) / accuracy among flagged degraded nodes. All use D86 except P-sweep.

{base_tab}

- **Layer 2 ranking is worth having:** at B = 1800, ours {next(x for x in t['baselines'] if x['s'] == 1.3 and x['row'] == 'ours' and x['B'] == 1800)['km_F14']:.3f} against P-randnode {next(x for x in t['baselines'] if x['s'] == 1.3 and x['row'] == 'P-randnode' and x['B'] == 1800)['km_F14']:.3f} and P-sweep {next(x for x in t['baselines'] if x['s'] == 1.3 and x['row'] == 'P-sweep' and x['B'] == 1800)['km_F14']:.3f}.
- **Probe choice adds little to detection.** P-random is level with ours at every budget. Information gain targets diagnosis, and evidence comes from
  whichever probe moves p; picking probes for evidence is not in this design.
- **P-IG's detection equals B = 0.** With unit cost it picks the near-perfect telemetry probe first, stops, and adds no evidence. That is an artefact of the
  out-of-fold matrix (§6).

Diagnosis under D86, all K and s, random base:

{diag_tab}

Cut investigations pull the Gate 5c mean down, so the uncut column is the fairer one. It is below 72.6 s at every B.

Probes used per investigation, K = 1%:

| Policy, B | cpu | mbw | cache | mem | telemetry | healthy share of seconds |
|---|---|---|---|---|---|---|
| D86, 300 | {mix['ours300']['used_cpu']:.2f} | {mix['ours300']['used_mbw']:.2f} | {mix['ours300']['used_cache']:.2f} | {mix['ours300']['used_mem']:.2f} | {mix['ours300']['used_tel']:.2f} | {mix['ours300']['healthy_share_of_seconds']:.2f} |
| D86, 7200 | {mix['ours7200']['used_cpu']:.2f} | {mix['ours7200']['used_mbw']:.2f} | {mix['ours7200']['used_cache']:.2f} | {mix['ours7200']['used_mem']:.2f} | {mix['ours7200']['used_tel']:.2f} | {mix['ours7200']['healthy_share_of_seconds']:.2f} |
| skip rule, 300 | {mix['ours_skipdone300']['used_cpu']:.2f} | {mix['ours_skipdone300']['used_mbw']:.2f} | {mix['ours_skipdone300']['used_cache']:.2f} | {mix['ours_skipdone300']['used_mem']:.2f} | {mix['ours_skipdone300']['used_tel']:.2f} | {mix['ours_skipdone300']['healthy_share_of_seconds']:.2f} |

The order is always mem, then cache: cache is the most informative 20-s probe because every cause moves it. Under the skip rule, B = 300, {mix['ours_skipdone300']['healthy_share_of_seconds']:.0%} of seconds
go to healthy nodes. That is the cost of searching.

## 5. D87 check, run before the grid

n_pool = 1000; run {a.d87}. No healthy growth is above 0 for any probe or bet, also at n_pool = 200, so the grid went ahead.

Mean log f on healthy nodes, fixed pool:

{d87a}

Mean log f per cause, bet after the mem probe has ruled out cause mem:

{d87b}

E[f | pool] under the healthy model, over 2,000 pools (bets that are not flat):

{d87c}

Means are 1.00–1.01. The fixed mbw pool is on the unlucky side: E[f | pool] is {next(x['fixed_pool'] for x in pool if x['probe'] == 'mbw' and x['bet'] == 'prior'):.2f} on the first-probe bet and {next(x['fixed_pool'] for x in pool if x['probe'] == 'mbw' and x['bet'] == 'only_mbw'):.2f} on the mbw-only bet, above its q95 of {next(x['q95'] for x in pool if x['probe'] == 'mbw' and x['bet'] == 'only_mbw'):.2f}.
Healthy log-growth is still negative on that pool. The pool seed was fixed before this check and was not changed.

The mem probe is worth noting: with cause mem in the mixture the first-probe bet loses {-growth('mem', 'prior', 'healthy'):.2f} nats on every node without cause mem, and scenarios never have that cause.

## 6. Gate 5d: macro-F1 {f1['macro_f1']:.4f}, and why not to trust it

Run {a.clf}: {f1['n']:,} pure 60-s windows, {clf['n_features']:,} features, 5 contiguous time folds per block with a 1-h purge. Macro-F1 by block: mono {clf['by_block']['mono']['macro_f1']:.4f}, multi {clf['by_block']['multi']['macro_f1']:.4f}.
Strict time order (first 60% to last 40%): {clf['forward']['macro_f1']:.4f}. With cpufreq and pagefail as extra classes: {clf['seven_class']['macro_f1']:.4f}. For context, Netti et al. report an RF F-score of 0.98 (arXiv 2007.14241, figure from D94, not checked here).

Out-of-fold counts, the a_tel likelihood as D94 specifies, with add-one smoothing [D]:

{conf_oof}

Zero errors in {f1['n']:,} windows contradicts F23 and F26, so I checked it (run {a.diag}).

{fam}

{cb_tab}

There is no label leak: permuted labels score {dg['permuted_labels']['macro_f1']:.3f}. The perfect score is a **program fingerprint**: top metrics are the injector's own activity
(branch instructions, page faults, anonymous pages). Each fault program runs identical code on one machine in one configuration, and pure windows drop the hard transitions. Change only single-core to
multi-core on the same node and macro-F1 falls to {cb[0]['macro_f1']:.2f} and {cb[1]['macro_f1']:.2f}, with cpu and cache going first and mbw and mem holding. Gate 5d is met as defined, but it does not show telemetry diagnosis works on another system.

Cross-block counts, both directions summed, used for the sensitivity rows:

{conf_cross}

Effect on diagnosis, accuracy among flagged degraded nodes, K = 1%. These rows were added before any grid result, and the skip-rule rows after seeing the main grid, because starvation hides any telemetry effect under D86 at B = 300.

{tel_tab}

The matrix changes accuracy little at B = 300 under the skip rule. The largest drop is the transfer row at B = 7200, s = 1.2: {sens('tel_transfer/ours/B7200', 1.2)['accuracy']:.3f} against {sens('main/ours/B7200', 1.2)['accuracy']:.3f}, which is the cost of trusting a perfect matrix on a world that is not.

**Prodigy secondary check: not run.** Its data files are HDF5 and neither `h5py` nor `tables` is installed. Rule 4 says ask first. The labels also show cpuoccupy, so D94's "healthy vs memleak only" is not quite right: `docs/data_cards/prodigy.md`.

## 7. Sensitivity (design §2, §4, §6)

{sens_tab}

- σ = 0.1 is the largest loss: at s = 1.3 the shift is {np.log(1.3) / 0.1:.1f} σ, so p-values rarely reach the small bins. Phase 6 must measure σ.
- Misspecification keeps validity (§3) and costs little at s ≥ 1.3, but recall at s = 1.1 falls to {sens('misspec/ours/B300', 1.1)['recall_end']:.3f}.
- k_primary = 2 helps most at s = 1.1, F(14 d) {sens('k_primary_2/ours/B300', 1.1)['km_F14']:.3f}.
- n_pool 200, pool per base and telemetry rows barely move Gate 5b, which is set by starvation under D86.
- D86 is sensitive to the prior: π0(h0) = 0.9 gives {sens('pi0_0.9/ours/B300', 1.3)['km_F14']:.3f}, because more healthy nodes get cleared and budget moves on.

## 8. Real base (unknown truth, no claim)

Null real base: {rn[('null', None, 0)]['distinct_flagged']:.0f} distinct nodes flagged at B = 0, {rn[('null', None, 300)]['distinct_flagged']:.0f} at B = 300. K = 1%, s = 1.3: {rn[('node', 1.3, 0)]['distinct_flagged']:.1f} at B = 0, {rn[('node', 1.3, 300)]['distinct_flagged']:.1f} at B = 300.
**This fall is built in, not a finding:** the simulation has no model of these nodes and draws healthy probes for every node without injection, so it clears them by assumption (D92). Which flags are real can only be tested on a cluster.

## 9. Corrections and disclosures

- **YAML bug.** `null` in `grid.validity_kinds` parsed as None, so the first full run ({a.run}) had no null bases. I quoted it, added a test, and ran the nulls in {a.addon}. Rows are merged on (scenario, variant, policy, B); the {len(rep)} repeated rows are identical (maximum difference in final log E {max(rep)}).
- **Add-on run was designed after seeing the first grid.** It holds the nulls, Gate 5a for the design and skip rules, skip rule at other budgets and telemetry rows on the skip rule and at B = 7200. All seeds here are development, and no gate or threshold changed.
- **Decided by me, per the plan or D90 unless noted:** Gate 5c read at B = 300 pooled over K and s, since the design names no budget; a probe that does not fit ends the block and carries the remainder; the node output counts only investigations after onset; battery order is cpu, mbw, cache, mem, telemetry; all probe state is taken at launch; add-one smoothing; RF settings fixed before any result (300 trees, balanced weights).
- **Runtime.** Full {full['n_runs']:,} runs in {full['wall_s'] / 60:.1f} min, add-on {add['n_runs']:,} runs in {add['wall_s'] / 60:.1f} min, 7 workers. The pilot's Layer 2 was 1.5 s per scenario against my 6 s estimate.
- **Failed runs left in place:** two classifier runs under `runs/phase5/` that stopped on code errors, before any result. Nothing was deleted.

## 10. Decisions needed before the sealed run (Phase 5F)

1. **Investigation rule.** D86 as written starves. I recommend adopting the skip rule: same validity, Gate 5c passes at B = 300, Gate 5b 0.625. Without this, Gate 5c is failed by design at any modest budget.
2. **Budget.** B = 300 cannot reach 0.8 under any rule tested. B = 7200 does, at {node_share:.4%} of node time. Gate 5b is your call; I have not changed it.
3. **Bets at low s.** Probes lower recall at s = 1.1. A mixture over s0 in the bet may fix it. This changes a formula, so it needs your go.
4. **Telemetry.** Keep D94's matrix, or use cross-block as the honest likelihood. I recommend cross-block: Gate 5d's 1.0 is a fingerprint, and the cost in simulation is small.
5. **Prodigy check.** Say if you want `h5py` installed.

## 11. Commands

```
uv run python scripts/download_data.py --config configs/phase0_data.yaml --download
uv run python scripts/phase5_d87_check.py --config configs/phase5.yaml                                # {a.d87}
uv run python -m hpcdiag.layer3.classifier --config configs/phase5.yaml                               # {a.clf}
uv run python scripts/phase5_d94_diag.py --config configs/phase5.yaml                                 # {a.diag}
uv run python -m hpcdiag.layer3.run_phase5 --config configs/phase5.yaml --mode pilot --workers 4      # 20261102-001332-pilot
uv run python -m hpcdiag.layer3.run_phase5 --config configs/phase5.yaml --mode full --workers 7       # {a.run}
uv run python -m hpcdiag.layer3.run_phase5 --config configs/phase5.yaml --mode addon --workers 7      # {a.addon}
uv run python scripts/phase5_tables.py --run {a.run} --addon {a.addon}
uv run python scripts/make_phase5_report.py --run {a.run} --addon {a.addon} --clf {a.clf} --diag {a.diag} --d87 {a.d87}
```
"""
    resolve("reports/phase_5.md").write_text(md)
    print("wrote reports/phase_5.md")


if __name__ == "__main__":
    main()
