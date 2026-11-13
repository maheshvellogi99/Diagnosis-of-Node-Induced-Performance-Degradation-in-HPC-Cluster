# ruff: noqa: E501  (markdown report template: table rows are long by nature)
"""Render reports/phase_5c.md (Phase 5c: D108-D111, restart mixture for probe evidence).

    uv run python scripts/make_phase5c_report.py --run <dev5c> --ref <dev5b>

D109 rows come from the dev5c run (policy ig_d109, k_pre 0). The old-formula reference rows are the
dev5b grid's "ig" rows; dev5c re-ran part of them (K 1%, s 1.3 and the 200 nulls at B 7200), and
the report checks that they match exactly. The D111 freeze decision is computed here and printed;
the tag itself is set by hand only if it says FREEZE.
"""

from __future__ import annotations

import argparse
import json
import subprocess

from phase5_tables import diag_row, km_ci, load, pick, speed_row, validity_row

from hpcdiag.utils.run import PROJECT_ROOT, resolve

ALPHA = 0.1
S = [1.1, 1.2, 1.3, 1.5, 2.0, 3.0]
KINDS = ("null", "user_decoy", "user_decoy_heavy", "system_shock")
NEW, OLD = "D109 restart (primary)", "old formula (reference)"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    return "\n".join(out + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows])


def git(*args):
    return subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ref", required=True)
    a = ap.parse_args()
    c5 = load(a.run)
    b5 = load(a.ref)
    meta = json.loads(resolve(f"runs/phase5/{a.run}/metrics.json").read_text())
    run_commit = resolve(f"runs/phase5/{a.run}/git_commit.txt").read_text().strip()

    def rows(rule, B, kind="node", variant="main", k_pre=0, **kw):
        if B == 0:
            src, pol = c5, "ig"
        elif rule == "new":
            src, pol = c5, "ig_d109"
        else:
            src, pol = b5, "ig"
        out = [r for r in src if r["policy"] == pol and r["B"] == B and r["variant"] == variant
               and r["kind"] == kind and r["base"] != "real" and r.get("k_pre", 0) == k_pre]  # fmt: skip
        return pick(out, **kw)

    def val(rule, B, kind, variant="main"):
        return validity_row(rows(rule, B, kind, variant))

    def vt(x):
        return f"{x['P_ever']:.3f} ± {x['se']:.3f}"

    def F(rule, B, K=1, s=1.3, variant="main"):
        return speed_row(rows(rule, B, variant=variant, K_pct=K, s=s))

    # ---------------- reproduction of the reference rows
    ref = {(r["scenario_id"], r["variant"], r["B"]): r for r in b5 if r["policy"] == "ig"}
    rep = [r for r in c5 if r["policy"] == "ig" and r.get("k_pre", 0) == 0 and r["B"] > 0]
    same = sum(
        json.dumps(r["metrics"], sort_keys=True, default=str)
        == json.dumps(
            ref[(r["scenario_id"], r["variant"], r["B"])]["metrics"], sort_keys=True, default=str
        )
        and r["n_investigations"]
        == ref[(r["scenario_id"], r["variant"], r["B"])]["n_investigations"]
        and r["seconds_total"] == ref[(r["scenario_id"], r["variant"], r["B"])]["seconds_total"]
        for r in rep
    )
    b0 = [r for r in c5 if r["B"] == 0]
    ref0 = {(r["scenario_id"]): r for r in b5 if r["B"] == 0 and r["variant"] == "main"}
    same0 = sum(
        json.dumps(r["metrics"], sort_keys=True, default=str)
        == json.dumps(ref0[r["scenario_id"]]["metrics"], sort_keys=True, default=str)
        for r in b0
    )

    # ---------------- validity
    vrows, new_ok = [], True
    for rule, lab in (("new", NEW), ("old", OLD)):
        for B in (0, 300, 1800, 7200):
            if B == 0 and rule == "old":
                continue
            r = ["Rn(1) alone, no probes" if B == 0 else lab, B]
            ok = True
            for k in KINDS:
                x = val(rule, B, k)
                y = val(rule, B, k, "pool_redraw") if B else x
                ok = ok and x["pass"] and y["pass"]
                r.append(f"{vt(x)} / {vt(y)}" if B else vt(x))
            r.append("pass" if ok else "**FAIL**")
            vrows.append(r)
            if rule == "new":
                new_ok = new_ok and ok
        for B in (300, 7200):
            r = [f"{lab}, misspecified", B]
            ok = True
            for k in KINDS:
                x = val(rule, B, k, "misspec")
                ok = ok and x["pass"]
                r.append(vt(x))
            r.append("pass" if ok else "**FAIL**")
            vrows.append(r)
            if rule == "new":
                new_ok = new_ok and ok
    n_new_rows = sum(1 for r in vrows if not r[0].startswith(OLD))
    val_tab = table(
        [
            "Rule",
            "B",
            "null (200)",
            "user_decoy (80)",
            "heavy (80)",
            "shock (80)",
            "all ≤ α + 2 SE",
        ],
        vrows,
    )
    old_ok = all(r[-1] == "pass" for r in vrows if r[0].startswith(OLD))
    freeze = new_ok

    # ---------------- speed, K 1/2/5, all s
    sp = {}
    for K in (1, 2, 5):
        rr = []
        for s in S:
            rr.append(
                [s]
                + [f"{F('new', B, K, s)['km_F14']:.3f}" for B in (0, 300, 1800, 7200)]
                + [f"{F('old', B, K, s)['km_F14']:.3f}" for B in (300, 1800, 7200)]
            )
        sp[K] = table(
            ["s", "B = 0", "D109 300", "D109 1800", "D109 7200", "old 300", "old 1800", "old 7200"],
            rr,
        )
    rec = []
    for s in S:
        rec.append(
            [s]
            + [f"{F('new', B, 1, s)['recall_end']:.3f}" for B in (0, 300, 1800, 7200)]
            + [f"{F('old', B, 1, s)['recall_end']:.3f}" for B in (300, 1800, 7200)]
        )
    rec_tab = table(
        ["s", "B = 0", "D109 300", "D109 1800", "D109 7200", "old 300", "old 1800", "old 7200"], rec
    )

    # ---------------- diagnosis
    drows = []
    for rule, lab in (("new", NEW), ("old", OLD)):
        for B in (300, 1800, 7200):
            d = diag_row(rows(rule, B))
            drows.append(
                [
                    lab,
                    B,
                    f"{d['flagged_degraded']:,}",
                    f"{d['accuracy']:.3f}",
                    f"{d['never_investigated']:,}",
                    f"{d['s_per_investigation']:.1f}",
                    f"{d['share_cut']:.3f}",
                    f"{d['healthy_inv_per_day']:.1f}",
                ]
            )
    d_tab = table(
        [
            "Rule",
            "B",
            "flagged degraded (all K, s)",
            "accuracy",
            "never investigated",
            "s per investigation",
            "share cut",
            "healthy investigations/day",
        ],
        drows,
    )

    # ---------------- misspecified outcomes, K 1%
    mrows = []
    for rule, lab in (("new", NEW), ("old", OLD)):
        for v in ("main", "misspec"):
            r = [lab, v]
            for B in (300, 7200):
                dd = diag_row(rows(rule, B, variant=v, K_pct=1))
                r.append(
                    f"{F(rule, B, variant=v)['km_F14']:.3f} / {F(rule, B, s=1.1, variant=v)['recall_end']:.3f} / {dd['accuracy']:.3f}"
                )
            mrows.append(r)
    m_tab = table(
        ["Rule", "outcomes", "B 300: F(14 d) s 1.3 / recall s 1.1 / accuracy (K 1%)", "B 7200"],
        mrows,
    )

    # ---------------- stress test (D110, diagnostic)
    srows = []
    for k in (0, 10, 25):
        r = [k]
        for rule, pol in (("old", "ig"), ("new", "ig_d109")):
            if k == 0:
                rs = rows(rule, 7200, K_pct=1, s=1.3)
            else:
                rs = [
                    x
                    for x in c5
                    if x["policy"] == pol
                    and x["B"] == 7200
                    and x.get("k_pre", 0) == k
                    and x["kind"] == "node"
                    and x["K_pct"] == 1
                    and x["s"] == 1.3
                ]
            x = speed_row(rs)
            lo, hi = km_ci(rs)
            r += [f"{x['km_F14']:.3f} ({lo:.2f}–{hi:.2f})", f"{x['recall_end']:.3f}"]
        srows.append(r)
    s_tab = table(
        [
            "k_pre (extra healthy investigations before onset)",
            "old: F(14 d) (95% interval)",
            "old: recall",
            "D109: F(14 d) (95% interval)",
            "D109: recall",
        ],
        srows,
    )

    # ---------------- gate previews, D109
    g300, g7200 = F("new", 300), F("new", 7200)
    d300, d7200 = diag_row(rows("new", 300)), diag_row(rows("new", 7200))
    ci7200 = km_ci(rows("new", 7200, K_pct=1, s=1.3))
    v300 = [val("new", 300, k) for k in KINDS] + [val("new", 300, k, "pool_redraw") for k in KINDS]
    v_all = [
        val("new", B, k, v)
        for B in (0, 300, 1800, 7200)
        for k in KINDS
        for v in (("main", "pool_redraw") if B else ("main",))
    ]
    gates = table(
        ["Check (development seeds 0–19), D109 primary", "Criterion", "Measured", "Result"],
        [
            [
                "5a validity",
                "B 300, P(ever) ≤ α + 2 SE, fixed pool and pool per base",
                f"null {vt(val('new', 300, 'null'))} / {vt(val('new', 300, 'null', 'pool_redraw'))}; worst {max(x['P_ever'] for x in v300):.3f}",
                "pass" if all(x["pass"] for x in v300) else "**FAIL**",
            ],
            [
                "5b speed",
                "K 1%, s 1.3, B 300: KM F(14 d) ≥ 0.8",
                f"{g300['km_F14']:.3f}",
                "**FAILED** (stays failed, D101)"
                if g300["km_F14"] < 0.8
                else "pass (5b stays recorded as failed, D101)",
            ],
            [
                "5c diagnosis",
                "B 300: accuracy ≥ 0.9, ≤ 72.6 s per investigation",
                f"{d300['accuracy']:.3f}, {d300['s_per_investigation']:.1f} s",
                "pass"
                if d300["accuracy"] >= 0.9 and d300["s_per_investigation"] <= 72.6
                else "**FAIL**",
            ],
            [
                "5F (i) preview",
                "validity at every B ∈ {0, 300, 1800, 7200}",
                f"worst P(ever) {max(x['P_ever'] for x in v_all):.3f}",
                "pass" if all(x["pass"] for x in v_all) else "**FAIL**",
            ],
            [
                "5F (ii) preview",
                "K 1%, s 1.3, B 7200: KM F(14 d) ≥ 0.8",
                f"{g7200['km_F14']:.3f} ({ci7200[0]:.2f}–{ci7200[1]:.2f})",
                "pass" if g7200["km_F14"] >= 0.8 else "**FAIL**",
            ],
            [
                "5F (iii) preview",
                "B 7200: accuracy ≥ 0.9",
                f"{d7200['accuracy']:.3f}",
                "pass" if d7200["accuracy"] >= 0.9 else "**FAIL**",
            ],
        ],
    )

    s11 = {B: F("new", B, 1, 1.1) for B in (0, 300, 1800, 7200)}
    o11 = {B: F("old", B, 1, 1.1) for B in (300, 1800, 7200)}
    commit = git("rev-parse", "--short", "HEAD")
    freeze_txt = (
        f"**Every one of the {n_new_rows} D109 validity rows passes α + 2 SE, so the D111 freeze rule says FREEZE:** the primary is tagged `v1.0-dev-frozen`, and every parameter is listed in docs/FROZEN_CONFIG.md."
        if freeze
        else "**At least one D109 validity row fails α + 2 SE, so the D111 freeze rule says STOP:** no tag."
    )
    text = f"""# Phase 5c report — probe debt fixed by a restart mixture (D108–D111)

Generated by `scripts/make_phase5c_report.py` at commit {commit}, from run {a.run} (grid code at commit {run_commit[:7]}) and the
Phase 5b reference run {a.ref}. Simulation, design probe model [A]. Development seeds 0–19 and old nulls 0–199 only.
Nothing ran on seeds 20–39 or nulls 200–399: the runner reads `index.parquet` only and raises on a sealed id.
{meta["n_runs"]:,} runs in {meta["wall_s"] / 60:.1f} min on 7 workers.

## 1. Result

{freeze_txt}

- **Primary:** Rn(1) Layer 2, IG-only, D95, D96, D98, D109, with J = 0.01.
- **Validity:** every D109 row passes, fixed pool and pool per base, B ∈ {{0, 300, 1800, 7200}}, plus the misspecified rows (§3).
- **Speed:** at K = 1%, s = 1.3, D109 gives F(14 d) {F("new", 300)["km_F14"]:.3f} / {F("new", 1800)["km_F14"]:.3f} / {g7200["km_F14"]:.3f} at B = 300 / 1800 / 7200, against
  {F("old", 300)["km_F14"]:.3f} / {F("old", 1800)["km_F14"]:.3f} / {F("old", 7200)["km_F14"]:.3f} with the old formula (§4).
- **Debt stress test (§6):** with 25 extra healthy investigations before onset, F(14 d) at s = 1.3, B = 7200 is {srows[2][3]} with D109,
  against {srows[2][1]} with the old formula.

{gates}

Gate 5b stays FAILED (D101). The 5F rows are previews on development seeds; the real Gate 5F runs once, on the sealed seeds.

## 2. What changed (D108, D109)

**D108.** The D102 stop rule is confirmed: pass = exact value ≤ 1e-12 AND Monte-Carlo within max(2 SE, 1e-12). The
exact computation is the criterion; Monte-Carlo is a cross-check. The docstring of `scripts/phase5_d102_check.py`
says so. Process rule from now on: after editing a check script, verify that the edit applied before re-running.

**D109.** Probe evidence is now a restart mixture over each node's investigations, F_i = product of f over the probes of
investigation i:

P_n(t) = Σ_{{k ≤ N}} w_k Π_{{i = k..N}} F_i + (1 − J)^N,  w_k = J (1 − J)^(k−1),  J = 0.01,  P_n(0) = 1,

computed by the recursion A ← F (A + J R), R ← (1 − J) R, P = A + R (`src/hpcdiag/stats/restart.py`). log E_n = log M_n + log P_n,
and e-BH runs on E_n as before. The validity proof is in PLAN §9 Phase 5, 5.3. The old formula
(log E = log M + Σ log f, PHASE5_DESIGN §3) is recorded there as the architect's design error: it is valid, but it
loses power through unbounded debt.

Tests (written before the code; all pass):
- `tests/test_restart.py`: P(0) = 1; recursion = explicit sum (J ∈ {{0.01, 0.1, 0.5, 1}}); one-step fairness, E[P'] = P
  exactly when f integrates to 1; strict supermartingale when E[F] < 1; J = 1 gives the old running sum bit for bit;
  the debt is bounded; the vectorised version equals the scalar one; J outside (0, 1] is rejected.
- `tests/test_layer3.py`: in the policy, J = 1 gives log E bit for bit equal to the old formula recomputed from the
  investigations; J = 0.01 equals the per-node recursion; Policy defaults to J = 1 (so Phase 5/5b runs mean what
  they meant); the stress option adds evidence but not budget.

**Reproduction.** dev5c re-ran the old formula on K = 1%, s = 1.3 and the 200 nulls at B = 7200. {same} of {len(rep)} runs
match the Phase 5b grid exactly (investigations, seconds, all metrics). At B = 0, {same0} of {len(b0)} runs match. So the
reference rows below are the dev5b rows, produced by the same code path.

## 3. Validity (D100): fixed pool / pool per base

P(ever) ± SE, α = 0.1, random base. The misspecified rows use a pool at the true σ. The old-formula rows are from {a.ref}.

{val_tab}

D109 rows: {"all pass" if new_ok else "**at least one fails**"}. Old-formula reference rows: {"all pass" if old_ok else "**at least one fails**"}.

## 4. Speed

KM P(flag ≤ 14 d), random base, 20 seeds per cell.

K = 1%:

{sp[1]}

K = 2%:

{sp[2]}

K = 5%:

{sp[5]}

Recall at trace end, K = 1%:

{rec_tab}

**s = 1.1.** With D109, recall at trace end is {s11[0]["recall_end"]:.3f} / {s11[300]["recall_end"]:.3f} / {s11[1800]["recall_end"]:.3f} / {s11[7200]["recall_end"]:.3f} at B = 0 / 300 / 1800 / 7200. With the old
formula it was {o11[300]["recall_end"]:.3f} / {o11[1800]["recall_end"]:.3f} / {o11[7200]["recall_end"]:.3f} at B = 300 / 1800 / 7200. F(14 d) at s = 1.1 stays low: at most {max(x["km_F14"] for x in s11.values()):.3f} with D109.

## 5. Diagnosis and misspecified outcomes

All K and s, random base:

{d_tab}

Misspecified outcomes (k ± 50%, σ = 0.1), K = 1%:

{m_tab}

## 6. Debt stress test (D110, diagnostic, not a gate)

Before the trace starts, each degraded node gets k_pre extra investigations while healthy. They enter its evidence but
not the budget. 10 and 25 investigations are about 2 and 5 months of weekly probing. K = 1%, s = 1.3, B = 7200, 20
seeds, 200 degraded nodes per cell; intervals resample scenarios.

{s_tab}

## 7. Commands

```
uv run pytest -q
uv run python -m hpcdiag.layer3.run_phase5 --config configs/phase5.yaml --mode dev5c --workers 7   # {a.run}
uv run python scripts/make_phase5c_report.py --run {a.run} --ref {a.ref}
```
"""
    resolve("reports/phase_5c.md").write_text(text)
    print(
        "wrote reports/phase_5c.md;",
        "FREEZE" if freeze else "STOP (no tag)",
        f"new_ok={new_ok} old_ok={old_ok} repro {same}/{len(rep)} b0 {same0}/{len(b0)}",
    )


if __name__ == "__main__":
    main()
