# ruff: noqa: E501  (markdown report template: table rows are long by nature)
"""Append the D105-D107 addendum to reports/phase_5b.md from the diagnosis run and the D102 runs.

    uv run python scripts/make_phase5b_addendum.py --diag <d105 run> --grid <dev5b run> \
        --d102 <run1> <run2> <run3> <run4>

Replaces everything from the addendum heading to the end of the file, so it can be re-run.
Run it after scripts/make_phase5b_report.py, which rewrites the file.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from hpcdiag.utils.run import resolve

HEAD = "## 10. Addendum (D105–D107)"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    return "\n".join(out + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows])


def commit_of(rid: str) -> str:
    return resolve(f"runs/phase5/{rid}/git_commit.txt").read_text().strip()[:7]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--d102", nargs=4, required=True)
    a = ap.parse_args()
    D = resolve(f"runs/phase5/{a.diag}")
    d = pd.read_parquet(D / "per_node.parquet")
    iv = pd.read_parquet(D / "per_investigation.parquet")
    runs = pd.read_csv(D / "runs.csv")

    # reproduction against the grid
    grid = {}
    with open(resolve(f"runs/phase5/{a.grid}/results.jsonl")) as fh:
        for line in fh:
            r = json.loads(line)
            if r["variant"] == "main":
                grid[(r["scenario_id"], r["policy"], r["B"])] = (
                    r["n_investigations"],
                    r["seconds_total"],
                )
    chk = runs[~runs["case"].str.contains("excl0")]
    keys = [(r.scenario_id, r.policy, int(r.B)) for r in chk.itertuples()]
    n_chk = sum(k in grid for k in keys)
    n_bad = sum(
        k in grid and grid[k] != (r.n_investigations, r.seconds_total)
        for k, r in zip(keys, chk.itertuples(), strict=True)
    )

    def agg(x):
        return pd.Series({
            "flag14": x["flag14"].mean(), "ever": x["flag_ever"].mean(),
            "cf14": x["cf_no_debt_flag14"].mean(), "cfever": x["cf_no_debt_flag_ever"].mean(),
            "pre": x["n_inv_pre"].mean(), "post14": x["n_inv_post_14d"].mean(),
            "clr_post": x["n_cleared_post"].mean(), "any_clr": (x["n_cleared_post"] > 0).mean(),
            "debt": x["debt_at_onset"].mean(), "ev14": x["probe_sum_post_14d"].mean(),
            "excl14": x["blocks_excluded_14d"].mean(), "excl_on": x["excluded_at_onset"].mean(),
            "logM14": x["logM_max_14d"].mean(), "logE14": x["logE_max_14d"].mean(),
        })  # fmt: skip

    g = d.groupby(["case", "s", "B"]).apply(agg, include_groups=False)

    def G(c, s, B):
        return g.loc[(c, s, B)]

    names = {
        "ig": "IG-only + D95",
        "ig_excl0": "IG-only, exclusion 0 (diagnostic)",
        "design_rule": "IG + design rule",
    }

    def row(c, s, B):
        x = G(c, s, B)
        return [names[c], s, B, f"{x.flag14:.3f}", f"{x.ever:.3f}", f"{x.pre:.2f}", f"{x.post14:.2f}",
                f"{x.clr_post:.2f} ({x.any_clr:.0%})", f"{x.debt:+.2f}", f"{x.ev14:+.2f}",
                f"{x.excl14:.1f} of 56 ({x.excl_on:.0%})", f"{x.logM14:+.2f}", f"{x.logE14:+.2f}",
                f"{x.cf14:.3f} / {x.cfever:.3f}"]  # fmt: skip

    head = ["Rule", "s", "B", "F(14 d)", "recall", "inv. before onset", "inv. in 14 d after",
            "cleared while degraded (nodes ≥ 1)", "probe log-sum at onset", "probe log-sum, 14 d after",
            "blocks excluded in 14 d (at onset)", "max log M, 14 d", "max log E, 14 d",
            "no-debt F(14 d) / recall"]  # fmt: skip
    main_tab = table(head, [row("ig", 1.1, 0), row("ig", 1.1, 1800), row("ig", 1.1, 7200),
                            row("ig_excl0", 1.1, 1800), row("ig_excl0", 1.1, 7200),
                            row("ig", 1.3, 300), row("ig", 1.3, 7200),
                            row("design_rule", 1.3, 300), row("design_rule", 1.3, 7200)])  # fmt: skip

    # log E before/after each investigation
    m = iv.merge(
        d[["scenario_id", "case", "B", "node", "s"]], on=["scenario_id", "case", "B", "node"]
    )
    lrows = []
    for (c, s, B), x in m.groupby(["case", "s", "B"]):
        for lab, y in (
            ("before onset", x[~x["after_onset"]]),
            ("after onset", x[x["after_onset"]]),
        ):
            if len(y):
                lrows.append([names[c], s, B, lab, f"{len(y):,}", f"{y['logE_before'].mean():+.2f}",
                              f"{y['logE_after'].mean():+.2f}", f"{y['dlog'].mean():+.3f}",
                              f"{y['cleared'].mean():.3f}"])  # fmt: skip
    le_tab = table(["Rule", "s", "B", "investigations", "n", "mean log E before", "mean log E after",
                    "mean probe log-sum", "share cleared"], lrows)  # fmt: skip

    # debt against onset time
    trows = []
    for c, s, B in (
        ("ig", 1.1, 1800),
        ("ig", 1.1, 7200),
        ("ig", 1.3, 7200),
        ("design_rule", 1.3, 7200),
    ):
        x = d[(d["case"] == c) & (d["s"] == s) & (d["B"] == B)].copy()
        slope = np.polyfit(x["onset"] / 4, x["debt_at_onset"], 1)[0]
        x["q"] = pd.qcut(x["onset"], 3, labels=["early", "middle", "late"])
        t = x.groupby("q", observed=True).agg(
            debt=("debt_at_onset", "mean"), f14=("flag14", "mean"), ever=("flag_ever", "mean")
        )
        trows.append(
            [names[c], s, B, f"{slope:+.3f}"]
            + [
                f"{t.loc[q, 'debt']:+.1f} / {t.loc[q, 'f14']:.3f} / {t.loc[q, 'ever']:.3f}"
                for q in ("early", "middle", "late")
            ]
        )
    o_days = (d["onset"].min() / 4, d["onset"].max() / 4)
    t_tab = table(
        [
            "Rule",
            "s",
            "B",
            "debt slope, nats per day before onset",
            "early onsets: debt / F(14 d) / recall",
            "middle",
            "late",
        ],
        trows,
    )

    i13 = G("ig", 1.3, 7200)
    a11, b11 = G("ig", 1.1, 1800), G("ig", 1.1, 7200)
    e18, e72 = G("ig_excl0", 1.1, 1800), G("ig_excl0", 1.1, 7200)
    dr3, dr7 = G("design_rule", 1.3, 300), G("design_rule", 1.3, 7200)
    pre_ig = iv[(iv["case"] == "ig") & (~iv["after_onset"])]
    per_inv_debt = pre_ig["dlog"].mean()
    post11 = {
        B: m[(m["case"] == "ig") & (m["s"] == 1.1) & (m["B"] == B) & m["after_onset"]][
            "dlog"
        ].mean()
        for B in (1800, 7200)
    }
    post13 = m[(m["case"] == "ig") & (m["s"] == 1.3) & (m["B"] == 7200) & m["after_onset"]][
        "dlog"
    ].mean()

    def tercile(c, s, B, q):
        x = d[(d["case"] == c) & (d["s"] == s) & (d["B"] == B)].copy()
        x["q"] = pd.qcut(x["onset"], 3, labels=["early", "middle", "late"])
        return x[x["q"] == q]

    late11, late13 = tercile("ig", 1.1, 7200, "late"), tercile("ig", 1.3, 7200, "late")
    dr_post = m[(m["case"] == "design_rule") & (m["B"] == 7200) & m["after_onset"]]
    r1 = json.loads(resolve(f"runs/phase5/{a.d102[0]}/metrics.json").read_text())["positive"]
    ex1 = max(p["exact_log_f"] for p in r1)
    slope72 = np.polyfit(
        *(lambda x: (x["onset"] / 4, x["debt_at_onset"]))(
            d[(d["case"] == "ig") & (d["s"] == 1.3) & (d["B"] == 7200)]
        ),
        1,
    )[0]

    # D107: the four D102 runs
    drows = []
    for rid in a.d102:
        mm = json.loads(resolve(f"runs/phase5/{rid}/metrics.json").read_text())
        pos = mm["positive"]
        mc = [p["mean_log_f"] for p in pos if p["bet"] == "max over all weights"]
        what = (
            f"{len(mc)} max-over-weights rows (Monte-Carlo {min(mc):.1e} to {max(mc):.1e})"
            if mc
            else "no max-over-weights rows"
        )
        nm = [p["exact_log_f"] for p in pos if p["bet"] == "no_mem"]
        mx = [p["exact_log_f"] for p in pos if p["bet"] == "max over all weights"]
        ex_txt = f", exact ≤ {max(mx):.1e}" if mx else ""
        drows.append(
            [
                rid,
                commit_of(rid),
                len(pos),
                f"{what}{ex_txt}; {len(nm)} flat no_mem rows (log f = {max(nm):.1e})"
                if pos
                else "none",
            ]
        )
    same12 = (
        resolve(f"runs/phase5/{a.d102[0]}/metrics.json").read_bytes()
        == resolve(f"runs/phase5/{a.d102[1]}/metrics.json").read_bytes()
    )
    d_tab = table(["Run", "commit", "marginal_positive", "positive rows"], drows)
    head_commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()

    text = f"""{HEAD}

Generated by `scripts/make_phase5b_addendum.py` at commit {head_commit}, from run {a.diag}. Diagnosis only (D105):
the design was not changed. The diagnosis re-ran K = 1%, random base, seeds 0–19 with the grid's seeds and code
path. {n_chk} of its runs have a grid twin, and {n_bad} differ in investigations or seconds. Each cell has 200 degraded nodes.
Definitions:
- "Probe log-sum at onset" ("debt") is Σ log f over the node's probes before onset.
- "Cleared while degraded" counts investigations after onset that ended with π(h0) ≥ 0.95.
- Exclusion counts blocks of 6 h; 14 days is 56 blocks.
- "No-debt" is a post-hoc counterfactual on the same run: each degraded node's probe evidence from before onset is
  removed, and e-BH is recomputed.
- "Exclusion 0" is a diagnostic re-run with the 7-day exclusion after clearing set to 0.

Neither counterfactual is a proposal.

### 10.1 s = 1.1: why recall falls at B = 7200 (D105)

{main_tab}

log E before and after each investigation (mean over investigations):

{le_tab}

Debt against onset time (onsets {o_days[0]:.1f}–{o_days[1]:.1f} days after the trace start; terciles of onset):

{t_tab}

**Finding.** The drop has two causes, which act together:
- **Probe debt before onset.** While a node is still healthy, every investigation multiplies its E by f < 1. The mean
  probe log-sum per investigation before onset is {per_inv_debt:+.2f} nats (IG-only, all cells above). A cleared node becomes eligible again
  after 7 days, so at B = 7200 every node is re-probed about weekly. The debt grows linearly with the healthy time
  ({slope72:+.3f} nats/day at B = 7200, s = 1.3 cells; terciles below). At onset it averages {b11.debt:+.2f} nats at B = 7200, against
  {a11.debt:+.2f} at B = 1800. The e-BH threshold is about 9.19 nats.
- **Weak evidence at s = 1.1.** After onset an investigation adds only {post11[7200]:+.3f} nats on average at B = 7200
  ({post11[1800]:+.3f} at B = 1800), so the debt is
  never repaid. Max log E in the 14 days after onset averages {b11.logE14:+.2f} at B = 7200, against {b11.logM14:+.2f}
  for Layer 2 alone.
- **Wrong clearings.** After onset, {b11.any_clr:.0%} of degraded nodes at B = 7200 are cleared at least once while
  degraded ({a11.any_clr:.0%} at B = 1800). Each clearing excludes the node for 7 days, which delays new evidence:
  {b11.excl14:.1f} of the first 56 blocks are excluded at B = 7200, against {a11.excl14:.1f} at B = 1800.

**Does the 7-day exclusion explain the drop?** Partly:
- Exclusion 0 removes the gap between the budgets: recall is {e18.ever:.3f} at B = 1800 and {e72.ever:.3f} at B = 7200
  (actual {a11.ever:.3f} and {b11.ever:.3f}).
- Removing only the debt also removes it, and reverses it: {a11.cfever:.3f} and {b11.cfever:.3f}.
- The debt is the larger factor. At B = 7200, recall is {b11.cfever:.3f} with the debt removed, against {e72.ever:.3f}
  with exclusion 0. Late onsets, which carry the most debt ({late11["debt_at_onset"].mean():+.1f} nats), reach recall
  {late11["flag_ever"].mean():.3f} at B = 7200.
- The exclusion matters because it stops post-onset evidence from repaying the debt.

### 10.2 IG + design rule, B = 7200, F(14 d) {dr7.flag14:.3f} (D106)

**Not a bug.** The run reproduces the grid exactly, and the code does what PLAN 5.3 and 5.5 say. The probes work after
onset: the mean probe log-sum per investigation is {dr_post["dlog"].mean():+.2f} nats, and the share cleared is
{dr_post["cleared"].mean():.3f} (table above).

The loss has the same mechanism as above, made worse by the design rule:
- **Debt.** At onset it averages {dr7.debt:+.2f} nats at B = 7200, against {dr3.debt:+.2f} at B = 300.
- **Exclusion.** The design rule excludes every investigated node for 7 days, diagnosed ones included. So a degraded
  node gets only {dr7.post14:.2f} investigations in its first 14 days, and is excluded for {dr7.excl14:.1f} of 56
  blocks; {dr7.excl_on:.0%} are excluded at onset.
- **Effect.** One or two investigations cannot repay the debt within 14 days. With the debt removed, F(14 d) would be
  {dr7.cf14:.3f}.

**Shared code.** The probe log-sum (`run_policy`, `lp`) is shared with the primary configuration. It is not a bug,
though: it is the PLAN 5.3 formula log E = log M + Σ log f over all probes ever run. The primary keeps F(14 d) at
s = 1.3, B = 7200 at {i13.flag14:.3f}, even for the late onsets ({late13["flag14"].mean():.3f}, debt
{late13["debt_at_onset"].mean():+.1f} nats): after onset it gets
{i13.post14:.2f} investigations in 14 days, at {post13:+.2f} nats each, and it does not exclude diagnosed nodes.

### 10.3 A fix seems needed: proposal only, not implemented (D105)

The debt is a property of the design, not of these traces, and it grows with deployment length:
- **Measured.** About {slope72:+.2f} nats per healthy day at B = 7200, linear over onsets up to {o_days[1]:.0f} days.
- **Extrapolated, not measured.** Over 180 healthy days the debt would reach about {180 * slope72:+.0f} nats. That is
  more than the {i13.ev14:+.1f} nats a node at s = 1.3 gains in its first 14 days after onset at B = 7200 (measured).
  So in a long deployment, the B = 7200 headline would lose power even at s = 1.3. Within these traces (onsets ≤
  {o_days[1]:.0f} days) the late-onset tercile still reaches {late13["flag14"].mean():.3f}, so the development numbers
  do not show it yet.

Proposal (needs your decision; it changes the PLAN 5.3 formula):

- **P1, restart mixture for probe evidence (recommended).** Treat the probe e-values of each node as their own
  restart mixture, P_n(t) = Σ_k w_k Π_{{probes i ≥ k}} f_i, with weights w_k fixed in advance. This is the same idea as
  the Rn(1) jumper of Layer 2. Then use log E_n = log M_n + log P_n.
  - Each term is a test martingale started at probe k, and fixed weights keep the mixture one.
  - Probe and job updates never happen at the same time, so the product should stay a test supermartingale. This
    needs a unit test and a proof note before use.
  - The cost of the debt before onset is then at most −log w_k, with k the first probe after onset, instead of
    growing without limit.
  - The weights (for example, the same jumper rate as Layer 2) must be fixed before any result. It would be checked
    on development seeds only, against validity (P(ever)) and speed.
- **P2, probe healthy nodes less often:** a longer exclusion after clearing. It slows the debt but does not bound it,
  and it lengthens the delay after a wrong clearing. Not recommended.
- **P3, do nothing and disclose:** state that the B = 7200 result holds for onsets within about 95 days of the start
  of probing.

Because a fix seems needed, the primary configuration was **not** tagged `v1.0-dev-frozen` (D107).

### 10.4 The four D102 check runs (D107)

{d_tab}

What changed between them:
1. **{a.d102[0]}** (commit {commit_of(a.d102[0])}): the stop rule was "2,000-pool Monte-Carlo mean of log f > 0".
   - The ten positive rows were the eight "max over all weights" rows plus two flat mem/no_mem bets.
   - The exact values, computed from the bin widths and already in this run, were at most {ex1:.1e} for all ten.
2. **{a.d102[1]}** (same commit): no code change. My edit of the script failed to apply (the formatter had rewrapped
   a line it searched for), and the chained command ran the check anyway. Its metrics.json is byte-identical to
   run 1's ({"yes" if same12 else "no"}).
3. **{a.d102[2]}** (commit {commit_of(a.d102[2])}, "standard error over pools"): the script now records the standard
   error over pools of the maximiser's healthy growth.
   - The stop rule became "exact > 1e-12, or Monte-Carlo > 2 SE".
   - Two rows remained: the flat no_mem bets, where f = 1 for every p, so log f = 2.2e-16 from rounding and SE = 0.
4. **{a.d102[3]}** (commit {commit_of(a.d102[3])}, "floating-point floor"): the Monte-Carlo test uses
   max(2 SE, 1e-12). Zero rows remained.

The numbers did not change between runs 1, 3 and 4; only the stop rule did. That rule change is mine, not D102's
literal "≤ 0". A Monte-Carlo maximum over 2,000 pools is biased upward and cannot show exactly 0, and the exact values
never exceeded {ex1:.1e}. Please confirm the rule ("exact ≤ 1e-12, Monte-Carlo within max(2 SE, 1e-12)") or set another.

### 10.5 Commands

```
uv run python scripts/phase5_d105_diag.py --config configs/phase5.yaml --workers 7      # {a.diag}
uv run python scripts/make_phase5b_addendum.py --diag {a.diag} --grid {a.grid} --d102 {" ".join(a.d102)}
```
"""
    rp = resolve("reports/phase_5b.md")
    old = rp.read_text()
    if HEAD in old:
        old = old[: old.index(HEAD)].rstrip() + "\n"
    rp.write_text(old + "\n" + text)
    print("wrote addendum; reproduction", n_chk, "bad", n_bad)
    return 0


if __name__ == "__main__":
    sys.exit(main())
