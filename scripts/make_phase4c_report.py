# ruff: noqa: E501  (markdown report template: table rows are long by nature)
"""Render reports/phase_4c.md from the D81 pilot and null diagnostic.

uv run python scripts/make_phase4c_report.py --pilot <run id> --diag <run id> --diag4b <run id>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

import numpy as np
import pandas as pd
from make_phase4_report import f, load
from phase4c_tables import decoy_rows, node_rows

from hpcdiag.utils.run import PROJECT_ROOT, resolve

LAB = {
    "lr_Sc_a": "v1",
    "lr_Sc_a_w1": "v1, w = 1",
    "R1": "R(1)",
    "R025": "R(0.25)",
    "Rn1": "Rn(1)",
    "Rn025": "Rn(0.25)",
}
DIAG_V = {
    "v1": "v1",
    "v1_w1": "v1, w = 1",
    "R1": "R(1)",
    "R025": "R(0.25)",
    "Rn1": "Rn(1)",
    "Rn025": "Rn(0.25)",
}
STREAM_LABEL = {
    "real": "real",
    "iid": "iid uniform",
    "perm": "shuffled across jobs",
    "perm_in_size": "shuffled within job size",
}


def sh(cmd):
    r = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def sim_bursts(n=200):
    sys.path.insert(0, str(PROJECT_ROOT / "tests"))
    from test_relative import _sim_bursts

    r = pd.DataFrame([_sim_bursts(s) for s in range(n)]).mean()
    return {k: float(v) for k, v in r.items()}, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--diag", required=True)
    ap.add_argument("--diag4b", required=True)
    a = ap.parse_args()
    res = load(a.pilot)
    dm = json.loads(resolve(f"runs/phase4/{a.diag}/metrics.json").read_text())
    d4b = json.loads(resolve(f"runs/phase4/{a.diag4b}/metrics.json").read_text())
    pil = json.loads(resolve(f"runs/phase4/{a.pilot}/metrics.json").read_text())
    cp = resolve(f"runs/phase4/{a.pilot}/git_commit.txt").read_text().strip()[:7]
    cd = resolve(f"runs/phase4/{a.diag}/git_commit.txt").read_text().strip()[:7]
    T = {(r["stream"], r["v"]): r for r in dm["table"]}
    for r in d4b["table"]:
        if r["v"] in ("v1", "v1_w1"):
            x = T[(r["stream"], r["v"])]
            if (
                x["P_ever_200"] != r["P_ever_200"]
                or abs(x["median_max_logM"] - r["median_max_logM"]) > 1e-12
            ):
                raise RuntimeError(f"Phase 4b cell not reproduced: {r}")
    rows = node_rows(res)
    N = {(r["s"], r["variant"]): r for r in rows}
    dec = decoy_rows(res)
    D = {(r["kind"], r["variant"]): r for r in dec}
    fl = pd.DataFrame(dm["flags"])
    fl["day"] = fl["date"].str[:10]
    day_max = {v: int(g.groupby("day").size().max()) for v, g in fl.groupby("version")}
    n_flag = fl["version"].value_counts().to_dict()
    dr = dm["drift"]
    dr_all = [abs(v) for d in dr.values() for v in d.values()]
    budget = float(np.log(980 / 0.1) + np.log(6 / 0.01))
    sim, nsim = sim_bursts()
    R1r, wr = T[("real", "R1")], T[("real", "v1_w1")]

    s = "# Phase 4c report — relative (cross-sectional) e-values (D78–D82)\n\n"
    s += "**Status: development evaluation only. No design decision is made here.** v1 (`lr_Sc_a`) remains the Layer-2 configuration. No scenario with seeds 20–39 and no null base 200–399 was run (D77, D82). The 200 old null bases of D81 (a) include placements 20–39, as in Phases 4 and 4b.\n\n"
    s += "## 1. Result\n\n"
    s += f"- **Validity:** R(1) flags {f(R1r['P_ever_200'])} ± {f(R1r['se_200'])} of 200 real-stream nulls, where v1 with w = 1 flags {f(wr['P_ever_200'])} ± {f(wr['se_200'])}. Relative evidence removes the burst failure of Phase 4b.\n"
    s += f"- **Decoys and shocks:** R(1) has P(ever) = {f(D[('user_decoy', 'R1')]['P_ever'])} on all three kinds (240 scenarios, bases 0–19).\n"
    s += f"- **Speed:** R(1) is slower than v1 at every s tested (section 5). At s = 1.3, K = 1%, KM median delay is {f(N[(1.3, 'R1')]['km_median_d'], 2)} d vs {f(N[(1.3, 'lr_Sc_a')]['km_median_d'], 2)} d and recall at trace end {f(N[(1.3, 'R1')]['recall_end'], 2)} vs {f(N[(1.3, 'lr_Sc_a')]['recall_end'], 2)}. Found within 14 d is {f(N[(1.3, 'R1')]['within14'], 2)} for both, far below the 0.8 target of Gate 4b. At s = 2.0 and 3.0 its recall at trace end is slightly higher than v1, on 50 nodes.\n"
    s += "- **R(0.25) is worse than R(1) at every s** on delay, within-14-d share and recall at trace end.\n"
    s += "- **The D78 proof does not hold as specified.** D80(1) exact enumeration gives mean update 8/9 and 19/18 for mixed job sizes (section 2). A node-weighted denominator (Rn, added by me, not adopted) is exact and gives almost the same numbers.\n\n"

    s += "## 2. Problem in the D78 validity proof\n\n"
    s += "D78 argues that node n's pairs are a uniform draw from the block's pairs in its partition, so E[e_{n,b} | …] = 1. The D80(1) enumeration shows this holds for equal job sizes but not for mixed ones.\n\n"
    s += "Hand case, confirmed by `tests/test_relative.py`: 3 nodes; job A has size 1, job B has size 2; 9 equally likely placements, each job drawn uniformly as in `inject/placement.py`. Node 0 holds both jobs in 2 placements, A only in 1, B only in 4, neither in 2.\n\n"
    s += "| f_A | f_B | f̄ (pair-weighted) | exact mean update | mean e given a pair | D78 claim |\n|---|---|---|---|---|---|\n"
    s += "| 3 | 0 | 1 | 8/9 | 6/7 | 1 |\n| 0 | 3 | 2 | 19/18 | — | 1 |\n\n"
    s += "The reason is that K_n, the number of pairs on the node, depends on which jobs are there, and the mean inside e divides by K_n. So R(λ) with the pair-weighted f̄ is not an e-process under the global null. Mean 1 does hold exactly when all jobs in the block have the same size.\n\n"
    s += "**Candidate correction (Rn), not adopted because it changes a D78 formula.** Use as denominator the mean of the node means a_{n,b} over the nodes of q with a pair in b. The ratios then sum to the number of active nodes in every placement. Node labels are exchangeable, so the mean is exactly 1 for any sizes and any temporal dependence. The full proof is in PLAN Phase 4.1. Rn equals R when every active node has one pair.\n\n"
    s += f"**Practical size of the error (section 7):** over the whole trace the average drift of R is at most {f(max(dr_all), 2)} nats in size, against an evidence budget of {f(budget, 2)} nats. The error is real but small here, and R and Rn give close results in sections 4–6.\n\n"

    s += "## 3. Tests (D80)\n\n| D80 item | Result |\n|---|---|\n"
    s += "| (1) Exact enumeration, mean 1 | Equal sizes: passes for R and Rn. Mixed sizes: **fails for R as specified** (the test pins 8/9 and 19/18); exact for Rn |\n"
    s += f"| (2) Burst simulation, n = {nsim}, 100 nodes, 120 blocks, mixed sizes | P(ever): R(1) {f(sim['R1'])}, R(0.25) {f(sim['R025'])}, Rn(1) {f(sim['Rn1'])}, full bet with w = 1 {f(sim['w1'])}. α = 0.1 |\n"
    s += "| (3) Means per partition, block-b jobs only | Passes |\n"
    s += "| (4) λ one constant fixed in advance; nodes with no pairs not updated | Passes |\n\n"
    s += "`src/hpcdiag/stats/evalues.py`: `relative_evalues`, `relative_logM`. `src/hpcdiag/layer2/accumulate.py`: `node_evidence_relative`, `block_grid`.\n\n"

    s += "## 4. Null diagnostic (D81 a)\n\n"
    s += f"Run `{a.diag}` (commit `{cd}`), null bases 0–199, n = 200 per cell, α = 0.1. v1 and v1 with w = 1 use the guard (a) p-values, as in Phase 4b. R and Rn use unguarded S_c p-values (D78). The v1 and v1 with w = 1 cells on the real and iid streams, and the v1 cells on the shuffled streams, match run `{a.diag4b}` exactly (checked by this script).\n\n"
    s += (
        "| p-value stream | "
        + " | ".join(DIAG_V.values())
        + " |\n|---|"
        + "---|" * len(DIAG_V)
        + "\n"
    )
    for k, lab in STREAM_LABEL.items():
        s += (
            f"| {lab} | "
            + " | ".join(f"{f(T[(k, v)]['P_ever_200'])} ± {f(T[(k, v)]['se_200'])}" for v in DIAG_V)
            + " |\n"
        )
    s += f"\nFlagged real-stream bases: R(1) {n_flag.get('R1', 0)}, v1 {n_flag.get('v1', 0)}, v1 with w = 1 {n_flag.get('v1_w1', 0)}. The most bases flagged on one day: R(1) {day_max.get('R1', 0)}, v1 with w = 1 {day_max.get('v1_w1', 0)}. The R(1) first-flag dates do not pile up the way the w = 1 dates do.\n\n"

    s += "## 5. Development pilot (D81 b)\n\n"
    s += f"Run `{a.pilot}` (commit `{cp}`). Seeds 0–4, node, K = 1%, random base: 5 scenarios and 50 degraded nodes per s. Delays in days.\n\n"
    s += "| s | variant | KM median delay | found within 14 d | KM P(≤ 14 d) | recall at trace end | mean FDP | co-allocated false nodes per scenario | unrelated false nodes per scenario |\n|---|---|---|---|---|---|---|---|---|\n"
    s += "".join(
        f"| {r['s']} | {LAB[r['variant']]} | {f(r['km_median_d'], 2)} | {f(r['within14'], 2)} | {f(r['km_F14'], 2)} | {f(r['recall_end'], 2)} | {f(r['fdp_mean'])} | {f(r['false_coalloc'], 1)} | {f(r['false_unrelated'], 1)} |\n"
        for r in rows
    )
    s += "\nv1 with w = 1 is fast, but it fails every null check, so its delays are not a usable result. It is shown as the reference D79 asks for.\n\n"

    s += "## 6. Decoys and shocks (D81 c)\n\n"
    s += "Random base, seeds 0–19, 4 values of s, 80 scenarios per kind. P(ever) ± SE.\n\n"
    s += "| kind | " + " | ".join(LAB.values()) + " |\n|---|" + "---|" * len(LAB) + "\n"
    for kind in ("user_decoy", "user_decoy_heavy", "system_shock"):
        s += (
            f"| {kind} | "
            + " | ".join(f"{f(D[(kind, v)]['P_ever'])} ± {f(D[(kind, v)]['se'])}" for v in LAB)
            + " |\n"
        )
    s += f"\nv1's flags are all on base {D[('user_decoy', 'lr_Sc_a')]['base_list']}, whose null also flags, as in Phase 4. The 7% null rate of R(1) comes from bases outside 0–19, so these 240 scenarios sit on 20 bases with no R(1) null flag. A zero here does not mean R(1) can never flag on a decoy scenario.\n\n"

    s += "## 7. Drift of the pair-weighted form\n\n"
    s += "Sum over blocks of log(mean of e over active nodes), R denominator, averaged over 200 bases. This is a population average, not any single node. A fair update gives 0. Compare with the budget of "
    s += (
        f"{f(budget, 2)} nats.\n\n| stream | "
        + " | ".join(dr["real"])
        + " |\n|---|"
        + "---|" * len(dr["real"])
        + "\n"
    )
    for k in ("real", "iid"):
        s += f"| {STREAM_LABEL[k]} | " + " | ".join(f(v, 2) for v in dr[k].values()) + " |\n"
    s += "\n## 8. Choices made while implementing\n\n"
    s += "- **Warm-up jobs:** the 6-h grid is extended backwards from the first evaluation, so jobs before it fall in 6-h blocks. D78 does not cover these jobs.\n"
    s += "- **Bets:** the existing `Sc` set, built on unguarded S_c, because D78 uses no guard. Bets do not depend on placement, so the proof is not affected.\n"
    s += "- **All f = 0 in a block:** e = 1.\n"
    s += "- **Blocks:** a block is (t_{b−1}, t_b]. Jobs ending after the last evaluation are not used, since no evaluation could see them.\n"
    s += "- **Jumper:** it mixes once per block and active node, so it takes fewer steps than v1, which mixes once per pair.\n"
    s += "- **Rn variants:** added beyond D79, before any R result was seen, because of section 2.\n\n"

    s += "## 9. Decisions\n\n| # | Decision |\n|---|---|\n"
    s += "| D78 | Relative e-values implemented as specified; validity proof fails for mixed sizes (section 2) |\n"
    s += "| D79 | R(1), R(0.25), v1 and v1 with w = 1 evaluated |\n"
    s += "| D80 | Tests written first; item (1) found the proof failure |\n"
    s += "| D81 | Development evaluation done |\n"
    s += "| D82 | Sealing unchanged |\n\n"
    s += "**Open, for Mahesh:** whether to adopt the Rn denominator, and whether R(1) or v1 is Layer 2.\n\n"

    s += "## 10. Limitations\n\n"
    s += "- Evaluated on the random base only. On the real placement, exchangeability within partition and block is an assumption [A] and has not been tested. Phase 4 found 9 real-placement null flags under v1.\n"
    s += "- The pilot has 5 scenarios per s, on development seeds. The speed differences are not precise.\n"
    s += "- All real-stream nulls share one trace (D52). For Rn this matches the proof, which fixes the p-values and randomises placement. For v1 it does not.\n"
    s += "- With degraded nodes present, straggler contamination still applies (D72): co-allocated jobs are late, and they enter e_{n,b} for healthy nodes.\n"
    s += "- Relative evidence cannot see a slowdown shared by all active nodes of a partition. K up to 5% was not tested in this phase.\n\n"

    s += "## 11. Reproduce\n\n```\n"
    s += "uv run python -m hpcdiag.layer2.run_layer2 --config configs/phase4_layer2.yaml --mode pilot4c --workers 7\n"
    s += "uv run python scripts/diag_4c_null.py --config configs/phase4_layer2.yaml --workers 7\n"
    s += f"uv run python scripts/phase4c_tables.py --run {a.pilot} --diag {a.diag}\n"
    s += f"uv run python scripts/make_phase4c_report.py --pilot {a.pilot} --diag {a.diag} --diag4b {a.diag4b}\n```\n\n"
    tests = [x for x in sh(["uv", "run", "pytest", "-q"]).splitlines() if " passed" in x][-1]
    s += f"`uv run pytest -q`: {tests}. Scenarios in the pilot: {pil['n_tasks']}.\n"
    (PROJECT_ROOT / "reports" / "phase_4c.md").write_text(s, encoding="utf8")
    print("written", sim, day_max, n_flag, round(max(dr_all), 2), round(budget, 2))


if __name__ == "__main__":
    main()
