# ruff: noqa: E501  (markdown report template: table rows are long by nature)
"""Render reports/phase_4b.md from the D73 pilot, the D76 diagnostic and the D70 generation run.

uv run python scripts/make_phase4b_report.py --pilot <run id> --diag <run id> --fresh <run id>
"""

from __future__ import annotations

import argparse
import json
import subprocess

import numpy as np
import pandas as pd
from make_phase4_report import det_table, f, km, km_median, load, p_ever, sel

from hpcdiag.utils.run import PROJECT_ROOT, resolve

VARIANTS = {"v1": "lr_Sc_a", "v2": "lr_Sc_a_blame"}
STREAM_LABEL = {
    "real": "real",
    "iid": "iid uniform",
    "perm": "shuffled across jobs",
    "perm_in_size": "shuffled within job size",
}


def sh(cmd):
    r = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def lam_row(rs, v):
    x = pd.DataFrame([g for r in rs for g in r["variants"][v]["lambda_degraded"]])
    return (
        float(np.average(x["mean_lam"].fillna(0), weights=x["n_after"])),
        float(np.average(x["mean_lam_multi"].fillna(0), weights=x["n_after_multi"])),
        float(x["n_after_multi"].sum() / x["n_after"].sum()),
    )


def pilot_rows(res):
    rows = []
    for s in sorted({r["s"] for r in res if r["kind"] == "node"}):
        rs = sel(res, kind="node", base="random", K=1, s=s)
        for lab, v in VARIANTS.items():
            d = det_table(rs, v)
            m = [r["variants"][v] for r in rs]
            la, lm, sh_multi = lam_row(rs, v)
            rows.append(
                {
                    "s": s,
                    "v": lab,
                    "scen": len(rs),
                    "nodes": len(d),
                    "km": km_median(d["t"], d["detected"]),
                    "f14": float(km(d["t"], d["detected"], [14])[0]),
                    "w14": float(d["within14"].mean()),
                    "end": float(d["detected"].mean()),
                    "fdp": float(np.mean([x["fdp_mean_over_time"] for x in m])),
                    "co": float(np.mean([x["false_nodes_coallocated"] for x in m])),
                    "un": float(np.mean([x["false_nodes_unrelated"] for x in m])),
                    "lam": la,
                    "lam_m": lm,
                    "multi": sh_multi,
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--diag", required=True)
    ap.add_argument("--fresh", required=True)
    a = ap.parse_args()
    res = load(a.pilot)
    dm = json.loads(resolve(f"runs/phase4/{a.diag}/metrics.json").read_text())
    fm = json.loads(resolve(f"runs/phase4b/{a.fresh}/metrics.json").read_text())
    pil = json.loads(resolve(f"runs/phase4/{a.pilot}/metrics.json").read_text())
    commit = {
        "pilot": resolve(f"runs/phase4/{a.pilot}/git_commit.txt").read_text().strip()[:7],
        "diag": resolve(f"runs/phase4/{a.diag}/git_commit.txt").read_text().strip()[:7],
        "fresh": resolve(f"runs/phase4b/{a.fresh}/git_commit.txt").read_text().strip()[:7],
    }
    rows = pilot_rows(res)
    nulls = sel(res, kind="null")
    n1, s1, c1 = p_ever(nulls, "lr_Sc_a")
    n2, s2, c2 = p_ever(nulls, "lr_Sc_a_blame")
    b1 = [r["base"] for r in nulls if r["variants"]["lr_Sc_a"]["ever_flagged"]]
    b2 = sorted(r["base"] for r in nulls if r["variants"]["lr_Sc_a_blame"]["ever_flagged"])
    T = {(r["stream"], r["v"]): r for r in dm["table"]}
    alpha = 0.1
    v2r = T[("real", "v2")]
    limit = alpha + 2 * v2r["se_200"]
    acf = dm["acf_p_lt_0.1"]
    se_acf = 1 / np.sqrt(dm["n_jobs"])
    fl = pd.DataFrame(dm["flags"])
    fl["day"] = fl["date"].str[:10]
    piv = fl.pivot_table(
        index="day", columns="version", values="base", aggfunc="size", fill_value=0
    )
    fl["week"] = (
        pd.to_datetime(fl["day"]).dt.to_period("W-SUN").dt.start_time.dt.strftime("%Y-%m-%d")
    )
    wk = fl.pivot_table(
        index="week", columns="version", values="base", aggfunc="size", fill_value=0
    )
    v2_top = int(wk["v2"].nlargest(5).sum())
    v1f = fl[fl["version"] == "v1"].sort_values("date")
    v2f = fl[fl["version"] == "v2"]
    w1f = fl[fl["version"] == "v1_w1"]
    busiest_w1 = piv["v1_w1"].nlargest(2)

    s = "# Phase 4b report — predictable blame allocation (D70–D77)\n\n"
    s += "**Status: D71 rejected (D75). v1 (`lr_Sc_a`) remains the Layer-2 configuration.** "
    s += "The Phase 4 gate results stand as reported: Gate 4a PASS, Gate 4b FAIL. "
    s += "No Layer-2 design was evaluated on the fresh seeds, which stay sealed (D77).\n\n"
    s += "## 1. Result\n\n"
    s += f"- v2 (D71) detects faster on development seeds, but flags **{f(v2r['P_ever_200'])} ± {f(v2r['se_200'])}** of the 200 old null bases on the real p-value trace, above the Gate 4a limit of {f(limit)} (α + 2 SE). v1 flags {f(T[('real', 'v1')]['P_ever_200'])} ± {f(T[('real', 'v1')]['se_200'])}.\n"
    s += f"- On iid uniform p-values v2 flags {f(T[('iid', 'v2')]['P_ever_200'])} ± {f(T[('iid', 'v2')]['se_200'])}, so the code and the proof hold. The failure comes from the p-values.\n"
    s += "- **The per-node e-process guarantee needs p-values that are valid given the past. Real Mode P p-values violate this through temporal clustering (D76 i).** "
    s += f"They are marginally calibrated, but 1{{p < 0.1}} has autocorrelation {f(acf['1'], 2)} at lag 1.\n"
    s += f"- **v1 passes Gate 4a empirically partly because w = 1/|N_j| dilutes bursts (D76 ii).** With w = 1 the same process flags {f(T[('real', 'v1_w1')]['P_ever_200'])} of real nulls but only {f(T[('iid', 'v1_w1')]['P_ever_200'])} of iid nulls.\n"
    s += "- Gate 4a on real data is therefore an empirical result, not a consequence of the theorem. The theorem applies to p-values that meet its condition.\n\n"

    s += "## 2. What was tested (D71, D72)\n\n"
    s += "v2 replaced w_j = 1/|N_j| with λ_{j,n} = M_n⁻ / Σ_{k ∈ N_j} M_k⁻, where M_k⁻ is node k's total jumper capital before the job's tie block. "
    s += "`blame_reference` (step by step) and `blame_logM` (vectorised) are in `src/hpcdiag/stats/evalues.py`. "
    s += "`tests/test_blame.py` checks hand cases, equal capital, single-node jobs, predictability across tie blocks, one-step fairness with λ fixed, and the D49 stopped mean under a simulated global null. The code stays in the repository to reproduce this report.\n\n"

    s += "## 3. Development pilot (D73)\n\n"
    s += f"Run `{a.pilot}` (commit `{commit['pilot']}`, {pil['n_tasks']} scenarios). Seeds 0–4, K = 1%, random base, plus the 20 original nulls. `KM` = Kaplan–Meier; delays in days.\n\n"
    s += "| s | version | scenarios | degraded nodes | KM median delay | found within 14 d | KM P(≤ 14 d) | recall at trace end | mean FDP | co-allocated false nodes per scenario | unrelated false nodes per scenario | mean λ after onset | mean λ, multi-node jobs |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    s += "".join(
        f"| {r['s']} | {r['v']} | {r['scen']} | {r['nodes']} | {f(r['km'], 2)} | {f(r['w14'], 2)} | {f(r['f14'], 2)} | {f(r['end'], 2)} | {f(r['fdp'])} | {f(r['co'], 1)} | {f(r['un'], 1)} | {f(r['lam'], 3)} | {f(r['lam_m'], 3)} |\n"
        for r in rows
    )
    s += f"\nMulti-node jobs give {f(rows[0]['multi'] * 100, 0)}% of the updates on degraded nodes after onset. For v1 the λ columns show w on the same updates.\n\n"
    s += "| version | null P(ever), 20 original nulls | flagged bases |\n|---|---|---|\n"
    s += f"| v1 | {f(n1)} ± {f(s1)} (n = {c1}) | {', '.join(b1)} |\n| v2 | {f(n2)} ± {f(s2)} (n = {c2}) | {', '.join(b2)} |\n\n"
    s += f"- **Power:** the v2 gains cannot be separated from the invalid null behaviour, so they are not claimed as real speed. Even so, the found-within-14-d share at s = 1.3 is {f(rows[1]['w14'], 2)}, below the 0.8 target.\n"
    s += "- **Co-allocated false flags (D72):** D71 did not reduce them on this sample of 5 scenarios per s.\n\n"

    s += "## 4. Null diagnostic (D76 iii)\n\n"
    s += f"Run `{a.diag}` (commit `{commit['diag']}`), on null bases 0–199, not on bases 200–399. Each stream keeps the warm-up mask of the real p-values. Random streams use seed 10000 + base.\n\n"
    s += "| p-value stream | v1 P(ever) | v2 P(ever) | v1 median max log M | v2 median max log M |\n|---|---|---|---|---|\n"
    for k, lab in STREAM_LABEL.items():
        x, y = T[(k, "v1")], T[(k, "v2")]
        s += f"| {lab} | {f(x['P_ever_200'])} ± {f(x['se_200'])} | {f(y['P_ever_200'])} ± {f(y['se_200'])} | {f(x['median_max_logM'], 2)} | {f(y['median_max_logM'], 2)} |\n"
    s += "\nn = 200 per cell, α = 0.1.\n\n"
    s += "- **iid:** code and proof hold for v2.\n"
    s += "- **Shuffled across jobs:** the real marginal distribution does not cause the failure.\n"
    s += "- **Shuffled within job size:** the link between p-value and job size does not cause it either.\n"
    s += "- **Real:** only the time-ordered stream inflates v2.\n\n"
    s += "### Direct check of D76 (ii): no dilution\n\n"
    s += "Added beyond the requested table: v1 with w = 1 instead of 1/|N_j|, so a job bets at full weight on every node.\n\n"
    s += "| p-value stream | v1, w = 1/\\|N_j\\| | v2, λ | v1 with w = 1 |\n|---|---|---|---|\n"
    for k in ("real", "iid"):
        x, y, z = T[(k, "v1")], T[(k, "v2")], T[(k, "v1_w1")]
        s += f"| {STREAM_LABEL[k]} | {f(x['P_ever_200'])} ± {f(x['se_200'])} | {f(y['P_ever_200'])} ± {f(y['se_200'])} | {f(z['P_ever_200'])} ± {f(z['se_200'])} |\n"
    s += "\nThe less a job's bet is diluted, the higher the false-flag rate on real p-values, while all three stay below α on iid p-values.\n\n"

    s += "## 5. Autocorrelation of 1{p < 0.1}\n\n"
    s += f"Real guard (a) S_c Mode P p-values in processing order (D14), {dm['n_jobs']:,} scored jobs. Lag is in jobs.\n\n"
    s += "| lag | 1 | 10 | 100 | 1000 |\n|---|---|---|---|---|\n"
    s += f"| autocorrelation | {f(acf['1'], 3)} | {f(acf['10'], 3)} | {f(acf['100'], 3)} | {f(acf['1000'], 3)} |\n\n"
    s += f"For independent indicators the standard error would be about 1/√n = {f(se_acf, 4)}. Which source produces the bursts (for example same-user campaigns) was not tested.\n\n"
    s += "### Calibration by job size\n\n"
    s += "| job size (nodes) | jobs | job-node pairs | mean p | share p < 0.1 |\n|---|---|---|---|---|\n"
    s += "".join(
        f"| {r['size'].replace('17-1000000', '≥ 17')} | {r['jobs']:,} | {r['pairs']:,} | {f(r['mean_p'])} | {f(r['share_p_lt_0.1'])} |\n"
        for r in dm["by_size"]
    )
    s += "\nThese are marginal shares within each size class. They do not capture time order.\n\n"

    s += "## 6. Dates of the null flags\n\n"
    s += f"Date of the first flag per base, real stream. All bases use the same p-value trace (D52), so a date recurring across bases points to the trace, not the placement. e-BH is evaluated every 6 h. v2 has {len(v2f)} flagged bases, v1 has {len(v1f)}, v1 with w = 1 has {len(w1f)}.\n\n"
    s += "**By week (Monday start):**\n\n| week of | v1 | v2 | v1 with w = 1 |\n|---|---|---|---|\n"
    s += "".join(f"| {d} | {r['v1']} | {r['v2']} | {r['v1_w1']} |\n" for d, r in wk.iterrows())
    s += f"\nThe five busiest weeks hold {v2_top} of the {len(v2f)} v2 flags. For v1 with w = 1, {int(busiest_w1.iloc[0])} flags fall on {busiest_w1.index[0]} and {int(busiest_w1.iloc[1])} on {busiest_w1.index[1]}.\n\n"
    s += "**Every v1 null flag:**\n\n| base | first flag (UTC) | node | jobs on node so far | median job size, 20 largest gains |\n|---|---|---|---|---|\n"
    s += "".join(
        f"| rand{int(r['base']):02d} | {r['date']} | {int(r['node'])} | {int(r['n_updates'])} | {f(float(r['median_size_top20']), 1)} |\n"
        for _, r in v1f.iterrows()
    )
    s += "\n<details><summary>Flags per day</summary>\n\n| day | v1 | v2 | v1 with w = 1 |\n|---|---|---|---|\n"
    s += "".join(f"| {d} | {r['v1']} | {r['v2']} | {r['v1_w1']} |\n" for d, r in piv.iterrows())
    s += "\n</details>\n\n"

    s += "## 7. Fresh scenarios: generated and sealed (D70, D77)\n\n"
    s += f"Run `{a.fresh}` (commit `{commit['fresh']}`) wrote {fm['n_scenarios']:,} scenarios in {f(fm['seconds'], 0)} s, listed in `data/processed/scenarios/index_fresh.parquet`.\n\n"
    s += "| kind | random base | real base |\n|---|---|---|\n"
    for k in ("node", "user_decoy", "user_decoy_heavy", "system_shock", "null"):
        s += f"| {k} | {fm['grid'].get(k + '|random', 0)} | {fm['grid'].get(k + '|real', 0)} |\n"
    s += "\n- **Sealed:** nothing runs on seeds 20–39 or nulls 200–399 until the final system (Layer 2 v1 + Phase 5 active probing) is frozen. D77 replaces the D74 run. No Layer-2 code reads `index_fresh.parquet`.\n"
    s += "- **Generation:** `src/hpcdiag/inject/run_fresh.py` reuses the Phase 3 keyed draws and checks. `run_inject.py`, `configs/phase3_inject.yaml` and `index.parquet` are unchanged.\n"
    s += "- **Disclosure:** random bases 20–39 were already used as null bases in Phase 4, and again in the null diagnostic above. Their placements are not new. Their degraded-node draws are new and have not been evaluated. Null bases 200–399 have not been used in any evaluation.\n\n"

    s += "## 8. Decisions\n\n"
    s += "| # | Decision |\n|---|---|\n"
    s += "| D70 | Fresh evaluation seeds 20–39 and nulls 200–399 generated; seeds 0–19 become development seeds |\n"
    s += "| D71 | Predictable blame allocation tested as v2 |\n"
    s += "| D72 | Honest validity statement added to PLAN Phase 4.1 |\n"
    s += "| D73 | Development pilot run |\n"
    s += "| D74 | Not run; replaced by D77 |\n"
    s += "| D75 | D71 rejected. v1 stays. No λ caps, batching or other tuning of D71 |\n"
    s += "| D76 | Temporal clustering recorded as a result, in PLAN Phase 4.1 and §12 |\n"
    s += "| D77 | Fresh seeds sealed until the final system is frozen |\n\n"

    s += "## 9. Limitations\n\n"
    s += "- The 200 real-stream nulls share one p-value trace (D52). Their SEs are conditional on this trace, and this is one trace. The iid and shuffled rows draw new randomness per base.\n"
    s += "- The D73 pilot has 5 scenarios per s, on development seeds. The 0.300 null rate is on old bases, so it is not a formal Gate 4a result.\n"
    s += "- The check of D76 (ii) changes bet size on every job, so it shows that dilution matters, not how much of v1's pass is due to dilution alone.\n"
    s += "- v1 is also exposed to this clustering. Its pass does not show the p-values are valid, and a new trace with stronger bursts could behave differently.\n\n"

    s += "## 10. Reproduce\n\n```\n"
    s += "uv run python -m hpcdiag.inject.run_fresh --config configs/phase4b_fresh.yaml\n"
    s += "uv run python -m hpcdiag.layer2.run_layer2 --config configs/phase4_layer2.yaml --mode pilot4b --workers 7\n"
    s += "uv run python scripts/diag_d71_null.py --config configs/phase4_layer2.yaml --workers 7\n"
    s += f"uv run python scripts/make_phase4b_report.py --pilot {a.pilot} --diag {a.diag} --fresh {a.fresh}\n```\n\n"
    s += f"`uv run pytest -q`: {sh(['uv', 'run', 'pytest', '-q']).splitlines()[-1]}.\n"
    (PROJECT_ROOT / "reports" / "phase_4b.md").write_text(s, encoding="utf8")
    print("written; v2 real", v2r["P_ever_200"], "limit", round(limit, 3))


if __name__ == "__main__":
    main()
