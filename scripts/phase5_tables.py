"""Tables for Phase 5 from a full run (docs/PHASE5_DESIGN.md §6-§7).

    uv run python scripts/phase5_tables.py --run <full run id> --addon <addon run id> \
        [--v1 20261010-094328-full]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from make_phase4_report import det_table, km, km_median

from hpcdiag.utils.run import resolve

ALPHA = 0.1
VALID_KINDS = ("null", "user_decoy", "user_decoy_heavy", "system_shock")


def load(run: str) -> list[dict]:
    with open(resolve(f"runs/phase5/{run}/results.jsonl")) as fh:
        return [json.loads(x) for x in fh]


def merged(*runs: str) -> list[dict]:
    """Rows of several runs; a repeated (scenario, variant, policy, B) keeps the first."""
    out, seen = [], set()
    for rid in runs:
        for r in load(rid):
            k = (r["scenario_id"], r["variant"], r["policy"], r["B"])
            if k not in seen:
                seen.add(k)
                out.append(r)
    return out


def pick(res, **kw):
    out = res
    for k, v in kw.items():
        if k == "random":
            out = [r for r in out if (r["base"] != "real") == v]
        else:
            out = [r for r in out if r[k] == v]
    return out


def wrap(rs):
    return [{**r, "variants": {"x": r["metrics"]}} for r in rs]


def speed_row(rs) -> dict:
    d = det_table(wrap(rs), "x")
    m = [r["metrics"] for r in rs]
    return {
        "scen": len(rs),
        "nodes": len(d),
        "km_F14": float(km(d["t"], d["detected"], [14])[0]),
        "km_median_d": km_median(d["t"], d["detected"]),
        "within14": float(d["within14"].mean()),
        "recall_end": float(d["detected"].mean()),
        "fdp_mean": float(np.mean([x["fdp_mean_over_time"] for x in m])),
        "false_coalloc": float(np.mean([x["false_nodes_coallocated"] for x in m])),
        "false_unrelated": float(np.mean([x["false_nodes_unrelated"] for x in m])),
    }


def km_ci(rs, n_boot=1000, seed=0) -> tuple[float, float]:
    """Percentile interval of KM F(14 d), resampling scenarios."""
    d = det_table(wrap(rs), "x")
    groups = [g for _, g in d.groupby("scenario")]
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n_boot):
        s = pd.concat([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        v.append(km(s["t"], s["detected"], [14])[0])
    return float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))


def validity_row(rs) -> dict:
    x = np.array([r["metrics"]["ever_flagged"] for r in rs], float)
    p = float(x.mean()) if len(x) else np.nan
    se = float(np.sqrt(p * (1 - p) / len(x))) if len(x) else np.nan
    return {"n": len(x), "P_ever": p, "se": se, "pass": bool(p <= ALPHA + 2 * se)}


def diag_row(rs) -> dict:
    fl = [d for r in rs for d in r["diagnosis"] if d.get("flagged")]
    inv = [d for d in fl if d["investigated"]]
    n_inv = sum(r["n_investigations"] for r in rs)
    sec = sum(r["seconds_total"] for r in rs)
    unc = sum(r["n_uncut"] for r in rs)
    correct_all = sum(d["correct"] for r in rs for d in r["diagnosis"])
    days = np.mean([r["seconds_total"] / r["seconds_per_day"] for r in rs if r["seconds_per_day"]])
    return {
        "flagged_degraded": len(fl),
        "accuracy": sum(d["correct"] for d in fl) / len(fl) if fl else np.nan,
        "never_investigated": len(fl) - len(inv),
        "accuracy_investigated": sum(d["correct"] for d in inv) / len(inv) if inv else np.nan,
        "s_per_investigation": sec / n_inv if n_inv else np.nan,
        "s_per_uncut": sum(r["seconds_uncut"] for r in rs) / unc if unc else np.nan,
        "share_cut": 1 - unc / n_inv if n_inv else np.nan,
        "s_per_correct": sec / correct_all if correct_all else np.nan,
        "healthy_inv_per_day": float(np.mean([r["healthy_investigations_per_day"] for r in rs])),
        "s_per_day": sec / len(rs) / days if len(rs) else np.nan,
    }


def confusion(rs) -> dict:
    rows: dict = {}
    for r in rs:
        for d in r["diagnosis"]:
            if d.get("flagged"):
                k = d["cause_true"]
                o = d["cause_out"] or "never"
                rows.setdefault(k, {}).setdefault(o, 0)
                rows[k][o] += 1
    return rows


def tables(res, v1, addon) -> dict:
    t: dict = {}
    main = pick(res, variant="main")
    budgets = sorted({r["B"] for r in pick(main, policy="ours")})
    s_list = sorted({r["s"] for r in res if r["kind"] == "node"})
    rows = []
    for kind in VALID_KINDS:
        for B in budgets:
            rows.append(
                {
                    "kind": kind,
                    "row": "ours",
                    "B": B,
                    **validity_row(pick(main, policy="ours", B=B, kind=kind, random=True)),
                }
            )
        for v in sorted({r["variant"] for r in res} - {"main"}):
            rs = pick(res, variant=v, policy="ours", B=300, kind=kind, random=True)
            if rs:
                rows.append({"kind": kind, "row": v, "B": 300, **validity_row(rs)})
        for p in ("ours_design_rule", "ours_skipdone"):
            rs = pick(main, policy=p, B=300, kind=kind, random=True)
            rows.append({"kind": kind, "row": p, "B": 300, **validity_row(rs)})
        rs = [
            r
            for r in v1
            if r["kind"] == kind and r["base"] != "real" and (r["seed"] is None or r["seed"] <= 19)
        ]
        x = np.array([r["variants"]["lr_Sc_a"]["ever_flagged"] for r in rs], float)
        rows.append(
            {
                "kind": kind,
                "row": "v1",
                "B": 0,
                "n": len(x),
                "P_ever": float(x.mean()),
                "se": float(np.sqrt(x.mean() * (1 - x.mean()) / len(x))),
                "pass": bool(x.mean() <= ALPHA + 2 * np.sqrt(x.mean() * (1 - x.mean()) / len(x))),
            }
        )
    t["validity"] = rows
    rows = []
    for K in (1, 2, 5):
        for s in s_list:
            for B in budgets:
                rs = pick(main, policy="ours", B=B, kind="node", K_pct=K, s=s, random=True)
                rows.append({"K": K, "s": s, "row": "ours", "B": B, **speed_row(rs)})
            vr = [
                r
                for r in v1
                if r["kind"] == "node" and r["base"] != "real" and r["K_pct"] == K and r["s"] == s
            ]
            vm = [
                {
                    "metrics": r["variants"]["lr_Sc_a"],
                    "scenario_id": r["scenario_id"],
                    "s": s,
                    "K_pct": K,
                    "seed": r["seed"],
                }
                for r in vr
            ]
            rows.append({"K": K, "s": s, "row": "v1", "B": 0, **speed_row(vm)})
    t["speed"] = rows
    gate = pick(main, policy="ours", B=300, kind="node", K_pct=1, s=1.3, random=True)
    t["gate5b_ci"] = km_ci(gate)
    rows = []
    for s in s_list:
        for p in ("ours", "P-all", "P-IG", "P-random", "P-randnode", "P-sweep"):
            for B in (60, 300, 1800):
                rs = pick(main, policy=p, B=B, kind="node", K_pct=1, s=s, random=True)
                rows.append({"s": s, "row": p, "B": B, **speed_row(rs), **diag_row(rs)})
    t["baselines"] = rows
    rows = []
    for B in budgets[1:]:
        rs = pick(main, policy="ours", B=B, kind="node", random=True)
        rows.append({"s": "all", "K": "all", "B": B, **diag_row(rs)})
        for s in s_list:
            rs2 = [r for r in rs if r["s"] == s]
            rows.append({"s": s, "K": "all", "B": B, **diag_row(rs2)})
    t["diagnosis"] = rows
    rs = pick(main, policy="ours", B=300, kind="node", random=True)
    t["confusion_B300"] = confusion(rs)
    rows = []
    sens = [("main", q, 300) for q in ("ours", "ours_design_rule", "ours_skipdone")]
    sens += [(v, "ours", 300) for v in sorted({r["variant"] for r in res} - {"main"})]
    tel = ("tel_crossblock", "tel_design", "tel_transfer")
    sens += [(v, "ours_skipdone", 300) for v in tel]
    sens += [("main", "ours", 7200)] + [(v, "ours", 7200) for v in tel]
    for v, p, B in sens:
        for s in s_list:
            rs = pick(res, variant=v, policy=p, B=B, kind="node", K_pct=1, s=s, random=True)
            rows.append({"row": f"{v}/{p}/B{B}", "s": s, **speed_row(rs), **diag_row(rs)})
    t["sensitivity"] = rows
    rows = []
    for p in ("ours", "ours_design_rule", "ours_skipdone"):
        rs = pick(main, policy=p, B=300, kind="node", random=True)
        rows.append({"rule": p, **diag_row(rs)})
        fl = [d for r in rs for d in r["diagnosis"] if d.get("flagged") and d["investigated"]]
        lo = [d for d in fl if d["pi_out"] < 0.95]
        rows[-1]["share_output_pi_below_tau"] = len(lo) / len(fl)
        rows[-1]["accuracy_output_pi_below_tau"] = sum(d["correct"] for d in lo) / max(len(lo), 1)
        hi = [d for d in fl if d["pi_out"] >= 0.95]
        rows[-1]["accuracy_output_pi_at_tau"] = sum(d["correct"] for d in hi) / max(len(hi), 1)
    t["gate5c_rules"] = rows
    t["confusion_rules"] = {
        p: confusion(pick(main, policy=p, B=300, kind="node", random=True))
        for p in ("ours_design_rule", "ours_skipdone")
    }
    rows = []
    for B in budgets:
        rs = pick(main, policy="ours", B=B, kind="node", K_pct=1, s=1.3, random=True)
        rows.append({"policy": "ours", "B": B, "km_F14": speed_row(rs)["km_F14"]})
    for B in (60, 300, 1800, 7200):
        rs = pick(main, policy="ours_skipdone", B=B, kind="node", K_pct=1, s=1.3, random=True)
        rows.append({"policy": "ours_skipdone", "B": B, "km_F14": speed_row(rs)["km_F14"]})
    t["d91"] = rows
    ok = [r["B"] for r in rows if r["km_F14"] >= 0.8 and r["policy"] == "ours"]
    t["d91_smallest_B"] = min(ok) if ok else None
    t["gate5b_rows"] = []
    for p in ("ours_design_rule", "ours_skipdone"):
        rs = pick(main, policy=p, B=300, kind="node", K_pct=1, s=1.3, random=True)
        t["gate5b_rows"].append({"policy": p, **speed_row(rs), "ci": km_ci(rs)})
    rows = []
    for p, B in (("ours", 300), ("ours", 7200), ("ours_skipdone", 300)):
        rs = [
            r
            for r in addon
            if r["policy"] == p
            and r["B"] == B
            and r["variant"] == "main"
            and r["kind"] == "node"
            and r["K_pct"] == 1
            and r["base"] != "real"
        ]
        n = sum(r["n_investigations"] for r in rs)
        used = {a: sum(r["probe_counts"][a] for r in rs) / n for a in rs[0]["probe_counts"]}
        sec = {
            a: sum(r["second_probe"][a] for r in rs)
            / max(sum(sum(r["second_probe"].values()) for r in rs), 1)
            for a in rs[0]["second_probe"]
        }
        rows.append(
            {
                "policy": p,
                "B": B,
                "investigations": n,
                **{f"used_{a}": v for a, v in used.items()},
                **{f"second_{a}": v for a, v in sec.items()},
                "healthy_share_of_seconds": float(
                    np.mean([r["healthy_seconds_per_day"] / r["seconds_per_day"] for r in rs])
                ),
            }
        )
    t["probe_mix"] = rows
    rows = []
    for B in sorted({r["B"] for r in res if r["base"] == "real"}):
        for kind in ("node", "null"):
            for s in s_list if kind == "node" else [None]:
                rs = [
                    r
                    for r in pick(main, policy="ours", B=B, kind=kind)
                    if r["base"] == "real" and r["s"] == s
                ]
                if rs:
                    rows.append(
                        {
                            "kind": kind,
                            "s": s,
                            "B": B,
                            "scen": len(rs),
                            "distinct_flagged": float(
                                np.mean([r["metrics"]["distinct_nodes_ever"] for r in rs])
                            ),
                            "flags_per_day": float(
                                np.mean([r["metrics"]["flags_per_day"] for r in rs])
                            ),
                            "healthy_s_per_day": float(
                                np.mean([r["healthy_seconds_per_day"] for r in rs])
                            ),
                        }
                    )
    t["real"] = rows
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--addon", required=True)
    ap.add_argument("--v1", default="20261010-094328-full")
    a = ap.parse_args()
    res = merged(a.run, a.addon)
    addon = load(a.addon)
    with open(resolve(f"runs/phase4/{a.v1}/results.jsonl")) as fh:
        v1 = [json.loads(x) for x in fh]
    t = tables(res, v1, addon)
    pilot = {}
    with open(resolve("runs/phase4/20261101-214937-pilot4c/results.jsonl")) as fh:
        for x in fh:
            r = json.loads(x)
            pilot[r["scenario_id"]] = r["variants"]["Rn1"]["max_logM_final"]
    same = [
        abs(r["max_logE_final"] - pilot[r["scenario_id"]])
        for r in res
        if r["B"] == 0 and r["scenario_id"] in pilot
    ]
    t["b0_matches_phase4c_Rn1"] = {"scenarios": len(same), "max_abs_diff": max(same)}
    out = resolve(f"runs/phase5/{a.run}/tables.json")
    out.write_text(json.dumps(t, indent=1, default=float))
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 500)
    for k, v in t.items():
        print(f"\n== {k}")
        print(
            pd.DataFrame(v).round(3).to_string(index=False)
            if isinstance(v, list) and v and isinstance(v[0], dict)
            else v
        )


if __name__ == "__main__":
    main()
