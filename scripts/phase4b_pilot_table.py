"""D73 table: v1 (lr_Sc_a) vs v2 (lr_Sc_a + D71) on development seeds.

uv run python scripts/phase4b_pilot_table.py --run <pilot4b run id>
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from make_phase4_report import det_table, km, km_median, load, p_ever, sel

VARIANTS = {"v1": "lr_Sc_a", "v2": "lr_Sc_a_blame"}


def lam_summary(rs, v):
    x = pd.DataFrame([g for r in rs for g in r["variants"][v]["lambda_degraded"]])
    return {
        "mean_lam_all": float(np.average(x["mean_lam"].fillna(0), weights=x["n_after"])),
        "mean_lam_multi": float(
            np.average(x["mean_lam_multi"].fillna(0), weights=x["n_after_multi"])
        ),
        "share_updates_multi": float(x["n_after_multi"].sum() / x["n_after"].sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    res = load(a.run)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    rows = []
    for s in sorted({r["s"] for r in res if r["kind"] == "node"}):
        rs = sel(res, kind="node", base="random", K=1, s=s)
        for lab, v in VARIANTS.items():
            d = det_table(rs, v)
            m = [r["variants"][v] for r in rs]
            g = pd.DataFrame([x for r in rs for x in r["variants"][v]["growth"]])
            rows.append(
                {
                    "s": s,
                    "v": lab,
                    "scen": len(rs),
                    "nodes": len(d),
                    "km_median_d": km_median(d["t"], d["detected"]),
                    "km_F14": float(km(d["t"], d["detected"], [14])[0]),
                    "within14_a": float(d["within14"].mean()),
                    "recall_end": float(d["detected"].mean()),
                    "fdp_mean": float(np.mean([x["fdp_mean_over_time"] for x in m])),
                    "false_coalloc": float(np.mean([x["false_nodes_coallocated"] for x in m])),
                    "false_unrelated": float(np.mean([x["false_nodes_unrelated"] for x in m])),
                    "growth_per_pair": float(g["growth_per_pair"].mean()),
                    **lam_summary(rs, v),
                }
            )
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    nulls = sel(res, kind="null")
    print()
    for lab, v in VARIANTS.items():
        p, se, n = p_ever(nulls, v)
        flagged = [r["base"] for r in nulls if r["variants"][v]["ever_flagged"]]
        dist = np.mean([r["variants"][v]["distinct_nodes_ever"] for r in nulls])
        print(
            f"{lab} null P(ever) = {p:.3f} ± {se:.3f} (n = {n}); bases {flagged}; "
            f"mean distinct nodes {dist:.3f}; flags/day "
            f"{np.mean([r['variants'][v]['flags_per_day'] for r in nulls]):.4f}"
        )
    t = np.array([r["timing_s"]["variants"]["lr_Sc_a_blame"] for r in res])
    print(f"v2 variant seconds: median {np.median(t):.1f}, max {t.max():.1f}")


if __name__ == "__main__":
    main()
