"""D81 (b), (c) tables from a pilot4c run, plus the drift of D78 from the diagnostic.

uv run python scripts/phase4c_tables.py --run <pilot4c run id> --diag <diag-4c run id>
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from make_phase4_report import det_table, km, km_median, load, p_ever, sel

from hpcdiag.utils.run import resolve

VARIANTS = ["lr_Sc_a", "lr_Sc_a_w1", "R1", "R025", "Rn1", "Rn025"]
DECOYS = ("user_decoy", "user_decoy_heavy", "system_shock")


def node_rows(res):
    rows = []
    for s in sorted({r["s"] for r in res if r["kind"] == "node"}):
        rs = sel(res, kind="node", base="random", K=1, s=s)
        for v in VARIANTS:
            d = det_table(rs, v)
            m = [r["variants"][v] for r in rs]
            rows.append(
                {
                    "s": s,
                    "variant": v,
                    "scen": len(rs),
                    "nodes": len(d),
                    "km_median_d": km_median(d["t"], d["detected"]),
                    "km_F14": float(km(d["t"], d["detected"], [14])[0]),
                    "within14": float(d["within14"].mean()),
                    "recall_end": float(d["detected"].mean()),
                    "fdp_mean": float(np.mean([x["fdp_mean_over_time"] for x in m])),
                    "false_coalloc": float(np.mean([x["false_nodes_coallocated"] for x in m])),
                    "false_unrelated": float(np.mean([x["false_nodes_unrelated"] for x in m])),
                }
            )
    return rows


def decoy_rows(res):
    rows = []
    for kind in DECOYS:
        rs = sel(res, kind=kind, base="random")
        for v in VARIANTS:
            p, se, n = p_ever(rs, v)
            fl = sorted({r["base"] for r in rs if r["variants"][v]["ever_flagged"]})
            rows.append(
                {
                    "kind": kind,
                    "variant": v,
                    "n": n,
                    "P_ever": p,
                    "se": se,
                    "flagged_scenarios": sum(r["variants"][v]["ever_flagged"] for r in rs),
                    "bases": len(fl),
                    "base_list": " ".join(fl[:8]),
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--diag", required=True)
    a = ap.parse_args()
    res = load(a.run)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 60)
    print(pd.DataFrame(node_rows(res)).round(3).to_string(index=False))
    print(pd.DataFrame(decoy_rows(res)).round(3).to_string(index=False))
    m = json.loads(resolve(f"runs/phase4/{a.diag}/metrics.json").read_text())
    print(json.dumps(m["drift"], indent=0))
    f = pd.DataFrame(m["flags"])
    print(f[f["version"].isin(["v1", "R1", "Rn1"])].sort_values("date").to_string(index=False))


if __name__ == "__main__":
    main()
