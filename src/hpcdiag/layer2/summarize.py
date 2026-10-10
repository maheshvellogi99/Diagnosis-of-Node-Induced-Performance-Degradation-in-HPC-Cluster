"""Summaries of Phase 4 results.jsonl (pilot and full runs).

uv run python -m hpcdiag.layer2.summarize --run <phase4 run id>
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from hpcdiag.utils.run import resolve

NODE_KEYS = [
    "recall_a",
    "recall_b",
    "recall_a_within_14d",
    "recall_b_within_14d",
    "median_delay_h",
    "median_delay_jobs",
    "fdp_mean_over_time",
    "fdp_at_onset_plus_7d",
    "fdp_at_onset_plus_14d",
    "false_nodes_coallocated",
    "false_nodes_unrelated",
    "false_nodes_ever",
]


def load(run_id: str) -> list[dict]:
    with open(resolve(f"runs/phase4/{run_id}/results.jsonl")) as fh:
        return [json.loads(line) for line in fh]


def null_table(res: list[dict], kinds=("null",)) -> pd.DataFrame:
    rows = []
    sel = [r for r in res if r["kind"] in kinds]
    for v in sel[0]["variants"] if sel else []:
        ever = np.array([r["variants"][v]["ever_flagged"] for r in sel], float)
        fpd = np.array([r["variants"][v]["flags_per_day"] for r in sel], float)
        n = len(ever)
        p = ever.mean()
        rows.append(
            {
                "variant": v,
                "n": n,
                "P_ever": p,
                "se": np.sqrt(p * (1 - p) / n),
                "flags_per_day_mean": fpd.mean(),
                "distinct_nodes_mean": np.mean(
                    [r["variants"][v]["distinct_nodes_ever"] for r in sel]
                ),
            }
        )
    return pd.DataFrame(rows)


def node_table(res: list[dict], kinds=("node", "node_partial")) -> pd.DataFrame:
    sel = [r for r in res if r["kind"] in kinds]
    rows = []
    for v in sel[0]["variants"] if sel else []:
        d = pd.DataFrame(
            [{k: r["variants"][v].get(k) for k in NODE_KEYS} for r in sel], dtype=float
        )
        dets = [x for r in sel for x in r["variants"][v]["detections"] if x["detected"]]
        row = {"variant": v, "n_scen": len(sel)}
        for k in NODE_KEYS:
            row[k] = float(d[k].mean()) if d[k].notna().any() else np.nan
        row["median_delay_h_pooled"] = (
            float(np.median([x["delay_h"] for x in dets])) if dets else np.nan
        )
        row["median_delay_jobs_pooled"] = (
            float(np.median([x["delay_jobs"] for x in dets])) if dets else np.nan
        )
        row["n_detected_nodes"] = len(dets)
        rows.append(row)
    return pd.DataFrame(rows)


def timing(res: list[dict]) -> dict:
    tot = np.array([r["timing_s"]["total"] for r in res])
    l1 = [r["timing_s"]["layer1_parts"] for r in res if "modeP7" in r["timing_s"]["layer1_parts"]]
    var = pd.DataFrame([r["timing_s"]["variants"] for r in res])
    other = np.array(
        [
            r["timing_s"]["total"]
            - r["timing_s"]["layer1"]
            - r["timing_s"]["pairs"]
            - sum(r["timing_s"]["variants"].values())
            - r["timing_s"]["baselines"]
            for r in res
        ]
    )
    l1d = pd.DataFrame(l1)
    primary_vars = [
        v
        for v in ("jumper_none", "jumper_a", "jumper_b", "lr_Sc_none", "lr_Sc_a", "lr_Sc_b")
        if v in var.columns
    ]
    l1_cols = [c for c in ("population", "modeP7", "modeP7_guard_a", "centred") if c in l1d]
    t_primary = (
        l1d[l1_cols].sum(axis=1).median()
        + np.median([r["timing_s"]["pairs"] for r in res])
        + var[primary_vars].sum(axis=1).median()
        + np.median([r["timing_s"]["baselines"] for r in res])
        + np.median(other)
    )
    t_ablation_uncached = float(
        np.median(
            [r["timing_s"]["total"] for r in res if "modeP7" in r["timing_s"]["layer1_parts"]]
        )
    )
    t_ablation_cached_null = float(
        np.median(
            [r["timing_s"]["total"] for r in res if "cached" in r["timing_s"]["layer1_parts"]]
            or [np.nan]
        )
    )
    return {
        "total_median_s": float(np.median(tot)),
        "total_max_s": float(tot.max()),
        "layer1_full_median_s": float(l1d.sum(axis=1).median()),
        "variants_median_s": var.median().round(2).to_dict(),
        "primary_set_est_s": float(t_primary),
        "ablation_set_uncached_median_s": t_ablation_uncached,
        "ablation_set_null_cached_median_s": t_ablation_cached_null,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    res = load(a.run)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(json.dumps(timing(res), indent=1))
    print(null_table(res).round(4).to_string(index=False))
    print(node_table(res).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
