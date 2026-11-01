"""D84: Rn(1) and v1 on the real-placement null. Development only, exploratory, unknown truth.

uv run python scripts/d84_real_check.py --config configs/phase4_layer2.yaml
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from hpcdiag.layer2 import accumulate as A
from hpcdiag.layer2 import metrics as MT
from hpcdiag.layer2 import prep as P
from hpcdiag.layer2 import run_layer2 as R
from hpcdiag.layer2 import select as SEL
from hpcdiag.utils.run import Run, resolve, run_guarded, start_run

PHASE2_LIST = [840, 287, 841, 80, 309, 81, 441, 42, 679, 40]


def flags_table(fl, snap, eval_ns):
    rows = []
    for n in np.flatnonzero(fl.any(axis=0)):
        hit = np.flatnonzero(fl[:, n])
        rows.append(
            {
                "node": int(n),
                "first_flag": str(pd.Timestamp(int(eval_ns[hit[0]]))),
                "evaluations_flagged": int(len(hit)),
                "max_logM": float(snap[:, n].max()),
            }
        )
    return pd.DataFrame(rows).sort_values("first_flag").reset_index(drop=True)


def concentration(pairs: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    x = pairs.loc[np.isfinite(p[pairs["k"].to_numpy()]), ["node_id", "user"]]
    x = x.assign(user=x["user"].astype(str))
    g = x.groupby(["node_id", "user"]).size().rename("c").reset_index()
    t = g.groupby("node_id")["c"].agg(jobs="sum", users="size", top="max")
    t["top_user_share"] = t["top"] / t["jobs"]
    return t


def _run(run: Run, config_path: str) -> int:
    R._init(config_path)
    cfg = R.G["cfg"]
    lc, l1 = cfg["layer2"], cfg["layer1"]
    meta = json.loads(
        (resolve(cfg["data"]["scenarios"]) / "null__real" / "scenario.json").read_text()
    )
    pop = R._null_pop()
    pc = P.mode_p_on(pop, "score_c", l1["window_hours"], l1["min_ref"])
    pa = P.mode_p_on(pop, "score_c", l1["window_hours"], l1["min_ref"], l1["guard_a_hours"])
    pairs = A.make_pairs(pop, P.load_placement(meta["job_nodes"]))
    J, m, eval_ns = lc["jumper_J"], lc["m_fixed"], R.G["eval_ns"]
    part = pd.factorize(pop["partition"].astype(str))[0]
    grid = A.block_grid(eval_ns, int(pop["end_ns"].min()))
    evs = {
        "Rn1": A.node_evidence_relative(pairs, pc, part, R.G["bets"]["Sc"], J, 1.0, grid, "node"),
        "v1": A.node_evidence(pairs, pa, A.weights(pairs), R.G["bets"]["Sc_a"], J),
    }
    conc = concentration(pairs, pc)
    out, tabs = {}, {}
    for v, ev in evs.items():
        snap = A.snapshots(ev, eval_ns, m)
        fl = SEL.ebh_over_time(snap, lc["alpha"], m)
        tabs[v] = flags_table(fl, snap, eval_ns) if fl.any() else pd.DataFrame()
        nm = MT.null_metrics(fl, eval_ns)
        out[v] = {
            "ever": bool(fl.any()),
            "distinct": int(fl.any(axis=0).sum()),
            "flags_per_day": nm["flags_per_day"],
            "first_flag_any": str(pd.Timestamp(int(eval_ns[fl.any(axis=1).argmax()])))
            if fl.any()
            else None,
        }
    rn = (
        tabs["Rn1"].merge(conc, left_on="node", right_index=True, how="left")
        if len(tabs["Rn1"])
        else tabs["Rn1"]
    )
    v1t = tabs["v1"].merge(conc, left_on="node", right_index=True, how="left")
    rset, vset, p2 = set(rn["node"]) if len(rn) else set(), set(v1t["node"]), set(PHASE2_LIST)
    s = conc["top_user_share"]
    summary = {
        **{f"{k}": v for k, v in out.items()},
        "overlap_Rn1_phase2": sorted(rset & p2),
        "overlap_Rn1_v1": sorted(rset & vset),
        "overlap_v1_phase2": sorted(vset & p2),
        "Rn1_nodes": sorted(rset),
        "v1_nodes": sorted(vset),
        "all_nodes_top_user_share": {
            "nodes": int(len(s)),
            "q10": float(s.quantile(0.1)),
            "q50": float(s.quantile(0.5)),
            "q90": float(s.quantile(0.9)),
            "q99": float(s.quantile(0.99)),
            "max": float(s.max()),
        },
        "Rn1_flags": rn.to_dict("records"),
        "v1_flags": v1t.to_dict("records"),
        "phase2_list_top_user_share": conc.reindex(PHASE2_LIST).reset_index().to_dict("records"),
    }
    run.write_json("metrics.json", summary)
    pd.set_option("display.width", 200)
    print(
        json.dumps(
            {k: v for k, v in summary.items() if "flags" not in k and "phase2_list" not in k},
            indent=0,
        )
    )
    print("Rn1\n", rn.round(3).to_string(index=False))
    print("v1\n", v1t.round(3).to_string(index=False))
    print(conc.reindex(PHASE2_LIST).round(3).to_string())
    run.finish("ok", {"Rn1_flagged": len(rset)})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    return run_guarded(start_run("phase4", a.config, tag="d84"), lambda r: _run(r, a.config))


if __name__ == "__main__":
    sys.exit(main())
