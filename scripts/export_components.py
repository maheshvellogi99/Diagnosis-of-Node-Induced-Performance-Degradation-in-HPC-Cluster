"""D121(b,c): export the Layer 2 and Layer 3 components of the frozen primary.

    uv run python scripts/export_components.py --config configs/phase5.yaml --out deliverables/models

Everything is rebuilt with the functions the runners use, from the frozen configs:
- Layer 2 (configs/phase4_layer2.yaml): the D61 shift-LR bet tables g_s0 on the centred score
  ("Sc", the bets the primary uses: run_phase5.layer2) and the Layer 2 settings.
- Layer 3 (configs/phase5.yaml): probe model and policy settings, the D96 state bets (G, P), the
  state table, the design healthy pools (fixed pool, seed evalue.pool_seed), the telemetry counts
  and the likelihood the primary reads (cross-block, add-one smoothed).
- The Antarex telemetry random forest (D94 settings, seed 0), fit on ALL 5-class windows. Its
  reported scores come from runs 20261102-003936-classifier (out-of-fold) and
  20261102-004249-d94-diag (cross-block); this exported model was not scored on held-out data.
Writes runs/phaseR/<id>/metrics.json with the SHA-256 of every exported file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier

from hpcdiag.layer2 import run_layer2 as R
from hpcdiag.layer3 import classifier as CL
from hpcdiag.layer3 import probes as PR
from hpcdiag.layer3 import setup as ST
from hpcdiag.utils.run import Run, load_config, resolve, run_guarded, start_run

CLF_RUNS = ("20261102-003936-classifier", "20261102-004249-d94-diag")


def layer2(out: Path, l2_config: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    R._init(l2_config)
    cfg = R.G["cfg"]
    bets = R.G["bets"]["Sc"]
    b = len(bets[0].density)
    df = pd.DataFrame({"p_lo": np.arange(b) / b, "p_hi": np.arange(1, b + 1) / b})
    for hb in bets:
        df[hb.label] = hb.density
    df.to_csv(out / "layer2_shift_bets_Sc.csv", index=False, float_format="%.17g")
    meta = {hb.label: {"warmup_jobs": hb.n} for hb in bets}
    keep = {k: cfg[k] for k in ("layer1", "layer2", "shift_bets", "theta_seed") if k in cfg}
    keep["primary_bets"] = "Sc (centred score), shift-LR histogram bets, D61"
    keep["warmup_jobs_per_bet"] = meta
    (out / "layer2_settings.yaml").write_text(yaml.safe_dump(keep, sort_keys=False))


def layer3(out: Path, cfg: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    v = ST.VARIANTS["main"]
    m = ST.policy_model(cfg["probes"], v)
    bel = ST.make_belief(cfg, v)
    names = list(m.names)
    keep = {k: cfg[k] for k in ("seed", "probes", "evalue", "policy", "telemetry")}
    keep["primary_policy"] = {"name": "ig_d109", "nodes": "rank", "probes": "ig_cost",
                              "exclusion": "cleared_skipdone", "probe_J": cfg["evalue"]["probe_J"]}  # fmt: skip
    (out / "layer3_settings.yaml").write_text(yaml.safe_dump(keep, sort_keys=False))
    means, hyp = PR.state_table(m, cfg["policy"]["s_grid"])
    st = pd.DataFrame(means, columns=names)
    st.insert(0, "hypothesis", [PR.HYPOTHESES[h] for h in hyp])
    st.insert(1, "s", [None] + [s for _ in PR.CAUSES for s in cfg["policy"]["s_grid"]])
    st.to_csv(out / "layer3_state_table.csv", index=False, float_format="%.17g")
    edges = bel.bets.edges
    rows = []
    for i, a in enumerate(names):
        for j in range(bel.bets.G.shape[1]):
            for k in range(len(edges) - 1):
                rows.append(
                    (
                        a,
                        int(j + 1),
                        edges[k],
                        edges[k + 1],
                        bel.bets.G[i, j, k],
                        bel.bets.P[i, j + 1, k],
                    )
                )
    pd.DataFrame(rows, columns=["probe", "state", "p_lo", "p_hi", "g_density", "P_bin"]).to_csv(
        out / "layer3_state_bets.csv", index=False, float_format="%.17g"
    )
    h0 = pd.DataFrame({"probe": np.repeat(names, len(edges) - 1),
                       "p_lo": np.tile(edges[:-1], len(names)), "p_hi": np.tile(edges[1:], len(names)),
                       "P_bin_h0": bel.bets.P[:, 0, :].ravel()})  # fmt: skip
    h0.to_csv(out / "layer3_h0_bin_probs.csv", index=False, float_format="%.17g")
    n_pool, ec = cfg["evalue"]["n_pool"], cfg["evalue"]
    pools = ST.fixed_pools(m.names, m.sigma, n_pool, [ec["pool_seed"], n_pool])
    pd.DataFrame({a: p for a, p in zip(names, pools, strict=True)}).to_csv(
        out / "layer3_design_healthy_pool.csv", index=False, float_format="%.17g"
    )
    for which in ("crossblock", "oof"):
        lik = pd.DataFrame(
            ST.telemetry_rows(cfg, which), index=PR.HYPOTHESES, columns=PR.HYPOTHESES
        )
        lik.to_csv(out / f"layer3_telemetry_likelihood_{which}.csv", float_format="%.17g")
    cnt = {"crossblock_counts (primary, D98)": cfg["telemetry"]["crossblock_counts"],
           "counts (out-of-fold, sensitivity)": cfg["telemetry"]["counts"],
           "rows/columns": list(PR.HYPOTHESES)}  # fmt: skip
    (out / "layer3_telemetry_counts.json").write_text(json.dumps(cnt, indent=1))


def telemetry_rf(out: Path, cfg: dict, log) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rc, seed = cfg["antarex"]["rf"], cfg["seed"]
    X, meta, names = CL.build_features(cfg, log)
    main = meta["fault"].isin(list(CL.CLASS_MAP)).to_numpy()
    y = meta[main]["fault"].map(CL.CLASS_MAP).to_numpy(dtype=str)
    rf = RandomForestClassifier(n_estimators=rc["n_estimators"], class_weight=rc["class_weight"],
                                n_jobs=rc["n_jobs"], random_state=seed)  # fmt: skip
    rf.fit(X[main], y)
    cols = [f"{m}|{s}" for m in names for s in CL.STAT_NAMES]  # column order of window_stats
    assert len(cols) == X.shape[1], (len(cols), X.shape)
    joblib.dump({"model": rf, "feature_names": cols, "metric_names": list(names), "stat_names": list(CL.STAT_NAMES), "classes": list(rf.classes_),
                 "class_map": CL.CLASS_MAP, "fit_on": "all 5-class Antarex windows (mono + multi)",
                 "scored_on_held_out": False}, out / "telemetry_rf_antarex.joblib")  # fmt: skip
    for r in CLF_RUNS:
        shutil.copy(resolve(f"runs/phase5/{r}/metrics.json"), out / f"metrics_{r}.json")
    return {"windows": int(main.sum()), "metrics": len(names), "features": int(X.shape[1])}


def _run(run: Run, out: Path) -> int:
    cfg = run.config
    layer2(out / "layer2", cfg["layer2_config"])
    layer3(out / "layer3", cfg)
    rf = telemetry_rf(out / "telemetry_rf", load_config("configs/phase5.yaml"), run.log)
    sums = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out.rglob("*")) if p.is_file() and "layer1" not in p.parts}  # fmt: skip
    run.write_json("metrics.json", {"rf": rf, "sha256": sums})
    print(json.dumps({"rf": rf, "files": len(sums)}))
    run.finish("ok", {"files": len(sums)})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    return run_guarded(
        start_run("phaseR", a.config, tag="components"), lambda r: _run(r, resolve(a.out))
    )


if __name__ == "__main__":
    sys.exit(main())
