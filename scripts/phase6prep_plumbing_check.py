"""D113 plumbing check: with scalar sigma and the Gaussian pool, the code after the D113 changes
must reproduce the dev5c runs bit for bit. K 1%, s 1.3, B 7200, seeds 0-4, and the first 20 nulls
(development only), policy ig_d109, variant main. Every field is compared except wall-clock
timings (layer2_s, sim_s, total_s). Exit 1 and STOP if anything differs.

    uv run python scripts/phase6prep_plumbing_check.py --config configs/phase5.yaml \
        --ref 20261113-223241-dev5c
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from hpcdiag.layer3 import run_phase5 as RP
from hpcdiag.utils.run import Run, resolve, run_guarded, start_run

TIMING = {"layer2_s", "sim_s", "total_s"}


def canon(r: dict) -> str:
    return json.dumps({k: v for k, v in r.items() if k not in TIMING}, sort_keys=True, default=str)


def _run(run: Run, config_path: str, ref: str) -> int:
    idx = pd.read_parquet(resolve("data/processed/scenarios/index.parquet"))
    rand = idx[idx["base"] != "real"]
    node = rand[
        (rand["kind"] == "node") & (rand["K_pct"] == 1) & (rand["s"] == 1.3) & (rand["seed"] <= 4)
    ]
    nulls = sorted(rand[rand["kind"] == "null"]["scenario_id"])[:20]
    sids = sorted(node["scenario_id"]) + nulls
    RP.check_sealed(sids)
    with open(resolve(f"runs/phase5/{ref}/results.jsonl")) as fh:
        refs = {
            r["scenario_id"]: r
            for r in map(json.loads, fh)
            if r["policy"] == "ig_d109"
            and r["B"] == 7200
            and r["variant"] == "main"
            and r.get("k_pre", 0) == 0
        }
    RP._init(config_path)
    rows, diff = [], []
    for sid in sids:
        out = RP._process((sid, [("main", "ig_d109", 7200)]))[0]
        out = json.loads(json.dumps(out, default=str))  # same serialisation as results.jsonl
        same = canon(out) == canon(refs[sid])
        rows.append(
            {"scenario_id": sid, "identical": same, "n_investigations": out["n_investigations"]}
        )
        if not same:
            diff.append(sid)
        run.log.info("%s identical=%s", sid, same)
    pd.DataFrame(rows).to_csv(run.dir / "comparison.csv", index=False)
    run.write_json(
        "metrics.json", {"n": len(sids), "identical": len(sids) - len(diff), "differ": diff}
    )
    print(f"{len(sids) - len(diff)} of {len(sids)} runs identical to {ref}")
    if diff:
        print("STOP: plumbing changed results:", diff)
        run.finish("differ", {"differ": len(diff)})
        return 1
    run.finish("ok", {"identical": len(sids)})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ref", required=True)
    a = ap.parse_args()
    return run_guarded(
        start_run("phase6prep", a.config, tag="plumbing-check"), lambda r: _run(r, a.config, a.ref)
    )


if __name__ == "__main__":
    sys.exit(main())
