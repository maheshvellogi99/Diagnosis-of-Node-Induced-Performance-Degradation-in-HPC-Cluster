"""Phase 6 analysis (D113-D115): campaign logs -> D113 inputs, empirical pool, Gate 6 numbers.

    uv run python scripts/phase6_analyze.py --config configs/phase5.yaml --logs cluster/logs/*.jsonl

Writes runs/phase6/<run_id>/: d113_inputs.yaml (only D113 fields; k entries that cannot be estimated
keep their frozen value and are listed), healthy_pool.csv, gate6.json. The procedure is fixed in
src/hpcdiag/phase6/analysis.py and docs/PHASE6_PROTOCOL.md §5. --out writes elsewhere (tests).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from hpcdiag.phase6 import analysis as AN
from hpcdiag.utils.run import load_config, start_run


def analyse(cfg: dict, logs: list[str], out: Path) -> dict:
    df = AN.load_logs(logs)
    ft = AN.fit(df)
    m, kept = AN.model_from_fit(ft, cfg["probes"])
    pc = cfg["policy"]
    ho = AN.held_out(df, ft, m, pc["s_grid"], pc["pi0_h0"])
    sc = load_config("configs/phase3_inject.yaml")["kinds"]["node"]
    s_all = list(sc["s"]) + list(sc.get("s_extra", []))
    g = AN.gate6(df, ft, ho, (min(s_all), max(s_all)))
    pool = out / "healthy_pool.csv"
    AN.write_pool(ft, pool)
    d = AN.d113_inputs(ft, m, kept, str(pool))
    (out / "d113_inputs.yaml").write_text(yaml.safe_dump(d, sort_keys=False))
    (out / "gate6.json").write_text(json.dumps(g, indent=1, default=str))
    return g


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--out", help="output folder instead of a run directory")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        g = analyse(cfg, a.logs, out)
    else:
        run = start_run("phase6", a.config, tag="analysis")
        g = analyse(cfg, a.logs, run.dir)
        run.finish("ok", {"hosts": len(g["hosts"]), "days": len(g["days"])})
    print(json.dumps(g["completeness"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
