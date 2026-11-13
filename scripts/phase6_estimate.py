"""D114 wall-clock estimate per tier from planned probe costs and kit settings. No measurement.

    uv run python scripts/phase6_estimate.py --config configs/phase5.yaml \
        --kit cluster/campaign.json

Battery = every probe once (configs/phase5.yaml probes.cost_s plus policy.tel_cost_s). Per machine
and campaign day: healthy_batteries_per_day batteries; reps_per_day x anomalies x intensities
anomaly repetitions (warm-up + battery + cool-down each); the same number of benchmark pairs (with
the anomaly: warm-up + benchmark + cool-down; without: benchmark). Writes
runs/phase6prep/<id>/estimate.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

from hpcdiag.utils.run import Run, run_guarded, start_run

DAYS = 3
HEALTHY_MIN = 1000
REPS_MIN = 10


def estimate(cfg: dict, kit: dict, machines: int, days: int = DAYS) -> dict:
    battery = sum(float(v) for v in cfg["probes"]["cost_s"].values()) + float(
        cfg["policy"]["tel_cost_s"]
    )
    w, c, b = float(kit["warmup_s"]), float(kit["cooldown_s"]), float(kit["benchmark_planned_s"])
    cells = len(kit["anomaly_names"]) * len(kit["intensities"])
    reps = int(kit["reps_per_day"])
    healthy_day = int(kit["healthy_batteries_per_day"]) * battery
    inject_day = cells * reps * (w + battery + c)
    bench_day = cells * reps * ((w + b + c) + b)
    per_day = healthy_day + inject_day + bench_day
    return {
        "machines": machines,
        "days": days,
        "battery_s": battery,
        "healthy_h_per_machine_day": healthy_day / 3600,
        "injection_h_per_machine_day": inject_day / 3600,
        "benchmark_h_per_machine_day": bench_day / 3600,
        "total_h_per_machine_day": per_day / 3600,
        "total_machine_h": per_day * machines * days / 3600,
        "healthy_runs_per_probe": int(kit["healthy_batteries_per_day"]) * machines * days,
        "healthy_ok": int(kit["healthy_batteries_per_day"]) * machines * days >= HEALTHY_MIN,
        "reps_per_anomaly_intensity": reps * machines * days,
        "reps_ok": reps * machines * days >= REPS_MIN,
        "min_healthy_machine_h": HEALTHY_MIN * battery / 3600,
        "min_days_if_24h": math.ceil(per_day / 86400),
    }


def _run(run: Run, kit_path: str) -> int:
    kit = json.loads(open(kit_path).read())
    rows = [estimate(run.config, kit, m) for m in (2, 3, 4)]
    run.write_json("estimate.json", {"kit": kit_path, "rows": rows})
    for r in rows:
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}))
    run.finish("ok", {"battery_s": rows[0]["battery_s"]})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--kit", required=True)
    a = ap.parse_args()
    return run_guarded(start_run("phase6prep", a.config, tag="estimate"), lambda r: _run(r, a.kit))


if __name__ == "__main__":
    sys.exit(main())
