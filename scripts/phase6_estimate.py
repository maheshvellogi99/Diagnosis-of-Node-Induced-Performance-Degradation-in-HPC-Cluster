"""D114/D119a wall-clock estimate per tier from planned probe costs and kit settings.

    uv run python scripts/phase6_estimate.py --config configs/phase5.yaml \
        --kit cluster/campaign.json

Battery = every probe once (configs/phase5.yaml probes.cost_s plus policy.tel_cost_s). For M
machines and D days, healthy batteries per machine and day = the D119a minimum
(cluster/kit_safety.min_batteries_per_day: >= 1000 fit-day and >= 300 test-day healthy runs per
probe, fit days = the first ceil(2D/3)). Per machine and day also reps_per_day x anomalies x
intensities anomaly repetitions (warm-up + battery + cool-down each) and as many benchmark pairs
(with the anomaly: warm-up + benchmark + cool-down; without: benchmark). Writes
runs/phase6prep/<id>/estimate.json.
"""

from __future__ import annotations

import argparse
import json
import sys

from hpcdiag.utils.run import PROJECT_ROOT, Run, run_guarded, start_run

sys.path.insert(0, str(PROJECT_ROOT / "cluster"))
import kit_safety as KS  # noqa: E402

REPS_MIN = 10
GRID = [(m, d) for m in (2, 3, 4) for d in (3, 6)]


def estimate(cfg: dict, kit: dict, machines: int, days: int) -> dict:
    battery = sum(float(v) for v in cfg["probes"]["cost_s"].values()) + float(
        cfg["policy"]["tel_cost_s"]
    )
    w, c, b = float(kit["warmup_s"]), float(kit["cooldown_s"]), float(kit["benchmark_planned_s"])
    cells = len(kit["anomaly_names"]) * len(kit["intensities"])
    reps = int(kit["reps_per_day"])
    nb = KS.min_batteries_per_day(machines, days)
    healthy_day = nb * battery
    inject_day = cells * reps * (w + battery + c)
    bench_day = cells * reps * ((w + b + c) + b)
    per_day = healthy_day + inject_day + bench_day
    size = KS.campaign_size(machines, days, nb)
    return {
        "machines": machines,
        "days": days,
        "battery_s": battery,
        "healthy_batteries_per_machine_day": nb,
        "fit_runs_per_probe": size["fit_runs_per_probe"],
        "test_runs_per_probe": size["test_runs_per_probe"],
        "healthy_h_per_machine_day": healthy_day / 3600,
        "injection_h_per_machine_day": inject_day / 3600,
        "benchmark_h_per_machine_day": bench_day / 3600,
        "total_h_per_machine_day": per_day / 3600,
        "total_machine_h": per_day * machines * days / 3600,
        "reps_per_anomaly_intensity": reps * machines * days,
        "reps_ok": reps * machines * days >= REPS_MIN,
    }


def _run(run: Run, kit_path: str) -> int:
    kit = json.loads((PROJECT_ROOT / kit_path).read_text())
    rows = [estimate(run.config, kit, m, d) for m, d in GRID]
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
