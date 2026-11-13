"""Campaign plans for the measurement kit (D114, D115). Standard library only.

    python3 cluster/kit_campaign.py --config cluster/campaign.json --kind healthy   --seed 1 --out plan.json
    python3 cluster/kit_campaign.py --config cluster/campaign.json --kind injection --seed 2 --out plan.json
    python3 cluster/kit_campaign.py --config cluster/campaign.json --kind benchmark --seed 3 --out plan.json

healthy:   n_batteries batteries of every probe (probe order shuffled per battery).
injection: {anomalies} x {intensities} x reps anomaly steps in random order, each with a battery of every
           probe (shuffled) while the anomaly runs.
benchmark: for each anomaly x intensity x rep, one benchmark with the anomaly and one without, in
           random order.
Plans for one day are generated per machine and per day (seed = day number x 100 + machine number);
the seeds are listed in the campaign log.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def healthy_plan(n_batteries: int, probes: list[str], seed: int) -> list[dict]:
    rng = random.Random(seed)
    return [{"type": "battery", "condition": "healthy", "block": b,
             "probes": rng.sample(probes, len(probes))} for b in range(n_batteries)]  # fmt: skip


def injection_plan(anomalies, intensities, reps: int, probes: list[str], seed: int) -> list[dict]:
    rng = random.Random(seed)
    steps = [{"type": "anomaly", "condition": "anomaly", "anomaly": a, "intensity": i, "rep": r}
             for a in anomalies for i in intensities for r in range(reps)]  # fmt: skip
    rng.shuffle(steps)
    for s in steps:
        s["probes"] = rng.sample(probes, len(probes))
    return steps


def benchmark_plan(anomalies, intensities, reps: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    steps = []
    for a in anomalies:
        for i in intensities:
            for r in range(reps):
                steps.append({"type": "benchmark", "condition": "anomaly", "anomaly": a,
                              "intensity": i, "rep": r})  # fmt: skip
                steps.append({"type": "benchmark", "condition": "healthy", "anomaly": None,
                              "intensity": i, "rep": r, "paired_with": a})  # fmt: skip
    rng.shuffle(steps)
    return steps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--kind", choices=["healthy", "injection", "benchmark"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument(
        "--n", type=int, help="healthy batteries (default: config healthy_batteries_per_day)"
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    c = json.loads(Path(a.config).read_text())
    if a.kind == "healthy":
        plan = healthy_plan(a.n or c["healthy_batteries_per_day"], c["probes"], a.seed)
    elif a.kind == "injection":
        plan = injection_plan(
            c["anomaly_names"], c["intensities"], c["reps_per_day"], c["probes"], a.seed
        )
    else:
        plan = benchmark_plan(c["anomaly_names"], c["intensities"], c["reps_per_day"], a.seed)
    Path(a.out).write_text(json.dumps(plan, indent=1))
    print(f"{len(plan)} steps -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
