# Phase 6-prep part 2 report (D116–D120)

14 Nov 2026. Development Mac only. Not done: no hardware work, no installs, no network calls except the
requested `git push`, and no run of Phase 5F. Sealed seeds 20–39 and nulls 200–399 were not touched. All
460 tests pass (`uv run pytest -q`), `ruff check` is clean, and the guard passes
(`PASS: 6 frozen files checked, policy 'ig_d109'`).

| Decision | Done | Evidence |
|---|---|---|
| D116 | The 40 formatting-only files are restored byte for byte from `v1.0-dev-frozen`; the 5 layer3 files are kept. New guard check (d): `src/` may differ from the tag only in `layer3/`, `freeze.py` and `phase6/`, covering committed, uncommitted and untracked files. | `git diff --stat v1.0-dev-frozen HEAD -- src` lists only those paths. Plumbing check 25/25 (run 20261114-004734-plumbing-check). 49 guard tests. |
| D117 | `telemetry.crossblock_counts` replaces `telemetry.counts` as the D113 field. Phase 6 telemetry classifier: per-second counter rates, the D94 random forest, leave-one-fit-day-out confusion matrix on fit days only, which replaces the frozen matrix only if every class has ≥ 30 fit-day samples. The tel probe records snapshot times. | `tests/test_phase6_analysis.py` (synthetic logs): matrix rows h0/cpu/cache/mbw/mem; test days never used; frozen matrix kept when a class has fewer than 30 samples |
| D118 | PROMPT 6 steps 2–3 are marked superseded; the protocol states the bin edges stay frozen and there are ≥ 1000 healthy runs per probe type. | `docs/PHASE_PROMPTS_5_6.md`, `docs/PHASE6_PROTOCOL.md` |
| D119(a) | The campaign must give ≥ 1000 fit-day and ≥ 300 test-day healthy runs per probe. `kit_safety.check_campaign_size` runs in the harness preflight. The plan is now 2 machines × 3 days × 250 batteries, giving 1000 fit-day and 500 test-day runs. The analysis writes the measured fit/test runs. | Estimate run 20261114-010029-estimate (protocol §7): 12.14 h per machine-day and 72.87 machine-h for 2 machines × 3 days; 7.94 h and 95.32 machine-h for 4 × 3. 18 new kit tests. |
| D119(b) | Measured tables in the simulator. `World.draw()`: Gaussian is the unchanged call; measured = model mean + resampled test-day healthy log x. Fixed pool = first 1000 fit-day runs. Pool per base = fit-day runs resampled with replacement. `cross_machine` variant. Misspecified row refused. Exact healthy-growth diagnostic. | 9 tests (`tests/test_d119.py`), written first. Gaussian path 25/25 identical (run 20261114-005516-plumbing-check). It fitted the simulator cleanly; no STOP needed. |
| D120 | `docs/tierB_machines.md` (empty template); protocol 4.0 and the kit README now require verifying the Triad parser on real STREAM output on the target. | docs |

**Disclosures:**
- **Superseded estimate run.** Run 20261114-010005-estimate is superseded. My rewrite of the script had
  failed to save, so it ran the old script with the new 250-battery setting. Run 20261114-010029 is the
  D119 table.
- **Estimate script location.** `scripts/phase6_estimate.py` imports `cluster/kit_safety.py`, so the kit
  and the estimate share one formula.

**Open, needs your confirmation (protocol §10):**
- **Cross-machine row:** one direction only (first host's pool, second host's outcomes), with the pool
  resampled with replacement.
- **Still `UNKNOWN`:** HPAS options and commit, the single-node benchmark and the MPI benchmark.

Commands: see protocol §7 and §8; `uv run python scripts/check_frozen.py --config configs/phase5.yaml --policy ig_d109`.
