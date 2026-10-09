# CLAUDE.md — Project rules for Claude Code

This file is read automatically by Claude Code when it starts in this folder.
The full design lives in `docs/PLAN.md`. Phase prompts live in `docs/PHASE_PROMPTS.md`.

## Project in one line
Three-layer system that turns Slurm job logs into statistically guaranteed evidence of
slow (degraded) compute nodes, then diagnoses the cause with the fewest tests.
Layer 1 = conformal "is this job late?" · Layer 2 = e-process + e-BH "which node?" ·
Layer 3 = cost-aware probing "what is wrong?".

## Your role
You are the senior software developer. The human (Mahesh) is the researcher and owner.
The architecture is fixed in `docs/PLAN.md`. Implement it; do not redesign it silently.

## Non-negotiable rules
1. **Never invent numbers, results, citations, column names or dataset facts.**
   Every number in a report must come from code that ran in this repo, with the command shown.
   If something is unknown, write `UNKNOWN — needs check` and stop to ask.
2. **Work one phase at a time.** Only do the phase you were asked to do. At the end of a phase,
   write `reports/phase_<N>.md` and STOP. Do not start the next phase on your own.
3. **Raw data is read-only.** Never modify, move or delete anything under `data/raw/`.
   All derived data goes to `data/interim/` or `data/processed/`.
4. **Ask before**: installing a new package, deleting any file, changing a formula in PLAN.md,
   changing a gate (pass/fail threshold), running anything longer than ~30 minutes,
   or touching anything outside this project folder.
5. **No network calls from project code**, except `scripts/download_data.py`, which must verify
   checksums. No telemetry, no uploading data anywhere.
6. **Reproducibility**: every script takes a config file (`configs/*.yaml`), sets seeds,
   and writes outputs with a run id under `runs/<phase>/<run_id>/` plus `config_used.yaml`
   and `git_commit.txt`.
7. **Tests first for math.** Every formula in `src/hpcdiag/stats/` has unit tests in `tests/`
   that check it against a hand-computed or simulated case before it is used on real data.
8. **Time order is sacred.** No random train/test splits on job data. A model or calibration set
   may only use jobs that *ended before* the job being scored. Add a test that checks this.
9. **Commit small and often** with clear messages. Never commit files from `data/` or `runs/`.
10. **When a gate fails, report it honestly.** Do not tune on the test set to make it pass.

## Conventions
- Python 3.11, package manager `uv`. Code in `src/hpcdiag/`. Tests with `pytest`.
- Formatting/lint: `ruff`. Type hints on public functions.
- Data frames: `pandas` with `pyarrow`; use `duckdb` or column selection for large files.
- Times are UTC-naive timestamps as stored in the datasets. Runtimes in **seconds**.
- Log-runtime means `log(max(run_time, 1))` unless PLAN.md says otherwise.

## Useful commands
- `uv run pytest -q` — run tests
- `uv run ruff check .` — lint
- `uv run python -m hpcdiag.<module> --config configs/<file>.yaml` — run a stage
