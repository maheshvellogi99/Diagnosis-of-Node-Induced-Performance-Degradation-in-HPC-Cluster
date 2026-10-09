# hpcdiag — From Job Evidence to Root Cause

Three-layer system that turns Slurm job logs into statistically guaranteed evidence of slow
(degraded) compute nodes, then diagnoses the cause with the fewest tests.

- Layer 1 — conformal "is this job late?"
- Layer 2 — node e-processes + e-BH "which node?"
- Layer 3 — cost-aware probing "what is wrong?"

Design: `docs/PLAN.md`. Rules for the coding agent: `CLAUDE.md`. Phase reports: `reports/`.

## Setup

```bash
brew install uv            # once
uv python install 3.11     # once
uv sync                    # creates .venv from uv.lock
uv run pytest -q
```

## Data

Raw files (read-only) live in `data/raw/<dataset>/` and are git-ignored:

- `data/raw/eagle/eagle_data.parquet` — NREL Eagle jobs, OEDI submission 5860, CC-BY 4.0
- `data/raw/pm100/job_table.parquet` — PM100 (Marconi100), Zenodo record 10127767, CC-BY 4.0

Checksums: `uv run python scripts/download_data.py --config configs/phase0_data.yaml`
(writes `data/raw/CHECKSUMS.txt`; `--verify` checks them). It does not download anything.

## Phase 0 commands

```bash
uv run python -m hpcdiag.io.build --config configs/phase0_data.yaml     # raw -> data/processed
uv run python -m hpcdiag.io.profile --config configs/phase0_data.yaml   # metrics.json + data cards
uv run pytest -q
```
