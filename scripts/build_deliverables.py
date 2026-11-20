"""D125: assemble deliverables/ for Google Drive (git-ignored; rebuilt by this script).

    uv run python scripts/build_deliverables.py

Run export_layer1.py, export_components.py and demo_pipeline.py first: they write deliverables/models
and deliverables/demo. This script copies the report, the architecture figure, FINAL_RESULTS.md, the
phase reports with their figures, PLAN / decisions / FROZEN_CONFIG, and writes README.md (index),
models/README.md and datasets.md (links, DOIs, licences, checksums; NO raw data). It never reads
or copies anything under data/ except data/raw/CHECKSUMS.txt (read only). It prints the folder
size.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables"
REP = ROOT / "Claude outputs" / "report"

REPORT_FILES = ["G34_Final_Report.tex", "architecture_fig.tex", "architecture_standalone.tex",
                "Project Report.pdf", "logo/logo-000.png"]  # fmt: skip
ARCH_FILES = ["architecture.png", "architecture_standalone.pdf"]
DOCS = ["docs/PLAN.md", "docs/decisions.md", "docs/FROZEN_CONFIG.md"]

DATASETS = [
    # name, record, DOI or id, licence, raw files (CHECKSUMS.txt paths), files URL
    ("PM100 (Marconi100, CINECA)", "https://zenodo.org/records/10127767", "Zenodo record 10127767",
     "CC-BY 4.0 (record page, checked 9 Oct 2026 by Mahesh)", ["pm100/job_table.parquet"]),
    ("NREL Eagle jobs", "https://data.openei.org/submissions/5860", "OEDI submission 5860",
     "CC-BY 4.0 (PLAN.md §4)", ["eagle/eagle_data.parquet"]),
    ("Antarex HPC fault dataset", "https://zenodo.org/records/2553224", "doi:10.5281/zenodo.2553224",
     "CC-BY 4.0 (given by Mahesh, D93)", ["antarex/Antarex.zip", "antarex/Readme_Antarex_Dataset.pdf"]),
    ("Prodigy artifact (Eclipse)", "https://zenodo.org/records/8079388", "doi:10.5281/zenodo.8079388",
     "CC-BY 4.0 (given by Mahesh, D93)",
     ["prodigy/eclipse_small_prod_dataset.tar", "prodigy/fe_eclipse_tsfresh_raw_CHI_2000.json"]),
]  # fmt: skip

INDEX = """# G34 final review package (development results; sealed evaluation pending)

Built by `scripts/build_deliverables.py` from git commit {commit}. Nothing here is raw data.

| Folder / file | What |
|---|---|
| `report/` | Final report source (`G34_Final_Report.tex`, `architecture_fig.tex`, logo). `Project Report.pdf` matches the .tex: rebuilt by the architect on 20 Nov 2026 with the D124 fixes (D126). The list of fixes is in `reports/phase_R.md`. |
| `architecture/` | Architecture figure (`architecture.png`, `architecture_standalone.pdf`) |
| `FINAL_RESULTS.md` | Every result with its run id |
| `reports/` | Phase reports 0 to R, with `figures/` |
| `docs/` | `PLAN.md` (design, decision log §16), `decisions.md` (copy of §16), `FROZEN_CONFIG.md` (frozen method, tag `v1.0-dev-frozen`) |
| `models/` | Exported components; see `models/README.md` |
| `demo/` | Layers 1 to 3 end to end on one development scenario; `summary.txt` first |
| `datasets.md` | Where the four public datasets come from, licences, checksums |

Code: the git repository (tag `v1.0-dev-frozen` for the frozen method, `main` for this package).
"""

MODELS = """# models/ — exported components (D121)

Development results; sealed evaluation pending. Load with the Python environment of the repo
(`uv sync`; LightGBM 4.7.0, scikit-learn 1.9.1, joblib 1.6.0, pandas).

**Layers 2 and 3 are statistical procedures (code + settings), not trained models.** Their files are
settings and tables that the code in `src/hpcdiag/` uses. The only trained models are the two Layer 1
LightGBM models and the telemetry random forest.

## layer1/ (trained)
| File | What |
|---|---|
| `layer1_q90.txt` | LightGBM quantile model, τ 0.9, predicts log(max(run time s, 1)); its prediction is q̂ in the score S = log T − q̂ |
| `layer1_q50.txt` | LightGBM quantile model, τ 0.5 (median), same target |
| `layer1_feature_encoding.json` | Column order, numeric columns, categorical columns with their training classes (label code = index in the sorted class list, unseen = −1), the training window |
| `layer1_export_check.json` | Re-fit check: max \\|diff\\| to the frozen predictions of run 20261010-003334-pm100-E-main-full on 102,120 jobs (0.0) |

Fitted on the 27,659 PM100 jobs that ended before 2020-06-19 15:55:59 (45-day window, seed 0, 200 trees,
learning rate 0.05, 50 leaves).

```python
import json, lightgbm as lgb, numpy as np, pandas as pd
enc = json.load(open("layer1/layer1_feature_encoding.json"))
def encode(df):
    X = df[enc["numeric"]].astype("float64").copy()
    for c in enc["categorical"]:
        cls = {v: i for i, v in enumerate(enc["classes"][c])}
        X[c] = df[c].astype(str).map(cls).fillna(enc["unseen_code"]).astype("int64")
    return X[enc["column_order"]]
q90 = lgb.Booster(model_file="layer1/layer1_q90.txt")
q_hat = q90.predict(encode(jobs))            # jobs: walltime_req_s, procs_req, mem_req, nodes_req, gpus_req, user, partition
score = np.log(np.maximum(jobs["run_time_s"], 1)) - q_hat
```
The p-value step (pooled conformal over the previous 7 days, the centring and the tie-breaker) is code:
`hpcdiag.layer2.prep` (`scenario_population`, `add_centred`, `mode_p_on`).

## layer2/ (procedure)
| File | What |
|---|---|
| `layer2_shift_bets_Sc.csv` | Bet functions g_s0(p) for s0 ∈ {{1.2, 1.3, 1.5, 2.0, 3.0}}: 20-bin histogram densities on [0, 1] (mean 1), built on the warm-up week of the centred score (D61) |
| `layer2_settings.yaml` | Layer 1 p-value settings (W = 7 d, ≥ 30 references, centring), Layer 2 settings (α 0.1, J 0.01, m 980, every 6 h), bet settings, jobs per bet |

`pd.read_csv("layer2/layer2_shift_bets_Sc.csv")`; g(p) = column value in the row with p_lo ≤ p < p_hi.

## layer3/ (procedure; assumed probe model, Phase 6 not run)
| File | What |
|---|---|
| `layer3_settings.yaml` | Probe model (k table, σ 0.05, costs), probe e-value settings (pool 1000, β 0.9, bin edges, J 0.01), policy (s grid, τ 0.95, 7-day exclusion), telemetry counts; primary policy `ig_d109` |
| `layer3_state_table.csv` | Mean of log x per probe for every (cause, s) state |
| `layer3_state_bets.csv` | D96 bets: per probe, state and p bin, the bet density g and the bin probability under that state |
| `layer3_h0_bin_probs.csv` | Bin probabilities of the conformal probe p-value under a healthy node |
| `layer3_design_healthy_pool.csv` | The fixed simulated healthy pool, 1000 log x values per probe (seed 7001) |
| `layer3_telemetry_counts.json` | Confusion counts: cross-block (used by the primary, D98) and out-of-fold |
| `layer3_telemetry_likelihood_crossblock.csv` / `_oof.csv` | P(telemetry label \\| true state), add-one smoothed, rows and columns h0, cpu, cache, mbw, mem |

## telemetry_rf/ (trained)
| File | What |
|---|---|
| `telemetry_rf_antarex.joblib` | dict: `model` (RandomForestClassifier, 300 trees, balanced, seed 0), `feature_names` (2,684 = 244 metrics × 11 statistics, order `metric|stat`), `metric_names`, `stat_names`, `classes`, `class_map` |
| `metrics_20261102-003936-classifier.json` | Out-of-fold scores (5-class macro-F1 1.0000) |
| `metrics_20261102-004249-d94-diag.json` | Cross-block scores (0.690 mono→multi, 0.854 multi→mono) |

**The exported random forest is fit on all 26,359 five-class Antarex windows. The reported scores come from
runs 20261102-003936-classifier (out-of-fold) and 20261102-004249-d94-diag (cross-block). The exported
model was not scored on held-out data.**

```python
import joblib
d = joblib.load("telemetry_rf/telemetry_rf_antarex.joblib")
label = d["model"].predict(X)   # X: (n, 2684) from hpcdiag.layer3.classifier.window_stats, same metric order
```
"""


def _copy(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def datasets_md() -> str:
    sums = {}
    for line in (ROOT / "data/raw/CHECKSUMS.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            sha, md5, size, path = line.split(maxsplit=3)
            sums[path] = (sha, md5, int(size))
    out = ["# Datasets (no raw data included)", "",
           "Download with `uv run python scripts/download_data.py --config configs/phase0_data.yaml`, which",
           "verifies these checksums. Data cards: `docs/data_cards/` in the repository.", ""]  # fmt: skip
    for name, url, ident, lic, files in DATASETS:
        out += [f"## {name}", f"- Record: {url} ({ident})", f"- Licence: {lic}", "",
                "| File | Bytes | SHA-256 | MD5 |", "|---|---|---|---|"]  # fmt: skip
        for f in files:
            sha, md5, size = sums[f]
            out.append(f"| `{f}` | {size:,} | `{sha}` | `{md5}` |")
        out.append("")
    return "\n".join(out)


def main() -> int:
    for need in ("models/layer1/layer1_q90.txt", "models/telemetry_rf/telemetry_rf_antarex.joblib",
                 "demo/summary.txt"):  # fmt: skip
        if not (OUT / need).is_file():
            print(f"missing {need}: run the export and demo scripts first")
            return 1
    for d in ("report", "architecture", "reports", "docs"):
        shutil.rmtree(OUT / d, ignore_errors=True)
    for f in REPORT_FILES:
        _copy(REP / f, OUT / "report" / f)
    for f in ARCH_FILES:
        _copy(REP / f, OUT / "architecture" / f)
    _copy(ROOT / "reports/FINAL_RESULTS.md", OUT / "FINAL_RESULTS.md")
    for f in sorted((ROOT / "reports").glob("*.md")):
        if f.name != "FINAL_RESULTS.md":
            _copy(f, OUT / "reports" / f.name)
    shutil.copytree(ROOT / "reports/figures", OUT / "reports/figures")
    for f in DOCS:
        _copy(ROOT / f, OUT / "docs" / Path(f).name)
    commit = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True, check=True).stdout.strip()  # fmt: skip
    (OUT / "README.md").write_text(INDEX.format(commit=commit))
    (OUT / "models/README.md").write_text(MODELS)
    (OUT / "datasets.md").write_text(datasets_md())
    size = subprocess.run(
        ["du", "-sh", str(OUT)], capture_output=True, text=True, check=True
    ).stdout
    files = sum(1 for p in OUT.rglob("*") if p.is_file())
    print(f"deliverables: {files} files, {size.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
