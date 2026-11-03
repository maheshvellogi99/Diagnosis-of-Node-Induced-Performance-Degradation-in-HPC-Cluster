"""D99: Prodigy artifact, secondary binary check (healthy vs anomalous), reported only.

    uv run python scripts/phase5_prodigy_check.py --config configs/phase5.yaml

One sample = one (job, node) series at 1 Hz. Features: the 11 statistics of the Antarex classifier
over the whole series of each metric (counters differenced). Random Forest, same settings as the
Antarex classifier. Splits: grouped 5-fold by job (a job never in both train and test), and leave
one application out (transfer). The Prodigy artifact's own method (healthy-only VAE) is not
reproduced; its published macro-F1 is context only (F25).
"""

from __future__ import annotations

import argparse
import sys

import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from hpcdiag.io.safe_pickle import loads
from hpcdiag.layer3 import classifier as C
from hpcdiag.utils.run import Run, resolve, run_guarded, start_run


def read(path, key):
    """Metric matrix, metric names and (job_id, component_id) per row, without pandas/PyTables.
    The id block is a pickled numpy object array: read with the restricted unpickler."""
    with h5py.File(path, "r") as h:
        g = h[key]
        names = [x.decode() for x in g["block0_items"][:]]
        vals = g["block0_values"][:]
        ids_names = [x.decode() for x in g["block1_items"][:]]
        ids = loads(g["block1_values"][0].tobytes())
    ids = np.asarray(ids)
    if ids.shape[0] != vals.shape[0]:
        ids = ids.T
    if ids_names != ["job_id", "component_id"] or ids.shape != (vals.shape[0], 2):
        raise ValueError(f"unexpected id block {ids_names} {ids.shape}")
    return vals, names, ids


def samples(cfg: dict):
    pc = cfg["prodigy"]
    d = resolve(pc["dir"])
    series, labels = [], []
    names_ref = None
    for hdf, key, lab in pc["files"]:
        vals, names, ids = read(d / hdf, key)
        if names_ref is None:
            names_ref = names
        if names != names_ref:
            raise ValueError("metric columns differ between files")
        lb = pd.read_csv(d / lab, dtype={"job_id": str, "component_id": str})
        df = pd.DataFrame({"job": ids[:, 0].astype(str), "node": ids[:, 1].astype(str)})
        for (j, n), idx in df.groupby(["job", "node"], sort=False).indices.items():
            row = lb[(lb["job_id"] == j) & (lb["component_id"] == n)]
            if len(row) != 1:
                raise ValueError(f"label rows for {j}/{n}: {len(row)}")
            x = vals[idx]
            x = x[np.argsort(x[:, names.index("timestamp")], kind="stable")]
            series.append(x)
            labels.append({**row.iloc[0].to_dict(), "file": lab, "n_seconds": len(x)})
    return series, pd.DataFrame(labels), names_ref


def features(series, names, share):
    keep = [i for i, n in enumerate(names) if n != "timestamp"]
    cum = []
    for i in keep:
        c = np.mean([C.is_cumulative(s[:, i].astype(float)) for s in series])
        cum.append(c >= share)
    X = []
    for s in series:
        x = s[:, keep].astype(float)
        for j, c in enumerate(cum):
            if c:
                x[:, j] = np.r_[0.0, np.diff(x[:, j])]
        X.append(C.window_stats(x[None])[0])
    return np.vstack(X), int(np.sum(cum)), len(keep)


def rf(rc, seed):
    return RandomForestClassifier(
        n_estimators=rc["n_estimators"],
        class_weight=rc["class_weight"],
        n_jobs=rc["n_jobs"],
        random_state=seed,
    )


def _run(run: Run) -> int:
    cfg = run.config
    rc, seed = cfg["antarex"]["rf"], cfg["seed"]
    series, lab, names = samples(cfg)
    X, n_cum, n_metrics = features(series, names, cfg["prodigy"]["cumulative_share"])
    y = lab["binary_anom"].to_numpy()
    groups = lab["job_id"].to_numpy()
    counts = lab.groupby(["file", "anom_name"]).size().to_dict()
    print("samples", len(y), "metrics", n_metrics, "differenced", n_cum, counts)
    pred = np.full(len(y), -1)
    cv = StratifiedGroupKFold(n_splits=cfg["prodigy"]["n_folds"], shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(cv.split(X, y, groups)):
        if set(groups[tr]) & set(groups[te]):
            raise RuntimeError("job in train and test")
        pred[te] = rf(rc, seed + k).fit(X[tr], y[tr]).predict(X[te])
    grouped = {
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "confusion": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "recall_by_anomaly": {
            a: float((pred[lab["anom_name"] == a] == y[lab["anom_name"] == a]).mean())
            for a in sorted(lab["anom_name"].unique())
        },
    }
    print("grouped by job", grouped)
    loao = []
    for app in sorted(lab["app_name"].unique()):
        te = (lab["app_name"] == app).to_numpy()
        p = rf(rc, seed).fit(X[~te], y[~te]).predict(X[te])
        loao.append(
            {
                "held_out_app": app,
                "n": int(te.sum()),
                "macro_f1": float(f1_score(y[te], p, average="macro", labels=[0, 1])),
                "accuracy": float((p == y[te]).mean()),
            }
        )
        print(loao[-1])
    run.write_json(
        "metrics.json",
        {
            "n_samples": len(y),
            "n_metrics": n_metrics,
            "n_differenced": n_cum,
            "counts": {f"{a}|{b}": v for (a, b), v in counts.items()},
            "seconds_per_sample_median": float(lab["n_seconds"].median()),
            "grouped_by_job": grouped,
            "leave_one_app_out": loao,
        },
    )
    run.finish("ok", {"macro_f1": round(grouped["macro_f1"], 4)})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    return run_guarded(start_run("phase5", a.config, tag="prodigy-check"), _run)


if __name__ == "__main__":
    sys.exit(main())
