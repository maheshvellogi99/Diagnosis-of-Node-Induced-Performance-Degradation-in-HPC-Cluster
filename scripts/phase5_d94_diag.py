"""Why is the Gate 5d macro-F1 1.0? Fingerprint checks on the Antarex classifier (D94).

    uv run python scripts/phase5_d94_diag.py --config configs/phase5.yaml

(a) top metrics by Random Forest importance; (b) metric families alone; (c) drop the top metrics
repeatedly; (d) train on one block, test on the other; (e) permuted labels on contiguous stretches.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score

from hpcdiag.layer3 import classifier as C
from hpcdiag.utils.run import Run, run_guarded, start_run


def rf(rc, seed):
    return RandomForestClassifier(
        n_estimators=rc["n_estimators"],
        class_weight=rc["class_weight"],
        n_jobs=rc["n_jobs"],
        random_state=seed,
    )


def _run(run: Run) -> int:
    cfg, log = run.config, run.log
    rc, seed = dict(cfg["antarex"]["rf"]), cfg["seed"]
    X, meta, names = C.build_features(cfg, log)
    k = len(C.STAT_NAMES)
    main = meta["fault"].isin(list(C.CLASS_MAP)).to_numpy()
    X, meta = X[main], meta[main].reset_index(drop=True)
    y = meta["fault"].map(C.CLASS_MAP).to_numpy(dtype=str)
    labels = C.H_ORDER
    out: dict = {}

    def macro(XX, yy=y):
        return C.cv(XX, yy, meta, rc, labels, seed)["macro_f1"]

    m = rf(rc, seed).fit(X, y)
    imp = m.feature_importances_.reshape(len(names), k)
    by_metric = pd.Series(imp.sum(axis=1), index=names).sort_values(ascending=False)
    print(by_metric.head(25).round(4).to_string())
    out["top_metrics"] = by_metric.head(40).to_dict()
    per_class = []
    for mn in by_metric.index[:12]:
        j = names.index(mn) * k + 2  # window mean
        row = {"metric": mn}
        for c in labels:
            row[c] = float(np.median(X[y == c, j]))
        per_class.append(row)
    t = pd.DataFrame(per_class)
    print(t.round(1).to_string(index=False))
    out["top_metric_medians"] = per_class

    fam = np.array([n.split(".")[0] for n in names])
    rows = []
    for f in C.FILES:
        cols = np.ravel([[i * k + j for j in range(k)] for i in np.flatnonzero(fam == f)])
        rows.append({"family": f, "macro_f1": macro(X[:, cols])})
        log.info("%s", rows[-1])
    out["families"] = rows

    drop, rows = [], []
    order = list(by_metric.index)
    for n_drop in (5, 20, 50, 100, 150):
        drop = set(order[:n_drop])
        keep = [i for i, n in enumerate(names) if n not in drop]
        cols = np.ravel([[i * k + j for j in range(k)] for i in keep])
        rows.append(
            {"dropped_top": n_drop, "macro_f1": macro(X[:, cols]), "metrics_left": len(keep)}
        )
        log.info("%s", rows[-1])
    out["drop_top"] = rows

    rows = []
    for a, b in (("mono", "multi"), ("multi", "mono")):
        tr, te = (meta["block"] == a).to_numpy(), (meta["block"] == b).to_numpy()
        p = rf(rc, seed).fit(X[tr], y[tr]).predict(X[te])
        cm = confusion_matrix(y[te], p, labels=labels)
        rows.append(
            {
                "train": a,
                "test": b,
                "macro_f1": float(f1_score(y[te], p, labels=labels, average="macro")),
                "per_class": dict(
                    zip(
                        labels,
                        f1_score(y[te], p, labels=labels, average=None).round(4).tolist(),
                        strict=True,
                    )
                ),
                "confusion": cm.tolist(),
            }
        )
        print(rows[-1])
    out["cross_block"] = rows

    rng = np.random.default_rng(seed)
    n = len(y)
    stretch = 60  # one hour of windows moved together
    ids = np.arange(n) // stretch
    perm = rng.permutation(ids.max() + 1)
    idx = np.argsort(perm[ids], kind="stable")
    r = C.cv(X, y[idx], meta, rc, labels, seed)
    out["permuted_labels"] = {"macro_f1": r["macro_f1"], "per_class_f1": r["per_class_f1"]}
    print("permuted", out["permuted_labels"])
    run.write_json("metrics.json", out)
    run.finish("ok", {})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    return run_guarded(start_run("phase5", a.config, tag="d94-diag"), _run)


if __name__ == "__main__":
    sys.exit(main())
