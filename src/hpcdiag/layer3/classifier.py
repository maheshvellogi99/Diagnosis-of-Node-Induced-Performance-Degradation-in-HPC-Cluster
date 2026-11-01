"""Telemetry classifier for Gate 5d (D94): Random Forest on Antarex CPU/memory blocks.

Per-second LDMS metrics, 60-s non-overlapping windows, Tuncer-style statistics per metric. Windows
are kept only if one fault label (or none) covers all 60 seconds. Splits are contiguous time
segments with a purge around the test segment, never random windows.

    uv run python -m hpcdiag.layer3.classifier --config configs/phase5.yaml
"""

from __future__ import annotations

import argparse
import re
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score

from hpcdiag.utils.run import Run, resolve, run_guarded, start_run

WINDOW = 60
FILES = ("meminfo", "procstat", "vmstat", "procsensors", "diff_perfevent")
ID_COLS = {"#Time", "Time_usec", "ProducerName", "component_id", "job_id"}
STAT_NAMES = (
    "min", "max", "mean", "var", "skew", "kurt", "p5", "p25", "p50", "p75", "p95",
)  # fmt: skip
CLASS_MAP = {"None": "healthy", "dial": "cpu", "ddot": "cache", "memeater": "mbw", "leak": "mem"}
H_ORDER = ["healthy", "cpu", "cache", "mbw", "mem"]  # order of HYPOTHESES
OTHER_FAULTS = ("cpufreq", "pagefail")  # reported, excluded from the H mapping (D94)


def window_stats(x: np.ndarray) -> np.ndarray:
    """x: (n_windows, WINDOW, n_metrics). Returns (n_windows, n_metrics * 11), order STAT_NAMES."""
    q = np.percentile(x, [5, 25, 50, 75, 95], axis=1)  # (5, W, M)
    with np.errstate(invalid="ignore", divide="ignore"):
        sk = skew(x, axis=1, bias=True)
        ku = kurtosis(x, axis=1, bias=True)
    s = [x.min(axis=1), x.max(axis=1), x.mean(axis=1), x.var(axis=1), sk, ku, *q]
    out = np.stack(s, axis=2)  # (W, M, 11)
    return np.nan_to_num(out.reshape(len(x), -1), nan=0.0, posinf=0.0, neginf=0.0)


def is_cumulative(v: np.ndarray) -> bool:
    """A counter: non-decreasing on at least 99.9% of steps and increasing on at least 1%."""
    d = np.diff(v[np.isfinite(v)])
    if len(d) < 2:
        return False
    return bool((d >= 0).mean() >= 0.999 and (d > 0).mean() >= 0.01)


def per_second(t: np.ndarray, v: np.ndarray, t0: int, n: int) -> np.ndarray:
    """(n, n_metrics) on the second grid t0..t0+n-1: last sample in each second, carried forward
    at most 5 s; NaN where older than that."""
    idx = (np.floor(t).astype(np.int64) - t0).clip(-1, n)
    keep = (idx >= 0) & (idx < n)
    out = np.full((n, v.shape[1]), np.nan, np.float32)
    out[idx[keep]] = v[keep]  # later rows overwrite earlier ones in the same second
    df = pd.DataFrame(out).ffill(limit=5)
    return df.to_numpy(np.float32)


def pure_windows(fault: np.ndarray) -> np.ndarray:
    """fault: fault type per second. Returns one label per window; '' if the window is mixed."""
    n = len(fault) // WINDOW
    f = fault[: n * WINDOW].reshape(n, WINDOW)
    same = (f == f[:, :1]).all(axis=1)
    return np.where(same, f[:, 0], "")


def time_folds(t: np.ndarray, n_folds: int, purge_s: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Contiguous equal-duration segments of [t.min, t.max]. Fold k tests on segment k and trains
    on everything further than purge_s from it. Returns (train idx, test idx) pairs."""
    edges = np.linspace(t.min(), t.max() + 1, n_folds + 1)
    out = []
    for k in range(n_folds):
        te = (t >= edges[k]) & (t < edges[k + 1])
        tr = (t < edges[k] - purge_s) | (t >= edges[k + 1] + purge_s)
        out.append((np.flatnonzero(tr), np.flatnonzero(te)))
    return out


def fault_type(s: pd.Series) -> np.ndarray:
    return s.astype(str).str.replace(r"_\d+$", "", regex=True).to_numpy(dtype=str)


def per_core_groups(cols: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    """Group per-core columns by metric; the rest are system-wide."""
    groups: dict[str, list[str]] = {}
    plain = []
    for c in cols:
        m = re.fullmatch(r"(per_core_.+?)(\d+)", c)
        if m:
            groups.setdefault(m.group(1), []).append(c)
        else:
            plain.append(c)
    return groups, plain


def load_metrics(path, t0: int, n: int) -> tuple[np.ndarray, list[str]]:
    """One LDMS csv as per-second metrics: system-wide columns, and for per-core metrics the sum and
    the maximum over cores. Counters become per-second differences."""
    tb = pacsv.read_csv(path)
    cols = [
        c.name
        for c in tb.schema
        if c.name not in ID_COLS and (pa.types.is_integer(c.type) or pa.types.is_floating(c.type))
    ]
    t = tb.column("#Time").to_numpy()
    groups, plain = per_core_groups(cols)
    names, arrs = [], []
    for c in plain:
        names.append(c)
        arrs.append(tb.column(c).to_numpy().astype(np.float64))
    for g, cs in groups.items():
        m = np.column_stack([tb.column(c).to_numpy().astype(np.float64) for c in cs])
        for lab, v in (("sum", np.nansum(m, axis=1)), ("max", np.nanmax(m, axis=1))):
            names.append(f"{g}_{lab}")
            arrs.append(v)
    del tb
    o = np.argsort(t, kind="stable")
    t, v = t[o], np.column_stack(arrs)[o]
    for j in range(v.shape[1]):
        if is_cumulative(v[:, j]):
            v[:, j] = np.r_[0.0, np.diff(v[:, j])]
    return per_second(t, v, t0, n), names


def build_features(cfg: dict, log) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    a = cfg["antarex"]
    Xs, metas, names_ref = [], [], None
    for block in a["blocks"]:
        d = resolve(block["dir"])
        lab = pd.read_csv(d / block["labels"], sep=";", dtype=str, keep_default_na=False)
        t = pd.to_numeric(lab["#Time"]).to_numpy(np.int64)
        t0, n = int(t.min()), int(t.max() - t.min() + 1)
        grid = np.full(n, "None", dtype=object)
        grid[t - t0] = fault_type(lab["#Fault"])
        pure = pure_windows(grid.astype(str))
        m_all, names = [], []
        for f in FILES:
            v, nm = load_metrics(d / "ldms/csv" / f, t0, n)
            m_all.append(v)
            names += [f"{f}.{x}" for x in nm]
            log.info("%s %s: %d metrics", block["name"], f, len(nm))
        m = np.concatenate(m_all, axis=1)
        del m_all
        W = len(pure)
        x = m[: W * WINDOW].reshape(W, WINDOW, m.shape[1])
        complete = np.isfinite(x).all(axis=(1, 2))
        keep = complete & (pure != "")
        Xs.append(window_stats(x[keep]).astype(np.float32))
        metas.append(
            pd.DataFrame(
                {
                    "block": block["name"],
                    "t": t0 + np.flatnonzero(keep) * WINDOW,
                    "fault": pure[keep],
                }
            )
        )
        log.info(
            "%s: %d windows, %d mixed, %d incomplete, kept %d",
            block["name"], W, int((pure == "").sum()), int((~complete).sum()), int(keep.sum()),
        )  # fmt: skip
        if names_ref is None:
            names_ref = names
        elif names != names_ref:
            common = [c for c in names_ref if c in names]
            pick_a = [names_ref.index(c) for c in common]
            pick_b = [names.index(c) for c in common]
            k = len(STAT_NAMES)
            ia = np.ravel([[i * k + j for j in range(k)] for i in pick_a])
            ib = np.ravel([[i * k + j for j in range(k)] for i in pick_b])
            Xs = [z[:, ia] for z in Xs[:-1]] + [Xs[-1][:, ib]]
            names_ref = common
    return np.concatenate(Xs), pd.concat(metas, ignore_index=True), names_ref


def fit_predict(X, y, tr, te, rc, seed):
    rf = RandomForestClassifier(
        n_estimators=rc["n_estimators"],
        class_weight=rc["class_weight"],
        n_jobs=rc["n_jobs"],
        random_state=seed,
    )
    rf.fit(X[tr], y[tr])
    return rf.predict(X[te])


def cv(X, y, meta, rc, labels, seed) -> dict:
    """Out-of-fold predictions: fold k tests on segment k of every block, purged."""
    pred = np.full(len(y), "", dtype=object)
    per_fold = []
    folds = {}
    for _, g in meta.groupby("block"):
        for k, (tr, te) in enumerate(time_folds(g["t"].to_numpy(), rc["n_folds"], rc["purge_s"])):
            a, b = folds.setdefault(k, ([], []))
            a.append(g.index.to_numpy()[tr])
            b.append(g.index.to_numpy()[te])
    for k, (a, b) in sorted(folds.items()):
        tr, te = np.concatenate(a), np.concatenate(b)
        pred[te] = fit_predict(X, y, tr, te, rc, seed + k)
        per_fold.append(
            float(f1_score(y[te], pred[te].astype(str), labels=labels, average="macro"))
        )
    ok = pred != ""
    yt, yp = y[ok], pred[ok].astype(str)
    return {
        "labels": labels,
        "n": int(ok.sum()),
        "macro_f1": float(f1_score(yt, yp, labels=labels, average="macro")),
        "per_class_f1": dict(
            zip(labels, f1_score(yt, yp, labels=labels, average=None).tolist(), strict=True)
        ),
        "per_fold_macro_f1": per_fold,
        "confusion": confusion_matrix(yt, yp, labels=labels).tolist(),
        "pred": pred,
    }


def forward(X, y, meta, rc, labels, seed) -> dict:
    """Train on the first share of each block, test on the rest, purged: strict time order."""
    tr, te = [], []
    for _, g in meta.groupby("block"):
        t = g["t"].to_numpy()
        cut = t.min() + rc["forward_train_share"] * (t.max() - t.min())
        tr.append(g.index.to_numpy()[t < cut - rc["purge_s"]])
        te.append(g.index.to_numpy()[t >= cut])
    tr, te = np.concatenate(tr), np.concatenate(te)
    p = fit_predict(X, y, tr, te, rc, seed)
    return {
        "macro_f1": float(f1_score(y[te], p, labels=labels, average="macro")),
        "n_train": len(tr),
        "n_test": len(te),
    }


def _run(run: Run) -> int:
    cfg, log = run.config, run.log
    t0 = time.time()
    rc, seed = cfg["antarex"]["rf"], cfg["seed"]
    X, meta, names = build_features(cfg, log)
    log.info("features %s (%.0fs)", X.shape, time.time() - t0)
    counts = meta.groupby(["block", "fault"]).size().unstack(0).fillna(0).astype(int)
    log.info("windows\n%s", counts)
    if not set(CLASS_MAP) <= set(meta["fault"]):
        raise RuntimeError(f"missing classes: {sorted(meta['fault'].unique())}")
    main = meta["fault"].isin(list(CLASS_MAP)).to_numpy()
    X5 = X[main]
    m5 = meta[main].reset_index(drop=True)
    y5 = m5["fault"].map(CLASS_MAP).to_numpy(dtype=str)
    res = cv(X5, y5, m5, rc, H_ORDER, seed)
    res.pop("pred")
    log.info("5-class macro-F1 %.4f %s", res["macro_f1"], res["per_class_f1"])
    by_block = {}
    for b in cfg["antarex"]["blocks"]:
        i = (m5["block"] == b["name"]).to_numpy()
        r = cv(X5[i], y5[i], m5[i].reset_index(drop=True), rc, H_ORDER, seed)
        by_block[b["name"]] = {"macro_f1": r["macro_f1"], "per_class_f1": r["per_class_f1"]}
        log.info("block %s macro-F1 %.4f", b["name"], r["macro_f1"])
    fw = forward(X5, y5, m5, rc, H_ORDER, seed)
    log.info("forward macro-F1 %.4f", fw["macro_f1"])
    y7 = meta["fault"].map(lambda f: CLASS_MAP.get(f, f)).to_numpy(dtype=str)
    labels7 = H_ORDER + list(OTHER_FAULTS)
    r7 = cv(X, y7, meta, rc, labels7, seed)
    r7.pop("pred")
    log.info("7-class macro-F1 %.4f %s", r7["macro_f1"], r7["per_class_f1"])
    run.write_json(
        "metrics.json",
        {
            "n_features": X.shape[1],
            "n_metrics": len(names),
            "windows": counts.to_dict(),
            "five_class": res,
            "by_block": by_block,
            "forward": fw,
            "seven_class": r7,
            "seconds": time.time() - t0,
        },
    )
    run.finish("ok", {"macro_f1": round(res["macro_f1"], 4)})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    return run_guarded(start_run("phase5", a.config, tag="classifier"), _run)


if __name__ == "__main__":
    sys.exit(main())
