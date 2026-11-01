"""D81 (a): null diagnostic for relative e-values on old null bases 0..199 (never 200..399, D77).

    uv run python scripts/diag_4c_null.py --config configs/phase4_layer2.yaml --workers 7

Streams as in Phase 4b (real, iid, perm, perm_in_size), each with its own seed per base. v1 and
v1 with w = 1 use the guard (a) p-values, as in Phase 4b, so their cells must reproduce it. R and Rn
(cfg["relative"]) use the unguarded S_c p-values. Also measures the drift of D78: the mean of e over
active nodes in each block, which is exactly 1 for a fair update averaged over nodes.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time

import numpy as np
import pandas as pd

from hpcdiag.layer2 import accumulate as A
from hpcdiag.layer2 import prep as P
from hpcdiag.layer2 import run_layer2 as R
from hpcdiag.layer2 import select as SEL
from hpcdiag.stats.evalues import relative_evalues
from hpcdiag.utils.run import Run, run_guarded, start_run

STREAMS = ("real", "iid", "perm", "perm_in_size")
BUCKETS = [(1, 1), (2, 2), (3, 4), (5, 16), (17, 10**6)]  # as in Phase 4b
L: dict = {}


def _init(config_path: str) -> None:
    R._init(config_path)
    cfg = R.G["cfg"]
    l1 = cfg["layer1"]
    pop = R._null_pop()
    pl = P.load_placement(cfg["d62"]["placement"])
    size = pl.drop_duplicates("row_id").set_index("row_id")["n_nodes"]
    L.update(
        pop=pop,
        p_a=P.mode_p_on(pop, "score_c", l1["window_hours"], l1["min_ref"], l1["guard_a_hours"]),
        p_c=P.mode_p_on(pop, "score_c", l1["window_hours"], l1["min_ref"]),
        size=size.reindex(pop["row_id"]).to_numpy(),
        part=pd.factorize(pop["partition"].astype(str))[0],
        grid=A.block_grid(R.G["eval_ns"], int(pop["end_ns"].min())),
    )


def stream(p: np.ndarray, name: str, seed: int) -> np.ndarray:
    ok = np.isfinite(p)
    rng = np.random.default_rng(seed)
    out = p.copy()
    if name == "iid":
        out[ok] = rng.uniform(size=int(ok.sum()))
    elif name == "perm":
        out[ok] = rng.permutation(p[ok])
    elif name == "perm_in_size":
        sz = L["size"]
        for lo, hi in BUCKETS:
            i = np.flatnonzero(ok & (sz >= lo) & (sz <= hi))
            out[i] = rng.permutation(p[i])
    return out


def drift(pairs: pd.DataFrame, p: np.ndarray, bets, m: int) -> list[float]:
    """Sum over blocks of log(mean of e over the block's active nodes), one value per bet. A
    positive value means nodes gain on average with no signal at all. Partition 1 has 880 of the
    980 nodes, so the mean over all active nodes is close to its own."""
    pk = pairs["k"].to_numpy()
    b = np.searchsorted(L["grid"], pairs["end_ns"].to_numpy(), side="left")
    use = np.isfinite(p[pk]) & (b < len(L["grid"]))
    pu = p[pk][use]
    f = np.column_stack([g(pu) for g in bets])
    br, _, e = relative_evalues(
        b[use], L["part"][pk[use]], pairs["node_id"].to_numpy()[use], f, m, "pair"
    )
    _, inv = np.unique(br, return_inverse=True)
    c = np.bincount(inv)
    return [float(np.log(np.bincount(inv, weights=e[:, k]) / c).sum()) for k in range(e.shape[1])]


def task(k: int) -> dict:
    cfg = R.G["cfg"]
    lc = cfg["layer2"]
    placement = P.load_placement(
        f"data/processed/scenarios/_bases/random_{k:02d}/job_nodes.parquet"
    )
    pairs = A.make_pairs(L["pop"], placement)
    w = A.weights(pairs)
    J, m, eval_ns = lc["jumper_J"], lc["m_fixed"], R.G["eval_ns"]
    res = {"base": k}
    for name in STREAMS:
        pa, pc = stream(L["p_a"], name, 10_000 + k), stream(L["p_c"], name, 10_000 + k)
        for v in ("v1", "v1_w1", *cfg["relative"]):
            if v in cfg["relative"]:
                denom, lam = cfg["relative"][v]
                ev = A.node_evidence_relative(
                    pairs, pc, L["part"], R.G["bets"]["Sc"], J, lam, L["grid"], denom
                )
            else:
                ww = w if v == "v1" else np.ones(len(w))
                ev = A.node_evidence(pairs, pa, ww, R.G["bets"]["Sc_a"], J)
            snap = A.snapshots(ev, eval_ns, m)
            fl = SEL.ebh_over_time(snap, lc["alpha"], m)
            res[f"{name}_{v}"] = bool(fl.any())
            res[f"{name}_{v}_maxlogM"] = float(snap.max())
            if name == "real" and fl.any():
                t = int(np.flatnonzero(fl.any(axis=1))[0])
                res.setdefault("flags", []).append(
                    {"version": v, "base": k, "date": str(pd.Timestamp(int(eval_ns[t])))}
                )
        if name in ("real", "iid"):
            res[f"drift_{name}"] = drift(pairs, pc, R.G["bets"]["Sc"], m)
    return res


def _run(run: Run, config_path: str, workers: int) -> int:
    cfg = run.config
    _init(config_path)
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(config_path,)) as pool:
        res = pool.map(task, range(200), chunksize=4)
    drop = ("flags", "drift_real", "drift_iid")
    d = pd.DataFrame([{k: v for k, v in r.items() if k not in drop} for r in res])
    d.to_csv(run.dir / "diag.csv", index=False)
    rows = [
        {
            "stream": name,
            "v": v,
            "P_ever_200": float(d[f"{name}_{v}"].mean()),
            "se_200": float(np.sqrt(d[f"{name}_{v}"].mean() * (1 - d[f"{name}_{v}"].mean()) / 200)),
            "median_max_logM": float(d[f"{name}_{v}_maxlogM"].median()),
        }
        for name in STREAMS
        for v in ("v1", "v1_w1", *cfg["relative"])
    ]
    labels = [b.label for b in R.G["bets"]["Sc"]]
    dr = {
        n: dict(zip(labels, np.mean([r[f"drift_{n}"] for r in res], axis=0).tolist(), strict=True))
        for n in ("real", "iid")
    }
    flags = [x for r in res for x in r.get("flags", [])]
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("drift, D78 pair denominator, sum of log mean e over blocks:", dr)
    run.write_json(
        "metrics.json",
        {"table": rows, "drift": dr, "flags": flags, "seconds": time.time() - t0},
    )
    run.finish("ok", {"seconds": round(time.time() - t0, 1)})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    return run_guarded(
        start_run("phase4", a.config, tag="diag-4c"), lambda r: _run(r, a.config, a.workers)
    )


if __name__ == "__main__":
    sys.exit(main())
