"""Why does v2 (D71) flag nulls? Diagnostics on Phase 3 null bases 0..199 (not fresh seeds).

    uv run python scripts/diag_d71_null.py --config configs/phase4_layer2.yaml --workers 7

v1 and v2 on the same placements with the p-value stream replaced by:
  real          the guard (a) S_c Mode P p-values
  iid           iid U(0, 1) (same warm-up mask): P(ever) must respect alpha if code and proof hold
  perm          real p-values randomly permuted over jobs (no time order, no link to job size)
  perm_in_size  permuted within job-size class (no time order, link to job size kept)
Versions: v1, v2, and on real and iid also v1 with w = 1 (no dilution, D76 ii).
Also: mean bet value by job size, autocorrelation of p, and what drives each real-stream v2 flag.
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
from hpcdiag.utils.run import Run, run_guarded, start_run

STREAMS = ("real", "iid", "perm", "perm_in_size")
# v1_w1: v1 with w = 1 instead of 1/|N_j| (no dilution): direct check of D76 (ii)
VERSIONS = {
    "real": ("v1", "v2", "v1_w1"),
    "iid": ("v1", "v2", "v1_w1"),
    "perm": ("v1", "v2"),
    "perm_in_size": ("v1", "v2"),
}
BUCKETS = [(1, 1), (2, 2), (3, 4), (5, 16), (17, 10**6)]
L: dict = {}


def _init(config_path: str) -> None:
    R._init(config_path)
    cfg = R.G["cfg"]
    l1 = cfg["layer1"]
    pop = R._null_pop()
    p = P.mode_p_on(pop, "score_c", l1["window_hours"], l1["min_ref"], l1["guard_a_hours"])
    pl = P.load_placement(cfg["d62"]["placement"])
    size = pl.drop_duplicates("row_id").set_index("row_id")["n_nodes"]
    L.update(pop=pop, p=p, size=size.reindex(pop["row_id"]).to_numpy())


def stream(name: str, seed: int) -> np.ndarray:
    p = L["p"]
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


def task(k: int) -> dict:
    cfg = R.G["cfg"]
    lc = cfg["layer2"]
    placement = P.load_placement(
        f"data/processed/scenarios/_bases/random_{k:02d}/job_nodes.parquet"
    )
    pairs = A.make_pairs(L["pop"], placement)
    w = A.weights(pairs)
    bets = R.G["bets"]["Sc_a"]
    J, m, eval_ns = lc["jumper_J"], lc["m_fixed"], R.G["eval_ns"]
    res = {"base": k}
    for name in STREAMS:
        p = stream(name, 10_000 + k)
        for v in VERSIONS[name]:
            if v == "v2":
                ev = A.node_evidence_blame(pairs, p, bets, J)
            else:
                ev = A.node_evidence(pairs, p, w if v == "v1" else np.ones(len(w)), bets, J)
            snap = A.snapshots(ev, eval_ns, m)
            fl = SEL.ebh_over_time(snap, lc["alpha"], m)
            res[f"{name}_{v}"] = bool(fl.any())
            res[f"{name}_{v}_maxlogM"] = float(snap.max())
            if name == "real" and fl.any():
                t = int(np.flatnonzero(fl.any(axis=1))[0])
                n = int(np.flatnonzero(fl[t])[0])
                e = ev[ev["node_id"] == n].reset_index(drop=True)
                gain = np.diff(np.r_[0.0, e["logM"].to_numpy()])
                sel = (e["end_ns"] <= eval_ns[t]).to_numpy()
                nn = pairs.loc[
                    (pairs["node_id"] == n) & np.isfinite(p[pairs["k"].to_numpy()]), "n_nodes"
                ].to_numpy()
                top = np.argsort(-gain[sel])[:20]
                res.setdefault("flags", []).append(
                    {
                        "version": v,
                        "base": k,
                        "date": str(pd.Timestamp(int(eval_ns[t]))),
                        "nodes_at_first_flag": int(fl[t].sum()),
                        "distinct_nodes_ever": int(fl.any(axis=0).sum()),
                        "node": n,
                        "t_days": (eval_ns[t] - eval_ns[0]) / (24 * 3600 * 1e9),
                        "n_updates": int(sel.sum()),
                        "mean_lam": float(e["lam"][sel].mean()) if v == "v2" else float("nan"),
                        "share_gain_top20": float(gain[sel][top].sum() / gain[sel].sum()),
                        "median_size_top20": float(np.median(nn[: len(sel)][sel][top])),
                    }
                )
    return res


def tables(cfg: dict) -> dict:
    p, sz = L["p"], L["size"]
    ok = np.isfinite(p) & np.isfinite(sz)
    bets = R.G["bets"]["Sc_a"]
    rows = []
    for lo, hi in BUCKETS:
        i = ok & (sz >= lo) & (sz <= hi)
        row = {
            "size": f"{lo}" if lo == hi else f"{lo}-{hi}",
            "jobs": int(i.sum()),
            "pairs": int(sz[i].sum()),
            "mean_p": float(p[i].mean()),
            "share_p_lt_0.1": float((p[i] < 0.1).mean()),
        }
        for b in bets:
            g = b(p[i])
            row[f"mean_{b.label}"] = float(g.mean())
            row[f"se_{b.label}"] = float(g.std() / np.sqrt(len(g)))
        rows.append(row)
    x = (p[ok] < 0.1).astype(float)
    x = x - x.mean()
    acf = {str(k): float(np.mean(x[:-k] * x[k:]) / np.mean(x * x)) for k in (1, 10, 100, 1000)}
    return {"by_size": rows, "acf_p_lt_0.1": acf, "n_jobs": int(ok.sum())}


def _run(run: Run, config_path: str, workers: int) -> int:
    cfg = run.config
    _init(config_path)
    t0 = time.time()
    tab = tables(cfg)
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init, initargs=(config_path,)) as pool:
        res = pool.map(task, range(200), chunksize=4)
    d = pd.DataFrame([{k: v for k, v in r.items() if k != "flags"} for r in res])
    d.to_csv(run.dir / "diag.csv", index=False)
    rows = []
    for name in STREAMS:
        for v in VERSIONS[name]:
            x = d[f"{name}_{v}"].to_numpy(float)
            y = x[:20]
            rows.append(
                {
                    "stream": name,
                    "v": v,
                    "P_ever_200": x.mean(),
                    "se_200": np.sqrt(x.mean() * (1 - x.mean()) / 200),
                    "P_ever_first20": y.mean(),
                    "median_max_logM": float(d[f"{name}_{v}_maxlogM"].median()),
                }
            )
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(t.round(3).to_string(index=False))
    print(pd.DataFrame(tab["by_size"]).round(4).T.to_string(header=False))
    print("acf", tab["acf_p_lt_0.1"], "jobs", tab["n_jobs"])
    fl = pd.DataFrame([x for r in res for x in r.get("flags", [])])
    print(fl.round(3).to_string(index=False))
    run.write_json(
        "metrics.json",
        {"table": rows, **tab, "flags": fl.to_dict("records"), "seconds": time.time() - t0},
    )
    run.finish("ok", {"seconds": round(time.time() - t0, 1)})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    return run_guarded(
        start_run("phase4", a.config, tag="diag-d71"), lambda r: _run(r, a.config, a.workers)
    )


if __name__ == "__main__":
    sys.exit(main())
