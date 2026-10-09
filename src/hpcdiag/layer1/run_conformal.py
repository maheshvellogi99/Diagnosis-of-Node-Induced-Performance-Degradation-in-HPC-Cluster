"""Phase 2 runner: quantile model + conformal p-values + ACI (PLAN.md §9 Phase 2).

    uv run python -m hpcdiag.layer1.run_conformal --config configs/phase2_conformal.yaml \
        --system {eagle,pm100} --mode {select,A,E} [--population {main,all}] [--fast]

select: PM100 window selection (2.7, D18). A: Mode A (adaptive). E: Mode E (exact, frozen model).
Outputs in runs/phase2/<run_id>/ (metrics.json, scores_eval.parquet, figures) and the §3.1
contract table data/processed/layer1_scores_<system>_<run_id>.parquet (Modes A and E).
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from hpcdiag.eval import conformal_metrics as CM
from hpcdiag.eval.plots import pvalue_hists
from hpcdiag.features.encoding import encode_frame
from hpcdiag.layer1 import data as D
from hpcdiag.layer1.baseline import run_split
from hpcdiag.layer1.baselines import UARPModel, guide_alarm, uarp_alarm, user_alarm
from hpcdiag.layer1.quantile import fit_quantile, log_runtime, pinball_loss
from hpcdiag.layer1.scores import mode_a_pvalues, mode_e_pvalues
from hpcdiag.layer1.splits import TimeIndex, fast_subset, mode_a_windows, split_times
from hpcdiag.stats.aci import aci_alarms
from hpcdiag.utils.checksums import verify
from hpcdiag.utils.run import Run, resolve, run_guarded, start_run

DAY = pd.Timedelta(days=1)
CONTRACT = pa.schema([
    ("system", pa.string()), ("row_id", pa.int64()), ("job_id", pa.int64()),
    ("end_time", pa.timestamp("us")), ("group_key", pa.string()), ("q_hat", pa.float64()),
    ("score", pa.float64()), ("p_value", pa.float64()), ("theta", pa.float64()),
    ("alarm_fixed", pa.bool_()), ("alarm_aci", pa.bool_()), ("model_version", pa.string()),
])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--system", required=True, choices=["eagle", "pm100"])
    ap.add_argument("--mode", required=True, choices=["select", "A", "E"])
    ap.add_argument("--population", default="main")
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    tag = f"{a.system}-{a.mode}-{a.population}-{'fast' if a.fast else 'full'}"
    run = start_run("phase2", a.config, tag=tag)
    return run_guarded(run, lambda r: _run(r, a.system, a.mode, a.population, a.fast))


# ----------------------------------------------------------------------------- data


def _load(cfg: dict, system: str, population: str, log):
    s = cfg["systems"][system]
    f = cfg["features"]
    base = D.load_jobs(resolve(s["jobs"]), f["numeric"] + f["categorical"],
                       cfg["filters"]["states"], cfg["filters"]["require_walltime"])
    if population not in cfg["populations"]:
        raise ValueError(f"unknown population {population}")
    pop = D.apply_variant(base, cfg["populations"][population] or {}, None)
    log.info("%s: base rows %d, population %s rows %d", system, len(base), population, len(pop))
    return base, pop


def _split_list(cfg: dict, system: str, base: pd.DataFrame) -> list[pd.Timestamp]:
    s = cfg["systems"][system]
    if system == "eagle":
        sp = s["splits"]
        return split_times(base["submit_time"], sp["first_split_offset_days"],
                           sp["split_every_days"], sp["test_days"])
    start, end = base["submit_time"].min(), base["submit_time"].max()
    t, out = start + pd.Timedelta(days=s["mode_a_first_day"]), []
    while t + DAY <= end:
        out.append(t)
        t += pd.Timedelta(days=s["split_every_days"])
    return out


def _train_days(cfg: dict, system: str) -> int:
    L = cfg["systems"][system]["train_days"]
    if L is None:
        raise ValueError(f"{system}.train_days not set (run --mode select first; D18)")
    return int(L)


# ----------------------------------------------------------------------------- run


def _run(run: Run, system: str, mode: str, population: str, fast: bool) -> int:
    cfg, log = run.config, run.log
    problems = verify(resolve(cfg["raw_dir"]), resolve(cfg["checksums_file"]))
    if problems:
        raise RuntimeError(f"raw checksum problems: {problems}")
    base, pop = _load(cfg, system, population, log)
    if mode == "select":
        return _select(run, system, pop, base)
    if mode == "A":
        return _mode_a(run, system, population, fast, base, pop)
    return _mode_e(run, system, population, fast, base, pop)


def _select(run: Run, system: str, pop: pd.DataFrame, base: pd.DataFrame) -> int:
    """PM100 window selection (PLAN 2.7, D18)."""
    cfg, log = run.config, run.log
    if system != "pm100":
        raise ValueError("selection is defined for pm100 only")
    sel, f = cfg["systems"][system]["selection"], cfg["features"]
    cal = cfg["mode_a"]["cal_days"]
    start = base["submit_time"].min()
    horizon = start + pd.Timedelta(days=sel["horizon_days"])
    pop = pop[pop["submit_time"] < horizon].reset_index(drop=True)   # never look past day 90
    pop = pop[pop["end_time"] < horizon].reset_index(drop=True)
    idx = TimeIndex.build(pop["end_time"], pop["submit_time"])
    days = [start + pd.Timedelta(days=sel["first_test_day"] + k) for k in range(sel["n_test_days"])]
    if days[-1] + DAY > horizon:
        raise RuntimeError("selection test days exceed the horizon")
    rows = []
    for L in sel["candidates"]:
        for t in days:
            tr, _, te = mode_a_windows(idx, t, L, cal, 1)
            if len(tr) == 0 or len(te) == 0:
                log.warning("L=%d day %s: empty train or test", L, t)
                continue
            xtr, xte, _ = encode_frame(pop.iloc[tr], pop.iloc[te], f["numeric"], f["categorical"])
            m = fit_quantile(xtr, log_runtime(pop["run_time_s"].iloc[tr]), sel["tau"],
                             cfg["quantile"]["lgbm_params"], cfg["seed"])
            y = log_runtime(pop["run_time_s"].iloc[te])
            q = m.predict(xte)
            rows.append({"L": L, "t": t, "train_rows": len(tr), "test_rows": len(te),
                         "loss_sum": float(np.sum(np.maximum(sel["tau"] * (y - q),
                                                             (sel["tau"] - 1) * (y - q)))),
                         "pinball": pinball_loss(y, q, sel["tau"])})
            log.info("L=%d %s train=%d test=%d pinball=%.4f", L, t.date(), len(tr), len(te),
                     rows[-1]["pinball"])
    r = pd.DataFrame(rows)
    pooled = (r.groupby("L")["loss_sum"].sum() / r.groupby("L")["test_rows"].sum()).to_dict()
    best = min(pooled.values())
    near = sorted(L for L, v in pooled.items() if v <= best * (1 + sel["tie_rel"]))
    chosen = near[0]
    r.to_csv(run.dir / "selection_per_day.csv", index=False)
    m = {"system": system, "rule": "lowest pooled pinball (tau 0.9, log runtime); within "
         f"{sel['tie_rel']:.0%} of best -> shortest window (D18)",
         "test_days": [str(t) for t in days], "horizon": str(horizon),
         "pooled_pinball": {str(k): v for k, v in pooled.items()},
         "within_tie_of_best": near, "chosen_train_days": chosen,
         "test_rows_per_L": r.groupby("L")["test_rows"].sum().astype(int).to_dict()}
    run.write_json("metrics.json", m)
    run.finish("ok", {"chosen_train_days": chosen, "pooled_pinball": pooled})
    log.info("pooled pinball %s -> chosen L = %d", pooled, chosen)
    return 0


def _features(cfg, tr, *others):
    f = cfg["features"]
    xtr, xs = None, []
    for o in others:
        xtr, xo, _ = encode_frame(tr, o, f["numeric"], f["categorical"], f["unseen_code"])
        xs.append(xo)
    return xtr, xs


def _mode_a(run: Run, system: str, population: str, fast: bool, base, pop) -> int:
    cfg, log = run.config, run.log
    L, cal_days = _train_days(cfg, system), cfg["mode_a"]["cal_days"]
    times_all = _split_list(cfg, system, base)
    times = fast_subset(times_all, cfg["fast_mode"]["n_splits"]) if fast else times_all
    log.info("Mode A: L=%d cal=%d; %d split times in list, %d used", L, cal_days,
             len(times_all), len(times))
    idx = TimeIndex.build(pop["end_time"], pop["submit_time"])
    qp, seed, n_min = cfg["quantile"]["lgbm_params"], cfg["seed"], cfg["groups"]["n_min"]

    if system == "eagle":
        p1 = pd.read_parquet(resolve(cfg["systems"][system]["phase1_predictions"]),
                             columns=["row_id", "yhat"]).set_index("row_id")["yhat"]
    else:
        base_idx = TimeIndex.build(base["end_time"], base["submit_time"])
    f = cfg["features"]

    theta_rng = np.random.default_rng(cfg["theta_seed"])     # D23: one generator, split order
    parts, skipped, timing = [], [], []
    for i, t in enumerate(times):
        t0 = time.time()
        tr, ca, te = mode_a_windows(idx, t, L, cal_days, cfg["mode_a"]["test_days"])
        if min(len(tr), len(ca), len(te)) == 0:
            skipped.append({"t": str(t), "train": len(tr), "cal": len(ca), "test": len(te)})
            log.warning("split %s skipped: train=%d cal=%d test=%d", t, len(tr), len(ca), len(te))
            continue
        dtr, dca, dte = pop.iloc[tr], pop.iloc[ca], pop.iloc[te]
        xtr, (xca, xte) = _features(cfg, dtr, dca, dte)
        ytr = log_runtime(dtr["run_time_s"])
        m90 = fit_quantile(xtr, ytr, cfg["quantile"]["taus"]["upper"], qp, seed)
        m50 = fit_quantile(xtr, ytr, cfg["quantile"]["taus"]["median"], qp, seed)
        s_ca = log_runtime(dca["run_time_s"]) - m90.predict(xca)
        q_te = m90.predict(xte)
        s_te = log_runtime(dte["run_time_s"]) - q_te
        th = theta_rng.uniform(size=len(te))
        r = mode_a_pvalues(s_ca, dca["user"].to_numpy(), dca["partition"].to_numpy(), s_te,
                           dte["user"].to_numpy(), dte["partition"].to_numpy(), n_min, th,
                           cfg["mode_a"]["bound_alpha"])
        uarp = UARPModel.fit(xtr, dtr["run_time_s"].to_numpy(), cfg["baselines"]["uarp_params"],
                             seed)
        q99, sig = uarp.predict(xte)
        if system == "eagle":
            that = p1.reindex(dte["row_id"]).to_numpy()
            if np.isnan(that).any():
                raise RuntimeError(f"Phase 1 prediction missing for {np.isnan(that).sum()} rows")
        else:
            res = run_split(base, base_idx, t, L, 1, f["numeric"], f["categorical"],
                            "run_time_s", f["unseen_code"], {})
            that = res.predictions.set_index("row_id")["yhat"].reindex(dte["row_id"]).to_numpy()
            if np.isnan(that).any():
                raise RuntimeError("B-guide prediction missing")
        out = dte[["row_id", "job_id", "submit_time", "end_time", "run_time_s", "state_raw",
                   "nodes_req", "walltime_req_s", "user", "partition"]].copy()
        out["split_t"] = t
        out["q_hat"], out["score"], out["p"], out["theta"] = q_te, s_te, r.p, th
        out["dec_qhat"], out["dec_wall"] = _deciles(m90.predict(xtr), q_te,
                                                    dtr["walltime_req_s"], dte["walltime_req_s"])
        out["q50_log"] = m50.predict(xte)
        out["group_key"], out["n_cal"] = r.group_key, r.n_cal
        out["ub_log"] = q_te + r.upper_offset
        out["alarm_guide"] = guide_alarm(dte["run_time_s"], that,
                                         cfg["baselines"]["guide_threshold"],
                                         cfg["baselines"]["guide_clip_pred_min_s"])
        out["alarm_uarp"] = uarp_alarm(dte["run_time_s"], q99, sig)
        out["alarm_user"] = user_alarm(dte["run_time_s"], dte["walltime_req_s"])
        parts.append(out)
        timing.append(time.time() - t0)
        log.info("split %d/%d %s train=%d cal=%d test=%d fa05=%.4f (%.1fs)", i + 1, len(times),
                 t, len(tr), len(ca), len(te), float(np.mean(r.p <= 0.05)), timing[-1])

    df = pd.concat(parts, ignore_index=True).sort_values(["end_time", "row_id"],
                                                         kind="stable").reset_index(drop=True)
    return _finish(run, system, "A", population, fast, df, "p", None,
                   {"train_days": L, "cal_days": cal_days, "n_split_times_list": len(times_all),
                    "n_split_times_used": len(times), "n_nonempty": len(parts),
                    "skipped": skipped, "seconds_per_split_mean": float(np.mean(timing)),
                    "seconds_total": float(np.sum(timing))})


def _mode_e(run: Run, system: str, population: str, fast: bool, base, pop) -> int:
    cfg, log = run.config, run.log
    L = _train_days(cfg, system)
    start = base["submit_time"].min()
    t0 = start + pd.Timedelta(days=L)
    tr = pop[pop["end_time"] < t0]
    post = pop[pop["end_time"] >= t0]
    if fast:
        horizon = pd.Timedelta(days=cfg["fast_mode"]["mode_e_days_after_freeze"])
        post = post[post["end_time"] < t0 + horizon]
    post = post.sort_values(["end_time", "row_id"], kind="stable").reset_index(drop=True)
    log.info("Mode E: freeze at %s (L=%d); train rows %d; post-freeze rows %d", t0, L, len(tr),
             len(post))
    tstart = time.time()
    xtr, (xpost,) = _features(cfg, tr, post)
    ytr = log_runtime(tr["run_time_s"])
    qp, seed = cfg["quantile"]["lgbm_params"], cfg["seed"]
    m90 = fit_quantile(xtr, ytr, cfg["quantile"]["taus"]["upper"], qp, seed)
    m50 = fit_quantile(xtr, ytr, cfg["quantile"]["taus"]["median"], qp, seed)
    q = m90.predict(xpost)
    s = log_runtime(post["run_time_s"]) - q
    theta = np.random.default_rng(cfg["theta_seed"]).uniform(size=len(post))
    r = mode_e_pvalues(s, post["user"].to_numpy(), post["partition"].to_numpy(), theta,
                       cfg["groups"]["n_min"])
    df = post[["row_id", "job_id", "submit_time", "end_time", "run_time_s", "state_raw",
               "nodes_req", "walltime_req_s", "user", "partition"]].copy()
    df["q_hat"], df["score"], df["theta"] = q, s, theta
    df["dec_qhat"], df["dec_wall"] = _deciles(m90.predict(xtr), q, tr["walltime_req_s"],
                                              post["walltime_req_s"])
    df["q50_log"] = m50.predict(xpost)
    df["p"], df["group_key"], df["p_user"] = r.p_fallback, r.key_fallback, r.p_user
    # test-day membership (same days as Mode A)
    days = pd.DatetimeIndex(_split_list(cfg, system, base))
    pos = np.searchsorted(days.values, df["submit_time"].values, side="right") - 1
    ok = pos >= 0
    df["test_day"] = False
    day_end = days.values[pos[ok]] + DAY.to_timedelta64()
    df.loc[ok, "test_day"] = df["submit_time"].values[ok] < day_end
    secs = time.time() - tstart
    return _finish(run, system, "E", population, fast, df, "p", "p_user",
                   {"train_days": L, "freeze_time": str(t0), "train_rows": len(tr),
                    "post_rows": len(df), "seconds_total": secs})


def _deciles(q_train, q_eval, wall_train, wall_eval) -> tuple[np.ndarray, np.ndarray]:
    """Decile index 0..9 of q_hat and of walltime_req_s, edges from the model's TRAINING rows
    (Step 1c of PROMPT 2B). searchsorted(side='right') puts values equal to an edge above it."""
    qs = np.linspace(0.1, 0.9, 9)
    eq = np.quantile(np.asarray(q_train, float), qs)
    ew = np.quantile(np.asarray(wall_train, float), qs)
    return (np.searchsorted(eq, np.asarray(q_eval, float), side="right"),
            np.searchsorted(ew, np.asarray(wall_eval, float), side="right"))


# ----------------------------------------------------------------------------- outputs


def _finish(run: Run, system: str, mode: str, population: str, fast: bool, df: pd.DataFrame,
            pcol: str, pcol_variant: str | None, info: dict) -> int:
    cfg, log = run.config, run.log
    a = cfg["alarms"]
    df["alarm_fixed"] = df[pcol] <= a["alpha_job"]
    users = df["user"].astype(str).to_numpy()
    one = np.zeros(len(df), dtype=int)
    aci_rates, aci_global = {}, {}
    for g in a["aci_gammas"]:
        al, _ = aci_alarms(df[pcol].to_numpy(), users, a["alpha_job"], g)   # descriptive (D25)
        df[f"alarm_aci_peruser_{g:g}"] = al
        aci_rates[f"{g:g}"] = float(al.mean())
        al, _ = aci_alarms(df[pcol].to_numpy(), one, a["alpha_job"], g)     # gate stream (D25)
        df[f"alarm_aci_global_{g:g}"] = al
        aci_global[f"{g:g}"] = float(al.mean())
    gdef = f"alarm_aci_global_{a['aci_gamma_default']:g}"
    alarm_cols = ["alarm_fixed"] + [f"alarm_aci_global_{g:g}" for g in a["aci_gammas"]]
    alarm_cols += [f"alarm_aci_peruser_{g:g}" for g in a["aci_gammas"]]
    alarm_cols += [c for c in ("alarm_guide", "alarm_uarp", "alarm_user") if c in df]

    m = {"system": system, "mode": mode, "population": population, "fast": fast, **info,
         "aci_long_run_rate_peruser_descriptive": aci_rates,
         "aci_long_run_rate_global": aci_global}
    m["main"] = CM.evaluate(df, pcol, cfg, alarm_cols)
    if mode == "A":
        m["gate2_aci"] = CM.gate2_aci(aci_global[f"{a['aci_gamma_default']:g}"], cfg)
    if mode == "E":
        m["gate2_mode_e"] = CM.gate2_mode_e(m["main"], cfg)
        sub = df[df["test_day"]]
        m["test_days_only"] = CM.evaluate(sub, pcol, cfg, alarm_cols)
        # D21: Gate 2 restricted to the Mode A test days (PM100: days >= 90); sensitivity only
        m["gate2_mode_e_test_days_sensitivity"] = CM.gate2_mode_e(m["test_days_only"], cfg)
        al, _ = aci_alarms(df[pcol_variant].to_numpy(), users, a["alpha_job"],
                           a["aci_gamma_default"])
        df["alarm_fixed_user_only"] = df[pcol_variant] <= a["alpha_job"]
        df["alarm_aci_user_only"] = al
        m["variant_per_user"] = CM.evaluate(
            df, pcol_variant, cfg, ["alarm_fixed_user_only", "alarm_aci_user_only"])
        m["variant_per_user"]["gate2_mode_e_style"] = CM.gate2_mode_e(m["variant_per_user"], cfg)
        m["variant_per_user_test_days"] = CM.evaluate(
            df[df["test_day"]], pcol_variant, cfg, ["alarm_fixed_user_only"])

    df.to_parquet(run.dir / "scores_eval.parquet", index=False)
    contract = pd.DataFrame({
        "system": system, "row_id": df["row_id"], "job_id": df["job_id"],
        "end_time": df["end_time"], "group_key": df["group_key"].astype(str),
        "q_hat": df["q_hat"], "score": df["score"], "p_value": df[pcol],
        "theta": df["theta"] if "theta" in df else np.nan,
        "alarm_fixed": df["alarm_fixed"], "alarm_aci": df[gdef],
        "model_version": (df["split_t"].astype(str) if "split_t" in df
                          else f"frozen:{info.get('freeze_time')}"),
    })
    out = resolve(f"data/processed/layer1_scores_{system}_{run.run_id}.parquet")
    pq.write_table(pa.Table.from_pandas(contract, schema=CONTRACT, preserve_index=False), out)
    m["contract_table"] = str(out.relative_to(resolve(".")))
    pvalue_hists(df[pcol].to_numpy(), df["user"].astype(str).to_numpy(),
                 cfg["evaluation"]["top_users_ks"],
                 f"{system} Mode {mode} ({population}{', fast' if fast else ''}): p-values",
                 run.dir / "figures" / "pvalue_hist.png")
    run.write_json("metrics.json", m)
    key = {"n": m["main"]["n"], "ks": round(m["main"]["ks"], 4), "fa": m["main"]["fa"],
           "aci_global": aci_global}
    run.finish("ok", key)
    log.info("done: %s", key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
