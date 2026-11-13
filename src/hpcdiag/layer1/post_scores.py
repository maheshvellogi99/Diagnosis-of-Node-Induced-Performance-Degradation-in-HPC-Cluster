"""Phase 2B post-processing of stored Layer-1 scores (PROMPT 2B Steps 2-3, 5-7, 9; D21-D26).

    uv run python -m hpcdiag.layer1.post_scores --config configs/phase2_conformal.yaml \
        --runs <phase2 run id> [<phase2 run id> ...]

For every input run (Mode A or E): tail checks (D24), global-stream ACI (D25), decile tables.
For Mode E runs additionally: Mode P (pooled sliding-window conformal on the frozen-model scores,
D22) for all window/grouping variants, Monte-Carlo noise from theta, Gate 2 (original) with
per-group drift, Gate 2-L2 (D26), the D21 test-day subset, and (PM100 main) the exploratory
per-node look (Step 9). No model is refitted.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats

from hpcdiag.eval import conformal_metrics as CM
from hpcdiag.layer1.run_conformal import CONTRACT
from hpcdiag.stats.aci import aci_alarms
from hpcdiag.stats.calibrators import mixture_calibrator, tail_checks
from hpcdiag.stats.conformal import sliding_window_pvalues
from hpcdiag.utils.run import Run, resolve, run_guarded, start_run

H_NS = 3600 * 10**9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tag", default="post")
    a = ap.parse_args()
    run = start_run("phase2", a.config, tag=a.tag)
    return run_guarded(run, lambda r: _run(r, a.runs))


# ----------------------------------------------------------------------------- helpers


class TailCache:
    """D24 tail check; the simulated null depends only on n, so it is cached by n."""

    def __init__(self, cfg: dict) -> None:
        self.t = cfg["tail"]
        self._null: dict[int, dict] = {}

    def __call__(self, p: np.ndarray) -> dict:
        p = np.asarray(p, float)
        p = p[np.isfinite(p)]
        n = len(p)
        if n not in self._null:
            self._null[n] = tail_checks(p, self.t["null_sims"], self.t["null_seed"],
                                        self.t["thresholds"], self.t["kappas"])["null_mean_f"]
        mf = float(mixture_calibrator(p, self.t["kappas"]).mean())
        return {"n": n, "p_le": {f"{x:g}": float(np.mean(p <= x)) for x in self.t["thresholds"]},
                "mean_f": mf, "null_mean_f": self._null[n],
                "mean_f_le_null_p95": bool(mf <= self._null[n]["p95"]),
                "kappas": list(self.t["kappas"])}


def _aci_global(p: np.ndarray, cfg: dict) -> dict:
    a = cfg["alarms"]
    one = np.zeros(len(p), dtype=int)
    return {f"{g:g}": float(aci_alarms(p, one, a["alpha_job"], g)[0].mean())
            for g in a["aci_gammas"]}


def _basic(p: np.ndarray, cfg: dict, tail: TailCache) -> dict:
    p = np.asarray(p, float)
    p = p[np.isfinite(p)]
    return {"n": int(len(p)), "ks": CM.ks_distance(p),
            "fa": CM.fa_by_alpha(p, cfg["alarms"]["alphas"]),
            "tail": tail(p), "aci_global": _aci_global(p, cfg)}


def _deciles(df: pd.DataFrame, p: np.ndarray, alpha: float) -> dict:
    out = {}
    for col in ("dec_qhat", "dec_wall"):
        if col in df:
            d = df[col].to_numpy()
            ok = np.isfinite(p)
            out[col] = [{"decile": k + 1, "n": int(((d == k) & ok).sum()),
                         "fa": float(np.mean(p[(d == k) & ok] <= alpha))
                         if ((d == k) & ok).any() else float("nan")} for k in range(10)]
    return out


def gate2_l2(m: dict, cfg: dict) -> dict:
    """D26: KS, FA at each alpha, P(p<=1e-3), mean f(p) vs null 95th pct, global ACI."""
    g, a = cfg["gate2_l2"], cfg["alarms"]
    fa_ok = {k: bool(abs(v - float(k)) <= g["fa_tol"]) for k, v in m["fa"].items()}
    p3 = m["tail"]["p_le"]["0.001"]
    aci = m["aci_global"][f"{a['aci_gamma_default']:g}"]
    checks = {
        "ks": {"value": m["ks"], "max": g["ks_max"], "ok": bool(m["ks"] <= g["ks_max"])},
        "fa": {"value": m["fa"], "tol": g["fa_tol"], "ok": fa_ok},
        "p_le_1e-3": {"value": p3, "max": g["p_le_1e3_max"], "ok": bool(p3 <= g["p_le_1e3_max"])},
        "mean_f": {"value": m["tail"]["mean_f"], "null_p95": m["tail"]["null_mean_f"]["p95"],
                   "ok": m["tail"]["mean_f_le_null_p95"]},
        "aci_global": {"value": aci, "target": a["alpha_job"], "tol": g["aci_tol"],
                       "ok": bool(abs(aci - a["alpha_job"]) <= g["aci_tol"])},
    }
    ok = (checks["ks"]["ok"] and all(fa_ok.values()) and checks["p_le_1e-3"]["ok"]
          and checks["mean_f"]["ok"] and checks["aci_global"]["ok"])
    return {"checks": checks, "pass": bool(ok)}


def _drift(df: pd.DataFrame, p: np.ndarray, alpha: float, k: int = 10) -> dict:
    """Per-group drift for the original Gate 2 report: FA by month, per top-k user FA and KS."""
    month = df["end_time"].dt.to_period("M").astype(str).to_numpy()
    by_month = [{"month": mo, "n": int((month == mo).sum()),
                 "fa": float(np.mean(p[month == mo] <= alpha))} for mo in sorted(set(month))]
    users = df["user"].astype(str)
    top = users.value_counts().head(k)
    by_user = []
    for u, n in top.items():
        pu = p[(users == u).to_numpy()]
        half = len(pu) // 2
        by_user.append({"user": u, "n": int(n), "fa": float(np.mean(pu <= alpha)),
                        "ks": CM.ks_distance(pu),
                        "fa_first_half": float(np.mean(pu[:half] <= alpha)),
                        "fa_second_half": float(np.mean(pu[half:] <= alpha))})
    return {"fa_by_month": by_month, "top_users": by_user}


# ----------------------------------------------------------------------------- Mode P


def mode_p(df: pd.DataFrame, cfg: dict, window_h: float, variant: str,
           theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mode P p-values on the stored Mode E scores (df in D14 order)."""
    mp = cfg["mode_p"]
    s = df["score"].to_numpy()
    e = df["end_time"].to_numpy().astype("datetime64[ns]").astype("int64")
    w = int(window_h * H_NS)
    if variant == "global":
        return sliding_window_pvalues(s, e, theta, w, mp["min_ref"])
    if variant == "qhat_decile":
        return sliding_window_pvalues(s, e, theta, w, mp["min_ref"], df["dec_qhat"].to_numpy())
    if variant == "user_first":
        # exploratory, our definition: user's own window if it holds >= min_ref jobs, else pooled
        pu, nu = sliding_window_pvalues(s, e, theta, w, mp["min_ref"],
                                        df["user"].astype(str).to_numpy())
        pg, ng = sliding_window_pvalues(s, e, theta, w, mp["min_ref"])
        use_u = np.isfinite(pu)
        return np.where(use_u, pu, pg), np.where(use_u, nu, ng)
    raise ValueError(variant)


def node_check(df: pd.DataFrame, p: np.ndarray, cfg: dict) -> dict:
    """Step 9, exploratory: per-node alarm rates on unmodified PM100 (NOT valid tests)."""
    nc = cfg["node_check"]
    ok = np.isfinite(p)
    jobs = pd.DataFrame({"row_id": df["row_id"].to_numpy()[ok],
                         "alarm": p[ok] <= nc["alpha"],
                         "user": df["user"].astype(str).to_numpy()[ok]})
    jn = pd.read_parquet(resolve(nc["job_nodes"]), columns=["row_id", "node_id"])
    x = jn.merge(jobs, on="row_id", how="inner")
    overall = float(jobs["alarm"].mean())
    g = x.groupby("node_id").agg(n=("alarm", "size"), alarms=("alarm", "sum"),
                                 users=("user", "nunique"))
    g = g[g["n"] >= nc["min_jobs"]].copy()
    g["rate"] = g["alarms"] / g["n"]
    g["binom_upper_p"] = stats.binom.sf(g["alarms"] - 1, g["n"], overall)
    var_obs = float(g["rate"].var(ddof=1))
    var_binom = float(np.mean(overall * (1 - overall) / g["n"]))
    top = g.sort_values(["binom_upper_p", "rate"], ascending=[True, False]).head(10)
    return {
        "warning": ("multi-node jobs make nodes dependent; these binomial p-values are NOT "
                    "valid tests"),
        "alpha": nc["alpha"], "overall_alarm_rate": overall, "n_nodes": int(len(g)),
        "min_jobs": nc["min_jobs"], "dispersion_ratio": var_obs / var_binom,
        "rate_quantiles": {q: float(g["rate"].quantile(q)) for q in (0.1, 0.5, 0.9, 0.99)},
        "n_nodes_rate_ge_2x": int((g["rate"] >= 2 * overall).sum()),
        "top_nodes": [{"node_id": int(i), "n": int(r["n"]), "rate": float(r["rate"]),
                       "users": int(r["users"]), "binom_upper_p": float(r["binom_upper_p"])}
                      for i, r in top.iterrows()],
    }


# ----------------------------------------------------------------------------- run


def _run(run: Run, run_ids: list[str]) -> int:
    cfg, log = run.config, run.log
    tail = TailCache(cfg)
    mp, alpha = cfg["mode_p"], cfg["alarms"]["alpha_job"]
    out: dict = {"inputs": run_ids, "per_run": {}}
    for rid in run_ids:
        d = resolve(f"runs/phase2/{rid}")
        meta = json.loads((d / "metrics.json").read_text())
        df = pd.read_parquet(d / "scores_eval.parquet")
        df = df.sort_values(["end_time", "row_id"], kind="stable").reset_index(drop=True)
        system, mode, pop = meta["system"], meta["mode"], meta["population"]
        log.info("%s: %s Mode %s %s, %d rows", rid, system, mode, pop, len(df))
        r: dict = {"system": system, "mode": mode, "population": pop, "fast": meta["fast"],
                   "source_commit": (d / "git_commit.txt").read_text().strip()}
        p = df["p"].to_numpy()
        r["as_run"] = _basic(p, cfg, tail)
        r["as_run"]["deciles_alpha_job"] = _deciles(df, p, alpha)
        if mode == "E":
            r["as_run"]["gate2_original"] = meta["gate2_mode_e"]
            r["as_run"]["drift"] = _drift(df, p, alpha)
            r["per_user_variant"] = _basic(df["p_user"].to_numpy(), cfg, tail)
            theta = df["theta"].to_numpy()
            r["mode_p"], keep = {}, {}
            for wname, wh in mp["windows_hours"].items():
                for var in mp["variants"]:
                    if var != "global" and wname != mp["main_window"]:
                        continue
                    if var == "qhat_decile" and "dec_qhat" not in df:
                        r["mode_p"][f"{var}_{wname}"] = {"skipped": "no dec_qhat in this run"}
                        continue
                    pp, nref = mode_p(df, cfg, wh, var, theta)
                    name = f"{var}_{wname}"
                    m = _basic(pp, cfg, tail)
                    m["n_warmup_nan"] = int(np.isnan(pp).sum())
                    m["gate2_l2"] = gate2_l2(m, cfg)
                    ok = np.isfinite(pp)
                    m["same_jobs_mode_e_as_run"] = _basic(p[ok], cfg, tail)
                    m["deciles_alpha_job"] = _deciles(df, pp, alpha)
                    if "test_day" in df:
                        sub = df["test_day"].to_numpy()
                        m["test_days"] = _basic(pp[sub], cfg, tail)
                        m["test_days"]["gate2_l2"] = gate2_l2(m["test_days"], cfg)
                    r["mode_p"][name] = m
                    keep[name] = pp
                    log.info("%s %s: n=%d ks=%.4f fa=%s mean_f=%.3f", rid, name, m["n"], m["ks"],
                             {k: round(v, 4) for k, v in m["fa"].items()}, m["tail"]["mean_f"])
            main = f"global_{mp['main_window']}"
            noise = []
            for sd in mp["theta_noise_seeds"]:
                th = np.random.default_rng(sd).uniform(size=len(df))
                pp, _ = mode_p(df, cfg, mp["windows_hours"][mp["main_window"]], "global", th)
                f = pp[np.isfinite(pp)]
                noise.append({"seed": sd, "ks": CM.ks_distance(f),
                              "fa": CM.fa_by_alpha(f, cfg["alarms"]["alphas"]),
                              "p_le_1e-3": float(np.mean(f <= 1e-3)),
                              "mean_f": float(mixture_calibrator(f, cfg["tail"]["kappas"]).mean())})
            r["mode_p_theta_noise"] = noise
            pd.DataFrame({"row_id": df["row_id"], "end_time": df["end_time"],
                          **{f"p_{k}": v for k, v in keep.items()}}).to_parquet(
                run.dir / f"modep_{rid}.parquet", index=False)
            pm = keep[main]
            ok = np.isfinite(pm)
            one = np.zeros(int(ok.sum()), dtype=int)
            aci, _ = aci_alarms(pm[ok], one, alpha, cfg["alarms"]["aci_gamma_default"])
            contract = pd.DataFrame({
                "system": system, "row_id": df["row_id"][ok], "job_id": df["job_id"][ok],
                "end_time": df["end_time"][ok], "group_key": "pooled_window_" + mp["main_window"],
                "q_hat": df["q_hat"][ok], "score": df["score"][ok], "p_value": pm[ok],
                "theta": theta[ok], "alarm_fixed": pm[ok] <= alpha, "alarm_aci": aci,
                "model_version": f"modeP:{mp['main_window']}:from:{rid}"})
            outp = resolve(
                f"data/processed/layer1_scores_{system}_{run.run_id}_modeP_{pop}.parquet")
            pq.write_table(pa.Table.from_pandas(contract, schema=CONTRACT, preserve_index=False),
                           outp)
            r["mode_p_contract_table"] = str(outp.relative_to(resolve(".")))
            if system == "pm100" and pop == "main":
                r["node_check_exploratory"] = node_check(df, pm, cfg)
        out["per_run"][rid] = r
        run.write_json("metrics.json", out)       # incremental save
    run.finish("ok", {"n_inputs": len(run_ids)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
