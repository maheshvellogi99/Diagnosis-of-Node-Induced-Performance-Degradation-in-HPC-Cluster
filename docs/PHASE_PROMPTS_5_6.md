# Phase prompts — part 2 (PROMPT 2B, addenda to PROMPTS 3–4, PROMPTS 5–6)

Written by the senior architect, 9 Oct 2026. Same rules as docs/PHASE_PROMPTS.md: paste one prompt, let Claude Code
finish and STOP, send the report for review, then paste the next one.

Order from here: **2B → 3 (+ addendum) → 4 (+ addendum) → 5 → 6.**

---

## PROMPT 2B — finish Phase 2 (paste now)

```
Read CLAUDE.md, docs/PLAN.md (Phase 2, Gate 2, D17-D21) and docs/LAYER3_DESIGN.md §0. We are still in Phase 2.
Do NOT start the full runs until Step 8.

Step 0 - housekeeping
The architect added docs/LAYER3_DESIGN.md and docs/PHASE_PROMPTS_5_6.md. Commit only these two files
("docs: Layer 3 design and prompts 2B-6") so the runs record a clean commit.

Step 1 - corrections to how the fast results were read (report text)
a) False-alarm rate by ACTUAL runtime bin and TIMEOUT share among alarms are conditioned on the outcome. A perfectly
   calibrated p-value also alarms more often on jobs that turned out long, and TIMEOUT jobs are by definition jobs that ran
   to their limit. Keep both tables, labelled "descriptive, outcome-conditioned - not a calibration check".
b) Delete the claim that the censoring assumption says TIMEOUT jobs should be under-represented among alarms. PLAN 2.2 only
   says a TIMEOUT job's own p-value is larger than its uncensored p-value would be.
c) Add the valid conditional checks: FA@0.05 by decile of q_hat (model prediction) and by decile of walltime_req_s (known at
   submit). Decile edges from the model's training rows.

Step 2 - the architect's exploratory finding (verify it, do not trust it)
On the fast-run Mode E scores the architect recomputed p-values with ONE pooled stream (all users, no Mondrian groups) over a
sliding 7-day window (definition in Step 3; warm-up n < 30 excluded; theta seed differs from ours):
- PM100 main: n 102,090; KS 0.0132; FA .0125 / .0523 / .0997; P(p<=1e-3) 0.00126; mean mixture calibrator f(p) 0.814
  (simulated-null percentiles for this n: 50th 0.80, 95th 1.155).
- Eagle main: n 371,874; KS 0.0218; FA .0122 / .0564 / .1058; P(p<=1e-3) 0.0021; P(p<=1e-4) 0.00071; mean f(p) 1.596
  (null 95th 1.039, 99th 1.234) -> Eagle's extreme tail is too heavy.
- Same jobs, Mode E as run: PM100 mean f(p) 1.419, P(p<=1e-3) 0.00317; Eagle 1.379, 0.00356.
- Per-user (and user-first fallback) streams are the main source of over-alarming: on PM100 a user-first 7-day window gave
  KS 0.052, and shorter windows were worse.
Disclosure for the report and D22: the architect chose W = 7 d after looking at W in {6 h, 1 d, 7 d, 365 d} with global and
q_hat-decile grouping on PM100 fast Mode E scores. Eagle was not used for the choice. Reproduce these numbers with your own
code in Step 3; if yours differ beyond Monte-Carlo noise from theta, report both and STOP.

Step 3 - D22: Mode P (pooled sliding-window conformal), candidate input to Layer 2
- Scores: the Mode E frozen-model scores S_j (no new model, no refit).
- Reference R_j: all jobs of the same population that come before j in the D14 order (end_time, then row_id) and have
  end_time >= end_time_j - W. No user/partition grouping.
- p_j = ( #{i in R_j : S_i > S_j} + theta_j * (1 + #{i in R_j : S_i = S_j}) ) / (|R_j| + 1), theta_j ~ U(0,1) from the
  theta_seed generator.
- If |R_j| < 30, p_j = NaN (warm-up): excluded from evaluation and from Layer 2.
- Main W = 7 days. Also report W in {6 h, 1 d, 365 d} and the q_hat-decile Mondrian variant (the architect looked at them).
- Tests first: (1) IID scores give uniform p (KS p > 0.01, 5 seeds), also with heavy ties; (2) the sliding-window result
  equals a brute-force O(n^2) computation on 2,000 random jobs; (3) no job that ends after j, or ties with j but has a larger
  row_id, enters R_j; (4) warm-up gives NaN.
- Validity statement for the report: uniformity needs exchangeability of S within the 7-day window plus job j [A]. Weaker
  than Mode E's whole-history assumption, still an assumption, checked empirically by Gate 2-L2.
- Known cost of pooling (write it down): without user conditioning, a user whose jobs are all slow produces small p-values on
  every node that user runs on. This is handled in Layer 2 (PROMPT 4 addendum), not here.

Step 4 - D23: Mode A tie-breaking
Replace the Mode A split-conformal p-value by its smoothed version
p_j = ( #{S_i > S_j} + theta_j * (1 + #{S_i = S_j}) ) / (n + 1). Add a ties test. Reason: PM100 Mode A "all" KS 0.30 came
from ties (conservative direction).

Step 5 - D24: tail checks (what Layer 2 needs)
Layer 2 bets with kappa down to 0.05, and f_kappa(p) = kappa * p^(kappa-1) is dominated by the smallest p-values; FA at
alpha >= 0.01 cannot certify that. For every mode and dataset report:
- P(p <= 1e-3) and P(p <= 1e-4) (null 0.001 and 0.0001);
- mean of f(p) = average over kappa in {0.05, 0.1, 0.25, 0.5} of kappa * p^(kappa-1), with its null distribution from 300
  simulated U(0,1) samples of the same n (fixed seed): 50th/95th/99th percentiles. Under uniform p this sample mean is
  usually BELOW 1 (heavy tail), so compare with the simulated null, not with 1.

Step 6 - D25: ACI judged against its own theory
Gibbs & Candes Prop. 4.1 bounds the long-run error of ONE stream by (max(alpha_1, 1 - alpha_1) + gamma) / (gamma * T).
Quote the exact statement from the paper if the PDF is in the project; if not, write "as recalled by the architect -
UNKNOWN, needs check". With gamma = 0.005 a per-user stream of T = 50 jobs has a bound near 4, i.e. no guarantee.
So: ACI for the gate = one global stream (all jobs, D14 order). Per-user ACI = descriptive only.

Step 7 - D26: Gate 2 revision (I approve this change)
- Gate 2 (original, Mode E per user): keep the criteria, report the FAIL honestly with per-group drift, as PLAN says.
- Gate 2-L2 (input to Layer 2), checked on PM100 main with Mode P, W = 7 d:
  KS <= 0.02; FA within alpha +/- 0.01 at alpha in {0.01, 0.05, 0.10}; P(p <= 1e-3) <= 0.002;
  mean f(p) <= simulated-null 95th percentile; global-stream ACI rate within alpha +/- 0.005.
  Also on PM100 test days >= 90 (D21). Same table for Eagle main (reported; Eagle does not feed Layer 2).
- The decile tables from Step 1c are reported, not gated.
- If PM100 Mode P fails Gate 2-L2 in the full run: STOP and report. Do not try more windows.

Step 8 - full runs, option (b) (approved, > 30 min)
Eagle Mode A main full (~50 min); Eagle Mode E main and all full; all PM100 runs full; Eagle Mode A "all" fast only.
Then Mode P (all W variants), tail checks and global ACI from the stored scores.

Step 9 - exploratory, NOT a result: is unmodified PM100 clean?
With Mode P (W = 7 d) alarms at 0.05: per node with >= 50 jobs, alarm rate, distinct users, binomial tail p against the
overall rate, and the dispersion ratio var(observed rate) / var(binomial). State clearly that multi-node jobs make nodes
dependent, so these p-values are not valid tests. The architect's quick look on fast scores: dispersion about 2.9;
several nodes near twice the overall rate across about 100 users each (e.g. 287, 840, 309, 80, 841).

Step 10 - reports/phase_2.md with all of the above, commands and run ids; D22-D26 in PLAN §16 and docs/decisions.md.
Commit. STOP.
```

---

## ADDENDUM to PROMPT 3 (paste together with PROMPT 3, after Phase 2 is reviewed)

```
Addendum to Phase 3 (approved by me; record as decisions):
A1. Two base traces for every scenario:
    (i)  "random-placement" base (PRIMARY): for each job, replace its node set by |N_j| distinct nodes drawn uniformly
         at random from the nodes of its partition (seeded, one draw per base seed). This removes any real slow node and any
         user-node link, so the only non-null nodes are the injected ones: the truth is exact.
    (ii) "real-placement" base (SECONDARY): the original node sets. Real PM100 may contain genuinely slow nodes (Phase 2,
         Step 9), so flags on non-injected nodes here are "unknown truth", never counted as false.
A2. Truth table gets a `cause` column for node scenarios, drawn per degraded node from {cpu, cache, mbw} (equal mix, seeded).
    memory-pressure causes are not injected as slowdowns (HPAS: memleak/memeater did not visibly slow applications,
    LAYER3_DESIGN F23); they enter only in Phase 5 simulations.
A3. Onset t0 never in the first 30 days, and never inside a submission gap listed in docs/data_cards/pm100.md.
A4. Keep s in {1.1, 1.2, 1.3, 1.5}; Phase 6.3 will measure real slowdowns and we will compare.
A5. Report per scenario: affected jobs, affected nodes, censored jobs, and for random placement the per-node job counts.
```

---

## ADDENDUM to PROMPT 4 (paste together with PROMPT 4)

```
Addendum to Phase 4 (approved by me; record as decisions):
B1. Layer-1 input = Mode P, W = 7 d (if Gate 2-L2 passed). If it did not pass, STOP before Phase 4 and ask me.
B2. Gate 4a is measured on the random-placement base (exact truth). Real-placement results are reported with flags on
    non-injected nodes labelled "unknown truth".
B3. User-decoy defence, implemented as a variant and compared (ablation): per-job weight
    w_j = 1 / ( |N_j| * max(1, c_{u,n}) ), where c_{u,n} = number of jobs of the same user that STARTED on node n in the 24 h
    before job j started. It is known at job start, so it stays predictable (F16). Test that it is computed only from jobs
    that started before j.
B4. The system-shock guard (a) detrends S before Mode P is applied: S'_j = S_j - median(S of jobs that ended in the 6 h
    before end_time_j); then Mode P on S'. Guard (b) unchanged.
B5. Add a tail test to the e-process tests: on simulated null streams with p-values from Mode P applied to IID scores,
    the empirical P(max M >= 1/alpha) <= alpha + Monte-Carlo error.
```

---

## PROMPT 5 — Layer 3 offline (classifier + cost-aware probe engine, simulation)

```
Read CLAUDE.md, docs/PLAN.md and docs/LAYER3_DESIGN.md fully. We are doing Phase 5 only.

Step 0 - merge and locate
a) Merge LAYER3_DESIGN.md into PLAN.md: §1 -> §2.1 as F22-F26; §2 -> §1.3 gap 4 and §1.4 C4; §3-§5 -> §9 Phases 5-6;
   §6 -> §2.2 as Q10-Q12. Show me the diff before committing. Record as a decision.
b) Ask me for the path of the Prodigy-format data used in the earlier Triplet project. Do not search my disk outside
   this project. Report which classes and how many runs/nodes it contains (closes Q10).

Step 1 - plan first; wait for my "go". Include expected runtimes.

Step 2 - telemetry classifier (src/hpcdiag/layer3/classifier.py)
Reproduce RF, Triplet and gated-Triplet with the SAME grouped 5-fold CV by job as the earlier project (seeded).
No threshold or hyper-parameter choice may use test folds. Export the RF out-of-fold confusion matrix (add-one smoothing)
as the a_tel likelihood table. Ask before installing PyTorch.

Step 3 - engine (src/hpcdiag/layer3/{tables,bayes,selection}.py), tests first:
- toy tables where the right first probe is known by hand;
- information gain equals brute-force enumeration;
- posterior sums to 1; a probe with identical rows has zero gain;
- stopping rules (tau = 0.95, budget B = 300 node-s, min gain/cost 1e-4) behave as specified.
Selection = argmax (H(pi) - E_o H(pi | o, a)) / c_a. Baselines B-all, B-IG (unit cost), B-cheap, B-tel, B-random.

Step 4 - design tables (configs/phase5_tables.yaml): micro-benchmark likelihoods written ONLY from LAYER3_DESIGN F23,
each entry marked [A], noise level a config value. These are never reported as measurements.

Step 5 - simulation study: 10,000 trials x 5 seeds; priors pi0(h0) in {0.05, 0.10, 0.30};
metrics accuracy, mean node-seconds, mean probes, expected calibration error; misspecification study with
Dirichlet-perturbed truth tables (concentration 100, 30, 10).

Step 6 - end-to-end simulation on Phase 4 outputs: for node scenarios (random-placement base), take the e-BH discoveries,
simulate probe outcomes for the true cause (h0 for false discoveries), report diagnosis accuracy and node-seconds per day
vs B-all on every discovery and vs a daily full battery on every active node.

Step 7 - reports/phase_5.md: tables and plots, Gate 5 pass/fail exactly as in LAYER3_DESIGN §4, everything labelled
"simulation". Commit. STOP.
```

---

## PROMPT 6 — Real hardware (only after written admin approval)

```
Read CLAUDE.md, docs/PLAN.md Phase 6 and the safety rules in it. We are doing Phase 6 only.
Admin approval text and the list of approved nodes are in docs/admin_approval.md (I will create it). If that file is
missing, STOP.

Step 0 - safety check, before any code runs on the cluster
- List the approved nodes, CPU architecture (must be x86-64 for membw), memory per node, scheduler.
- Confirm in writing in reports/phase_6_safety.md: no iometadata, no iobandwidth, no netoccupy; memleak capped at a
  config value below 25% of node memory with a watchdog; every anomaly has an end time; only approved nodes.
- Show me the exact commands you will run. Wait for my "go".

Step 1 - build HPAS (github.com/peaclab/HPAS, record commit hash) and the probes (a_cpu, a_mbw = STREAM, a_cache, a_mem,
a_tel capture) in a separate folder cluster/ with its own README. No network calls except the documented git clone.

Step 2 - healthy baselines: each probe >= 30 runs per node over >= 3 different days. Fix outcome bin edges from these
quantiles and COMMIT them before any anomaly run (pre-registration).

Step 3 - injection campaign: {cpuoccupy, cachecopy, membw, memleak} x 2 intensities x >= 10 repetitions x all probes,
randomised order, start/stop timestamps logged. Time split: tables estimated on the first 2/3 of campaign days, engine
evaluated on the last 1/3.

Step 4 - straggler check: 2-4-node MPI benchmark jobs with one anomaly on one node; slowdown factor s per anomaly x
intensity (closes Q12).

Step 5 - end-to-end demonstration (pipeline only; say so in the report).

Step 6 - reports/phase_6.md: measured tables, costs, held-out accuracy and node-seconds vs baselines, s values vs
Phase 3, every number traced to a log file, Gate 6, admin sign-off. Commit. STOP.
```
