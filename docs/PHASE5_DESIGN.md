# PHASE5_DESIGN.md — Layer 3 v2: active probing as evidence + cause diagnosis (simulation)

Author: senior architect (chat), 1 Nov 2026. Status: approved design for Phase 5.
**Supersedes PROMPT 5 in docs/PHASE_PROMPTS_5_6.md and LAYER3_DESIGN.md §3–§4** (LAYER3_DESIGN §1–§2 facts and
corrections still stand; §5 Phase 6 still stands). Tags: [V] verified source · [D] design choice · [A] assumption.

---

## 0. Idea in five lines

1. Layer 2 (logs, relative e-values Rn(1)) is valid but slow: it is used to **rank suspects**, not only to flag them.
2. Every 6 h, a small **probe budget** (node-seconds/day) is spent on the top-ranked suspects.
3. Each probe result becomes a **conformal p-value** against a pool of healthy probe runs, then an **e-value** that is
   multiplied into the node's evidence. e-BH on the combined evidence keeps the FDR guarantee (§3).
4. The same probe results update a **posterior over causes** {healthy, cpu, cache, mbw, mem}; probes are chosen by
   expected information gain per node-second (Rish et al. use unit cost, F24; dividing by cost is ours [D]).
5. Headline result: **detection delay and diagnosis accuracy vs probe budget**, with FDR still controlled.

## 1. Layer 2 inside Phase 5 (frozen at the start of Phase 5)

Rn(1) from Phase 4c (node-mean denominator, λ = 1, 6-h blocks, unguarded centred S_c, Mode P W = 7 d, LR jumper
{no bet, g_1.2 … g_3.0}, J = 0.01), e-BH α = 0.1, m = 980. Call its per-node capital E_n^log.
(Adopted as D83 in Step 0 of PROMPT 5v2, after the real-placement check D84.)

## 2. Probe model (simulation) [A — replaced by Phase 6 measurements]

Probes a ∈ {cpu, mbw, cache, mem} produce a scalar x where **larger = worse** (time ratio; for mbw the inverse
bandwidth ratio; for mem the deficit ratio of MemAvailable vs baseline). Healthy: log x ~ N(0, σ_a²).
Under cause h on a node degraded with job slowdown s: log x ~ N(δ_{h,a}(s), σ_a²), δ_{h,a}(s) = k_{h,a} · log s.

Design sensitivity table k_{h,a} [A], written only from F23 (HPAS: membw lowers STREAM bandwidth strongly,
cachecopy does not; cpuoccupy slows CPU work; memory anomalies do not slow applications):

| cause \ probe | cpu (20 s) | mbw = STREAM (20 s) | cache (20 s) | mem (1 s) |
|---|---|---|---|---|
| cpu   | 1.0 | 0.1 | 0.3 | 0 |
| cache | 0.3 | 0.0 | 1.0 | 0 |
| mbw   | 0.1 | 1.0 | 0.3 | 0 |
| mem   | 0   | 0.1 | 0   | 1.0 (fixed shift 0.5, not scaled by s) |

σ_a = 0.05 for all probes (main) [A]; sensitivity σ ∈ {0.1}; sensitivity k_primary ∈ {2.0} (diagonal entries).
Costs in parentheses [D]; Phase 6 measures them. Node scenarios have cause ∈ {cpu, cache, mbw} (Phase 3 A2);
h_mem never occurs in scenarios (no runtime effect, F23) but stays in the hypothesis set.
A node is in state h = cause only if degraded and t ≥ onset; otherwise h0. Probe noise is independent per run [A].

Telemetry probe (a_tel, 60 s): used for **diagnosis only, never in e-values** (its validity would rest on a model).
Likelihood = RF out-of-fold confusion matrix if the Prodigy data has the classes; otherwise the design confusion
[A]: correct class 0.80; cpu/cache/mbw confused with each other 0.08 each (F23, F26); healthy row 0.96 correct.

## 3. Probe e-values (validity)

Healthy pool P_a = n_pool healthy runs of probe a (simulation: n_pool = 200 drawn once from the healthy model, fixed;
Phase 6: measured baselines). For a new run x on node n:

  p = ( #{y ∈ P_a : y > x} + θ · (1 + #{y ∈ P_a : y = x}) ) / (n_pool + 1),  θ ~ U(0,1).

Under H0_n (node healthy at probe time) and exchangeability of x with P_a [A], p is super-uniform, whatever the
past, because x is a fresh measurement and the choice of node and probe uses only past information.

Bet: f_a(p) = (1 − β) + β · g_a(p), β = 0.5 [D], where g_a = histogram density (20 bins, renormalised to integrate
to exactly 1) of p under the cause mixture "uniform over {cpu, cache, mbw}" at s0 = 1.3, built once by simulation
before evaluation. ∫₀¹ f_a = 1, f_a ≥ 0.5 > 0 (a probe can at most halve a node's capital).

Combined evidence: E_n ← E_n · f_a(p) applied to every jumper state of node n, at the probe's completion time,
in time order with log updates. Each factor is conditionally valid given everything before it, so E_n is an
e-process under H0_n and e-BH at the 6-h evaluations keeps the stopping-time argument of D49.
Same-type repeats on one node are allowed only in different investigations; the shared pool makes their p-values
slightly dependent [A] — n_pool = 200 keeps this small; state it.

Co-allocation note: probing a healthy node that shared slow jobs with a degraded node yields h0 outcomes, so its
capital tends to shrink — probes also help clear contaminated co-nodes. Measure it.

## 4. Policy

- **Schedule.** At each 6-h evaluation: budget per block = B/4 node-s. Candidates = nodes ranked by log E_n,
  excluding nodes investigated in the last 7 days. Investigate in rank order until the block budget is spent.
- **Diagnosis posterior.** Prior π0(h0) = 0.5, causes uniform [D]; sensitivity π0(h0) ∈ {0.1, 0.9}. Gaussian
  likelihood of log x per probe (σ_a, k table) and the telemetry confusion row; exact Bayes over 5 hypotheses.
- **Probe choice.** a* = argmax over unused probes of [H(π) − E_x H(π | x, a)] / c_a (expectation by Gauss–Hermite
  quadrature or 500 Monte-Carlo draws).
- **Stop an investigation** when max π ≥ 0.95, or every probe type was used, or the block budget is spent.
  If π(h0) ≥ 0.95 the node is "cleared" (not re-investigated for 7 d; its e-process is NOT reset).
- **Output per flagged node:** cause = argmax over causes of π, its probability, node-seconds spent, probe sequence.

## 5. Baselines (same budget B)

P-all (full battery on each investigated node, pre-planned), P-IG (Rish: information gain, unit cost),
P-random (random probe order), P-randnode (investigate random nodes instead of the Layer-2 ranking — shows the
value of Layer 2), P-sweep (Rish-style periodic detection: the full battery on nodes in round-robin, no Layer 2).
B ∈ {0, 60, 300, 1800} node-s/day; B = 0 is Layer 2 alone.

## 6. Evaluation (development only; sealed seeds untouched)

Random base, node scenarios seeds 0–19, K ∈ {1, 2, 5}%, s ∈ {1.1 … 3.0}; old null bases 0–199; user_decoy,
user_decoy_heavy, system_shock seeds 0–19. Real base reported as unknown truth.
Metrics: KM P(flag ≤ 14 d) and KM median delay; recall at trace end; mean FDP; null P(ever); decoy/shock P(ever);
top-1 cause accuracy among flagged degraded nodes; node-seconds per day and per correct diagnosis; healthy nodes
probed per day; co-allocated false flags with and without probes.
Misspecification: the policy and bets use the design table; outcomes come from a perturbed table
(each k ± 50%, uniform, seeded) and σ = 0.1. Validity must still hold (p-values are conformal); report power loss.

## 7. Gate 5 (pre-registered)

- **5a validity (binding):** with probes on at B = 300 s/day, null P(ever) ≤ α + 2 SE on the 200 old null bases;
  decoy and shock P(ever) ≤ α + 2 SE (80 each, pooled over s).
- **5b speed:** K = 1%, s = 1.3, B = 300 s/day: KM P(flag ≤ 14 d) ≥ 0.8 [D target; report honestly if missed].
- **5c diagnosis:** top-1 cause accuracy ≥ 0.9 among flagged degraded nodes (design table, σ = 0.05), with mean
  node-seconds per investigation ≤ 0.6 × P-all.
- **5d classifier:** if the Prodigy data is available, RF macro-F1 within ±0.03 of the earlier 0.970 under the same
  grouped CV.
Everything is labelled "simulation, design probe model [A]". Real numbers come from Phase 6.

## 8. Final evaluation (later, one shot)

After Phase 5 (and Phase 6 tables if available) the system is frozen; then seeds 20–39, nulls 200–399, all kinds,
run once (Phase 5F). No change after that run.
