# LAYER3_DESIGN.md — Layer 3 (Phases 5–6), finalised after reading the Layer 3 papers

Author: senior architect (chat), 9 Oct 2026. Status: **approved design, not yet merged into PLAN.md.**
When Phase 5 starts, Claude Code merges §1 into PLAN §2.1 (as F22–F26), §2 into PLAN §1.3/§1.4,
and §3–§5 into PLAN §9 Phases 5–6, then records the merge as a decision. Until then this file is the source.

Tags as in PLAN §0: **[V]** verified in a paper we hold (section given) · **[D]** our design choice · **[A]** assumption to test.

---

## 0. Summary in five lines

1. Every telemetry diagnosis paper we hold assumes 1 Hz monitoring on **every** node plus labelled anomaly runs (F22–F26).
2. Passive telemetry **confuses CPU, cache and memory-bandwidth anomalies** (F23, F26). A memory-bandwidth benchmark (STREAM)
   separates two of them directly (F23). That is the technical reason for *active* probes, not only cost.
3. Rish et al. choose probes by **information gain with every probe costing the same**, and they need a **periodic detection
   probe set on all nodes** (F24). We replace detection with Layer 2 (free, from logs) and weight information gain by cost [D].
4. Memory leak / memory-eater anomalies did **not** slow applications in the HPAS experiments (F23). Layer 2 (runtime-based)
   will therefore mainly surface CPU, cache and memory-bandwidth problems; Layer 3's hypothesis set reflects that.
5. On a real cluster we must **never** run the I/O anomalies (they caused filesystem outages, F23/F25) and we skip the network
   anomaly (needs SHMEM; produced contention only in two-node runs, F23/F25).

---

## 1. Verified facts from the Layer 3 papers (to become PLAN F22–F26)

| # | Fact | Source |
|---|---|---|
| F22 | Tuncer et al.: LDMS collects 721 metrics/s per node on Volta (52 nodes); features computed per node "and do not account for the interaction and correlation between multiple nodes"; window W = 45 s, alarm only after C = 5 consistent windows; random forest finds 98% of injected anomalies with 0.08% false alarms; LDMS uses "less than 0.1 percent of a core"; evaluation = 5-fold stratified CV over application runs; in 4-node runs one synthetic anomaly is injected on **one of the four nodes** | Tuncer TPDS 2019, §3.1, §3.4, §4.1, §4.6, §5.1, §5.3, §5.4, §5.11 |
| F23 | HPAS: 8 user-space anomaly generators (cpuoccupy, cachecopy, membw, memeater, memleak, netoccupy, iometadata, iobandwidth) with intensity knobs; membw uses x86 non-temporal store instructions (MOVNT*); netoccupy uses the SHMEM API; 256 instances of the I/O anomalies caused outages of Voltrino's Lustre filesystem; membw "significantly reduces memory bandwidth" measured by STREAM while cachecopy "has no significant impact on memory bandwidth"; cachecopy, cpuoccupy and membw slow applications most; memleak and memeater "do not visibly affect performance" (no swap; too large → crash); RF diagnosis: "the cachecopy, cpuoccupy, and membw anomalies are sometimes mistaken for each other. This could be due to the lack of metrics representing memory bandwidth"; code at github.com/peaclab/HPAS | Ates et al. ICPP 2019, Table 1, §3.3.3, §3.4, §4 (Figs. 4, 8), §5 (Figs. 9–10) |
| F24 | Rish et al.: active probing picks the next probe that maximises information gain I(X; T \| previous outcomes) (Fig. 6a, eq. 3); probes are counted, i.e. **unit cost**; detection relies on a small probe set that covers all nodes and runs on a schedule; single-fault assumption and noisy-OR likelihoods in experiments; optimal probe-set selection is NP-hard; active probing used 60–75% fewer probes than pre-planned probing on real applications; dynamics handled with DBNs | Rish et al. TNN 2005, §II–§VII, §VIII, §XI |
| F25 | Prodigy: VAE trained only on healthy samples; LDMS samples all Eclipse compute nodes at 1 Hz, ≈10 TB/day; HPAS cpuoccupy, cachecopy, membw, memleak used; I/O anomaly runs were terminated by administrators; the network anomaly caused contention only with two-node runs; macro-F1 0.95 (Eclipse) / 0.88 (Volta); anomaly threshold chosen as "the threshold that results in the highest F1-score in the **test** dataset"; output is binary (healthy/anomalous), not the anomaly type | Aksar et al. SC'23, §3, §4.1, §5.1–5.2, §5.4.4, §6.1 |
| F26 | Aksar et al. (ALBADross runtime): 60-s windows with 15-s skip; 156 node-level metrics after dropping per-core metrics; 6 applications × 5 labels (healthy + 4 HPAS anomalies); the anomaly runs on **every** node of the job; margin-sampling active learning reaches the supervised F1 with far fewer labels (0.78 F1 with 16× fewer samples); cachecopy and membw are the most-queried (most confused) classes; LDMS runs on all 1,488 Eclipse nodes; feature extraction for 1,488 windows × 2,000 features ≈ 28.5 s on one node | Aksar et al. TPDS 2024, Abstract, §IV-B, §IV-C, §IV-E, §V-B (Fig. 5), §V-D |

DOIs (all checked against the PDFs): Tuncer 10.1109/TPDS.2018.2870403 · HPAS 10.1145/3337821.3337907 · Rish 10.1109/TNN.2005.853423 · Prodigy 10.1145/3581784.3607076 · Aksar 10.1109/TPDS.2024.3365462.

---

## 2. Corrections to PLAN §1.3 gap 4 and contribution C4

**Correction 1 (attribution).** PLAN calls our selection rule "information-gain-per-cost (Rish 2005)". Rish uses information
gain with unit cost (F24). Dividing by cost is **our** design [D]. Write it that way in the paper.

**Correction 2 (the cost argument).** "Always-on telemetry is expensive per node" is **not** supported: LDMS uses < 0.1% of a core
(F22). What *is* supported: it needs a monitoring infrastructure on every node and produces ≈10 TB/day on a 1,488-node system (F25);
supervised diagnosis needs labelled anomaly runs (F26); and passive telemetry confuses CPU/cache/memory-bandwidth causes (F23, F26).

**Gap 4 (new wording).** Existing diagnosis methods assume 1 Hz telemetry from every node plus labelled anomaly data (F22, F25, F26),
and they still confuse CPU, cache and memory-bandwidth causes (F23, F26). Probe-based diagnosis (F24) needs periodic detection probes
on all nodes and treats every probe as equally costly. No method we hold starts from scheduler logs only, probes **only nodes that
a statistically controlled detector has flagged**, and chooses probes by information per unit of node time.

**C4 (new wording).** Cost-weighted active probing (information gain ÷ node-seconds), started only on Layer-2 discoveries, with a
telemetry probe whose likelihood is the measured confusion matrix of a telemetry classifier, and micro-benchmark probes that
separate the classes telemetry confuses.

---

## 3. Layer 3 design [D unless marked]

**3.1 Hypotheses H** (single cause per flagged node, as in Rish's single-fault experiments, F24):
h0 healthy (a Layer 2 false discovery or a transient), h_cpu (CPU contention, cf. cpuoccupy), h_cache (cache contention, cachecopy),
h_mbw (memory-bandwidth contention, membw), h_mem (memory pressure, memleak/memeater). Network and I/O causes are out of scope
(reasons in §0.5); "other/unknown" is reported when no hypothesis reaches the stopping threshold.

**3.2 Prior.** π0(h0) = 0.10 (Layer 2 runs e-BH at α = 0.1, so on average at most 10% of flagged nodes are healthy — an average,
not a per-node fact, hence [A]); the rest uniform over the four causes. Sensitivity: π0(h0) ∈ {0.05, 0.10, 0.30}.

**3.3 Probes A, outcomes and costs** (cost c_a = node-seconds the node is withheld from users; Phase 5 uses the planned durations
below, Phase 6 replaces them with measured ones):

| Probe | What it runs | Outcome (discretised) | Planned cost |
|---|---|---|---|
| a_tel | 60-s capture of /proc meminfo, vmstat, procstat (the samplers used in F25/F26) → telemetry classifier | predicted class | 60 s |
| a_cpu | fixed-work CPU loop on all cores | time ratio vs this node's healthy baseline: ≤1.05 / 1.05–1.20 / >1.20 | 20 s |
| a_mbw | STREAM triad | bandwidth ratio vs baseline: ≥0.95 / 0.80–0.95 / <0.80 | 20 s |
| a_cache | cache-resident working-set loop (L2- and L3-sized) | time ratio bins as a_cpu | 20 s |
| a_mem | read MemAvailable from /proc/meminfo | ratio vs baseline: ≥0.9 / 0.7–0.9 / <0.7 | 1 s |

Bin edges are [D] for Phase 5 and are re-fixed from healthy-baseline quantiles in Phase 6 **before** any anomaly run.

**3.4 Likelihoods P(o | h, a).**
- a_tel: the classifier's grouped-CV confusion matrix (rows = true class, columns = predicted), with add-one smoothing [D].
  Classes the training data lacks get a uniform row and are reported as a limitation.
- micro-benchmarks: Phase 5 uses **design tables** written from F23 only (membw lowers STREAM; cachecopy does not; cpuoccupy slows a
  CPU loop) with explicit noise — these are [A] and are never reported as results. Phase 6 replaces them with tables estimated from
  the injection campaign.

**3.5 Selection, update, stop.** a* = argmax_a [H(π) − E_o H(π | o, a)] / c_a over probes not yet run; Bayes update; stop when
max_h π(h) ≥ τ = 0.95, or the budget B = 300 node-s is spent, or the best ratio < 1e-4 bits/s. Output: diagnosis, posterior,
node-seconds spent, probe sequence. Exact computation (|H| = 5, |A| = 5): no approximation needed.

**3.6 Baselines.** B-all (run every probe; pre-planned battery), B-IG (Rish: information gain, unit cost), B-cheap (cheapest first
until τ), B-tel (telemetry probe only, i.e. the passive approach), B-random (random order until τ).

---

## 4. Phase 5 — Layer 3 offline

5.1 **Telemetry classifier.** Locate the Prodigy-format data used in the earlier Triplet project (path from Mahesh — UNKNOWN until he
gives it). Report which classes it actually contains. Reproduce RF / Triplet / gated-Triplet with the **same grouped 5-fold CV**
(by job). Export the RF confusion matrix → a_tel likelihood. Do not tune thresholds on test folds (contrast: F25 chose its threshold on
the test set).

5.2 **Engine** (`src/hpcdiag/layer3/`): entropy, expected posterior entropy, selection, Bayes update, stopping, all in tests first:
hand-computed toy tables where the right first probe is known; information gain equals brute-force enumeration; posterior sums to 1;
a probe whose rows are identical has zero gain.

5.3 **Simulation study.** Draw h ~ π0, draw outcomes from the tables, run every policy, 10,000 trials × 5 seeds.
Metrics: accuracy, mean node-seconds, mean probes, posterior calibration (expected calibration error).
Misspecification study: the policy plans with the design tables, outcomes come from Dirichlet-perturbed tables
(concentration ∈ {100, 30, 10}); report the accuracy drop.

5.4 **End-to-end on PM100 scenarios (simulation).** Take Layer 2 discoveries from Phase 4 on node scenarios whose truth table has a
`cause` column (PROMPT 3 addendum); simulate probe outcomes for the true cause (h0 for false discoveries); report diagnosis accuracy and
node-seconds per day, against (i) B-all on every discovery and (ii) Rish-style periodic detection: the full battery on every active
node once per day.

**Gate 5.** (i) RF macro-F1 within ±0.03 of the earlier 0.970 under the same grouped CV; (ii) all engine tests pass;
(iii) with correct tables, the cost-weighted policy's accuracy is within 1 percentage point of B-all at ≤ 60% of B-all's mean
node-seconds [D — target, report honestly if missed]; (iv) misspecification curve reported. Everything in Phase 5 is labelled
"simulation"; no real-hardware claim is made before Phase 6.

---

## 5. Phase 6 — real hardware (needs written admin approval)

**Preconditions.** Admin approval in writing; 2–4 dedicated **x86-64 Linux** nodes (membw needs x86 non-temporal stores, F23 — it will
not build on Apple-silicon Macs); Slurm or direct SSH; no root needed (HPAS is user space, F23).

**Safety rules (non-negotiable).** Never run iometadata or iobandwidth (filesystem outages, F23; runs terminated by admins, F25).
Do not run netoccupy (shared network; SHMEM; effect only in two-node runs, F23/F25). memleak capped well below node memory, with a
watchdog that kills it. Every anomaly has an explicit end time. Only on the approved nodes.

**6.1 Healthy baselines.** Each probe ≥ 30 runs per node over ≥ 3 different days; fix outcome bin edges from these quantiles
**before** any anomaly run (pre-registered in the decision log).

**6.2 Injection campaign.** {cpuoccupy, cachecopy, membw, memleak} × 2 intensities × ≥ 10 repetitions × all probes, randomised
order, start/stop logged to the second. Estimate P(o | h, a) on the first ⅔ of campaign days; evaluate the engine and baselines on the
last ⅓ (time split, no random split).

**6.3 Straggler check (validates Phase 3).** Run 2–4-node MPI benchmark jobs (NPB or mini-apps, as in F22) with one anomaly on one
node; measure the job slowdown factor s per anomaly × intensity. Compare with the Phase 3 values s ∈ {1.1, 1.2, 1.3, 1.5}.

**6.4 End-to-end demonstration.** Small benchmark workload, injection at random times on one node, Layer 3 triggered on that node;
report time-to-diagnosis, accuracy and node-seconds. Layer 1–2 statistics on 2–4 nodes are **not** meaningful (e-BH over 4 nodes);
this run demonstrates the pipeline only. Layer 2's statistical claims come from PM100 (Phases 3–4).

**Fallback if no cluster access.** One x86-64 Linux lab workstation: 6.1, 6.2 (single node, no MPI) still give real likelihood tables;
6.3 is dropped and stated as a limitation.

**Gate 6.** Measured tables and costs; held-out accuracy and node-seconds vs all baselines; every number traced to campaign logs;
admin sign-off recorded; safety checklist signed.

---

## 6. Open questions (to become PLAN Q10–Q12)

| Q | Question | Closed in |
|---|---|---|
| Q10 | Where is the Prodigy-format data from the earlier project, and which classes does it contain? | Phase 5 step 0 |
| Q11 | Admin answer: nodes, CPU architecture, scheduler, permission for HPAS (CPU/cache/membw/memleak only) | before Phase 6 |
| Q12 | Measured slowdown s per anomaly — does it match Phase 3's range? | Phase 6.3 |
