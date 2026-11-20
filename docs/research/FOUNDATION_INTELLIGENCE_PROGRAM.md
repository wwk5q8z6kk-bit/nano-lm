# Nano Foundation Intelligence Program

The November 17 proposal below preserves its original experiment portfolio. The November 20 [project charter](../PROJECT_CHARTER.md) places NanoScribe outside active work, leaves model size open, and prioritizes matched general-language baselines. Use the [current execution plan](../EXECUTION_PLAN.md) for training decisions.

**Status:** Proposed research plan and engineering roadmap (reconciled against `origin/master` @ `acc9b7e`); not a claim of implementation or 2B experimental validation.
**Date:** 2026-11-17
**Target envelope:** Preferred ~2B deployed learned parameters; hard ceiling strictly `< 4,000,000,000` deployed learned parameters across all weights, embeddings, routers, experts, adapters, and auxiliary learned modules.

---

## 0. Authority Alignment & Mission

This document operates under [PROJECT_AUTHORITY.md](../PROJECT_AUTHORITY.md) and [PROJECT_CHARTER.md](../PROJECT_CHARTER.md). It does not override [ACTIVE_NOW.md](../ACTIVE_NOW.md), [CAPABILITY_LADDER.md](../CAPABILITY_LADDER.md), or tagged empirical artifacts in `papers/` and `trajectory/`.

1. **Original application assumption (superseded November 20).** The November 17 proposal used medical scribing as a proving application. The current model program measures general language, reasoning and transfer on separate benchmarks.
2. **Strict parameter ceiling (`< 4B` total deployed):** Every candidate counts all deployed learned parameters (`backbone + embeddings + output head + all MoE experts + routers + adapters + auxiliary learned verifiers/fast weights`). A model with 2B active and 5B total parameters is out-of-cap.
3. **Two strictly separated evaluation tracks (no conflation):**
   - **Track 1 — Self-Contained Foundation Model:** Naked checkpoint weights only; zero external retrieval, Python execution, tools, or helper models.
   - **Track 2 — Grounded Intelligent System:** Compact foundation model co-designed with explicit evidence ledger (`nano/`), immutable source store, deterministic verifiers (`fabric/`, `nanoscribe/`), tools, and adaptive compute control.
   Both frontiers must be reported independently. System reliability gains must never be misrepresented as naked-weight capability.
4. **No premature 2B pretraining commitment:** All mechanism hypotheses must pass `T0`–`T3` on `3M–350M` controlled scaling ladders and match modern 2026 baselines (`GDN-2`, `Mamba-3`, `Qwen3.5-2B`, `Dense GQA + Muon`) before committing to a `2B` pretrain run.
5. **Scientific safety:** No PHI or private owner material; no clinical deployment claims without external human/clinician dual-rater validation; no rewrites of frozen `papers/`/`trajectory/`/`artifacts/` evidence; no unapproved paid confirmatory or materially costly runs.
6. **Pareto objective (not a single IQ scalar):**

$$
F(a, \theta; W) = \bigl(Q_{\text{gen}},\; Q_{\text{reason}},\; Q_{\text{grounded}},\; Q_{\text{OOD}},\; Q_{\text{calibration}},\; Q_{\text{memory}},\; -C_{\text{train}},\; -C_{\text{infer}},\; -L_{p95},\; -E_{\text{joule}},\; -M_{\text{peak}}\bigr)
$$

subject to $P_{\text{deployed}} < 4 \times 10^9$ and non-negotiable evidence-conservation floors.

---

## 1. Primary Research Thesis: Grounded Adaptive State Intelligence

**Falsifiable hypothesis:** A compact learned model achieves substantially higher reliable task capability per unit of compute and memory when it:
1. avoids forcing every rare literal, identifier, or provenance chain through lossy weight or fixed-state compression (dual-channel lossless evidence + compressed semantic state),
2. maintains geometrically well-conditioned editable fast state with decoupled erase, geometric transformation, and write controls,
3. allocates recursive refinement, retrieval, tool execution, and verification according to expected value of information rather than uniform token emission, and
4. separates typed epistemic states (`supported`, `denied/absent`, `missing/unmentioned`, `uncertain`, `conflicting`, `superseded`) so uncertainty drives abstention and human review rather than silent fabrication.

### Falsifiers
- Reject extra architectural complexity if matched-FLOP/matched-memory `Dense GQA` or 2026 `GDN-2`/`Mamba-3` baselines match capability and calibration.
- Reject geometric transition constraints if gains hold only on synthetic group automata ($Z_2$, $Z_4$, $S_3$) without transferring to OOD retrieval, code, math, and grounded extraction.
- Keep dual-memory purely system-level (`Track 2`) if evidence retrieval latency, deletion semantics, or storage overhead eliminates its Pareto advantage.
- Reject learned halting if it collapses (as observed in `X3_sgrd_delta`) or fails to beat explicit bounded search under a shared multi-question FLOP budget.

---

## 2. Fundamental Research Questions

| ID | Question | Core Experimental Probe |
|----|----------|-------------------------|
| **RQ1** | **Intelligence substrate:** Which mixer learns transferable abstractions best per FLOP, parameter, and byte: dense attention, recurrent SSM/DeltaNet, hybrid, or external-state systems? | `X10`, `X11`, `X17` |
| **RQ2** | **Memory and state:** Does intelligence require distinct fast state, persistent episodic/semantic memory, and immutable exact-source storage? | `X05`, `X07`, `X14` |
| **RQ3** | **Geometry and dynamics:** Do orthogonal/Householder/rotational/dissipative updates improve length transfer and conditioning over unconstrained or contraction-only recurrences? | `X03`, `X04` |
| **RQ4** | **Computation and reasoning:** Can adaptive depth, recursive refinement, verification, and search improve verified accuracy per marginal joule/second? | `X08`, `RE01–RE05` |
| **RQ5** | **Grounded truth:** Does model-proposal + deterministic source/verifier co-design outperform both unconstrained LMs and brittle deterministic rules? | `X01`, `X02`, `X15`, `X16` |
| **RQ6** | **Continual learning:** What belongs in slow weights vs fast weights vs retrievable memory vs immutable evidence under deletion and drift? | `X14`, `MM06–MM14` |
| **RQ7** | **Causality and action:** Does action-conditioned world-state prediction beat next-token imitation on interventional tasks? | `SY04`, `SY06` |
| **RQ8** | **Scientific transfer:** Do gains generalize across unseen templates, non-medical domains (code/incident/doc QA), languages, lengths, and MPS/CUDA devices? | `X16`, `X18` |

---

## 3. Stage-Gate Program (`G0`–`G10`)

| Gate | Work Package | Deliverable | Exit Criterion |
|---|---|---|---|
| **G0** | Reconcile remote vs local worktrees (`X00`) | Verified inventory of `master`, `codex/p1-h7-scribing-path`, `nanolm-v3`, `nanolm-lab`, `nanolm-posttrain` | Every result tagged `VERIFIED`, `CONFOUNDED`, `INVALID`, `CROSS_BRANCH`, or `LOCAL_WORKTREE_UNMERGED` |
| **G1** | Trustworthy science harness (`X01`, `X08`) | Evaluator negative controls (oracle, all-missing, shuffled, wrong-patient, offset-corrupt, abstain), contamination & license manifests | Evaluator fails closed on corrupted gold; independent fixtures pass |
| **G2** | Strong matched baselines (`X05`, `X10`, `C01`) | Matched `Dense GQA` (`AdamW` + `Muon`), `Qwen3.5-2B`, `MiniCPM5-2B`, `GDN-2`, `Mamba-3` harnesses with total parameter census | Identical prompt/token/FLOP accounting across baselines |
| **G3** | Mechanism laboratory (`X03`–`X08`) | Multi-seed tests of Householder, `GDN-2`, `Mamba-3/SO(2)`, `Diff-GQA`, `AttnRes/mHC`, dual memory, VoI controller | Each P0/P1 innovation receives `ACCEPT`, `SCOPED_ONLY`, `REJECT`, or `PARK` |
| **G4** | P1 repaired grounding path (`X02`, `X15`) | Lossless candidate-to-note lineage (`evidence-grounded-v1`), repaired cross-field temporal/denial boundaries | Zero silent assertion drops; fresh unexposed cohort + external human review before clinical claims |
| **G5** | Small-model transfer (`X10`–`X14`) | 125M–350M 3-seed matched iso-FLOP portfolio across domains and sequence lengths | Pareto improvement over strongest matched baseline beyond toy tasks |
| **G6** | Data & training optimization (`X12`–`X14`) | Data quality/dedup manifests, static vs learning-progress curriculum, optimizer/objective interference reports | Reproducible scaling curves without test leakage |
| **G7** | Flagship authorization (`D05`) | Concrete `~2B` spec, kernel benchmarks, dataset rights, budget, independent holdouts | Explicit owner authorization before launching material/confirmatory `2B` run |
| **G8** | Independent 2B model result (`X19`) | Trained `<4B` Nano candidate vs best matched competitor | Held-out 15-pillar, OOD, calibration, and MPS/CUDA energy Pareto |
| **G9** | Grounded intelligent system (`X15`, `X16`, `X20`) | Dual-track model vs system scores on P1 and non-medical DomainPacks | Verified state transitions, deletion, human review readiness |
| **G10** | Scientific publication | Reproducible report, negative results, novelty audit, immutable artifacts | All claims bound by recorded denominators and external replication |

Parallelism rule: `G0`, `G1`, `G2`, and `G3`/`G4` progress in parallel on `$0` local/CPU/MPS or within existing approved budgets. `G7` is strictly gated by `G0`–`G6`.

---

## 4. Document Map

- [ARCHITECTURE_PORTFOLIO.md](ARCHITECTURE_PORTFOLIO.md) — Competing architectures `A`–`G`, strict `<4B` census, and external reference models
- [INNOVATION_REGISTER.md](INNOVATION_REGISTER.md) — 91 candidate mechanisms (`AX`, `MM`, `RP`, `RE`, `OP`, `DT`, `SY`), statuses, and admission rubric
- [MATHEMATICAL_RESEARCH.md](MATHEMATICAL_RESEARCH.md) — Formal equations, stability boundaries, dual-memory formulation, and falsifiers
- [EVALUATION_CONTRACT.md](EVALUATION_CONTRACT.md) — 15-pillar suite spec, T0–T6 gates, contamination checks, and evaluator negative controls
- [TRAINING_AND_COMPUTE.md](TRAINING_AND_COMPUTE.md) — L0–L5 ladder, data provenance/licensing, checkpoint cards, MPS/Kaggle/RunPod rules
- [EXPERIMENT_REGISTER.md](EXPERIMENT_REGISTER.md) — Forensic reconciliation of `H6`/`H7`/`C12–C15`/`NanoLM-v3`/`Track A+B` and queue `X00–X20`
- [RESULTS_PARETO.md](RESULTS_PARETO.md) — Quality, OOD, memory, and MPS latency crossover ledger with confounds flagged
- [NOVELTY_AND_PRIOR_ART.md](NOVELTY_AND_PRIOR_ART.md) — 2024–2026 literature registry separating prior art from Nano hypotheses
- [EXTERNAL_REPLICATION.md](EXTERNAL_REPLICATION.md) — Verification commands, SHA manifests, and independent audit procedures
