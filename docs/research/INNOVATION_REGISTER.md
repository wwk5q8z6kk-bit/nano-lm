# Innovation Register & Admission Rubric

All 91 research candidates (`AX01–AX15`, `MM01–MM15`, `RP01–RP08`, `RE01–RE12`, `OP01–OP12`, `DT01–DT15`, `SY01–SY14`) are tracked below.  
**Rule:** Literature mechanisms are **prior art**, not Nano inventions. Statuses: `PROPOSED`, `PROTOTYPED`, `MEASURED`, `REPLICATED`, `ACCEPTED`, `SCOPED_ONLY`, `REJECTED`, `INVALID`, `PARKED`.

---

## Admission Rubric (7 Mandatory Axes)

Every candidate is scored before promotion to `125M+` runs:
1. Directly attacks a measured failure mode (`H6`, `C12–C14`, `EXP-1..5`, or 15-pillar tail error);
2. Strength of independent prior evidence (including 2025–2026 primary literature);
3. Experimental cost per discriminating signal (`$0` local / cheap `L0` first);
4. Mathematical soundness (analytic bounds + empirical Jacobian/conditioning checks);
5. Kernel/runtime compatibility across CPU, Apple Silicon MPS, and CUDA;
6. Cross-domain transfer (non-medical reasoning, code, evidence QA, OOD lengths);
7. Zero violation of the `<4B` parameter cap or evidence-conservation invariants.

---

## 5.A Sequence Mixing, Architectural Depth, and Routing (`AX01`–`AX15`)

| ID | Pri | Mechanism | Precise Question | Decisive Measure | Status |
|---|---|---|---|---|---|
| `AX01` | P0 | Dense GQA control | How much `NanoLM-v3` gain remains at matched optimizer/data/params? | Iso-FLOP BPB + multi-seed eval | `MEASURED` (Track A `46M`: `-0.73%` vs `T0+Muon`; `EXP-3` confounded by `+14.1%` non-emb params) |
| `AX02` | P0 | Gated DeltaNet-2 | Do channelwise erase/write controls beat $\beta\in(0,2)$ Householder on delayed multi-key edits? | Multi-key overwrite/erase + OOD | `PROPOSED` (reference lab `X04`) |
| `AX03` | P0 | Mamba-3 | Do complex selective $SO(2)$/MIMO dynamics beat Householder on non-commutative state & $Z_4$? | Group OOD, length transfer, speed | `PROPOSED` (reference lab `X03`) |
| `AX04` | P0 | Kimi Delta Attention (KDA) | Do per-channel forget gates beat scalar decay at equal state bytes? | Retention/interference & NLL | `PROPOSED` |
| `AX05` | P0 | Global attention anchors (`DDDA`) | Minimum anchor density (`1:2`, `1:4`, `1:6`, `1:8`) for exact recall? | Exact needle recall vs KV bytes | `MEASURED` (`P2_delta` `0%` anchors collapsed `11.65` copy loss; `3:1` preserves copy) |
| `AX06` | P1 | Latent compressed attention (MLA) | Can global anchors compress KV rank without hurting exact value copy? | Exact copied value F1 + KV bytes | `PROPOSED` |
| `AX07` | P1 | Differential attention (`Diff-GQA`) | Does $\text{softmax}(Q_1 K_1^\top)-\lambda\text{softmax}(Q_2 K_2^\top)$ improve real retrieval beyond `79k` toy runs? | Multi-needle suite vs Dense/MLA | `PROTOTYPED` (`EXP-2` `79.5k` toy `3.1%->11.3%`; `EXP-3` standalone `G2` lagged `G0` on math/code) |
| `AX08` | P1 | Multi-scale local/global attention | Subquadratic cross-chunk retrieval at 64k–128k? | Cross-chunk recall @ 64k/128k | `PROPOSED` |
| `AX09` | P1 | Block Attention Residuals (`AttnRes`) | Does Nano Laminar Depth-AttnRes beat 2026 `AttnRes` at matched width/depth? | Validation BPB, gradient flow, `tok/s` | `PROTOTYPED` (`G3_depth_attnres` `1.8M` tok only; needs `X06`) |
| `AX10` | P1 | Manifold hyper-connections (`mHC`) | Does Birkhoff-polytope/manifold constraint stabilize residual expressivity? | Norm drift & training stability | `PROPOSED` (`X06`) |
| `AX11` | P1 | Adaptive depth / early exit | Route hard tokens (`ATRD`) without halting collapse? | Accuracy vs mean/p95 FLOPs | `PROTOTYPED` (`X3_sgrd` collapsed; `X5_atrd` pending in `nanolm-lab`) |
| `AX12` | P2 | Layer reuse / recurrent depth | Does shared-weight looping improve reasoning at iso-FLOPs? | Task success vs wall-clock | `REJECTED` at `46M` pretrain (`R1_loop` `+8.26%`, `R2` `-0.08%` iso-FLOP) |
| `AX13` | P2 | Tensor/low-rank factorization | Active FLOP savings without expressivity collapse? | Quality/FLOP & conditioning | `PROPOSED` |
| `AX14` | P2 | Sparse MoE under 4B total | MoE vs dense at matched total params & matched active FLOPs separately | Capability, load balance, `batch=1` latency | `PROTOTYPED` (`nanolm-v3-3.9b-moe` spec; `4.90 tok/s` MPS decode bottleneck) |
| `AX15` | P3 | Masked diffusion / iterative editing | Non-AR refinement for structured JSON/repair vs AR? | Exact schema/repair per second | `PROPOSED` |

---

## 5.B Learning, Memory, Knowledge, and Representation (`MM01`–`MM15`)

| ID | Pri | Mechanism | Precise Question | Decisive Measure | Status |
|---|---|---|---|---|---|
| `MM01` | P0 | Source-preserving dual memory | Separate immutable evidence bytes from lossy fast state? | Exact copy fidelity & byte cost | `PROPOSED` (`X07` / `nano/mechanisms/`) |
| `MM02` | P0 | Typed epistemic states | Separate `supported`, `denied/absent`, `missing`, `uncertain`, `conflicting`, `superseded`? | Classwise F1, selective risk/coverage | `MEASURED` (`H6` failed state calibration; `grounded.py` prototyped) |
| `MM03` | P0 | Robust span transport | Zero silent assertion drops across parser $\to$ verifier $\to$ renderer? | False-discard count & lineage audit | `MEASURED` (`C14` lost `13/13` free assertions; `X02` diagnostic) |
| `MM04` | P0 | Multi-timescale state | Working fast state, episodic log, semantic store | Interference across sessions | `PROPOSED` |
| `MM05` | P1 | Engram conditional lookup | Static facts via $O(1)$ hash lookup vs weights under `<4B` cap? | Knowledge accuracy + table bytes | `PROTOTYPED` (`X6_pkm_muon` in `nanolm-lab`) |
| `MM06` | P1 | Titans online memory | Neural test-time memory vs explicit retrieval on repeated updates? | Update/forget curves over horizon | `PROPOSED` |
| `MM07` | P1 | Nested learning/update rates | Slow/fast plasticity stability–plasticity trade-off? | Retention vs adaptation curve | `PROPOSED` |
| `MM08` | P1 | In-place test-time training | Fast-weight adaptation without corrupting global weights or user isolation? | Long-context delta + isolation check | `PROPOSED` |
| `MM09` | P1 | Memory consolidation/replay | Verified episodic replay against catastrophic forgetting? | Prior-task retention post-update | `PROPOSED` |
| `MM10` | P1 | Event-graph / temporal-state memory | Valid time, system time, supersession, contradictions (`nano/`) | Adversarial chronology & correction | `PROTOTYPED` (`nano/temporal.py`, `nano/contracts.py`) |
| `MM11` | P1 | Uncertainty-weighted writes | Gate unsafe memory commits until validation/human confirmation? | False-write & recall rates | `PROPOSED` |
| `MM12` | P2 | Interference-aware associative memory | Key orthogonalization/whitening for multi-key capacity? | Capacity under distractor noise | `PROPOSED` |
| `MM13` | P2 | Retrieval-distillation into weights | When to consolidate frequent retrievals while preserving deletability? | Retention, retrain cost, deletion | `PROPOSED` |
| `MM14` | P2 | Online test-time controller | Decide retain/evict/retrieve/fast-update/verify? | Task success per memory byte | `PROPOSED` |
| `MM15` | P3 | Holographic/VSA binding | Complex/phase binding on compositional generalization? | Collision rate & OOD composition | `PROPOSED` |

---

## 5.C Input Representation & Multimodality (`RP01`–`RP08`)

| ID | Pri | Mechanism | Precise Question | Decisive Measure | Status |
|---|---|---|---|---|---|
| `RP01` | P0 | Tokenizer audit | Fragmentation of drugs, labs, IDs, code literals, multilingual text? | Exact copy & bytes/char | `PROPOSED` (`X09`) |
| `RP02` | P1 | Byte latent patching (`BLT`) | Dynamic entropy-based byte patches vs BPE? | Exact value copy at iso-FLOP | `PROPOSED` |
| `RP03` | P1 | Hybrid byte–subword path | Byte identity for rare strings + subword efficiency elsewhere | Long-tail precision & parity | `PROPOSED` |
| `RP04` | P1 | Type-aware value channels | Side channel for numeric/date/unit literals | OOD numeric/unit exactness | `PROPOSED` |
| `RP05` | P1 | Information-adaptive granularity | Allocate compute to high-surprise regions | FLOPs vs rare-fact recall | `PROPOSED` |
| `RP06` | P2 | Semantic/event latent patches | Multi-granularity document/event encoding | Cross-doc reasoning | `PROPOSED` |
| `RP07` | P2 | Multimodal backbone adapters | Audio/chart encoders feeding Nano Core within `<4B` total | Transfer & total deployed count | `PROPOSED` |
| `RP08` | P2 | Spatial-temporal 3D geometry | Equivariant geometry for physical simulators only | Unseen-rotation intervention error | `PROPOSED` |

---

## 5.D Reasoning, Formal Methods, and Test-Time Compute (`RE01`–`RE12`)

| ID | Pri | Mechanism | Precise Question | Decisive Measure | Status |
|---|---|---|---|---|---|
| `RE01` | P0 | Verifiable reasoning harness | Executable math/code/logic rewards at fixed budgets | Success/joule & verified pass@1 | `PROTOTYPED` (`suite_15pillar.py`, `nanolm_posttrain/eval_suite.py`) |
| `RE02` | P0 | Adaptive compute allocation | VoI allocation across items under shared FLOP budget | Shared-budget aggregate score | `PROPOSED` (`X08`) |
| `RE03` | P1 | Tiny recursive reasoning (`TRM`) | Compact core iterated across refinement steps | Puzzle/ARC transfer vs latency | `PROPOSED` |
| `RE04` | P1 | Search with verification | Best-of-N / tree search + validator vs budget | Verified accuracy per candidate | `PROPOSED` |
| `RE05` | P1 | Learn to abstain | Missing premises trigger ask/abstain instead of fabrication | Selective risk at fixed coverage | `PROTOTYPED` (`fabric/`, `nanoscribe/`) |
| `RE06` | P1 | Latent vs verbal CoT | Recurrent latent steps vs emitted `<think>` tokens | Quality/FLOP & inspectability | `PROPOSED` |
| `RE07` | P1 | Tool-use judgement | Call solver/compiler vs weight answer | Net success minus call cost | `MEASURED` (Track B `When2Call` / tool recovery) |
| `RE08` | P1 | Typed intermediate programs | DSL/program intermediate representation for arithmetic/logic | Exactness & runtime errors | `PROPOSED` |
| `RE09` | P2 | Formal proof certificates | Lean/SAT/SMT verified certificates on formal tasks | Checker acceptance, 0 false proofs | `PROPOSED` |
| `RE10` | P2 | Algorithm execution traces | Execution traces + failed-hypothesis repair | Length extrapolation | `PROPOSED` |
| `RE11` | P2 | Self-correction with external evidence | Grounded belief revision without self-grading | Revision gain & calibration | `PROPOSED` |
| `RE12` | P3 | Neural stack/tape memory | External differentiable stack/tape for DFS/graph tasks | Arbitrary-length extrapolation | `PROPOSED` |

---

## 5.E Optimization, Training Dynamics, and Geometry (`OP01`–`OP12`)

| ID | Pri | Mechanism | Precise Question | Decisive Measure | Status |
|---|---|---|---|---|---|
| `OP01` | P0 | `Muon` vs `AdamW` matched | Isolate optimizer effect from architecture effect | Iso-FLOP & iso-token curves | `MEASURED` (`-6.33%` BPB on Dense `46M`; `X11` needed at larger scale) |
| `OP02` | P0 | State transition stability | Spectral norms vs nonlinear write-gated Jacobian SVDs | `4x`/`16x` length stability | `PROPOSED` (`X03` / `nano/mechanisms/`) |
| `OP03` | P1 | $\mu$P hyperparameter transfer | Transfer optimal LR across widths (`35M->125M->350M->2B`) | Scale-invariant loss curves | `PROPOSED` |
| `OP04` | P1 | Norm & init (`QK-Norm`, `Canon` `[0,0,1]`) | Prevent logit/attention drift and step-0 graft disruption | Divergence rate & step-0 KL | `MEASURED` (`I2` graft achieved `0.000000` step-0 KL) |
| `OP05` | P1 | Low-precision robustness (`bf16`/`fp8`) | Numeric parity & MPS `RMSNorm` dtype mismatches | Throughput + max diff | `MEASURED` (MPS `BFloat16`/`float` `rms_norm` mismatch flagged in `gen3.py`) |
| `OP06` | P1 | Householder update rank & $\beta$ range | $\beta\in(0,1)$ vs $\beta\in(0,2)$ vs multi-reflection rank-$r$ | $Z_2$, $Z_4$, $S_3$ & language transfer | `MEASURED` (`EXP-1`: `100%` $Z_2$, `99.37%` $S_3$; `X03`/`X04` replicates) |
| `OP07` | P1 | Cayley / Lie-group $SO(d)$ transition | Skew-symmetric rotations vs reflections | Length transfer vs kernel cost | `PROPOSED` |
| `OP08` | P2 | Dissipative continuous-time flow | $\dot z = (\Omega - DD^\top)z + Bu$ bounded autonomous energy | Long-horizon gradient norm | `PROPOSED` |
| `OP09` | P2 | Symplectic latent dynamics | Physical simulators with conserved invariants only | Energy drift over rollouts | `PROPOSED` (strictly scoped to physics sims) |
| `OP10` | P2 | Low-rank curvature preconditioners | Compare against `Muon` at iso-wall-clock | Quality per second | `PROPOSED` |
| `OP11` | P2 | Meta-learned update rule | Isolated sandbox adaptation stability | OOD transfer & divergence | `PROPOSED` |
| `OP12` | P3 | Renormalization hierarchy | Multi-scale temporal state operators | Cross-timescale transfer | `PROPOSED` |

---

## 5.F Data, Curriculum, Self-Supervision, and Continual Learning (`DT01`–`DT15`)

| ID | Pri | Mechanism | Precise Question | Decisive Measure | Status |
|---|---|---|---|---|---|
| `DT01` | P0 | Dataset lineage & dedup | Licenses, near-dup, MinHash, train/eval zero overlap | Audit pass + dataset cards | `PROTOTYPED` (`LAB-DATA-01` SHA-256 `867beb0d…`) |
| `DT02` | P0 | Data mixture controls | Frozen uniform vs tuned vs surprise/learning-progress | Per-domain BPB + forgetting | `MEASURED` (`EXP-4`: web rose `1.933->2.019` under Surprise-DoReMi) |
| `DT03` | P0 | Negative-control evaluator checks | Prevent H7 all-missing corruption | Evaluator fails closed | `PROTOTYPED` (`annotated_evaluation.py` + `X01`) |
| `DT04` | P1 | DCLM-style quality filtering | Reasoning/knowledge density filtering | Downstream gain over raw crawl | `PROPOSED` |
| `DT05` | P1 | Difficulty curriculum | Progressive depth/context/adversarial noise | OOD transfer at equal tokens | `PROPOSED` |
| `DT06` | P1 | Provenance-tracked synth teachers | Multi-teacher verified traces with disagreement audit | Real-corpus generalization | `PROTOTYPED` (`Track B` `Qwen3-8B` / `32B`) |
| `DT07` | P1 | Multi-token prediction (`MTP`) | Auxiliary lookahead learning + speculative draft | Capability + acceptance rate | `PROTOTYPED` (`nanolm/model/mtp.py`, not yet isolated) |
| `DT08` | P1 | Verifier-guided RL (`GRPO`/`RLVR`) | Executable rewards with KL anchor against reward hacking | Verified pass + OOD regression | `PROTOTYPED` (`nanolm-lab` Phase 2 `B5`) |
| `DT09` | P1 | Top-K + tail-mass KD | Correct tail KL vs off-policy SFT forgetting | Forgetting, KL, calibration | `MEASURED` (`Track B`: `B1` SFT dropped GSM8K `75->55.6%`; `B3` KD `73.2%`) |
| `DT10` | P1 | Cross-task regularization | Does scribing transfer to code/evidence QA? | Non-medical transfer delta | `PROPOSED` (`X16`) |
| `DT11` | P2 | Continual weight adaptation | Frozen backbone + LoRA vs replay vs fast weights | Plasticity/retention/privacy | `PROPOSED` (`X14`) |
| `DT12` | P2 | Procedural practice worlds | Generator/solver/verifier self-play | Fresh adversarial success | `PROPOSED` |
| `DT13` | P2 | Counterfactual augmentation | Swap entities, dates, negations, causes | Invariance & causal accuracy | `PROPOSED` |
| `DT14` | P2 | Learning-progress curriculum (`LP`) | Improvement/FLOP minus noise & retention penalties | Gain per FLOP without drift | `PROPOSED` (`X12`) |
| `DT15` | P3 | Sandboxed self-modification | Safe bounded plasticity under frozen safety invariants | Long-lived drift stability | `PROPOSED` |

---

## 5.G Grounding, Causality, Control, and System Optimization (`SY01`–`SY14`)

| ID | Pri | Mechanism | Precise Question | Decisive Measure | Status |
|---|---|---|---|---|---|
| `SY01` | P0 | Evidence-to-output conservation | Trace every candidate offset across parser/checker/note | Zero silent losses (`X02`) | `MEASURED` (`C14` lost `13/13`; `grounded.py` fixes 12/13; `retain-08` fix in `X02`) |
| `SY02` | P0 | Dual model vs system scorecards | Never mix naked weights with system tools | Separate Competition 1 & 2 sheets | `PROTOTYPED` (`nanolm-lab` Phase 2 PREREG) |
| `SY03` | P1 | Conformal selective prediction | Bounded-risk sets under shift assumptions | Risk at fixed coverage | `PROPOSED` |
| `SY04` | P1 | Action-conditioned world planning | Latent dynamics vs text-only planning | Simulator goal attainment | `PROPOSED` |
| `SY05` | P1 | Active evidence gathering | Cost-aware ask/retrieve/inspect policy | Useful steps per call | `PROPOSED` |
| `SY06` | P1 | Causal interventions | Counterfactual program/world edits | Intervention consistency | `PROTOTYPED` (`nano/causal.py`) |
| `SY07` | P1 | Speculative / `MTP` decoding | Quality-preserving speedup | p50/p95 `tok/s` & exact match | `PROPOSED` |
| `SY08` | P1 | Quantization / QAT (`int4`/`q8`) | Evidence fidelity & OOD calibration under quant | Grounded exact delta & joules | `PROPOSED` (`X18`) |
| `SY09` | P1 | GPU kernel & UMA audit | Fair MPS/CUDA profiles (`512`..`128k`, cold/warm) | `tok/s`, peak MB, joules/task | `MEASURED` (`C15` `2.08x` MPS prefix-KV; `v3` `ctx=512` vs `4096` crossover) |
| `SY10` | P1 | Commit-gated KV eviction | Evict KV only after durable evidence ledger commit | Zero uncommitted span loss | `MEASURED` (`CommitGatedKVCacheLedger` `99.981%` KV savings) |
| `SY11` | P2 | Multi-agent vs single+tools | Same total compute budget comparison | Quality per total FLOP/dollar | `PROPOSED` |
| `SY12` | P2 | Error-aware control flow | Rollback, partial retention, human escalation | Recovery rate & review load | `PROTOTYPED` (`nanoscribe/grounded.py`) |
| `SY13` | P3 | Hardware-aware NAS | Constrained search over legal blocks after cost | Pareto gain net of search | `PROPOSED` |
| `SY14` | P3 | Reversible inference/ckpt | Activation/cache trade-off without numeric drift | Peak memory & parity | `PROPOSED` |
