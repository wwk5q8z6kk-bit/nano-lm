# PREREG PHASE 2 — Compact Intelligence System, Information-Dense Learning, RLVR, ATD+Memory Architecture & 120B Head-to-Head

*Registered before Phase 2 GPU execution on `nanolm/discovery-lab-v1`.*

## 1. Overriding Research Question
**Can a `< 4B` (`~2B`) model obtain substantially more intelligence, reliability, and task-completion capability per unit of compute by rethinking (a) neural computation & persistent memory, (b) information-dense / verification-driven training (`rsLoRA` vs Full Fine-Tuning + RLVR), and (c) a 5-engine cooperating system architecture — and where can it close or reverse the gap against `8B`, `32B`, and `~120B` (`openai/gpt-oss-120b`) models across two strictly separated competitions?**

---

## 2. Two Strictly Separated Competitions (Preventing Conflation)

### Competition 1 — The Foundation Model (Naked Weights, Zero External Assistance)
- **Rules:** Single model checkpoint, zero external Python execution, zero external retrieval, zero external verifiers.
- **Benchmarks (`6,000+` held-out items):**
  1. Public `lm_eval` (`GSM8K` 1,319, `MATH-500` 500, `HumanEval` 164, `IFEval` 541, `ARC-C` 1,172, `WinoGrande` 1,267, `TruthfulQA` 817).
  2. `132`-item deterministic 8-pillar benchmark (`think=0` & `think=1`).
  3. New **`LAB-FRONTIER-HARD` (`200`-item adversarial multi-step suite, zero overlap):** competition combinatorics/number theory/algebra, multi-function stateful coding, 4-turn chained API error recovery, adversarial nested schema conservation, and unseen technical specification synthesis.

### Competition 2 — The Complete Intelligent System (`5` Cooperating Engines)
- **Architecture (`lab/system/`):**
  1. **Reasoning Engine:** Entropy-triggered bounded recurrent reasoning + **anti-circularity n-gram/state divergence monitor** (prevents runaway `<think>` loops).
  2. **Memory Engine:** Episodic working scratchpad + hybrid dense/lexical persistent knowledge store with **confidence-calibrated sufficiency & contradiction gating**.
  3. **Verification Engine:** Deterministic code test synthesis & execution, symbolic/numeric dual-check for math, JSON/Mermaid AST validation, and explicit epistemic status tagging (`VERIFIED_COMPUTATION`, `VERIFIED_SCHEMA`, `SUPPORTED_EVIDENCE`, `PLAUSIBLE_UNVERIFIED`, `UNCERTAIN_ABSTAIN`).
  4. **Tool Execution Engine:** Live sandboxed Python + stateful multi-API environment with error inspection and iterative repair ($K \le 3$).
  5. **Adaptive Computation Controller:** Routes each query/step using `NanoLM`'s own predictive Shannon entropy & task structure so easy queries execute in **1 fast pass** while hard problems receive targeted reasoning, retrieval, tool execution, and verification.
- **Fairness Matrix:** Evaluates:
  - `NanoLM-2B System` vs `NanoLM-2B Naked` (isolating system gain & compute cost per problem),
  - `NanoLM-2B System` vs **Naked `Qwen3-8B`, `Qwen3-32B`, and `openai/gpt-oss-120b`** (measuring where `< 4B` system intelligence closes or reverses the `60×` parameter gap),
  - `NanoLM-2B System` vs **`Qwen3-8B System`, `Qwen3-32B System`, and `openai/gpt-oss-120b System` operating under identical 5-engine system conditions** (measuring true system-conditioned foundation scaling & capability-per-FLOP / capability-per-dollar!).

---

## 3. Preregistered Controlled Experiments

### Experiment `LAB-ARCH-04` (Track A Original Architecture — Adaptive Token-Routed Depth & Persistent Memory)
Fixes the two mathematical flaws diagnosed in `X3_sgrd_delta` (inverted surprise sign in halting logit + non-residual `inject` resetting Delta state) and tests two new mechanisms at `600M` tokens (`n=2` seeds, matched active non-embedding parameters `~29.4M` against `X4_nanolm_v2_muon` `0.9575 ± 0.0008` BPB):
1. **`X5_atrd_muon` (Adaptive Token-Routed Residual Depth + Anti-Circular Orthogonality):**
   - Pass 1 runs `NanoLM-v2` (`DDDADDDADA`); computes token predictive entropy $u_t = H(p_1(\cdot \mid x_{\le t}))$; routes **only the top `25%` highest-uncertainty tokens** through a step-conditioned zero-init residual refinement block regularized by anti-circular cosine penalty $\max(0, \cos(\Delta h^{(2)}_t, \Delta h^{(1)}_t) - 0.5)^2$.
   - **Falsification rule:** Reject if `X5_atrd_muon` fails to beat `X4_nanolm_v2_muon` (`0.9575` BPB) at matched tokens OR fails to beat `T0_dense + Muon` (`0.9645` BPB) at matched FLOPs.
2. **`X6_pkm_muon` (`NanoLM-v2` + Sparse Persistent Associative Knowledge Memory Bank):**
   - Replaces dense FFN width slice with a `1,024`-slot persistent associative memory bank querying only `top-8` slots per token with confidence gating (reporting active vs persistent memory parameters separately).
   - **Falsification rule:** Reject if `X6_pkm_muon` fails to improve knowledge/textbook/web BPB per active FLOP over `X4_nanolm_v2_muon`.

### Experiment `LAB-POST-04` & `LAB-RLVR-01` (Track B Superior Learning & Verification-Driven Post-Training)
1. **Controlled `rsLoRA` vs Full Fine-Tuning (`FFT`) (`B4_rslora` vs `B4_fft`):**
   - Identical verified `Qwen3-32B` teacher trajectories and Proper Top-64+Tail Logit KD (`alpha_kd=0.85`), comparing `rsLoRA (r=32, 2.0% trainable, lr=5e-5)` vs `Full Fine-Tuning (100% trainable, lr=8e-6, cosine warmup)`.
2. **Information-Dense Difficulty-Aware Curriculum (`Uniform` vs `InfoDense-Hard`) + Category Marginal Attribution:**
   - Score a `1,400`-item candidate pool with Student (`Qwen3-1.7B`) 0-shot execution/surprise and Teacher (`Qwen3-32B`) verified solutions; compare training on `500` uniform verified items vs `500` **high-learnability items** (where Student fails or exhibits high entropy AND Teacher passes deterministic verification), plus leave-one-category-out / per-category marginal tracking.
3. **Verification-Driven RLVR (`B5_rlvr_2b`):**
   - Online `GRPO` (`G=4` rollouts/prompt, deterministic execution/math/tool/JSON rewards, `kl_beta=0.04` reference anchor) on top of the winning `B4` checkpoint.

---

## 4. Compute & Budget Envelope
- Current spend: `\$39.07` / `\$180.00`.
- Planned Phase 2 compute:
  - `1 × H200 141GB` or `H100 80GB` (`~\$3.19–\$3.99/h`, ~2.5h = `~\$9.50`): `Qwen3-32B` verified teacher generation + `B4` (`rsLoRA` vs `FFT` + `Uniform` vs `InfoDense`) + `B5` (`RLVR`) + live head-to-head evaluation of `NanoLM-B4/B5`, `Qwen3-8B`, `Qwen3-32B`, and `openai/gpt-oss-120b` across Competition 1 & Competition 2!
  - `2 × RTX 5090` (`\$1.39/h`, ~1.6h = `~\$4.50`): `LAB-ARCH-04` multi-seed (`n=2`) `X5_atrd_muon` & `X6_pkm_muon`.
- Estimated Phase 2 spend: **~\$14–\$18** (cumulative ~`\$55` / `\$180`, leaving `>\$120` untouched).
