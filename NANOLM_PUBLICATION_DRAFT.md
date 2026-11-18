# NanoLM Ω: Single-Pass Wave-Metriplectic State Transport, Open-Entity Free-Energy Planning, and Provenance-Verified Neural Reasoning under Sub-4B Constraints

**Draft Type:** Reproducible Research Paper Manuscript (`nanolm/architecture-invention-v1`, `$0.00` Cloud Spend)

---

## Abstract
Compact language models under a strict `4B` parameter limit face three bottlenecks: quadratic $O(T)$ KV cache growth in standard Transformers, closed-entity permutation limits and brittle penalty hacks in neural planners, and disconnected symbolic verification during tool-assisted reasoning. Evaluating all candidates under strict iso-parameter bounds (`±0.5%`), exact FLOP accounting, and zero-leakage train–step duality (`<= 1.91e-6`), we investigate combining multi-scale causal wave surprise, $\beta \in (0, 1.98)$ orientation-reversing Householder WY state transport, surprise-evicted `SCAM-32` episodic slots, Layer-0 Per-Head `Gated-VRes`, and `QK-RMSNorm`. On `4.10M`-token multi-domain training (`Kaggle 2x Tesla T4`), single-pass **`G4_NoSparseRefiner`** achieves **`2.1112 ± 0.0081` ID BPB** (beating Transformer `B0` at `2.1188`) and **`2.1079 ± 0.0082` `2x`-OOD BPB** (beating `B0` `2.1526` by `+0.0447 BPB`) with **`1.80 MB`** `32k` state (`28.0x` smaller than hybrid `X8` at `50.54 MB`), while cleanly falsifying iso-parameter conditional Pass-2 latent recycling (`2.1175` BPB) and dual vocabulary unembedding (`78.56M` vs `41.58M` `4k` FLOPs). Outside sequence modeling, a vocabulary-free character-span graph planner with learned Expected Free Energy (`OmegaEFEPlanner`) reaches **`83.33%` Super-OOD** verified repair across `N = 16..24` dependencies (`0.00%` premature regression) where `Sym(12)` baselines collapse to `4.17%–5.28%`, and wiring `nano/` provenance into early-halting dual-spec verification reaches **`100.0%` (`60/60`)** hint-free accuracy at **`2.225` cost/win** vs `28.33%` (`4.588`) for keyword-routed orchestration.

---

## 1. Introduction & Forensic Motivation
Audit of prior hybrid iterations (`X8_margin_atrd`) exposed uncounted dual vocabulary GEMMs (`1.81x–2.65x` throughput slowdown) alongside contractive delta rules incapable of finite non-Abelian state flips (`S3_4x ~ 0.172`). Simultaneously, prior cognitive planners (`C4b/C5`) hardcoded `12` entity embeddings (`M0..M11`) and environment-specific action penalties (`suff_pen[4]/suff_pen[6]`). `NANOLM Ω` tests whether unifying exact recurrent physics, value-residual stability, entity-unbounded relational planning, and formal `nano/` epistemic contracts resolves all three bottlenecks simultaneously without cloud overspend.

---

## 2. Mathematical Foundations & Guarantees
1. **Orientation-Reversing WY Metriplectic Recurrence (`G3`/`G4`):** Half-subspace reflections $(I - \beta_t u_t u_t^\top)$ ($\beta_t \in (0, 1.98)$) composed with dissipative writes satisfy unconditional $\|A_t\|_2 \le 1$, maintain IEEE-754 finite backpropagation strictly below $C_{\text{crit}} \approx 41.6$ (`min(chunk_size, 16)` in `G4`), and pair with `CausalKeyShift` (`[0.7, 0.3]`) and `SCAM-32` for exact `< 1.91e-6` batched–autoregressive equivalence.
2. **Layer-0 Per-Head `Gated-VRes` + `QK-RMSNorm`:** Mixing depth-0 values via $v_t^{(\ell)} = \alpha_t^{(\ell)} v_{\text{raw},t}^{(\ell)} + (1-\alpha_t^{(\ell)}) v_t^{(0)}$ preserves clean lexical identity across compressed recurrent depths without adding $O(T)$ KV growth.
3. **Smoothed Surprise & Honest Accounting:** Using $s_t = \sqrt{\|x_t^{\text{re}}-\bar{w}_{t-1}\|_{\text{rms}}^2 + 10^{-6}} + \operatorname{softplus}(\cdot)$ avoids `sqrt(0)` backward singularities, while single-unembed `G4` uses `1` $|V|d$ pass (`41.58M` `4k` FLOPs at `20.56M`) vs `X8`'s `2` passes (`78.56M`).
4. **Open-Entity Graph & EFE Control:** Character-convolved node embeddings feed 3-hop upstream/downstream GRU belief propagation and policy logits $\hat{V}_a + \beta_{\text{epist}}(\hat{u})\hat{I}_a - \lambda_{\text{risk}}(\hat{u})\hat{R}_a$.
5. **Symbolic–Neural Unification:** `OmegaUnifiedCognitiveSystem` enforces `nano.kernel.Identity.may_join`, `EpistemicStatus` (`DIRECT_DOCUMENTATION`, `SUPERSEDED`, `CONFLICTING`, `NOT_FOUND`), `CausalBasis`, and early halting when `Channel A == Channel C`.

---

## 3. Empirical Results & Honest Falsifications

### 3.1 Multi-Domain CUDA Tournament (`20.55M`, Kaggle `2x Tesla T4`, `1,000` Steps × `2` Seeds)
- **Single-Pass Winner (`g4_no_sparse_refiner`):** `2.1112 ± 0.0081` ID BPB, `2.1079 ± 0.0082` `2x` OOD BPB (`web=1.8500`, `textbook=1.6771`, `math=2.4438`, `code=2.4741`), `41.58M` FLOP/tok, `1.80 MB` `32k` state (`7,398.6 tok/s`, `0` skipped grads). Beats `B0_Transformer` (`2.1188` ID / `2.1526` OOD from `EXP-INV-06A`) and Cycle 3 `G3_HMWR` (`2.1342` / `2.1322`).
- **Ablation Proof of `Gated-VRes`:** Removing `Gated-VRes` (`g4_no_vres`) drops BPB to `2.1508 ± 0.0146` ID / `2.1480 ± 0.0142` OOD (`-0.0396 BPB`) and loses `-15.1pp` `4x` OOD Overwrite (`0.7339` vs `0.8850`).
- **Falsification of `H-Ω04` (Pass-2 Latent Refiner):** Adding Pass-2 `_LatentTokenRefiner` (`g4_omega`) degrades corpus BPB (`2.1175 ± 0.0032` ID / `2.1148 ± 0.0032` OOD) and dilutes structural `EpistAUROC` (`0.5260` vs `0.6185`, even after rescuing uncalibrated sign inversion from `0.3854` via multiplicative readout $u^{\text{mem}}(0.5 + \sigma(z^{\text{lat}}))$).
- **Hybrid Pareto Point (`x8_dual_unembed_atrd`):** Achieves `2.0797 ± 0.0066` ID / `2.0759` `2x` OOD BPB via 2 full `GQA` layers at `1.89x` FLOPs (`78.56M`) and `28.0x` `32k` KV cache (`50.54 MB`).

### 3.2 Open-Entity Super-OOD Cognition (`OMEGA-EXP-02`, `248k`, `3` Seeds)
`omega_efe_planner` achieves `84.72%` ID, `95.56%` OOD (`9–12`), and **`83.33%` Super-OOD (`16–24`)** with `0.00%` premature patches across all regimes, crushing `c4b_sym12` (`5.28%` Super-OOD) and `c5_shielded` (`4.17%`); ablating graph propagation collapses Super-OOD to `35.83%`, and ablating EFE risk drops OOD to `75.56%`.

### 3.3 Unified `10`-Capability Benchmark (`OMEGA-EXP-03`, `60` Unlabelled Tasks)
`omega_unified_system` achieves **`100.0%` (`60/60`)** and **`100.0%` abstention/conflict** (`1.817` calls, `2.225` cost/win) vs `legacy_keyword_system_v2` (`28.33%`, `4.588` cost/win, `0%` abstention) and `omega_no_slw_provenance` (`78.33%`).

---

## 4. Limitations & Open Work
1. **Systems Fusion & Prefill Streaming:** Without custom Triton/Metal WY+Wave kernels, `222M` unchunked `32k` prefill peaks at `7.88 GiB` (`EXP-INV-06B`); `1024`-tok streaming prefill and fused shaders (`Cycle 4A`) are the immediate prerequisite to pretraining past `41M` tokens (`Cycle 4B`).
2. **Pretraining Scale & Clinical Separation:** Multi-domain corpus experiments remain `20.55M` / `4.10M`-token horizon checks (`Flagship 3.884B` verified for structural census, not pre-trained here); `Omega` cognition/system benchmarks measure deterministic/synthetic tasks and make no clinical NanoScribe deployment claims.
