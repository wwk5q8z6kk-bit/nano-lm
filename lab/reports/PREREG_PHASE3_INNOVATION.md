# Pre-Registration: Phase 3 — Epistemic Margin Routing, Latent Belief Planning, & `<4B` vs `120B` System v2 (`LAB-COG-02`, `LAB-ARCH-05`, `LAB-SYS-02`, `LAB-TRACKB-03`)

**Timestamp:** 2026-11-17  
**Branch:** `nanolm/discovery-lab-v1`  
**Remaining Autonomous Budget Before Phase 3:** `\$106.04` (`\$73.96 / \$180.00` spent; Phase 3 cap `<= \$18.00`)

## 1. Hypotheses Under Test

1. **`H_COG_02` (`C4` Uncertainty-Gated Latent Belief-Space Planning + Graph $\alpha$-Equivariance):**
   - *Mechanism:* Iterating $L=3$ uncertainty-gated belief-graph message-passing steps over `[Z_t; M_mat]` ($O((K+N)^2 d)$ FLOPs) + 3-token predicate-context source-span pointers + epistemic sufficiency gating (`C4a_ulbp_latent_plan`), combined with training-time module-symbol $\alpha$-renaming $\pi \in \text{Sym}(12)$ (`C4b_ulbp_equivariant`).
   - *Success / Falsification Criterion (`n=3` seeds):* Must exceed `C3_ebwm_counterfactual` (`30.6%` OOD full resolution, `70.4%` value exactness) by `>= +20 pp` on Standard-OOD (`7–9` modules) AND achieve `>= 50%` full verified resolution on Extreme-OOD (`10–12` modules) where non-equivariant models collapse.

2. **`H_ARCH_05` (`X8_margin_atrd_muon` — Top-2 Hypothesis-Margin Routing vs Shannon Tail Dispersion):**
   - *Mechanism:* Replacing vocabulary-size-sensitive Shannon entropy $H_t$ in `CausalEntropyRefiner` (`X5_atrd_muon`) with vocabulary-invariant Top-2 probability margin ambiguity $u_t^{\text{margin}} = 1 - (p_{t,(1)} - p_{t,(2)}) \in [0, 1]$, margin-weighted copy protection, and 32-bucket aligned `max_k` gather shapes.
   - *Success / Falsification Criterion (`n=2` seeds, `600M` tokens each, `~29.4M` active non-emb):* Match or beat `X4_nanolm_v2_muon` (`0.9575` BPB) while cutting 512-tok OOD random copy loss by `>= 50%` relative to `X5_atrd_muon` (`3.065 -> <= 1.50` nats) with zero 24GB allocator OOMs.

3. **`H_SYS_02` (`NanoLMSystem-v2` — Belief-Revision Working Memory + Error-Triggered `think=1` + Dual-Channel Math Arbiter):**
   - *Mechanism:* (a) Superseding invalidated state keys and escalating `AGENTIC_TOOL_LOOP` to `think=True` strictly on turns with unresolved API errors; (b) running unanchored analytical `think=True` + unanchored Python execution in `COMPUTE_VERIFY_MATH`, invoking a neutral arbiter only upon disagreement.
   - *Fairness Rule:* Evaluated under **100% identical system code** across `1.72B`, `3.08B`, `8B`, `32B`, and `120B` (`openai/gpt-oss-120b`), strictly separated from Competition 1 (Naked).

4. **`H_TRACKB_03` (`NanoLM-B6-2B` L2-Anchored Stratified FFT+KD+RLVR & Stage-3 `NanoLM-B6-3B`):**
   - *Mechanism:* Enforcing a `>= 22%` per-category stratified quota (including disjoint multi-turn API error-recovery trajectories) verified from `openai/gpt-oss-120b` + `Qwen3-32B`, trained via Pretrained-Weight L2-Anchored `100%` FFT (`\lambda_0 = 0.05`) + Top-64+Tail KD + `G=8` online `GRPO` at `1.72B` (`NanoLM-B6-2B`), plus Stage-3 scaling transfer onto `SmolLM3-3B` (`3.08B < 4.0B`, `NanoLM-B6-3B`).
