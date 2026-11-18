# Research Log — Autonomous Discovery Chronicle (`Cycles 1, 2 & Ω`)

## Cycle 1 — Invention of `HSPA` & Diagnosis of E2 Failures (`pod 2v2dt0j536rh58`, `L40S`, `$1.3026`)
- **Action:** Implemented `nanolm_holonomy/` (`SpinorRegisterField`, `PredictiveCodingCell`, `HolonomicRegisterSidecar`) + `ExperimentRegistry`.
- **Empirical Findings (`X1–X4`):**
  - `X2` (`14M`, 2 seeds): `Gen1_H2_HSPA_PredCode` (`0.3723 ± 0.00235` BPB) beat `B0_Transformer` (`0.3794`), `B1_Abelian_SSM` (`0.3826`), and `Gen2_H3` (`0.3803`), lifting `state` probes `13x` (`1.04% -> 13.54%`).
  - `X3` (`3.845B`): Held VRAM flat at `8,318 MB` at `16K` vs `13,824 MB` Transformer, but uncompiled PyTorch scan was `8.2x` slower in training and `3.2x` slower in decode (`135.0` vs `41.8 ms/tok`).
  - `X4`: Early-stopped SFT + harder RLVR unlocked `18/72` GRPO updates (`2/4` coding).

## Cycle 2 — Fused OpenAI Triton Adjoint, `32K` Crossover & Failure Diagnosis (`pod vboeyg7n3dk7k3`, `L40S`, `$0.7021`)
- **Theorem & Kernel:** Derived exact quaternion reverse-mode adjoint (`_raw_scan_dispatch` over time-reversed conjugated rotors storing `1` tensor). Verified `float64 gradcheck` and CUDA Triton equivalence (`rel_err < 1e-4` after fixing scale-normalized assertion in `test_holonomy.py`).
- **Empirical Findings (`C2-X0..X4`):**
  - `C2-X0`: Triton scan achieved **`11.08x–35.00x`** `fwd+bwd` speedup and **`2.33x`** lower VRAM (`529.04 ms -> 15.12 ms` at `L=4096`).
  - `C2-X2`: Across 3 seeds, `Gen1_H2_PredCode` improved to **`0.3693 ± 0.00041` BPB** at **`49,633 tok/s`** (`3.77x` over Cycle 1); `Gen3_H4_WavePred` lifted `state` probes to `15.28%` (`22.1x`) yet scored `0.3696` BPB (`Rejected vs Gen1_H2` under `ExperimentRegistry`).
  - `C2-X3`: At `32,768` tokens (`logits_to_keep=1`), **`HSPA-4B` surpassed `Transformer-4B` in BOTH decode latency (`74.264 < 75.026 ms/tok`) and peak VRAM (`9,261.8 < 13,888.0 MB`, `-4,626.2 MB`)**.
  - `C2-X4`: `NanoLM-1.7B-Holonomic-v2` reached `35/56 (62.50%)` vs external `8.19B` `Qwen3-8B` `39/56 (69.64%)`, winning `tool_use` (`4/4 vs 2/4`) and `math_logic` (`2/4 vs 1/4`), but exposing two critical gaps: (1) 3-ep standard LoRA still damaged Python `coding` (`1/4`), and (2) `SyntaxTTC` had `0` triggers because errors were runtime/arithmetic, not AST syntax.

## `NANOLM Ω` Cycle — Roofline Pre-Flight, `100%` $S_4$ `16x` OOD, Spectral Nullspace LoRA & Beating `Qwen3-8B` (`pod jiwcmt2i7qspab`, `A40`, `$0.3201`)
- **Inventions Implemented:** `ComputePlanner`, `HSPA-Ω` (`su2_lie_exponential` + `so3_projective_adjoint`), `SpectralNullspaceLoRALinear` ($\Delta W V_k \equiv 0$), and zero-oracle `OmegaCognitiveController`.
- **Empirical Breakthroughs (`EXP-Ω1..Ω3`):**
  - `EXP-Ω1`: Resolved both the $S_4$ Cayley pole and $\pm q$ sign ambiguity — `HSPA-Ω` hit **`100.00% ± 0.00%` at `L=32..512` (`16x` OOD)** while every ablation scored `1.8%–6.0%`.
  - `EXP-Ω2`: `Spectral_Nullspace_LoRA` beat both `Standard_LoRA` (`+4.4703 -> +3.8293` Dom-A forgetting) and `Proximal_L2_LoRA` (`+4.3863 -> +3.8293`) on `3/3` seeds at matched Dom-B plasticity.
  - `EXP-Ω3`: **`NanoLM-Ω-1.7B` (`1.747B`) `Naked_Greedy` achieved `40/56 (71.43%)`** (`coding=3/4, math=4/4, tools=4/4, inst=4/4`), **outperforming `4.69x` larger `Qwen/Qwen3-8B` (`39/56 = 69.64%`)** on identical single-pass greedy decoding with **`3.76x` less VRAM (`4,403` vs `16,541 MB`)**. Diagnosed why unconditional `SYMBOLIC_COMPUTE` overrides on `OmegaCognitiveController` (`39/56`) helped `dia-04` (`2/4 -> 3/4`) yet overwrote `tol-02` and `mem-01`.
