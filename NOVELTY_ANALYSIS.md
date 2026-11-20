# Novelty Analysis — How `HSPA-Ω` Differs From Prior Sequence & Post-Training Paradigms

| Dimension | Standard Transformer (GQA/RoPE) | Mamba / GLA / Abelian SSM | Householder / DeltaNet | **`HSPA` & `HSPA-Ω` (This Branch)** |
|---|---|---|---|---|
| **State Transition Group** | Non-stateful attention (all-to-all $O(T^2)$) | Commutative diagonal / real decay ($S_4$ collapses to $\mathbb{Z}_2$) | $O(d_h^2)$ low-rank rank-1 reflections | **Compact $SU(2)^G \to SO(3)$ quaternionic holonomy** (`4`-float spinor / fiber) |
| **Algebraic $S_4$ `16x` Extrapolation** | `3.91%` (`TC^0` collapse) | `5.99%` (`U(1)` abelian collapse) | $d_h\times d_h$ matrix state required | **`100.00% ± 0.00%` (`EXP-Ω1`, `5,784` params)** |
| **Memory & KV Cache @ `32,768` (`3.85B`)** | `3,072.0 MB` KV (`13,888 MB` total) | $O(1)$ vector state | $O(1)$ matrix state $d_h \times d_h$ | **`1.781 MB` state (`1,725x` smaller; `9,262 MB` flat VRAM; faster decode `74.26 < 75.03 ms`)** |
| **Parallel Training Backward** | FlashAttention recompute | Scan buffers or Pallas/CUDA | Chunk WY triangular solve | **Exact time-reversed conjugate spinor scan saving `1` tensor (`35.00x` Triton speedup)** |
| **Channel / Inference Block** | Static single-pass `SwiGLU` | Static `SwiGLU` | Static `SwiGLU` | **2-step precision-weighted `PredictiveCodingCell` (`0.3693` vs `0.3798` BPB)** |
| **Adapter Continual Plasticity** | Standard LoRA (overwrites coding) | Full FT / standard adapters | Standard LoRA / FFT | **`SpectralNullspaceLoRA` ($\Delta W V_k \equiv 0$, `-0.641` forgetting, `coding 0/4 -> 3/4`)** |
