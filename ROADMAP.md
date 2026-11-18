# NanoLM Architecture & Intelligence Roadmap (`HSPA-Ω -> Next Horizon`)

See [NANOLM_RESEARCH_BACKLOG.md](NANOLM_RESEARCH_BACKLOG.md) for the full 24-hypothesis ranked portfolio.

---

## Completed Milestones (`Cycles 1, 2 & Ω`)
- **Fused Non-Abelian Triton Engine (`C2-X0`, `C2-X2`):** `35.00x` speedup over PyTorch Blelloch scan (`L=4096`), `O(1)`-activation conjugated reverse adjoint, and multi-seed BPB dominance (`0.3693` vs `0.3798`).
- **`32,768`-Token `3.845B` Crossover (`C2-X3`):** Flat `9,262 MB` VRAM & `74.26 ms/tok` decode beating `Transformer-4B` (`13,888 MB` & `75.03 ms/tok`).
- **Exact `16x` OOD $S_4$ Composition (`EXP-Ω1`):** `HSPA-Ω` (`su2_lie_exponential + so3_projective_adjoint`) achieved `100.00% ± 0.00%` up to `L=512`.
- **Spectral Nullspace Anti-Forgetting (`EXP-Ω2`, `EXP-Ω3`):** Cut forgetting across `3/3` seeds (`+4.47 -> +3.83`) and lifted `NanoLM-Ω-1.7B Naked_Greedy` to **`40/56 (71.43%)`**, outperforming `4.69x` larger `Qwen/Qwen3-8B` (`39/56 = 69.64%`).

---

## Immediate Next Priorities
1. **Priority #1 — Uncertainty-Gated `OmegaCognitiveController-v2`:**
   Require schema-key & syntactic validity preservation before `SYMBOLIC_COMPUTE` overwrites an already valid JSON answer (fixing the `tol-02` / `mem-01` regressions so `Cognitive_System` strictly beats `40/56`, targeting `44+/56`).
2. **Priority #2 — Persistent CUDA-Graph Decode & Triton Projection Fusion (`H-Ω20`):**
   Eliminate eager per-layer Python launch overhead on `HSPA_4B` (`~74 ms/tok -> < 12 ms/tok`) and speed up `SpectralNullspaceLoRA` inference by folding `P_perp = I - V_k V_k^T` directly into merged `W_eff` at eval time (**0% inference overhead** when frozen!).
3. **Priority #3 — `3:1` Hybrid `HSPA-Ω` + `Diff-GQA` Multi-Billion-Token Pretraining (`125M -> 350M`, Free Kaggle `T4 x 2` + `Muon`):**
   Embed `HSPA-Ω` (`LieExp + SO(3)`) + `PredictiveCodingCell` + `Diff-GQA` sparse anchors at `125M–350M` using `Muon` + `Top-2 Margin` routing.
