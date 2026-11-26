# NanoLM Architecture & Intelligence Roadmap (`HSPA-Ω -> Next Horizon`)

See [NANOLM_RESEARCH_BACKLOG.md](NANOLM_RESEARCH_BACKLOG.md) for the full 24-hypothesis ranked portfolio.

---

## Completed Milestones (`Cycles 1, 2, Ω & Ω-v2`)
- **Fused Non-Abelian Triton Engine (`C2-X0`, `C2-X2`):** `35.00x` speedup over PyTorch Blelloch scan (`L=4096`), `O(1)`-activation conjugated reverse adjoint, and multi-seed BPB dominance (`0.3693` vs `0.3798`).
- **`32,768`-Token `3.845B` Crossover (`C2-X3`):** Flat `9,262 MB` VRAM & `74.26 ms/tok` decode beating `Transformer-4B` (`13,888 MB` & `75.03 ms/tok`).
- **Exact `16x` OOD $S_4$ Composition (`EXP-Ω1`) & Full-LM $S_4$ Resolution (`EXP-Ω4`):** `HSPA-Ω` hit `100.00%` up to `L=512`, and `Gen4_Omega_SO3_PredCode` solved full-LM unbroken $S_4$ (`100.00%` `L32..128`, `93.06%` `8x` OOD `L256`; honest caveat: `res_mix_gate=0` in `7/8` layers trailed `Gen1_H2` on 800-step LM BPB `0.3775` vs `0.3701` in `EXP-Ω5`).
- **Zero-Overhead Nullspace Folding + Static CUDA-Graph Decode (`EXP-Ω6`):** `3.10x` decode at `1.808B` (`51.023 ms/tok`), `2.35x` at `3.841B <= 4B` (`100.291 ms/tok`), `7.55x` `[2048,2048]` linear GEMM, `1.61x` HF `1.747B` end-to-end (`6/6` exact match across `196` layers).
- **Spectral Nullspace Anti-Forgetting + `OmegaCognitiveController-v2` (`EXP-Ω2`, `EXP-Ω3`, `EXP-Ω7`):** Cut forgetting (`+4.47 -> +3.83`), lifted `NanoLM-Ω-1.7B` (`Naked_Greedy`) to `40/56 (71.43%)`, and upgraded to **`OmegaCognitiveController-v2` at `41/56 (73.21%, diagrams 4/4, tools 4/4, coding 3/4, cap/B=0.4190)`** on Free Kaggle `T4` (`$0.00 USD`).
- **Cycle 4 First-Principles Interaction Intelligence (`EXP-FP1..EXP-FP4`, `$0.00` Compute):** Proved `201x` bitemporal exact value recall via `CausalSourceCopyGate` (`96.67%` vs `0.48%` nocopy), validated all `8` primitives in $\mathcal{U}_t$ + `VoI` Pareto dominance (`100% @ 3.375` steps), and demonstrated `WorldController_v3` beating `3x` `EqualCompute_SC3` (`15/30 = 50.0%` vs `9/30 = 30.0%`) on held-out `FP-5Axis-30` while preserving `41/56 (73.21%)` on `benchmark_15d`.

---

## Immediate Next Priorities (Cycle 5 — Kaggle / Google Cloud / RunPod)
1. **Priority #1 — `Gen1_H2_PredCode` + `CausalSourceCopyGate` Fusion (`BEST` BPB + `96.7%` Copy):**
   Port the `O(d)` FP32-safe `CausalSourceCopyGate` directly onto `Gen1_H2_PredCode` (which retained `BEST` BPB under both `AdamW` and `Muon`) and test progressive `res_mix_gate` unfreezing to unify `S_4` OOD, macro BPB, and lossless bitemporal recall.
2. **Priority #2 — Structural Causal Model (`do()`-Calculus) Verifier for `WorldController_v4`:**
   Close the `1/6` `causal_vs_associational` gap on `FP-5Axis-30` by adding zero-oracle causal-graph backdoor/frontdoor adjustment verification alongside `AppendOnlySourceLedger`.
3. **Priority #3 — `3:1` Hybrid `HSPA-Ω` + `Diff-GQA` Pretraining (`125M -> 350M`, Google Cloud / Kaggle / RunPod):**
   Scale `HSPA-Ω` (`LieExp + SO(3)`) + `PredictiveCodingCell` + `CausalSourceCopyGate` + `Diff-GQA` sparse anchors at `125M–350M` using `Muon` + `Top-2 Margin` routing.
