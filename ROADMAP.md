# NanoLM Architecture & Intelligence Roadmap (`HSPA-Ω -> Next Horizon`)

See [NANOLM_RESEARCH_BACKLOG.md](NANOLM_RESEARCH_BACKLOG.md) for the full 24-hypothesis ranked portfolio.

---

## Completed Milestones (`Cycles 1, 2, Ω & Ω-v2`)
- **Fused Non-Abelian Triton Engine (`C2-X0`, `C2-X2`):** `35.00x` speedup over PyTorch Blelloch scan (`L=4096`), `O(1)`-activation conjugated reverse adjoint, and multi-seed BPB dominance (`0.3693` vs `0.3798`).
- **`32,768`-Token `3.845B` Crossover (`C2-X3`):** Flat `9,262 MB` VRAM & `74.26 ms/tok` decode beating `Transformer-4B` (`13,888 MB` & `75.03 ms/tok`).
- **Exact `16x` OOD $S_4$ Composition (`EXP-Ω1`) & Full-LM $S_4$ Resolution (`EXP-Ω4`):** `HSPA-Ω` hit `100.00%` up to `L=512`, and `Gen4_Omega_SO3_PredCode` solved full-LM unbroken $S_4$ (`100.00%` `L32..128`, `93.06%` `8x` OOD `L256`; honest caveat: `res_mix_gate=0` in `7/8` layers trailed `Gen1_H2` on 800-step LM BPB `0.3775` vs `0.3701` in `EXP-Ω5`).
- **Zero-Overhead Nullspace Folding + Static CUDA-Graph Decode (`EXP-Ω6`):** `3.10x` decode at `1.808B` (`51.023 ms/tok`), `2.35x` at `3.841B <= 4B` (`100.291 ms/tok`), `7.55x` `[2048,2048]` linear GEMM, `1.61x` HF `1.747B` end-to-end (`6/6` exact match across `196` layers).
- **Spectral Nullspace Anti-Forgetting + `OmegaCognitiveController-v2` (`EXP-Ω2`, `EXP-Ω3`, `EXP-Ω7`):** Cut forgetting (`+4.47 -> +3.83`), lifted `NanoLM-Ω-1.7B` (`Naked_Greedy`) to `40/56 (71.43%)`, and upgraded to **`OmegaCognitiveController-v2` at `41/56 (73.21%, diagrams 4/4, tools 4/4, coding 3/4, cap/B=0.4190)`** on Free Kaggle `T4` (`$0.00 USD`).

---

## Immediate Next Priorities
1. **Priority #1 — Progressive `res_mix_gate` Schedule (`Gen4-Progressive`):**
   Replace static `res_mix_gate=0.0` initialization on layers `1..L-1` with a progressive unfreezing/cosine schedule so `Gen4` preserves `100%` $S_4$ algebraic composition at layer 0 while beating `Gen1_H2`'s `0.3701` general-text BPB (`EXP-Ω5` non-promotion gap).
2. **Priority #2 — End-to-End `StaticGraphHSPADecoder` Controller Integration:**
   Wire captured CUDA-Graph step decode directly into `OmegaCognitiveController-v2` multi-turn inference and benchmark BF16 `A40`/`L40S` (`40/56 -> 44+/56`).
3. **Priority #3 — `3:1` Hybrid `HSPA-Ω` + `Diff-GQA` Multi-Billion-Token Pretraining (`125M -> 350M`, Free Kaggle `T4 x 2` + `Muon`):**
   Embed `HSPA-Ω` (`LieExp + SO(3)`) + `PredictiveCodingCell` + `Diff-GQA` sparse anchors at `125M–350M` using `Muon` + `Top-2 Margin` routing.
