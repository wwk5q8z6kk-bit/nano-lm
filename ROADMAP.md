# NanoLM Architecture Research Roadmap

Guided strictly by experimental evidence from [EXPERIMENTS.md](EXPERIMENTS.md), [PERFORMANCE.md](PERFORMANCE.md), and [NOVELTY_ANALYSIS.md](NOVELTY_ANALYSIS.md), keeping total parameter count strictly **`<= 4,000,000,000` (`4B`)**.

---

## Generation 1 (Completed) — `HSPA_Spinor` & `HSPA_PredCode` (`BEST` Promoted)
- Proved non-abelian $SU(2)$ spinor transport beats matched commutative $U(1)$ SSM (`0.3800` vs `0.3826 BPB`).
- Discovered that combining $SU(2)$ register transport with **precision-weighted `PredictiveCodingCell` (`Gen1_H2`)** beats Transformer baseline on 2 seeds (`0.3723` vs `0.3794`), yields a `13x` state-probe gain (`13.54%` vs `1.04%`), and holds 4B-class persistent VRAM flat at `8.32 GB` (`1.78 MB` state vs `1,536 MB` KV at `16K`).

---

## Generation 2 (Evaluated & Revised)
- **Completed negative ablation:** `Gen2_H3` tested Wavepacket Conv + Kuramoto coupling *without* `PredictiveCodingCell` (`0.3803` -> Rejected cleanly).
- **Generation 2B (Immediate Next Architectural Experiment):** Combine verified winner `PredictiveCodingCell` (`Gen1_H2`) WITH `ShortConv(k=4)` causal wavepacket pre-filter (`Gen2_H3_WavePred`) in a controlled 3-seed study against `Gen1_H2` (`0.3723`) to test whether local syntax convolution + predictive coding compounds both `code` (`0.2014` Transformer gap) and `state` (`0.4438`) simultaneously.

---

## Kernel & Systems Milestone (Critical Path to 4B Pretraining)
1. **Fused Triton `hamilton_product` + Hillis-Steele Scan Kernel:** Eliminate the `~8.2x` Python training slowdown and `~3.2x` BF16 4B decode gap by computing the associative $SU(2)$ tree scan and chunk boundaries inside GPU SRAM/registers (`float32` accumulation, `bf16` I/O), cutting activation VRAM from `6.7 GB` to `< 1.8 GB` at `14M` scale and targeting `< 25 ms/tok` at `4B` scale.
2. **Always-Chunked Prefill Boundary:** Enforce `chunk_size = 512–1024` universally even at `T = 512` (removing the transient `10.88 GB` X3 `T=512` spike down to `~8.3 GB`).

---

## Scaling & Foundation Post-Training Milestones (`<= 4B`)
1. **Extended `S_4` & Symbolic Algorithmic Curriculum (`X1-v2`):** Train fused-kernel `Gen1_H2` and `B0_Transformer` to convergence (`5,000+` steps on `8M` params with length curriculum) to test whether non-abelian `SU(2)` achieves `>90%` compositional length generalization on non-commutative groups where Transformers degrade.
2. **Pretraining Scale Ramp (`14M -> 350M -> 2.0B -> 3.85B HSPA-PredCode`):** Validate scaling exponent on multi-billion-token real open code/math/web corpora (`<= 4B` strict cap) before making any capability comparison against larger Transformer checkpoints.
3. **KL-Stabilized Foundation GRPO (`X4-v2`):** Tune GRPO `kl_beta` (`0.08` vs `0.03`) and merge `HolonomicRegisterSidecar` weights or `torch.compile` step functions so coding (`2/4` -> `3/4+`), math (`4/4`), structured JSON (`3/4`), and planning (`1/4+`) hold simultaneously without throughput loss.
