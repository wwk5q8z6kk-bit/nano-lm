# NANOLM Ω — Twenty-Four Ranked Scientific Hypotheses (`H-Ω01` to `H-Ω24`)

Each hypothesis specifies the limitation addressed, mathematical mechanism, falsifiable prediction, and explicit kill criterion (`Task E`).

---

## Architecture, Operators & Capacity Allocation

1. **`H-Ω01` (Zero-Vocab Latent Surprise Routing vs. Dual-Unembed `X8`):** — **PARTIALLY CONFIRMED / REFINED (`OMEGA-EXP-01`):**  
   *Claim & Outcome:* Eliminated `47.9%` of true `4k` FLOPs at `20.5M` (`40.91M` vs `78.56M`) and `28.0x` `32k` KV state (`1.80 MB` vs `50.54 MB`). `G4_Omega` beat `B0` (`2.1175` vs `2.1188` ID; `2.1148` vs `2.1526` `2x` OOD), while `X8` retained higher short/mid-range exact copy BPB (`2.0797`) via its 2 exact `GQA` layers at `1.92x` FLOPs, and uncalibrated internal gating diluted `EpistAUROC` (`0.3854` -> `0.5260` after multiplicative readout vs `0.8151` `X8`).
2. **`H-Ω02` (Per-Head Layer-0 `Gated-VRes` + `QK-RMSNorm` on Metriplectic Wave Synthesis):** — **CONFIRMED (`OMEGA-EXP-01` + `01-MECH`):**  
   *Claim & Outcome:* Adding `v~_l = alpha_l * v_l + (1 - alpha_l) * v_0` and `QK-RMSNorm` closed `G3_HMWR`'s `-0.0154 BPB` gap and **beat `B0_Transformer` on both ID (`2.1112` vs `2.1188`) and `2x` OOD (`2.1079` vs `2.1526`)** while beating `g4_no_vres` (`2.1508`) by `+0.0396 BPB` and `+15.1pp` `OW_4x`.
3. **`H-Ω03` (Reflection Gates `beta in (0, 2)` Necessary for Non-Abelian Group Tracking):** — **CONFIRMED (`EXP-INV-01`):**  
   *Claim:* Contractive updates (`beta <= 1`) cannot flip orientation inside finite subspaces, causing exponential errors on parity / $S_3$ / long-range needle state permutations (`EXP-INV-01`: `g2_no_reflection` `0.364` vs `0.999`).
4. **`H-Ω04` (Conditional Pass-2 `O(1)`-State Latent Refinement Improves `G4` Reasoning Without KV Growth):** — **FALSIFIED AT `20.5M` ISO-PARAM BUDGET (`OMEGA-EXP-01`):**  
   *Claim & Outcome:* Under fixed `~20.56M` parameters, redirecting FFN capacity into Pass-2 `_LatentTokenRefiner` degraded both corpus BPB (`g4_omega` `2.1175` ID / `2.1148` OOD vs single-pass `g4_no_sparse_refiner` **`2.1112` ID / `2.1079` OOD**) and structural OOD calibration (`EpistAUROC 0.5260` vs `0.6185`), making single-pass `G4` strictly Pareto-dominant among constant-state models.
5. **`H-Ω05` (Fixed Capacity Budget Breakdown Under Iso-Parameter Constraints):**  
   *Claim:* Sequence operator parameter share exceeding `~7.0 d^2` (`G2_MPRN = 8.5 d^2 -> ffn=64`) starves channel FFN transformations and degrades real corpus BPB despite high synthetic accuracy; keeping operator cost `<= 6.5 d^2` unlocks both. *(Confirmed in Cycle 3; re-verified under `g4_omega`.)*
6. **`H-Ω06` (Asymmetric Key-Shifted Addressing `CausalKeyShift` Solves Single-Layer Induction Binding):**  
   *Claim:* Convolving `(q, k, u, scam_q, scam_k)` while leaving `v` unshifted decouples contextual trigger detection from verbatim payload storage (`OW_4x`: `0.146 -> 0.977`). *(Confirmed Cycles 1–3.)*
7. **`H-Ω07` (Content-Dependent Attention-over-Depth Residuals Outperform Identity Skip Connections):**  
   *Claim:* Softmax routing over depth states $H_{0:\ell}$ plus uncertainty-gated synthesis beats flat residual addition on **both** `S3_4x` (`+0.095`) and real corpus BPB (`+0.0125`). *(Confirmed Cycle 3.)*
8. **`H-Ω08` (Fixed-Slot `SCAM-K` Circumvents Rank-`H*d_h^2` Associative Interference):**  
   *Claim:* Surprise-evicted verbatim slots maintain constant-memory long-range retrieval across arbitrary context (`Needle_OOD: 0.729 -> 0.999` with vs without SCAM).
9. **`H-Ω09` (WY Finite-Precision Chunk Bound $C < C_{\text{crit}} \approx 41.6$ Prevents Reflection Gradient Explosion):**  
   *Claim:* Choosing $C=32 < 41.6$ keeps `||(I-L)^{-1}||_2^2 < 10^29 \ll 3.4e38$ while keeping exact mathematical nilpotent inversion. *(Confirmed in `F2`.)*
10. **`H-Ω10` (Short-Horizon Inductive Bias Crossover Phenomenon `250` vs `1,000` Steps):**  
    *Claim:* Local dilated convolutions (`I3_HRWN`) exhibit artificially low `250`-step loss but saturate past step `500`, whereas structured Wave+Metriplectic+Depth recurrence (`G3`/`G4`) crosses over permanently (`+0.0312` BPB in `EXP-INV-06A`).

## Epistemic Uncertainty, Cognition & Decision-Theoretic Planning

11. **`H-Ω11` (Vocabulary-Free Character-Span Graph Encoder Breaks the `Sym(12)` Entity Ceiling):** — **CONFIRMED (`OMEGA-EXP-02`):**  
    *Outcome:* `omega_efe_planner` reached **`84.72%` ID, `95.56%` OOD (`9–12`), `83.33%` Super-OOD (`16–24`)** vs `C4b` (`5.28%`) and `C5` (`4.17%–7.78%`).
12. **`H-Ω12` (Learned Expected Free-Energy Policy Matches/Exceeds Hardcoded Action-Index Shields):** — **CONFIRMED (`OMEGA-EXP-02`):**  
    *Outcome:* Learned EFE (`0.00%` premature rate across ID/OOD/Super-OOD vs `9.17%` unshielded `C5`); ablating risk penalty (`omega_no_efe_risk`) dropped OOD `95.56% -> 75.56%` and Super-OOD `83.33% -> 68.06%`.
13. **`H-Ω13` (Directed Upstream/Downstream Belief Propagation Resolves Distractor-First Multi-Hop DAGs):** — **CONFIRMED (`OMEGA-EXP-02`):**  
    *Outcome:* Removing graph message passing (`omega_no_graph_prop`) collapsed Super-OOD resolution **`83.33% -> 35.83%`** (`-47.50pp`) and contradiction revision (`83.38% -> 12.93%`).
14. **`H-Ω14` (Retrieval-Normalized Surprise Uncertainty Fixes Positive-Orthant Precision Inversion):** — **CONFIRMED WITH BOUNDARY CONDITION (`EXP-INV-01` + `OMEGA-EXP-01-MECH`):**  
    *Outcome:* Structural `surp_score / retr_rms` maintains positive OOD calibration (`0.6185` in single-pass `G4`, `0.686` `G3_HMWR`, `0.934` `g2_no_precision`), whereas discriminative hidden probes (`score_proj(norm(h))`) fail under out-of-vocabulary covariate shift unless un-squashed by `mem_unc` (`0.3854 -> 0.5260`).
15. **`H-Ω15` (Unified Hint-Free Structural+Neural Router Dominates Legacy Keyword Routing):** — **CONFIRMED (`OMEGA-EXP-03`):**  
    *Outcome:* `omega_unified_system` reached **`100.0%` (`60/60`)** across paraphrased math, code, causal, SLW, and continual episodes vs `28.33%` (`17/60`) for `legacy_keyword_system_v2`.

## Verification, Symbolic Integration & Optimization Dynamics

16. **`H-Ω16` (Early-Halting Dual-Spec Verification Cuts Compute/Win Without Accuracy Loss):** — **CONFIRMED (`OMEGA-EXP-03`):**  
    *Outcome:* Maintained `100.0%` accuracy while cutting `avg_model_calls` (`1.883 -> 1.817`) and `cost/verified_win` (`2.325 -> 2.225` vs `4.588` legacy).
17. **`H-Ω17` (`nano/` Identity + Provenance SLW Ledger Eliminates Stale/Absent/Conflicting Hallucinations):** — **CONFIRMED (`OMEGA-EXP-03`):**  
    *Outcome:* `omega_unified_system` hit `100.0%` `abstention_and_conflict_acc` and `100%` SLW memory vs `78.33%` (`30%` SLW, `50%` abstention) for `omega_no_slw_provenance`.
18. **`H-Ω18` (Causal Link Attribution `CausalBasis` Stops Temporal-Association Confounding):** — **CONFIRMED (`OMEGA-EXP-03`):**  
    *Outcome:* `causal_reasoning = 100.0%` vs `50.0%` for `naked_pass1` and `legacy_keyword_system_v2`.
19. **`H-Ω19` (Continual Memory Preservation Zeroes Catastrophic Forgetting Across Sequential Domains):** — **CONFIRMED (`OMEGA-EXP-03`):**  
    *Outcome:* `continual_consolidation = 100.0%` vs `40.0%` without provenance (`omega_no_slw_provenance`) and `0.0%` legacy.
20. **`H-Ω20` (`Muon` Pure-CE Late-Horizon Slope Crossover vs. `AdamW`):**  
    *Claim:* Matrix orthogonalization converges slower initially at step `250` (`2.505` vs `2.443`) but exhibits steeper `-0.359 BPB` decay from `250 -> 1,000` steps (`-0.309` for `AdamW`), warranting `3,000–10,000`-step crossover testing without early `JEP` penalty (`+0.034 BPB`).
21. **`H-Ω21` (Streaming Chunked Prefill Bounding `32k` Peak VRAM Under $O(1)$ Recurrent State):**  
    *Claim:* Splitting prefill into `C_stream = 1024` sequential state updates cuts `G3/G4` `222M` `32k` peak VRAM from `7.88 GiB` to `< 1.1 GiB` while changing logits by `< 1e-5`.
22. **`H-Ω22` (Fused Triton/Metal WY+Wave Shader Recovers `>= 80%` Dense Training Throughput):**  
    *Claim:* Eliminating Python `for c in range(n_chunks)` kernel dispatch in register-resident SRAM turns `7.82 GFLOP/tok` `G3/G4` `3.89B` from memory-bound `406 ms/tok` into `> 25 tok/s` hardware speed.
23. **`H-Ω23` (`G4` Uncertainty + `SCAM` Grounding Transfer Directly to Clinical P1 `H6` Failure Gates):**  
    *Claim:* Replacing static GQA pointer trunks with `G4` native uncertainty + `SCAM` span registers directly closes `H6`'s three failed epistemic gates (`absence`, `conflict`, `uncertainty`).
24. **`H-Ω24` (Self-Verifying Synthetic Curriculum Multiplication via Exact Rational/Program Checkers):**  
    *Claim:* Training small `<4B` models solely on sandbox-verified rational/differential trajectories avoids distributional hallucination amplification common in unverified self-play.
