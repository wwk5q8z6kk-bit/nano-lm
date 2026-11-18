# Experiment U-02 — Cross-Paradigm Mechanistic Showdown (Part A Micro-Lab & Part B Real Corpus)

- **Experiment ID:** `EXP-U02`
- **Status:** Preregistered before execution
- **Hardware / Cost:** Local Apple Silicon M3 Max (`cpu` / `mps`), `$0.00` paid compute
- **Parameter Ceiling:** Strict iso-parameter parity in Part A (`~55.5K–57.5K` trainable parameters across all `10` mechanisms, `±2%`).

## Competing Mechanisms Evaluated (`M0..M9`)

1. `M0_Dense_GQA_RoPE` — Causal Self-Attention with RoPE ($\theta=10{,}000$) + QK-Norm + SwiGLU baseline.
2. `M1_GDN1_Cone_Closed` — Exact `NextGenDeltaMixer` physics: `ShortConv(F.silu)` positive-orthant keys, rank-1 Householder $\beta \in (0,1)$, scalar decay, zero-init closed `F.silu(W_g x)` gate.
3. `M2_NanoOmega_FullSphere_SFOR` — Linear conv + zero-mean L2 keys (`FullSphere`, **H-08**), $\beta \in (0,2)$, open `+1.5` `SFOR` gate (**H-13**).
4. `M3_KDA_DeltaProduct2_AttnRes` — **Candidate Hybrid Mechanism (`H-09 + H-34`):** Rank-2 Householder `DeltaProduct-2` ($\beta_1,\beta_2 \in (0,2)$) + channelwise `KDA` decay + selective Delta write + `SFOR` + `Block AttnRes` depth softmax aggregation.
5. `M4_BOBO_DualFastOp` — Bilinear dual-probe outer-product coincidence (`BOBO`-style, **H-04**).
6. `M5_HSPA_LieSO3` — Exact $SU(2) \to SO(3)$ Rodrigues quaternion spinor transport (**H-11**).
7. `M6_NEBN_BayesFilter` — Exponential-family natural-parameter Bayes precision filter (**H-16**).
8. `M7_SkewCayley_Flow` — Continuous-time skew-symmetric minus dissipative Cayley ODE step (**H-10**).
9. `M8_TRM_Recursive_Loop` — Shared `2`-layer recursive core iterated `K=3` loops with deep supervision, evaluated raw (`step 3`), with **Monotonic Margin Guard** (**H-31**), and with **LoopCD** (**H-35**).
10. `M9_DualChannel_Pointer` — `M2_NanoOmega_FullSphere_SFOR` backbone + strictly causal content copy-pointer channel (**H-23**).

## Four Decisive Tasks & Kill Criteria (Part A, `n=3` Seeds)

1. **`S4_Holonomy_8x`**: Non-abelian $S_4$ generator product prediction; train `L=24`, eval ID (`L=24`) & **`8×` OOD (`L=192`)**. Tests algebraic composition (`H-09`, `H-10`, `H-11`, `H-30`).
2. **`MQAR_Selective_Overwrite_4x`**: Multi-query key-value stream with `>=35%` value overwrites amidst distractors; train `T=48`, eval ID (`T=48`) & **`4×` OOD (`T=192`, `N_writes=40`)**. Reports overall AND specifically `overwritten_key_acc`.
3. **`Unseen_OpenVocab_Copy`**: Train literal copying strictly on token IDs `[4..43]`; eval both ID (`[4..43]`) AND completely **`Unseen` held-out sub-vocab `[44..67]`** (`0` training exposures!). Tests **H-23** (`DualChannel_Pointer` vs pure weights).
4. **`Parity_Dyck_4x`**: Joint binary parity + clamped Dyck parenthesis depth state tracking; train `T=40`, eval ID (`T=40`) & **`4×` OOD (`T=160`)**.

- **Causality Gate:** Every mechanism must pass `verify_strict_causality()` (`max_future_leak < 1e-6`) before training starts.
- **Scale-Up Rule:** Top 4–5 genuinely distinct winners/controls advance to **Part B (`>=1,000` steps)** on real subword corpus `/Volumes/Express1M2/nanolm-lab/data/v1/`.
