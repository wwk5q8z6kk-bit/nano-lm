# EXP-U07 Protocol — First-Principles Algorithmic & Equation-Level Enhancements (`M13`–`M16`)

- **Experiment ID:** `EXP-U07`
- **Status:** Preregistered BEFORE execution
- **Hardware / Cost:** Local Apple Silicon M3 Max (`cpu` 5 workers for Part A & A2, `mps` for Part B), `$0.00` paid cloud compute

## 1. Theoretical First-Principles Diagnosis of `EXP-U06` Gaps
`EXP-U06` established the superiority of $SO(3)$ spinor transport (`M10`, `M12`) for natural BPB and hierarchy rollups and $O(T \cdot K)$ Sparse Overwrite-Aware Pointers (`M11`, `M12`) for overwrite MQAR and `4×` real-haystack recall, while exposing three mathematical root failures:
1. **`GDN1` Chunked WY Catastrophic Cancellation (`M11 seed=0` `4×` `NaN`):** Neumann doubling $(I-L)(I+L^2)\cdots(I+L^{16})$ amplifies $|L^{16}| \approx 4.47\times 10^{12}$ under aligned keys (`code` indentation, `copy_256`, $\beta \to 1.9$), causing `~1.05e6` float32 cancellation error per chunk and exploding at `L=512`.
2. **`HSPA` Random-Copy & Rare-MQAR Gap (`copy_256`: `M12=0.642` vs `M11 s1=0.952`):** (a) `HSPALieSO3` maintains only `48` floats of continuous state per layer, information-theoretically unable to retain membership of `256` uniform IID tokens to drive `ptr_gate(h_final)`; (b) unnormalized `ptr_q · ptr_k` lacked a translation-invariant local lexical suffix similarity metric; (c) `ptr_gate(h_final)` had no explicit causal access to candidate match sharpness or base-LM uncertainty.
3. **`HSPA` Finite-Horizon Dissipation & Gauge Loss (`S4_Holonomy_8x`: `~0.05` OOD; `Hierarchy` depth ceiling):** `decay = sigmoid(0) = 0.5` yields a `1`-token half-life ($0.5^{16} = 1.5\times 10^{-5}$ gradient vanishing), $S_0 = 0 \in \mathbb{R}^3$ lacks an undamped group holonomy accumulator $q_t \in SU(2)$, local `(scale_tok, val_tok)` bigrams lack pre-mixer receptive coupling, and $3\times 3$ matrix doubling runs `6.76×` slower than a gauge-decoupled quaternion + scalar scan.

## 2. Three First-Principles Mathematical Upgrades
1. **Invention A — Exact Unit-Triangular WY Solve:** Replace Neumann doubling in `GDN1Mixer._forward_chunked_wy` with `torch.linalg.solve_triangular(I + L, I, upper=False, unitriangular=True)` (exact float32 solve; `1.57e-5` rel error even under fully aligned keys `beta=1.9, decay=0.999`).
2. **Invention B — Evidential Dual-Basis Cosine Sparse Pointer (`pointer_mode="sparse_evid"`):**
   - Bounded cosine similarity in contextual (`q_ctx · k_ctx`) and translation-invariant 4-gram lexical embedding (`u_lex · u_lex`, taps `[0.125, 0.25, 0.5, 1.0]`, symmetric `I`-initialized projection -> exact cosine `+1.0` for any unseen repeated 4-gram).
   - Explicitly causal 6-feature Bayes/MDL evidence gate `(has_exact, log_n_exact, max_cos, exact0_cos, ptr_ent, detach(base_ent))` added to `ptr_gate(ptr_norm(h_final))`.
3. **Invention C — `GaugeSU2HolonomyMixer`:** Gauge-decoupled $O(\log T)$ Kogge–Stone $SU(2)$ holonomy scan $q_t = \Delta q_t q_{t-1}$ (`6.76×` faster than $3\times 3$ matrix scan) + multi-timescale co-moving body-frame scan (`a_bias` spaced `[0.0, 1.5, 3.0, 4.5]`) + $C^\infty$ Rodrigues map + $\pm 1$-invariant 6D $SO(3)$ adjoint readout `[c1(q), c2(q)]` alongside `S_t`.

## 3. Contenders & Exact Iso-Parameter Parity (`verified`)
All contenders strictly match total params at `d=32, V=72` (`55,075–55,521`, `±0.4%`) and non-embedding params at `d=128` (`852,027–852,801`, `±0.05%`):
- `M12_HSPA_SparseOWPointer` (control from `EXP-U06`): `d=32 ffn=72 (55,075)`, `d=128 ffn=474 (non_emb=852,451)`
- `M13_HSPA_EvidCosSparsePtr` (`HSPA` + Invention B): `d=32 ffn=70 (55,499)`, `d=128 ffn=463 (non_emb=852,587)`
- `M14_GaugeSU2_Pure` (Invention C pure, no pointer): `d=32 ffn=44 (55,521)`, `d=128 ffn=462 (non_emb=852,801)`
- `M15_GaugeSU2_EvidCosSparsePtr` (Inventions B + C): `d=32 ffn=35 (55,371)`, `d=128 ffn=429 (non_emb=852,075)`
- `M16_FullSphereTri_EvidCosSparsePtr` (Inventions A + B): `d=32 ffn=68 (55,227)`, `d=128 ffn=301 (non_emb=852,027)`

## 4. Experiment Suite & Falsification Criteria
- **Part A (`cpu`, 5 workers):** `5` contenders × `5` micro-lab tasks × `3` seeds (`200` steps).
- **Part A2 (`cpu`, 5 workers, Algebraic Convergence Probe):** `S4_Holonomy_8x` & `Hierarchy_Denom_Rollup_4x` at **`600` steps × `2` seeds** across `{M0, M5, M12, M14, M15}` -> tests whether gauge-decoupled `SU(2)` holonomy breaks the 200-step algebraic optimization floor.
- **Part B (`mps`, `data/v1`, `d=128`):** `5` contenders × `1,000` steps × `2` seeds (`B=8, T=128`), evaluated at **`1× (128)`, `2× (256)`, `4× (512)`** BPB, `copy_{32..256}`, and real-haystack MQAR (`common` & `rare_tail` at `T128, T256, T512`).
- **Success / Falsification Criteria:**
  - **Criterion A (Triangular WY Stability):** `M16` produces `0` `NaN`/`Inf` values across every training step and `1×/2×/4×` eval probe on both seeds.
  - **Criterion B (Evidential Cosine Pointer Closes Copy & Rare MQAR Gap):** `M13` and `M15` lift `copy_256` from `M12`'s `0.6416` to **`>= 0.90`** and `T512_N4` rare MQAR from `0.4375` to **`>= 0.60`** while keeping `MQAR_Overwrite_4x` `>= 0.92` and `Unseen_OpenVocab_Copy >= 0.95`.
  - **Criterion C (Gauge-Decoupled $SU(2)$ Holonomy & Multi-Timescale Win):** `M14` and `M15` markedly lift `S4_Holonomy_8x` (200-step & 600-step probe) and `Hierarchy_Denom_Rollup_4x` above `M5`/`M12` (`0.054` / `0.262`), beat `M12` MPS throughput (`> 6,260 tok/s`), and match or beat `M12` natural BPB across `web/textbook/math/code`.
