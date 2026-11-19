# EXP-U06 Protocol — Spinor-Pointer Synthesis (`M10`–`M12`) & $O(\log T)$ Parallel `HSPA-LieSO3` Scan

- **Experiment ID:** `EXP-U06`
- **Status:** Preregistered BEFORE execution
- **Hardware / Cost:** Local Apple Silicon M3 Max (`cpu` 5 workers for Part A, `mps` for Part B), `$0.00` paid cloud compute

## 1. Hypothesis Being Tested
In `EXP-U02`, **`M5_HSPA_LieSO3`** ($O(T)$ exact $SU(2) \to SO(3)$ spinor transport) led **`Hierarchy_Denom_Rollup_4x` (`0.2588`)** and **real `textbook@128` (`1.9336`)**, whereas **`M9_DualChannel_Pointer`** (`FullSphere + SFOR + Dense O(T^2) Pointer`) led **`Unseen_OpenVocab_Copy` (`0.9619`)**, **`MQAR_Overwrite_4x` (`0.5764`, `ov=0.5421`)**, **`code@256` (`2.1983`)**, and **`math@256` (`2.3172`)**, but suffered:
1. $O(T^2)$ pointer score matrix cost (`[B, T, T]`),
2. Overwrite blurring in `MQAR_Overwrite_4x` (`ov=0.5421` vs static unseen `0.9619`) because plain softmax splits weight across repeated past key positions, and
3. Depth-wise key rank contraction in its `FullSphere` backbone (`10.65 -> 5.31 / 32`).

We test the **2×2 Factorial Synthesis (`{FullSphere, HSPA_LieSO3} × {Dense O(T^2) Pointer, Sparse Overwrite-Aware O(T·K) Pointer}`)** against pure controls **`M0_Dense_GQA_RoPE`** and **`M5_HSPA_LieSO3`**:
- `M10_HSPA_DensePointer` (`HSPA` parallel scan + `M9` dense pointer)
- `M11_FullSphere_SparseOWPointer` (`FullSphere` + sparse recency-ranked `K=32` overwrite-aware pointer)
- `M12_HSPA_SparseOWPointer` (`HSPA` parallel scan + sparse recency-ranked `K=32` overwrite-aware pointer)

## 2. Exact Algorithmic Upgrades Verified Before Run
- **`HSPALieSO3Mixer` $O(\log T)$ Kogge–Stone Parallel Scan:** Exact affine matrix doubling over $3\times 3$ Rodrigues rotations (`test_hspa_parallel_scan_matches_sequential`: `< 1e-4` forward and backward gradient diff vs sequential on CPU and MPS).
- **$O(T \cdot K)$ Sparse Overwrite-Aware Causal Pointer (`K = 32`):**
  - Strictly causal ($j < t$) slots: `n_exact = 8` recency-ordered exact previous occurrences of the current token (`rank 0` = most recent occurrence -> **latest overwrite value**) + `n_local = 24` deduplicated local window positions, with learned slot recency biases and learned log-distance decay.

## 3. Protocol & Parameter Parity
- **Part A (Micro-Lab):** `d_model = 32, V = 72`, `55.0K–56.0K` trainable params (`±2%`), `5` tasks (`S4_Holonomy_8x`, `MQAR_Selective_Overwrite_4x`, `Unseen_OpenVocab_Copy`, `Parity_Dyck_4x`, `Hierarchy_Denom_Rollup_4x`) × `3` seeds × `200` steps.
- **Part B (Real Subword Corpus `data/v1`):** `d_model = 128, n_heads = 4, n_layers = 4`, matched on **non-embedding backbone params (`851,858–853,313`, `±0.15%`)** (`~5.05M` total):
  - `M0_Dense_GQA_RoPE` (`ffn=384`), `M5_HSPA_LieSO3` (`ffn=496`), `M9_DualChannel_Pointer` (`ffn=312`), `M10_HSPA_DensePointer` (`ffn=474`), `M11_FullSphere_SparseOWPointer` (`ffn=312`), `M12_HSPA_SparseOWPointer` (`ffn=474`).
  - Train `1,000` steps × `2` seeds (`B=8, T=128`, `AdamW lr=1.5e-3, wd=0.01`, `100` warmup + cosine decay) on MPS.
  - Evaluate **`1×` (`T=128`), `2×` (`T=256`), and `4×` (`T=512`)** validation BPB across `web`, `textbook`, `math`, `code`; zero-shot `copy_{32,64,128,256}`; and `val_web` real-haystack MQAR (`T128_N4`, `T256_N4`, `T512_N4`, common vs rare).

## 4. Falsification & Success Criteria
1. **Sparse Overwrite Win Criterion:** `M11` and `M12` must lift `MQAR_Selective_Overwrite_4x` `ood_overwritten_acc` markedly above dense-pointer `M9`/`M10` (`0.542`) without sacrificing `Unseen_OpenVocab_Copy` (`>= 0.94`) or Part B `4×` (`L=512`) `code`/`math` BPB.
2. **Spinor + Pointer Complementarity Criterion:** If `M10`/`M12` preserve `M5`'s `Hierarchy_Denom_Rollup_4x` & `textbook` BPB lead WHILE matching or beating `M9`/`M11` on `code@512`, `math@512`, and `copy_256`, the synthesis earns frontier core status; if instead pointer gating/HSPA kinematics interfere negatively with either natural BPB or exact copy, record the exact trade-off honestly.
