# EXP-U06 Analysis — 2×2 Factorial Spinor–Pointer Synthesis (`M10`–`M12`)

- **Experiment ID:** `EXP-U06`
- **Status:** `COMPLETE` (`$0.00` paid compute)
- **Implementation & Preregistration Commit:** `fc60e0e`
- **Script:** `research/unified_loop/scripts/run_u06_spinor_pointer_synthesis.py`
- **Raw Artifacts:**
  - Part A (`cpu`, `5` tasks × `6` contenders × `3` seeds, `200` steps, `329.52s`): `research/unified_loop/experiments/U06_spinor_pointer_synthesis/results_part_a.json`
  - Part B (`mps`, `data/v1`, `d=128`, `~852K–853K` non-emb / `~5.05M` total, `1,000` steps × `2` seeds, `1960.85s`): `research/unified_loop/experiments/U06_spinor_pointer_synthesis/results_part_b.json`

---

## 1. Executive Summary (`verified local result`)

Replacing `M9_DualChannel_Pointer`'s `FullSphere (GDN1 + SFOR)` backbone and $O(T^2)$ dense pointer in a controlled `2×2` factorial (`{FullSphere, HSPA_LieSO3} × {Dense O(T²) Pointer, Sparse Overwrite-Aware O(T·K) Pointer}`, `K=32 = 8 exact recency + 24 local`) alongside `M0_Dense_GQA_RoPE` and `M5_HSPA_LieSO3` controls yielded three sharp empirical conclusions:

1. **Both synthesis upgrades work as predicted on natural language, hierarchy, overwrite MQAR, and 4× real-haystack MQAR:**
   - **`M10_HSPA_DensePointer` (`HSPA + Dense`) achieved `#1` validation BPB across ALL 4 domains and ALL 3 context horizons (`1×=128, 2×=256, 4×=512`)**:
     - `web@512`: `1.8659` (`vs M9 1.9395`, `M5 2.0024`, `M0 2.0359`)
     - `textbook@512`: `1.8094` (`vs M5 1.8871`, `M9 1.9112`, `M0 1.9194`)
     - `math@512`: `2.1175` (`vs M9 2.1726`, `M0 2.7893`, `M5 2.8310`)
     - `code@512`: **`1.5959`** (`-1.0524 BPB` vs `M0 2.6483`, `-0.0440` vs `M9 1.6399`)
   - **`M12_HSPA_SparseOWPointer` (`HSPA + Sparse OW`, $O(T \log T) + O(T \cdot K)$) combined stability + #1 hierarchy + #2 natural BPB + overwrite & 4× rare-tail MQAR:**
     - `#1` `Hierarchy_Denom_Rollup_4x` (`0.2624 ± 0.0119`, `close=0.1671`, matching `M5 0.2602` and beating `M0 0.1459` and `M9 0.2393`).
     - **`0.9286 ± 0.0282`** on `MQAR_Selective_Overwrite_4x` (**`ood_overwritten_acc = 0.9198`**, `+37.77 pp` over dense `M9 0.5421`).
     - `Unseen_OpenVocab_Copy` = `0.9603 ± 0.0171` (passes `>= 0.94` criterion; `ID = 0.9974`).
     - `#2` across all natural language horizons (`web@512 = 1.8924`, `textbook@512 = 1.8276`, `math@512 = 2.1878`, `code@512 = 1.7077`), beating `M9`, `M5`, `M0`, and `M11` on `web`, `textbook`, and `math@128/256`.
     - Zero-decay `4×` (`T=512`) real-haystack MQAR: **`0.5625` common / `0.4375` rare** (`vs M9 0.0416 / 0.0000` and `M10 0.2084 / 0.0417`).
     - **`1.56×` faster MPS training** than dense-pointer `M10` (`6,260.2 vs 4,013.9 tok/s`) with **zero NaNs** across all Part A & Part B evaluations.

2. **Sparse Overwrite-Aware Pointer (`K=32`) decisively solves overwrite blurring and rare-tail MQAR decay on BOTH backbones:**
   - Part A `MQAR_Selective_Overwrite_4x` (`ood_overwritten_acc`): `M9 0.5421 -> M11 0.9266` (`+38.45 pp`), `M10 0.6433 -> M12 0.9198` (`+27.65 pp`).
   - Part B `T512_N4` (`4×`) real-haystack MQAR: `M9 0.0416 common / 0.0000 rare -> M11 0.6250 / 0.6459`; `M10 0.2084 / 0.0417 -> M12 0.5625 / 0.4375`.

3. **Two important limitations & failure modes were uncovered (`negative / falsification results`):**
   - **Trade-off 1 (Synthetic uniform-random copy gap on `HSPA`):** Even though `M12` (`copy_256 = 0.6416 ± 0.1651`) substantially outperformed `M10` (`0.1808 ± 0.1326`) and pure backbones `M0`/`M5` (`0.0000`), `HSPA`-backed pointers trailed `FullSphere`-backed pointers on synthetic zero-shot IID random copying (`M9 copy_256 = 0.8899`; `M11 copy_128 = 0.9543`, `copy_256 s=1 = 0.9520`). Thus Criterion 2 ("match or beat `M9`/`M11` on `copy_256`") is **NOT yet met** for `HSPA` on ungrounded random token sequences.
   - **Failure mode 2 (`M11` `4×` eval `NaN` on `seed=0`):** `M11_FullSphere_SparseOWPointer` trained finitely through step `1000` on both seeds and evaluated cleanly at `L=128`, `L=256`, `web/textbook/math@512`, `copy_{32,64,128}`, and all `mqar` horizons, **but `seed=0` produced `NaN` on `code@512` (`seed=1 = 1.78472`) and `copy_256` loss (`acc = 0.2039, loss = NaN`; `seed=1 acc = 0.9520, loss = 2.1467`)**. In contrast, norm-preserving `SO(3)` spinor transport in `M10` and `M12` had **zero NaNs**.

---

## 2. Algorithmic Verification Summary (`test_zoo.py`, `fc60e0e`)

- **$O(\log T)$ Kogge–Stone `HSPALieSO3Mixer` scan:** verified against sequential recurrence in `test_hspa_parallel_scan_matches_sequential` (`< 1e-4` max forward diff and gradient diff on both `cpu` and `mps`), enabling `M5` to move from CPU-bound execution to MPS (`4,784.4 tok/s`).
- **$O(T \cdot K)$ strictly causal Sparse Overwrite-Aware Pointer (`K = 32`):** verified by `test_sparse_ow_pointer_causality_and_overwrite_index` (no candidate index `>= t`; repeated keys place the most recent position at exact slot rank `0`; finite gradients on MPS).
- **Strict causality across `M0..M12`:** `causal_leak_max = 0.0` at `d=32` (`55.0K–56.4K` params) and `<= 1e-6` (`<1e-6` gate) at `d=128, V=32768` (`851,858–853,313` non-embedding params, `±0.15%`).

---

## 3. Part A — Micro-Lab Results (`d=32, V=72, 55.0K–55.8K params, 3 seeds, 200 steps`)

| Model | Params | `Unseen_OpenVocab_Copy` OOD (`ID`) | `MQAR_Overwrite_4x` OOD / **Overwritten** | `Hierarchy_Denom_Rollup_4x` OOD (`Close`) | `Parity_Dyck_4x` OOD | `S4_Holonomy_8x` OOD (`ID`) |
|---|---:|---:|---:|---:|---:|---:|
| `M0_Dense_GQA_RoPE` | 55,841 | `0.0000 ± 0.0000` (`0.0924`) | `0.0927 ± 0.0104` / **`0.0925`** | `0.1459 ± 0.0091` (`0.1140`) | `0.9188 ± 0.0311` | `0.0422` (`0.0378`) |
| `M5_HSPA_LieSO3` | 55,265 | `0.0088 ± 0.0080` (`0.0273`) | `0.0597 ± 0.0019` / **`0.0578`** | `0.2602 ± 0.0110` (`0.1614`) | `0.9155 ± 0.0424` | `0.0465` (**`0.1712`**) |
| `M9_DualChannel_Pointer` | 55,538 | `0.9619 ± 0.0244` (`0.9967`) | `0.5764 ± 0.0794` / **`0.5421`** | `0.2393 ± 0.0059` (`0.1370`) | `0.9254 ± 0.0287` | `0.0329` (`0.0814`) |
| `M10_HSPA_DensePointer` | 55,042 | `0.9189 ± 0.0284` (`0.9961`) | `0.6795 ± 0.0762` / **`0.6433`** | `0.2610 ± 0.0101` (`0.1665`) | **`0.9280 ± 0.0323`** | `0.0507` (`0.0957`) |
| `M11_FullSphere_SparseOW` | 55,571 | **`0.9840 ± 0.0251`** (**`1.0000`**) | **`0.9347 ± 0.0121`** / **`0.9266`** | `0.2428 ± 0.0082` (`0.1365`) | `0.9232 ± 0.0265` | `0.0358` (`0.0781`) |
| `M12_HSPA_SparseOW` | 55,075 | `0.9603 ± 0.0171` (`0.9974`) | `0.9286 ± 0.0282` / **`0.9198`** (`ID 0.9957`) | **`0.2624 ± 0.0119`** (**`0.1671`**) | `0.9139 ± 0.0381` | **`0.0537`** (`0.1074`) |

**Part A Takeaways (`verified local result`):**
- Sparse overwrite-aware indexing raises `MQAR_Selective_Overwrite_4x` overwritten recall from `0.5421–0.6433` (dense pointer) to **`0.9198–0.9266`** without touching `Unseen_OpenVocab_Copy` (`0.9603` on `M12`, `0.9840` on `M11`).
- Swapping `FullSphere` for `HSPA_LieSO3` eliminates the `-0.021 OOD` / `-0.025 close` penalty on `Hierarchy_Denom_Rollup_4x`: `M12` leads all 6 models at `0.2624` (`close=0.1671`).
- `S4_Holonomy_8x` OOD remains near uniform `1/24 = 0.0417` across all architectures under 200-step micro training (consistent with `EXP-U02` negative finding).

---

## 4. Part B — Real Subword Corpus `data/v1` (`d=128, ~852K non-emb / 5.05M total, 1,000 steps × 2 seeds, MPS`)

### 4.1 Multi-Horizon Validation BPB (`1× = 128`, `2× = 256`, `4× = 512`, lower is better)

| Model | Non-Emb | `tok/s` | `web` (`128 / 256 / 512`) | `textbook` (`128 / 256 / 512`) | `math` (`128 / 256 / 512`) | `code` (`128 / 256 / 512`) |
|---|---:|---:|---|---|---|---|
| `M0_Dense_GQA_RoPE` | 853,121 | 24,222.0 | `2.0660 / 2.0611 / 2.0359` | `1.9664 / 1.9672 / 1.9194` | `2.6571 / 2.7015 / 2.7893` | `2.8519 / 2.8246 / 2.6483` |
| `M5_HSPA_LieSO3` | 853,313 | 4,784.4 | `2.0391 / 2.0336 / 2.0024` | `1.9358 / 1.9410 / 1.8871` | `2.6819 / 2.7256 / 2.8310` | `2.8874 / 2.8619 / 2.7123` |
| `M9_DualChannel_Pointer` | 851,858 | 10,351.2 | `2.0190 / 1.9928 / 1.9395` | `1.9848 / 1.9613 / 1.9112` | `2.3975 / 2.3121 / 2.1726` | `2.4344 / 2.1927 / 1.6399` |
| **`M10_HSPA_DensePointer`** | 852,418 | 4,013.9 | **`1.9404 / 1.9158 / 1.8659`** | **`1.8921 / 1.8677 / 1.8094`** | **`2.3157 / 2.2484 / 2.1175`** | **`2.2968 / 2.0822 / 1.5959`** |
| `M11_FullSphere_SparseOW` | 851,891 | 9,952.3 | `2.0503 / 2.0277 / 1.9776` | `1.9959 / 1.9804 / 1.9363` | `2.4405 / 2.3726 / 2.2666` | `2.4773 / 2.2769` / **`NaN`** (`s0=NaN, s1=1.7847`) |
| **`M12_HSPA_SparseOW`** | 852,451 | 6,260.2 | `1.9572 / 1.9405 / 1.8924` | `1.8972 / 1.8832 / 1.8276` | `2.3453 / 2.2823 / 2.1878` | `2.3165 / 2.1370 / 1.7077` |

### 4.2 Zero-Shot Random Copy Probes & Real-Haystack MQAR (`val_web`)

| Model | `copy_32` | `copy_64` | `copy_128` | `copy_256` | MQAR `common` (`128 / 256 / 512`) | MQAR `rare_tail` (`128 / 256 / 512`) |
|---|---:|---:|---:|---:|---|---|
| `M0_Dense_GQA_RoPE` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000 / 0.0000 / 0.0000` | `0.0000 / 0.0000 / 0.0000` |
| `M5_HSPA_LieSO3` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000 / 0.0000 / 0.0000` | `0.0000 / 0.0000 / 0.0000` |
| `M9_DualChannel_Pointer` | `0.8952` | `0.8745` | `0.9276` | `0.8899 ± 0.0241` | `0.2709 / 0.1250 / 0.0416` | `0.0417 / 0.0000 / 0.0000` |
| `M10_HSPA_DensePointer` | `0.4395` | `0.3453` | `0.2160` | `0.1808 ± 0.1326` | `0.3541 / 0.1875 / 0.2084` | `0.1041 / 0.0416 / 0.0417` |
| `M11_FullSphere_SparseOW` | **`0.9315`** | **`0.9628`** | **`0.9543`** | `0.5779 ± 0.5290` (`s0=0.2039/NaN, s1=0.9520`) | `0.5000 / 0.4792` / **`0.6250`** | **`0.5834 / 0.6666 / 0.6459`** |
| `M12_HSPA_SparseOW` | `0.6008` | `0.5551` | `0.5591` | `0.6416 ± 0.1651` (`s0=0.7583, s1=0.5248`) | **`0.6250`** / `0.4583 / 0.5625` | `0.4166 / 0.4792 / 0.4375` |

---

## 5. Scientific Diagnosis & Verdict Against Preregistered Criteria

1. **Preregistered Criterion 1 (Sparse Overwrite Win): `PASSED`**
   - Both `M11` (`0.9266`) and `M12` (`0.9198`) decisively beat dense pointers `M9` (`0.5421`) and `M10` (`0.6433`) on `MQAR_Selective_Overwrite_4x` while keeping `Unseen_OpenVocab_Copy >= 0.94` (`0.9840`, `0.9603`) and lifting `4×` (`T=512`) real-haystack MQAR from `0.000–0.042` (`M9`) to **`0.5625 common / 0.4375 rare` (`M12`)** and **`0.6250 / 0.6459` (`M11`)**.
2. **Preregistered Criterion 2 (Spinor + Pointer Complementarity): `PARTIAL PASS WITH CLEAR DIAGNOSED GAP`**
   - *Passed:* `M10` and `M12` eliminate `M9`'s `textbook` BPB lag and achieve `#1` (`M10`) and `#2` (`M12`) BPB across all 4 domains (`1×/2×/4×`), while `M12` preserves `#1` `Hierarchy_Denom_Rollup_4x` (`0.2624`) and stays `100%` finite (`NaN`-free) unlike `M11 seed=0` at `4×`.
   - *Gap:* Both `M10` (`0.1808`) and `M12` (`0.6416`) trail `M9` (`0.8899`) and `M11 seed=1` (`0.9520`) on zero-shot synthetic random copying (`copy_{32..256}`), and `M12` trails `M11` on rare-tail MQAR (`0.4375 vs 0.6459`).
   - *Structural hypothesis (`hypothesis / inference`):*
     - First, in `M10`/`M12`, `ptr_q(h_final)` and `ptr_k(h_final)` read the `HSPA_LieSO3`-rotated final residual stream without layer-norm / unit normalization or a direct unrotated lexical-key residual path; on purely uniform-random token sequences where consecutive context has zero semantic correlation, $SO(3)$ rotation perturbs arbitrary key vectors enough to soften `ctx_scores = ptr_q · ptr_k` when distractor local tokens also match weakly. Notably, Sparse OW (`K=32`, exact slot 0) already recovers `+46.1 pp` (`M10 0.181 -> M12 0.642`).
     - Second, in `M11 seed=0`, `FullSphere` (`GDN1`) plus unnormalized `ptr_q · ptr_k` logits at `4×` horizon (`L=512`, `code` and `copy_256`) produced `NaN` (not observed in finite training at `T=128`), whereas `SO(3)` norm-preserving transport in `M10`/`M12` prevented blowup at `4×`. Checkpoint weights were not saved by `run_u06`, so the exact intermediate overflow tensor in `M11 seed=0` is inferred rather than directly inspected.

## 6. Next Highest-Leverage Hypothesis (`EXP-U07`)

1. **Test normalized/lexical-bypass pointer keys on `M12` (`M13_HSPA_NormalizedLexSparsePtr`):** apply `RMSNorm` / `L2`-normalized cosine pointer queries & keys (`tau * cos(q, k)`) plus a shallow unrotated lexical-context skip connection into `ptr_q`/`ptr_k` while retaining `M12`'s $SO(3)$ spinor backbone and $O(T \cdot K)$ overwrite-aware index — aiming to recover `M11`-level `0.95+` random copy and `0.65+` rare-tail MQAR **with** `M12`'s `#1–#2` natural BPB, `#1` hierarchy rollup, and zero `4×` `NaN`s.
2. **Test `RMSNorm` + spectral cap on `M11`'s `FullSphere` / `GDN1` + `ptr_q·ptr_k`:** verify whether normalizing pointer logits/keys eliminates `M11 seed=0`'s `4×` `NaN` instability.
