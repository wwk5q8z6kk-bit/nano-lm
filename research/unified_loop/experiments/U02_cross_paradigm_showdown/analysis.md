# EXP-U02 Analysis: Cross-Paradigm Neural Mechanism Showdown (Part A Micro-Lab & Part B Real Corpus)

- **Status:** Verified local results (`$0.00` paid cloud compute, local Apple Silicon CPU + MPS)
- **Scripts:**
  - Part A: `research/unified_loop/scripts/run_u02_cross_paradigm_showdown.py` (`426.7s` wall across 5 workers, `n=3` seeds)
  - Part B: `research/unified_loop/scripts/run_u02_part_b_real_corpus.py` (`819.88s` wall across MPS + 4 CPU workers, `n=2` seeds)
- **Raw Artifacts:**
  - `research/unified_loop/experiments/U02_cross_paradigm_showdown/results_part_a.json`
  - `research/unified_loop/experiments/U02_cross_paradigm_showdown/results_part_b.json`
- **Mechanism Library:** `research/unified_loop/mechanisms/zoo.py` (`causal_leak_max = 0.0` verified at both `d=32` and `d=128`; `Woodbury` rank-1 Cayley `< 1e-5` vs `linalg.solve`; `GDN1` `C=32` chunked WY `< 1e-4` forward & backward vs sequential on CPU and MPS).

---

## 1. Part A: Iso-Parameter Cross-Paradigm Micro-Lab (`55.3K–56.4K` Params, `5` Tasks × `3` Seeds)

| Mechanism | Params | `S4` ID (`L=24`) | `S4` `8x` OOD | `MQAR` `4x` OOD (`ov`) | `Unseen Copy` ID / Unseen | `Parity+Dyck` `4x` OOD | `Hier Rollup` `4x` OOD (`close`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `M0_Dense_GQA_RoPE` | 55,841 | 0.0378 | 0.0422±0.0023 | 0.0927 (`0.0925`) | 0.0924 / **0.0000** | 0.9188±0.0311 | 0.1459 (`0.1140`) |
| `M1_GDN1_Cone_Closed` | 56,401 | 0.0573 | 0.0404±0.0032 | 0.0590 (`0.0576`) | 0.0273 / **0.0000** | 0.9174±0.0233 | 0.2463 (`0.1384`) |
| `M2_FullSphere_SFOR` | 55,761 | 0.0820 | 0.0335±0.0063 | 0.0597 (`0.0578`) | 0.0345 / **0.0000** | 0.9203±0.0231 | 0.2396 (`0.1378`) |
| `M3_KDA_DeltaProd2` | 55,777 | 0.0814 | **0.0436±0.0021** | 0.0597 (`0.0576`) | 0.0326 / **0.0000** | 0.9120±0.0182 | 0.2326 (`0.1360`) |
| `M4_BOBO_DualFastOp` | 55,617 | 0.0456 | 0.0407±0.0035 | 0.0597 (`0.0578`) | 0.0260 / **0.0000** | 0.9234±0.0272 | 0.2447 (`0.1357`) |
| `M5_HSPA_LieSO3` | 55,265 | **0.1029** | 0.0426±0.0036 | 0.0597 (`0.0578`) | 0.0267 / 0.0088 | 0.9029±0.0396 | **0.2588 (`0.1633`)** |
| `M6_NEBN_BayesFilter` | 55,633 | 0.0443 | 0.0414±0.0015 | 0.0579 (`0.0569`) | 0.0234 / **0.0000** | 0.9229±0.0201 | 0.2454 (`0.1357`) |
| `M7_SkewCayley_Flow` | 56,129 | 0.0417 | 0.0415±0.0020 | 0.0612 (`0.0598`) | 0.0339 / **0.0000** | **0.9279±0.0207** | 0.2560 (`0.1534`) |
| `M8_TRM_Loop` (raw/guard/cd) | 55,713 | 0.0456 | 0.0426 (g=0.0441) | 0.0753 (g=0.0857) | 0.0690 / **0.0000** | 0.9141 (g=0.9162) | 0.1332 (g=0.1416) |
| **`M9_DualChannel_Pointer`** | 55,538 | 0.0814 | 0.0329±0.0041 | **0.5764 (`0.5421`)** | **0.9967 / 0.9619** | 0.9254±0.0287 | 0.2393 (`0.1370`) |

### Key Part A Findings (Both Positive & Negative)
1. **Unseen Open-Vocabulary Copying & Selective Overwrite MQAR (`H-23` Verified):** Every pure-weight mechanism (`M0..M8`) scored **`0.0000`** (`0.0088` on `M5`) on unseen out-of-distribution tokens (`[44..67]`) and `0.058–0.093` on `4×` MQAR. Only **`M9_DualChannel_Pointer`** solved unseen open-vocab copying (**`0.9967` ID / `0.9619 ± 0.0244` Unseen OOD**) and reached **`0.5764 ± 0.0794`** (`0.5421` on overwritten keys) on `4×` MQAR. *(Still short of `1.00` on overwritten keys because pointer softmax can split weight across duplicate past occurrences when not paired with hard key-erasure.)*
2. **Geometric / Lie Recurrence Beats Attention ~1.8× on `4×` Deeper Hierarchical Accumulation:** On `Hierarchy_Denom_Rollup_4x` (`depth <= 2` -> `depth 4..8`, mod `17`), both RoPE Attention baselines (`M0 = 0.1459`, `M8 = 0.1332`) lagged behind all recurrent models (`0.2326–0.2588`), led by exact rotational **`M5_HSPA_LieSO3` (`0.2588`, `close=0.1633`)** and **`M7_SkewCayley_Flow` (`0.2560`, `close=0.1534`)**. **Negative bound:** even the top learned score (`close=0.1633` vs `1/17=0.0588` chance) remains far below `nano/`'s symbolic rollup engine (`100%` exact).
3. **S4 8× Negative Result & TRM Monotonic Guard:** At `200` micro-steps (`56K`), `M5_HSPA_LieSO3` led ID (`0.1029` vs `0.0378` Attention), but ALL mechanisms fell to chance (`~0.042`) at `8×` length (`L=192`). On `M8_TRM_Recursive_Loop`, Monotonic Margin Guard (`H-31`) improved over raw step-3 on every algorithmic OOD task (`0.0441 > 0.0426`, `0.0857 > 0.0753`, `0.9162 > 0.9141`, `0.1416 > 0.1332`), whereas naive `LoopCD` (`H-35`) hurt on MQAR (`0.0662`) and Hierarchy (`0.1094`).

---

## 2. Part B: Real Subword Corpus Transfer (`/Volumes/Express1M2/nanolm-lab/data/v1/`, `d=128`, `1,000` Steps, `n=2` Seeds)

Matched within **`±0.15%`** on non-embedding backbone parameters (`851,858–854,273`, `~5.05M` total including tied `vocab=32,768` embedding).

| Contender (Device, Non-Emb Params) | `tok/s` | `web` (`128` / `256`) | `textbook` (`128` / `256`) | `math` (`128` / `256`) | `code` (`128` / `256`) | `copy_32 / 64 / 128` | Real-Haystack MQAR (`128 / 256` common) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `M0_Dense_GQA_RoPE` (`mps`, 853,121) | **26,652** | 2.0660 / 2.0611 | 1.9663 / 1.9672 | 2.6573 / 2.7016 | 2.8517 / 2.8245 | 0.000 / 0.000 / 0.000 | 0.0000 / 0.0000 |
| `M1_GDN1_Cone_Closed` (`mps`, 852,241) | 12,258 | 2.1376 / 2.1298 | 2.0457 / 2.0492 | 2.8021 / 2.8407 | 3.1180 / 3.1013 | 0.000 / 0.000 / 0.000 | 0.0000 / 0.0000 |
| **`M5_HSPA_LieSO3`** (`cpu`, 853,313) | 2,253 | 2.0369 / 2.0324 | **1.9336** / **1.9392** | 2.6749 / 2.7186 | 2.8805 / 2.8547 | 0.000 / 0.000 / 0.000 | 0.0000 / 0.0000 |
| `M7_SkewCayley_Flow` (`cpu`, 854,273) | 1,269 | 2.0609 / 2.0536 | 1.9681 / 1.9719 | 2.7066 / 2.7435 | 2.9259 / 2.8991 | 0.000 / 0.000 / 0.000 | 0.0000 / 0.0000 |
| **`M9_DualChannel_Pointer`** (`mps`, 851,858) | 9,256 | **2.0219 / 1.9960** | 1.9844 / 1.9619 | **2.4053 / 2.3172** | **2.4364 / 2.1983** | **0.910 / 0.933 / 0.926** | **0.3125 / 0.1875** |

### Per-Layer Key Geometry (`hd = 32`, `val_web`, Seed `0`)
- `M0_Dense_GQA_RoPE`: Layer `0 -> 3` effective rank `10.728 -> 8.575 -> 4.390 -> 3.919` (`cos = 0.3628 -> 0.5851`).
- `M1_GDN1_Cone_Closed`: `12.815 -> 7.413 -> 5.435 -> 4.449` (`cos = 0.2695 -> 0.6225`).
- `M9_DualChannel_Pointer`: `10.647 -> 8.343 -> 6.155 -> 5.305` (`cos = 0.3192 -> 0.5897`).
- `M7_SkewCayley_Flow`: `13.400 -> 7.958 -> 5.362 -> 4.458` (`cos = 0.2768 -> 0.5260` — lowest deeper-layer key cosine of all key-based models).

### Scientific Synthesis from `EXP-U02`
1. **Structural Content Copying + FullSphere (`M9`) Unlocks Code, Math, and `2×` Extrapolation:** `M9_DualChannel_Pointer` beat RoPE Attention (`M0`) by **`-0.626 BPB` on `code@256`** (`2.1983` vs `2.8245`), **`-0.384 BPB` on `math@256`** (`2.3172` vs `2.7016`), and **`-0.065 BPB` on `web@256`** (`1.9960` vs `2.0611`), while uniquely solving zero-shot `copy_128 = 0.9257` and real-haystack MQAR (`0.3125` common at `T128_N4`). *(Caveat: `M9` currently uses an $O(T^2)$ prefix pointer map over `[B, T, T]`.)*
2. **Legacy `S1` DeltaNet Physics (`M1_GDN1`) Is Uncompetitively Weak:** `M1_GDN1_Cone_Closed` finished **last on every single real-corpus domain** (`2.1376` `web`, `3.1180` `code`, `+0.116` to `+0.903 BPB` behind `M9`, and `+0.10` behind `M5` on `web`/`textbook`). Removing the `SiLU` orthant constraint + closed gate + scalar-cone restriction (moving to `M9` or `M5`) yields an immediate real-corpus gain.
3. **Pure $O(T)$ Spinor Rotation (`M5_HSPA_LieSO3`) Beats Attention on Natural Text & Hierarchies:** `M5` achieved the best `textbook@128` (`1.9336`) and `#2` overall `web` (`2.0369 -> 2.0324`) with strictly $O(T)$ memory, while beating Attention on `Hierarchy_Denom_Rollup_4x`, proving orthogonal geometric state carries high value for non-copying semantic abstraction — and naturally invites **combining an `M5_HSPA_LieSO3` semantic backbone with a sparse/selective `M9`-style exact pointer**.
4. **Open Problem Clearly Diagnosed:** Even `FullSphere + SFOR` (`M9`) and `SkewCayley` (`M7`) still suffer depth-wise key rank contraction (`10.6–13.4 -> 4.5–5.3 / 32`), explaining why `M9` drops to `0.0416` on `rare_tail_ids` MQAR (`20000..32700`) and relies heavily on its explicit pointer for copying.
