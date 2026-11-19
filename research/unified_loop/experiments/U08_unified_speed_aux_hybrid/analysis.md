# EXP-U08 Analysis — Unified Speed `HSPA` Body Scan, Auxiliary Base-LM Loss, Multi-Timescale `M17`, Hybrid Semantic–Lexical `M18`, & Key-Rank Health Audit

- **Experiment ID:** `EXP-U08`
- **Status:** `COMPLETE` (`$0.00` paid compute, `has_nonfinite = False` everywhere)
- **Implementation & Preregistration Commit:** `3e6bfd7`
- **Script:** `research/unified_loop/scripts/run_u08_unified_speed_aux_hybrid.py`
- **Raw Artifacts (`allow_nan=False` strict RFC-8259 JSON):**
  - Part A (`cpu`, `5` contenders × `6` tasks × `3` seeds, `200` steps, `337.16s`): `research/unified_loop/experiments/U08_unified_speed_aux_hybrid/results_part_a.json`
  - Part A2 (`cpu`, `5` contenders × `3` tasks × `2` seeds, `600` steps, `per_seed` saved, `262.76s`): `research/unified_loop/experiments/U08_unified_speed_aux_hybrid/results_part_a2_algebra.json`
  - Part B (`mps`, `data/v1`, `d=128`, `851,691–852,716` non-emb / `~5.05M` total, `1,000` steps × `2` seeds + `L0→L3` Rank Audit, `1,201.01s`): `research/unified_loop/experiments/U08_unified_speed_aux_hybrid/results_part_b.json`

---

## 1. Executive Summary Across All Five Preregistered Criteria (`verified local result`)

1. **Criterion 1 (Speed Unification): `PASSED` (`1.58×` faster MPS training, zero accuracy loss on LM BPB)**
   - Replacing `HSPA`'s $3\times 3$ matrix Kogge–Stone doubling with the gauge-decoupled $SU(2)$ body scan (`48` dims) and replacing the CPU candidate double loop (`<= 1024`) with vectorized on-device `torch.topk` accelerated `M13_Ctrl` from **`7,439.9 tok/s` (`EXP-U07`) to `11,754.0 tok/s` (`1.58×`)** while matching/improving natural BPB (`web@512: 1.8764 vs 1.8766`, `textbook@512: 1.8190 vs 1.8222`, `code@512: 1.6535 vs 1.6596`). Mean `copy_256` on `M13_Ctrl` (`0.7169 ± 0.1723`) was noisy across its 2 seeds (`0.595`, `0.839`), whereas **`M13_AuxBase` (`0.9543 ± 0.0255`)** and **`M18_AuxBase` (`0.9543 ± 0.0095`, `11,063.6 tok/s`)** exceeded `0.95`.

2. **Criterion 2 (`AuxBase` Loss `λ = 0.25`): `MIXED — HELPS PART B & 600-STEP ALGEBRA, REGRESSES 200-STEP UNSEEN COPY ON M13/M18`**
   - **Wins:** On `data/v1` (`Part B`), `M13_AuxBase` beat `M13_Ctrl` on **every domain and horizon (`12/12` cells)**: `web@512: 1.8697 vs 1.8764`, `textbook@512: 1.8107 vs 1.8190`, `math@512: 2.1403 vs 2.1605`, **`code@512: 1.6285 vs 1.6535`**, lifted `copy_256` (`0.9543 vs 0.7169`) and `T512_N4` rare MQAR (`0.8959 vs 0.8333`), lifted `MQAR_Overwrite_4x` (`0.9985`), and lifted `600`-step `S4_Holonomy_8x` OOD (`0.5273 vs 0.4126`; `M18_AuxBase` hit `0.6081 ± 0.0691`).
   - **Negative regression:** At `200` steps (`Part A`), constant `aux_base_weight = 0.25` degraded `Unseen_OpenVocab_Copy` OOD (`M13: 0.9824 -> 0.6211 ± 0.3242`, `M18: 0.6348 ± 0.3152`, vs `M17: 0.9876`, `M16: 0.9788`) and `Parity_Dyck_4x` on `M13` (`0.8991 -> 0.8588`). Constant auxiliary cross-entropy over closed ID tokens `[4..43]` can over-sharpen closed-vocab base predictions against unseen OOD pointer copying unless annealed or gated.

3. **Criterion 3 (Multi-Timescale Damped `M17`): `FALSIFIED` on Natural BPB & Hierarchy (`negative finding`)**
   - Even with `HSPA`'s compact `48`-dim readout (`ffn=461 vs 463`, eliminating `M14`/`M15`'s `144`-dim `hol_6d` bloat), `M17_AuxBase` (`CausalDepthwiseConv` + `a_bias [0.0..4.5]`) **still regressed on all 4 `data/v1` domains** (`web@512 = 1.9810 vs M13_AuxBase 1.8697`, `code@512 = 1.7308 vs 1.6285`), regressed on `600`-step `S4` (`0.2768 vs 0.5273`) & `Hierarchy` (`0.2759 vs 0.3236`), and deepened `L0→L3` collapse (`copy L3 u_in = 1.19–1.95 / 12`), while lifting `copy_256` (`0.9986`) and `T512_N4` rare MQAR (`0.9375`). Thus forcing fixed slow decay (`sigmoid(4.5) = 0.989`) + pre-mixer conv blurs local transition prediction cleanly won by `M13`'s unconstrained adaptive decay + `M18`'s hybrid pointer.

4. **Criterion 4 (Hybrid Semantic–Lexical `M18`): `PARTIAL WIN ON SYSTEM FRONTIER; ALIAS TRANSFER AT 4× REMAINS UNSOLVED`**
   - On the new cross-alias `MQAR_Alias_Semantic_4x` test (`[4..11] <-> [12..19]`), `M18_AuxBase` led Part A (`ID = 0.5868`, `OOD = 0.0776 ± 0.0611` vs `0.020–0.026` elsewhere), yet by `600` steps (`Part A2`) ALL models scored near chance (`0.030–0.068` OOD). Scalar token-salience landmarks (`ptr_salience(h_n)`) alone do not reliably route opposite-alias queries across `4×` context.
   - However, on **real data & algebra**, **`M18_AuxBase`** ran at **`11,063.6 tok/s` (`1.34×` faster than `M13_AuxBase`)**, topped `Part B` `textbook`/`code` behind `M13_AuxBase` (`web@512 = 1.8736`, `textbook@512 = 1.8144`, `math@512 = 2.1556`, `code@512 = 1.6468`), matched `0.9543 ± 0.0095` `copy_256`, hit `0.9791` `T128`/`T256` rare MQAR, and led `Part A2` `S4_Holonomy_8x` (`0.6081 ± 0.0691`, both seeds converged: `s0=0.6569, s1=0.5592`) and `Hierarchy_Denom_Rollup_4x` (`0.3376 ± 0.0716`, `s0=0.3882`)!

5. **Criterion 5 (`L0→L3` Key-Rank Health Gate): `WARNING — CRITICAL RANK CONTRACTION DIAGNOSED BEFORE SCALE-UP` (`negative finding`)**
   - All models completed with `has_nonfinite = False`.
   - **Critical finding:** Even `HSPA` (`M13`, `M17`, `M18`) AND `FullSphere` (`M16`) lose **`70–85%` of effective state rank between `L0` and `L3`** on `val_code_256` (`max=32`): `L0 = 11.1–13.1 → L3 = 1.75–3.73 / 32` (`state_offdiag_cos = 0.66–0.87`), with `M18_AuxBase` retaining slightly higher `L3` rank (`2.74–3.73 / 32`) than `M13` (`1.75–3.49`) and `M16` (`1.81–2.31`, whose `gdn1_k` also dropped `10.17–11.69 → 3.58–6.03 / 32`). Fixing deep-layer rank contraction is mandatory before `>30M` scaling.

---

## 2. Part A — Micro-Lab (`d=32, V=72, 55.23K–55.53K params, 3 seeds, 200 steps`)

| Contender | Params | `aux` | `Unseen_OpenVocab_Copy` OOD (`ID`) | `MQAR_Overwrite_4x` OOD (`ov`) | `MQAR_Alias_Sem_4x` OOD (`ID`) | `Hierarchy_4x` OOD (`Close`) | `Parity_Dyck_4x` OOD | `S4_8x` OOD (`ID`) |
|---|---:|---:|---|---|---|---|---|---|
| `M13_Ctrl` | 55,499 | 0.0 | `0.9824 ± 0.0096` (`1.000`) | `0.9926` (`0.9917`) | `0.0260` (`0.5243`) | **`0.2627`** (`0.1614`) | `0.8991` | `0.0586` (`0.1263`) |
| `M13_AuxBase` | 55,499 | 0.25 | `0.6211 ± 0.3242` (`1.000`) | **`0.9985` (`0.9983`)** | `0.0201` (`0.5174`) | `0.2575` (`0.1597`) | `0.8588` | `0.0489` (`0.1758`) |
| `M17_AuxBase` | 55,371 | 0.25 | **`0.9876 ± 0.0039`** (`1.000`) | `0.9664` (`0.9620`) | `0.0237` (`0.5182`) | `0.2560` (`0.1501`) | `0.9203` | **`0.1428`** (**`0.2402`**) |
| `M18_AuxBase` | 55,532 | 0.25 | `0.6348 ± 0.3152` (`1.000`) | `0.9928` (`0.9920`) | **`0.0776`** (**`0.5868`**) | `0.2577` (`0.1583`) | `0.9203` | `0.0430` (`0.0898`) |
| `M16_AuxBase` | 55,227 | 0.25 | `0.9788 ± 0.0248` (`1.000`) | `0.9978` (`0.9976`) | `0.0227` (`0.5122`) | `0.2391` (`0.1365`) | **`0.9232`** | `0.0634` (`0.0879`) |

---

## 3. Part A2 — `600`-Step Algebraic & Cross-Alias Convergence Probe (`2 seeds`, `per_seed` recorded)

| Contender | `S4_Holonomy_8x` OOD (`ID`) & `[s0, s1 OOD]` | `Hierarchy_Denom_Rollup_4x` OOD (`Close`) & `[s0, s1 OOD]` | `MQAR_Alias_Semantic_4x` OOD (`ID`) |
|---|---|---|---|
| `M5_Pure` | `0.3664 ± 0.4146` (`0.9160`) `[0.0732, 0.6595]` | `0.2788 ± 0.0082` (`0.1889`) `[0.2846, 0.2730]` | `0.0617` (`0.0638`) |
| `M13_Ctrl` | `0.4126 ± 0.1533` (`1.0000`) `[0.5210, 0.3042]` | `0.3085 ± 0.0429` (`0.1897`) `[0.3389, 0.2782]` | **`0.0676`** (`0.5234`) |
| `M13_AuxBase` | `0.5273 ± 0.4172` (`0.9961`) `[0.8223, 0.2323]` | `0.3236 ± 0.0638` (`0.1918`) `[0.3688, 0.2785]` | `0.0543` (`0.5182`) |
| `M17_AuxBase` | `0.2768 ± 0.3366` (`0.5439`) `[0.5148, 0.0387]` | `0.2759 ± 0.0016` (`0.1947`) `[0.2770, 0.2747]` | `0.0297` (`0.5365`) |
| **`M18_AuxBase`** | **`0.6081 ± 0.0691` (`1.0000`) `[0.6569, 0.5592]`** | **`0.3376 ± 0.0716` (`0.1906`) `[0.3882, 0.2869]`** | `0.0672` (`0.1536`) |

---

## 4. Part B — Real Subword Corpus `data/v1` (`d=128, 1,000 steps × 2 seeds, MPS`) + `L0→L3` Rank Audit

### 4.1 Multi-Horizon Validation BPB (`1× = 128`, `2× = 256`, `4× = 512`, lower is better)

| Contender | Non-Emb | `tok/s` | `web` (`128 / 256 / 512`) | `textbook` (`128 / 256 / 512`) | `math` (`128 / 256 / 512`) | `code` (`128 / 256 / 512`) |
|---|---:|---:|---|---|---|---|
| `M13_Ctrl` | 852,587 | **11,754.0** | `1.9507 / 1.9309 / 1.8764` | `1.8879 / 1.8720 / 1.8190` | `2.3266 / 2.2603 / 2.1605` | `2.3030 / 2.1127 / 1.6535` |
| **`M13_AuxBase`** | 852,587 | 8,256.8 | **`1.9423 / 1.9228 / 1.8697`** | **`1.8791 / 1.8627 / 1.8107`** | **`2.3081 / 2.2454 / 2.1403`** | **`2.2382 / 2.0600 / 1.6285`** |
| `M17_AuxBase` | 851,691 | 7,636.4 | `2.0629 / 2.0381 / 1.9810` | `2.0102 / 1.9900 / 1.9518` | `2.4529 / 2.3766 / 2.2594` | `2.4525 / 2.2397 / 1.7308` |
| **`M18_AuxBase`** | 852,716 | 11,063.6 | `1.9492 / 1.9296 / 1.8736` | `1.8858 / 1.8664 / 1.8144` | `2.3204 / 2.2628 / 2.1556` | `2.2636 / 2.0869 / 1.6468` |
| `M16_AuxBase` | 852,027 | 7,845.4 | `2.0551 / 2.0284 / 1.9729` | `1.9937 / 1.9751 / 1.9352` | `2.4176 / 2.3426 / 2.2302` | `2.3984 / 2.1897 / 1.7024` |

### 4.2 Zero-Shot Copy, Real-Haystack MQAR (`val_web`), & `L0→L3` Effective Rank Audit (`max_state_rank = 32`)

| Contender | `copy_32 / 64 / 128 / 256` | MQAR `common` (`128 / 256 / 512`) | MQAR `rare_tail` (`128 / 256 / 512`) | `val_code_256` State Rank `L0→L1→L2→L3` (`s0 / s1`) |
|---|---|---|---|---|
| `M13_Ctrl` | `0.7641 / 0.6795 / 0.6981 / 0.7169` | `0.8334 / 0.9167 / 0.8125` | `0.9166 / 0.8958 / 0.8333` | `11.71→5.00→1.95→3.49` / `11.11→6.45→4.08→1.75` |
| `M13_AuxBase` | `0.9748 / 0.9489 / 0.9483 / 0.9543` | `0.7916 / 0.8541 / 0.7292` | **`0.9583 / 0.9375 / 0.8959`** | `11.22→2.59→6.14→2.57` / `11.40→4.86→2.27→1.76` |
| `M17_AuxBase` | **`1.0000 / 1.0000 / 0.9990 / 0.9986`** | **`0.9167 / 0.9791 / 0.8959`** | `0.8750` / **`0.9791 / 0.9375`** | `12.56→2.16→3.23→2.53` / `13.06→5.25→6.07→2.09` |
| **`M18_AuxBase`** | `0.9708 / 0.9544 / 0.9439 / 0.9543` | `0.8333 / 0.9166 / 0.8125` | **`0.9791 / 0.9791`** / `0.7709` | `11.77→5.28→1.76→3.73` / `11.80→8.45→2.47→2.74` |
| `M16_AuxBase` | `0.9990 / 0.9980 / 0.9970 / 0.9967` | **`0.9375`** / `0.8750 / 0.8125` | `0.8959 / 0.8750 / 0.8958` | `9.21→10.94→9.84→2.31` / `11.70→11.66→5.96→1.81` |

*(Input/key rank details on `val_code_256`: `HSPA u_in` [`max=12`] dropped `7.57–7.95 → 2.23–3.29` in `M13`/`M18`, `6.66–6.73 → 2.10–2.87` in `M17`; `M16 gdn1_k` [`max=32`] dropped `10.17–11.69 → 3.58–6.03`. Full per-layer cosine data is stored in `results_part_b.json`.)*

---

## 5. Next Required First-Principles Loop Before RunPod Scale (`EXP-U09` Prescription)

1. **Solve `L0 → L3` Effective-Rank Collapse (`eff_rank: 11.5 → 1.8–3.7 / 32`):** Test `AuxBase` warmup/annealing (`0.25 -> 0.05` so `Unseen_OpenVocab_Copy` stays `0.98+`), spectral/cosine off-diagonal decorrelation regularizer across deep blocks (`L1–L3`), and orthogonal / `Block-AttnRes` skip mixing (`M3`'s depth routing inside `M13`/`M18`) — verify `L3 state_eff_rank >= 8 / 32` while preserving `M13_AuxBase`'s `1.870 / 1.629` BPB and `0.954` copy before spending RunPod dollars at `30M–125M`.
2. **Solve Cross-Alias Retrieval (`MQAR_Alias_Semantic_4x`):** Replace scalar token salience with a lightweight key-hash / cached-state cosine semantic top-`8` lookup so opposite aliases match without relying on identical token IDs.
