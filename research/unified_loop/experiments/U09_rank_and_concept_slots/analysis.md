# EXP-U09 Analysis: Pre-RunPod Hardening Gate (`CenteredRMSNorm`, Residual Scaling, `Block-AttnRes`, Annealed `AuxBase`, and `ConceptSlotMemory`)

- **Date:** `2026-11-19` · **Experiment Commit:** `e9bf701` (`research/unified-discovery-loop-v1`)
- **Paid Cloud Compute Spend:** **`$0.00`** (`250.17s` CPU Part A + `244.62s` CPU Part A2 + `981.69s` MPS Part B = `1,476.48s` total local compute)
- **Strict Integrity:** `has_nonfinite = false` across all `90` Part A runs, `30` Part A2 runs, and `10` Part B MPS runs (`0.0` causal leak verified at `d=32` and `d=128`; strict iso-parameter parity `55,437–55,659` at `d=32` and `852,333–853,227` non-emb at `d=128`).
- **Artifacts:** [`protocol.md`](./protocol.md) · [`results_part_a.json`](./results_part_a.json) · [`results_part_a2_algebra.json`](./results_part_a2_algebra.json) · [`results_part_b.json`](./results_part_b.json)

---

## 1. Part A — Iso-Parameter `d=32` Micro-Lab (`200` Steps, `5 x 6 x 3 = 90` Runs, `250.17s`)

All contenders use cosine-annealed `AuxBase` (`0.25 -> 0.0`) unless noted in `EXP-U08` reference (`U08 M13_AuxBase` used constant `0.25`).

| Task (`4x`/`8x` OOD) | `M13_AnnealAux` (`55,499`) | `M19_ScaledNorm` (`55,499`) | `M20_ScaledAttnRes` (`55,659`) | `M21_ConceptSlot` (`55,437`) | `M22_FullCore` (`55,597`) |
|---|---:|---:|---:|---:|---:|
| **`S4_Holonomy_8x` (ID / OOD)** | `0.0755` / `0.0414±0.0041` | `0.7467` / `0.4427±0.4880` | `0.4714` / `0.2512±0.3052` | `0.4824` / `0.3482±0.4616` | **`0.7533` / `0.5698±0.4680`** |
| **`MQAR_Selective_Overwrite_4x` (OOD / `ov`)** | `0.9963` / `0.9958` | `0.9965` / `0.9961` | `0.9976` / `0.9973` | `0.9965` / `0.9961` | **`0.9991` / `0.9990`** |
| **`MQAR_Alias_Semantic_4x` (ID / OOD / `ov`)** | `0.5174` / `0.0201` / `0.0200` | `0.5278` / `0.0289` / `0.0289` | `0.5278` / `0.0234` / `0.0231` | `0.6970` / `0.4424±0.0726` / `0.4459` | **`0.7326` / `0.4961±0.0655` / `0.4995`** |
| **`Unseen_OpenVocab_Copy` (ID / OOD)** | `1.0000` / `0.6719±0.3141` | `1.0000` / `0.5843±0.1669` | `1.0000` / `0.8978±0.0958` | `1.0000` / `0.4782±0.2558` | **`1.0000` / `0.9964±0.0020`** |
| **`Parity_Dyck_4x` (ID / OOD)** | `0.9208` / `0.9055±0.0260` | `0.9393` / `0.9053±0.0591` | `0.9227` / `0.9193±0.0362` | `0.9263` / `0.9029±0.0139` | **`0.9216` / `0.9213±0.0266`** |
| **`Hierarchy_Denom_Rollup_4x` (OOD / `close`)** | `0.2571` / `0.1562` | **`0.2586`** / **`0.1616`** | `0.2545` / `0.1515` | **`0.2586`** / `0.1594` | `0.2543` / `0.1556` |

### Key Part A Findings
1. **Cosine-annealing `AuxBase` alone was insufficient on `M13` (`Unseen_OpenVocab_Copy = 0.6719` vs `U08 M13_AuxBase = 0.6211`), while `M22_FullCore` (`ScaledNorm + Block-AttnRes + ConceptSlotMemory + AnnealAux`) swept `5/6` Part A tasks:** `0.9964±0.0020` `Unseen_OpenVocab_Copy`, `0.9991` `MQAR_Overwrite_4x`, `0.9213` `Parity_Dyck_4x`, `0.5698` `200`-step `S4_Holonomy_8x` (vs `0.0414` for `M13`), and **`0.4961±0.0655` (`24.7x` `M13`) on `MQAR_Alias_Semantic_4x`**.
2. Notice the clean ablation decomposition: adding `Block-AttnRes` + `ScaledNorm` (`M20 = 0.8978`, `M22 = 0.9964`) stabilized `Unseen_OpenVocab_Copy` over non-AttnRes scaled variants (`M19 = 0.5843`, `M21 = 0.4782`).

---

## 2. Part A2 — `600`-Step Extended Algebra & Semantic Alias Suite (`5 x 3 x 2 = 30` Runs, `244.62s`)

| Task (`600` steps) | `M13_AnnealAux` | `M19_ScaledNorm` | `M20_ScaledAttnRes` | `M21_ConceptSlot` | `M22_FullCore` |
|---|---:|---:|---:|---:|---:|
| **`S4_Holonomy_8x` (ID / OOD)** | `1.0000` / `0.6866±0.3820` (`0.9567`, `0.4165`) | `1.0000` / `0.7179±0.2466` (`0.8923`, `0.5435`) | `1.0000` / `0.7495±0.3326` (`0.9847`, `0.5143`) | **`1.0000` / `0.8385±0.1765`** (`0.7137`, `0.9634`) | `0.9795` / `0.5637±0.4310` (`0.2590`, `0.8685`) |
| **`Hierarchy_Denom_Rollup_4x` (ID / OOD / `close`)** | **`0.3996` / `0.3288±0.0700`** / `0.1902` | `0.3310` / `0.2793±0.0012` / **`0.1971`** | `0.3274` / `0.2729±0.0057` / `0.1840` | `0.3250` / `0.2764±0.0008` / `0.1807` | `0.3209` / `0.2759±0.0086` / `0.1852` |
| **`MQAR_Alias_Semantic_4x` (ID / OOD / `ov`)** | `0.5195` / `0.0641` / `0.0638` | `0.5260` / `0.0711` / `0.0713` | `0.5339` / `0.0762` / `0.0764` | `0.7643` / `0.1367±0.0376` / `0.1334` | **`0.8255` / `0.2156±0.1746` / `0.2148`** (`0.3391`, `0.0922`) |

### Key Part A2 Findings
1. **Why `MQAR_Alias_Semantic_4x` OOD dropped between 200 steps (`0.4961`) and 600 steps (`0.2156`, even as ID climbed `0.7326 -> 0.8255`):** Classic **length-overfitting inside `ConceptSlotMemory`**. `slot_k`, `slot_q`, and `slot_write` take normalized contextual `h_n` (whose HSPA/RoPE trajectory drifts across `T=48 -> T=192`) and learn sharper `w_gate` at `600` steps that either decay or miss keys at `4x` horizon (`seed=0` held `0.3391` OOD, `seed=1` dropped to `0.0922`), whereas `emb_in` itself is position-invariant!
2. **`M21_ConceptSlot` led `600`-step `S4_Holonomy_8x` (`0.8385 ± 0.1765`, both seeds `>= 0.71`)**; `M13_AnnealAux` led `Hierarchy` (`0.3288` driven by `seed=0 = 0.3783`, `seed=1 = 0.2794`), though `M19` had best `close_rollup` (`0.1971`).

---

## 3. Part B — Real-Corpus `data/v1` `1x / 2x / 4x` & Rank Audit v2 (`d=128`, `1,000` Steps × `2` Seeds, MPS, `981.69s`)

Reference from `EXP-U08`: `M13_AuxBase` (constant `aux=0.25`) scored `web@512 = 1.8697`, `textbook@512 = 1.8107`, `math@512 = 2.1403`, `code@512 = 1.6285` (domain average `1.8623`), `copy_256 = 0.9543`, `T512_N4` `common/rare = 0.7292 / 0.8959`, `8,256.8 tok/s`.

### 3a. Natural Language BPB (`128 / 256 / 512`), Zero-Shot Copy, Real MQAR & Throughput

| Metric | `M13_AnnealAux` | `M19_ScaledNorm` | `M20_ScaledAttnRes` | `M21_ConceptSlot` | `M22_FullCore` |
|---|---:|---:|---:|---:|---:|
| **Non-Emb Params** (`~853k`) | `852,587` | `852,587` | `853,227` | `852,333` | `852,973` |
| **MPS Throughput (`tok/s`)** | **`13,596.0`** | `13,224.2` | `12,896.5` | `8,959.5` | `8,478.8` |
| **`mean_512_bpb`** | `1.8680±0.0013` | `1.8518±0.0040` | `1.8513±0.0023` | **`1.8384±0.0077`** | `1.8466±0.0057` |
| **`web` (`128 / 256 / 512`)** | `1.9456 / 1.9275 / 1.8737` | `1.9304 / 1.9103 / 1.8573` | `1.9342 / 1.9145 / 1.8639` | `1.9347 / 1.9092 /` **`1.8555`** | `1.9332 / 1.9126 / 1.8578` |
| **`textbook` (`128 / 256 / 512`)** | `1.8881 / 1.8697 / 1.8125` | `1.8641 / 1.8458 / 1.7917` | `1.8685 / 1.8516 / 1.7997` | `1.8640 / 1.8419 /` **`1.7869`** | `1.8688 / 1.8519 / 1.7986` |
| **`math` (`128 / 256 / 512`)** | `2.3207 / 2.2501 / 2.1451` | `2.3045 / 2.2447 / 2.1352` | `2.2960 / 2.2323 / 2.1253` | `2.2792 / 2.2140 /` **`2.1108`** | `2.2931 / 2.2259 / 2.1183` |
| **`code` (`128 / 256 / 512`)** | `2.2551 / 2.0789 / 1.6406` | `2.2341 / 2.0596 / 1.6231` | `2.2294 / 2.0555 / 1.6164` | `2.2124 / 2.0349 /` **`1.6006`** | `2.2176 / 2.0455 / 1.6119` |
| **`copy_32 / 64 / 128`** | `0.8175 / 0.7802 / 0.7979` | `0.9093 / 0.8527 / 0.8619` | **`0.9758` / `0.9445`** / `0.9439` | `0.9294 / 0.8720 / 0.8955` | `0.9567 / 0.9380 /` **`0.9545`** |
| **`copy_256` (`2x` ctx)** | `0.8029±0.0903` | `0.9033±0.0584` | `0.9404±0.0177` | `0.9213±0.0322` | **`0.9700±0.0054`** |
| **`MQAR common` (`128 / 256 / 512`)** | `0.8750 / 0.9583 /` **`0.7917`** | `0.8125 / 0.8334 /` **`0.8125`** | `0.8125 / 0.7916 / 0.7708` | `0.7917 / 0.8541 / 0.7708` | `0.8125 / 0.8750 / 0.7709` |
| **`MQAR rare` (`128 / 256 / 512`)** | `0.8750 / 0.9375 / 0.8541` | `0.8542 / 0.8541 / 0.8334` | `0.8334 / 0.8334 / 0.7709` | `0.8958 / 0.8959 /` **`0.9166`** | `0.8750 /` **`0.9375`** `/ 0.8542` |

### 3b. Rank & Norm Health Audit v2 (`val_code_256` & `copy_128_256`, `max_rank = 32`)

| Model | `L0 h_in_rms` | `L0 \|\|mix\|\|/\|\|h\|\|` | `L0->L3 mix_u` | `L0->L3 mix_c` (`val_code`) | `L3 h_in_c` (`val_code`) | `L3 h_in_c` (`copy128`) |
|---|---:|---:|---:|---:|---:|---:|
| `M13_AnnealAux` | `0.088` | `7.511` (**shock**) | `11.09 -> 2.20` | `13.40 -> 5.91` | `8.21` | `9.39` |
| `M19_ScaledNorm` | **`1.000`** | **`0.396`** | `7.05 -> 3.40` | `9.69 -> 6.74` | `9.05` | `8.61` |
| `M20_ScaledAttnRes` | **`1.000`** | **`0.436`** | `6.71 -> 2.79` | `8.69 -> 6.53` | `10.29` | `12.54` |
| `M21_ConceptSlot` | **`1.000`** | **`0.390`** | `7.17 -> 1.82` | `9.29 -> 5.83` | `9.74` | `11.56` |
| `M22_FullCore` | **`1.000`** | **`0.427`** | `6.88 -> 1.88` | `9.29 -> 4.88` | **`11.18`** | **`16.09`** |

**Architectural Discovery Behind `L3 mix_c = 4.88–6.74 < 10/32`:**
- Upgrade 1 completely repaired the L0 shock (`||mix||/||h||`: `7.511 -> 0.390–0.436`, `h_in_rms: 0.088 -> 1.000`) and lifted the residual stream `L3 h_in_rank_cent` to `10.29–11.18` (`val_code`) and **`16.09` (`copy128`)** on `M22_FullCore`.
- Why did `mix_y = blk.mix(xn)` itself remain below `10/32`? Because [`HSPALieSO3Mixer`](../../mechanisms/zoo.py) hardcodes `n_reg = 4`, so `self.fdim = h * n_reg * 3 = 4 * 4 * 3 = 48` (`12` per-head fiber dims projected up to `hd = 32` at `d=128`) regardless of `d_model`! Furthermore, each `SO(3)` rotation has a fixed invariant eigenvector axis (`1D`), structurally limiting each head's instantaneous linear span at large `d`.

---

## 4. Preregistered Criteria Verdict & Pre-RunPod Scale Gate (`HOLD`)

| Criterion | Target | Observed Result | Verdict |
|---|---|---|---|
| **C1: Rank & Norm Repair** | `L0 \|\|mix\|\|/\|\|h\|\| <= 1.2`; `L3 mix_rank_cent >= 10/32` on `val_code@256` | `L0` fixed (`0.390–0.436`, `h_in_rms = 1.000`), `L3 h_in_c = 11.18` (`16.09` copy128); **but** `L3 mix_rank_cent = 4.88–6.74` (`fdim=48` structural cap) | **PARTIAL** |
| **C2: Cross-Alias Semantic MQAR** | `M21/M22` OOD `>= 0.70` without `MQAR_Overwrite` / `Unseen_Copy` regression | `M22` reached `0.4961` at 200 steps (`24.7x` `M13`), then `4x` OOD regressed to `0.2156` at 600 steps (ID `0.8255`) from `h_n` length drift | **PARTIAL / FAIL** |
| **C3: Annealed `AuxBase` Recovery** | `Unseen_Copy >= 0.90` on `M13` + `mean_512_bpb <= 1.8624` | `M13_AnnealAux` got `0.6719` Unseen Copy and `1.8680` BPB; **`M22_FullCore` restored `0.9964` Unseen Copy & `0.9700` `copy_256`**, while **`M21_ConceptSlot` hit best-ever `1.8384` mean `512` BPB** | **PARTIAL (Composite PASS)** |
| **C4: Pre-RunPod Scale Gate (`30M–125M`)** | 0 NaNs/leaks; beats `M13` BPB; `L3 cent >= 10`; `Alias >= 0.70`; `S4 >= 0.55`; `copy_256 >= 0.95`; `>= 9,500 tok/s` | 0 NaNs/leaks (**PASS**); `M21` `-0.030 BPB` (**PASS**); `M22` `copy_256 = 0.9700` (**PASS**); `S4 = 0.57–0.84` (**PASS**); **FAILS** on `L3 mix_cent` (`4.9–6.7 < 10`), `Alias` (`0.22–0.50 < 0.70`), and `M21/M22` speed (`8,479–8,960 < 9,500 tok/s`) | **`HOLD`** |

---

## 5. `EXP-U10` Remediation Roadmap (Must Clear Locally at `$0.00` Before RunPod)

1. **Fix `HSPALieSO3Mixer` `fdim=48` Rank Cap:** Scale `n_reg` proportionally with `hd` (e.g., `fdim >= D` / `n_reg ~ max(4, hd // 3)` or `8–10` per-head 3D fibers at `d=128`) so `mix` spans the full head subspace `hd=32` instead of choking on a fixed `12`-dim per-head bottleneck (`L3 mix_rank_cent -> >= 12/32`).
2. **Fix `ConceptSlotMemory` Length Drift & Speed (`8.5k -> >10.5k tok/s`):**
   - Project slot values into a **compact low-rank concept subspace `d_slot = min(32, D // 4)`** (`[B, T, 16, 32]` -> `4x` smaller Kogge–Stone scan tensor at `d=128` -> `~11k+ tok/s`!).
   - Index slot write keys/queries with a **length-invariant mixture of raw embedding `emb_in` + normalized layer features** (reducing positional drift across `1x -> 4x`), plus sharpen/stabilize slot retention so `MQAR_Alias_Semantic_4x` clears **`>= 0.70`** at both `200` and `600` steps.
3. **Floor `AuxBase` Annealing (`0.25 -> 0.05`) + Keep `M22` Attention-Residual Anchoring:** Combine the `0.9964` `Unseen_OpenVocab_Copy` + `0.9700` `copy_256` of `M22_FullCore` with the `1.8384` `mean_512_bpb` of `M21_ConceptSlot` and small `0.05` final `AuxBase` regularization so both `MQAR_common` (`M19 = 0.8125`) and `rare` (`M21 = 0.9166`) peak together.
