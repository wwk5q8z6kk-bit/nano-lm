# EXP-U09 Preregistration: Pre-RunPod Hardening Gate (`CenteredRMSNorm` + Residual Scaling + `Block-AttnRes` + Cosine-Annealed `AuxBase` + Differentiable `ConceptSlotMemory`)

- **Experiment ID:** `EXP-U09`
- **Date Preregistered:** `2026-11-19`
- **Branch / Worktree:** `research/unified-discovery-loop-v1`
- **Compute Budget:** `$0.00` paid cloud compute (local CPU multi-process + local MPS GPU only)
- **Prerequisite Diagnosis:** `task-1537` root-cause diagnostic on `M0` & `M13` (`data/v1`, `d=128, H=4`, `val_code@256`):
  - `M13 @300` exhibited a **`7.48x` L0 residual shock** (`||h_0|| = 1.0`, `RMS = 0.088`, while `RMSNorm(h_0)` outputs `RMS = 1.0`, wiping initial token embeddings in layer 0) and **`L3` DC-offset rank decay** (`uncent=1.97`, `cent=4.25/32`, `cos_u=0.823`, `cos_c=0.081`, `||mix||/||h||=0.19`).

## Architectural Upgrades Evaluated (Zero Edits to `nanoscribe/` or Frozen Artifacts)
1. **Upgrade 1a (`ScaledNorm`, `M19`):** Replace `RMSNorm` with `CenteredRMSNorm` (subtracts last-dim mean before `rms_norm` to strip DC drift), initialize residual stream at unit RMS (`h = sqrt(d_model) * normalize(emb(x))`), and scale `mix.o.weight` & `ffn.w2.weight` by `1 / sqrt(2 * n_layers)`.
2. **Upgrade 1b (`ScaledAttnRes`, `M20`):** Enable depthwise `Block-AttnRes` (`attn_res=True`) with `CenteredRMSNorm` key normalization over block outputs `[h_0..h_l]`.
3. **Upgrade 2 (`ConceptSlot`, `M21`):** Differentiable $O(T \log T \cdot C \cdot D)$ (`C = 16`) causal `ConceptSlotMemory` wired **before** `sparse_evid` pointer mixing:
   - Strictly causal writes `(h_{s-1} -> emb_in_s)` for `s = 1..T-2` via Kogge-Stone parallel affine scan (`M_s[c] = (1 - alpha_{s,c}) M_{s-1}[c] + alpha_{s,c} emb_in_s`).
   - Query `t` reads strictly prior state `M_{t-1}` (`t <= 1` guaranteed zero memory / `g_slot == 0`) and mixes into `probs_base` via norm-aware gate `(1 - g_slot) * probs_base + g_slot * probs_slot`.
4. **Upgrade 3 (`AnnealAux`):** Cosine-anneal `AuxBase` weight from `0.25 -> 0.0` across training steps (early backbone alignment without late-stage logit calibration penalty).
5. **Full Core (`M22`):** Combines Upgrades 1a + 1b + 2 + 3 (`M22_HSPA_FullCore_U09`).

## Iso-Parameter Contenders (Strict Causal Parity Verified)
| Contender Label | Zoo Mechanism ID | `d=32, V=72` Params (`[54.5k, 56.5k]`) | `d=128` FFN Hidden | `d=128` Non-Emb Params (`[851.5k, 853.5k]`) | `AuxBase` Schedule |
| :--- | :--- | ---: | ---: | ---: | :--- |
| `M13_AnnealAux` | `M13_HSPA_EvidCosSparsePtr` | `55,499` | `463` | `852,587` | Cosine `0.25 -> 0.0` |
| `M19_ScaledNorm` | `M19_HSPA_ScaledNorm_EvidPtr` | `55,499` | `463` | `852,587` | Cosine `0.25 -> 0.0` |
| `M20_ScaledAttnRes` | `M20_HSPA_ScaledAttnRes_EvidPtr` | `55,659` | `463` | `853,227` | Cosine `0.25 -> 0.0` |
| `M21_ConceptSlot` | `M21_HSPA_ConceptSlot_EvidPtr` | `55,437` | `460` | `852,333` | Cosine `0.25 -> 0.0` |
| `M22_FullCore` | `M22_HSPA_FullCore_U09` | `55,597` | `460` | `852,973` | Cosine `0.25 -> 0.0` |

## Preregistered Evaluation Protocol
- **Part A (CPU, `d=32`, `5 x 6 x 3 = 90` runs, `200` steps):** All 5 contenders across `{S4_Holonomy_8x, MQAR_Selective_Overwrite_4x, MQAR_Alias_Semantic_4x, Unseen_OpenVocab_Copy, Parity_Dyck_4x, Hierarchy_Denom_Rollup_4x}`, seeds `{0, 1, 2}`. Output: `results_part_a.json`.
- **Part A2 (CPU, `d=32`, `5 x 3 x 2 = 30` runs, `600` steps):** Extended learning horizon on `{S4_Holonomy_8x, Hierarchy_Denom_Rollup_4x, MQAR_Alias_Semantic_4x}`, seeds `{0, 1}`, saving `per_seed`. Output: `results_part_a2_algebra.json`.
- **Part B (MPS, `d=128, H=4`, `5 x 2 = 10` runs, `1,000` steps):** `/Volumes/Express1M2/nanolm-lab/data/v1`, seeds `{0, 1}`, evaluating `128 / 256 / 512` domain BPB (`web, textbook, math, code`), `copy_{32,64,128,256}`, real-haystack MQAR (`common` & `rare_tail` across `128/256/512`), throughput (`tok/s`), and `audit_layer_ranks_v2` (both uncentered & time-centered effective rank + `||mix||/||h||` across L0–L3). Output: `results_part_b.json`.

## Preregistered Success Criteria & Pre-RunPod Scale Gate
- **Criterion 1 (Rank & Norm Repair):** Best `U09` (`M19/M20/M22`) resolves `L0` residual shock (`L0 ||mix||/||h|| <= 1.2`, vs `7.48` for `M13`) and `L3` collapse (`val_code@256` L3 time-centered mixer rank `>= 10 / 32`, vs `4.25` in `M13`) without degrading BPB.
- **Criterion 2 (Cross-Alias Semantic Recall):** `M21` / `M22` reaches `>= 0.70` OOD accuracy on `MQAR_Alias_Semantic_4x` (Part A or Part A2, vs `0.067` in `EXP-U08`) while preserving `MQAR_Selective_Overwrite_4x` and `Unseen_OpenVocab_Copy`.
- **Criterion 3 (Annealed `AuxBase` Calibration):** `M13_AnnealAux` restores `200`-step `Unseen_OpenVocab_Copy` OOD (`>= 0.90` vs `0.621` constant `0.25` in `U08`) and matches or improves `data/v1` `512`-tok mean BPB (`<= 1.8624` vs `M13_AuxBase` in `U08`).
- **Criterion 4 (Pre-RunPod Scale Gate `30M-125M` Readiness):** **PASS** iff ALL FOUR hold:
  1. Zero NaN / Inf and `0.0` causal leak across CPU & MPS;
  2. Best `U09` candidate strictly beats `M13_AnnealAux` and `U08` `M13_AuxBase` (`1.8624`) on mean `512`-tok `data/v1` BPB;
  3. Rank/norm repair passes (`L3 cent_rank >= 10/32`, `L0 ||mix||/||h|| <= 1.2`);
  4. `MQAR_Alias_Semantic_4x` OOD `>= 0.70`, `S4_Holonomy_8x` `>= 0.55`, `copy_256 >= 0.95`, and MPS throughput `>= 9,500 tok/s`.
  Otherwise status is **HOLD** and no paid RunPod campaign is authorized yet.
