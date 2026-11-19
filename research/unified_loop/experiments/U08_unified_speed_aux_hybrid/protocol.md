# EXP-U08 Protocol — Unified Speed-Opt `HSPA` Body Scan, Auxiliary Base-LM Loss, Multi-Timescale `M17`, Hybrid Semantic–Lexical Pointer `M18`, & Key-Rank Health Audit

- **Experiment ID:** `EXP-U08`
- **Status:** Preregistered BEFORE execution
- **Hardware / Cost:** Local Apple Silicon M3 Max (`cpu` 5 workers for Part A & Part A2, `mps` for Part B), `$0.00` paid cloud compute

## 1. Hypotheses & Mechanistic Rationale
`EXP-U07` proved that **Invention A (`solve_triangular`)** eliminated `FullSphere`'s `4×` `NaN`s (`M16` `copy_256 = 0.9977`) and **Invention B (`sparse_evid`)** Pareto-improved `HSPA` (`M13` `copy_256 = 0.9405`, `T512_N4` rare MQAR `0.8125`, overwrite `0.9944`, `#1` Hierarchy `0.2628`, `#1` BPB `web@512=1.8766, code@512=1.6596`), while exposing four remaining gaps:
1. **`HSPA` MPS Speed Bottleneck (`7,440 tok/s`) + CPU↔MPS Candidate Sync:** Matrix $3\times 3$ Kogge–Stone doubling + CPU double-loop candidate indexing cost `~101.6 ms/step` (`B=8, T=128`), whereas our gauge-decoupled $SU(2)$ quaternion + scalar body scan outputting only damped 3D `S_t` (`48` dims, zero un-decayed `hol_6d` bloat) + vectorized on-device candidate lookup cuts step time to `~62.7 ms/step` (**`1.62×` full-step speedup**, `7.6e-6` exact equivalence).
2. **Pointer–Backbone Gradient Starvation on Pure Algebra (`Part A2` `S4`: `M12`/`M15` `0.045` vs `M5`/`M14` `0.489–0.516`):** When `sparse_evid` dominates early token loss, the backbone receives suppressed gradient on structural transitions. Adding auxiliary base-LM loss $\mathcal{L} = \mathcal{L}_{\text{mix}} + 0.25\,\mathcal{L}_{\text{base}}$ prevents backbone starvation without impairing pointer routing.
3. **Multi-Timescale Damped `HSPA` (`M17`):** Unlike `M14`/`M15` (which exported `144`-dim un-decayed `hol_6d` and starved FFN), `M17` keeps `HSPA`'s compact `48`-dim damped `S_t` while adding `CausalDepthwiseConv(k=4)` linear residual + log-spaced `a_bias` `[0.0, 1.5, 3.0, 4.5]` for nested depth persistence.
4. **Exact-Token-Only Pointer Ceiling on Paraphrase/Alias Retrieval (`M18`):** Replacing `8` of `24` local slots with `8` strictly causal top-salience semantic landmark slots (`ptr_salience(h_n)`) enables causal retrieval across distinct entity aliases (`[4..11] <-> [12..19]`) without hurting exact lexical recall or $O(T \cdot K)$ scaling.

## 2. Contenders & Exact Verified Iso-Parameter Parity
All contenders match `55.0K–55.5K` total params at `d=32, V=72` and `851.7K–852.7K` non-embedding params at `d=128`:
- `M13_Ctrl` (`M13_HSPA_EvidCosSparsePtr`, `aux_base_weight=0.0`): `d=32 ffn=70 (55,499)`, `d=128 ffn=463 (non_emb=852,587)`
- `M13_AuxBase` (`M13_HSPA_EvidCosSparsePtr`, `aux_base_weight=0.25`): `d=32 ffn=70 (55,499)`, `d=128 ffn=463 (non_emb=852,587)`
- `M17_AuxBase` (`M17_HSPA_MultiScale_EvidCosSparsePtr`, `aux_base_weight=0.25`): `d=32 ffn=68 (55,371)`, `d=128 ffn=461 (non_emb=851,691)`
- `M18_AuxBase` (`M18_HSPA_HybridSemLexPtr`, `aux_base_weight=0.25`): `d=32 ffn=70 (55,532)`, `d=128 ffn=463 (non_emb=852,716)`
- `M16_AuxBase` (`M16_FullSphereTri_EvidCosSparsePtr`, `aux_base_weight=0.25`): `d=32 ffn=68 (55,227)`, `d=128 ffn=301 (non_emb=852,027)`

*(Part A2 also includes `M5_Pure` (`M5_HSPA_LieSO3`, `ffn=78, 55,265`) as unpointered algebraic control.)*

## 3. Suite & Falsification Criteria
- **Part A (`cpu`, 5 workers):** `5` contenders × `6` tasks (`S4_Holonomy_8x`, `MQAR_Selective_Overwrite_4x`, `Unseen_OpenVocab_Copy`, `Parity_Dyck_4x`, `Hierarchy_Denom_Rollup_4x`, and new `MQAR_Alias_Semantic_4x`) × `3` seeds (`200` steps).
- **Part A2 (`cpu`, 5 workers, `600` steps, per-seed saved):** `M5_Pure`, `M13_Ctrl`, `M13_AuxBase`, `M17_AuxBase`, `M18_AuxBase` × `{S4_Holonomy_8x, Hierarchy_Denom_Rollup_4x, MQAR_Alias_Semantic_4x}` × `2` seeds.
- **Part B (`mps`, `data/v1`, `d=128`, `1,000` steps × `2` seeds, `1×/2×/4×` eval):** Standard BPB/copy/MQAR evaluation + **`L0→L3` Effective Rank & Mean Off-Diag Cosine Audit** on `val_code@256` and `copy_128`.
- **Falsification Criteria:**
  1. **Criterion 1 (Speed Unification):** `M13_Ctrl` reaches `>= 1.5×` higher MPS `tok/s` than `EXP-U07` `M13` (`7,440 tok/s`, target `>= 11,500 tok/s`) while matching `web@512 <= 1.89`, `code@512 <= 1.68`, `copy_256 >= 0.90`.
  2. **Criterion 2 (AuxBase Loss):** `M13_AuxBase` improves `Part A2` `S4_Holonomy_8x` / `Hierarchy_Denom_Rollup_4x` or `Part B` natural BPB without degrading `copy_256` (`>= 0.90`) or `MQAR_Overwrite_4x` (`>= 0.98`).
  3. **Criterion 3 (Multi-Timescale Damped `M17`):** `M17_AuxBase` improves `Hierarchy_Denom_Rollup_4x` and/or `data/v1` `4×` BPB over `M13_AuxBase` without `M14`/`M15` regression (`web@512 <= 1.90`).
  4. **Criterion 4 (Hybrid Semantic+Lexical `M18`):** `M18_AuxBase` beats exact-only `M13_AuxBase` on `MQAR_Alias_Semantic_4x` OOD without regressing `Unseen_OpenVocab_Copy`, `MQAR_Overwrite_4x`, or `data/v1` BPB.
  5. **Criterion 5 (Key-Rank Health Gate):** Trained `M13`/`M17`/`M18` maintain healthy `L0→L3` effective rank (no `S1_125m`/`M1`-style collapse to `<= 4.5/32`), `has_nonfinite = False` everywhere.
