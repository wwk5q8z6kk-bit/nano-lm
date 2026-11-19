# Nano LM — Verified Unified Reality Map & Core-and-Web Frontier State

**Worktree:** `/Users/mac/Projects/nano-lm/.worktrees/nanolm-unified-loop` (`research/unified-discovery-loop-v1`, base `f82d82a`)  
**Updated:** `2026-11-18` · **Paid Compute Added in Loop:** `$0.00` (Local Apple Silicon CPU + MPS only)  
**Rule:** Every claim is strictly labeled (`verified local result` | `verified artifact on disk` | `external peer branch artifact` | `contradicted / invalid` | `unverified / proposed`). Frozen historical evidence (`H6`, `H7`, `papers/`) remains untouched.

---

## 1. Strategic Architecture Shift: From 1D Medical Ladder to "Milky Way" Core-and-Web

Earlier governance organized Nano LM as a linear 9-stage product ladder (`P1 Master Scribing -> P2..P9`) anchored on `NanoScribe` (`DomainPack #1`). Empirical work in `EXP-U01` through `EXP-U05` showed why a domain-general sub-4B frontier model requires a **Galactic Core + Interconnected Capability Web** instead:

```text
               [ Outer Domain Adapters — Thin, Replaceable Schemas ]
        Finance/Treasury · SRE/Systems · Legal/SLA · Code/Math · Clinical (Passive Legacy)
                                          ▲
                                          │ declarative schemas (`nano/schema.py`, `nano/units.py`)
               [ Spiral Arms — Mutually Coupled Capability Web ]
    Exact Copy & Recall ◄──► Hierarchical & Algebraic Composition ◄──► Epistemic Control & Verification
              ▲                              ▲                                   ▲
              └──────────────► Long-Horizon State & Natural Modeling ◄───────────┘
                                          ▲
                                          │ emerges from
               [ Galactic Core — Minimal Shared Primitives (`zoo.py` + `nano/`) ]
    1. Causal Content Copy-Pointer Channel (`M9`, solves Unseen Copy & boosts Code/Math BPB)
    2. Geometric Orthogonal Recurrence (`M5_HSPA_LieSO3` / `M7_SkewCayley_Flow`, beats Attention on Hierarchies & Textbook BPB)
    3. Exact Unit/Denomination + Dynamic Hierarchy Algebra (`nano/units.py`, `nano/schema.py`, `nano/grounded_engine.py`)
    4. Inference-Time Monotonic Margin Guard / Abstention (`M8` Guard, `E24` clean entropy-margin routing)
```

- **`nanoscribe/` Posture:** Refactored into a clean declarative schema plugin (`CLINICAL_ENCOUNTER_SCHEMA` over `nano/`, `e7f4933`), verified at `235/276` end-to-end grounded retention with `0/7` hazards (`0` `N-03`), and **quarantined as a passive legacy domain regression suite** — zero frontier architecture decisions depend on it.

---

## 2. Verified Outcomes from Autonomous Loop (`EXP-U01` through `EXP-U05`, `$0.00` Paid Compute)

| Exp ID | Commit(s) | Scope & Artifact | Verified Headline Results (`verified local result`) |
|---|---|---|---|
| **EXP-U01** | `1dac170`, `01bdda7`, `adcfdbb` | Forensic autopsy of `X4_46m`, `S0_125m`, `S1_125m` (`600M`-tok checkpoints) — `U01_real_125m_autopsy/results.json` | **`S1_125m` Delta key collapse diagnosed:** Past Layer 0, Delta keys collapsed to `eff_rank = 2.9–11.3 / 64` (`val_code`) and `2.06–5.92 / 64` (`copy_probe`), `offdiag_cos = 0.503–0.888`, `decay_a = 0.036–0.366`. At `L=2048` (`2×`), text `web` BPB improved (`1.0821 -> 1.0068`), while `copy_1024` collapsed **`28×` deeper** in `S1` (`0.0021`, loss `11.6306`) than `S0` (`0.0582`, loss `9.628`). Zero-shot `val_web` MQAR `T256_N1`: `S1 = 0.0833/0.1667` vs `S0 = 0.2500/0.2500` (`T1024_N8`: `0.00–0.08`). |
| **EXP-U02 (Part A)** | `a4179f3`, `5a178ae` | Iso-param micro-lab (`55.3K–56.4K`), `10` mechanisms × `5` tasks × `3` seeds (`426.7s`) — `results_part_a.json` | 1) **Only `M9_DualChannel_Pointer` solved `Unseen_OpenVocab_Copy` (`0.9619 ± 0.0244` vs `0.0000` for `M0–M4, M6–M8`) & led `MQAR_4x` (`0.5764 ± 0.0794`, `ov=0.5421` vs `0.058–0.093`).** 2) **Geometric state doubled Attention on `Hierarchy_Denom_Rollup_4x` (`depth 4..8`):** `M5_HSPA_LieSO3 = 0.2588 (close=0.1633)`, `M7_SkewCayley_Flow = 0.2560 (close=0.1534)` vs `M0 = 0.1459 (close=0.1140)`. 3) `Parity_Dyck_4x`: `M7` led (`0.9279`). 4) `S4_Holonomy_8x`: `M5` led ID (`0.1029`), all ~chance (`~0.042`) at `8×` OOD at 200 steps. |
| **EXP-U02 (Part B)** | `5a178ae` | Real corpus transfer (`data/v1`, `d=128`, `~853K` non-emb / `~5.05M` total, `1,000` steps × `2` seeds, `819.88s`) — `results_part_b.json`, `analysis.md` | 1) **`M9_DualChannel_Pointer` crushed all non-pointer baselines across `web`, `math`, and `code`:** `web@256 = 1.9960` (`vs M0 2.0611`, `M1 2.1298`), `math@256 = 2.3172` (`vs M0 2.7016`), **`code@256 = 2.1983` (`-0.626 BPB` vs `M0 2.8245`, `-0.903 BPB` vs `M1 3.1013`)**; zero-shot `copy_32/64/128 = 0.910/0.933/0.926` vs `0.000` for `M0, M1, M5, M7`; MQAR `T128_N4 common = 0.3125` (`0.0000` elsewhere). 2) **Pure $O(T)$ `M5_HSPA_LieSO3` beat RoPE Attention (`M0`) & `GDN1` (`M1`):** best `textbook@128 = 1.9336` (`vs M0 1.9663`, `M1 2.0457`) and `#2` `web@256 = 2.0324`. 3) Legacy `M1_GDN1` finished last on all 4 domains (`code@128 = 3.1180`) and repeated Layer `0->3` key rank collapse (`12.82 -> 4.45 / 32`, `cos = 0.6225`). |
| **EXP-U03** | `f7ccb71`, `adcfdbb` | Scaffold attribution (`E23 C7-Omega`, `3` seeds) + routing paradox (`E24 HEM-ATRD`, `5` seeds) — `U03_scaffold_routing_audit/results.json` | 1) `C7-Omega`: Python `SymbolicBeliefState` binary features gave `0.000` gain; `A4_frontier_sym0` (`sym_feats=OFF`, no shield) hit `1.0000 ± 0.0000` Deep-OOD (`0.9967` Stress, `0.0033` Unsafe) via frontier-relative De Bruijn gauge (`\0`), matched by a `0`-param deterministic rule (`1.0000`). 2) `E24`: `H1_x5` (`1.0000 ± 0.0000`) beat `H4_hem` (`0.7469 ± 0.3621`) and `H3_mm` (`0.6922 ± 0.1851`) due to `u_sel * dx_total` grad scaling bug (`margin_shield_refiner.py:173`), Pass-2 starvation/variance, `min_ambiguity_floor=0.46`, and Lyapunov shield rejecting `-17.2 pp` valid Pass-2 fixes in `H3`. |
| **EXP-U04** | `5c1be63`, `09f688c`, `adcfdbb`, `e7f4933` | Grounded verifier diagnosis + Qwen2.5-1.5B (`C12–C14`) — `U04_nanoscribe_verifier_qwen/results.json` | Fixed determiner & false-conflict drops (`200/276 -> 228/276` Gold, `242/276` menu, `235/276` e2e, `72/72` useful notes, `0` `N-03`, `0/7` hazards). Live `Qwen2.5-1.5B-Instruct` on `C14` (`166.14s` MPS, prefix cache): `33/80 (41.2%)` asserted (`65/120` all, `0` custody violations) vs `0`-param deterministic menu rule `63/80 (78.8%)` (`91/120` all). |
| **EXP-U05** | `be9cd07`, `e7f4933`, `283b832`, `01e42c6` | Domain-pure modular foundation (`nano/units.py`, `nano/schema.py`, `nano/grounded_engine.py`) across 4 domains — `U05_cross_domain_hierarchy_denom/results.json` | **`zero_domain_branches_verified = True`** in `nano/`. Identical engine across Finance (`24`), SRE (`24`), and Legal (`24`): **`Full_Modular_Nano = 72/72 (100.0%)`** (`35/35` equiv/rollup, `24/24` hazard/fail-closed) vs `DenomOnly_FlatTopics = 66/72 (91.7%)` and `FlatString_NoDenom_NoHierarchy = 17/72 (23.6%)`, plus legacy Clinical `C12–C14` (`235/276` e2e, `0/7` hazards). |

---

## 3. Historical Context & External Branch Status (Unmodified)

- **Early Own-Stack Checkpoints (`3.15M`, `10.0M`, `159.3M`), `H6` (`3.29M`, `REJECTED`), `H7` (`INVALID_GOLD`):** Preserved frozen on `origin/master`.
- **Track B (`B6–B8-3B`):** No local weights on disk; earlier `8-Pillar` (`85.61%` vs `Comp-1 12%` with `think=1`) depended on `System-v4` (`69%` `SmolLM3-3B Base + System-v4` > `66%` `B8-3B`).
- **Active Peer Cloud/Local Jobs:** Left untouched per strict non-interference policy (`PARALLEL_PROCESS_AND_COMPUTE_REGISTRY.json`).

---

## 4. Highest-Leverage Next Architectural Directions (Sub-4B Frontier)

1. **Synthesize `M5_HSPA_LieSO3` + Selective Sparse Pointer (`H-55` candidate):** `M5` led pure-$O(T)$ semantic/textbook BPB & hierarchical rollups while `M9` led exact copy/code/math by `-0.63 BPB`. Replacing `M9`'s `FullSphere+GDN1` backbone (which still suffered `10.6 -> 5.3` key rank collapse) with an `M5` quaternion/Lie state + $O(T \log T)$ or landmark/bloom exact copy pointer is the clearest next mechanism experiment.
2. **Overwrite-Aware Recency/Delta Pointer:** In `Part A`, `M9` hit `0.962` on static open-vocab copy yet dropped to `0.542` on overwritten keys (`MQAR_Selective_Overwrite_4x`) because plain softmax blurs repeated past keys; hard/decayed recency suppression or key-invalidation inside the pointer is directly testable at `$0`.
3. **Prevent Depth-Wise Key Rank Collapse (`rank 13 -> 4/32`):** Orthogonality regularization, non-collapsing spectral constraints, or pure skew-symmetric `M7`/`M5` transport without SiLU orthant bias needs scaled ablation before any Track A `>100M` pretrain.
