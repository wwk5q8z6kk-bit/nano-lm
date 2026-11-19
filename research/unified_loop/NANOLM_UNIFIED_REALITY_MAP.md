# Nano LM — Verified Unified Reality Map & Core-and-Web Frontier State

**Worktree:** `/Users/mac/Projects/nano-lm/.worktrees/nanolm-unified-loop` (`research/unified-discovery-loop-v1`, base `f82d82a`)  
**Updated:** `2026-11-19` · **Paid Compute Added in Loop:** `$0.00` (Local Apple Silicon CPU + MPS only)  
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
    1. Sparse Overwrite-Aware Causal Pointer (`M11`/`M12`, O(T·K), solves Unseen Copy + Overwrite & 4× Rare MQAR)
    2. O(log T) Spinor Recurrence (`M5`/`M10`/`M12` `HSPA_LieSO3`, leads Hierarchy & pairs cleanly with Pointers on BPB)
    3. Exact Unit/Denomination + Dynamic Hierarchy Algebra (`nano/units.py`, `nano/schema.py`, `nano/grounded_engine.py`)
    4. Inference-Time Monotonic Margin Guard / Abstention (`M8` Guard, `E24` clean entropy-margin routing)
```

- **`nanoscribe/` Posture:** Refactored into a clean declarative schema plugin (`CLINICAL_ENCOUNTER_SCHEMA` over `nano/`, `e7f4933`), verified at `235/276` end-to-end grounded retention with `0/7` hazards (`0` `N-03`), and **quarantined as a passive legacy domain regression suite** — zero frontier architecture decisions depend on it.

---

## 2. Verified Outcomes from Autonomous Loop (`EXP-U01` through `EXP-U06`, `$0.00` Paid Compute)

| Exp ID | Commit(s) | Scope & Artifact | Verified Headline Results (`verified local result`) |
|---|---|---|---|
| **EXP-U01** | `1dac170`, `01bdda7`, `adcfdbb` | Forensic autopsy of `X4_46m`, `S0_125m`, `S1_125m` (`600M`-tok checkpoints) — `U01_real_125m_autopsy/results.json` | **`S1_125m` Delta key collapse diagnosed:** Past Layer 0, Delta keys collapsed to `eff_rank = 2.9–11.3 / 64` (`val_code`) and `2.06–5.92 / 64` (`copy_probe`), `offdiag_cos = 0.503–0.888`, `decay_a = 0.036–0.366`. At `L=2048` (`2×`), text `web` BPB improved (`1.0821 -> 1.0068`), while `copy_1024` collapsed **`28×` deeper** in `S1` (`0.0021`, loss `11.6306`) than `S0` (`0.0582`, loss `9.628`). Zero-shot `val_web` MQAR `T256_N1`: `S1 = 0.0833/0.1667` vs `S0 = 0.2500/0.2500` (`T1024_N8`: `0.00–0.08`). |
| **EXP-U02 (Part A)** | `a4179f3`, `5a178ae` | Iso-param micro-lab (`55.3K–56.4K`), `10` mechanisms × `5` tasks × `3` seeds (`426.7s`) — `results_part_a.json` | 1) **Only `M9_DualChannel_Pointer` solved `Unseen_OpenVocab_Copy` (`0.9619 ± 0.0244` vs `0.0000` for `M0–M4, M6–M8`) & led `MQAR_4x` (`0.5764 ± 0.0794`, `ov=0.5421` vs `0.058–0.093`).** 2) **Geometric state doubled Attention on `Hierarchy_Denom_Rollup_4x` (`depth 4..8`):** `M5_HSPA_LieSO3 = 0.2588 (close=0.1633)`, `M7_SkewCayley_Flow = 0.2560 (close=0.1534)` vs `M0 = 0.1459 (close=0.1140)`. 3) `Parity_Dyck_4x`: `M7` led (`0.9279`). 4) `S4_Holonomy_8x`: `M5` led ID (`0.1029`), all ~chance (`~0.042`) at `8×` OOD at 200 steps. |
| **EXP-U02 (Part B)** | `5a178ae` | Real corpus transfer (`data/v1`, `d=128`, `~853K` non-emb / `~5.05M` total, `1,000` steps × `2` seeds, `819.88s`) — `results_part_b.json`, `analysis.md` | 1) **`M9_DualChannel_Pointer` crushed all non-pointer baselines across `web`, `math`, and `code`:** `web@256 = 1.9960` (`vs M0 2.0611`, `M1 2.1298`), `math@256 = 2.3172` (`vs M0 2.7016`), **`code@256 = 2.1983` (`-0.626 BPB` vs `M0 2.8245`, `-0.903 BPB` vs `M1 3.1013`)**; zero-shot `copy_32/64/128 = 0.910/0.933/0.926` vs `0.000` for `M0, M1, M5, M7`; MQAR `T128_N4 common = 0.3125` (`0.0000` elsewhere). 2) **Pure $O(T)$ `M5_HSPA_LieSO3` beat RoPE Attention (`M0`) & `GDN1` (`M1`):** best `textbook@128 = 1.9336` (`vs M0 1.9663`, `M1 2.0457`) and `#2` `web@256 = 2.0324`. 3) Legacy `M1_GDN1` finished last on all 4 domains (`code@128 = 3.1180`) and repeated Layer `0->3` key rank collapse (`12.82 -> 4.45 / 32`, `cos = 0.6225`). |
| **EXP-U03** | `f7ccb71`, `adcfdbb` | Scaffold attribution (`E23 C7-Omega`, `3` seeds) + routing paradox (`E24 HEM-ATRD`, `5` seeds) — `U03_scaffold_routing_audit/results.json` | 1) `C7-Omega`: Python `SymbolicBeliefState` binary features gave `0.000` gain; `A4_frontier_sym0` (`sym_feats=OFF`, no shield) hit `1.0000 ± 0.0000` Deep-OOD (`0.9967` Stress, `0.0033` Unsafe) via frontier-relative De Bruijn gauge (`\0`), matched by a `0`-param deterministic rule (`1.0000`). 2) `E24`: `H1_x5` (`1.0000 ± 0.0000`) beat `H4_hem` (`0.7469 ± 0.3621`) and `H3_mm` (`0.6922 ± 0.1851`) due to `u_sel * dx_total` grad scaling bug (`margin_shield_refiner.py:173`), Pass-2 starvation/variance, `min_ambiguity_floor=0.46`, and Lyapunov shield rejecting `-17.2 pp` valid Pass-2 fixes in `H3`. |
| **EXP-U04** | `5c1be63`, `09f688c`, `adcfdbb`, `e7f4933` | Grounded verifier diagnosis + Qwen2.5-1.5B (`C12–C14`) — `U04_nanoscribe_verifier_qwen/results.json` | Fixed determiner & false-conflict drops (`200/276 -> 228/276` Gold, `242/276` menu, `235/276` e2e, `72/72` useful notes, `0` `N-03`, `0/7` hazards). Live `Qwen2.5-1.5B-Instruct` on `C14` (`166.14s` MPS, prefix cache): `33/80 (41.2%)` asserted (`65/120` all, `0` custody violations) vs `0`-param deterministic menu rule `63/80 (78.8%)` (`91/120` all). |
| **EXP-U05** | `be9cd07`, `e7f4933`, `283b832`, `01e42c6` | Domain-pure modular foundation (`nano/units.py`, `nano/schema.py`, `nano/grounded_engine.py`) across 4 domains — `U05_cross_domain_hierarchy_denom/results.json` | **`zero_domain_branches_verified = True`** in `nano/`. Identical engine across Finance (`24`), SRE (`24`), and Legal (`24`): **`Full_Modular_Nano = 72/72 (100.0%)`** (`35/35` equiv/rollup, `24/24` hazard/fail-closed) vs `DenomOnly_FlatTopics = 66/72 (91.7%)` and `FlatString_NoDenom_NoHierarchy = 17/72 (23.6%)`, plus legacy Clinical `C12–C14` (`235/276` e2e, `0/7` hazards). |
| **EXP-U06 (Parts A & B)** | `fc60e0e` | `2×2` Spinor–Pointer Synthesis (`M10`–`M12` + `M0, M5, M9`) + $O(\log T)$ Kogge–Stone `HSPA` scan + $O(T \cdot K)$ Sparse OW Pointer (`329.52s` Part A + `1,960.85s` MPS Part B `1×/2×/4×`) — `U06_spinor_pointer_synthesis/analysis.md` | 1) **`M10_HSPA_DensePointer` swept `#1` BPB across all 4 domains & `1×/2×/4×`** (`web@512=1.8659`, `textbook@512=1.8094`, `math@512=2.1175`, **`code@512=1.5959` vs `M0 2.6483`**). 2) **Sparse OW Pointer (`K=32`) solved overwrite & `4×` real MQAR:** `MQAR_Overwrite_4x` `ov`: `M11=0.9266`, `M12=0.9198` (`vs M9 0.5421`); `T512_N4` MQAR `common/rare`: **`M11=0.6250/0.6459`**, **`M12=0.5625/0.4375`** (`vs M9 0.0416/0.0000`). `M12` also kept `#1` `Hierarchy_Denom_Rollup_4x` (`0.2624`), `1.56×` MPS speedup over `M10`, and zero NaNs. 3) **Diagnosed limits:** `HSPA` pointers trailed `FullSphere` on IID `copy_256` (`M12=0.6416`, `M10=0.1808` vs `M9=0.8899`, `M11 s1=0.9520`); `M11 s=0` hit `NaN` on `code@512` & `copy_256` loss (`s=1` finite). |

---

## 3. Historical Context & External Branch Status (Unmodified)

- **Early Own-Stack Checkpoints (`3.15M`, `10.0M`, `159.3M`), `H6` (`3.29M`, `REJECTED`), `H7` (`INVALID_GOLD`):** Preserved frozen on `origin/master`.
- **Track B (`B6–B8-3B`):** No local weights on disk; earlier `8-Pillar` (`85.61%` vs `Comp-1 12%` with `think=1`) depended on `System-v4` (`69%` `SmolLM3-3B Base + System-v4` > `66%` `B8-3B`).
- **Active Peer Cloud/Local Jobs:** Left untouched per strict non-interference policy (`PARALLEL_PROCESS_AND_COMPUTE_REGISTRY.json`).

---

## 4. Highest-Leverage Next Architectural Directions (Sub-4B Frontier)

1. **Close `M12` IID Random-Copy & Rare-MQAR Gap (`EXP-U07`):** Test `RMSNorm` / `L2`-normalized cosine pointer queries & keys (`tau * cos(q, k)`) plus an unrotated shallow lexical-context bypass into `ptr_q`/`ptr_k` on `M12_HSPA_SparseOWPointer`, aiming to lift `copy_256` (`0.6416 -> 0.95+`) and `T512` rare MQAR (`0.4375 -> 0.65+`) while preserving `M12`'s `#1` `Hierarchy_Denom_Rollup_4x` (`0.2624`), `#1–#2` natural BPB, `0.920` overwrite MQAR, `1.56×` MPS speedup, and zero `4×` `NaN`s.
2. **Stabilize `M11` (`FullSphere + SparseOW`) at `4×` (`L=512`):** Test whether `RMSNorm` / cosine pointer logits and spectral capping eliminate `M11 seed=0`'s `code@512` & `copy_256` `NaN` failure without degrading `M11`'s `0.952–0.984` copy and `0.625/0.646` `T512` MQAR.
3. **Prevent Depth-Wise Key Rank Collapse (`rank 13 -> 4/32`) & Scale Screening:** Combine anti-collapse orthogonality regularization with `M12`/`M13` before scaling to `30M–100M` on `data/v1`.
