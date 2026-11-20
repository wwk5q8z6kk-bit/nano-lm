# NANOLM Ω — Master Research Plan & Architectural Program

**Branch:** `nanolm/architecture-invention-v1` (isolated worktree) · **Active Spend on Branch:** **`$0.00`** · **Hard Parameter Ceiling:** `< 4,000,000,000` deployed parameters (`flagship_4b = 3.881B–3.890B`) · **Canonical Authority Anchor:** [`docs/PROJECT_AUTHORITY.md`](docs/PROJECT_AUTHORITY.md), [`docs/ACTIVE_NOW.json`](docs/ACTIVE_NOW.json) (`P1_SCRIBING` / `NanoScribe`, `H6` terminal, untouched).

---

## 1. Strategic Mandate & Epistemic Rules

`NANOLM Ω` transitions the program from incremental Transformer tuning (`NanoLM-v2/v3` `X4/X8`) to **first-principles architecture, cognitive planning, and symbolic-neural unification**, governed by four hard scientific constraints:

1. **Strict Parameter & FLOP Honesty:** Every architecture competes at `±0.5%` deployed parameter parity (`~115k` mechanistic, `~20.55M` multi-seed corpus, `~3.88B–3.89B` flagship `< 4B`) with explicit accounting of multiple vocabulary projections (`X8` runs `2` full $|V| \times d$ GEMMs per token) and persistent `32k`-context memory footprints (`1.80 MB` vs `50.54 MB` at `20.5M`; `83.71 MiB` vs `1,441.4 MiB` at `3.88B`).
2. **Zero Promotional Filtering (`LineageRegistry` & Multi-Horizon Discipline):** Short-horizon (`250`-step) overfitting cannot promote a candidate; architectures must pass mechanistic OOD (`4x` sequence length on $S_3$, noisy Overwrite, 8-Needle retrieval), epistemic AUROC, and `>= 1,000`-step multi-domain corpus crossover (`web`, `textbook`, `math`, `code`) + `2x` OOD context extrapolation (`T = 512`).
3. **Closed-Entity & Heuristic Elimination:** Cognitive planners must operate over arbitrary unseen entity cardinalities (`N = 4..24+`) without fixed `M0..M11` vocabulary slots and without hardcoded action-index shields (`suff_pen[4]`, `suff_pen[6]`).
4. **Compute & Evidence Discipline:** `$0.00` cloud spend on this branch by running exclusively on local Apple M3 Max (`64 GB` UMA) and Kaggle Free GPU (`2x NVIDIA Tesla T4`), leaving concurrent sibling RunPod pod `y05unbw7rthg69` and canonical evidence paths completely untouched, with cryptographic `SHA256SUMS` over every empirical JSON and preserved `.pt` checkpoint.

---

## 2. Program Phases Executed (`Cycles 1–3` -> `Cycle Ω`)

| Phase | Scope & Artifacts | Key Scientific Decision | Status |
|---|---|---|---|
| **Baseline Audit (`Tasks A–D`)** | Inspected canonical `origin/master`, `.worktrees/nanolm-lab`, `.worktrees/nanolm`, `.worktrees/nanolm-posttrain`, and `Express1M2` logs | Reconciled `$84.86` / `$92.11` / `$74.86` RunPod ledgers; isolated `X8` `2x` unembed slowdown (`1.81x–2.65x`), `C4b/C5` `Sym(12)` ceiling, and `NanoLMSystem` `task_hint`/keyword oracle. | **COMPLETE** |
| **Cycles 1–3 (`nanolm_discovery/`)** | `EXP-INV-01..06` (`discovery_results.json`, `kaggle_cuda_results.json`, `lineage_registry.json`) | Proved $\beta \in (0, 2)$ Householder WY + `CausalKeyShift` + `SCAM-32` + `Depth-Attention` (`G3_HMWR`: `OOD_Comp = 0.739`, crossed `I3_HRWN` by `+0.0312 BPB` & beat `B0` at `2x` OOD `2.1322` vs `2.1526`). | **COMPLETE** |
| **Omega Backbone (`OMEGA-EXP-01`)** | `nanolm_omega/backbone.py`, M3 Max `OMEGA-EXP-01-MECH`, Kaggle `2x T4` `1,000`-step tournament | **Single-pass `g4_no_sparse_refiner`** (`G3 + QK-RMSNorm + Gated-VRes`) **beat `B0_Transformer` on ID (`2.1112` vs `2.1188`) and `2x` OOD (`2.1079` vs `2.1526`)**, beat `g4_omega` (`2.1175`), crushed `g4_no_vres` (`2.1508`), and cut `32k` KV state `28.0x` vs `X8` (`1.80 MB` vs `50.54 MB`). Falsified iso-param Pass-2 `LatentTokenRefiner` (`H-Ω04`). | **COMPLETE** |
| **Omega Cognition (`OMEGA-EXP-02`)** | `nanolm_omega/cognition_efe.py` (`HardCausalSWEEnv`, `N=4..24` Super-OOD) | `omega_efe_planner` (`248,328` params) achieved **`84.72%` ID, `95.56%` OOD (`9–12`), `83.33%` Super-OOD (`16–24`), `0.00%` premature rate** vs `C4b` (`5.28%` Super-OOD) & `C5` (`4.17%–7.78%`). | **COMPLETE** |
| **Omega Unified System (`OMEGA-EXP-03`)** | `nanolm_omega/unified_system.py` + `nano/` contracts (`OMEGA-BENCH-01`, `60` tasks × `6` pillars) | `omega_unified_system` scored **`100.0%` (`60/60`)**, **`100.0%` abstention/conflict**, and **`2.225` cost/win** vs `legacy_keyword_system_v2` (`28.33%`, `4.588` cost/win) and `omega_no_slw_provenance` (`78.33%`). | **COMPLETE** |

---

## 3. Gated Near-Term Execution Plan (`Cycles 4A–6`)

1. **Cycle 4A (Free Kaggle + M3 Max — Runtime & Memory Wall):** Implement `1,024`-token streaming chunked prefill (cutting `222M` `32k` peak VRAM `7,876.7 MiB -> < 1.1 GiB`) and fused Triton/Metal `WY + Wave + SCAM` kernels targeting `>= 80%` dense `B0` training throughput while keeping `< 1e-4` step duality.
2. **Cycle 4B (Free Kaggle Quota — Chinchilla `41M`-Token Horizon Law):** Test promoted **Single-Pass `G4` (`g4_no_sparse_refiner`)** against `B0_Transformer` and `X8` across `10,000` steps (`AdamW` vs Pure-CE `Muon`) evaluating length extrapolation out to `4,096` tokens (`16x` train context) where `O(T)` GQA KV layers saturate.
3. **Cycle 5 (`NanoScribe P1` Bridge):** Wire Single-Pass `G4` structural uncertainty $u_t^{\text{mem}} = s_t / \sqrt{\|y^{\text{met}}\|_{\text{rms}}^2 + \|y^{\text{scam}}\|_{\text{rms}}^2 + 10^{-3}}$ (`EpistAUROC=0.6185`, `Entropy AUROC=0.7409–0.7852`) and `SCAM-32` registers into synthetic P1 scribing extraction (`absence`, `conflict`, `uncertainty` gates — zero clinical claims without human dual-IAA).
4. **Cycle 6 (`3.884B` Flagship Authorization Gate):** Only after Cycle 4A/4B confirm Chinchilla scaling parity and kernel efficiency, request explicit owner budget sign-off inside `$180` RunPod envelope for flagship `3.884B` pretraining.
