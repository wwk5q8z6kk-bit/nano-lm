# NANOLM Ω — Repository-Wide Baseline & Forensic Verification Report (`TASK A–C`)

> **Scope:** Independent first-principles audit conducted inside the active locked worktree on branch `nanolm/holonomic-discovery-v1` (PR `#174`), cross-checking canonical `origin/master` (`acc9b7e`) and sibling research branches (`nanolm/discovery-lab-v1`, `nanolm/arch-evolution-v3`, `nanolm/autonomous-discovery-v1`, `nanolm/architecture-invention-v1`, `nanolm/posttrain-v1`) on 2026-11-18. All statuses distinguish **Demonstrated** (re-executed / SHA-verified artifacts), **Existing** (code on branch), **Failed / Falsified** (negative evidence preserved), **Obsolete / Relabeled**, and **Proposed**.

---

## 1. Repository & Multi-Branch Lineage Map (`TASK A`)

| Worktree / Branch | Latest Commit / Role | Parameter Ceiling (`<= 4B`) | Primary Packages / Contribution |
|---|---|---:|---|
| Canonical `origin/master` (`acc9b7e`) | Foundation integration (`PR #118`) | N/A | `nano/` (symbolic substrate), `nano_ai/` (clinical H6 + H7 freeze), `benchmarks/`, `fabric/`, `trajectory/` |
| `nanolm/posttrain-v1` (`813cfab`, PR `#171`) | `nanolm_frontier_v1` (`$0.3699` A40) | `3,845M` / `1.753B` | `nanolm_frontier/`, `nanolm_posttrain/`, 15-dim objective suite (`56` tasks) |
| **`nanolm/holonomic-discovery-v1` (Active, PR `#174`)** | **HSPA Cycles 1–2 + `NANOLM Ω`** | **`3.845B` (`<= 4B`) & `1.747B`** | `nanolm_holonomy/` (`HSPA`, fused OpenAI Triton kernel, `ComputePlanner`, `HSPA-Ω`, `SpectralNullspaceLoRA`, `OmegaCognitiveController`) |
| `nanolm/discovery-lab-v1` (`0256d52`, read-only ref) | Phases 1–4 (`$92.11` 19 pods) | `1.72B` / `3.08B` | `lab/` (`X8_margin_atrd_muon`, `C4b`/`C5` De Bruijn cognition, `NanoLM-B6-2B/3B`, `NanoLMSystem-v3`) |
| `nanolm/arch-evolution-v3` (`65ba250`, read-only ref) | Gen-3 to Gen-5 physics & Muon (`60/60` tests) | `2.073B` / `3.929B` MoE | `nanolm/` (`Householder-Delta`, `Diff-GQA`, `Depth-AttnRes`, `Canon-SwiGLU`, `NorCautiousMuon`, M3 Max `4k` profile) |
| `nanolm/autonomous-discovery-v1` (`d7730ce`, read-only ref) | `ERN` Epistemic Renormalization (`2` cycles) | `3.874B` (`25.17M` runs) | `Block-WY` recursive inverse (`NaN` fix), `ERN-Gen3` (`99.94%` 4x-OOD parity, `58.59%` 64-slot recall) |
| `nanolm/architecture-invention-v1` (`b8af1c6`, read-only ref) | `G3_HMWR` Cycles 1–3 (`$0.00` M3 Max + Kaggle T4) | `3.89B` (`20.5M` 1k steps) | `nanolm_discovery/`, `4x` horizon crossover on Kaggle `2x Tesla T4` |

---

## 2. Verification of Prior Program & Phase 1–4 Claims (`TASK B & TASK C`)

| Historical / Phase Claim | Verified Evidence Source | Audit Verdict & Discrepancies Found |
|---|---|---|
| **Symbolic Substrate (`nano/`)** | `nano/` + `scripts/check_docs_integrity.py` | **Demonstrated / Intact:** Frozen evidence paths (`papers/`, `trajectory/`, `freeze/`) untouched; unit tests pass. |
| **Clinical `nano_ai/` `H6`** | `artifacts/nano_h6/H6_TERMINAL_DECISION.json` | **Failed (`REJECTED`, Terminal):** One-shot Kaggle eval (`5,000` decisions): overall `3,901/5,000 (78.02%)`, held value `76.23%` passed, **absence (`48.18%`), conflict (`57.20%`), uncertain (`76.00%`) failed**. Copying worked; epistemic state calibration failed. |
| **Follow-up `nano_ai/` `H7`** | `artifacts/nano_h7/H7_TERMINAL_DECISION.json` & `artifacts/nano_h7/screen-v1/manifest.json` | **Discrepancy Confirmed:** `H7_TERMINAL_DECISION.json` is absent in this branch checkout, while `screen-v1/manifest.json` confirms evaluation-only zero-overlap fixture generation (`1,000` records, `0` PHI); `nanolm-lab` synthesis confirms `H7` terminal verdict recorded `INVALID_GOLD_ANNOTATIONS`. |
| **Early `1.54B` / `1.98B` Claims** | Cross-worktree audit (`nanolm-lab`) | **Relabeled / Unpretrained:** `1.54B` wrapped unmodified `Qwen/Qwen2.5-1.5B-Instruct`; `1.98B` was never pretrained to frontier parity due to pretraining FLOP economics. |
| **Track A `X8_margin_atrd_muon` (`41` runs, `24.6B` tok)** | `nanolm-lab`: `lab/reports/LAB-ARCH-02_results.md`, `NANOLM_UNIFIED_RESEARCH_SYNTHESIS.md` | **Demonstrated (with scope caveat):** `0.9543 ± 0.0005` BPB (`-7.32%` vs `T0_dense/adamw`, `n=2` seeds, `600M` tok/seed, `-80%` KV). **Caveat:** Short-budget (`123k`-tok) hybrid mixer surgery (`LAB-INT-01`) failed to close a `+0.254` BPB gap (`1.2018` vs `0.9481`), so post-hoc mixer replacement is not lossless without sustained retraining. |
| **Cognitive `C4b` / `C5` (`LAB-COG-01..03`)** | `nanolm-lab`: `lab/reports/lab_cog_02_results.json` | **Demonstrated (domain-scoped):** `C5_de_bruijn_canonical` (`1.019M` params, `n=3` seeds) lifted Extreme-OOD (`10–12` mod) repo repair from `0.0%` (`C0–C2`) and `64.5%` (`C4b`) to `78.5% ± 13.3%`. **Inconsistency / Gap:** Separate small-state model (`1M` params), not integrated end-to-end inside a single differentiable `2B–4B` LM weight stack. |
| **Phase 1–4 `NanoLM-B6-2B/3B` + `System-v3`** | `nanolm-lab`: `SPEND_LEDGER.md`, `NANOLM_UNIFIED_RESEARCH_SYNTHESIS.md` | **Demonstrated (with variance caveat):** `NanoLM-B6-2B` (`1.72B`, `52.0%` `System-v3`) and `NanoLM-B6-3B-Fixed` (`3.08B`, `70.8% MATH-500`) lifted capabilities substantially. **Inconsistency Found:** Multi-turn Harmony reasoning budget in `System-v2` broke `gpt-oss-120b` tools (`75% -> 15%`), and hard-token SFT vs logit KD traded benchmark score (`70.8%` MATH) against multi-turn JSON/tool compliance (`15%` Conservation JSON on `B6-3B-Fixed` vs `95%` on GRPO-only `B6-3B`). |
| **`nanolm_frontier_v1` E1 & E2 (`$0.3699` A40)** | `artifacts/nanolm_frontier_v1/{e1_tournament,e2_foundation_15d}.json` | **Demonstrated (reproduced on disk):** E1 `C0..C4` (`~38M`) showed pure-PyTorch `solve_triangular` ran `5.3x–12.2x` slower than attention; E2 showed 5-epoch SFT collapsed `Qwen3-1.7B` Python `coding` (`3/4 -> 0/4`) and left `71/72` GRPO groups with zero reward variance. |
| **HSPA Cycle 1 & Cycle 2 (`$1.3026 + $0.7021` L40S)** | `artifacts/nanolm_holonomy_v1/*.json`, `artifacts/nanolm_holonomy_v2/*.json` | **Demonstrated (reproduced on disk):** Cycle-2 `C2-X0` Triton scan accelerated `fwd+bwd` **`11.08x–35.00x`**; `C2-X2` `Gen1_H2` (`0.3693` BPB) beat `B0` (`0.3798`); `C2-X3` `HSPA-4B` (`3.845B`) beat `Transformer-4B` at `32,768` tokens in both VRAM (`9,261.8` vs `13,888.0 MB`) and decode speed (`74.26` vs `75.03 ms/tok`); `C2-X4` hit `35/56` vs `Qwen3-8B` `39/56`. |

---

## 3. Baseline Ledger & Budget Reconciliation (`TASK B`)

- **Historical Pre-Phase-1:** Kaggle H6 verification (`$0.00`) + H6 RunPod dev (`$3.70` in `H6_TERMINAL_DECISION.json`).
- **Sibling Track `nanolm-lab` (`SPEND_LEDGER.md`):** `19` pods across Phases 1–4 = **`$92.11` USD** (`$39.07` P1 + `$34.89` P2 + `$10.90` P3 + `$7.25` P4). *Note:* The `$95.14` figure noted in prior prompts reflected the `$180.00 - $84.86` balance right after Phase 3 prior to Phase 4 (`$7.25`), plus concurrent multi-worktree shared RunPod usage.
- **This Research Track (`nanolm-frontier` + `nanolm-holonomy` Cycles 1–2):**
  - `nanolm_frontier_v1` (pod `58b79zdtdqcr9n`, A40): **`$0.3699`**
  - `nanolm_holonomy_v1` (pod `2v2dt0j536rh58`, L40S): **`$1.3026`**
  - `nanolm_holonomy_v2` (pod `vboeyg7n3dk7k3`, L40S): **`$0.7021`**
  - Subtotal prior to `NANOLM Ω`: **`$2.3746` USD** (all pods verified terminated; `0` pods orphaned).
- **Live Account State Verified Before `NANOLM Ω` Launch:** `runpodctl user` reported `clientBalance = $88.6893` (above `$20.00` owner reserve), with pre-flight Roofline selection verifying `EXP-Omega1..3` fit inside `48 GB` (`peak safe = 7.09 GB`) for `< $0.25` USD.
