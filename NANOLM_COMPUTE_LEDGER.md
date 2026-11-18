# NANOLM Ω — Verified Compute & Financial Ledger

**Audit Timestamp:** 2026-11-18  
**Autonomous Program Budget Policy:** `$180.00` autonomous envelope (`$200.00` hard cap, `$20.00` untouched owner reserve); Kaggle free quota (`30.00h/wk` GPU, `20.00h/wk` TPU) and local Apple M3 Max prioritized first.  
**Active Session Cloud Spend (`nanolm/architecture-invention-v1`):** **`$0.00`**

---

## 1. Reconciliation of Cross-Worktree Cloud Spend vs. `$95.14` Reported Balance

Inspection of sibling worktree ledgers and live CLI state reconciles every dollar cleanly:

| Source / Ledger Record | Scope | Reported Cost / Balance | Status / Notes |
|---|---|---:|---|
| `.worktrees/nanolm-lab/lab/reports/SPEND_LEDGER.md` | Phases 1–3 (`18` RunPod pods: `H100 NVL`, `RTX 4090`, `RTX 5090`, `L40S`, `A40`, `H200 SXM`) | **`$84.86`** realized (`$180.00 - $84.86 = $95.14` remaining) | Explains exact `$95.14` figure quoted in directive; all 18 pods terminated. |
| `.worktrees/nanolm-lab/lab/reports/SPEND_LEDGER.md` | Phase 4 (`1` pod `mqmxyjlqeevye8` `H200 SXM`, `NanoLM-B6-3B-Fixed` + `System-v3`) | **`+$7.25`** (`$92.11` total cumulative / `$87.89` remaining in envelope) | Pod terminated cleanly. |
| `.worktrees/nanolm/docs/nanolm/EMPIRICAL_V2_H100_AND_ENHANCEMENTS.json` | Separate worktree pod `eiebuwcuv8j0dj` (`NVIDIA H100 80GB`, `4.39/hr`, `11.39m`) | Est. `$0.8335` / observed account delta `$1.8616` -> balance `$81.3463` | Independent sibling worktree run. |
| `.worktrees/nanolm-posttrain/.../SPEND_AND_PROVENANCE.json` | Separate worktree pod `58b79zdtdqcr9n` (`NVIDIA A40`, `Qwen3-1.7B`, `2,257s`) | Pod est. `$0.3699` / account balance **`$74.8589`** (delta `$5.0923` during window) | Account balance shared across concurrent peer pods in workspace. |
| `runpodctl get pod` (live audit, `2026-11-18`) | Live active peer pod `y05unbw7rthg69` (`nanolm-scale02-v3-125m`, `1x L40S`, `RUNNING`) | Actively accruing billed RunPod balance (owned by sibling `nanolm-lab` session) | **Strictly left untouched per isolation guardrails; zero RunPod pods launched or modified here.** |

---

## 2. Current Branch Compute & Quota Accounting (`$0.00` Realized Paid Cost)

Because an active sibling RunPod pod (`y05unbw7rthg69`) already shares the RunPod account, **all experiments on branch `nanolm/architecture-invention-v1` execute exclusively on zero-cost local Apple Silicon and free Kaggle GPU quota**:

| Experiment Campaign | Backend Host | Hardware Spec | Compute Volume | Paid Cloud Cost |
|---|---|---|---|---:|
| `EXP-INV-01` (Cycles 1–3), `EXP-INV-02`, `EXP-INV-03`, `EXP-INV-04`, `EXP-INV-05A`, `EXP-INV-05B` | Local Apple Silicon | Apple M3 Max (`16` cores, `64 GB` UMA, `torch 2.11.0`, `mps`) | `115k` 3-seed mechanistic sweeps + `8.65M–47.64M` multi-domain corpus runs + `3.89B` live `bfloat16` profiling | **`$0.00`** |
| `EXP-INV-06A` & `EXP-INV-06B` (`hassaneljesr/nanolm-discovery-cuda-v1`) | Kaggle Free GPU | `2x NVIDIA Tesla T4` (`CUDA 13.0`, `14.56 GiB`/GPU) | `2,280.7s` (`0.63h` of `30.00h/wk` free quota): `4` × `20.55M` × `1,000` steps × `2` seeds (`32.77M` tokens) + `222M` `512–32k` KV wall | **`$0.00`** |
| `OMEGA-EXP-01-MECH`, `OMEGA-EXP-02` (`HardCausalSWEEnv`), `OMEGA-EXP-03` (`OMEGA-BENCH-01`) | Local Apple Silicon | Apple M3 Max (`64 GB` UMA) | `3,894.8s` initial full run + `326.0s` multiplicative router recalibration (`4` `115k` 3-seed mech arms + `g4_uncalibrated_router` control, `7` cognitive arms x `3` seeds, `60`-task unified benchmark; `2` local `.pt` checkpoints) | **`$0.00`** |
| `OMEGA-EXP-01` (`hassaneljesr/nanolm-omega-cuda-v1`, v4+v5) | Kaggle Free GPU | `2x NVIDIA Tesla T4` (`CUDA 13.0`, `float32`) | Multi-process Dual-GPU `4` x `20.55M` x `1,000` steps x `2` seeds (`32.77M` tokens; `skipped_grad_steps=0` on v5; `g4_omega_s42.pt` `82.3 MB` checkpoint preserved) | **`$0.00`** |

**Total Paid Spend by This Branch (`nanolm/architecture-invention-v1`):** **`$0.00`** (`0.0%` of RunPod budget touched).
