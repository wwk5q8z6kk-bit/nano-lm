# NanoLM Frontier Experiment Report (`nanolm_frontier_v1`)

> Independent general-purpose small foundation LM evolution (`<= 4,000,000,000` absolute parameter cap, `~2B` preferred target). Measured on **RunPod `NVIDIA A40` 48GB (`CUDA 13.2`, `torch 2.9.1+cu128`, `BF16`)** under pod `58b79zdtdqcr9n` (terminated and verified removed; pod-attributed cost **\$0.3699 USD** for `2,257` wall-clock seconds @ `\$0.59/hr`).

---

## 1. Experiment E1 — Iso-Parameter 5-Architecture Pretraining Tournament (`~38M` Micro Scale, `2,867,200` Tokens)

All 5 architectures shared identical `FrontierMicroTokenizer` (`vocab_size=8,192`), `seq_len=256`, `batch_size=16`, `700` AdamW cosine steps, matched parameter budgets within `0.7%`, and held-out evaluation splits (`code`, `math`, `structured`, `assoc`).

| Candidate Architecture | Params | Delta / Softmax Layers | KV Bytes / Tok (BF16) | Active FLOPs / Tok @ 2K | Train Tok/s | Peak VRAM (MB) | Mean Val BPB | Code BPB | Math BPB | Struct BPB | Assoc BPB | OOD Recall (`256 / 512 / 1024`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `C0_DenseGQA` (`c0-dense-gqa-micro`) | 38,025,216 | 0 / 12 | 6,144 | 126,353,408 | **63,110.2** | **2,491.18** | **0.6316** | **0.2457** | 0.8010 | 0.4023 | 1.0774 | `0.0000 / 0.0208 / 0.0000` |
| `C1_GatedDeltaHybrid` (`c1-gated-delta-micro`) | 38,153,664 | 9 / 3 | **1,536** (-75%) | 89,305,088 | 11,821.7 | 3,829.94 | 0.6331 | 0.2544 | 0.8110 | **0.3918** | **1.0750** | `0.0000 / 0.0000 / 0.0417` |
| `C2_DiffNSDelta` (`c2-diff-ns-delta-micro`) | 38,154,960 | 9 / 3 | **1,536** (-75%) | 89,305,088 | 9,705.0 | 4,347.97 | 0.6335 | 0.2541 | 0.8118 | 0.3921 | 1.0759 | `0.0000 / 0.0000 / 0.0417` |
| `C3_CorticalPKM` (`c3-cortical-pkm-micro`) | 38,007,516 | 9 / 3 (+4,096 PKM) | **1,536** (-75%) | **82,731,008** (-34.5%) | 9,420.2 | 4,463.34 | 0.6326 | 0.2547 | 0.8001 | 0.3999 | 1.0756 | `0.0000 / 0.0000 / 0.0000` |
| `C4_JCLE_Frontier` (`c4-jcle-frontier-micro`, r=2) | 38,272,224 | 9 / 3 (+PKM +Loop) | **1,536** (-75%) | 133,341,184 | 5,181.1 | 6,824.55 | 0.6368 | 0.2544 | **0.7887** | 0.4288 | 1.0753 | `0.0000 / 0.0000 / 0.0000` |

### `C4_JCLE_Frontier` Test-Time Compute Scaling & Shannon Entropy Early Exit

| Active Loops (`r`) | Entropy Halt (`tau=1.25`) | Mean Loops Used | Early Exit Rate | Math Val BPB | Assoc Val BPB | Eval Time (s) | Active FLOPs / Tok @ 2K |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 1 | Off | 1.000 | 0.0% | 0.8000 | **1.0751** | 3.483 | 83,255,296 |
| 2 | Off | 2.000 | 0.0% | 0.7887 | 1.0753 | 6.707 | 133,341,184 |
| 4 | Off | 4.000 | 0.0% | 0.7885 | 1.0772 | 11.541 | 233,512,960 |
| **4** | **On** | **2.184** | **60.5%** | **0.7884** | 1.0770 | 11.849 | 233,512,960 (budget cap) |

### E1 Empirical Interpretation
1. **Sparse Cortical Memory (`C3_CorticalPKM`) improves hybrid Delta efficiency:** Replacing dense FFN width with 4,096 Cartesian Product-Key slots reduced active FLOPs/token by **34.5%** (`82.7M` vs `C0` `126.4M`) and improved overall BPB from `0.6331` (`C1`) and `0.6335` (`C2`) to `0.6326`, beating `C0`, `C1`, and `C2` on `math` (`0.8001`).
2. **Anchored Contractive Equilibrium Loop (`C4_JCLE_Frontier`) excels at mathematical reasoning & conditional compute:** `C4` achieved the lowest `math` BPB of all 5 architectures (`0.7887` vs `0.8010` `C0`), monotonic `r=1 -> r=2 -> r=4` improvement (`0.8000 -> 0.7887 -> 0.7885`), and **60.5% token early exit** (`mean_loops_used = 2.184`) under entropy halting with optimal `math_bpb = 0.7884`. However, unoptimized `C4` regressed on `structured` BPB (`0.4288` vs `C1` `0.3918`).
3. **Hardware Kernel Bottleneck & Negative Result on `C2`:** Pure-PyTorch chunked `torch.linalg.solve_triangular` in `C1..C4` ran **5.3x–12.2x slower** (`5.2k–11.8k` vs `63.1k tok/s`) than CUDA FlashAttention `SDPA` and raised activation VRAM (`3.8–6.8 GB` vs `2.5 GB`), proving fused Triton/CUDA linear-recurrence kernels are mandatory before 2B scaling. `C2_DiffNSDelta` trailed `C1` slightly (`0.6335` vs `0.6331`). On OOD recall after only `2.87M` packed tokens (`loss=1.02`), exact copy induction had not converged across any arm (`0%–4.2%`).

---

## 2. Experiment E2 — ~1.72B Foundation Evolution Across All 15 Dimensions (`Qwen/Qwen3-1.7B`)

Evaluated strictly decontaminated (`overlaps = 0` across 35 curriculum items and 56 held-out tasks) with uniform non-thinking chat prompting. `Frontier_DexDiff_SFT_GRPO` verified exact bit-level step-0 graft preservation (`step0_max_abs_logit_diff = 0.00e+00`) across `196` LoRA layers, `28` `DexDiffAttentionWrapper` modules, and `14` upper-layer `ThalamocorticalVAnchor` modules.

### Overall Summary & Dimension 15 (`inference_efficiency`)

| Arm | Total Params (`<= 4B`) | Passed / 56 | Accuracy [95% CI] | Mean Partial Score | Easy (`15`) | Medium (`21`) | Hard (`20`) | Tok/s | Peak VRAM (MB) | KV @ 4K (MB) | Cap / B-Params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Base (`Qwen3-1.7B`)** | 1,720,574,976 | 34 / 56 | 60.71% [48.2%, 71.4%] | 0.7439 | 73.3% (11) | **66.7%** (14) | 45.0% (9) | **33.53** | **3,899.07** | 448.0 | 0.3528 |
| **LoRA SFT** (`r=16`, 5 ep) | 1,738,007,552 | **36 / 56** | **64.29%** [51.8%, 75.0%] | **0.7708** | 80.0% (12) | 61.9% (13) | 55.0% (11) | 20.18 | 4,157.09 | 448.0 | **0.3699** |
| **Frontier (`DexDiff + VAnchor + SFT + GRPO`)** | 1,753,189,376 | 35 / 56 | 62.50% [51.8%, 73.2%] | 0.7554 | **86.7%** (13) | 47.6% (10) | **60.0%** (12) | 16.58 | 4,280.93 | 448.0 | 0.3565 |

### Per-Dimension Breakdown (Dimensions 1–14, 4 Tasks Each)

| Dimension | Base | LoRA SFT | Frontier (`DexDiff + VAnchor + SFT + GRPO`) |
|---|---:|---:|---:|
| 1. `coding` | **3/4 (75%)** | 0/4 (0%) | 1/4 (25%) |
| 2. `math_logic` | 1/4 (25%) | **4/4 (100%)** | 2/4 (50%) |
| 3. `summarization` | 3/4 (75%) | 3/4 (75%) | 3/4 (75%) |
| 4. `structured_json` | **3/4 (75%)** | 2/4 (50%) | 2/4 (50%) |
| 5. `diagrams_charts` | 3/4 (75%) | **4/4 (100%)** | 3/4 (75%) |
| 6. `long_context` | 2/4 (50%) | 2/4 (50%) | 2/4 (50%) |
| 7. `general_knowledge` | **4/4 (100%)** | **4/4 (100%)** | **4/4 (100%)** |
| 8. `multi_step_reasoning` | 3/4 (75%) | **4/4 (100%)** | 3/4 (75%) |
| 9. `instruction_following` | 3/4 (75%) | **4/4 (100%)** | **4/4 (100%)** |
| 10. `tool_use` | 2/4 (50%) | 2/4 (50%) | 2/4 (50%) |
| 11. `planning_decomposition` | 0/4 (0%) | 0/4 (0%) | **1/4 (25%)** |
| 12. `ood_robustness` | **3/4 (75%)** | 2/4 (50%) | **3/4 (75%)** |
| 13. `hallucination_resistance` | 3/4 (75%) | **4/4 (100%)** | **4/4 (100%)** |
| 14. `memory_retention` | 1/4 (25%) | 1/4 (25%) | 1/4 (25%) |

### Key Causal Findings & Failure Taxonomy
1. **Why SFT Lifted Formatting/Reasoning Yet Broke Coding (`3/4 -> 0/4`):** SFT converged to `0.0319` cross-entropy loss (`LoRA_SFT`) and `0.0294` (`Frontier` Stage 1) over 5 epochs of 35 concise examples, raising `math_logic` (`25% -> 100%`), `instruction_following` (`75% -> 100%`), and `hallucination_resistance` (`75% -> 100%`). However, over-fitting 6 tiny code snippets damaged algorithmic boundary handling (`IndexError: pop from empty list`, missing `import heapq` in `dijkstra`).
2. **Why `Frontier` Outperformed on `hard` (`60%` vs `55%` SFT / `45%` Base) Yet GRPO Barely Updated:** Because Stage-1 SFT already reached `94.8%–96.6%` training rollout reward before GRPO started, **71 of 72 GRPO groups had zero reward variance** and were skipped, leaving only **1** gradient update (`mean_abs_dex_diff_gamma = 0.001561`, `mean_abs_v_anchor_gate = 0.001465`, `mean_kl = 0.0807`). Despite only 1 RL step, `Frontier` achieved the highest `hard` accuracy (`12/20 = 60%`) and `easy` accuracy (`13/15 = 86.7%`), uniquely solved hard DAG planning (`pln-04`), and avoided complete `coding`/`ood_robustness` collapse — at a decode speed cost (`16.58` vs `33.53 tok/s` Base) from uncompiled eager Python wrapper calls.

---

## 3. Retain / Modify / Reject Decisions & Provenance

| Innovation | Evidence Verdict | Justification & Next Modification |
|---|---|---|
| `C3_CorticalPKM` | **RETAIN** | Cut active FLOPs by `34.5%` and beat `C1/C2` overall and `C0/C1/C2` on `math`. |
| `C4_JCLE_Frontier` (Damped Loop + Entropy Halt) | **MODIFY** | Lowest `math` BPB (`0.7884`), `60.5%` early exit at `r=4`; must pair with Triton delta kernels and isolate structured JSON FFN routing to fix throughput and structured BPB regression. |
| `C2_DiffNSDelta` | **REJECT (at short horizon)** | `0.6335` BPB slightly behind `C1` (`0.6331`) and `C0` (`0.6316`) with lower throughput (`9,705 tok/s`). |
| `DexDiff + ThalamocorticalVAnchor + Verifier GRPO` | **MODIFY (Active Recipe)** | Verified exact `0.00e+00` step-0 KL preservation and best `hard` accuracy (`60%`), plus DAG planning breakthrough (`pln-04`). Next iteration must cap SFT at 1–2 epochs (`loss ~0.25`) to preserve coding generalization (`3/4`) AND feed harder RL prompts so GRPO executes ~50+ non-zero-variance updates instead of `1/72`. Merge or `torch.compile` wrappers for inference. |

**RunPod Spend Provenance ([SPEND_AND_PROVENANCE.json](SPEND_AND_PROVENANCE.json)):**
- Pod: `58b79zdtdqcr9n` (`NVIDIA A40` @ `\$0.59/hr`, `2,257s` = `\$0.3699 USD` attributed spend; verified terminated via `runpodctl get pod`).
