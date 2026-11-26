# Empirical Results: Pre-Registered `40.38M` (`M`-Scale) Value-Residual Confirmation Gate & Hopper `R0-v2` Gated DeltaNet Benchmark

**Completed:** `2026-11-26` · **Branch:** `research/valueres-42m-confirm-and-r0` ([Draft PR #214](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/214))  
**Pre-registration commit (immutable before any run):** `e05d9c2cc54ff6335487dd31eb7ed2705f33bd03` (`PROTOCOL.md`, `valueres_42m_jax.py`, `score_42m_gate.py` untouched)  
**Compute & Teardown Verified:**
- **Part I (`40.38M` Confirmation Gate):** `1 × Cloud TPU v5litepod-4` (`4` TPU v5e chips, `us-east1-c`, spot, `~2.85 h` = `~$4.10` GCP credits), `12` full `799,539,200`-token runs = **`9,594,470,400` (`9.59B`) unique Corpus `v2` tokens**, `12 × 7` zero-shot benchmark suites, `36` causal knockout ablations; node **self-deleted automatically on completion** (`us-east1-c` TPU list = `[]`).
- **Part II (Hopper `R0-v2` Benchmark):** `1 × NVIDIA H100 80GB HBM3` (`RunPod`, `499.5s` wall time = **`$0.5592` actual balance delta**), **pod deleted and verified** (`pods_after = []`).

---

## 1. Executive Verdicts (Unaltered Mechanical Outputs)

| Gate | Mechanical Verdict | Decisive Empirical Finding |
|---|---|---|
| **`G1_42m_architecture_gate`** (`score_42m_gate.py` on `results/runs_42m/run_*.json`) | **`REJECT_VALUE_RES_PROMOTION_AT_42M`** | At Chinchilla horizon (`40.38M` params, `800M` unique tokens = `19.8` tok/param), `value_res`'s `+0.0166` `14M` lead shrinks by **`64.5%`** to **`+0.00590 ± 0.00085` mix BPB** ($t = 11.96$; equal-wall-clock `+0.00556`, $t = 11.23$), missing the `+0.0080` `G1` promotion threshold. Moreover, **`74.6%` (`+0.00440 / +0.00590`) of that gain is matched by a generic `+7`-scalar Layer-0 residual skip (`resid_x0_skip`)**, leaving a non-significant `+0.00150` BPB gap ($t = 2.56 < 4.303$). |
| **`G2_causal_v0_mechanism_gate`** (`score_42m_gate.py`) | **`CONFIRMED_LATE_POSITIONAL_V0_TRANSPORT`** | On trained `value_res` checkpoints across all 3 seeds, permuting $v_0$ across sequence positions (`perm_v0`) degrades held-out `mix` BPB by **`+0.9744` BPB**, and ablating late layers (`ablate_late_6_7`: **`+1.0168` BPB**) causes **`598×`** more damage than ablating middle layers (`ablate_mid_3_4`: `+0.0017` BPB). |
| **`F1_stage_c_gate`** (`r0_bench_hopper.py` -> `results/r0_hopper/r0_hopper_result.json`) | **`ADVANCE_STAGE_C_ON_HOPPER`** | Upgrading `triton==3.7.1` fixes `fla 0.6.0` Hopper bug `#640` (`FP32` output/grad rel err `0.12%–0.20%`; `BF16` `0.33%–0.60%`; `0/100` non-finite steps). `546.9M` Gated DeltaNet hits **`56,133` tok/s** (`184.2 TFLOPS` = **`18.6%` of H100 datasheet peak `> 15%` gate**, `84.8%` of FlashAttention speed at `8×2048`) and **`48,261` tok/s** with cached top-32 teacher KD (`~$15.50` community H100 per `1B` KD tokens). |

---

## 2. Part I — `40.38M` (`M`-Scale, `800M` Tokens, `3` Seeds) Tournament Results

### 2.1 Held-Out BPB (`655,360` Tokens) & Throughput (`seeds = 1000, 1001, 1002`)

| Arm | Exact Params | Equal-Token `mix` BPB (Mean ± SD) | Per-Seed `mix` (`1000, 1001, 1002`) | $\bar{\Delta}_{\text{tok}}$ vs `dense` ($t$-stat) | Equal-Time `mix` BPB ($\bar{\Delta}_{\text{time}}$, $t$-stat) | Mean Tok/s/Chip | `code` BPB ($\bar{\Delta}$) | `math` BPB ($\bar{\Delta}$) |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| `dense` (`PreNorm + Muon`) | `40,379,904` | `1.0163 ± 0.0006` | `1.0160, 1.0170, 1.0158` | `0.0000` (ref) | `1.0163` (ref) | `261,395` | `0.8282` | `1.1407` |
| `resformer_fixed` (`0.5/0.5` SVRA) | `40,379,904` | `1.0149 ± 0.0001` | `1.0150, 1.0149, 1.0148` | `+0.0014` ($t = 3.73$, NS) | `1.0156` (`+0.0006`, $t = 1.89$) | `257,085` | `0.8232` (`+0.0049`) | `1.1374` (`+0.0033`) |
| `resid_x0_skip` (`+7` $\alpha_\ell$ $h_0$ skip) | `40,379,911` | `1.0119 ± 0.0008` | `1.0119, 1.0126, 1.0111` | `+0.0044` ($t = 25.40$) | `1.0129` (`+0.0033`, $t = 18.92$) | `256,452` | `0.8207` (`+0.0074`) | `1.1336` (`+0.0071`) |
| **`value_res` (`+7` $\lambda_\ell$ $v_0$ skip)** | **`40,379,911`** | **`1.0104 ± 0.0003`** | **`1.0102, 1.0102, 1.0107`** | **`+0.0059` ($t = 11.96$)** | **`1.0107` (`+0.0056`, $t = 11.23$)** | **`259,412`** | **`0.8167` (`+0.0115`)** | **`1.1308` (`+0.0100`)** |

**Key paired contrasts (`df = 2`, two-sided $t_{0.95} = 4.303$):**
- `value_res` vs `resformer_fixed` (`ACL 2025` uniform `0.5/0.5` SVRA): **`+0.00453 ± 0.00038` BPB ($t = 20.74$, significant)** — learned depth scheduling substantially outperforms uniform `0.5/0.5` mixing (`resformer_fixed` hurts `textbook` by `-0.0012` BPB because middle layers `l = 3, 4` prefer `< 9%` $v_0$).
- `value_res` vs `resid_x0_skip` (`+7` scalar Layer-0 residual skip): **`+0.00150 ± 0.00101` BPB ($t = 2.560$, NOT significant; per-seed `+0.0017, +0.0024, +0.0004`)**. Nearly three-quarters (`74.6%`) of `value_res`'s Chinchilla-regime improvement is captured by a generic scalar skip from $h_0$ into block inputs.

### 2.2 Downstream 7-Task Zero-Shot Evaluation (`tasks-v1.json`, 3-Seed Mean)

| Arm | `ARC-Easy` (`acc_norm`) | `ARC-Challenge` (`acc_norm`) | `HellaSwag` (`acc_norm`) | `PIQA` (`acc_norm`) | `WinoGrande` (`acc`) | `OpenBookQA` (`acc_norm`) | `LAMBADA` (`acc` / `ppl`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dense` | `0.3977` | `0.2335` | `0.2823` | `0.5885` | **`0.5104`** | **`0.2807`** | `0.2331` / `173.65` |
| `resformer_fixed` | `0.4091` | **`0.2338`** | `0.2837` | `0.5905` | `0.4944` | `0.2633` | `0.2185` / `193.47` |
| **`resid_x0_skip`** | **`0.4143`** | `0.2292` | `0.2846` | **`0.5912`** | `0.5069` | `0.2780` | **`0.2415` / `154.29`** |
| `value_res` | `0.4088` | `0.2261` | **`0.2862`** | `0.5907` | `0.5086` | `0.2713` | `0.2272` / `162.50` |

`value_res`'s `+0.0059` BPB held-out perplexity edge does **not** translate into superior downstream task accuracy over `dense` or `resid_x0_skip` (trailing `resid_x0_skip` on `ARC-Easy`, `LAMBADA`, `OpenBookQA`, `ARC-Challenge`, and `PIQA`), reinforcing `G1`'s decision **not** to complicate the core `Dense + Muon` architecture with per-layer value skips.

### 2.3 Normalized Vector-Norm Telemetry & Causal Knockout Ablations (`3`-Seed Mean on `value_res`)

| Layer $\ell$ | Learned $\lambda_\ell$ | Nominal $v_0$ Coeff ($1 - \lambda_\ell$) | $\|v_\ell\|_{\text{rms}}$ | $\|v_0\|_{\text{rms}}$ | **True Normalized $v_0$ Norm Share** | $\cos(v_\ell, v_0)$ |
|---:|---:|---:|---:|---:|---:|---:|
| `1` | `0.7393` | `0.2607` | `0.0505` | `0.2242` | **`0.5772`** | `+0.0591` |
| `2` | `0.7395` | `0.2605` | `0.0625` | `0.2242` | **`0.5246`** | `+0.0048` |
| `3` | `0.9192` | `0.0808` | `0.0776` | `0.2242` | **`0.1908`** | `+0.1431` |
| `4` | `0.9451` | `0.0549` | `0.1269` | `0.2242` | **`0.0875`** | `+0.1097` |
| `5` | `0.6938` | `0.3062` | `0.1817` | `0.2242` | **`0.3323`** | `+0.0900` |
| `6` | `0.2099` | `0.7901` | `0.3687` | `0.2242` | **`0.6622`** | `-0.0230` |
| `7` | `0.1230` | `0.8770` | `0.4019` | `0.2242` | **`0.7588`** | `-0.0211` |

- **Empirical resolution of PR #211's vector-norm question:** In early layers (`l = 1, 2`), $\|v_\ell\|_{\text{rms}} \ll \|v_0\|_{\text{rms}}$, so a `26.1%` nominal $v_0$ scalar produces a **`52–58%`** effective vector-norm share; conversely, in Layer `7`, $\|v_7\|_{\text{rms}} = 0.4019 > \|v_0\|_{\text{rms}} = 0.2242$ (`1.79×`), so an `87.7%` nominal $v_0$ scalar corresponds to a **`75.88%` true vector-norm share** (`\cos \approx -0.021`).
- **Convergent U-shaped Layer-0 demand in `resid_x0_skip`:** Across all 3 seeds, `resid_x0_skip` learned $\alpha_7 \in [0.0623, 0.0779]$ (`mean 0.0726`, placing `92.7%` nominal weight on $h_0$ at Layer 7's input!), showing that tied-embedding PreNorm Transformers primarily want direct un-rotated input embeddings at the **final layer** before tied output projection—whether delivered via $v_0$ or $h_0$ (or simply eliminated by untying output embeddings at larger scale).

---

## 3. Part II — Hopper `R0-v2` Gated DeltaNet Benchmark (`H100 80GB HBM3`, `results/r0_hopper/r0_hopper_result.json`)

| Workload (`24L, d=896, 7×128 heads, SwiGLU 4864, V=151,936`, BF16) | Micro-Batch | Steady Step (ms) | Tok/s | Model TFLOPS (`6N`) | % of H100 `989` TFLOPS Peak | % of `817.1` TFLOPS GEMM Ceiling | Peak VRAM |
|---|---|---:|---:|---:|---:|---:|---:|
| **`GatedDeltaNet` (`546.9M`, `triton=3.7.1`)** | `4×1024` | `108.4` | `37,779` | `124.0` | `12.5%` | `15.2%` | `18.25 GiB` |
| **`GatedDeltaNet` (`546.9M`, `triton=3.7.1`)** | `8×1024` | `160.9` | `50,916` | `167.1` | **`16.9%`** | `20.4%` | `25.20 GiB` |
| **`GatedDeltaNet` (`546.9M`, `triton=3.7.1`)** | `16×1024` | `292.0` | `56,106` | `184.1` | **`18.6%`** | `22.5%` | `39.16 GiB` |
| **`GatedDeltaNet` (`546.9M`, `triton=3.7.1`)** | **`8×2048`** | **`291.9`** | **`56,133`** | **`184.2`** | **`18.6%` (`PASS ≥ 15%`)** | **`22.5%`** | **`39.16 GiB`** |
| `GatedDeltaNet` + **Cached Top-32 Sparse Teacher KD** | `8×2048` | `339.5` | **`48,261`** | `158.4` | **`16.0%`** | `19.4%` | `41.48 GiB` |
| `Softmax FlashAttention` (`527.0M` reference) | `16×1024` | `238.2` | `68,787` | `217.5` | `22.0%` | `26.6%` | `31.94 GiB` |
| `Softmax FlashAttention` (`527.0M` reference) | `8×2048` | `247.5` | `66,188` | `209.3` | `21.2%` | `25.6%` | `31.94 GiB` |

- **Profiler & Stage C Cost Projection:** CUDA kernel time splits across `recurrent_triton` (`14.1%`), `gemm` (`12.3%`), `norm_activation_elementwise` (`29.4%`), `optimizer` (`3.0%`), and `other/chunked-CE` (`41.1%` — dominated by `V = 151,936` chunked cross-entropy backward). At `48,261` tok/s (`0.0000s` async prefetch wait), **1B tokens of cached-logit sparse KD on `547M` Gated DeltaNet requires `5.76` H100-hours (`$15.48` at `$2.69/hr` community / `$25.27` at `$4.39/hr` secure)**, unblocking `RESEARCH-PLAN §20.6` Stage C once the free `135M` Stage B conversion test passes.

---

## 4. Program-Level Implications

1. **Close cross-depth scalar skips (`ValueRes`, `ResFormer SVRA`, `Block AttnRes`, `x0-skip`) for the Dense PreNorm + Muon trunk:**  
   With `H-13 Block AttnRes` closed on wall-clock at `14M` (PR #203) and `ValueRes` failing `G1` at `40.38M / 800M` (`+0.0059 < +0.0080` BPB, `74.6%` explained by generic final-layer $h_0$ access, no 7-task gain), **clean `PreNorm GQA + Muon (1% floor)` remains the undefeated dense Pareto baseline**. (A targeted ablation of **untied output head** vs tied embeddings is the natural structural follow-up to test whether Layer-7's $h_0/v_0$ pull is simply compensating for tied-embedding output rotation.)
2. **Proceed on the `SmolLM2 → Gated DeltaNet` Hybrid Conversion Track (`Stage B` -> Hopper `Stage C`):**  
   With Hopper `R0-v2` verified (`triton==3.7.1`, `56.1k` tok/s, `$15.50 / 1B` KD tokens), Hopper GPU execution is no longer blocked; once the free `135M` `Stage B` inheritance gate completes (`B1`/`B2`), `Stage C` (`0.5B`) can execute cleanly within a `$20–$50` H100 envelope.
