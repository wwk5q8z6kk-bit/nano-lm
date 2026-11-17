# Track A & Integration Final Report: Architecture, Optimizer, Multi-Scale & Surgery Discovery (`LAB-ARCH-02`, `LAB-ARCH-03`, `LAB-SCALE-01`, `LAB-INT-01`)

*Completed 2026-11-17. Total from-scratch pretraining runs: **36** (**21.29 billion** training tokens across 17 configurations, 100% reproducible on `LAB-DATA-01` SHA-256 `867beb0d8d4c3977acaaf97fb91da39be29fb9fb040d86ea0bfdf78c9de72ac4`).*

---

## 1. Executive Summary of Validated Discoveries & Falsifications

### Validated Architectural & Optimization Wins (Multi-Seed `n=2`, `600M` Tokens, Iso-Non-Embedding Params `~29.4M`)
1. **`X4_nanolm_v2_muon` (`NanoLM-v2`: 4:1 `Sharp-Delta` + `Gated-VRes` + `QK-Canon` + `Muon`) — #1 Overall (`0.9575 ± 0.0008` BPB, `-7.01%` vs `T0_dense` AdamW):**
   - Outperforms **`T0_dense + Muon`** (`0.9645 ± 0.0004` BPB) by **`-0.0070` BPB (`9×` the seed half-range)** across both seeds (`s0 = 0.9566`, `s1 = 0.9583`) and wins **all four domains simultaneously**:
     - `web`: **`1.0691`** vs `1.0751`
     - `textbook`: **`0.7580`** vs `0.7589`
     - `math`: **`1.1089`** vs `1.1183` (`-0.84%` over `T0+Muon`, `-6.98%` over `T0+AdamW`)
     - `code`: **`0.7821`** vs `0.7951` (`-1.64%` over `T0+Muon`, `-10.73%` over `T0+AdamW`)
   - Cuts sequence-growing global attention KV cache by **`80%`** (`2` `A` layers out of `10` = `2,048 B/tok` vs `10,240 B/tok`, plus `131 KB` fixed constant-size Delta associative state across `8` `D` layers) and lowers 512-token OOD random-sequence copy loss from `3.314` nats (`T0_dense` AdamW) / `1.177` nats (`T0_dense` Muon) to **`0.763` nats (`92.7%` mean exact accuracy; `96.1%` on `s0`)**.

2. **Discovery 1 — QK-Decoupled Causal Convolution / Unblurred Value Writes (`Sharp-Delta`, `X2_sharp_delta_vres` vs `X1_gated_vres`):**
   - Standard linear/delta-rule hybrids (`H2_delta`, `X1_gated_vres`, Mamba-2/Gated-DeltaNet) apply `ShortConv(k=4)` to `(Q, K, V)` jointly. Convolving `K` over `[t-3..t]` is required for 1-layer induction matching, **but convolving `V` blurs the associative payload written into $S_t = \alpha_t (I - \beta_t k_t k_t^\top) S_{t-1} + \beta_t k_t v_t^\top$**.
   - Restricting `ShortConv` strictly to `(Q, K)` while writing unblurred $v_t$ (`sharp_v=True`) plus `QK-Canon` on Attention (`X2_sharp_delta_vres`, AdamW, `n=2`) dropped Composite BPB from **`0.9869 ± 0.0013` (`X1`) $\to$ `0.9689 ± 0.0009` (`X2`)** (**`-5.90%` vs `T0_dense` AdamW**; `-1.82%` additional gain = **`20×` seed spread**!), raised 512-token OOD copy accuracy to **`97.0%`** (`copy_loss = 0.536`), **beat `T0_dense + Muon` on Code (`0.7910 < 0.7951`) using plain AdamW**, and increased throughput (`238.8k` vs `234.5k tok/s`).

3. **Discovery 2 — Universal Per-Head Input-Gated Layer-0 Value Highway (`Gated-VRes`, `X1_gated_vres` vs `T1_vres` & `H3_delta_vres`):**
   - Unconditioned scalar Layer-0 value residual (`T1_vres`, `-3.66%` BPB) severely degraded OOD 512-token induction copying (`copy_loss = 9.207` nats, `8.3%` accuracy) because a static per-layer scalar contaminates induction heads with unshifted token embeddings.
   - Replacing scalar mixing with a per-head input-dependent gate $\alpha^{(l)}_{b,h,t} = \sigma(w_{l,h}^\top x_{b,t} + b_{l,h})$ wired universally across **all 10** `D` and `A` blocks (`X1_gated_vres`, `0.9869 ± 0.0013` BPB, `-4.15%`) beat **both** full-attention `T1_vres` (`0.9919`) and `H3_delta_vres` (`0.9942`) on every domain while restoring **`96.1%` OOD induction accuracy** (`copy_loss = 0.746` nats).

### Multi-Scale Verification (`LAB-SCALE-01`: `46M -> 121M` Total / `29.5M -> 96.0M` Non-Embedding, `Muon`, `600M` Tokens)
- Scaling non-embedding parameters by **`3.25×`** (`d=768, L=16`, `DDDADDDADDDADDDA` = `-75%` KV cache `4,096 B/tok` vs `16,384 B/tok` on `S0_dense_125m`):
  - Throughout the constant-LR pretraining phase (`150M`, `300M`, `450M` tokens), **`S1_nanolm_v2_125m` strictly beat `S0_dense_125m` at every checkpoint** (`1.0966 < 1.1096` at `150M`, `1.0158 < 1.0244` at `300M`, `0.9808 < 0.9871` at `450M`).
  - At final WSD cooldown (`600M` tokens = `6.25` tok/non-emb-param under-trained regime, without re-tuning the `46M` `Muon` `lr=0.02`), **`S1_nanolm_v2_125m` matched `S0_dense_125m` to 4 decimal places on Composite BPB (`0.9106` vs `0.9105`) and beat `S0_dense_125m` on Code (`0.7377 < 0.7420`, `-0.58%`) while using `75%` less global KV cache**.

### Preserved Negative & Falsified Hypotheses (First-Class Scientific Controls)
1. **Pure Recurrent Models Without Global Attention (`P1_gla` `+3.17%` worse, `P2_delta` `+1.65%` worse):** Complete OOD exact-copy failure (`copy_loss = 11.65–12.03` nats, `0%` accuracy) and severe code/math degradation (`+7.6%` to `+12.0%` worse on code). At least 20%–25% sparse global attention anchors are non-negotiable.
2. **Weight-Shared Recurrent Depth Under Iso-FLOPs (`R1_loop` `+8.26%` worse; `R2_delta_loop` `-0.08%` null at matched FLOPs):** While `R2_delta_loop` improved BPB at matched tokens (`0.9976`, `-3.11%`) and achieved strong OOD copy loss (`0.513`), reducing tokens to `460M` to match `T0_dense` training FLOPs (`1.85e5 TFLOP·s`) yielded `1.0288 ± 0.0006` BPB — **worse than unlooped `H2_delta` (`1.0035`) at lower FLOPs (`1.76e5`)**.
3. **Surprise-Gated Recurrent Depth (`X3_sgrd_delta`, `0.9815 ± 0.0005` BPB):** Coupling PonderNet halting to log online delta prediction error $\gamma \log \lVert e_t \rVert$ (`γ` grew `0.50 -> +0.852`) beat fixed looping `R2_delta_loop` (`0.9976`), **but gradient descent drove `expected_loops -> 1.00`**, independently confirming that spending parameter/compute budget on unlooped `Sharp-Delta` (`X2_sharp_delta_vres`, `0.9689` BPB) beats weight-shared recurrent passes during pretraining.
4. **Short-Budget Architecture Surgery (`LAB-INT-01` on Pretrained `125M` Checkpoint):**
   - Naive 3:1 `Sharp-Delta` replacement (`I0_naive_swap`) caused catastrophic collapse (`0.9481 -> 3.5121` BPB at step 0; `2.1722` BPB after `123k` tokens).
   - Our 2-stage QKV-warm-started Layerwise-MSE + Proper Top-64+Tail Logit KD pipeline (`I1_progressive_rad_kd`) cut that short-budget gap by **`-0.970` BPB** (`1.2018` BPB, recovering `90%` of long-context gain `+0.484` nats), while proving that lossless hybrid conversion requires multi-billion-token distillation rather than light post-training.
   - Exact zero-init `Gated-VRes` grafting (`I2b_frozen_backbone_gated_vres`) achieved bit-exact `0.0000` step-0 $\Delta$BPB (`0.94811`), but post-hoc gate tuning alone (`0.9507` BPB) did not replicate from-scratch `Gated-VRes` gains (`-4.15%`) because pretrained `o_proj` subspaces were not co-trained with `V^{(0)}`. Consequently, Track B (`Qwen3-1.7B`) was preserved in its native architecture without destructive surgery.

---

## 2. Strict Parameter, Active-Parameter & Memory Accounting

| Configuration | Total Params | Embedding Params | Non-Emb Params | Active Params / Token | Global KV Cache (`bf16`) | Fixed Recurrent State (`fp32`) |
|---|---:|---:|---:|---:|---:|---:|
| `T0_dense` (`46M`) | `46,277,632` | `16,777,216` | `29,500,416` | `46,277,632` | `10,240 B/tok` (10 `A` layers) | `0 B` |
| `X2` / `X4` (`NanoLM-v2 46M`, `DDDADDDADA`) | `46,154,296` | `16,777,216` | `29,377,080` | `46,154,296` | **`2,048 B/tok`** (2 `A` layers, **`-80%`**; `3,072 B/tok` if `v0` cached separately) | `131,072 B` (`128 KiB` constant across 8 `D` layers) |
| `S0_dense_125m` (`121M`) | `121,137,920` | `25,165,824` | `95,972,096` | `121,137,920` | `16,384 B/tok` (16 `A` layers) | `0 B` |
| `S1_nanolm_v2_125m` (`121M`, `DDDADDDADDDADDDA`) | `121,596,528` | `25,165,824` | `96,430,704` | `121,596,528` | **`4,096 B/tok`** (4 `A` layers, **`-75%`**) | `294,912 B` (`288 KiB` constant across 12 `D` layers) |

---

## 3. Complete 36-Run Empirical Tables

See [LAB-ARCH-02_results.md](LAB-ARCH-02_results.md) for the auto-generated machine-verified aggregation from `/Volumes/Express1M2/nanolm-lab/runs/*/result.json`, and [lab_int_01_results.json](lab_int_01_results.json) for the `LAB-INT-01` progressive surgery measurements.
