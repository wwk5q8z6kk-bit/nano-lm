# Pre-Registered Protocol: `40.38M` (`M`-Scale) Value-Residual Confirmation Gate & Hopper `R0` Gated DeltaNet Fix

**Registered before any execution of `valueres_42m_jax.py` or `r0_bench_hopper.py` (`2026-11-25`).**  
**Base lineage:** `origin/master` (`3e36bf8`) · Incorporates all P0 audit controls from Draft PR #211, #212, and #213.

---

## Part I — `40.38M` (`M`-Scale, `800M` Tokens, `3` Seeds) Value-Residual Confirmation Tournament

### 1. Motivation & Audit Controls Addressed

In PR #203 (`14.29M` `S`-scale, `98.30M` tokens/arm, `3` seeds), `value_res` (`+7` scalar Layer-0 value skips) improved held-out `mix` BPB by `+0.0166 ± 0.0009` ($t = 31.75$) and `code` BPB by `+0.0349` ($t = 14.54$) at zero runtime/VRAM cost, outperforming `10`-layer untied `dense10` (`+1.48M` params) while `H-13 Block AttnRes` lost `-0.0383` raw / `-0.0281` steady-state BPB at equal wall-clock. Because PR #203's pre-registered `D1` fell through to `CLOSE_DEPTH_ROUTING_AT_14M`, and subsequent P0 audit PR #211 identified four requirements before architecture promotion, this protocol tests all four at **`2.83×` model scale (`40.38M`) and `8.14×` token horizon (`800M` unique tokens = `19.81` tok/param, `9.60B` total trained tokens across `12` runs)**:

1. **Prior-art comparator (`resformer_fixed`, Zhou et al. ACL 2025 *ResFormer* SVRA):** Tests whether **learned** per-layer mixing $v_\ell \leftarrow \lambda_\ell v_\ell + (1 - \lambda_\ell) v_0$ (`+7` scalars, where `14M` learned $\lambda_7 \approx 0.09$) measurably beats **parameter-free fixed `0.5 / 0.5`** Layer-0 value mixing ($v_\ell \leftarrow 0.5 v_\ell + 0.5 v_0$).
2. **Matched-parameter first-layer shortcut control (`resid_x0_skip`):** Tests whether `value_res`'s gain requires **attention value-head transport ($v_0$)** or is matched by a generic **`+7`-scalar Layer-0 residual-stream skip** $x_{\text{in}}^{(\ell)} \leftarrow \alpha_\ell h_\ell + (1 - \alpha_\ell) h_0$ (initialized at $\alpha_\ell = 1.0$ so step-0 matches `dense` while $\partial \mathcal{L} / \partial \alpha_\ell = \langle \nabla h_{\text{in}}^{(\ell)}, h_\ell - h_0 \rangle \ne 0$ from step 1).
3. **Normalized vector-norm telemetry & causal knockout ablations:** Logs per-layer $\|v_\ell\|_{\text{rms}}$, $\|v_0\|_{\text{rms}}$, cosine similarity $\cos(v_\ell, v_0)$, and **true vector-norm share** $\mathbb{E}[\|(1-\lambda_\ell)v_0\|_2 / (\|\lambda_\ell v_\ell\|_2 + \|(1-\lambda_\ell)v_0\|_2)]$, plus four zero-cost inference ablations on every trained checkpoint: `no_skip` ($\lambda_\ell \leftarrow 1$), `ablate_late_6_7` ($\lambda_{6,7} \leftarrow 1$), `ablate_mid_3_4` ($\lambda_{3,4} \leftarrow 1$), and `perm_v0` (permuting $v_0$ across sequence positions).
4. **Cryptographic provenance & strict paired-significance scorer (`score_42m_gate.py`):** Every output JSON records SHA256 hashes of the trainer source, `manifest.json`, all four `val_*.bin` shards, `tasks-v1.json`, `train.bin` byte size + `64 MiB` header hash, and the saved `.npz` checkpoint, plus 7-task zero-shot benchmark scores (`hellaswag, arc_easy, arc_challenge, piqa, winogrande, openbookqa, lambada`).

### 2. Experimental Matrix & Recipe (`valueres_42m_jax.py`)

- **Corpus:** `Corpus v2` (`gs://nanolm-lab-eu-0478806098/corpus/v2/train.bin`, `10.0B` decontaminated tokens, zero epoch repetition) + canonical held-out `val_{web,textbook,math,code}.bin` (`655,360` tokens, `np.random.default_rng(0)`) + `tasks-v1.json`.
- **Optimizer & Schedule:** `optax.contrib.muon` (`lr = 4e-3`, `wd = 0.1`, `adam_weight_decay = 0.1`, `consistent_rms = 0.2`, global grad clip `1.0` — matching `train_jax.py` `--opt muon` exactly), `bs = 64`, `seq = 512` (`32,768` tok/step), `warmup = 500`, cosine decay to `lr_floor = 0.01`, **`24,400` steps (`122` compiled XLA `jax.lax.scan` chunks of `200` steps) = `799,539,200` tokens/run**.
- **Hardware & Parallel Execution:** `1 × Cloud TPU v5litepod-4` (`4` TPU v5e chips, `BF16` compute with `FP32` norm/softmax/loss). Within each seed (`1000, 1001, 1002`), **all 4 arms train simultaneously in parallel**, one arm per chip, with cyclic Latin-square chip assignment `chip = (arm_idx + seed_idx) % 4` and JIT compilation completed before starting `train_sec`.

| Arm | Definition (`d=512, layers=8, ffn=1408, heads=8, kv=4, V=32,768`) | Exact Params | Extra Params |
|---|---|---:|---:|
| `dense` | Standard PreNorm GQA Transformer (tied embeddings) | `40,379,904` | `0` |
| `resid_x0_skip` | Block input $x_{\text{in}}^{(\ell)} = \alpha_\ell h_\ell + (1 - \alpha_\ell) h_0$ ($\ell=1..7$, init $\alpha_\ell=1.0$) | `40,379,911` | `+7` |
| `resformer_fixed` | Fixed uniform SVRA $v_\ell = 0.5 v_\ell + 0.5 v_0$ ($\ell=1..7$, ACL 2025 prior art) | `40,379,904` | `0` |
| `value_res` | Learned affine SVRA $v_\ell = \lambda_\ell v_\ell + (1 - \lambda_\ell) v_0$ ($\ell=1..7$, init $\lambda_\ell=0.5$) | `40,379,911` | `+7` |

### 3. Pre-Registered Predictions (`M1–M4`) & Mechanical Decision Rules (`G1–G2`)

- **M1 (Scale retention):** `value_res` beats `dense` on 3-seed `mix` BPB by `0.008–0.020` BPB at `40.38M` / `800M` tokens (`code` gain `≥ +0.015` BPB) with `< 2%` step-time delta.
- **M2 (Mechanism specificity over generic $h_0$ skip):** `value_res` beats `resid_x0_skip` by `≥ 0.004` BPB (`|t| > 4.303`).
- **M3 (Learned $\lambda_\ell$ vs fixed `0.5/0.5` ResFormer):** Late layers (`l = 6, 7`) again learn $\lambda_\ell < 0.25$ with effective normalized $v_0$ vector-norm share `> 0.65`, giving `value_res` a `0.002–0.006` BPB edge over `resformer_fixed` (or, if `< 0.002`, establishing that parameter-free `0.5/0.5` SVRA suffices).
- **M4 (Positional $v_0$ knockout):** Position-shuffling $v_0$ (`perm_v0`) degrades `value_res` held-out `mix` BPB by `≥ 0.05` BPB in every seed, and ablating late layers (`ablate_late_6_7`) hurts more than ablating middle layers (`ablate_mid_3_4`).

**Mechanical Decision Rules (`score_42m_gate.py`, two-sided 95% paired $|t| > 4.303$, `df = 2`, enforced on BOTH equal-token and equal-wall-clock):**
- **`G1_42m_architecture_gate`:**
  - Define `clears_base(A)` $\iff$ $\bar{\Delta}_{\text{tok}}(A, \text{dense}) \ge 0.008$ (`sig_95` & `all_positive`), $\bar{\Delta}_{\text{time}}(A, \text{dense}) \ge 0.008$ (`sig_95` & `all_positive`), and $\bar{\Delta}_{\text{tok}}(A, \text{resid\_x0\_skip}) \ge 0.004$ (`sig_95` & `all_positive`).
  - `ADOPT_LEARNED_VALUE_RES_AT_42M` iff `clears_base(value_res)` **and** $\bar{\Delta}_{\text{tok}}(\text{value\_res}, \text{resformer\_fixed}) \ge 0.002$ (`sig_95`).
  - `ADOPT_RESFORMER_FIXED_05_AT_42M` iff `clears_base(resformer_fixed)` and (`not clears_base(value_res)` or $\bar{\Delta}_{\text{tok}}(\text{value\_res}, \text{resformer\_fixed}) < 0.002$ or non-significant) — enforcing Occam's razor in favor of zero extra parameters when indistinguishable.
  - `ADOPT_VALUE_RES_FAMILY_TIE_LEARNED_VS_FIXED` iff `clears_base(value_res)` holds, `resformer_fixed` misses `clears_base`, and the `value_res` vs `resformer_fixed` margin is `< 0.002` or non-significant.
  - `REJECT_VALUE_RES_PROMOTION_AT_42M` otherwise.
- **`G2_causal_v0_mechanism_gate`:**
  - `CONFIRMED_LATE_POSITIONAL_V0_TRANSPORT` iff in all 3 seeds `perm_v0` degrades `value_res` `mix` BPB by $\ge 0.05$ BPB **and** `ablate_late_6_7` degrades `mix` BPB strictly more than `ablate_mid_3_4`.

---

## Part II — Hopper `R0-v2` Gated DeltaNet Fix & Utilization Gate (`RESEARCH-PLAN §20.6` F1)

- **Root cause of Nov 24 `R0` failure:** On H100 80GB (`r0_result.json`), `fla==0.6.0` raised `RuntimeError` inside `chunk_bwd_dqkwg` because the base container carried `triton==3.5.1` (`3.4.0 <= triton < 3.7.1` Hopper backward bug `#640`).
- **Minimal surgical fix (`r0_bench_hopper.py`):** Upgrade `triton==3.7.1` (with automatic `tilelang` fallback and an immediate `[1, 64, 2, 64]` forward+backward verification probe before benchmark timing) on the identical `547M` (`24L, d=896, 7×128 heads, SwiGLU 4864, V=151,936`) workload, under `r0_run.py`'s 3-layer self-terminating RunPod controller with a hard **`$4.00` cap** (`< $1.50` expected spend).
- **Pre-registered `RESEARCH-PLAN §20.6` F1 decision rule (preserved unchanged):**
  - `ADVANCE_STAGE_C_ON_HOPPER` iff `correctness` passes (`rel_err < 1e-2` in BF16 / `< 1e-4` in FP32) **and** `bench_gdn` steady-state training achieves **$\ge 15\%$ of H100 datasheet dense BF16 peak (`989 TFLOPS`, i.e. $\ge 148.35$ TFLOPS)** without NaN/Inf in `stability`.
  - `HOLD_500M_STAGE_C_OPTIMIZE_KERNELS_OR_DOWNSIZE` if steady-state GDN utilization is `< 15%` (or `< 8%` = compute-bound kernel bottleneck).
