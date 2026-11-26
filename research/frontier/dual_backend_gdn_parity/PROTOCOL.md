# Pre-Registered Protocol: Dual-Backend `PyTorch/CUDA (fla 0.6.0) <-> JAX/XLA (Cloud TPU)` Gated DeltaNet Parity & `R0-TPU` Hardware Routing Gate

**Registered before executing `kaggle_fla_probe.py` (CUDA GPU) or `r0_tpu_bench_and_parity.py` (Cloud TPU) (`2026-11-26`).**  
**Base lineage:** `origin/master` (`3e36bf8`) · Enforces strict cross-backend parity across RunPod/Kaggle CUDA GPUs and Google Cloud TPUs (`v5e` / `v6e`) prior to `Stage C` (`0.5B -> sub-4B` hybrid conversion).

---

## 1. Architectural Invariant & Motivation

Every frontier Nano LM architecture (`Dense GQA` and `Hybrid Gated DeltaNet + GQA`) must run interchangeably across **PyTorch/CUDA (`fla==0.6.0`)** and **JAX/XLA (`Cloud TPU v5e/v6e`)** with **1:1 lossless checkpoint compatibility**, **verified numerical forward/backward parity**, and **measured cost/throughput routing**.

To eliminate all approximations between CUDA Triton (`fla.layers.GatedDeltaNet`) and XLA TPU, `dual_gdn_core.py` implements the exact `fla==0.6.0` layer contract in both frameworks:
- **Exact 13-tensor per-layer parameter schema:** `q_proj.weight [HK, D]`, `k_proj.weight [HK, D]`, `v_proj.weight [HV*V, D]`, `a_proj.weight [HV, D]`, `b_proj.weight [HV, D]`, `A_log [HV]`, `dt_bias [HV]`, `q_conv1d.weight [HK, 1, 4]`, `k_conv1d.weight [HK, 1, 4]`, `v_conv1d.weight [HV*V, 1, 4]`, `g_proj.weight [HV*V, D]`, `o_norm.weight [V]`, `o_proj.weight [D, HV*V]`.
- **Exact operator semantics:**
  1. Depthwise causal `ShortConvolution(W=4, silu)` accumulated in `FP32` via 4 register-aligned causal shifts (zero layout-transpose overhead on TPU).
  2. Per-head `L2Norm(eps=1e-6)` on `q, k`, `beta = sigmoid(b_proj(x))` (`* 2` if `allow_neg_eigval`), and log-decay `g = -exp(A_log) * softplus(a_proj(x) + dt_bias)`.
  3. **Chunkwise WY Parallel Representation (`BT = 64`):** Solves `(I + U_strict)^{-1}` across all chunks in parallel via **6 batched `64x64` Neumann matrix multiplications (`(I+N)(I+N^2)(I+N^4)(I+N^8)(I+N^{16})(I+N^{32})`, exact because `N^{64} = 0` for strictly lower-triangular `64x64` `N`)** at `jax.lax.Precision.HIGHEST` on TPU MXUs, followed by `jax.lax.scan` over `T / 64` inter-chunk states `[B, H, K, V]`.
  4. Output `FusedRMSNormGated(eps=1e-5, activation='swish')` in `FP32` followed by `o_proj`.

---

## 2. Pre-Registered Parity & Hardware Gates (`score_dual_parity.py`)

- **`P1_checkpoint_bridge_bit_exact` (`PASS` iff `roundtrip_max_abs_diff == 0.0` and exact parameter counts hold):**
  - Verifies lossless `PyTorch fla state_dict -> NPZ/Safetensors -> JAX PyTree -> PyTorch fla state_dict` round-trip (`0.0` bit drift) and exact parameter counts (`151,928,140` for `SmolLM2-135M` `22 GDN + 8 GQA`; `546,865,360` for `R0` `24L` `GatedDeltaNet`).
- **`P2_cuda_fla_vs_jax_fp32_parity` (`PASS` iff `fp32_logit_max_abs_err <= 1e-4`, `fp32_loss_abs_err <= 1e-5`, and `fp32_grad_max_rel_err <= 2e-3`):**
  - Compares live CUDA `fla==0.6.0` Triton kernels (`kaggle_fla_probe.py` on GPU) directly against JAX/XLA (`r0_tpu_bench_and_parity.py` on Cloud TPU) on the exact same exported hybrid weights and token inputs in `FP32`.
- **`P3_cuda_fla_vs_jax_bf16_and_multistep_parity` (`PASS` iff `bf16_loss_abs_err <= 2e-3` nats and `step10_loss_abs_err <= 5e-3` nats):**
  - Compares native mixed-precision forward loss and a `10`-step `AdamW` training trajectory starting from identical initial weights and batches between live CUDA `fla==0.6.0` (`BF16` on `sm >= 80`, `FP16` Tensor Core on `sm_75` Turing `T4`) and JAX Cloud TPU (`BF16` MXU).
- **`P4_tpu_r0_stability_and_throughput_gate` (`PASS` iff `nonfinite_steps_100 == 0` AND (`tpu_gdn_547m_mfu >= 0.15` OR `tpu_gdn_vs_attn_speed_ratio >= 0.60` at `seq >= 2048`):**
  - Benchmarks `546.9M` (`24L, d=896, 7x128, SwiGLU 4864, V=151,936`) pure `GatedDeltaNet`, `541.8M` `18 GDN + 6 GQA` Stage-C hybrid, and `527.0M` pure `Softmax GQA` across `seq in {2048, 4096, 8192}` (1-chip and 4-chip `pmap`, with and without cached top-32 sparse teacher KD) plus a `100`-step stability soak test on Cloud TPU.
- **`D_ROUTING` (Unified Cross-Hardware Routing Verdict):**
  - `DUAL_BACKEND_PARITY_VERIFIED_READY_FOR_STAGE_C` iff **`P1 AND P2 AND P3 AND P4`** all pass, unlocking interchangeable `Stage C` execution across Cloud TPU (`v5e`/`v6e`) and Hopper GPUs (`H100`).
