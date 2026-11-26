"""Hardware Preflight Profiler and Compute/Cost Scaling Estimator for NanoLM."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any

import torch

from nanolm.config import NanoLMConfig
from nanolm.model.transformer import NanoLMForCausalLM


@dataclass(frozen=True, slots=True)
class ComputeEstimate:
    model_name: str
    target_tokens_billions: float
    total_training_flops: float
    min_zepto_vram_gb_bf16_adamw: float
    min_vram_gb_with_grad_ckpt: float
    estimated_8xh100_hours: float
    estimated_8xh100_cost_usd: float
    estimated_4xb200_hours: float
    estimated_4xb200_cost_usd: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_full_training_budget(
    cfg: NanoLMConfig,
    target_tokens_billions: float = 100.0,
    h100_mfu: float = 0.42,
    h100_hourly_8x_usd: float = 19.92,  # ~$2.49/GPU-hr RunPod Spot/Community reference
    b200_mfu: float = 0.45,
    b200_hourly_4x_usd: float = 21.60,  # ~$5.40/GPU-hr RunPod B200 reference
) -> ComputeEstimate:
    """Estimate training FLOPs, optimizer/activation VRAM, wall-clock hours, and USD cost."""
    acct = cfg.parameter_accounting()
    n_active = acct.active_parameters_per_token
    n_total = acct.total_deployed_parameters + acct.training_only_parameters
    tokens = target_tokens_billions * 1e9

    # Standard 6 * N_active * D training FLOP rule
    total_flops = 6.0 * n_active * tokens

    # Memory: bf16 weights (2B) + bf16 grads (2B) + fp32 master weights & AdamW m,v (12B) = 16 bytes * N_total
    opt_state_gb = (16.0 * n_total) / (1024**3)
    # Activation estimate for micro_batch=4, seq=4096
    act_full_gb = (4 * 4096 * cfg.hidden_size * acct.effective_depth * 12) / (1024**3)
    act_ckpt_gb = (4 * 4096 * cfg.hidden_size * acct.effective_depth * 2.5) / (1024**3)

    # 8x H100 SXM bf16 peak ~ 8 * 989 TFLOP/s = 7.912e15 FLOP/s
    eff_8xh100_flops = 8 * 989e12 * h100_mfu
    h100_hours = total_flops / eff_8xh100_flops / 3600.0

    # 4x B200 bf16 peak ~ 4 * 2250 TFLOP/s = 9.0e15 FLOP/s
    eff_4xb200_flops = 4 * 2250e12 * b200_mfu
    b200_hours = total_flops / eff_4xb200_flops / 3600.0

    return ComputeEstimate(
        model_name=cfg.name,
        target_tokens_billions=target_tokens_billions,
        total_training_flops=total_flops,
        min_zepto_vram_gb_bf16_adamw=round(opt_state_gb + act_full_gb, 2),
        min_vram_gb_with_grad_ckpt=round(opt_state_gb + act_ckpt_gb, 2),
        estimated_8xh100_hours=round(h100_hours, 2),
        estimated_8xh100_cost_usd=round(h100_hours * h100_hourly_8x_usd, 2),
        estimated_4xb200_hours=round(b200_hours, 2),
        estimated_4xb200_cost_usd=round(b200_hours * b200_hourly_4x_usd, 2),
    )


@torch.inference_mode()
def benchmark_live_inference(
    cfg: NanoLMConfig,
    prompt_len: int = 256,
    decode_tokens: int = 32,
    device: str = "mps",
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Any]:
    """Instantiate real model weights (including 2B-4B!) on device and measure prefill & KV-cached decode."""
    t_init0 = time.perf_counter()
    model = NanoLMForCausalLM(cfg)
    model.strip_training_only_heads()
    model = model.to(device=device, dtype=dtype).eval()
    if device == "mps":
        torch.mps.synchronize()
    init_sec = time.perf_counter() - t_init0

    x = torch.randint(0, cfg.vocab_size, (1, prompt_len), device=device)
    # Warmup
    _ = model(input_ids=x[:, :16], use_cache=True)
    if device == "mps":
        torch.mps.synchronize()

    # Prefill timing
    t0 = time.perf_counter()
    out = model(input_ids=x, use_cache=True)
    if device == "mps":
        torch.mps.synchronize()
    prefill_sec = max(1e-6, time.perf_counter() - t0)

    # Autoregressive KV-cached decode timing
    kv = out.past_key_values
    next_tok = out.logits[:, -1:, :].argmax(dim=-1)
    t1 = time.perf_counter()
    for _ in range(decode_tokens):
        out = model(input_ids=next_tok, past_key_values=kv, use_cache=True)
        kv = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
    if device == "mps":
        torch.mps.synchronize()
    decode_sec = max(1e-6, time.perf_counter() - t1)

    # Actual KV cache bytes allocated
    kv_bytes = 0
    if kv is not None:
        for entry in kv:
            kv_bytes += entry.k_or_latent.numel() * entry.k_or_latent.element_size()
            kv_bytes += entry.v_or_rope.numel() * entry.v_or_rope.element_size()

    acct = cfg.parameter_accounting()
    res = {
        "model_name": cfg.name,
        "device": device,
        "dtype": str(dtype),
        "deployed_parameters": model.count_deployed_parameters(),
        "active_parameters_per_token": acct.active_parameters_per_token,
        "init_seconds": round(init_sec, 3),
        "prompt_len": prompt_len,
        "prefill_tok_per_s": round(prompt_len / prefill_sec, 1),
        "decode_tokens": decode_tokens,
        "decode_tok_per_s": round(decode_tokens / decode_sec, 2),
        "measured_kv_cache_mb": round(kv_bytes / (1024**2), 3),
    }
    del model, out, kv, x
    if device == "mps":
        torch.mps.empty_cache()
    return res
