"""Probe untrained HSPA rank, prefix invariance and forward timing on synthetic tokens."""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from research.unified_loop.mechanisms.zoo import UnifiedBlockLM, count_trainable_params


MECHANISM = "M22_HSPA_FullCore_U09"
WIDTH, HEADS, VOCAB = 128, 4, 256
SEEDS = (11, 29, 47)


def count_non_emb_params(model):
    return count_trainable_params(model) - model.emb.weight.numel()


def model_for(registers, ffn_width, seed, device="cpu"):
    torch.manual_seed(seed)
    return UnifiedBlockLM(
        VOCAB, WIDTH, MECHANISM, n_heads=HEADS, ffn_hidden_override=ffn_width,
        hspa_registers=registers,
    ).eval().to(device)


def layer_ranks(output):
    # Numerical rank uses the float32 SVD tolerance even when accumulating on CPU.
    x = output.detach().cpu().float()
    x = x - x.mean(dim=1, keepdim=True)
    values = torch.linalg.svdvals(x.reshape(-1, WIDTH))
    tol = values[0] * max(x.shape[0] * x.shape[1], WIDTH) * torch.finfo(x.dtype).eps
    # Match U09's centered rank: normalize each time row, then compute spectral
    # entropy separately for every sample and projected output head.
    per_head = F.normalize(x.unflatten(-1, (HEADS, WIDTH // HEADS)), dim=-1).transpose(1, 2)
    energy = torch.linalg.svdvals(per_head).square()
    p = energy / energy.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    effective = (-(p * p.clamp_min(1e-12).log()).sum(dim=-1)).exp()
    return {"centered_numerical_rank": int((values > tol).sum()), "tolerance": float(tol),
            "mean_head_centered_effective_rank": float(effective.mean())}


def forward_timing(model, tokens, repeats):
    synchronize = torch.mps.synchronize if tokens.device.type == "mps" else lambda: None
    for _ in range(2):
        model(tokens)
    synchronize()
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        model(tokens)
        synchronize()
        durations.append(time.perf_counter() - start)
    median = statistics.median(durations)
    return {"seconds": durations, "median_tokens_per_second": tokens.numel() / median}


@torch.inference_mode()
def run(device, repeats):
    # A single SwiGLU hidden unit adds the same number of weights at every width.
    reference = count_non_emb_params(model_for(4, 460, 0))
    wide = count_non_emb_params(model_for("full_head", 460, 0))
    unit = wide - count_non_emb_params(model_for("full_head", 459, 0))
    matched_ffn = 460 - round((wide - reference) / unit)
    arms = [(4, 460, "U09"), ("full_head", 460, "same_ffn"),
            ("full_head", matched_ffn, "nearest_parameter_count")]
    generator = torch.Generator().manual_seed(20261119)
    tokens = torch.randint(VOCAB, (2, 256), generator=generator).to(device)
    other = torch.randint(VOCAB, tokens.shape, generator=generator).to(device)
    rows = []
    for seed in SEEDS:
        for registers, hidden, label in arms:
            model = model_for(registers, hidden, seed, device)
            outs = []
            hooks = [block.mix.register_forward_hook(lambda module, args, out: outs.append(out))
                     for block in model.blocks]
            try:
                logits = model(tokens)
            finally:
                for hook in hooks:
                    hook.remove()
            if not torch.isfinite(logits).all() or any(not x.isfinite().all() for x in outs):
                raise RuntimeError(f"nonfinite output in {seed=} {label=}")
            ranks = [layer_ranks(out) for out in outs]
            errors = []
            for length in (7, 129):
                changed = torch.cat([tokens[:, :length], other[:, length:]], dim=1)
                expected = logits[:, :length]
                suffix_error = (model(changed)[:, :length] - expected).abs().max().item()
                crop_error = (model(tokens[:, :length]) - expected).abs().max().item()
                errors.append({"length": length, "changed_suffix_max_abs": suffix_error,
                               "cropped_max_abs": crop_error})
                if max(suffix_error, crop_error) > 1e-4:
                    raise RuntimeError(f"unstable prefix in {seed=} {label=} {errors[-1]}")
            mix = model.blocks[-1].mix
            rank_bound = min(WIDTH, mix.fdim)
            if any(r["centered_numerical_rank"] > rank_bound for r in ranks):
                raise RuntimeError("observed numerical rank exceeds projection bound")
            rows.append({
                "arm": label, "initialization_seed": seed, "registers_per_head": mix.r,
                "fiber_width": mix.fdim, "ffn_width": hidden, "nonembedding_parameters": count_non_emb_params(model),
                "global_projection_rank_bound": rank_bound, "layers": ranks, "prefix_errors": errors,
                "forward_timing": forward_timing(model, tokens, repeats),
            })
    return {"schema": "nano.u10.synthetic_register_preflight.v1", "device": device, "torch": str(torch.__version__),
            "platform": platform.platform(), "cpu_threads": torch.get_num_threads(), "input_seed": 20261119,
            "batch": 2, "length": 256, "vocabulary": VOCAB, "d_model": WIDTH, "heads": HEADS,
            "prefix_absolute_tolerance": 1e-4, "timing_warmups": 2, "timing_repeats": repeats, "measurements": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.threads, args.repeats) < 1:
        parser.error("threads and repeats must be positive")
    torch.set_num_threads(args.threads)
    result = run(args.device, args.repeats)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
