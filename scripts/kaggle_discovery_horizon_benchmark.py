#!/usr/bin/env python3
"""EXP-INV-06: Extended 4x Horizon & CUDA Discrete GPU Long-Context Stress Benchmark.

Designed to run inside Kaggle free GPU sessions (Tesla T4 / P100 / L4) as well as any
NVIDIA CUDA host with read-only access to the decontaminated `manifest.json` + `.bin` corpus.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanolm_discovery.config import (
    ArchitectureConfig,
    ParadigmKind,
    compute_parameter_census,
    make_iso_param_suite,
    retune_iso_param_variant,
)
from nanolm_discovery.eval_benchmarks import CorpusStream
from nanolm_discovery.models import NanoLMDiscoveryModel
from nanolm_discovery.optim import build_optimizer


def _mean_std(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    mean = float(statistics.fmean(vals))
    std = float(statistics.stdev(vals)) if len(vals) > 1 else 0.0
    return {"mean": mean, "std": std, "n": len(vals)}


def _sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _select_cuda_amp_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.bfloat16 if device.type == "mps" else torch.float32
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def run_cuda_invariant_audit(device: torch.device, amp_dtype: torch.dtype) -> dict[str, Any]:
    """Verify exact numel, parallel-vs-step duality, and strict causality on target CUDA GPU."""
    mech_suite = make_iso_param_suite("mech")
    out: dict[str, Any] = {}
    for kind in (ParadigmKind.B0_TRANSFORMER, ParadigmKind.I1_PHMN, ParadigmKind.I3_HRWN, ParadigmKind.G3_HMWR):
        cfg = mech_suite[kind]
        torch.manual_seed(7)
        model = NanoLMDiscoveryModel(cfg).to(device).eval()
        census = compute_parameter_census(cfg)
        assert model.deployed_parameter_count() == census.total_deployed_parameters
        x = torch.randint(0, cfg.vocab_size, (2, 48), device=device)
        with torch.no_grad():
            par = model(x)
            states = None
            steps_l = []
            for t in range(x.shape[1]):
                st = model.forward_step(x[:, t], states=states)
                steps_l.append(st.logits[:, 0])
                states = st.states
            seq = torch.stack(steps_l, dim=1)
        duality_err = float((par.logits - seq).abs().max().item())

        x2 = x.clone()
        t0 = 23
        x2[:, t0 + 1 :] = (x2[:, t0 + 1 :] + 5) % cfg.vocab_size
        with torch.no_grad():
            o1 = model(x)
            o2 = model(x2)
        causality_err = float((o1.logits[:, : t0 + 1] - o2.logits[:, : t0 + 1]).abs().max().item())

        # Mixed-precision backward finiteness check
        model.train()
        y = torch.randint(0, cfg.vocab_size, (2, 48), device=device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            mo = model(x, targets=y)
        mo.loss.backward()
        grads_finite = all(bool(torch.isfinite(p.grad).all().item()) for p in model.parameters() if p.grad is not None)
        assert duality_err < 1e-4, f"{kind.value} CUDA duality_err={duality_err}"
        assert causality_err < 1e-5, f"{kind.value} CUDA causality_err={causality_err}"
        assert grads_finite, f"{kind.value} CUDA non-finite grads under {amp_dtype}"
        out[kind.value] = {
            "duality_max_err": duality_err,
            "causality_max_err": causality_err,
            "amp_dtype": str(amp_dtype).replace("torch.", ""),
            "grads_finite": grads_finite,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def run_single_horizon_arm(
    cfg: ArchitectureConfig,
    stream: CorpusStream,
    *,
    seed: int,
    device: torch.device,
    optimizer_kind: str,
    jep_weight: float,
    steps: int,
    batch_size: int,
    seq_len: int,
    checkpoints: tuple[int, ...],
    eval_tokens_per_domain: int,
) -> dict[str, Any]:
    amp_dtype = _select_cuda_amp_dtype(device)
    use_scaler = device.type == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cfg_run = replace(cfg, jep_head_enabled=(jep_weight > 0.0))
    model = NanoLMDiscoveryModel(cfg_run).to(device)
    base_lr = 1.2e-3 if cfg.d_model <= 256 else (8e-4 if cfg.d_model <= 384 else 6e-4)
    opt = build_optimizer(model, kind=optimizer_kind, lr=base_lr, lr_muon=base_lr * 2.0, weight_decay=0.05)
    warmup = max(20, steps // 10)

    curve: dict[str, float] = {}
    tail_losses: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync_cuda(device)
    t0 = time.perf_counter()

    ckpt_set = set(checkpoints)
    for step in range(steps):
        model.train()
        if step < warmup:
            lr_mult = float(step + 1) / float(warmup)
        else:
            prog = float(step - warmup) / float(max(1, steps - warmup))
            lr_mult = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog))
        for g in opt.param_groups:
            base = (base_lr * 2.0) if g.get("kind") == "muon" else base_lr
            g["lr"] = base * lr_mult

        x, y = stream.sample_train_batch(batch_size, seq_len, rng, device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type in ("cuda", "mps")):
            out = model(x, targets=y, jep_weight=jep_weight)
            loss = out.loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss on {cfg.name} seed={seed} step={step}: {loss.item()}")
        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if step >= steps - 25:
            tail_losses.append(float(out.ce_loss.item()))

        human_step = step + 1
        if human_step in ckpt_set and human_step < steps:
            v_mid = stream.evaluate_validation_bpb(
                model,
                device=device,
                seq_len=seq_len,
                eval_tokens_per_domain=min(65_536, eval_tokens_per_domain),
                batch_size=batch_size,
                use_bf16=(device.type in ("cuda", "mps")),
                amp_dtype=amp_dtype,
            )
            curve[str(human_step)] = float(v_mid["composite_bpb"])

    _sync_cuda(device)
    train_sec = time.perf_counter() - t0
    train_tokens = steps * batch_size * seq_len

    val_id = stream.evaluate_validation_bpb(
        model,
        device=device,
        seq_len=seq_len,
        eval_tokens_per_domain=eval_tokens_per_domain,
        batch_size=batch_size,
        use_bf16=(device.type in ("cuda", "mps")),
        amp_dtype=amp_dtype,
    )
    curve[str(steps)] = float(val_id["composite_bpb"])
    val_ood2x = stream.evaluate_validation_bpb(
        model,
        device=device,
        seq_len=seq_len * 2,
        eval_tokens_per_domain=min(65_536, eval_tokens_per_domain),
        batch_size=max(4, batch_size // 2),
        use_bf16=(device.type in ("cuda", "mps")),
        amp_dtype=amp_dtype,
    )
    peak_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
        if device.type == "cuda"
        else 0.0
    )
    res = {
        "seed": seed,
        "device": str(device),
        "amp_dtype": str(amp_dtype).replace("torch.", ""),
        "deployed_params": model.deployed_parameter_count(),
        "optimizer": optimizer_kind,
        "jep_weight": jep_weight,
        "train_tokens": train_tokens,
        "train_seconds": train_sec,
        "tok_per_sec": train_tokens / max(1e-6, train_sec),
        "tail_train_ce": float(statistics.fmean(tail_losses)),
        "learning_curve_bpb": curve,
        "composite_bpb": val_id["composite_bpb"],
        "domain_bpb": val_id["domain_bpb"],
        "ood2x_composite_bpb": val_ood2x["composite_bpb"],
        "peak_vram_mb": peak_mb,
    }
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return res


def run_exp_inv_06a_horizon(
    stream: CorpusStream,
    *,
    steps: int,
    seeds: list[int],
    batch_size: int = 16,
    seq_len: int = 256,
    checkpoint_cb=None,
) -> dict[str, Any]:
    suite = make_iso_param_suite("discovery_18m", vocab_size=stream.vocab_size)
    arms = (
        ("b0_transformer_adamw", suite[ParadigmKind.B0_TRANSFORMER], "adamw", 0.0),
        ("i3_hrwn_adamw", suite[ParadigmKind.I3_HRWN], "adamw", 0.0),
        ("g3_hmwr_adamw", suite[ParadigmKind.G3_HMWR], "adamw", 0.0),
        ("g3_hmwr_muon_ce", suite[ParadigmKind.G3_HMWR], "muon", 0.0),
    )
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpts = tuple(sorted({max(1, steps // 4), max(1, steps // 2), max(1, (3 * steps) // 4), steps}))
    domains = ("web", "textbook", "math", "code")
    out: dict[str, Any] = {}

    for arm_name, cfg, opt_kind, jep_w in arms:
        runs = [
            run_single_horizon_arm(
                cfg,
                stream,
                seed=sd,
                device=dev,
                optimizer_kind=opt_kind,
                jep_weight=jep_w,
                steps=steps,
                batch_size=batch_size,
                seq_len=seq_len,
                checkpoints=ckpts,
                eval_tokens_per_domain=65_536,
            )
            for sd in seeds
        ]

        agg = {
            "deployed_params": runs[0]["deployed_params"],
            "optimizer": opt_kind,
            "jep_weight": jep_w,
            "train_tokens_per_seed": runs[0]["train_tokens"],
            "tok_per_sec": _mean_std([r["tok_per_sec"] for r in runs]),
            "composite_bpb": _mean_std([r["composite_bpb"] for r in runs]),
            "ood2x_composite_bpb": _mean_std([r["ood2x_composite_bpb"] for r in runs]),
            "domain_bpb": {d: _mean_std([r["domain_bpb"][d] for r in runs]) for d in domains},
            "learning_curve_bpb": {
                str(c): _mean_std([r["learning_curve_bpb"][str(c)] for r in runs])["mean"] for c in ckpts
            },
            "peak_vram_mb": max(r["peak_vram_mb"] for r in runs),
            "per_seed": runs,
        }
        out[arm_name] = agg
        print(
            f"[CUDA-06A] {arm_name:22s} params={agg['deployed_params']:,} "
            f"compBPB={agg['composite_bpb']['mean']:.4f}±{agg['composite_bpb']['std']:.4f} "
            f"ood2x={agg['ood2x_composite_bpb']['mean']:.4f} tok/s={agg['tok_per_sec']['mean']:.0f}",
            flush=True,
        )
        if checkpoint_cb is not None:
            checkpoint_cb(out)
    return out


def run_exp_inv_06b_cuda_kv_wall(device: torch.device, vocab_size: int) -> dict[str, Any]:
    """Measure discrete 16 GB CUDA GPU VRAM + decode latency wall vs context length (512 -> 32k)."""
    if device.type != "cuda":
        return {"skipped": "CUDA device unavailable"}
    amp_dtype = _select_cuda_amp_dtype(device)
    probe_b0 = ArchitectureConfig(
        name="b0_probe_210m",
        paradigm=ParadigmKind.B0_TRANSFORMER,
        vocab_size=vocab_size,
        d_model=1024,
        n_layers=16,
        n_heads=16,
        n_kv_heads=8,
        head_dim=64,
        ffn_hidden=2816,
        max_seq_len=32_768,
        chunk_size=32,
        scam_slots=32,
    )
    probe_g3_base = replace(probe_b0, name="g3_probe_210m", paradigm=ParadigmKind.G3_HMWR)
    probe_g3 = retune_iso_param_variant(probe_g3_base, probe_b0, name="g3_probe_210m", align=32)

    results: dict[str, Any] = {}
    for label, cfg in (("b0_transformer_probe", probe_b0), ("g3_hmwr_probe", probe_g3)):
        census = compute_parameter_census(cfg)
        torch.cuda.empty_cache()
        gc.collect()
        with device:
            prev_dt = torch.get_default_dtype()
            torch.set_default_dtype(amp_dtype)
            try:
                model = NanoLMDiscoveryModel(cfg, skip_init_weights=True)
                with torch.no_grad():
                    for p in model.parameters():
                        if p.ndim >= 2:
                            p.normal_(mean=0.0, std=0.02)
            finally:
                torch.set_default_dtype(prev_dt)
        model.eval()
        ctx_rows: dict[str, Any] = {}
        for ctx in (512, 2048, 8192, 16384, 32768):
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                prompt = torch.randint(0, cfg.vocab_size, (1, ctx), device=device)
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
                    _sync_cuda(device)
                    t0 = time.perf_counter()
                    out = model(prompt, return_states=True)
                    _sync_cuda(device)
                    prefill_s = time.perf_counter() - t0
                    assert torch.isfinite(out.logits[:, -1:]).all()
                    states = out.states
                    tok = out.logits[:, -1:].argmax(dim=-1)
                    del out

                    dec_t: list[float] = []
                    for _ in range(10):
                        _sync_cuda(device)
                        ts = time.perf_counter()
                        sout = model.forward_step(tok, states=states)
                        _sync_cuda(device)
                        dec_t.append(time.perf_counter() - ts)
                        states = sout.states
                        tok = sout.logits[:, -1:].argmax(dim=-1)
                        del sout
                peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
                mean_dec = float(statistics.fmean(dec_t[2:]))
                ctx_rows[f"ctx_{ctx}"] = {
                    "status": "OK",
                    "peak_vram_mb": peak_mb,
                    "prefill_tok_per_sec": float(ctx / max(1e-6, prefill_s)),
                    "decode_ms_per_tok": mean_dec * 1000.0,
                    "decode_tok_per_sec": 1.0 / max(1e-6, mean_dec),
                }
                del prompt, states
            except torch.cuda.OutOfMemoryError:
                ctx_rows[f"ctx_{ctx}"] = {
                    "status": "OOM",
                    "peak_vram_mb": float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)),
                }
            finally:
                torch.cuda.empty_cache()
                gc.collect()
            print(f"[CUDA-06B] {label:22s} ctx={ctx:5d} -> {ctx_rows[f'ctx_{ctx}']}", flush=True)
        results[label] = {
            "deployed_params": model.deployed_parameter_count(),
            "state_mb_512_bf16": census.state_bytes_at_512_bf16 / (1024.0 * 1024.0),
            "state_mb_32768_bf16": census.state_bytes_at_32768_bf16 / (1024.0 * 1024.0),
            "contexts": ctx_rows,
        }
        del model
        torch.cuda.empty_cache()
        gc.collect()
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="EXP-INV-06 Kaggle/CUDA Extended Horizon Benchmark")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-json", default="kaggle_cuda_results.json")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 137])
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_dtype = _select_cuda_amp_dtype(device)
    gpu_names = (
        [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else ["cpu"]
    )
    vram_gb = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024.0**3), 2)
        if torch.cuda.is_available()
        else 0.0
    )
    stream = CorpusStream(args.data_dir)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema": "nanolm.discovery.kaggle_cuda.v1",
        "hardware": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": getattr(torch.version, "cuda", None),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_names": gpu_names,
            "gpu_vram_gb": vram_gb,
            "amp_dtype": str(amp_dtype).replace("torch.", ""),
            "paid_compute_usd": 0.0,
        },
        "corpus_manifest_digest": {
            "vocab": stream.vocab_size,
            "train_tokens_total": stream.manifest.get("train_tokens_total"),
            "train_sha256": stream.manifest.get("train_sha256"),
        },
    }

    def _save() -> None:
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("=== CUDA Cross-Hardware Invariant Audit ===", flush=True)
    report["cuda_invariant_audit"] = run_cuda_invariant_audit(device, amp_dtype)
    _save()

    print(f"=== EXP-INV-06A: 4x Token Horizon ({args.steps} steps, seeds={args.seeds}) ===", flush=True)
    report["exp_inv_06a_horizon_4x"] = run_exp_inv_06a_horizon(
        stream,
        steps=args.steps,
        seeds=args.seeds,
        checkpoint_cb=lambda d: (report.update({"exp_inv_06a_horizon_4x": d}), _save()),
    )
    _save()

    print("=== EXP-INV-06B: Discrete CUDA VRAM & Long-Context KV-Cache Wall (512 -> 32k) ===", flush=True)
    report["exp_inv_06b_cuda_kv_wall"] = run_exp_inv_06b_cuda_kv_wall(device, stream.vocab_size)
    _save()
    print(f"COMPLETED EXP-INV-06; wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
