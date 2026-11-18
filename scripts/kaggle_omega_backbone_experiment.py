#!/usr/bin/env python3
"""OMEGA-EXP-01: Kaggle CUDA Extended 4x-Horizon (1,000 steps) Backbone Tournament.

Compares parameter-matched (~20.55M) & honest-FLOP-accounted arms:
  - `x8_dual_unembed_atrd`: X8 Margin-ATRD reproducing dual-vocab-unembedding overhead + beta in (0, 1)
  - `g4_omega`: G3 tri-scale wave-metriplectic + QK-RMSNorm + Per-Head Gated-VRes + zero-vocab LatentTokenRefiner
  - `g4_no_vres`: Ablation without Gated-VRes (testing Layer-0 value highway on G4)
  - `g4_no_sparse_refiner`: Ablation without Pass-2 LatentTokenRefiner (testing sparse latent refinement on G4)
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanolm_discovery.eval_benchmarks import CorpusStream
from nanolm_omega.backbone import (
    OmegaBackboneConfig,
    OmegaVariant,
    build_omega_model,
    compute_omega_census,
    make_omega_suite,
)


def _mean_std(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    return {
        "mean": float(statistics.fmean(vals)),
        "std": float(statistics.stdev(vals)) if len(vals) > 1 else 0.0,
        "n": len(vals),
    }


def _select_amp_dtype(device: torch.device) -> torch.dtype:
    if device.type == "mps":
        return torch.bfloat16
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(device)
        return torch.bfloat16 if major >= 8 else torch.float32
    return torch.float32


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def run_cuda_omega_invariant_audit(device: torch.device, amp_dtype: torch.dtype) -> dict[str, Any]:
    mech_suite = make_omega_suite("mech")
    out: dict[str, Any] = {}
    for var in OmegaVariant:
        cfg = mech_suite[var]
        torch.manual_seed(11)
        m = build_omega_model(cfg).to(device).eval()
        c = compute_omega_census(cfg)
        assert m.deployed_parameter_count() == c.total_deployed_parameters
        x1 = torch.randint(0, cfg.vocab_size, (2, 40), device=device)
        x2 = x1.clone()
        x2[:, 20:] = (x2[:, 20:] + 7) % cfg.vocab_size
        with torch.no_grad():
            causality_err = float((m(x1).logits[:, :20] - m(x2).logits[:, :20]).abs().max().item())
            duality_err = None
            if var != OmegaVariant.X8_DUAL_UNEMBED_ATRD:
                par = m(x1).logits
                st = None
                seq = []
                for t in range(x1.shape[1]):
                    so = m.forward_step(x1[:, t], states=st)
                    seq.append(so.logits[:, 0])
                    st = so.states
                duality_err = float((par - torch.stack(seq, dim=1)).abs().max().item())
                assert duality_err < 1e-4, f"{var.value} duality_err={duality_err}"
        assert causality_err < 1e-5, f"{var.value} causality_err={causality_err}"
        m.train()
        y = torch.randint(0, cfg.vocab_size, (2, 40), device=device)
        use_amp = (device.type in ("cuda", "mps")) and (amp_dtype != torch.float32)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            mo = m(x1, targets=y)
        mo.loss.backward()
        grads_finite = all(bool(torch.isfinite(p.grad).all().item()) for p in m.parameters() if p.grad is not None)
        assert grads_finite
        out[var.value] = {
            "params": c.total_deployed_parameters,
            "unembed_passes": c.unembed_passes,
            "causality_max_err": causality_err,
            "duality_max_err": duality_err,
            "grads_finite": grads_finite,
        }
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def run_single_arm(
    cfg: OmegaBackboneConfig,
    stream: CorpusStream,
    *,
    seed: int,
    device: torch.device,
    steps: int,
    batch_size: int = 16,
    seq_len: int = 256,
    ckpt_out_dir: Path | None = None,
    eval_tokens_per_domain: int = 65_536,
) -> dict[str, Any]:
    amp_dtype = _select_amp_dtype(device)
    use_amp = (device.type in ("cuda", "mps")) and (amp_dtype != torch.float32)
    use_scaler = False
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = build_omega_model(cfg).to(device)
    census = compute_omega_census(cfg)
    base_lr = 8e-4
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.95), weight_decay=0.05)
    warmup = max(20, steps // 10)
    ckpt_steps = {max(1, steps // 4), max(1, steps // 2), max(1, (3 * steps) // 4), steps}
    curve: dict[str, float] = {}
    tail_losses: list[float] = []
    skipped_grad_steps = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    t0 = time.perf_counter()

    for step in range(steps):
        model.train()
        if step < warmup:
            lr_mult = float(step + 1) / float(warmup)
        else:
            prog = float(step - warmup) / float(max(1, steps - warmup))
            lr_mult = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog))
        for g in opt.param_groups:
            g["lr"] = base_lr * lr_mult

        x, y = stream.sample_train_batch(batch_size, seq_len, rng, device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(x, targets=y)
            loss = out.loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss on {cfg.name} seed={seed} step={step}: {loss.item()}")
        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grads_ok = bool(torch.isfinite(gnorm)) and all(
                bool(torch.isfinite(p.grad).all().item()) for p in model.parameters() if p.grad is not None
            )
            if not grads_ok:
                opt.zero_grad(set_to_none=True)
                skipped_grad_steps += 1
                scaler.update()
                continue
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grads_ok = bool(torch.isfinite(gnorm)) and all(
                bool(torch.isfinite(p.grad).all().item()) for p in model.parameters() if p.grad is not None
            )
            if not grads_ok:
                opt.zero_grad(set_to_none=True)
                skipped_grad_steps += 1
                continue
            opt.step()

        if step >= steps - 25:
            tail_losses.append(float(out.ce_loss.item()))

        h_step = step + 1
        if h_step in ckpt_steps and h_step < steps:
            v_mid = stream.evaluate_validation_bpb(
                model,
                device=device,
                seq_len=seq_len,
                eval_tokens_per_domain=eval_tokens_per_domain,
                batch_size=batch_size,
                use_bf16=use_amp,
                amp_dtype=amp_dtype,
            )
            curve[str(h_step)] = float(v_mid["composite_bpb"])

    _sync(device)
    train_s = time.perf_counter() - t0
    train_tokens = steps * batch_size * seq_len

    val_id = stream.evaluate_validation_bpb(
        model,
        device=device,
        seq_len=seq_len,
        eval_tokens_per_domain=eval_tokens_per_domain,
        batch_size=batch_size,
        use_bf16=use_amp,
        amp_dtype=amp_dtype,
    )
    curve[str(steps)] = float(val_id["composite_bpb"])
    val_ood2x = stream.evaluate_validation_bpb(
        model,
        device=device,
        seq_len=seq_len * 2,
        eval_tokens_per_domain=eval_tokens_per_domain,
        batch_size=max(4, batch_size // 2),
        use_bf16=use_amp,
        amp_dtype=amp_dtype,
    )
    peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)) if device.type == "cuda" else 0.0

    ckpt_sha = None
    if ckpt_out_dir is not None and cfg.variant == OmegaVariant.G4_OMEGA and seed == 42:
        ckpt_out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_file = ckpt_out_dir / "g4_omega_s42.pt"
        torch.save({"config": cfg.name, "seed": seed, "state_dict": model.state_dict()}, ckpt_file)
        ckpt_sha = hashlib.sha256(ckpt_file.read_bytes()).hexdigest()

    ref_tau = float(model.refiner.ema_tau.item()) if getattr(model, "refiner", None) is not None else None
    res = {
        "seed": seed,
        "device": str(device),
        "deployed_params": model.deployed_parameter_count(),
        "unembed_passes": census.unembed_passes,
        "true_flops_per_token_4k": census.true_flops_per_token_4k,
        "state_bytes_t32768_bf16": census.state_bytes_at_32768_bf16,
        "train_tokens": train_tokens,
        "train_seconds": train_s,
        "tok_per_sec": train_tokens / max(1e-6, train_s),
        "tail_train_ce": float(statistics.fmean(tail_losses)) if tail_losses else float(out.ce_loss.item()),
        "learning_curve_bpb": curve,
        "composite_bpb": val_id["composite_bpb"],
        "domain_bpb": val_id["domain_bpb"],
        "ood2x_composite_bpb": val_ood2x["composite_bpb"],
        "peak_vram_mb": peak_mb,
        "refiner_ema_tau": ref_tau,
        "skipped_grad_steps": skipped_grad_steps,
        "checkpoint_sha256": ckpt_sha,
    }
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return res


def main() -> int:
    parser = argparse.ArgumentParser(description="OMEGA-EXP-01 Kaggle CUDA Backbone Tournament")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-json", default="omega_kaggle_cuda_results.json")
    parser.add_argument("--ckpt-dir", default="/kaggle/working")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 137])
    parser.add_argument("--eval-tokens-per-domain", type=int, default=65_536)
    parser.add_argument("--variants", type=str, nargs="+", default=[v.value for v in OmegaVariant])
    parser.add_argument("--worker-variant", type=str, default=None)
    parser.add_argument("--worker-seed", type=int, default=None)
    parser.add_argument("--worker-device", type=str, default=None)
    parser.add_argument("--worker-out", type=str, default=None)
    args = parser.parse_args()

    stream = CorpusStream(args.data_dir)
    ckpt_dir = Path(args.ckpt_dir)
    suite = make_omega_suite("discovery_18m", vocab_size=stream.vocab_size)

    if args.worker_variant is not None:
        if args.worker_seed is None or args.worker_device is None or args.worker_out is None:
            raise ValueError("--worker-variant requires --worker-seed, --worker-device, and --worker-out")
        var = OmegaVariant(args.worker_variant)
        worker_dev = torch.device(args.worker_device)
        res = run_single_arm(
            suite[var],
            stream,
            seed=args.worker_seed,
            device=worker_dev,
            steps=args.steps,
            ckpt_out_dir=ckpt_dir,
            eval_tokens_per_domain=args.eval_tokens_per_domain,
        )
        out_file = Path(args.worker_out)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
        return 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_dtype = _select_amp_dtype(device)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema": "nanolm.omega.exp01_cuda.v1",
        "hardware": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": getattr(torch.version, "cuda", None),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ["cpu"],
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

    print("=== OMEGA-EXP-01 CUDA Invariant Audit ===", flush=True)
    report["cuda_invariant_audit"] = run_cuda_omega_invariant_audit(device, amp_dtype)
    _save()

    domains = ("web", "textbook", "math", "code")
    arms_out: dict[str, Any] = {}
    ckpts = tuple(sorted({max(1, args.steps // 4), max(1, args.steps // 2), max(1, (3 * args.steps) // 4), args.steps}))

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    for var in [OmegaVariant(v) for v in args.variants]:
        cfg = suite[var]
        if gpu_count >= 2 and len(args.seeds) >= 2:
            with tempfile.TemporaryDirectory() as tmp_dir:
                procs: list[tuple[int, str, Path, subprocess.Popen[bytes]]] = []
                for idx, sd in enumerate(args.seeds):
                    w_dev = f"cuda:{idx % gpu_count}"
                    w_out = Path(tmp_dir) / f"{var.value}_s{sd}.json"
                    cmd = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--data-dir",
                        args.data_dir,
                        "--ckpt-dir",
                        str(ckpt_dir),
                        "--steps",
                        str(args.steps),
                        "--eval-tokens-per-domain",
                        str(args.eval_tokens_per_domain),
                        "--worker-variant",
                        var.value,
                        "--worker-seed",
                        str(sd),
                        "--worker-device",
                        w_dev,
                        "--worker-out",
                        str(w_out),
                    ]
                    procs.append((sd, w_dev, w_out, subprocess.Popen(cmd, cwd=str(ROOT))))
                runs = []
                for sd, w_dev, w_out, proc in procs:
                    rc = proc.wait()
                    if rc != 0:
                        raise RuntimeError(f"Worker subprocess failed for {var.value} seed={sd} on {w_dev} (exit={rc})")
                    runs.append(json.loads(w_out.read_text(encoding="utf-8")))
        else:
            runs = [
                run_single_arm(
                    cfg,
                    stream,
                    seed=sd,
                    device=device,
                    steps=args.steps,
                    ckpt_out_dir=ckpt_dir,
                    eval_tokens_per_domain=args.eval_tokens_per_domain,
                )
                for sd in args.seeds
            ]
        agg = {
            "deployed_params": runs[0]["deployed_params"],
            "unembed_passes": runs[0]["unembed_passes"],
            "true_flops_per_token_4k": runs[0]["true_flops_per_token_4k"],
            "state_bytes_t32768_bf16": runs[0]["state_bytes_t32768_bf16"],
            "train_tokens_per_seed": runs[0]["train_tokens"],
            "tok_per_sec": _mean_std([r["tok_per_sec"] for r in runs]),
            "composite_bpb": _mean_std([r["composite_bpb"] for r in runs]),
            "ood2x_composite_bpb": _mean_std([r["ood2x_composite_bpb"] for r in runs]),
            "domain_bpb": {d: _mean_std([r["domain_bpb"][d] for r in runs]) for d in domains},
            "learning_curve_bpb": {str(c): _mean_std([r["learning_curve_bpb"][str(c)] for r in runs])["mean"] for c in ckpts},
            "peak_vram_mb": max(r["peak_vram_mb"] for r in runs),
            "checkpoint_sha256": next((r["checkpoint_sha256"] for r in runs if r["checkpoint_sha256"]), None),
            "per_seed": runs,
        }
        arms_out[var.value] = agg
        report["omega_exp_01_corpus"] = arms_out
        _save()
        print(
            f"[OMEGA-01-CUDA] {var.value:22s} params={agg['deployed_params']:,} "
            f"flops={agg['true_flops_per_token_4k']/1e6:.2f}M "
            f"compBPB={agg['composite_bpb']['mean']:.4f}±{agg['composite_bpb']['std']:.4f} "
            f"ood2x={agg['ood2x_composite_bpb']['mean']:.4f} tok/s={agg['tok_per_sec']['mean']:.0f}",
            flush=True,
        )

    print(f"COMPLETED OMEGA-EXP-01 CUDA; wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
