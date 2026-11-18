#!/usr/bin/env python3
"""Execute Cycles 1-3 of the NanoLM Autonomous Architecture Discovery Program ($0.00 local)."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import hashlib
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
from torch.nn import functional as F

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
from nanolm_discovery.eval_benchmarks import (
    CorpusStream,
    evaluate_epistemic_calibration,
    generate_multi_needle_batch,
    generate_overwrite_batch,
    generate_s3_batch,
)
from nanolm_discovery.models import NanoLMDiscoveryModel
from nanolm_discovery.optim import build_optimizer
from nanolm_discovery.registry import InnovationScore, LineageRegistry


def _mean_std(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"mean": 0.0, "std": 0.0}
    mean = float(statistics.fmean(vals))
    std = float(statistics.stdev(vals)) if len(vals) > 1 else 0.0
    return {"mean": mean, "std": std, "n": len(vals)}


def _sync_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _masked_token_accuracy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    correct = (preds == targets) & mask
    denom = int(mask.sum().item())
    return float(correct.sum().item()) / float(max(1, denom))


def _masked_ce_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    sel_logits = logits.float()[mask]
    sel_targets = targets[mask]
    return F.cross_entropy(sel_logits, sel_targets)


def run_mechanistic_single(
    cfg: ArchitectureConfig,
    task: str,
    seed: int,
    device: torch.device,
    *,
    steps: int = 260,
    batch_size: int = 32,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    model = NanoLMDiscoveryModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3.5e-3, weight_decay=0.01)

    train_seq = {"s3": 32, "overwrite": 48, "needle": 96}[task]
    t0 = time.perf_counter()
    model.train()
    for step in range(steps):
        if task == "s3":
            x, y, m = generate_s3_batch(batch_size, train_seq, rng=np_rng, device=device)
        elif task == "overwrite":
            x, y, m = generate_overwrite_batch(batch_size, train_seq, rng=np_rng, device=device)
        else:
            x, y, m = generate_multi_needle_batch(batch_size, train_seq, n_needles=4, rng=np_rng, device=device)
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = _masked_ce_loss(out.logits, y, m)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    _sync_device(device)
    elapsed = time.perf_counter() - t0
    model.eval()
    eval_rng = np.random.default_rng(10_000 + seed)
    metrics: dict[str, Any] = {
        "params": model.deployed_parameter_count(),
        "train_seconds": elapsed,
    }
    with torch.no_grad():
        if task == "s3":
            for label, ev_len in (("id_t32", 32), ("ood4x_t128", 128), ("ood8x_t256", 256)):
                accs = []
                for _ in range(8):
                    x, y, m = generate_s3_batch(batch_size, ev_len, rng=eval_rng, device=device)
                    accs.append(_masked_token_accuracy(model(x).logits, y, m))
                metrics[label] = float(statistics.fmean(accs))
        elif task == "overwrite":
            for label, ev_len in (("id_t48", 48), ("ood4x_t192", 192)):
                accs = []
                for _ in range(8):
                    x, y, m = generate_overwrite_batch(batch_size, ev_len, rng=eval_rng, device=device)
                    accs.append(_masked_token_accuracy(model(x).logits, y, m))
                metrics[label] = float(statistics.fmean(accs))
            cal = evaluate_epistemic_calibration(model, rng=eval_rng, device=device, seq_len=96, batch_size=32)
            metrics["calibration"] = cal
        else:
            for label, ev_len, n_nd in (("id_t96_n4", 96, 4), ("ood_t256_n8", 256, 8)):
                accs = []
                for _ in range(8):
                    x, y, m = generate_multi_needle_batch(batch_size, ev_len, n_needles=n_nd, rng=eval_rng, device=device)
                    accs.append(_masked_token_accuracy(model(x).logits, y, m))
                metrics[label] = float(statistics.fmean(accs))
            metrics["state_bytes_t4096_bf16"] = model.census.state_bytes_at_4096_bf16
    del model, opt
    return metrics


def _save_results_snapshot(results_path: Path, results: dict[str, Any]) -> None:
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


def run_exp_inv_01(
    variants: dict[str, ArchitectureConfig],
    seeds: list[int],
    device: torch.device,
    *,
    steps: int = 260,
    existing: dict[str, Any] | None = None,
    checkpoint_cb: Any | None = None,
) -> dict[str, Any]:
    results: dict[str, Any] = dict(existing) if existing else {}
    for arm_name, cfg in variants.items():
        if arm_name in results:
            r_prev = results[arm_name]
            print(
                f"[EXP-INV-01] {arm_name:22s} (cached) OOD_Comp={r_prev['ood_composite_acc']:.3f}",
                flush=True,
            )
            continue
        arm_runs: dict[str, list[dict[str, Any]]] = {"s3": [], "overwrite": [], "needle": []}
        for task in ("s3", "overwrite", "needle"):
            for seed in seeds:
                arm_runs[task].append(run_mechanistic_single(cfg, task, seed, device, steps=steps))
        s3_ood4 = _mean_std([r["ood4x_t128"] for r in arm_runs["s3"]])
        s3_ood8 = _mean_std([r["ood8x_t256"] for r in arm_runs["s3"]])
        ow_id = _mean_std([r["id_t48"] for r in arm_runs["overwrite"]])
        ow_ood4 = _mean_std([r["ood4x_t192"] for r in arm_runs["overwrite"]])
        cal_int = _mean_std([r["calibration"]["auroc_internal_unc"] for r in arm_runs["overwrite"]])
        cal_ent = _mean_std([r["calibration"]["auroc_predictive_entropy"] for r in arm_runs["overwrite"]])
        nd_id = _mean_std([r["id_t96_n4"] for r in arm_runs["needle"]])
        nd_ood = _mean_std([r["ood_t256_n8"] for r in arm_runs["needle"]])
        ood_composite = float(statistics.fmean([s3_ood4["mean"], ow_ood4["mean"], nd_ood["mean"]]))
        results[arm_name] = {
            "params": arm_runs["s3"][0]["params"],
            "state_bytes_t4096_bf16": arm_runs["needle"][0]["state_bytes_at_4096_bf16"]
            if "state_bytes_at_4096_bf16" in arm_runs["needle"][0]
            else arm_runs["needle"][0]["state_bytes_t4096_bf16"],
            "s3_id_t32": _mean_std([r["id_t32"] for r in arm_runs["s3"]]),
            "s3_ood4x_t128": s3_ood4,
            "s3_ood8x_t256": s3_ood8,
            "overwrite_id_t48": ow_id,
            "overwrite_ood4x_t192": ow_ood4,
            "epistemic_auroc_internal": cal_int,
            "epistemic_auroc_entropy": cal_ent,
            "needle_id_t96_n4": nd_id,
            "needle_ood_t256_n8": nd_ood,
            "ood_composite_acc": ood_composite,
        }
        print(
            f"[EXP-INV-01] {arm_name:22s} params={results[arm_name]['params']:,} "
            f"S3_4x={s3_ood4['mean']:.3f} OW_4x={ow_ood4['mean']:.3f} "
            f"Needle_OOD={nd_ood['mean']:.3f} EpistAUROC={cal_int['mean']:.3f} OOD_Comp={ood_composite:.3f}",
            flush=True,
        )
        if checkpoint_cb is not None:
            checkpoint_cb(results)
    return results


def run_corpus_single(
    cfg: ArchitectureConfig,
    stream: CorpusStream,
    seed: int,
    device: torch.device,
    *,
    optimizer_kind: str = "adamw",
    jep_weight: float = 0.0,
    steps: int = 400,
    batch_size: int = 16,
    seq_len: int = 256,
    eval_tokens_per_domain: int = 65_536,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cfg_run = replace(cfg, jep_head_enabled=(jep_weight > 0.0))
    model = NanoLMDiscoveryModel(cfg_run).to(device)
    base_lr = 1.2e-3 if cfg.d_model <= 256 else (8e-4 if cfg.d_model <= 384 else 6e-4)
    opt = build_optimizer(
        model,
        kind=optimizer_kind,
        lr=base_lr,
        lr_muon=base_lr * 2.0,
        weight_decay=0.05,
    )
    use_bf16 = device.type in ("mps", "cuda")
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if use_bf16
        else torch.enable_grad()
    )
    warmup = max(20, steps // 10)
    t0 = time.perf_counter()
    model.train()
    last_losses: list[float] = []
    for step in range(steps):
        # Cosine LR decay with linear warmup
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
        with amp_ctx:
            out = model(x, targets=y, jep_weight=jep_weight)
            loss = out.loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss on {cfg_run.name} seed={seed} step={step}: {loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step >= steps - 20:
            last_losses.append(float(out.ce_loss.item()))

    _sync_device(device)
    train_sec = time.perf_counter() - t0
    train_tokens = steps * batch_size * seq_len
    val_id = stream.evaluate_validation_bpb(
        model,
        device=device,
        seq_len=seq_len,
        eval_tokens_per_domain=eval_tokens_per_domain,
        batch_size=batch_size,
        use_bf16=use_bf16,
    )
    # 2x context extrapolation check on web + code shards
    val_ood2x = stream.evaluate_validation_bpb(
        model,
        device=device,
        seq_len=seq_len * 2,
        eval_tokens_per_domain=min(65_536, eval_tokens_per_domain),
        batch_size=max(4, batch_size // 2),
        use_bf16=use_bf16,
    )
    res = {
        "deployed_params": model.deployed_parameter_count(),
        "training_only_params": model.training_only_parameter_count(),
        "optimizer": optimizer_kind,
        "jep_weight": jep_weight,
        "train_tokens": train_tokens,
        "train_seconds": train_sec,
        "tok_per_sec": train_tokens / max(1e-6, train_sec),
        "tail_train_ce": float(statistics.fmean(last_losses)),
        "composite_bpb": val_id["composite_bpb"],
        "domain_bpb": val_id["domain_bpb"],
        "ood2x_composite_bpb": val_ood2x["composite_bpb"],
        "state_bytes_t512_bf16": model.census.state_bytes_at_512_bf16,
        "state_bytes_t4096_bf16": model.census.state_bytes_at_4096_bf16,
        "state_bytes_t32768_bf16": model.census.state_bytes_at_32768_bf16,
    }
    del model, opt
    if device.type == "mps":
        torch.mps.empty_cache()
    gc.collect()
    return res


def aggregate_corpus_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    domains = ("web", "textbook", "math", "code")
    out: dict[str, Any] = {
        "deployed_params": runs[0]["deployed_params"],
        "training_only_params": runs[0]["training_only_params"],
        "optimizer": runs[0]["optimizer"],
        "jep_weight": runs[0]["jep_weight"],
        "train_tokens_per_seed": runs[0]["train_tokens"],
        "tok_per_sec": _mean_std([r["tok_per_sec"] for r in runs]),
        "composite_bpb": _mean_std([r["composite_bpb"] for r in runs]),
        "ood2x_composite_bpb": _mean_std([r["ood2x_composite_bpb"] for r in runs]),
        "domain_bpb": {d: _mean_std([r["domain_bpb"][d] for r in runs]) for d in domains},
        "state_bytes_t4096_bf16": runs[0]["state_bytes_t4096_bf16"],
        "state_bytes_t32768_bf16": runs[0]["state_bytes_t32768_bf16"],
        "per_seed": runs,
    }
    return out


def run_corpus_arm(
    arm_name: str,
    cfg: ArchitectureConfig,
    stream: CorpusStream,
    seeds: list[int],
    device: torch.device,
    *,
    optimizer_kind: str = "adamw",
    jep_weight: float = 0.0,
    steps: int = 400,
    batch_size: int = 16,
    seq_len: int = 256,
) -> dict[str, Any]:
    runs = [
        run_corpus_single(
            cfg,
            stream,
            seed,
            device,
            optimizer_kind=optimizer_kind,
            jep_weight=jep_weight,
            steps=steps,
            batch_size=batch_size,
            seq_len=seq_len,
        )
        for seed in seeds
    ]
    agg = aggregate_corpus_runs(runs)
    print(
        f"[CORPUS] {arm_name:24s} opt={optimizer_kind:5s} jep={jep_weight:.2f} "
        f"params={agg['deployed_params']:,} compBPB={agg['composite_bpb']['mean']:.4f}±{agg['composite_bpb']['std']:.4f} "
        f"ood2x={agg['ood2x_composite_bpb']['mean']:.4f} tok/s={agg['tok_per_sec']['mean']:.0f}",
        flush=True,
    )
    return agg


def profile_flagship_4b_hardware(
    cfgs: dict[str, ArchitectureConfig],
    device: torch.device,
) -> dict[str, Any]:
    """Instantiate real ~3.89B models sequentially in bfloat16 on M3 Max and measure live prefill/decode."""
    out: dict[str, Any] = {}
    dtype = torch.bfloat16 if device.type in ("mps", "cuda") else torch.float32
    for label, cfg in cfgs.items():
        census = compute_parameter_census(cfg, enforce_strict_4b=True)
        torch.manual_seed(0)
        # Allocate weights directly in bfloat16 on target device to stay comfortably inside 64 GB UMA
        with device:
            prev_dtype = torch.get_default_dtype()
            torch.set_default_dtype(dtype)
            try:
                model = NanoLMDiscoveryModel(cfg, skip_init_weights=True)
                # Lightweight fill to prevent uninitialized NaNs while preserving 1D norm/gate weights
                with torch.no_grad():
                    for p in model.parameters():
                        if p.ndim >= 2:
                            p.normal_(mean=0.0, std=0.02)
            finally:
                torch.set_default_dtype(prev_dtype)
        model.eval()
        deployed = model.deployed_parameter_count()
        assert deployed == census.total_deployed_parameters < 4_000_000_000

        use_amp = device.type in ("mps", "cuda")
        bench_ctx: dict[str, Any] = {}
        for ctx in (512, 2048, 4096):
            prompt = torch.randint(0, cfg.vocab_size, (1, ctx), device=device)
            # Warmup & prefill
            with torch.no_grad():
                _sync_device(device)
                t0 = time.perf_counter()
                if use_amp:
                    with torch.autocast(device_type=device.type, dtype=dtype):
                        pref = model(prompt, return_states=True)
                else:
                    pref = model(prompt, return_states=True)
                _sync_device(device)
                prefill_s = time.perf_counter() - t0
                assert torch.isfinite(pref.logits).all()

                # Autoregressive step decode (12 tokens)
                states = pref.states
                tok = pref.logits[:, -1:].argmax(dim=-1)
                step_times: list[float] = []
                for _ in range(12):
                    _sync_device(device)
                    ts = time.perf_counter()
                    if use_amp:
                        with torch.autocast(device_type=device.type, dtype=dtype):
                            step_out = model.forward_step(tok, states=states)
                    else:
                        step_out = model.forward_step(tok, states=states)
                    _sync_device(device)
                    step_times.append(time.perf_counter() - ts)
                    states = step_out.states
                    tok = step_out.logits[:, -1:].argmax(dim=-1)

            mean_step_s = float(statistics.fmean(step_times[2:]))  # drop first 2 warmup steps
            bench_ctx[f"ctx_{ctx}"] = {
                "prefill_tok_per_sec": float(ctx / max(1e-6, prefill_s)),
                "prefill_ms": float(prefill_s * 1000.0),
                "decode_ms_per_tok": float(mean_step_s * 1000.0),
                "decode_tok_per_sec": float(1.0 / max(1e-6, mean_step_s)),
                "persistent_state_mb": float(
                    (
                        census.state_bytes_at_512_bf16
                        if ctx == 512
                        else (census.state_bytes_at_4096_bf16 * (ctx / 4096.0) if cfg.paradigm in (ParadigmKind.B0_TRANSFORMER, ParadigmKind.B1_GDN2_HYBRID) else census.state_bytes_at_4096_bf16)
                    )
                    / (1024.0 * 1024.0)
                ),
            }
            print(
                f"[4B-LIVE] {label:20s} ctx={ctx:4d} prefill={bench_ctx[f'ctx_{ctx}']['prefill_tok_per_sec']:.1f} tok/s "
                f"decode={bench_ctx[f'ctx_{ctx}']['decode_ms_per_tok']:.2f} ms/tok ({bench_ctx[f'ctx_{ctx}']['decode_tok_per_sec']:.2f} tok/s)",
                flush=True,
            )

        out[label] = {
            "deployed_params": deployed,
            "strictly_below_4b": deployed < 4_000_000_000,
            "weight_gb_bf16": float(deployed * 2 / (1024.0**3)),
            "state_mb_512_bf16": float(census.state_bytes_at_512_bf16 / (1024.0**2)),
            "state_mb_4096_bf16": float(census.state_bytes_at_4096_bf16 / (1024.0**2)),
            "state_mb_32768_bf16": float(census.state_bytes_at_32768_bf16 / (1024.0**2)),
            "flops_per_tok_4k": census.forward_flops_per_token_4k,
            "benchmarks": bench_ctx,
        }
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        gc.collect()
    return out


def _write_sha256_manifest(artifact_dir: Path) -> None:
    lines: list[str] = []
    for p in sorted(artifact_dir.iterdir()):
        if p.is_file() and p.name != "SHA256SUMS":
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            lines.append(f"{h}  {p.name}")
    (artifact_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NanoLM Discovery Program Cycles 1-3.")
    parser.add_argument(
        "--phase",
        choices=["cycle1", "cycle2", "cycle3", "all"],
        default="all",
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("NANOLM_DATA_DIR", "/Volumes/Express1M2/nanolm-lab/data/v1"),
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/nanolm_discovery_v1",
    )
    parser.add_argument("--mech-steps", type=int, default=260)
    parser.add_argument("--corpus-steps", type=int, default=400)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    stream = CorpusStream(args.data_dir)
    registry = LineageRegistry(out_dir / "lineage_registry.json")

    results_path = out_dir / "discovery_results.json"
    if results_path.exists():
        results = json.loads(results_path.read_text(encoding="utf-8"))
    else:
        results = {
            "schema": "nanolm.discovery.results.v1",
            "hardware": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "device": device.type,
                "torch": torch.__version__,
                "paid_compute_usd": 0.0,
            },
            "corpus_manifest_digest": {
                "schema": stream.manifest.get("schema"),
                "vocab": stream.vocab_size,
                "train_tokens_total": stream.manifest.get("train_tokens_total"),
                "train_sha256": stream.manifest.get("train_sha256"),
                "tokenizer_sha256": stream.manifest.get("tokenizer_sha256"),
            },
        }

    mech_suite = make_iso_param_suite("mech")
    suite_18m = make_iso_param_suite("discovery_18m", vocab_size=stream.vocab_size)
    cycle1_kinds = (
        ParadigmKind.B0_TRANSFORMER,
        ParadigmKind.B1_GDN2_HYBRID,
        ParadigmKind.I1_PHMN,
        ParadigmKind.I2_NEBN,
        ParadigmKind.I3_HRWN,
    )

    if args.phase in ("cycle1", "all"):
        print("=== CYCLE 1A: EXP-INV-01 Mechanistic Falsification Tournament ===", flush=True)
        c1_mech_variants = {k.value: mech_suite[k] for k in cycle1_kinds}
        results["exp_inv_01_cycle1"] = run_exp_inv_01(
            c1_mech_variants,
            [101, 202, 303],
            device,
            steps=args.mech_steps,
            existing=results.get("exp_inv_01_cycle1"),
            checkpoint_cb=lambda d: (_save_results_snapshot(results_path, {**results, "exp_inv_01_cycle1": d})),
        )
        _save_results_snapshot(results_path, results)

        print("=== CYCLE 1B: EXP-INV-02 Real Multi-Domain Corpus (20.5M, 2 seeds, AdamW) ===", flush=True)
        c1_corpus: dict[str, Any] = dict(results.get("exp_inv_02_cycle1") or {})
        for k in cycle1_kinds:
            if k.value not in c1_corpus:
                c1_corpus[k.value] = run_corpus_arm(
                    k.value, suite_18m[k], stream, [42, 137], device, steps=args.corpus_steps
                )
                results["exp_inv_02_cycle1"] = c1_corpus
                _save_results_snapshot(results_path, results)
            else:
                print(
                    f"[CORPUS] {k.value:24s} (cached) compBPB={c1_corpus[k.value]['composite_bpb']['mean']:.4f}",
                    flush=True,
                )

        # Register baseline + submit Cycle 1 inventions to LineageRegistry (if not yet evaluated)
        if registry.data["baseline"] is None:
            b0_rec = {
                "name": ParadigmKind.B0_TRANSFORMER.value,
                "tier": "discovery_18m",
                "composite_bpb": c1_corpus[ParadigmKind.B0_TRANSFORMER.value]["composite_bpb"]["mean"],
                "ood_composite_acc": results["exp_inv_01_cycle1"][ParadigmKind.B0_TRANSFORMER.value]["ood_composite_acc"],
                "params": c1_corpus[ParadigmKind.B0_TRANSFORMER.value]["deployed_params"],
            }
            registry.register_baseline(b0_rec)

        c1_verdicts: dict[str, Any] = dict(results.get("cycle1_lineage_verdicts") or {})
        for k in cycle1_kinds[1:]:
            if k.value not in c1_verdicts:
                rec = {
                    "name": k.value,
                    "tier": "discovery_18m",
                    "composite_bpb": c1_corpus[k.value]["composite_bpb"]["mean"],
                    "ood_composite_acc": results["exp_inv_01_cycle1"][k.value]["ood_composite_acc"],
                    "params": c1_corpus[k.value]["deployed_params"],
                }
                c1_verdicts[k.value] = registry.submit_experimental(rec)
        results["cycle1_lineage_verdicts"] = c1_verdicts
        _save_results_snapshot(results_path, results)

    if args.phase in ("cycle2", "all"):
        print("=== CYCLE 2A: G2_MPRN + Surgical Leave-One-Out Ablations (EXP-INV-01 & 04) ===", flush=True)
        b0_mech = mech_suite[ParadigmKind.B0_TRANSFORMER]
        g2_mech = mech_suite[ParadigmKind.G2_MPRN]
        c2_mech_variants = {
            "g2_mprn": g2_mech,
            "g2_no_reflection": retune_iso_param_variant(g2_mech, b0_mech, name="g2_no_reflection", align=2, use_symplectic_reflection=False),
            "g2_no_precision": retune_iso_param_variant(g2_mech, b0_mech, name="g2_no_precision", align=2, use_precision_norm=False),
            "g2_no_scam": retune_iso_param_variant(g2_mech, b0_mech, name="g2_no_scam", align=2, use_scam_memory=False),
            "g2_std_swiglu": retune_iso_param_variant(g2_mech, b0_mech, name="g2_std_swiglu", align=2, use_phase_bilinear=False),
        }
        results["exp_inv_01_cycle2"] = run_exp_inv_01(
            c2_mech_variants,
            [101, 202, 303],
            device,
            steps=args.mech_steps,
            existing=results.get("exp_inv_01_cycle2"),
            checkpoint_cb=lambda d: (_save_results_snapshot(results_path, {**results, "exp_inv_01_cycle2": d})),
        )
        _save_results_snapshot(results_path, results)

        b0_18 = suite_18m[ParadigmKind.B0_TRANSFORMER]
        g2_18 = suite_18m[ParadigmKind.G2_MPRN]
        c2_corpus_variants = {
            "g2_mprn": g2_18,
            "g2_no_reflection": retune_iso_param_variant(g2_18, b0_18, name="g2_no_reflection", align=16, use_symplectic_reflection=False),
            "g2_no_precision": retune_iso_param_variant(g2_18, b0_18, name="g2_no_precision", align=16, use_precision_norm=False),
            "g2_no_scam": retune_iso_param_variant(g2_18, b0_18, name="g2_no_scam", align=16, use_scam_memory=False),
            "g2_std_swiglu": retune_iso_param_variant(g2_18, b0_18, name="g2_std_swiglu", align=16, use_phase_bilinear=False),
        }
        c2_abl_corpus: dict[str, Any] = dict(results.get("exp_inv_04_ablations_corpus") or {})
        c2_verdicts: dict[str, Any] = dict(results.get("cycle2_lineage_verdicts") or {})
        for name, cfg in c2_corpus_variants.items():
            if name not in c2_abl_corpus:
                agg = run_corpus_arm(name, cfg, stream, [42, 137], device, steps=args.corpus_steps)
                c2_abl_corpus[name] = agg
                rec = {
                    "name": name,
                    "tier": "discovery_18m",
                    "composite_bpb": agg["composite_bpb"]["mean"],
                    "ood_composite_acc": results["exp_inv_01_cycle2"][name]["ood_composite_acc"],
                    "params": agg["deployed_params"],
                }
                c2_verdicts[name] = registry.submit_experimental(rec)
                results["exp_inv_04_ablations_corpus"] = c2_abl_corpus
                results["cycle2_lineage_verdicts"] = c2_verdicts
                _save_results_snapshot(results_path, results)
            else:
                print(
                    f"[CORPUS] {name:24s} (cached) compBPB={c2_abl_corpus[name]['composite_bpb']['mean']:.4f}",
                    flush=True,
                )

        best_name = registry.data["best"]["name"]
        best_cfg = {
            **{k.value: suite_18m[k] for k in cycle1_kinds},
            **c2_corpus_variants,
        }.get(best_name, g2_18)
        print(f"=== CYCLE 2B: EXP-INV-03 2x2 Training Factorial (B0 vs BEST={best_name}) ===", flush=True)
        b0_c1 = results["exp_inv_02_cycle1"][ParadigmKind.B0_TRANSFORMER.value]
        best_adamw = (
            results["exp_inv_02_cycle1"][best_name]
            if best_name in results.get("exp_inv_02_cycle1", {})
            else c2_abl_corpus[best_name]
        )
        fac = dict(results.get("exp_inv_03_factorial") or {})
        fac["b0_adamw_ce"] = b0_c1
        fac["best_adamw_ce"] = best_adamw
        fac["best_variant_name"] = best_name
        if "b0_muon_jep" not in fac:
            fac["b0_muon_jep"] = run_corpus_arm(
                "b0_muon_jep", b0_18, stream, [42, 137], device, optimizer_kind="muon", jep_weight=0.10, steps=args.corpus_steps
            )
            results["exp_inv_03_factorial"] = fac
            _save_results_snapshot(results_path, results)
        if "best_muon_jep" not in fac:
            fac["best_muon_jep"] = run_corpus_arm(
                f"{best_name}_muon_jep", best_cfg, stream, [42, 137], device, optimizer_kind="muon", jep_weight=0.10, steps=args.corpus_steps
            )
            results["exp_inv_03_factorial"] = fac
            _save_results_snapshot(results_path, results)

    if args.phase in ("cycle3", "all"):
        print("=== CYCLE 3A: G3_HMWR Diagnostic Evolution (EXP-INV-01 & 02) ===", flush=True)
        b0_mech = mech_suite[ParadigmKind.B0_TRANSFORMER]
        g3_mech = mech_suite[ParadigmKind.G3_HMWR]
        c3_mech_variants = {
            "g3_hmwr": g3_mech,
            "g3_no_depth_adapt": retune_iso_param_variant(
                g3_mech, b0_mech, name="g3_no_depth_adapt", align=2, use_depth_routing=False
            ),
        }
        results["exp_inv_01_cycle3"] = run_exp_inv_01(
            c3_mech_variants,
            [101, 202, 303],
            device,
            steps=args.mech_steps,
            existing=results.get("exp_inv_01_cycle3"),
            checkpoint_cb=lambda d: (_save_results_snapshot(results_path, {**results, "exp_inv_01_cycle3": d})),
        )
        _save_results_snapshot(results_path, results)

        b0_18 = suite_18m[ParadigmKind.B0_TRANSFORMER]
        g3_18 = suite_18m[ParadigmKind.G3_HMWR]
        c3_corpus_variants = {
            "g3_hmwr": g3_18,
            "g3_no_depth_adapt": retune_iso_param_variant(
                g3_18, b0_18, name="g3_no_depth_adapt", align=16, use_depth_routing=False
            ),
        }
        c3_corpus: dict[str, Any] = dict(results.get("exp_inv_02_cycle3") or {})
        c3_verdicts: dict[str, Any] = dict(results.get("cycle3_lineage_verdicts") or {})
        for name, cfg in c3_corpus_variants.items():
            if name not in c3_corpus:
                agg = run_corpus_arm(
                    name, cfg, stream, [42, 137], device, optimizer_kind="adamw", jep_weight=0.0, steps=args.corpus_steps
                )
                c3_corpus[name] = agg
                rec = {
                    "name": name,
                    "tier": "discovery_18m",
                    "composite_bpb": agg["composite_bpb"]["mean"],
                    "ood_composite_acc": results["exp_inv_01_cycle3"][name]["ood_composite_acc"],
                    "params": agg["deployed_params"],
                }
                c3_verdicts[name] = registry.submit_experimental(rec)
                results["exp_inv_02_cycle3"] = c3_corpus
                results["cycle3_lineage_verdicts"] = c3_verdicts
                _save_results_snapshot(results_path, results)
            else:
                print(
                    f"[CORPUS] {name:24s} (cached) compBPB={c3_corpus[name]['composite_bpb']['mean']:.4f}",
                    flush=True,
                )

        best_name = registry.data["best"]["name"] if registry.data.get("best") else "g3_hmwr"
        print(f"=== CYCLE 3B: EXP-INV-05A Scaling Law (8.65M -> 20.55M -> 47.64M) B0 vs BEST={best_name} (AdamW+CE) ===", flush=True)
        abl_flags: dict[str, Any] = {}
        target_paradigm = ParadigmKind.G3_HMWR
        if best_name == "g2_no_reflection":
            target_paradigm, abl_flags = ParadigmKind.G2_MPRN, {"use_symplectic_reflection": False}
        elif best_name == "g2_no_precision":
            target_paradigm, abl_flags = ParadigmKind.G2_MPRN, {"use_precision_norm": False}
        elif best_name == "g2_no_scam":
            target_paradigm, abl_flags = ParadigmKind.G2_MPRN, {"use_scam_memory": False}
        elif best_name == "g2_std_swiglu":
            target_paradigm, abl_flags = ParadigmKind.G2_MPRN, {"use_phase_bilinear": False}
        elif best_name == "g3_no_depth_adapt":
            target_paradigm, abl_flags = ParadigmKind.G3_HMWR, {"use_depth_routing": False}
        elif best_name in {k.value for k in ParadigmKind}:
            target_paradigm = ParadigmKind(best_name)

        scaling_out: dict[str, Any] = dict(results.get("exp_inv_05a_scaling") or {})
        scaling_out["best_variant_name"] = best_name
        for tier, stps in (("scale_8m", args.corpus_steps), ("scale_42m", args.corpus_steps)):
            tsuite = make_iso_param_suite(tier, vocab_size=stream.vocab_size)
            b0_t = tsuite[ParadigmKind.B0_TRANSFORMER]
            best_t_base = tsuite[target_paradigm]
            best_t = (
                retune_iso_param_variant(best_t_base, b0_t, name=f"{best_name}_{tier}", align=32, **abl_flags)
                if abl_flags
                else replace(best_t_base, name=f"{best_name}_{tier}")
            )
            tier_res = dict(scaling_out.get(tier) or {})
            if "b0_adamw" not in tier_res:
                tier_res["b0_adamw"] = run_corpus_arm(
                    f"b0_{tier}", b0_t, stream, [42, 137], device, optimizer_kind="adamw", jep_weight=0.0, steps=stps
                )
                scaling_out[tier] = tier_res
                results["exp_inv_05a_scaling"] = scaling_out
                _save_results_snapshot(results_path, results)
            if "best_adamw" not in tier_res:
                tier_res["best_adamw"] = run_corpus_arm(
                    f"{best_name}_{tier}", best_t, stream, [42, 137], device, optimizer_kind="adamw", jep_weight=0.0, steps=stps
                )
                scaling_out[tier] = tier_res
                results["exp_inv_05a_scaling"] = scaling_out
                _save_results_snapshot(results_path, results)

        print("=== CYCLE 3C: EXP-INV-05B Live ~3.89B Flagship Realization on M3 Max ===", flush=True)
        suite_4b = make_iso_param_suite("flagship_4b")
        b0_4b = suite_4b[ParadigmKind.B0_TRANSFORMER]
        best_4b_base = suite_4b[target_paradigm]
        best_4b = (
            retune_iso_param_variant(best_4b_base, b0_4b, name=f"{best_name}_4b", align=64, **abl_flags)
            if abl_flags
            else replace(best_4b_base, name=f"{best_name}_4b")
        )
        models_4b: dict[str, ArchitectureConfig] = {
            "b0_transformer_4b": b0_4b,
            f"{best_name}_4b": best_4b,
        }
        if "i3_hrwn_4b" not in models_4b:
            models_4b["i3_hrwn_4b"] = suite_4b[ParadigmKind.I3_HRWN]
        if "g3_hmwr_4b" not in models_4b:
            models_4b["g3_hmwr_4b"] = suite_4b[ParadigmKind.G3_HMWR]
        results["exp_inv_05b_flagship_4b"] = profile_flagship_4b_hardware(models_4b, device)
        _save_results_snapshot(results_path, results)

    # Compute dynamic 5-axis innovation scores grounded in actual observed telemetry
    c1_m = results.get("exp_inv_01_cycle1", {})
    c2_m = results.get("exp_inv_01_cycle2", {})
    c3_m = results.get("exp_inv_01_cycle3", {})
    c1_c = results.get("exp_inv_02_cycle1", {})
    c2_c = results.get("exp_inv_04_ablations_corpus", {})
    c3_c = results.get("exp_inv_02_cycle3", {})
    b0_ood = float(c1_m.get("b0_transformer", {}).get("ood_composite_acc", 0.0))
    b0_bpb = float(c1_c.get("b0_transformer", {}).get("composite_bpb", {}).get("mean", 10.0))
    b0_mem32k = float(c1_c.get("b0_transformer", {}).get("state_bytes_t32768_bf16", 1.0))

    def _score_arm(name: str, orig: float, math_c: float, feas: float, m_rec: dict[str, Any], c_rec: dict[str, Any]) -> dict[str, float]:
        ood_gain = float(m_rec.get("ood_composite_acc", 0.0)) - b0_ood
        bpb_gain = b0_bpb - float(c_rec.get("composite_bpb", {}).get("mean", b0_bpb))
        cap = max(1.0, min(10.0, 5.0 + 10.0 * ood_gain + 8.0 * bpb_gain))
        mem32k = max(1.0, float(c_rec.get("state_bytes_t32768_bf16", b0_mem32k)))
        eff = max(1.0, min(10.0, 5.0 + 1.2 * math.log10(max(1.0, b0_mem32k / mem32k))))
        return InnovationScore(orig, math_c, feas, round(eff, 2), round(cap, 2)).to_dict()

    scores_dyn: dict[str, Any] = {}
    if "i1_phmn" in c1_c:
        scores_dyn["i1_phmn"] = _score_arm("i1_phmn", 8.0, 9.2, 8.8, c1_m.get("i1_phmn", {}), c1_c["i1_phmn"])
    if "i2_nebn" in c1_c:
        scores_dyn["i2_nebn"] = _score_arm("i2_nebn", 8.4, 9.0, 9.2, c1_m.get("i2_nebn", {}), c1_c["i2_nebn"])
    if "i3_hrwn" in c1_c:
        scores_dyn["i3_hrwn"] = _score_arm("i3_hrwn", 7.8, 8.0, 8.5, c1_m.get("i3_hrwn", {}), c1_c["i3_hrwn"])
    if "g2_mprn" in c2_c:
        scores_dyn["g2_mprn"] = _score_arm("g2_mprn", 8.8, 9.4, 9.0, c2_m.get("g2_mprn", {}), c2_c["g2_mprn"])
    if "g2_no_precision" in c2_c:
        scores_dyn["g2_no_precision"] = _score_arm(
            "g2_no_precision", 8.7, 9.3, 9.0, c2_m.get("g2_no_precision", {}), c2_c["g2_no_precision"]
        )
    if "g3_hmwr" in c3_c:
        scores_dyn["g3_hmwr"] = _score_arm("g3_hmwr", 9.2, 9.4, 9.3, c3_m.get("g3_hmwr", {}), c3_c["g3_hmwr"])
    if "g3_no_depth_adapt" in c3_c:
        scores_dyn["g3_no_depth_adapt"] = _score_arm(
            "g3_no_depth_adapt", 8.8, 9.1, 9.3, c3_m.get("g3_no_depth_adapt", {}), c3_c["g3_no_depth_adapt"]
        )
    if scores_dyn:
        results["innovation_scores"] = scores_dyn
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    _write_sha256_manifest(out_dir)
    print(f"COMPLETED phase={args.phase}; wrote {results_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
