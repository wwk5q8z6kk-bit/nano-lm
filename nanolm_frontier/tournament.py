"""RunPod/Local Experiment Runner for E1 (5-Architecture Tournament) and E2 (15-Dim Foundation Evolution)."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import gc
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
from torch.nn import functional as F

from nanolm_frontier.architectures import (
    ArchitectureFamily,
    FrontierArchConfig,
    FrontierCausalLM,
    build_micro_tournament_configs,
)
from nanolm_frontier.benchmark_15d import (
    CAPABILITY_DIMENSIONS,
    BenchmarkDim,
    BenchmarkTask,
    bootstrap_accuracy_ci,
    build_benchmark_15d_suite,
    compute_efficiency_score,
    verify_benchmark_response,
)
from nanolm_frontier.curriculum import (
    FrontierMicroTokenizer,
    build_decontaminated_curriculum,
    build_ood_recall_probes,
    build_pretrain_tournament_datasets,
    verify_zero_contamination,
)
from nanolm_frontier.surgery_and_rlvr import (
    DexDiffAttentionWrapper,
    GRPOConfig,
    ThalamocorticalVAnchor,
    apply_chat_template_safe,
    apply_frontier_surgery,
    grpo_verifier_posttrain,
    train_sft_curriculum,
)
from nanolm_posttrain.architecture import ABSOLUTE_MAX_PARAMETERS, ParameterBudgetExceededError
from nanolm_posttrain.lora import LoRAConfig, apply_lora, lora_state_dict, trainable_parameter_summary


def _select_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _reset_peak_mem(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_mem_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return round(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0), 2)
    return 0.0


def _autocast_ctx(device: torch.device) -> Any:
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _eval_bpb(
    model: FrontierCausalLM,
    seqs: list[list[int]],
    bytes_per_tok: float,
    device: torch.device,
    *,
    batch_size: int = 16,
    active_loops: int | None = None,
    adaptive_entropy_halt: bool = False,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_loops = 0.0
    total_exit = 0.0
    n_batches = 0
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = torch.tensor(seqs[i : i + batch_size], dtype=torch.long, device=device)
            x = batch[:, :-1]
            y = batch[:, 1:]
            with _autocast_ctx(device):
                out = model(
                    x,
                    active_loops=active_loops,
                    adaptive_entropy_halt=adaptive_entropy_halt,
                    return_loop_stats=True,
                )
            logits, stats = out
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
            total_loss += float(loss.item())
            total_tokens += int(y.numel())
            total_loops += float(stats["mean_loops"])
            total_exit += float(stats["early_exit_rate"])
            n_batches += 1
    mean_nats = total_loss / max(total_tokens, 1)
    bpb = mean_nats / (max(bytes_per_tok, 1e-6) * math.log(2.0))
    return {
        "bpb": round(bpb, 4),
        "loss_nats": round(mean_nats, 4),
        "mean_loops": round(total_loops / max(n_batches, 1), 3),
        "early_exit_rate": round(total_exit / max(n_batches, 1), 3),
    }


def _eval_ood_recall(
    model: FrontierCausalLM,
    probes: dict[int, list[tuple[list[int], int]]],
    device: torch.device,
    *,
    active_loops: int | None = None,
    adaptive_entropy_halt: bool = False,
) -> dict[str, float]:
    model.eval()
    acc_map: dict[str, float] = {}
    with torch.no_grad():
        for seq_len, items in probes.items():
            # Process in mini-batches for speed
            hits = 0
            for i in range(0, len(items), 16):
                chunk = items[i : i + 16]
                prompts = torch.tensor([c[0] for c in chunk], dtype=torch.long, device=device)
                targets = torch.tensor([c[1] for c in chunk], dtype=torch.long, device=device)
                with _autocast_ctx(device):
                    logits = model(
                        prompts,
                        active_loops=active_loops,
                        adaptive_entropy_halt=adaptive_entropy_halt,
                    )
                preds = logits[:, -1, :].argmax(dim=-1)
                hits += int((preds == targets).sum().item())
            acc_map[f"len_{seq_len}"] = round(hits / max(len(items), 1), 4)
    return acc_map


def run_e1_architecture_tournament(
    *,
    device: torch.device | None = None,
    train_sequences: int = 2_400,
    train_steps: int = 700,
    batch_size: int = 16,
    lr: float = 8e-4,
    seed: int = 42,
    configs: tuple[FrontierArchConfig, ...] | None = None,
    vocab_size: int = 8_192,
    ood_lengths: tuple[int, ...] = (256, 512, 1024),
    ood_probes_per_length: int = 48,
    max_val_batches: int | None = None,
) -> dict[str, Any]:
    """Run Experiment E1: Controlled Iso-Param / Iso-Token Tournament across C0..C4."""
    dev = device or _select_device()
    tok = FrontierMicroTokenizer(vocab_size=vocab_size)
    configs_to_run = configs or build_micro_tournament_configs(vocab_size=tok.vocab_size, max_seq_len=2_048)
    data = build_pretrain_tournament_datasets(tok, train_sequences=train_sequences, seq_len=256, seed=1337)
    train_seqs = data["train_seqs"]
    val_splits = {
        k: (v[0][: max_val_batches * batch_size] if max_val_batches else v[0], v[1])
        for k, v in data["val_splits"].items()
    }
    ood_probes = build_ood_recall_probes(tok, lengths=ood_lengths, probes_per_length=ood_probes_per_length, seed=9991)

    results: list[dict[str, Any]] = []
    for cfg in configs_to_run:
        torch.manual_seed(seed)
        model = FrontierCausalLM(cfg).to(dev)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(train_steps, 1))

        _reset_peak_mem(dev)
        _sync_device(dev)
        t0 = time.perf_counter()
        n_train = len(train_seqs)
        tokens_processed = 0
        final_losses: list[float] = []

        model.train()
        for step in range(train_steps):
            start_idx = (step * batch_size) % n_train
            if start_idx + batch_size <= n_train:
                raw_batch = train_seqs[start_idx : start_idx + batch_size]
            else:
                raw_batch = train_seqs[start_idx:] + train_seqs[: (start_idx + batch_size) % n_train]
            batch = torch.tensor(raw_batch, dtype=torch.long, device=dev)
            x, y = batch[:, :-1], batch[:, 1:]
            optimizer.zero_grad(set_to_none=True)
            with _autocast_ctx(dev):
                logits = model(x)
                loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            tokens_processed += int(y.numel())
            if step >= train_steps - 20:
                final_losses.append(float(loss.detach().cpu().item()))

        _sync_device(dev)
        elapsed_s = max(time.perf_counter() - t0, 1e-6)
        peak_mb = _peak_mem_mb(dev)

        val_metrics: dict[str, Any] = {}
        bpb_vals: list[float] = []
        for dom, (vseqs, vbpt) in val_splits.items():
            res = _eval_bpb(model, vseqs, vbpt, dev, batch_size=batch_size)
            val_metrics[dom] = res
            bpb_vals.append(res["bpb"])
        mean_val_bpb = round(sum(bpb_vals) / len(bpb_vals), 4)
        ood_recall = _eval_ood_recall(model, ood_probes, dev)

        entry: dict[str, Any] = {
            "config": cfg.parameter_breakdown(),
            "train_steps": train_steps,
            "tokens_processed": tokens_processed,
            "train_seconds": round(elapsed_s, 2),
            "train_tokens_per_sec": round(tokens_processed / elapsed_s, 1),
            "peak_vram_mb": peak_mb,
            "final_train_loss": round(sum(final_losses) / max(len(final_losses), 1), 4),
            "mean_val_bpb": mean_val_bpb,
            "domain_val": val_metrics,
            "ood_recall": ood_recall,
        }

        # For C4_JCLE_Frontier, also measure test-time compute scaling r=1, 2, 4 and entropy halt
        if cfg.uses_jcle:
            sweep: list[dict[str, Any]] = []
            for r_val, halt_flag in ((1, False), (2, False), (4, False), (4, True)):
                t_eval = time.perf_counter()
                v_assoc = _eval_bpb(
                    model,
                    val_splits["assoc"][0],
                    val_splits["assoc"][1],
                    dev,
                    active_loops=r_val,
                    adaptive_entropy_halt=halt_flag,
                )
                v_math = _eval_bpb(
                    model,
                    val_splits["math"][0],
                    val_splits["math"][1],
                    dev,
                    active_loops=r_val,
                    adaptive_entropy_halt=halt_flag,
                )
                ood_r = _eval_ood_recall(
                    model,
                    ood_probes,
                    dev,
                    active_loops=r_val,
                    adaptive_entropy_halt=halt_flag,
                )
                _sync_device(dev)
                dt = max(time.perf_counter() - t_eval, 1e-6)
                sweep.append(
                    {
                        "active_loops": r_val,
                        "adaptive_entropy_halt": halt_flag,
                        "assoc_bpb": v_assoc["bpb"],
                        "math_bpb": v_math["bpb"],
                        "mean_loops_used": v_assoc["mean_loops"],
                        "early_exit_rate": v_assoc["early_exit_rate"],
                        "ood_recall": ood_r,
                        "eval_seconds": round(dt, 3),
                        "active_flops_per_token_2k": cfg.active_flops_per_token(2048, loops=r_val),
                    }
                )
            entry["jcle_test_time_scaling"] = sweep

        print(
            f"[E1] {cfg.name:24s} bpb={mean_val_bpb:.4f} ood={ood_recall} "
            f"tok/s={entry['train_tokens_per_sec']:.0f} vram={peak_mb:.1f}MB",
            flush=True,
        )
        results.append(entry)
        del model, optimizer, scheduler
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "experiment": "E1_architecture_tournament",
        "device": str(dev),
        "vocab_size": tok.vocab_size,
        "train_sequences": train_sequences,
        "train_steps": train_steps,
        "arms": results,
    }


def _evaluate_model_15d(
    model: Any,
    tokenizer: Any,
    tasks: tuple[BenchmarkTask, ...],
    device: torch.device,
    *,
    arm_name: str,
) -> dict[str, Any]:
    model.eval()
    _reset_peak_mem(device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    total_params = sum(p.numel() for p in model.parameters())
    if total_params > ABSOLUTE_MAX_PARAMETERS:
        raise ParameterBudgetExceededError(f"{arm_name} has {total_params:,} > 4B")

    # Estimate KV cache footprint at 4k tokens from HF config
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "num_hidden_layers"):
        n_l = int(cfg.num_hidden_layers)
        n_kv = int(getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads", 8)))
        d_h = int(getattr(cfg, "head_dim", int(cfg.hidden_size) // int(cfg.num_attention_heads)))
        kv_bytes_per_tok = 2 * n_l * n_kv * d_h * 2
    else:
        kv_bytes_per_tok = 0
    kv_mb_4k = (kv_bytes_per_tok * 4096) / (1024.0 * 1024.0)

    per_task: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    total_gen_tokens = 0
    total_gen_seconds = 0.0

    for t in tasks:
        msgs = [{"role": "system", "content": t.system_prompt}, {"role": "user", "content": t.user_prompt}]
        prompt_text = apply_chat_template_safe(tokenizer, msgs, add_generation_prompt=True)
        enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)
        prompt_len = int(enc["input_ids"].shape[1])

        _sync_device(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out_ids = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=t.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )
        _sync_device(device)
        dt = max(time.perf_counter() - t0, 1e-6)
        gen_ids = out_ids[0, prompt_len:]
        total_gen_tokens += int(gen_ids.numel())
        total_gen_seconds += dt
        latencies_ms.append(dt * 1000.0)

        reply = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        verdict = verify_benchmark_response(t, reply)
        per_task.append(
            {
                "task_id": t.task_id,
                "dim": t.dim.value,
                "difficulty": t.difficulty,
                "passed": bool(verdict.passed),
                "score": float(verdict.score),
                "reason": verdict.reason,
                "gen_tokens": int(gen_ids.numel()),
                "latency_ms": round(dt * 1000.0, 2),
                "response_preview": reply[:200],
            }
        )

    peak_mb = _peak_mem_mb(device)
    lat_sorted = sorted(latencies_ms)
    p50_ms = lat_sorted[len(lat_sorted) // 2] if lat_sorted else 0.0
    p95_ms = lat_sorted[min(len(lat_sorted) - 1, int(0.95 * len(lat_sorted)))] if lat_sorted else 0.0
    tps = total_gen_tokens / max(total_gen_seconds, 1e-6)

    outcomes = [1.0 if r["passed"] else 0.0 for r in per_task]
    mean_acc, ci_lo, ci_hi = bootstrap_accuracy_ci(outcomes)
    mean_partial = round(sum(r["score"] for r in per_task) / max(len(per_task), 1), 4)

    by_dim: dict[str, dict[str, Any]] = {}
    for d in CAPABILITY_DIMENSIONS:
        sub = [r for r in per_task if r["dim"] == d.value]
        passed_n = sum(1 for r in sub if r["passed"])
        by_dim[d.value] = {
            "passed": passed_n,
            "total": len(sub),
            "accuracy": round(passed_n / max(len(sub), 1), 4),
            "mean_score": round(sum(r["score"] for r in sub) / max(len(sub), 1), 4),
        }

    eff = compute_efficiency_score(
        capability_accuracy=mean_acc,
        total_parameters=total_params,
        tokens_per_sec=tps,
        kv_cache_mb_at_4k=kv_mb_4k,
    )
    by_dim[BenchmarkDim.INFERENCE_EFFICIENCY.value] = {
        **eff,
        "p50_ms": round(p50_ms, 2),
        "p95_ms": round(p95_ms, 2),
        "peak_vram_mb": peak_mb,
        "total_parameters": total_params,
    }

    by_diff: dict[str, dict[str, Any]] = {}
    for diff_name in ("easy", "medium", "hard"):
        sub = [r for r in per_task if r["difficulty"] == diff_name]
        p = sum(1 for r in sub if r["passed"])
        by_diff[diff_name] = {"passed": p, "total": len(sub), "accuracy": round(p / max(len(sub), 1), 4)}

    failures = Counter(r["reason"] for r in per_task if not r["passed"])
    return {
        "arm": arm_name,
        "total_parameters": total_params,
        "passed": int(sum(outcomes)),
        "total_tasks": len(per_task),
        "accuracy": mean_acc,
        "bootstrap_ci95": [ci_lo, ci_hi],
        "mean_partial_score": mean_partial,
        "by_dimension": by_dim,
        "by_difficulty": by_diff,
        "failure_taxonomy": dict(failures.most_common(12)),
        "tasks": per_task,
    }


def _load_hf_model_and_tokenizer(model_id: str, device: torch.device) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if device.type == "cuda" else (torch.float16 if device.type == "mps" else torch.float32)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
    total = sum(p.numel() for p in model.parameters())
    if total > ABSOLUTE_MAX_PARAMETERS:
        raise ParameterBudgetExceededError(f"{model_id} has {total:,} > 4B limit")
    return model, tok


def _apply_and_verify_step0_surgery(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    lora_cfg: LoRAConfig,
) -> dict[str, Any]:
    """Attach surgical adapters and verify zero-init identity preserves exact step-0 logits."""
    sample = tokenizer("def add(a, b):\n    return", return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad():
        base_logits = model(**sample).logits.float().clone()
    surgery_info = apply_frontier_surgery(
        model,
        lora_config=lora_cfg,
        enable_dex_diff=True,
        enable_v_anchor=True,
    )
    model.eval()
    with torch.no_grad():
        graft_logits = model(**sample).logits.float()
    max_diff = float((base_logits - graft_logits).abs().max().item())
    if max_diff > 1e-4:
        raise RuntimeError(f"Step-0 graft equivalence violated: max_abs_diff={max_diff}")
    surgery_info["step0_max_abs_logit_diff"] = max_diff
    return surgery_info


def run_e2_foundation_evolution(
    *,
    model_id: str = "Qwen/Qwen3-1.7B",
    device: torch.device | None = None,
    sft_epochs: int = 5,
    sft_lr: float = 2e-4,
    grpo_rounds: int = 3,
    grpo_group_size: int = 4,
    enable_thinking_disabled: bool = True,
) -> dict[str, Any]:
    """Run Experiment E2 across Base, Standard LoRA SFT, and Frontier (DexDiff + VAnchor + SFT + GRPO)."""
    del enable_thinking_disabled  # Applied uniformly in apply_chat_template_safe
    dev = device or _select_device()
    eval_tasks = build_benchmark_15d_suite()
    curriculum = build_decontaminated_curriculum(seed=2026)
    decontam = verify_zero_contamination(curriculum, eval_tasks)

    arms_out: list[dict[str, Any]] = []

    # Arm 0: Base Model
    model0, tok = _load_hf_model_and_tokenizer(model_id, dev)
    arm0 = _evaluate_model_15d(model0, tok, eval_tasks, dev, arm_name="Base")
    print(f"[E2] Base passed={arm0['passed']}/{arm0['total_tasks']} acc={arm0['accuracy']:.4f}", flush=True)
    arms_out.append(arm0)
    del model0
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # Arm 1: Standard LoRA SFT
    model1, tok = _load_hf_model_and_tokenizer(model_id, dev)
    lora_cfg = LoRAConfig(rank=16, alpha=32.0)
    apply_lora(model1, lora_cfg)
    arm1_summary = trainable_parameter_summary(model1)
    sft1_stats = train_sft_curriculum(model1, tok, curriculum, device=dev, epochs=sft_epochs, lr=sft_lr)
    arm1 = _evaluate_model_15d(model1, tok, eval_tasks, dev, arm_name="LoRA_SFT")
    arm1["training_stats"] = {"parameter_summary": arm1_summary, "sft": sft1_stats}
    print(f"[E2] LoRA_SFT passed={arm1['passed']}/{arm1['total_tasks']} acc={arm1['accuracy']:.4f}", flush=True)
    arms_out.append(arm1)
    del model1
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # Arm 2: NanoLM Frontier (DexDiff + ThalamocorticalVAnchor + LoRA + SFT + Verifier-Guided GRPO)
    model2, tok = _load_hf_model_and_tokenizer(model_id, dev)
    surgery_info = _apply_and_verify_step0_surgery(model2, tok, dev, lora_cfg)

    sft2_stats = train_sft_curriculum(model2, tok, curriculum, device=dev, epochs=sft_epochs, lr=sft_lr)
    # Focus GRPO on algorithmic/reasoning/structured/tool/planning/ood/memory curriculum subset for high signal
    rl_dims = {
        BenchmarkDim.CODING,
        BenchmarkDim.MATH_LOGIC,
        BenchmarkDim.STRUCTURED_JSON,
        BenchmarkDim.TOOL_USE,
        BenchmarkDim.PLANNING_DECOMPOSITION,
        BenchmarkDim.OOD_ROBUSTNESS,
        BenchmarkDim.MEMORY_RETENTION,
        BenchmarkDim.INSTRUCTION_FOLLOWING,
        BenchmarkDim.HALLUCINATION_RESISTANCE,
    }
    rl_subset = [ex for ex in curriculum if ex.dim in rl_dims][:24]
    grpo_stats = grpo_verifier_posttrain(
        model2,
        tok,
        rl_subset,
        device=dev,
        config=GRPOConfig(rounds=grpo_rounds, group_size=grpo_group_size, max_new_tokens=192),
    )

    # Measure learned non-zero magnitude of surgical gates
    diff_norms = [float(m.gamma.detach().abs().mean().cpu().item()) for m in model2.modules() if isinstance(m, DexDiffAttentionWrapper)]
    anchor_norms = [float(m.anchor_gate.detach().abs().mean().cpu().item()) for m in model2.modules() if isinstance(m, ThalamocorticalVAnchor)]

    arm2 = _evaluate_model_15d(model2, tok, eval_tasks, dev, arm_name="Frontier_DexDiff_SFT_GRPO")
    arm2["training_stats"] = {
        "surgery": surgery_info,
        "sft": sft2_stats,
        "grpo": grpo_stats,
        "mean_abs_dex_diff_gamma": round(sum(diff_norms) / max(len(diff_norms), 1), 6),
        "mean_abs_v_anchor_gate": round(sum(anchor_norms) / max(len(anchor_norms), 1), 6),
    }
    adapter_count = len(lora_state_dict(model2))
    arm2["training_stats"]["trainable_tensor_count"] = adapter_count
    print(
        f"[E2] Frontier_DexDiff_SFT_GRPO passed={arm2['passed']}/{arm2['total_tasks']} "
        f"acc={arm2['accuracy']:.4f} step0_diff={surgery_info['step0_max_abs_logit_diff']:.2e}",
        flush=True,
    )
    arms_out.append(arm2)
    del model2
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "experiment": "E2_foundation_15d_evolution",
        "model_id": model_id,
        "device": str(dev),
        "decontamination": decontam,
        "arms": arms_out,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="NanoLM Frontier E1 + E2 Tournament CLI")
    parser.add_argument("--mode", choices=("all", "e1", "e2"), default="all")
    parser.add_argument("--out-dir", type=str, default="artifacts/nanolm_frontier_v1")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--e1-steps", type=int, default=700)
    parser.add_argument("--e1-sequences", type=int, default=2_400)
    parser.add_argument("--sft-epochs", type=int, default=5)
    parser.add_argument("--grpo-rounds", type=int, default=3)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = _select_device()
    meta: dict[str, Any] = {
        "device": str(dev),
        "cuda_name": torch.cuda.get_device_name(0) if dev.type == "cuda" else "non-cuda",
        "torch_version": torch.__version__,
    }

    payload: dict[str, Any] = {"meta": meta}
    if args.mode in ("all", "e1"):
        e1_res = run_e1_architecture_tournament(
            device=dev,
            train_sequences=args.e1_sequences,
            train_steps=args.e1_steps,
        )
        payload["e1"] = e1_res
        (out_dir / "e1_tournament.json").write_text(json.dumps(e1_res, indent=2), encoding="utf-8")

    if args.mode in ("all", "e2"):
        e2_res = run_e2_foundation_evolution(
            model_id=args.model_id,
            device=dev,
            sft_epochs=args.sft_epochs,
            grpo_rounds=args.grpo_rounds,
        )
        payload["e2"] = e2_res
        (out_dir / "e2_foundation_15d.json").write_text(json.dumps(e2_res, indent=2), encoding="utf-8")

    (out_dir / "frontier_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"FRONTIER_TOURNAMENT_COMPLETE out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
