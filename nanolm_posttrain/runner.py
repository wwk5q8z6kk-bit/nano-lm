#!/usr/bin/env python3
"""Evaluate and LoRA post-train a ``<= 4B`` NanoLM foundation checkpoint across 6 domains.

Evaluates deterministic capability verifiers across:
- ``coding`` (isolated AST + unit assertion execution)
- ``reasoning`` (multi-step mathematical & logical exact answers)
- ``summarization`` (strict word budget, key-fact recall, zero hallucinated numbers)
- ``structured_extraction`` (strict JSON schema & value conservation)
- ``diagrams_and_charts`` (Mermaid syntax/quoted-label/topology + quantitative chart series conservation)
- ``long_context`` (multi-needle synthesis across 30-paragraph documents)

When ``--adapt-epochs > 0``, attaches pure-PyTorch LoRA adapters (enforcing the
4B parameter ceiling), trains on the disjoint ``build_foundation_train_curriculum()``
examples with prompt-masked SFT, and re-evaluates on the held-out suite.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch

from nanolm_posttrain.architecture import ABSOLUTE_MAX_PARAMETERS, ParameterBudgetExceededError
from nanolm_posttrain.eval_suite import (
    FoundationTask,
    build_foundation_eval_suite,
    build_foundation_train_curriculum,
    verify_foundation_response,
)
from nanolm_posttrain.lora import (
    LoRAConfig,
    apply_lora,
    build_chat_sft_tensors,
    lora_state_dict,
    train_masked_sft_step,
    trainable_parameter_summary,
)


def _select_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def generate_chat_reply(
    model: Any,
    tokenizer: Any,
    *,
    system_prompt: str,
    user_prompt: str,
    device: str,
    max_new_tokens: int = 256,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    model.eval()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    cont = output[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(cont, skip_special_tokens=True).strip()


def evaluate_suite(
    model: Any,
    tokenizer: Any,
    tasks: tuple[FoundationTask, ...],
    *,
    device: str,
    max_new_tokens: int = 256,
) -> dict[str, Any]:
    started = time.perf_counter()
    by_domain: dict[str, list[float]] = defaultdict(list)
    by_domain_pass: dict[str, list[int]] = defaultdict(list)
    records: list[dict[str, Any]] = []

    for task in tasks:
        reply = generate_chat_reply(
            model,
            tokenizer,
            system_prompt=task.system_prompt,
            user_prompt=task.user_prompt,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        verdict = verify_foundation_response(task, reply)
        dom = task.domain.value
        by_domain[dom].append(verdict.score)
        by_domain_pass[dom].append(1 if verdict.passed else 0)
        records.append(
            {
                "task_id": task.task_id,
                "domain": dom,
                "passed": verdict.passed,
                "score": verdict.score,
                "reason": verdict.reason,
                "response": reply,
            }
        )

    total_passed = sum(1 for r in records if r["passed"])
    mean_score = sum(r["score"] for r in records) / max(len(records), 1)
    domain_summary = {
        dom: {
            "pass_rate": round(sum(by_domain_pass[dom]) / len(by_domain_pass[dom]), 4),
            "mean_score": round(sum(by_domain[dom]) / len(by_domain[dom]), 4),
            "tasks": len(by_domain[dom]),
        }
        for dom in sorted(by_domain)
    }
    return {
        "task_count": len(records),
        "passed_count": total_passed,
        "pass_rate": round(total_passed / max(len(records), 1), 4),
        "mean_score": round(mean_score, 4),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "by_domain": domain_summary,
        "tasks": records,
    }


def run_foundation_experiment(
    checkpoint_dir: Path,
    *,
    adapt_epochs: int = 6,
    lora_rank: int = 16,
    learning_rate: float = 2e-4,
    seed: int = 42,
    save_lora_path: Path | None = None,
) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    resolved = checkpoint_dir.resolve(strict=True)
    device, dtype = _select_device()
    tokenizer = AutoTokenizer.from_pretrained(str(resolved), local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        str(resolved),
        local_files_only=True,
        trust_remote_code=False,
        dtype=dtype,
    ).to(device)
    model.eval()

    base_params = sum(p.numel() for p in model.parameters())
    if base_params > ABSOLUTE_MAX_PARAMETERS:
        raise ParameterBudgetExceededError(
            f"checkpoint has {base_params:,} parameters > {ABSOLUTE_MAX_PARAMETERS:,}"
        )

    eval_tasks = build_foundation_eval_suite()
    baseline_eval = evaluate_suite(model, tokenizer, eval_tasks, device=device)

    adaptation_info: dict[str, Any] | None = None
    adapted_eval: dict[str, Any] | None = None

    if adapt_epochs > 0:
        curriculum = build_foundation_train_curriculum()
        wrapped_layers = apply_lora(model, LoRAConfig(rank=lora_rank, alpha=float(lora_rank * 2)))
        param_info = trainable_parameter_summary(model)
        triples = [(t.system_prompt, t.user_prompt, t.reference_response) for t in curriculum]
        batches = build_chat_sft_tensors(tokenizer, triples, device=device)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)

        epoch_losses: list[float] = []
        train_start = time.perf_counter()
        for _ in range(adapt_epochs):
            step_losses = [
                train_masked_sft_step(model, optimizer, inp_ids, lbl_ids)
                for inp_ids, lbl_ids in batches
            ]
            epoch_losses.append(round(sum(step_losses) / len(step_losses), 6))
        train_seconds = round(time.perf_counter() - train_start, 3)

        if save_lora_path is not None:
            save_lora_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(lora_state_dict(model), save_lora_path)

        adapted_eval = evaluate_suite(model, tokenizer, eval_tasks, device=device)
        adaptation_info = {
            "curriculum_size": len(curriculum),
            "epochs": adapt_epochs,
            "lora_rank": lora_rank,
            "learning_rate": learning_rate,
            "wrapped_linear_layers": wrapped_layers,
            **param_info,
            "epoch_losses": epoch_losses,
            "train_seconds": train_seconds,
        }

    cfg_bytes = (resolved / "config.json").read_bytes() if (resolved / "config.json").is_file() else b""
    return {
        "schema_version": "nanolm.foundation_posttrain_eval.v1",
        "checkpoint_id": resolved.name,
        "config_sha256": hashlib.sha256(cfg_bytes).hexdigest() if cfg_bytes else None,
        "device": device,
        "dtype": str(dtype),
        "base_parameter_count": base_params,
        "within_2b_4b_ceiling": sum(p.numel() for p in model.parameters()) <= ABSOLUTE_MAX_PARAMETERS,
        "baseline_evaluation": baseline_eval,
        "lora_adaptation": adaptation_info,
        "adapted_evaluation": adapted_eval,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Local <= 4B HF checkpoint snapshot")
    parser.add_argument("--adapt-epochs", type=int, default=6)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args(argv)

    result = run_foundation_experiment(
        args.checkpoint,
        adapt_epochs=args.adapt_epochs,
        lora_rank=args.lora_rank,
        learning_rate=args.lr,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
