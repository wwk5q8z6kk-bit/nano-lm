#!/usr/bin/env python3
"""Minimal 50-step QLoRA canary trainer for RunPod pod (stdout loss logging)."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

LOSS_MARKERS = (1, 10, 25, 50)


def _format_prompt(instruction: str, user_input: str) -> str:
    if user_input.strip():
        return f"### Instruction:\n{instruction}\n\n### Input:\n{user_input}\n\n### Response:\n"
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-url", required=True)
    parser.add_argument("--output-dir", default="/workspace/qlora_canary_adapter")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--log-file", default="/workspace/qlora_canary.log")
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    if not torch.cuda.is_available():
        print("MODEL_LOAD_FAIL: cuda unavailable", flush=True)
        return 1

    print(f"MODEL_LOAD_START base={args.base_model}", flush=True)
    raw = urllib.request.urlopen(args.dataset_url, timeout=120).read()
    rows = json.loads(raw)
    texts = []
    for row in rows:
        prompt = _format_prompt(row["instruction"], row.get("input", ""))
        texts.append({"text": prompt + row["output"]})

    ds = Dataset.from_list(texts)
    print(f"DATASET_OK rows={len(ds)}", flush=True)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    print("MODEL_LOADED", flush=True)

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    print("QLORA_INIT_OK", flush=True)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(batch["text"], truncation=True, max_length=2048)

    tokenized = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_file)

    class LossLogger(TrainerCallback):
        def __init__(self) -> None:
            self.step_losses: dict[int, float] = {}

        def on_init_end(self, args, state, control, **kwargs):  # noqa: ANN001
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
            if not logs or "loss" not in logs:
                return
            step = int(state.global_step)
            loss = float(logs["loss"])
            self.step_losses[step] = loss
            line = f"STEP_LOSS step={step} loss={loss:.6f}"
            print(line, flush=True)
            with log_path.open("a") as handle:
                handle.write(line + "\n")
            if step in LOSS_MARKERS:
                print(f"MILESTONE step={step} loss={loss:.6f}", flush=True)

    logger_cb = LossLogger()
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=5,
        logging_steps=1,
        save_steps=args.max_steps,
        save_total_limit=1,
        bf16=True,
        gradient_checkpointing=True,
        report_to=[],
        logging_first_step=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
        callbacks=[logger_cb],
    )
    print("TRAIN_START", flush=True)
    try:
        trainer.train()
    except Exception as exc:
        print(f"TRAIN_FAIL: {exc}", flush=True)
        import traceback

        traceback.print_exc()
        return 1
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print("ADAPTER_SAVED", flush=True)
    print(f"ADAPTER_PATH {out_dir}", flush=True)

    final = logger_cb.step_losses.get(args.max_steps) or (
        logger_cb.step_losses[max(logger_cb.step_losses)] if logger_cb.step_losses else None
    )
    first = logger_cb.step_losses.get(1) or (
        logger_cb.step_losses[min(logger_cb.step_losses)] if logger_cb.step_losses else None
    )
    summary = {
        "final_loss": final,
        "step_1_loss": first,
        "loss_decreased": bool(first is not None and final is not None and final < first),
        "loss_curve": [{"step": s, "loss": v} for s, v in sorted(logger_cb.step_losses.items())],
        "adapter_path": str(out_dir),
    }
    summary_path = out_dir / "canary_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("TRAIN_DONE", flush=True)
    print(json.dumps(summary), flush=True)
    return 0 if final is not None and final == final else 1


if __name__ == "__main__":
    raise SystemExit(main())
