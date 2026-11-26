"""Reproducible training, WSD/Cosine scheduling, BPB validation, and atomic checkpointing for NanoLM."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as F

from nanolm.config import NanoLMConfig
from nanolm.data.packing import PackedDocumentDataset
from nanolm.model.transformer import NanoLMForCausalLM


def detect_best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass(frozen=True, slots=True)
class TrainConfig:
    run_id: str = "nanolm-baseline"
    max_steps: int = 300
    micro_batch_size: int = 8
    grad_accum_steps: int = 2
    peak_lr: float = 1e-3
    min_lr_ratio: float = 0.10
    warmup_steps: int = 30
    schedule: str = "wsd"  # "wsd" (Warmup-Stable-Decay) or "cosine"
    decay_fraction: float = 0.20  # Last 20% of steps decay in WSD schedule
    weight_decay: float = 0.10
    grad_clip: float = 1.0
    eval_every: int = 50
    save_every: int = 100
    seed: int = 42
    precision: str = "bf16"  # "bf16", "fp16", or "fp32"
    gradient_checkpointing: bool = False
    checkpoint_dir: str = "experiments/runs/default"


@dataclass(frozen=True, slots=True)
class TrainResult:
    run_id: str
    model_name: str
    deployed_parameters: int
    active_parameters_per_token: int
    steps_completed: int
    tokens_trained: int
    wall_seconds: float
    tokens_per_second: float
    initial_val_loss: float
    final_val_loss: float
    initial_val_bpb: float
    final_val_bpb: float
    final_checkpoint_path: str
    history: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_lr(step: int, cfg: TrainConfig) -> float:
    """Compute step learning rate under Warmup-Stable-Decay (WSD) or Cosine schedule."""
    if step <= cfg.warmup_steps:
        return cfg.peak_lr * (step / max(1, cfg.warmup_steps))
    if cfg.schedule == "wsd":
        decay_steps = max(1, int(cfg.max_steps * cfg.decay_fraction))
        stable_end = cfg.max_steps - decay_steps
        if step <= stable_end:
            return cfg.peak_lr
        progress = (step - stable_end) / decay_steps
        # 1-sqrt or cosine decay in final phase; cosine-to-floor is smooth and strong for small LMs
        factor = cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return cfg.peak_lr * factor
    # Standard cosine
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    factor = cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return cfg.peak_lr * factor


@torch.no_grad()
def evaluate_dataset_bpb(
    model: NanoLMForCausalLM,
    dataset: PackedDocumentDataset,
    bytes_per_token: float,
    batch_size: int = 8,
    max_batches: int = 12,
    device: str = "cpu",
    autocast_dtype: torch.dtype | None = None,
) -> tuple[float, float]:
    """Return (mean_nats_per_token, bits_per_byte) on held-out packed blocks.

    Bits-Per-Byte (BPB) = (nats_per_token / ln(2)) / bytes_per_token, making cross-tokenizer
    and cross-architecture comparisons strictly invariant to vocabulary size!
    """
    was_training = model.training
    model.eval()
    try:
        total_nll = 0.0
        total_valid_tokens = 0
        n_batches = min(max_batches, max(1, dataset.num_blocks // max(1, batch_size)))
        for b_idx in range(n_batches):
            lo = (b_idx * batch_size) % max(1, dataset.num_blocks - batch_size + 1)
            idx = list(range(lo, min(dataset.num_blocks, lo + batch_size)))
            x = torch.from_numpy(dataset.inputs[idx].astype("int64")).to(device)
            y = torch.from_numpy(dataset.targets[idx].astype("int64")).to(device)
            pos = torch.from_numpy(dataset.positions[idx].astype("int64")).to(device)
            doc = torch.from_numpy(dataset.docs[idx].astype("int64")).to(device)

            if autocast_dtype is not None and device in ("cuda", "mps"):
                with torch.autocast(device_type=device, dtype=autocast_dtype):
                    out = model(input_ids=x, position_ids=pos, doc_ids=doc)
            else:
                out = model(input_ids=x, position_ids=pos, doc_ids=doc)

            nll = F.cross_entropy(
                out.logits.view(-1, model.config.vocab_size).float(),
                y.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            valid = int((y != -100).sum().item())
            total_nll += float(nll.item())
            total_valid_tokens += valid

        mean_loss = total_nll / max(1, total_valid_tokens)
        bits_per_token = mean_loss / math.log(2.0)
        bpb = bits_per_token / max(1e-6, bytes_per_token)
        return mean_loss, bpb
    finally:
        if was_training:
            model.train()


def save_atomic_checkpoint(
    path: str | Path,
    model: NanoLMForCausalLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    train_cfg: TrainConfig,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    """Write checkpoint atomically via temporary file rename to prevent partial corruption."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    payload = {
        "schema": "nanolm.checkpoint.v1",
        "step": step,
        "model_config": asdict(model.config),
        "train_config": asdict(train_cfg),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "extra": extra_meta or {},
    }
    torch.save(payload, tmp)
    tmp.replace(target)
    return target


def load_checkpoint(
    path: str | Path,
    model: NanoLMForCausalLM,
    optimizer: torch.optim.Optimizer | None = None,
    device: str = "cpu",
) -> int:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return int(payload["step"])


def run_training(
    model_cfg: NanoLMConfig,
    train_cfg: TrainConfig,
    train_dataset: PackedDocumentDataset,
    val_dataset: PackedDocumentDataset,
    val_bytes_per_token: float,
    device: str | None = None,
    resume_from: str | Path | None = None,
) -> TrainResult:
    """Execute reproducible pretraining/midtraining run with exact step-indexed data & WSD/Cosine LR."""
    dev = device or detect_best_device()
    torch.manual_seed(train_cfg.seed)

    model = NanoLMForCausalLM(model_cfg).to(dev)
    model.gradient_checkpointing = train_cfg.gradient_checkpointing

    decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": train_cfg.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=train_cfg.peak_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    autocast_dtype: torch.dtype | None = None
    if train_cfg.precision == "bf16" and dev in ("cuda", "mps"):
        autocast_dtype = torch.bfloat16
    elif train_cfg.precision == "fp16" and dev in ("cuda", "mps"):
        autocast_dtype = torch.float16

    start_step = 0
    if resume_from is not None and Path(resume_from).is_file():
        start_step = load_checkpoint(resume_from, model, optimizer=optimizer, device=dev)

    init_loss, init_bpb = evaluate_dataset_bpb(
        model,
        val_dataset,
        bytes_per_token=val_bytes_per_token,
        batch_size=train_cfg.micro_batch_size,
        device=dev,
        autocast_dtype=autocast_dtype,
    )

    history: list[dict[str, Any]] = [
        {"step": start_step, "val_loss": round(init_loss, 4), "val_bpb": round(init_bpb, 4)}
    ]
    ckpt_dir = Path(train_cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tokens_per_step = train_cfg.micro_batch_size * train_cfg.grad_accum_steps * train_dataset.seq_len
    t0 = time.perf_counter()

    for step in range(start_step + 1, train_cfg.max_steps + 1):
        lr = compute_lr(step, train_cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_ce = 0.0
        step_mtp = 0.0

        for micro_idx in range(train_cfg.grad_accum_steps):
            sub_step = (step - 1) * train_cfg.grad_accum_steps + micro_idx
            batch = train_dataset.sample_batch(
                batch_size=train_cfg.micro_batch_size,
                step=sub_step,
                seed=train_cfg.seed,
                device=dev,
            )
            if autocast_dtype is not None:
                with torch.autocast(device_type=dev, dtype=autocast_dtype):
                    out = model(
                        input_ids=batch.input_ids,
                        targets=batch.targets,
                        position_ids=batch.position_ids,
                        doc_ids=batch.doc_ids,
                    )
                    assert out.loss is not None
                    loss = out.loss / train_cfg.grad_accum_steps
            else:
                out = model(
                    input_ids=batch.input_ids,
                    targets=batch.targets,
                    position_ids=batch.position_ids,
                    doc_ids=batch.doc_ids,
                )
                assert out.loss is not None
                loss = out.loss / train_cfg.grad_accum_steps

            loss.backward()
            step_ce += float(out.ce_loss.item()) / train_cfg.grad_accum_steps if out.ce_loss is not None else 0.0
            if out.mtp_loss is not None:
                step_mtp += float(out.mtp_loss.item()) / train_cfg.grad_accum_steps

        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip).item())
        optimizer.step()

        if step % train_cfg.eval_every == 0 or step == train_cfg.max_steps:
            val_loss, val_bpb = evaluate_dataset_bpb(
                model,
                val_dataset,
                bytes_per_token=val_bytes_per_token,
                batch_size=train_cfg.micro_batch_size,
                device=dev,
                autocast_dtype=autocast_dtype,
            )
            elapsed = max(1e-6, time.perf_counter() - t0)
            tps = (step - start_step) * tokens_per_step / elapsed
            rec: dict[str, Any] = {
                "step": step,
                "train_ce": round(step_ce, 4),
                "train_mtp": round(step_mtp, 4),
                "val_loss": round(val_loss, 4),
                "val_bpb": round(val_bpb, 4),
                "grad_norm": round(grad_norm, 4),
                "lr": round(lr, 7),
                "tok_per_s": round(tps, 1),
            }
            rec.update({k: round(v, 4) for k, v in model.moe_load_stats().items()})
            history.append(rec)

        if step % train_cfg.save_every == 0 or step == train_cfg.max_steps:
            save_atomic_checkpoint(
                ckpt_dir / f"step_{step:06d}.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                train_cfg=train_cfg,
                extra_meta={"history_tail": history[-1]},
            )

    if dev == "mps":
        torch.mps.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()
    wall = max(1e-6, time.perf_counter() - t0)
    steps_run = max(1, train_cfg.max_steps - start_step)
    total_tokens = steps_run * tokens_per_step

    final_ckpt = save_atomic_checkpoint(
        ckpt_dir / "latest.pt",
        model=model,
        optimizer=optimizer,
        step=train_cfg.max_steps,
        train_cfg=train_cfg,
        extra_meta={"history": history},
    )

    acct = model_cfg.parameter_accounting()
    final_val_loss = float(history[-1]["val_loss"])
    final_val_bpb = float(history[-1]["val_bpb"])
    result = TrainResult(
        run_id=train_cfg.run_id,
        model_name=model_cfg.name,
        deployed_parameters=acct.total_deployed_parameters,
        active_parameters_per_token=acct.active_parameters_per_token,
        steps_completed=train_cfg.max_steps,
        tokens_trained=total_tokens,
        wall_seconds=round(wall, 3),
        tokens_per_second=round(total_tokens / wall, 1),
        initial_val_loss=round(init_loss, 4),
        final_val_loss=round(final_val_loss, 4),
        initial_val_bpb=round(init_bpb, 4),
        final_val_bpb=round(final_val_bpb, 4),
        final_checkpoint_path=str(final_ckpt),
        history=history,
    )
    (ckpt_dir / "train_result.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result
