"""Native Nano training loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nanoscribe.native.checkpoint import load_checkpoint, save_checkpoint
from nanoscribe.native.config import NativeTrainConfig, config_for_run, smoke_config
from nanoscribe.native.data import NativeBatchIterator, load_train_examples
from nanoscribe.native.losses import LossBreakdown, compute_batch_loss
from nanoscribe.native.model import build_native_model


@dataclass(frozen=True, slots=True)
class TrainResult:
    run_id: str
    steps_completed: int
    final_loss: float
    device: str
    checkpoint: str | None
    smoke: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "steps_completed": self.steps_completed,
            "final_loss": self.final_loss,
            "device": self.device,
            "checkpoint": self.checkpoint,
            "smoke": self.smoke,
        }


def _resolve_device(cpu_smoke: bool) -> str:
    import torch

    if cpu_smoke or not torch.cuda.is_available():
        return "cpu"
    return "cuda"


def train_native(
    cfg: NativeTrainConfig,
    *,
    resume: bool = False,
) -> TrainResult:
    import torch

    from nanoscribe.native.corpus.registry import assert_corpus_launch_allowed

    assert_corpus_launch_allowed(
        cfg.dataset_path,
        purpose=cfg.purpose if not cfg.cpu_smoke else "trainer_smoke",
        cpu_smoke=cfg.cpu_smoke,
    )

    build = build_native_model(cfg)
    model = build.model
    device = _resolve_device(cfg.cpu_smoke)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.peak_lr)
    start_step = 0
    if resume:
        payload = load_checkpoint(cfg, model, optimizer=optimizer)
        start_step = int(payload.get("step", 0))

    examples = load_train_examples(cfg.dataset_path)
    final_loss = 0.0
    step = start_step
    epoch = 0
    while step < cfg.max_steps:
        iterator = NativeBatchIterator(examples, cfg.batch_size, seed=cfg.seed + epoch)
        for batch in iterator:
            if step >= cfg.max_steps:
                break
            prompts = [item.prompt for item in batch]
            targets = [item.target for item in batch]
            optimizer.zero_grad(set_to_none=True)
            breakdown = compute_batch_loss(model, prompts, targets, cfg)
            total_tensor = breakdown.lm + breakdown.span_port
            if cfg.evidence_aware:
                total_tensor = (
                    total_tensor
                    + cfg.loss_weights.evidence_align * breakdown.evidence_align
                    + cfg.loss_weights.assertion_state * breakdown.assertion_state
                )
            total_tensor.backward()
            optimizer.step()
            final_loss = float(breakdown.total.detach()) if hasattr(breakdown.total, "detach") else float(breakdown.total)
            step += 1
            if step % max(1, cfg.max_steps // 5) == 0 or step >= cfg.max_steps:
                save_checkpoint(model, cfg, step=step, optimizer=optimizer, extra={"loss": final_loss})
        epoch += 1

    ckpt = None
    if step > start_step:
        path = save_checkpoint(model, cfg, step=step, optimizer=optimizer, extra={"loss": final_loss})
        ckpt = str(path)
    return TrainResult(
        run_id=cfg.run_id,
        steps_completed=step,
        final_loss=final_loss,
        device=device,
        checkpoint=ckpt,
        smoke=cfg.cpu_smoke,
    )


def train_run_id(run_id: str, *, cpu_smoke: bool = False, resume: bool = False) -> TrainResult:
    cfg = config_for_run(run_id, cpu_smoke=cpu_smoke)
    return train_native(cfg, resume=resume)


def cpu_smoke_train(variant: str = "native_a") -> TrainResult:
    from nanoscribe.native.config import NativeVariant

    v = NativeVariant.NATIVE_B if variant.endswith("_b") else NativeVariant.NATIVE_A
    cfg = smoke_config(variant=v)
    return train_native(cfg)
