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
    bits_per_byte: float = float("nan")
    integrity: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "steps_completed": self.steps_completed,
            "final_loss": self.final_loss,
            "device": self.device,
            "checkpoint": self.checkpoint,
            "bits_per_byte": self.bits_per_byte,
            "integrity": self.integrity,
            "smoke": self.smoke,
        }


def _resolve_device(cpu_smoke: bool) -> str:
    """Prefer CUDA, then Apple MPS, then CPU.

    MPS was previously ignored, so every local run on Apple Silicon fell back to
    CPU while the integrated GPU sat idle. Measured on this machine for a 30M arm
    at batch 32: CPU 0.348 steps/s vs MPS 1.396 steps/s — a 4.0x speedup that
    turns a 13h local screen into 3.2h and removes any need to rent a GPU for it.

    Set NANO_FORCE_CPU=1 to opt out (useful when isolating an MPS-specific
    numerical difference).
    """
    import os

    import torch

    if cpu_smoke or os.environ.get("NANO_FORCE_CPU"):
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        # Some ops still lack MPS kernels; fall back per-op rather than failing.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps"
    return "cpu"


def train_native(
    cfg: NativeTrainConfig,
    *,
    resume: bool = False,
) -> TrainResult:
    import torch

    from nanoscribe.native.corpus.registry import assert_corpus_launch_allowed
    from nanoscribe.native.integrity import (
        assert_bits_per_byte_plausible,
        bits_per_byte,
        run_startup_gate,
    )

    assert_corpus_launch_allowed(
        cfg.dataset_path,
        purpose=cfg.purpose if not cfg.cpu_smoke else "trainer_smoke",
        cpu_smoke=cfg.cpu_smoke,
    )

    build = build_native_model(cfg)
    model = build.model
    examples = load_train_examples(cfg.dataset_path)

    # Startup integrity gate — runs on the CPU model before the device move, on
    # every run, and raises rather than warns. Each check corresponds to a
    # defect in artifacts/DEFECT_INDEX.md that shipped as PASSING code and
    # inflated a result: D1.1 (missing causal mask), D2.1/D2.2 (target absent
    # from the loss), D3.1 (tokenizer silently capping the prompt). A one-off
    # check that already passed is not protection.
    integrity = run_startup_gate(model, examples, cfg)

    device = _resolve_device(cfg.cpu_smoke)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.peak_lr)
    start_step = 0
    if resume:
        payload = load_checkpoint(cfg, model, optimizer=optimizer)
        start_step = int(payload.get("step", 0))
    final_loss = 0.0
    final_bpb = float('nan')
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
            # Bits per byte, tracked every step. `lm` is mean CE in nats over
            # supervised (target) positions and the tokenizer is character
            # level, so nats-per-target-char is `lm`; scale by chars/bytes to
            # land on a per-byte figure that is comparable across tokenizers.
            lm_nats = float(breakdown.lm.detach()) if hasattr(breakdown.lm, "detach") else float(breakdown.lm)
            target_chars = sum(len(t) for t in targets)
            target_bytes = sum(len(t.encode("utf-8")) for t in targets)
            final_bpb = bits_per_byte(lm_nats * target_chars, max(1, target_bytes))
            step += 1
            if step % max(1, cfg.max_steps // 5) == 0 or step >= cfg.max_steps:
                save_checkpoint(model, cfg, step=step, optimizer=optimizer, extra={"loss": final_loss})
        epoch += 1

    # Information-theoretic floor rather than a remembered loss value. The
    # causal-mask leak presented as loss ~= 0.002, i.e. ~0.003 bits/byte — a
    # compression rate that is impossible on real text. See DEFECT_INDEX D1.1.
    if step > start_step:
        assert_bits_per_byte_plausible(final_bpb, step=step)

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
        bits_per_byte=final_bpb,
        integrity=integrity.to_dict(),
    )


def train_run_id(run_id: str, *, cpu_smoke: bool = False, resume: bool = False) -> TrainResult:
    cfg = config_for_run(run_id, cpu_smoke=cpu_smoke)
    return train_native(cfg, resume=resume)


def cpu_smoke_train(variant: str = "native_a") -> TrainResult:
    from nanoscribe.native.config import NativeVariant

    v = NativeVariant.NATIVE_B if variant.endswith("_b") else NativeVariant.NATIVE_A
    cfg = smoke_config(variant=v)
    return train_native(cfg)
