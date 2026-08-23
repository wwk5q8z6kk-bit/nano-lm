"""Native Nano training configuration — NATIVE-A decoder, NATIVE-B evidence-aware."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from nanoscribe.native.factorial import FACTORIAL_CELLS, FactorialCell


class NativeVariant(str, Enum):
    NATIVE_A = "native_a_decoder_baseline"
    NATIVE_B = "native_b_evidence_aware"


@dataclass(frozen=True, slots=True)
class LossWeights:
    lm: float = 1.0
    span_port: float = 1.0
    evidence_align: float = 0.0
    assertion_state: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "lm": self.lm,
            "span_port": self.span_port,
            "evidence_align": self.evidence_align,
            "assertion_state": self.assertion_state,
        }


@dataclass(frozen=True, slots=True)
class NativeTrainConfig:
    run_id: str
    variant: NativeVariant
    cell: FactorialCell | None
    seed: int
    params_m: int = 30
    vocab_size: int = 4098
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    max_seq: int = 512
    max_steps: int = 200
    batch_size: int = 16
    peak_lr: float = 3e-4
    dataset_path: str = "artifacts/campaign/p1_distill_train_v1.json"
    checkpoint_dir: str = "artifacts/native_checkpoints"
    cpu_smoke: bool = False
    loss_weights: LossWeights = field(default_factory=LossWeights)

    @property
    def evidence_aware(self) -> bool:
        return self.variant == NativeVariant.NATIVE_B

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "variant": self.variant.value,
            "seed": self.seed,
            "params_m": self.params_m,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "max_seq": self.max_seq,
            "max_steps": self.max_steps,
            "batch_size": self.batch_size,
            "peak_lr": self.peak_lr,
            "dataset_path": self.dataset_path,
            "checkpoint_dir": self.checkpoint_dir,
            "cpu_smoke": self.cpu_smoke,
            "loss_weights": self.loss_weights.as_dict(),
            "cell": self.cell.cell_id if self.cell else None,
        }


def default_loss_weights(variant: NativeVariant) -> LossWeights:
    if variant == NativeVariant.NATIVE_A:
        return LossWeights(lm=1.0, span_port=1.0, evidence_align=0.0, assertion_state=0.0)
    return LossWeights(lm=0.5, span_port=1.0, evidence_align=0.5, assertion_state=0.25)


def config_for_run(run_id: str, *, cpu_smoke: bool = False) -> NativeTrainConfig:
    for cell in FACTORIAL_CELLS:
        for seed in cell.seeds:
            if run_id == f"{cell.cell_id}_s{seed}":
                variant = (
                    NativeVariant.NATIVE_B
                    if cell.arch.value == "evidence_bottleneck"
                    else NativeVariant.NATIVE_A
                )
                return NativeTrainConfig(
                    run_id=run_id,
                    variant=variant,
                    cell=cell,
                    seed=seed,
                    cpu_smoke=cpu_smoke,
                    loss_weights=default_loss_weights(variant),
                )
    raise ValueError(f"unknown run_id: {run_id}")


def smoke_config(*, variant: NativeVariant = NativeVariant.NATIVE_A) -> NativeTrainConfig:
    return NativeTrainConfig(
        run_id="smoke_cpu",
        variant=variant,
        cell=None,
        seed=0,
        max_steps=2,
        batch_size=2,
        d_model=64,
        n_layers=2,
        n_heads=2,
        cpu_smoke=True,
        loss_weights=default_loss_weights(variant),
    )


def checkpoint_path(cfg: NativeTrainConfig, step: int | None = None) -> Path:
    base = Path(cfg.checkpoint_dir) / cfg.run_id
    if step is None:
        return base / "latest.pt"
    return base / f"step_{step:06d}.pt"
