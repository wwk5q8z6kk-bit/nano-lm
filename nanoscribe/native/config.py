"""Native Nano training configuration — NATIVE-A decoder, NATIVE-B evidence-aware."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from nanoscribe.native.factorial import (
    ArchFactor,
    FACTORIAL_CELLS,
    FactorialCell,
    ROUND2_PROMOTIONS,
    round2_run_id,
)


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


def _estimate_param_count(d_model: int, n_layers: int, *, evidence_aware: bool, vocab_size: int = 4098) -> int:
    emb = vocab_size * d_model
    block = n_layers * (4 * d_model * d_model + 2 * d_model * (4 * d_model))
    evidence = 2 * d_model * d_model if evidence_aware else 0
    return emb + block + evidence


def dims_for_target_params_m(
    target_m: float,
    *,
    evidence_aware: bool,
    vocab_size: int = 4098,
) -> tuple[int, int, int]:
    """Pick (d_model, n_layers, n_heads) closest to target param budget."""
    target = int(target_m * 1e6)
    best = (256, 8, 8)
    best_diff = float("inf")
    for n_layers in (10, 12, 14, 16):
        for d_model in range(384, 1025, 32):
            n_heads = max(4, d_model // 64)
            count = _estimate_param_count(d_model, n_layers, evidence_aware=evidence_aware, vocab_size=vocab_size)
            diff = abs(count - target)
            if diff < best_diff:
                best_diff = diff
                best = (d_model, n_layers, n_heads)
    return best


def _variant_for_arch(arch: ArchFactor) -> NativeVariant:
    return NativeVariant.NATIVE_B if arch == ArchFactor.EVIDENCE_BOTTLENECK else NativeVariant.NATIVE_A


def config_for_run(run_id: str, *, cpu_smoke: bool = False) -> NativeTrainConfig:
    from nanoscribe.native.factorial import canonical_run_id, legacy_run_id

    for promo in ROUND2_PROMOTIONS:
        for seed in promo.seeds:
            rid = round2_run_id(promo, seed)
            if run_id == rid:
                variant = _variant_for_arch(promo.arch)
                d_model, n_layers, n_heads = dims_for_target_params_m(
                    promo.params_m,
                    evidence_aware=variant == NativeVariant.NATIVE_B,
                )
                return NativeTrainConfig(
                    run_id=rid,
                    variant=variant,
                    cell=None,
                    seed=seed,
                    params_m=promo.params_m,
                    d_model=d_model,
                    n_layers=n_layers,
                    n_heads=n_heads,
                    max_steps=200,
                    batch_size=16,
                    peak_lr=3e-4,
                    cpu_smoke=cpu_smoke,
                    loss_weights=default_loss_weights(variant),
                )

    for cell in FACTORIAL_CELLS:
        for seed in cell.seeds:
            if run_id in {canonical_run_id(cell, seed), legacy_run_id(cell, seed)}:
                variant = _variant_for_arch(cell.arch)
                return NativeTrainConfig(
                    run_id=canonical_run_id(cell, seed),
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
