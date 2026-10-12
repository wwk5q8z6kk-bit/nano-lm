"""Frozen H2 recipe for Nano's direct state and evidence-pointer objective.

H2 keeps Nano v0.1's exact transformer trunk, adds only 8,665 bounded output
parameters, and trains the full model on the already sealed H1 train partition.
No benchmark or fresh-confirmation partition is imported by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tokenizers
import torch
from torch import Tensor
from torch.nn import functional as torch_functional

from nano_ai.adapters import state_span
from nano_ai.adapters.anchor_checkpoint import FROZEN_NANO_V01
from nano_ai.contract import FieldState
from nano_ai.training import pointer_data, state_span_data, train_state_span
from nano_ai.training.pointer_data import (
    IGNORE_SPAN_INDEX,
    POINTER_PROMPT_TEMPLATE_ID,
    POINTER_SUPERVISION_VERSION,
    STATE_ORDER,
    STATE_POINTER_COUNTS,
    PointerSupervision,
    encode_pointer_partition,
    load_pointer_tokenizer,
)
from nano_ai.training.pointer_model import (
    NANO_POINTER_PARAMETER_COUNT,
    NANO_TRUNK_PARAMETER_COUNT,
    POINTER_HEAD_PARAMETER_COUNT,
    NanoPointerModel,
    PointerModelOutput,
)
from nano_ai.training.state_span_data import (
    DATASET_SCHEMA_VERSION,
    TARGET_GRAMMAR_VERSION,
    canonical_json_bytes,
)
from nano_ai.training.train_state_span import (
    ADAM_BETAS,
    ADAM_EPSILON,
    BATCH_SIZE,
    COSINE_FLOOR,
    EPOCHS,
    GRADIENT_CLIP,
    LOG_EVERY_STEPS,
    PEAK_LEARNING_RATE,
    TRAINING_SEEDS,
    WARMUP_FRACTION,
    WEIGHT_DECAY,
    TrainingInputError,
    _seed_training,
    grouped_batch_indices,
    learning_rate_at,
    load_training_bundle,
)
from nano_ai.training.train_state_span import (
    _resolve_device as _resolve_base_device,
)

POINTER_TRAINING_REPORT_SCHEMA_VERSION = "nano.pointer-span-training-report.v0"
POINTER_TRAINING_RECIPE_VERSION = "nano-pointer-span-supervision-v0"
STATE_LOSS_WEIGHT = 1.0
POINTER_LOSS_WEIGHT = 1.0
STATE_CLASS_COUNTS: tuple[int, ...] = (46_050, 4_950, 3_000, 3_000, 3_000)
STATE_CLASS_WEIGHTS: tuple[float, ...] = tuple(
    (sum(STATE_CLASS_COUNTS) / len(STATE_CLASS_COUNTS)) / count
    for count in STATE_CLASS_COUNTS
)
STATE_LOSS_DEFINITION = (
    "train_inverse_frequency_weighted_cross_entropy_mean_by_weight_mass"
)
_DETERMINISTIC_CUBLAS_CONFIGS = frozenset({":4096:8", ":16:8"})


@dataclass(frozen=True, slots=True)
class PointerBatch:
    """One right-padded tensor batch with explicit active pointer slots."""

    token_ids: Tensor
    attention_mask: Tensor
    pointer_mask: Tensor
    state_labels: Tensor
    span_starts: Tensor
    span_ends: Tensor
    span_mask: Tensor


@dataclass(frozen=True, slots=True)
class PointerLoss:
    """Differentiable H2 loss components and stable aggregation counts."""

    total: Tensor
    state: Tensor
    pointer: Tensor
    state_count: int
    state_weight_sum: float
    pointer_count: int


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def collate_pointer_batch(
    records: Sequence[PointerSupervision],
    *,
    device: str,
) -> PointerBatch:
    """Right-pad one selected batch without changing any supervision index."""

    frozen = tuple(records)
    if not frozen or any(not isinstance(item, PointerSupervision) for item in frozen):
        raise TrainingInputError(
            "pointer batches must contain PointerSupervision records"
        )
    maximum = max(len(item.token_ids) for item in frozen)

    token_ids: list[list[int]] = []
    attention_mask: list[list[bool]] = []
    pointer_mask: list[list[bool]] = []
    for item in frozen:
        padding = maximum - len(item.token_ids)
        token_ids.append([*item.token_ids, *([0] * padding)])
        attention_mask.append([*item.attention_mask, *([False] * padding)])
        pointer_mask.append([*item.pointer_mask, *([False] * padding)])
    return PointerBatch(
        token_ids=torch.tensor(token_ids, dtype=torch.long, device=device),
        attention_mask=torch.tensor(attention_mask, dtype=torch.bool, device=device),
        pointer_mask=torch.tensor(pointer_mask, dtype=torch.bool, device=device),
        state_labels=torch.tensor(
            [item.state_labels for item in frozen],
            dtype=torch.long,
            device=device,
        ),
        span_starts=torch.tensor(
            [item.span_starts for item in frozen],
            dtype=torch.long,
            device=device,
        ),
        span_ends=torch.tensor(
            [item.span_ends for item in frozen],
            dtype=torch.long,
            device=device,
        ),
        span_mask=torch.tensor(
            [item.span_mask for item in frozen],
            dtype=torch.bool,
            device=device,
        ),
    )


def pointer_objective(
    output: PointerModelOutput,
    batch: PointerBatch,
) -> PointerLoss:
    """Compute state CE plus mean start/end CE on active Patient slots only."""

    if batch.token_ids.ndim != 2:
        raise TrainingInputError("pointer token batch must have shape [batch, tokens]")
    batch_size, sequence_length = batch.token_ids.shape
    expected_state_shape = (batch_size, len(STATE_ORDER))
    expected_pointer_shape = (
        batch_size,
        sequence_length,
        len(STATE_ORDER),
        2,
    )
    if tuple(batch.attention_mask.shape) != (batch_size, sequence_length) or tuple(
        batch.pointer_mask.shape
    ) != (batch_size, sequence_length):
        raise TrainingInputError("token masks and token batch have incompatible shapes")
    if tuple(batch.state_labels.shape) != expected_state_shape:
        raise TrainingInputError("state labels have an incompatible shape")
    if tuple(output.state_logits.shape) != (
        batch_size,
        len(STATE_ORDER),
        len(STATE_ORDER),
    ):
        raise TrainingInputError("state logits and labels have incompatible shapes")
    if (
        tuple(output.start_logits.shape) != expected_pointer_shape
        or tuple(output.end_logits.shape) != expected_pointer_shape
    ):
        raise TrainingInputError(
            "pointer logits and token batch have incompatible shapes"
        )
    expected_label_shape = (batch_size, len(STATE_ORDER), 2)
    if (
        tuple(batch.span_starts.shape) != expected_label_shape
        or tuple(batch.span_ends.shape) != expected_label_shape
        or tuple(batch.span_mask.shape) != expected_label_shape
    ):
        raise TrainingInputError("pointer labels and mask have incompatible shapes")
    if bool(torch.any(batch.pointer_mask & ~batch.attention_mask)):
        raise TrainingInputError("pointer mask must remain inside the attention mask")
    if bool(
        torch.any(batch.state_labels < 0)
        or torch.any(batch.state_labels >= len(STATE_ORDER))
    ):
        raise TrainingInputError("state labels contain an invalid class index")
    if bool(torch.any(batch.span_mask[:, :, 1] & ~batch.span_mask[:, :, 0])):
        raise TrainingInputError("active pointer slots must be left-packed")
    if not bool(torch.all(batch.span_starts[~batch.span_mask] == IGNORE_SPAN_INDEX)):
        raise TrainingInputError("inactive start labels must use the ignore index")
    if not bool(torch.all(batch.span_ends[~batch.span_mask] == IGNORE_SPAN_INDEX)):
        raise TrainingInputError("inactive end labels must use the ignore index")
    active_count = int(batch.span_mask.sum().item())
    if active_count < 1:
        raise TrainingInputError("pointer batch has no active evidence supervision")

    active_starts = batch.span_starts[batch.span_mask]
    active_ends = batch.span_ends[batch.span_mask]
    if bool(
        torch.any(active_starts < 0)
        or torch.any(active_ends < active_starts)
        or torch.any(active_ends >= sequence_length)
    ):
        raise TrainingInputError("active pointer labels are out of bounds")
    expected_counts = torch.tensor(
        STATE_POINTER_COUNTS,
        dtype=torch.long,
        device=batch.state_labels.device,
    )[batch.state_labels]
    if not bool(torch.equal(batch.span_mask.sum(dim=-1), expected_counts)):
        raise TrainingInputError("state labels and pointer-slot arity disagree")

    row_indices = (
        torch.arange(batch_size, device=batch.token_ids.device)
        .view(batch_size, 1, 1)
        .expand_as(batch.span_mask)[batch.span_mask]
    )
    invalid_prefix = (~batch.pointer_mask).long().cumsum(dim=1)
    invalid_through_end = invalid_prefix[row_indices, active_ends]
    before_start_indices = (active_starts - 1).clamp_min(0)
    invalid_before_start = invalid_prefix[row_indices, before_start_indices]
    invalid_before_start = torch.where(
        active_starts > 0,
        invalid_before_start,
        torch.zeros_like(invalid_before_start),
    )
    if bool(torch.any(invalid_through_end != invalid_before_start)):
        raise TrainingInputError("active pointer labels escape the Patient token mask")

    state_weights = torch.tensor(
        STATE_CLASS_WEIGHTS,
        dtype=output.state_logits.dtype,
        device=output.state_logits.device,
    )
    flat_state_labels = batch.state_labels.reshape(-1)
    weighted_state_losses = torch_functional.cross_entropy(
        output.state_logits.reshape(-1, len(STATE_ORDER)),
        flat_state_labels,
        weight=state_weights,
        reduction="none",
    )
    state_weight_sum_tensor = state_weights[flat_state_labels].sum()
    state_loss = weighted_state_losses.sum() / state_weight_sum_tensor
    valid_positions = batch.pointer_mask[:, :, None, None]
    floor = torch.finfo(output.start_logits.dtype).min
    start_logits = output.start_logits.masked_fill(~valid_positions, floor)
    end_logits = output.end_logits.masked_fill(~valid_positions, floor)
    start_loss = torch_functional.cross_entropy(
        start_logits.permute(0, 2, 3, 1).reshape(-1, sequence_length),
        batch.span_starts.reshape(-1),
        ignore_index=IGNORE_SPAN_INDEX,
    )
    end_loss = torch_functional.cross_entropy(
        end_logits.permute(0, 2, 3, 1).reshape(-1, sequence_length),
        batch.span_ends.reshape(-1),
        ignore_index=IGNORE_SPAN_INDEX,
    )
    pointer_loss = 0.5 * (start_loss + end_loss)
    return PointerLoss(
        total=STATE_LOSS_WEIGHT * state_loss + POINTER_LOSS_WEIGHT * pointer_loss,
        state=state_loss,
        pointer=pointer_loss,
        state_count=batch_size * len(STATE_ORDER),
        state_weight_sum=float(state_weight_sum_tensor.detach().item()),
        pointer_count=active_count,
    )


def _optimizer(model: NanoPointerModel) -> torch.optim.AdamW:
    decay = [parameter for parameter in model.parameters() if parameter.dim() >= 2]
    no_decay = [parameter for parameter in model.parameters() if parameter.dim() < 2]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=PEAK_LEARNING_RATE,
        betas=ADAM_BETAS,
        eps=ADAM_EPSILON,
    )


def _loss_aggregates() -> dict[str, float | int]:
    return {
        "state_sum": 0.0,
        "state_count": 0,
        "state_weight_sum": 0.0,
        "pointer_sum": 0.0,
        "pointer_count": 0,
    }


def _accumulate_loss(values: dict[str, float | int], loss: PointerLoss) -> None:
    values["state_sum"] += float(loss.state.item()) * loss.state_weight_sum
    values["state_count"] += loss.state_count
    values["state_weight_sum"] += loss.state_weight_sum
    values["pointer_sum"] += float(loss.pointer.item()) * loss.pointer_count
    values["pointer_count"] += loss.pointer_count


def _finish_loss(values: Mapping[str, float | int]) -> dict[str, float]:
    state = float(values["state_sum"]) / float(values["state_weight_sum"])
    pointer = float(values["pointer_sum"]) / int(values["pointer_count"])
    return {
        "loss": STATE_LOSS_WEIGHT * state + POINTER_LOSS_WEIGHT * pointer,
        "state_loss": state,
        "pointer_loss": pointer,
    }


def _development_loss(
    model: NanoPointerModel,
    records: Sequence[PointerSupervision],
    *,
    device: str,
) -> dict[str, float]:
    model.eval()
    totals = _loss_aggregates()
    with torch.no_grad():
        for start in range(0, len(records), BATCH_SIZE):
            batch = collate_pointer_batch(
                records[start : start + BATCH_SIZE], device=device
            )
            loss = pointer_objective(
                model(batch.token_ids, batch.attention_mask), batch
            )
            _accumulate_loss(totals, loss)
    model.train()
    return _finish_loss(totals)


def _save_checkpoint(path: Path, model: NanoPointerModel) -> dict[str, Any]:
    state_dict = {
        name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
    }
    try:
        with path.open("xb") as handle:
            torch.save(state_dict, handle)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise RuntimeError(f"could not create checkpoint {path.name}") from exc
    snapshot = path.read_bytes()
    return {"filename": path.name, "sha256": _sha256(snapshot), "bytes": len(snapshot)}


def _write_json_no_clobber(path: Path, value: Any) -> None:
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def _source_hashes() -> dict[str, str]:
    files = {
        "data_generator": Path(state_span_data.__file__),
        "base_model": Path(__file__).with_name("model.py"),
        "pointer_data": Path(pointer_data.__file__),
        "pointer_model": Path(__file__).with_name("pointer_model.py"),
        "state_span_adapter": Path(state_span.__file__),
        "h1_training_loader": Path(train_state_span.__file__),
        "training": Path(__file__),
    }
    return {name: _sha256(path.read_bytes()) for name, path in sorted(files.items())}


def _require_unchanged_sources(expected: Mapping[str, str]) -> None:
    if _source_hashes() != dict(expected):
        raise TrainingInputError("pointer training source changed during execution")


def _state_class_counts(
    records: Sequence[PointerSupervision],
) -> tuple[int, ...]:
    counts = [0] * len(STATE_ORDER)
    for record in records:
        for state_label in record.state_labels:
            counts[state_label] += 1
    return tuple(counts)


def _require_frozen_training_state_distribution(
    records: Sequence[PointerSupervision],
) -> tuple[int, ...]:
    observed = _state_class_counts(records)
    if observed != STATE_CLASS_COUNTS:
        raise TrainingInputError(
            "pointer training state-class distribution is not the frozen H2 recipe"
        )
    return observed


def _resolve_pointer_device(requested: str) -> str:
    """Resolve H2 compute and fail early if deterministic CUDA is not configured."""

    resolved = _resolve_base_device(requested)
    if (
        resolved == "cuda"
        and os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        not in _DETERMINISTIC_CUBLAS_CONFIGS
    ):
        raise TrainingInputError(
            "deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8"
        )
    return resolved


def train_pointer_candidate(
    *,
    data_dir: Path,
    base_checkpoint: Path,
    tokenizer_path: Path,
    output_dir: Path,
    seed: int,
    device: str,
) -> Mapping[str, Any]:
    """Run one fixed H2 seed and create a never-overwritten candidate."""

    if seed not in TRAINING_SEEDS:
        raise TrainingInputError(
            f"seed must be one of the frozen seeds {TRAINING_SEEDS}"
        )
    output = Path(output_dir)
    if output.exists():
        raise TrainingInputError("candidate output directory must not exist")

    started = time.monotonic()
    source_hashes = _source_hashes()
    resolved_device = _resolve_pointer_device(device)
    bundle = load_training_bundle(Path(data_dir))
    tokenizer = load_pointer_tokenizer(Path(tokenizer_path))
    train_records = encode_pointer_partition(
        tokenizer, bundle.train, expected_split="train"
    )
    dev_records = encode_pointer_partition(tokenizer, bundle.dev, expected_split="dev")
    observed_train_state_counts = _require_frozen_training_state_distribution(
        train_records
    )
    _seed_training(seed)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise TrainingInputError("candidate output directory must not exist") from exc

    model = NanoPointerModel.from_frozen_base(Path(base_checkpoint))
    model.to(resolved_device).train()
    optimizer = _optimizer(model)
    first_epoch_batches = grouped_batch_indices(
        bundle.train,
        batch_size=BATCH_SIZE,
        seed=seed,
        epoch=1,
    )
    if any(len(batch) != BATCH_SIZE for batch in first_epoch_batches):
        raise TrainingInputError("frozen training worlds do not fill exact batches")
    steps_per_epoch = len(first_epoch_batches)
    total_steps = steps_per_epoch * EPOCHS
    global_step = 0
    epoch_reports: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []

    print(
        json.dumps(
            {
                "event": "training_start",
                "recipe": POINTER_TRAINING_RECIPE_VERSION,
                "seed": seed,
                "device": resolved_device,
                "train_records": len(train_records),
                "dev_records": len(dev_records),
                "steps": total_steps,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for epoch in range(1, EPOCHS + 1):
        batches = grouped_batch_indices(
            bundle.train,
            batch_size=BATCH_SIZE,
            seed=seed,
            epoch=epoch,
        )
        totals = _loss_aggregates()
        epoch_started = time.monotonic()
        for indices in batches:
            global_step += 1
            learning_rate = learning_rate_at(global_step, total_steps=total_steps)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            batch = collate_pointer_batch(
                [train_records[index] for index in indices],
                device=resolved_device,
            )
            loss = pointer_objective(
                model(batch.token_ids, batch.attention_mask), batch
            )
            optimizer.zero_grad(set_to_none=True)
            loss.total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRADIENT_CLIP
            )
            optimizer.step()
            _accumulate_loss(totals, loss)
            if global_step == 1 or global_step % LOG_EVERY_STEPS == 0:
                print(
                    json.dumps(
                        {
                            "event": "training_step",
                            "epoch": epoch,
                            "step": global_step,
                            "steps": total_steps,
                            "loss": round(float(loss.total.item()), 6),
                            "state_loss": round(float(loss.state.item()), 6),
                            "pointer_loss": round(float(loss.pointer.item()), 6),
                            "gradient_norm": round(float(gradient_norm.item()), 6),
                            "learning_rate": learning_rate,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        train_loss = _finish_loss(totals)
        dev_loss = _development_loss(model, dev_records, device=resolved_device)
        _require_unchanged_sources(source_hashes)
        checkpoint_name = f"epoch-{epoch}.pt" if epoch < EPOCHS else "candidate.pt"
        checkpoint = _save_checkpoint(output / checkpoint_name, model)
        checkpoints.append(checkpoint)
        epoch_report = {
            "epoch": epoch,
            "train_loss": train_loss["loss"],
            "state_loss": train_loss["state_loss"],
            "pointer_loss": train_loss["pointer_loss"],
            "dev_loss": dev_loss["loss"],
            "dev_state_loss": dev_loss["state_loss"],
            "dev_pointer_loss": dev_loss["pointer_loss"],
            "seconds": time.monotonic() - epoch_started,
            "checkpoint": checkpoint,
        }
        epoch_reports.append(epoch_report)
        print(
            json.dumps({"event": "epoch_complete", **epoch_report}, sort_keys=True),
            flush=True,
        )

    candidate = checkpoints[-1]
    report = {
        "schema_version": POINTER_TRAINING_REPORT_SCHEMA_VERSION,
        "recipe": POINTER_TRAINING_RECIPE_VERSION,
        "status": "complete",
        "seed": seed,
        "device": resolved_device,
        "parameter_count": NANO_POINTER_PARAMETER_COUNT,
        "trunk_parameter_count": NANO_TRUNK_PARAMETER_COUNT,
        "pointer_head_parameter_count": POINTER_HEAD_PARAMETER_COUNT,
        "architecture_identity": FROZEN_NANO_V01.architecture_identity,
        "base_checkpoint_sha256": FROZEN_NANO_V01.checkpoint_sha256,
        "tokenizer_sha256": FROZEN_NANO_V01.tokenizer_sha256,
        "dataset_manifest_sha256": bundle.manifest_sha256,
        "dataset": {
            "schema_version": DATASET_SCHEMA_VERSION,
            "target_grammar": TARGET_GRAMMAR_VERSION,
            "train_sha256": bundle.manifest["train"]["sha256"],
            "dev_sha256": bundle.manifest["dev"]["sha256"],
            "train_records": len(bundle.train),
            "dev_records": len(bundle.dev),
        },
        "hyperparameters": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "paired_variants_per_world": 4,
            "peak_learning_rate": PEAK_LEARNING_RATE,
            "warmup_fraction": WARMUP_FRACTION,
            "cosine_floor": COSINE_FLOOR,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRADIENT_CLIP,
            "adam_betas": list(ADAM_BETAS),
            "adam_epsilon": ADAM_EPSILON,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "state_class_order": [state.value for state in STATE_ORDER],
            "state_class_counts": {
                state.value: count
                for state, count in zip(
                    STATE_ORDER, observed_train_state_counts, strict=True
                )
            },
            "state_class_weights": list(STATE_CLASS_WEIGHTS),
            "state_loss_weight": STATE_LOSS_WEIGHT,
            "state_loss_definition": STATE_LOSS_DEFINITION,
            "pointer_loss_weight": POINTER_LOSS_WEIGHT,
            "pointer_loss_definition": (
                "mean_of_start_and_end_cross_entropy_active_slots"
            ),
            "patient_token_masked": True,
            "prompt_template_id": POINTER_PROMPT_TEMPLATE_ID,
            "supervision_version": POINTER_SUPERVISION_VERSION,
            "uncertain_pointer_count": STATE_POINTER_COUNTS[
                STATE_ORDER.index(FieldState.UNCERTAIN)
            ],
            "causal_pointer_heads": True,
            "deterministic_algorithms": True,
            "full_trunk_trainable": True,
            "world_grouped_batches": True,
        },
        "epochs": epoch_reports,
        "candidate": candidate,
        "source_sha256": source_hashes,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "tokenizers": tokenizers.__version__,
            "cuda": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(torch.cuda.current_device())
                if resolved_device == "cuda"
                else None
            ),
            "cublas_workspace_config": (
                os.environ.get("CUBLAS_WORKSPACE_CONFIG")
                if resolved_device == "cuda"
                else None
            ),
            "platform": platform.platform(),
            "seconds": time.monotonic() - started,
        },
        "selection_note": (
            "This is an unselected H2 development candidate. Historical fresh-v0 "
            "and the sealed fresh-v1 confirmation partition were not read. Causal "
            "pointer logits at earlier transcript tokens can use prefix context only."
        ),
    }
    _require_unchanged_sources(source_hashes)
    _write_json_no_clobber(output / "training_report.json", report)
    print(
        json.dumps(
            {
                "event": "training_complete",
                "seed": seed,
                "candidate": candidate,
                "report": "training_report.json",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return report


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Train Nano's H2 pointer candidate")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=root / "checkpoints" / "anchors" / "nano_v01_scribe.pt",
    )
    parser.add_argument(
        "--tokenizer", type=Path, default=root / "sft" / "tokenizer.json"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=TRAINING_SEEDS, required=True)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    train_pointer_candidate(
        data_dir=args.data_dir,
        base_checkpoint=args.base_checkpoint,
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "POINTER_TRAINING_RECIPE_VERSION",
    "POINTER_TRAINING_REPORT_SCHEMA_VERSION",
    "STATE_CLASS_COUNTS",
    "STATE_CLASS_WEIGHTS",
    "STATE_LOSS_DEFINITION",
    "PointerBatch",
    "PointerLoss",
    "collate_pointer_batch",
    "pointer_objective",
    "train_pointer_candidate",
]
