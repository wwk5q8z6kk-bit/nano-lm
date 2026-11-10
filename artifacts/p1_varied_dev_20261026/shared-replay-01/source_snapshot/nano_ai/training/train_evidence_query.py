"""Frozen architecture-only H3 trainer for Nano's evidence-query heads.

H3 changes only the bounded state/pointer head.  It preserves H2's tokenizer,
prompt, generated data family, weighted state-plus-pointer objective, optimizer,
schedule, epoch count, and seeds.  The original training split is partitioned
by world identity before tokenization: worlds 0000--2799 carry gradients and
worlds 2800--2999 are training-only calibration data.  The development and
fresh-confirmation partitions are never encoded or used for selection here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import tokenizers
import torch

from nano_ai.adapters import state_span
from nano_ai.adapters.anchor_checkpoint import FROZEN_NANO_V01
from nano_ai.adapters.state_span import parse_state_span_summary
from nano_ai.contract import FieldState
from nano_ai.training import (
    evaluate_pointer,
    pointer_data,
    state_span_data,
    train_pointer,
    train_state_span,
)
from nano_ai.training.evaluate_pointer import build_pointer_inference_inputs
from nano_ai.training.evidence_query_model import (
    ARCHITECTURE_VERSION,
    EVIDENCE_QUERY_HEAD_PARAMETER_COUNT,
    NANO_EVIDENCE_QUERY_PARAMETER_COUNT,
    NanoEvidenceQueryPointerModel,
)
from nano_ai.training.pointer_data import (
    POINTER_PROMPT_TEMPLATE_ID,
    POINTER_SUPERVISION_VERSION,
    STATE_ORDER,
    STATE_POINTER_COUNTS,
    PointerSupervision,
    encode_pointer_partition,
    load_pointer_tokenizer,
)
from nano_ai.training.pointer_model import NANO_TRUNK_PARAMETER_COUNT
from nano_ai.training.state_span_data import (
    DATASET_SCHEMA_VERSION,
    TARGET_GRAMMAR_VERSION,
    StateSpanExample,
    canonical_json_bytes,
)
from nano_ai.training.train_pointer import (
    POINTER_LOSS_WEIGHT,
    STATE_CLASS_COUNTS,
    STATE_CLASS_WEIGHTS,
    STATE_LOSS_DEFINITION,
    STATE_LOSS_WEIGHT,
    collate_pointer_batch,
    pointer_objective,
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

EVIDENCE_QUERY_TRAINING_REPORT_SCHEMA_VERSION = "nano.evidence-query-training-report.v0"
EVIDENCE_QUERY_TRAINING_RECIPE_VERSION = "nano-evidence-query-architecture-only-v0"

FIT_WORLD_START = 0
FIT_WORLD_STOP = 2_800
CALIBRATION_WORLD_START = FIT_WORLD_STOP
CALIBRATION_WORLD_STOP = 3_000
FIT_WORLD_COUNT = FIT_WORLD_STOP - FIT_WORLD_START
CALIBRATION_WORLD_COUNT = CALIBRATION_WORLD_STOP - CALIBRATION_WORLD_START
VARIANTS_PER_WORLD = 4
FIT_RECORD_COUNT = FIT_WORLD_COUNT * VARIANTS_PER_WORLD
CALIBRATION_RECORD_COUNT = CALIBRATION_WORLD_COUNT * VARIANTS_PER_WORLD
FIT_STATE_CLASS_COUNTS: tuple[int, ...] = (42_980, 4_620, 2_800, 2_800, 2_800)
CALIBRATION_STATE_CLASS_COUNTS: tuple[int, ...] = (3_070, 330, 200, 200, 200)
CALIBRATION_SELECTION_SLICES = (
    "absence",
    "missing_target",
    "uncertain_target",
    "conflicting_target",
)
CALIBRATION_THRESHOLD_POLICY = "minimal_zero_wrong_presented_inclusive_v1"

_WORLD_ID_RE = re.compile(r"train-world-(\d{4})\Z")
_DETERMINISTIC_CUBLAS_CONFIGS = frozenset({":4096:8", ":16:8"})


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def training_source_paths() -> Mapping[str, Path]:
    """Return the exact source files authenticated by every H3 report."""

    training_dir = Path(__file__).resolve().parent
    return {
        "base_model": training_dir / "model.py",
        "data_generator": Path(state_span_data.__file__).resolve(),
        "evidence_query_inference": training_dir / "evidence_query_inference.py",
        "evidence_query_model": training_dir / "evidence_query_model.py",
        "h1_training_loader": Path(train_state_span.__file__).resolve(),
        "h2_objective": Path(train_pointer.__file__).resolve(),
        "pointer_data": Path(pointer_data.__file__).resolve(),
        "pointer_decoder": Path(evaluate_pointer.__file__).resolve(),
        "state_span_adapter": Path(state_span.__file__).resolve(),
        "training": Path(__file__).resolve(),
    }


def expected_training_source_paths() -> Mapping[str, Path]:
    """Compatibility alias for strict report/evaluator source authentication."""

    return training_source_paths()


def _source_hashes() -> dict[str, str]:
    paths = training_source_paths()
    missing = sorted(name for name, path in paths.items() if not path.is_file())
    if missing:
        raise TrainingInputError(
            "H3 training source is incomplete: " + ", ".join(missing)
        )
    return {name: _sha256(path.read_bytes()) for name, path in sorted(paths.items())}


def _require_unchanged_sources(expected: Mapping[str, str]) -> None:
    if _source_hashes() != dict(expected):
        raise TrainingInputError(
            "evidence-query training source changed during execution"
        )


def _world_index(example: StateSpanExample) -> int:
    if not isinstance(example, StateSpanExample):
        raise TrainingInputError("H3 data must contain StateSpanExample records")
    match = _WORLD_ID_RE.fullmatch(example.world_id)
    if match is None or example.split != "train":
        raise TrainingInputError("H3 source record is not a frozen training world")
    return int(match.group(1))


def split_fit_calibration_worlds(
    examples: Sequence[StateSpanExample],
) -> tuple[tuple[StateSpanExample, ...], tuple[StateSpanExample, ...]]:
    """Split H2 training worlds before encoding and prove the boundary is exact."""

    frozen = tuple(examples)
    if len(frozen) != FIT_RECORD_COUNT + CALIBRATION_RECORD_COUNT:
        raise TrainingInputError("H3 requires the exact 3,000-world H2 training split")

    grouped: dict[int, list[StateSpanExample]] = {}
    for example in frozen:
        index = _world_index(example)
        if index < FIT_WORLD_START or index >= CALIBRATION_WORLD_STOP:
            raise TrainingInputError("H3 training world index is outside 0000-2999")
        grouped.setdefault(index, []).append(example)
    if set(grouped) != set(range(FIT_WORLD_START, CALIBRATION_WORLD_STOP)):
        raise TrainingInputError("H3 training world identities are not complete")
    expected_variants = {"normal", "missing", "uncertain", "conflicting"}
    for index, records in grouped.items():
        if (
            len(records) != VARIANTS_PER_WORLD
            or {record.variant for record in records} != expected_variants
            or len({record.target_field for record in records}) != 1
        ):
            raise TrainingInputError(
                f"H3 training world {index:04d} is not one complete paired family"
            )

    fit = tuple(
        example
        for example in frozen
        if FIT_WORLD_START <= _world_index(example) < FIT_WORLD_STOP
    )
    calibration = tuple(
        example
        for example in frozen
        if CALIBRATION_WORLD_START <= _world_index(example) < CALIBRATION_WORLD_STOP
    )
    if len(fit) != FIT_RECORD_COUNT or len(calibration) != CALIBRATION_RECORD_COUNT:
        raise RuntimeError("H3 fit/calibration split cardinality drifted")
    if {record.world_id for record in fit} & {
        record.world_id for record in calibration
    }:
        raise RuntimeError("H3 fit and calibration worlds overlap")
    return fit, calibration


def _record_partition_sha256(examples: Sequence[StateSpanExample]) -> str:
    return _sha256(
        b"".join(canonical_json_bytes(example.to_dict()) for example in examples)
    )


def _transcript_multiset_sha256(examples: Sequence[StateSpanExample]) -> str:
    transcript_hashes = sorted(
        _sha256(example.transcript.encode("utf-8")) for example in examples
    )
    return _sha256("\n".join(transcript_hashes).encode("utf-8"))


def _partition_identity(
    examples: Sequence[StateSpanExample],
    *,
    role: str,
    first_world: int,
    final_world: int,
) -> dict[str, Any]:
    worlds = {example.world_id for example in examples}
    return {
        "role": role,
        "records": len(examples),
        "worlds": len(worlds),
        "first_world_id": f"train-world-{first_world:04d}",
        "final_world_id": f"train-world-{final_world:04d}",
        "records_sha256": _record_partition_sha256(examples),
        "transcript_multiset_sha256": _transcript_multiset_sha256(examples),
        "gradient_bearing": role == "fit",
    }


def _state_class_counts(records: Sequence[PointerSupervision]) -> tuple[int, ...]:
    counts = [0] * len(STATE_ORDER)
    for record in records:
        for label in record.state_labels:
            counts[label] += 1
    return tuple(counts)


def _require_partition_state_distributions(
    fit_records: Sequence[PointerSupervision],
    calibration_records: Sequence[PointerSupervision],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    fit_counts = _state_class_counts(fit_records)
    calibration_counts = _state_class_counts(calibration_records)
    if fit_counts != FIT_STATE_CLASS_COUNTS:
        raise TrainingInputError("H3 fit state-class distribution is not frozen")
    if calibration_counts != CALIBRATION_STATE_CLASS_COUNTS:
        raise TrainingInputError(
            "H3 calibration state-class distribution is not frozen"
        )
    if (
        tuple(
            fit + calibration
            for fit, calibration in zip(fit_counts, calibration_counts, strict=True)
        )
        != STATE_CLASS_COUNTS
    ):
        raise RuntimeError("H3 partition class counts do not reconstruct H2")
    return fit_counts, calibration_counts


def _resolve_evidence_query_device(requested: str) -> str:
    """Resolve compute and fail before I/O if deterministic CUDA is not configured."""

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


def _optimizer(model: NanoEvidenceQueryPointerModel) -> torch.optim.AdamW:
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
        "state_weight_sum": 0.0,
        "pointer_sum": 0.0,
        "pointer_count": 0,
    }


def _accumulate_loss(values: dict[str, float | int], loss: Any) -> None:
    values["state_sum"] += float(loss.state.item()) * loss.state_weight_sum
    values["state_weight_sum"] += loss.state_weight_sum
    values["pointer_sum"] += float(loss.pointer.item()) * loss.pointer_count
    values["pointer_count"] += loss.pointer_count


def _finish_loss(values: Mapping[str, float | int]) -> dict[str, float]:
    if float(values["state_weight_sum"]) <= 0 or int(values["pointer_count"]) <= 0:
        raise RuntimeError("H3 training loss aggregation is empty")
    state = float(values["state_sum"]) / float(values["state_weight_sum"])
    pointer = float(values["pointer_sum"]) / int(values["pointer_count"])
    return {
        "loss": STATE_LOSS_WEIGHT * state + POINTER_LOSS_WEIGHT * pointer,
        "state_loss": state,
        "pointer_loss": pointer,
    }


def _save_checkpoint(
    path: Path,
    model: NanoEvidenceQueryPointerModel,
) -> dict[str, Any]:
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
    return {
        "filename": path.name,
        "sha256": _sha256(snapshot),
        "bytes": len(snapshot),
    }


def _write_json_no_clobber(path: Path, value: Any) -> None:
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def _validate_rate_bucket(
    value: object,
    label: str,
    *,
    allow_empty: bool = False,
) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "numerator",
        "denominator",
        "rate",
    }:
        raise TrainingInputError(f"{label} calibration bucket is malformed")
    numerator = value["numerator"]
    denominator = value["denominator"]
    rate = value["rate"]
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator < 0
        or (denominator == 0 and not allow_empty)
        or not 0 <= numerator <= denominator
        or isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or float(rate) != (numerator / denominator if denominator else 0.0)
    ):
        raise TrainingInputError(f"{label} calibration bucket is inconsistent")
    return value


def _validate_metric_set(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "slices",
        "selection",
        "wrong_presented",
    }:
        raise TrainingInputError(f"{label} calibration metrics are malformed")
    slices = value["slices"]
    if not isinstance(slices, dict) or set(slices) != {
        *CALIBRATION_SELECTION_SLICES,
        "overall",
    }:
        raise TrainingInputError(f"{label} calibration slices are malformed")
    for name, bucket in slices.items():
        _validate_rate_bucket(bucket, f"{label}.{name}")
    wrong_presented = _validate_rate_bucket(
        value["wrong_presented"],
        f"{label}.wrong_presented",
        allow_empty=True,
    )
    selection = value["selection"]
    if not isinstance(selection, dict) or set(selection) != {
        "macro_joint",
        "overall_joint",
    }:
        raise TrainingInputError(f"{label} calibration selection is malformed")
    macro = sum(
        float(slices[name]["rate"]) for name in CALIBRATION_SELECTION_SLICES
    ) / len(CALIBRATION_SELECTION_SLICES)
    overall = float(slices["overall"]["rate"])
    if selection["macro_joint"] != macro or selection["overall_joint"] != overall:
        raise TrainingInputError(f"{label} calibration ranking is inconsistent")
    if wrong_presented["denominator"] > slices["overall"]["denominator"]:
        raise TrainingInputError(f"{label} presented denominator is impossible")
    return value


def _validate_calibration_summary(value: object) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "uncalibrated",
        "global_threshold",
        "calibrated",
        "threshold_policy",
    }:
        raise TrainingInputError("H3 calibration result has unexpected keys")
    uncalibrated = _validate_metric_set(value["uncalibrated"], "uncalibrated")
    calibrated = _validate_metric_set(value["calibrated"], "calibrated")
    threshold = value["global_threshold"]
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0.0 <= float(threshold) <= 1.0
    ):
        raise TrainingInputError("H3 global calibration threshold is invalid")
    if value["threshold_policy"] != CALIBRATION_THRESHOLD_POLICY:
        raise TrainingInputError("H3 calibration threshold policy drifted")
    if calibrated["wrong_presented"]["numerator"] != 0:
        raise TrainingInputError("H3 calibrated result is not zero-wrong-presented")
    if uncalibrated["slices"]["overall"][
        "denominator"
    ] != CALIBRATION_RECORD_COUNT * len(STATE_ORDER):
        raise TrainingInputError("H3 calibration overall denominator drifted")
    return value


def _calibrate_model(
    model: NanoEvidenceQueryPointerModel,
    examples: Sequence[StateSpanExample],
    records: Sequence[PointerSupervision],
    *,
    device: str,
) -> Mapping[str, Any]:
    """Call the shared inference authority; no trainer-local decoder is allowed."""

    try:
        from nano_ai.training.evidence_query_inference import (
            CalibrationGold,
            batched_evidence_query_inference,
            select_global_threshold,
        )
    except ImportError as exc:  # pragma: no cover - source preflight
        raise TrainingInputError(
            "shared H3 evidence-query calibration is unavailable"
        ) from exc

    inputs = build_pointer_inference_inputs(examples, records)
    model.eval()
    with torch.inference_mode():
        inference = batched_evidence_query_inference(
            model,
            inputs,
            device=device,
            batch_size=BATCH_SIZE,
        )
    gold_rows = tuple(
        parse_state_span_summary(example.target, example.transcript)
        for example in examples
    )
    gold = tuple(
        CalibrationGold(example_id=example.example_id, proposals=proposals)
        for example, proposals in zip(examples, gold_rows, strict=True)
    )
    selection = select_global_threshold(inference, gold)
    summary = selection.to_dict()
    return _validate_calibration_summary(summary)


def _epoch_selection_key(epoch_report: Mapping[str, Any]) -> tuple[float, float, int]:
    epoch = epoch_report.get("epoch")
    calibration = epoch_report.get("calibration")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise TrainingInputError("H3 epoch report has an invalid epoch")
    validated = _validate_calibration_summary(calibration)
    ranking = validated["uncalibrated"]["selection"]
    return (
        float(ranking["macro_joint"]),
        float(ranking["overall_joint"]),
        -epoch,
    )


def select_epoch_report(
    epoch_reports: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Select by frozen calibration ranking, breaking exact ties earlier."""

    reports = tuple(epoch_reports)
    if len(reports) != EPOCHS or {report.get("epoch") for report in reports} != set(
        range(1, EPOCHS + 1)
    ):
        raise TrainingInputError("H3 selection requires exactly three epoch reports")
    return max(reports, key=_epoch_selection_key)


def _dataset_file_identity(data_dir: Path, filename: str) -> dict[str, Any]:
    path = Path(data_dir) / filename
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise TrainingInputError(f"H3 dataset file is unavailable: {filename}") from exc
    return {"filename": filename, "bytes": size}


def train_evidence_query_candidate(
    *,
    data_dir: Path,
    base_checkpoint: Path,
    tokenizer_path: Path,
    output_dir: Path,
    seed: int,
    device: str,
) -> Mapping[str, Any]:
    """Train one fixed H3 seed and create all evidence without overwriting."""

    if seed not in TRAINING_SEEDS:
        raise TrainingInputError(
            f"seed must be one of the frozen seeds {TRAINING_SEEDS}"
        )
    output = Path(output_dir)
    if output.exists():
        raise TrainingInputError("candidate output directory must not exist")

    started = time.monotonic()
    resolved_device = _resolve_evidence_query_device(device)
    source_hashes = _source_hashes()
    bundle = load_training_bundle(Path(data_dir))
    fit_examples, calibration_examples = split_fit_calibration_worlds(bundle.train)

    # The boundary above is intentionally established on source records before
    # either subset reaches the tokenizer.
    tokenizer = load_pointer_tokenizer(Path(tokenizer_path))
    fit_records = encode_pointer_partition(
        tokenizer,
        fit_examples,
        expected_split="train",
    )
    calibration_records = encode_pointer_partition(
        tokenizer,
        calibration_examples,
        expected_split="train",
    )
    fit_counts, calibration_counts = _require_partition_state_distributions(
        fit_records,
        calibration_records,
    )

    _seed_training(seed)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise TrainingInputError("candidate output directory must not exist") from exc

    model = NanoEvidenceQueryPointerModel.from_frozen_base(Path(base_checkpoint))
    if (
        model.architecture_version != ARCHITECTURE_VERSION
        or model.parameter_count != NANO_EVIDENCE_QUERY_PARAMETER_COUNT
        or model.head_parameter_count != EVIDENCE_QUERY_HEAD_PARAMETER_COUNT
        or model.trunk.config.parameter_count != NANO_TRUNK_PARAMETER_COUNT
    ):
        raise RuntimeError("H3 model identity drifted before training")
    model.to(resolved_device).train()
    optimizer = _optimizer(model)

    first_epoch_batches = grouped_batch_indices(
        fit_examples,
        batch_size=BATCH_SIZE,
        seed=seed,
        epoch=1,
    )
    if any(len(batch) != BATCH_SIZE for batch in first_epoch_batches):
        raise TrainingInputError("H3 fit worlds do not fill exact batches")
    steps_per_epoch = len(first_epoch_batches)
    total_steps = steps_per_epoch * EPOCHS
    global_step = 0
    epoch_reports: list[dict[str, Any]] = []

    print(
        json.dumps(
            {
                "event": "training_start",
                "recipe": EVIDENCE_QUERY_TRAINING_RECIPE_VERSION,
                "seed": seed,
                "device": resolved_device,
                "fit_records": len(fit_records),
                "calibration_records": len(calibration_records),
                "development_records_used": 0,
                "steps": total_steps,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for epoch in range(1, EPOCHS + 1):
        batches = grouped_batch_indices(
            fit_examples,
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
                [fit_records[index] for index in indices],
                device=resolved_device,
            )
            loss = pointer_objective(
                model(batch.token_ids, attention_mask=batch.attention_mask),
                batch,
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
        _require_unchanged_sources(source_hashes)
        checkpoint = _save_checkpoint(output / f"epoch-{epoch}.pt", model)
        calibration = _calibrate_model(
            model,
            calibration_examples,
            calibration_records,
            device=resolved_device,
        )
        model.train()
        _require_unchanged_sources(source_hashes)
        epoch_report = {
            "epoch": epoch,
            "train_loss": train_loss["loss"],
            "state_loss": train_loss["state_loss"],
            "pointer_loss": train_loss["pointer_loss"],
            "seconds": time.monotonic() - epoch_started,
            "checkpoint": checkpoint,
            "calibration": calibration,
        }
        epoch_reports.append(epoch_report)
        print(
            json.dumps({"event": "epoch_complete", **epoch_report}, sort_keys=True),
            flush=True,
        )

    selected = select_epoch_report(epoch_reports)
    selected_checkpoint = dict(selected["checkpoint"])
    candidate = {"epoch": selected["epoch"], **selected_checkpoint}
    selected_calibration = selected["calibration"]

    report = {
        "schema_version": EVIDENCE_QUERY_TRAINING_REPORT_SCHEMA_VERSION,
        "recipe": EVIDENCE_QUERY_TRAINING_RECIPE_VERSION,
        "status": "complete",
        "seed": seed,
        "device": resolved_device,
        "architecture_version": ARCHITECTURE_VERSION,
        "parameter_count": NANO_EVIDENCE_QUERY_PARAMETER_COUNT,
        "trunk_parameter_count": NANO_TRUNK_PARAMETER_COUNT,
        "evidence_query_head_parameter_count": EVIDENCE_QUERY_HEAD_PARAMETER_COUNT,
        "architecture_identity": FROZEN_NANO_V01.architecture_identity,
        "base_checkpoint_sha256": FROZEN_NANO_V01.checkpoint_sha256,
        "tokenizer_sha256": FROZEN_NANO_V01.tokenizer_sha256,
        "dataset_manifest_sha256": bundle.manifest_sha256,
        "dataset": {
            "schema_version": DATASET_SCHEMA_VERSION,
            "target_grammar": TARGET_GRAMMAR_VERSION,
            "source_manifest": {
                **_dataset_file_identity(Path(data_dir), "manifest.json"),
                "sha256": bundle.manifest_sha256,
            },
            "source_train": {
                **_dataset_file_identity(Path(data_dir), "train.jsonl"),
                "sha256": bundle.manifest["train"]["sha256"],
                "records": len(bundle.train),
                "worlds": bundle.manifest["train"]["worlds"],
            },
            "source_dev": {
                **_dataset_file_identity(Path(data_dir), "dev.jsonl"),
                "sha256": bundle.manifest["dev"]["sha256"],
                "records": len(bundle.dev),
                "worlds": bundle.manifest["dev"]["worlds"],
                "usage": "source_authentication_only",
            },
            "fit": {
                **_partition_identity(
                    fit_examples,
                    role="fit",
                    first_world=FIT_WORLD_START,
                    final_world=FIT_WORLD_STOP - 1,
                ),
                "state_class_counts": {
                    state.value: count
                    for state, count in zip(STATE_ORDER, fit_counts, strict=True)
                },
            },
            "calibration": {
                **_partition_identity(
                    calibration_examples,
                    role="calibration",
                    first_world=CALIBRATION_WORLD_START,
                    final_world=CALIBRATION_WORLD_STOP - 1,
                ),
                "state_class_counts": {
                    state.value: count
                    for state, count in zip(
                        STATE_ORDER, calibration_counts, strict=True
                    )
                },
            },
        },
        "hyperparameters": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "paired_variants_per_world": VARIANTS_PER_WORLD,
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
            "state_class_weight_source_counts": {
                state.value: count
                for state, count in zip(STATE_ORDER, STATE_CLASS_COUNTS, strict=True)
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
            "full_context_evidence_queries": True,
            "shared_state_classifier": True,
            "deterministic_algorithms": True,
            "full_trunk_trainable": True,
            "world_grouped_batches": True,
        },
        "epochs": epoch_reports,
        "candidate": candidate,
        "calibration": {
            "selected_epoch": selected["epoch"],
            **selected_calibration,
        },
        "dev_used_for_selection": False,
        "fresh_v1_accessed": False,
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
            "The selected H3 epoch was chosen using calibration worlds "
            "2800-2999. Gradients used only disjoint fit worlds 0000-2799. The "
            "inspected "
            "development partition was authenticated as part of the frozen H2 "
            "data bundle but received no inference, loss, threshold, or selection "
            "access. Historical fresh-v0 and sealed fresh-v1 were not read."
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
    parser = argparse.ArgumentParser(
        description="Train Nano's H3 evidence-query candidate"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=root / "checkpoints" / "anchors" / "nano_v01_scribe.pt",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=root / "sft" / "tokenizer.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=TRAINING_SEEDS, required=True)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    train_evidence_query_candidate(
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
    "CALIBRATION_RECORD_COUNT",
    "CALIBRATION_SELECTION_SLICES",
    "CALIBRATION_THRESHOLD_POLICY",
    "CALIBRATION_WORLD_COUNT",
    "EVIDENCE_QUERY_TRAINING_RECIPE_VERSION",
    "EVIDENCE_QUERY_TRAINING_REPORT_SCHEMA_VERSION",
    "FIT_RECORD_COUNT",
    "FIT_WORLD_COUNT",
    "expected_training_source_paths",
    "select_epoch_report",
    "split_fit_calibration_worlds",
    "train_evidence_query_candidate",
    "training_source_paths",
]
