"""Frozen H6 state-conditioned-boundary trainer for Nano's scribe core.

H6 changes only one representation mechanism on top of H5: the model's
detached soft semantic-state posterior selects zero-initialized residuals for
the existing evidence-boundary queries.
It reuses byte-exact H5 fit and calibration data plus H5's tokenizer, prompt,
objective, optimizer, schedule, exposure, seeds, threshold policy, and
selection order.  This module never opens development, benchmark,
historical-fresh, sealed-confirmation, or legacy training-record artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import tokenizers
import torch

from nano_ai.adapters.anchor_checkpoint import FROZEN_NANO_V01
from nano_ai.contract import FieldState
from nano_ai.training import replay_mixture_data, state_conditioned_evidence_query_model
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
from nano_ai.training.state_conditioned_evidence_query_model import (
    ARCHITECTURE_VERSION,
)
from nano_ai.training.state_conditioned_evidence_query_model import (
    EVIDENCE_QUERY_HEAD_PARAMETER_COUNT_H6 as EVIDENCE_QUERY_HEAD_PARAMETER_COUNT,
)
from nano_ai.training.state_conditioned_evidence_query_model import (
    NANO_EVIDENCE_QUERY_PARAMETER_COUNT_H6 as NANO_EVIDENCE_QUERY_PARAMETER_COUNT,
)
from nano_ai.training.state_conditioned_evidence_query_model import (
    NanoStateConditionedEvidenceQueryPointerModel as NanoEvidenceQueryPointerModel,
)
from nano_ai.training.state_span_data import (
    DATASET_SCHEMA_VERSION,
    TARGET_GRAMMAR_VERSION,
    StateSpanExample,
)
from nano_ai.training.train_evidence_query import (
    _accumulate_loss,
    _calibrate_model,
    _finish_loss,
    _loss_aggregates,
    _optimizer,
    _resolve_evidence_query_device,
    _save_checkpoint,
    _write_json_no_clobber,
    select_epoch_report,
)
from nano_ai.training.train_evidence_query_h4 import (
    CALIBRATION_STATE_CLASS_COUNTS,
    PRESERVED_H3_SOURCE_SHA256,
    preserved_source_paths,
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
    COSINE_FLOOR,
    GRADIENT_CLIP,
    LOG_EVERY_STEPS,
    PEAK_LEARNING_RATE,
    WARMUP_FRACTION,
    WEIGHT_DECAY,
    TrainingInputError,
    _seed_training,
    grouped_batch_indices,
    learning_rate_at,
)

H6_TRAINING_REPORT_SCHEMA_VERSION = "nano.evidence-query-h6-training-report.v1"
H6_TRAINING_RECIPE_VERSION = "nano-evidence-query-h6-state-boundary-residual-v1"

FIT_RECORD_COUNT = replay_mixture_data.FIT_RECORDS
CALIBRATION_RECORD_COUNT = replay_mixture_data.CALIBRATION_RECORDS
FIT_WORLD_COUNT = replay_mixture_data.TOTAL_FIT_WORLDS
CALIBRATION_WORLD_COUNT = replay_mixture_data.CALIBRATION_WORLDS
SOURCE_WORLD_COUNT = replay_mixture_data.SOURCE_WORLDS
SOURCE_RECORD_COUNT = replay_mixture_data.SOURCE_RECORDS
VARIANTS_PER_WORLD = replay_mixture_data.VARIANTS_PER_WORLD
TRAINING_SEEDS = replay_mixture_data.TRAINING_SEEDS
BATCH_SIZE = replay_mixture_data.BATCH_SIZE
EPOCHS = replay_mixture_data.EPOCHS
STEPS_PER_EPOCH = replay_mixture_data.STEPS_PER_EPOCH
STEPS_PER_SEED = replay_mixture_data.STEPS_PER_SEED

H5_DATASET_FILE_IDENTITY = {
    "manifest": {
        "filename": "manifest.json",
        "bytes": 31_213,
        "sha256": "2569e7b27b53ef741fc92d1545ca323367155d960e77ff0e6852b32db0d32f31",
    },
    "fit": {
        "filename": "fit.jsonl",
        "bytes": 9_134_638,
        "sha256": "79f7581efbd989f48a474bdd2079e7ca163c2a2682bb48fb8398bd99febe22fb",
    },
    "calibration": {
        "filename": "calibration.jsonl",
        "bytes": 734_627,
        "sha256": "ce0562ccb44ee83963eace0d873773addaee8e49f29499a6b720b16335930e70",
    },
}

REPORT_DATASET_KEYS = frozenset(
    {
        "schema_version",
        "generator",
        "generator_sha256",
        "selection_policy",
        "target_grammar",
        "normalization",
        "training_identity",
        "source_manifest",
        "source_fit",
        "source_calibration",
        "fit",
        "calibration",
        "sources",
        "overlap_audit",
        "restrictions",
    }
)

FIT_STATE_CLASS_COUNTS: tuple[int, ...] = tuple(
    replay_mixture_data.EXPECTED_STATE_CLASS_COUNTS[state.value]
    for state in STATE_ORDER
)

_TRAINING_SELECTION_NOTE = (
    "The selected H6 epoch used only H5's byte-exact 200-world "
    "training-calibration partition. Gradients used only H5's byte-exact fixed "
    "1,400-world legacy plus 1,400-world surface replay fit mixture. The "
    "training command "
    "read no legacy training-record artifact, development, benchmark, "
    "historical-fresh, or sealed-confirmation records. H5's trunk, evidence "
    "reads, state head, boundary projections, objective, optimizer, exposure, "
    "seeds, inference policy, calibration, and selection rules were preserved. "
    "Only detached-posterior, zero-initialized state residuals on the boundary "
    "queries changed."
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def changed_source_paths() -> Mapping[str, Path]:
    """Return the H6 executable authorities authenticated during training."""

    return {
        "model": Path(state_conditioned_evidence_query_model.__file__).resolve(),
        "training": Path(__file__).resolve(),
    }


def _hash_paths(paths: Mapping[str, Path], *, label: str) -> dict[str, str]:
    missing = sorted(name for name, path in paths.items() if not path.is_file())
    if missing:
        raise TrainingInputError(f"{label} source is incomplete: " + ", ".join(missing))
    return {name: _sha256(path.read_bytes()) for name, path in sorted(paths.items())}


def _require_preserved_h3_sources() -> dict[str, str]:
    observed = _hash_paths(preserved_source_paths(), label="preserved H3")
    if observed != dict(PRESERVED_H3_SOURCE_SHA256):
        changed = sorted(
            name
            for name in set(observed) | set(PRESERVED_H3_SOURCE_SHA256)
            if observed.get(name) != PRESERVED_H3_SOURCE_SHA256.get(name)
        )
        raise TrainingInputError(
            "H6 preserved H3 source hash mismatch: " + ", ".join(changed)
        )
    return observed


def _changed_source_hashes() -> dict[str, str]:
    return _hash_paths(changed_source_paths(), label="changed H6")


def _require_unchanged_sources(changed_expected: Mapping[str, str]) -> None:
    _require_preserved_h3_sources()
    if _changed_source_hashes() != dict(changed_expected):
        raise TrainingInputError("H6 changed source changed during execution")


def _read_file(path: Path, *, label: str) -> bytes:
    try:
        return Path(path).read_bytes()
    except OSError as exc:
        raise TrainingInputError(f"H6 {label} is unavailable") from exc


def _file_sha256(path: Path, *, label: str) -> str:
    return _sha256(_read_file(path, label=label))


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
        raise TrainingInputError("H6 fit state-class distribution drifted")
    if calibration_counts != CALIBRATION_STATE_CLASS_COUNTS:
        raise TrainingInputError("H6 calibration state-class distribution drifted")
    if (
        tuple(
            fit + calibration
            for fit, calibration in zip(fit_counts, calibration_counts, strict=True)
        )
        != STATE_CLASS_COUNTS
    ):
        raise RuntimeError("H6 partitions do not reconstruct frozen class counts")
    return fit_counts, calibration_counts


def _require_exact_training_steps(
    examples: Sequence[StateSpanExample], *, seed: int
) -> Mapping[int, tuple[tuple[int, ...], ...]]:
    batches_by_epoch: dict[int, tuple[tuple[int, ...], ...]] = {}
    for epoch in range(1, EPOCHS + 1):
        batches = grouped_batch_indices(
            examples,
            batch_size=BATCH_SIZE,
            seed=seed,
            epoch=epoch,
        )
        if len(batches) != STEPS_PER_EPOCH or any(
            len(batch) != BATCH_SIZE for batch in batches
        ):
            raise TrainingInputError("H6 exposure is not exactly 350 full batches")
        flattened = [index for batch in batches for index in batch]
        if sorted(flattened) != list(range(FIT_RECORD_COUNT)):
            raise TrainingInputError("H6 epoch does not expose each fit record once")
        batches_by_epoch[epoch] = batches
    if sum(len(value) for value in batches_by_epoch.values()) != STEPS_PER_SEED:
        raise TrainingInputError("H6 exposure is not exactly 1,050 steps per seed")
    return batches_by_epoch


def _require_input_hashes(
    *,
    data_dir: Path,
    base_checkpoint: Path,
    tokenizer_path: Path,
    expected: Mapping[str, str],
) -> None:
    paths = {
        "manifest": Path(data_dir) / "manifest.json",
        "fit": Path(data_dir) / "fit.jsonl",
        "calibration": Path(data_dir) / "calibration.jsonl",
        "base_checkpoint": Path(base_checkpoint),
        "tokenizer": Path(tokenizer_path),
    }
    observed = {name: _file_sha256(path, label=name) for name, path in paths.items()}
    if observed != dict(expected):
        raise TrainingInputError("H6 authenticated input changed during execution")


def _dataset_file_identity(path: Path) -> dict[str, Any]:
    snapshot = _read_file(path, label=path.name)
    return {
        "filename": path.name,
        "bytes": len(snapshot),
        "sha256": _sha256(snapshot),
    }


def _require_frozen_h5_dataset(data_dir: Path) -> dict[str, str]:
    """Require the byte-exact H5 manifest, fit, and calibration authorities."""

    root = Path(data_dir)
    observed = {
        role: _dataset_file_identity(root / identity["filename"])
        for role, identity in H5_DATASET_FILE_IDENTITY.items()
    }
    if observed != H5_DATASET_FILE_IDENTITY:
        raise TrainingInputError("H6 input is not the byte-exact frozen H5 dataset")
    return {
        role: identity["sha256"] for role, identity in H5_DATASET_FILE_IDENTITY.items()
    }


def train_evidence_query_h6_candidate(
    *,
    data_dir: Path,
    base_checkpoint: Path,
    tokenizer_path: Path,
    output_dir: Path,
    seed: int,
    device: str,
) -> Mapping[str, Any]:
    """Train one H6 seed after complete fail-closed, no-output preflight."""

    if seed not in TRAINING_SEEDS:
        raise TrainingInputError(
            f"seed must be one of the frozen seeds {TRAINING_SEEDS}"
        )
    output = Path(output_dir)
    if output.exists():
        raise TrainingInputError("candidate output directory must not exist")

    started = time.monotonic()
    resolved_device = _resolve_evidence_query_device(device)
    preserved_hashes = _require_preserved_h3_sources()
    changed_hashes = _changed_source_hashes()

    base_sha256 = _file_sha256(Path(base_checkpoint), label="base checkpoint")
    tokenizer_sha256 = _file_sha256(Path(tokenizer_path), label="tokenizer")
    if base_sha256 != FROZEN_NANO_V01.checkpoint_sha256:
        raise TrainingInputError("H6 base checkpoint identity drifted")
    if tokenizer_sha256 != FROZEN_NANO_V01.tokenizer_sha256:
        raise TrainingInputError("H6 tokenizer identity drifted")

    frozen_data_hashes = _require_frozen_h5_dataset(Path(data_dir))
    bundle = replay_mixture_data.load_replay_mixture_dataset(Path(data_dir))
    if bundle.input_sha256 != frozen_data_hashes:
        raise TrainingInputError("H6 replay loader returned another dataset identity")
    tokenizer = load_pointer_tokenizer(Path(tokenizer_path))
    fit_records = encode_pointer_partition(
        tokenizer,
        bundle.fit,
        expected_split="train",
    )
    calibration_records = encode_pointer_partition(
        tokenizer,
        bundle.calibration,
        expected_split="train",
    )
    fit_counts, calibration_counts = _require_partition_state_distributions(
        fit_records, calibration_records
    )
    batches_by_epoch = _require_exact_training_steps(bundle.fit, seed=seed)

    authenticated_inputs = {
        **bundle.input_sha256,
        "base_checkpoint": base_sha256,
        "tokenizer": tokenizer_sha256,
    }
    _require_unchanged_sources(changed_hashes)
    _require_input_hashes(
        data_dir=Path(data_dir),
        base_checkpoint=Path(base_checkpoint),
        tokenizer_path=Path(tokenizer_path),
        expected=authenticated_inputs,
    )

    _seed_training(seed)
    model = NanoEvidenceQueryPointerModel.from_frozen_base(Path(base_checkpoint))
    if (
        model.architecture_version != ARCHITECTURE_VERSION
        or model.parameter_count != NANO_EVIDENCE_QUERY_PARAMETER_COUNT
        or model.head_parameter_count != EVIDENCE_QUERY_HEAD_PARAMETER_COUNT
        or model.trunk.config.parameter_count != NANO_TRUNK_PARAMETER_COUNT
    ):
        raise RuntimeError("H6 model identity drifted before training")
    model.to(resolved_device).train()
    optimizer = _optimizer(model)
    _require_input_hashes(
        data_dir=Path(data_dir),
        base_checkpoint=Path(base_checkpoint),
        tokenizer_path=Path(tokenizer_path),
        expected=authenticated_inputs,
    )
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise TrainingInputError("candidate output directory must not exist") from exc

    global_step = 0
    epoch_reports: list[dict[str, Any]] = []
    print(
        json.dumps(
            {
                "event": "training_start",
                "recipe": H6_TRAINING_RECIPE_VERSION,
                "seed": seed,
                "device": resolved_device,
                "fit_records": len(fit_records),
                "fit_legacy_worlds": SOURCE_WORLD_COUNT,
                "fit_surface_worlds": SOURCE_WORLD_COUNT,
                "calibration_records": len(calibration_records),
                "development_records_used": 0,
                "steps": STEPS_PER_SEED,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for epoch in range(1, EPOCHS + 1):
        totals = _loss_aggregates()
        epoch_started = time.monotonic()
        for indices in batches_by_epoch[epoch]:
            global_step += 1
            learning_rate = learning_rate_at(global_step, total_steps=STEPS_PER_SEED)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            batch = collate_pointer_batch(
                [fit_records[index] for index in indices],
                device=resolved_device,
            )
            loss = pointer_objective(
                model(batch.token_ids, attention_mask=batch.attention_mask), batch
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
                            "steps": STEPS_PER_SEED,
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
        _require_unchanged_sources(changed_hashes)
        _require_input_hashes(
            data_dir=Path(data_dir),
            base_checkpoint=Path(base_checkpoint),
            tokenizer_path=Path(tokenizer_path),
            expected=authenticated_inputs,
        )
        checkpoint = _save_checkpoint(output / f"epoch-{epoch}.pt", model)
        calibration = _calibrate_model(
            model,
            bundle.calibration,
            calibration_records,
            device=resolved_device,
        )
        model.train()
        _require_unchanged_sources(changed_hashes)
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

    if global_step != STEPS_PER_SEED:
        raise RuntimeError("H6 completed with an unexpected optimizer-step count")
    selected = select_epoch_report(epoch_reports)
    selected_checkpoint = dict(selected["checkpoint"])
    candidate = {"epoch": selected["epoch"], **selected_checkpoint}

    fit_identity = dict(bundle.manifest["partitions"]["fit"])
    calibration_identity = dict(bundle.manifest["partitions"]["calibration"])
    dataset_report = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "generator": replay_mixture_data.GENERATOR_VERSION,
        "generator_sha256": bundle.manifest["generator_sha256"],
        "selection_policy": replay_mixture_data.SELECTION_POLICY_VERSION,
        "target_grammar": TARGET_GRAMMAR_VERSION,
        "normalization": dict(bundle.manifest["normalization"]),
        "training_identity": dict(bundle.manifest["training_identity"]),
        "source_manifest": _dataset_file_identity(Path(data_dir) / "manifest.json"),
        "source_fit": {
            **_dataset_file_identity(Path(data_dir) / "fit.jsonl"),
            "records": len(bundle.fit),
            "worlds": FIT_WORLD_COUNT,
            "legacy_worlds": SOURCE_WORLD_COUNT,
            "surface_worlds": SOURCE_WORLD_COUNT,
            "gradient_bearing": True,
        },
        "source_calibration": {
            **_dataset_file_identity(Path(data_dir) / "calibration.jsonl"),
            "records": len(bundle.calibration),
            "worlds": CALIBRATION_WORLD_COUNT,
            "namespace": "train-calibration",
            "gradient_bearing": False,
            "reused_unchanged_from_h4": True,
        },
        "fit": {
            **fit_identity,
            "state_class_counts": {
                state.value: count
                for state, count in zip(STATE_ORDER, fit_counts, strict=True)
            },
        },
        "calibration": {
            **calibration_identity,
            "state_class_counts": {
                state.value: count
                for state, count in zip(STATE_ORDER, calibration_counts, strict=True)
            },
        },
        "sources": dict(bundle.manifest["sources"]),
        "overlap_audit": dict(bundle.manifest["overlap_audit"]),
        "restrictions": dict(bundle.manifest["restrictions"]),
    }
    if set(dataset_report) != REPORT_DATASET_KEYS:
        raise RuntimeError("H6 training dataset report contract drifted")

    report = {
        "schema_version": H6_TRAINING_REPORT_SCHEMA_VERSION,
        "recipe": H6_TRAINING_RECIPE_VERSION,
        "status": "complete",
        "seed": seed,
        "device": resolved_device,
        "architecture_version": ARCHITECTURE_VERSION,
        "parameter_count": NANO_EVIDENCE_QUERY_PARAMETER_COUNT,
        "trunk_parameter_count": NANO_TRUNK_PARAMETER_COUNT,
        "evidence_query_head_parameter_count": EVIDENCE_QUERY_HEAD_PARAMETER_COUNT,
        "architecture_identity": FROZEN_NANO_V01.architecture_identity,
        "base_checkpoint_sha256": base_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "dataset_manifest_sha256": bundle.manifest_sha256,
        "dataset": dataset_report,
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
            "steps_per_epoch": STEPS_PER_EPOCH,
            "total_steps": STEPS_PER_SEED,
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
            "soft_state_conditioned_boundaries": True,
            "state_conditioning_location": "start_end_query_residual",
            "state_posterior_detached": True,
            "gold_state_teacher_forced": False,
            "state_conditioning_parameter_count": 640,
            "state_conditioning_zero_initialized": True,
            "deterministic_algorithms": True,
            "full_trunk_trainable": True,
            "world_grouped_batches": True,
        },
        "epochs": epoch_reports,
        "candidate": candidate,
        "calibration": {
            "selected_epoch": selected["epoch"],
            **selected["calibration"],
        },
        "dev_used_for_selection": False,
        "legacy_record_artifact_accessed": False,
        "fresh_v1_accessed": False,
        "preserved_source_sha256": preserved_hashes,
        "changed_source_sha256": changed_hashes,
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
        "selection_note": _TRAINING_SELECTION_NOTE,
    }
    _require_unchanged_sources(changed_hashes)
    _require_input_hashes(
        data_dir=Path(data_dir),
        base_checkpoint=Path(base_checkpoint),
        tokenizer_path=Path(tokenizer_path),
        expected=authenticated_inputs,
    )
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
        description="Train Nano's frozen H6 replay-mixture candidate"
    )
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
    train_evidence_query_h6_candidate(
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
    "BATCH_SIZE",
    "CALIBRATION_RECORD_COUNT",
    "CALIBRATION_STATE_CLASS_COUNTS",
    "CALIBRATION_WORLD_COUNT",
    "EPOCHS",
    "FIT_RECORD_COUNT",
    "FIT_STATE_CLASS_COUNTS",
    "FIT_WORLD_COUNT",
    "H5_DATASET_FILE_IDENTITY",
    "H6_TRAINING_RECIPE_VERSION",
    "H6_TRAINING_REPORT_SCHEMA_VERSION",
    "SOURCE_RECORD_COUNT",
    "SOURCE_WORLD_COUNT",
    "STEPS_PER_EPOCH",
    "STEPS_PER_SEED",
    "TRAINING_SEEDS",
    "VARIANTS_PER_WORLD",
    "changed_source_paths",
    "train_evidence_query_h6_candidate",
]
