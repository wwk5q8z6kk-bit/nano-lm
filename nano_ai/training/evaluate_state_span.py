"""Sealed-development comparison for Nano's native state/span intervention.

This evaluator has one deliberately narrow evidence boundary: the deterministic
``state_span_data`` development split.  It does not import, discover, or read any
historical benchmark partition.  A manifest digest supplied by the caller roots
the development data identity; direct checkpoint digests or digest-pinned
training reports root candidate identities.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import platform
import re
import stat
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from nano_ai.adapters.anchor_checkpoint import (
    DECODE_POLICY_ID,
    FROZEN_NANO_V01,
    PROMPT_TEMPLATE_ID,
)
from nano_ai.adapters.deterministic_v0 import _extract_fields
from nano_ai.adapters.legacy_summary import LegacySummarySolver
from nano_ai.adapters.state_checkpoint import (
    STATE_PROMPT_TEMPLATE_ID,
    STATE_SPAN_PROMPT_INSTRUCTION,
    build_state_span_prompt,
)
from nano_ai.adapters.state_span import (
    StateSpanFormatError,
    StateSpanProposal,
    StateSpanSolver,
    parse_state_span_summary,
)
from nano_ai.contract import (
    FIELD_ORDER,
    FieldName,
    FieldOutput,
    FieldState,
    NanoInput,
    NanoOutput,
)
from nano_ai.evaluation import EvaluationReport, evaluate_solver
from nano_ai.fixtures import FixtureCase
from nano_ai.training import state_span_data
from nano_ai.training.model import (
    NANO_MODEL_CONFIG,
    NanoGPT,
    load_frozen_nano_state_dict,
    load_verified_nano_state_dict,
)
from nano_ai.training.state_span_data import (
    DATASET_SCHEMA_VERSION,
    DEV_SEED,
    DEV_WORLDS,
    MANIFEST_SCHEMA_VERSION,
    TARGET_GRAMMAR_VERSION,
    TRAIN_SEED,
    TRAIN_WORLDS,
    StateSpanExample,
    generate_split,
)

DEVELOPMENT_EVALUATION_SCHEMA_VERSION = "nano.state-span-development-evaluation.v0"
TRAINING_REPORT_SCHEMA_VERSION = "nano.state-span-training-report.v0"
TRAINING_RECIPE_VERSION = "nano-native-state-span-sft-v0"
DEVELOPMENT_PARTITION_ID = "native-state-span-dev-v0"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_NEW_TOKENS = 128

_LABEL_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_PRESENTED_STATES = frozenset({FieldState.SUPPORTED, FieldState.ABSENT})
_EXPECTED_ISOLATION = {
    "worlds_disjoint": True,
    "transcripts_disjoint": True,
    "open_value_lexicons_disjoint": True,
    "question_templates_disjoint": True,
    "answer_templates_disjoint": True,
    "denial_phrases_disjoint": True,
    "uncertainty_phrases_disjoint": True,
    "fresh_v0_read_by_generator": False,
}
_MANIFEST_KEYS = {
    "schema_version",
    "target_grammar",
    "generator_sha256",
    "tokenizer_sha256",
    "base_checkpoint_sha256",
    "train",
    "dev",
    "isolation",
}
_MANIFEST_SECTION_KEYS = {
    "seed",
    "records",
    "worlds",
    "sha256",
    "transcript_multiset_sha256",
    "state_field_quota",
}
_TRAINING_REPORT_KEYS = {
    "schema_version",
    "recipe",
    "status",
    "seed",
    "device",
    "parameter_count",
    "architecture_identity",
    "base_checkpoint_sha256",
    "tokenizer_sha256",
    "dataset_manifest_sha256",
    "dataset",
    "prompt",
    "hyperparameters",
    "epochs",
    "candidate",
    "source_sha256",
    "runtime",
    "selection_note",
}
_CHECKPOINT_KEYS = {"filename", "sha256", "bytes"}


class DevelopmentEvaluationError(ValueError):
    """A sealed input, artifact identity, or evaluation invariant is invalid."""


@dataclass(frozen=True, slots=True)
class CandidateCheckpoint:
    """One explicitly named and digest-pinned candidate checkpoint."""

    label: str
    path: Path
    sha256: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or _LABEL_RE.fullmatch(self.label) is None:
            raise DevelopmentEvaluationError(
                "candidate label must match [a-z0-9][a-z0-9._-]{0,127}"
            )
        object.__setattr__(self, "path", Path(self.path))
        _require_sha256(self.sha256, "candidate checkpoint")
        if not isinstance(self.provenance, Mapping) or any(
            not isinstance(key, str) for key in self.provenance
        ):
            raise DevelopmentEvaluationError(
                "candidate provenance must be a string-keyed mapping"
            )


@dataclass(frozen=True, slots=True)
class DevelopmentBundle:
    """The exact verified development snapshot used for model selection."""

    manifest: Mapping[str, Any]
    manifest_sha256: str
    dev_sha256: str
    examples: tuple[StateSpanExample, ...]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DevelopmentEvaluationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _read_regular_file(path: Path, *, role: str) -> bytes:
    try:
        with Path(path).open("rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise DevelopmentEvaluationError(f"{role} is not a regular file")
            return handle.read()
    except DevelopmentEvaluationError:
        raise
    except OSError as exc:
        raise DevelopmentEvaluationError(f"{role} is unavailable") from exc


def _read_verified_file(path: Path, expected_sha256: str, *, role: str) -> bytes:
    expected = _require_sha256(expected_sha256, role)
    snapshot = _read_regular_file(path, role=role)
    observed = _sha256(snapshot)
    if not hmac.compare_digest(observed, expected):
        raise DevelopmentEvaluationError(
            f"{role} SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return snapshot


def _reject_json_constant(value: str) -> None:
    raise DevelopmentEvaluationError(f"non-finite JSON constant is forbidden: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DevelopmentEvaluationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json(snapshot: bytes, *, role: str) -> Any:
    try:
        return json.loads(
            snapshot.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except DevelopmentEvaluationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise DevelopmentEvaluationError(f"{role} is invalid JSON") from exc


def _parse_json_lines(snapshot: bytes, *, role: str) -> tuple[StateSpanExample, ...]:
    records: list[StateSpanExample] = []
    for line_number, line in enumerate(snapshot.splitlines(), 1):
        try:
            value = _parse_json(line, role=f"{role} line {line_number}")
            records.append(StateSpanExample.from_dict(value))
        except (TypeError, ValueError) as exc:
            raise DevelopmentEvaluationError(
                f"{role} has an invalid record at line {line_number}"
            ) from exc
    if not records:
        raise DevelopmentEvaluationError(f"{role} is empty")
    return tuple(records)


def _state_field_quota(per_state_field: int) -> dict[str, int]:
    return {
        f"{state.value}:{field.value}": per_state_field
        for state in (
            FieldState.MISSING,
            FieldState.UNCERTAIN,
            FieldState.CONFLICTING,
        )
        for field in FIELD_ORDER
    }


def _transcript_multiset_sha256(examples: Sequence[StateSpanExample]) -> str:
    hashes = sorted(_sha256(example.transcript.encode("utf-8")) for example in examples)
    return _sha256("\n".join(hashes).encode("utf-8"))


def _validate_manifest_metadata(
    section: object,
    *,
    split: str,
    seed: int,
    worlds: int,
) -> Mapping[str, Any]:
    if not isinstance(section, dict) or set(section) != _MANIFEST_SECTION_KEYS:
        raise DevelopmentEvaluationError(f"manifest {split} section is invalid")
    for key in ("sha256", "transcript_multiset_sha256"):
        _require_sha256(section[key], f"manifest {split} {key}")
    if (
        section["seed"] != seed
        or section["records"] != worlds * 4
        or section["worlds"] != worlds
        or section["state_field_quota"]
        != _state_field_quota(worlds // len(FIELD_ORDER))
    ):
        raise DevelopmentEvaluationError(f"manifest {split} recipe is not frozen")
    return section


def _validate_development_records(examples: tuple[StateSpanExample, ...]) -> None:
    if len(examples) != DEV_WORLDS * 4:
        raise DevelopmentEvaluationError("development record count is not frozen")
    if any(example.split != "dev" for example in examples):
        raise DevelopmentEvaluationError("development data contains another split")
    if len({example.example_id for example in examples}) != len(examples):
        raise DevelopmentEvaluationError("development example IDs are not unique")
    grouped: dict[str, list[StateSpanExample]] = defaultdict(list)
    for example in examples:
        grouped[example.world_id].append(example)
    if len(grouped) != DEV_WORLDS:
        raise DevelopmentEvaluationError("development world count is not frozen")
    expected_variants = {"normal", "missing", "uncertain", "conflicting"}
    for world_id, world_examples in grouped.items():
        if (
            len(world_examples) != 4
            or {example.variant for example in world_examples} != expected_variants
            or len({example.target_field for example in world_examples}) != 1
        ):
            raise DevelopmentEvaluationError(
                f"development world {world_id} is not a complete paired family"
            )
    if examples != generate_split("dev"):
        raise DevelopmentEvaluationError(
            "development records do not equal deterministic frozen generation"
        )


def load_development_bundle(
    data_dir: str | Path,
    *,
    expected_manifest_sha256: str,
) -> DevelopmentBundle:
    """Verify only the sealed development snapshot; the train file is not read."""

    root = Path(data_dir)
    manifest_snapshot = _read_verified_file(
        root / "manifest.json",
        expected_manifest_sha256,
        role="state/span manifest",
    )
    manifest = _parse_json(manifest_snapshot, role="state/span manifest")
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        raise DevelopmentEvaluationError("state/span manifest schema is invalid")
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise DevelopmentEvaluationError("state/span manifest version is unsupported")
    if manifest["target_grammar"] != TARGET_GRAMMAR_VERSION:
        raise DevelopmentEvaluationError("state/span target grammar changed")
    if manifest["tokenizer_sha256"] != FROZEN_NANO_V01.tokenizer_sha256:
        raise DevelopmentEvaluationError("manifest tokenizer is not frozen Nano v0.1")
    if manifest["base_checkpoint_sha256"] != FROZEN_NANO_V01.checkpoint_sha256:
        raise DevelopmentEvaluationError("manifest base is not frozen Nano v0.1")
    if manifest["isolation"] != _EXPECTED_ISOLATION:
        raise DevelopmentEvaluationError("manifest isolation contract changed")

    generator_snapshot = _read_regular_file(
        Path(state_span_data.__file__), role="state/span generator source"
    )
    if manifest["generator_sha256"] != _sha256(generator_snapshot):
        raise DevelopmentEvaluationError(
            "state/span generator source changed after the manifest was sealed"
        )

    _validate_manifest_metadata(
        manifest["train"], split="train", seed=TRAIN_SEED, worlds=TRAIN_WORLDS
    )
    section = _validate_manifest_metadata(
        manifest["dev"], split="dev", seed=DEV_SEED, worlds=DEV_WORLDS
    )

    dev_snapshot = _read_verified_file(
        root / "dev.jsonl",
        section["sha256"],
        role="state/span development data",
    )
    examples = _parse_json_lines(dev_snapshot, role="state/span development data")
    _validate_development_records(examples)
    if _transcript_multiset_sha256(examples) != section["transcript_multiset_sha256"]:
        raise DevelopmentEvaluationError(
            "development transcript multiset does not match the manifest"
        )
    return DevelopmentBundle(
        manifest=manifest,
        manifest_sha256=_sha256(manifest_snapshot),
        dev_sha256=_sha256(dev_snapshot),
        examples=examples,
    )


def _checkpoint_from_report(
    report_path: Path,
    checkpoint: Mapping[str, Any],
    *,
    label: str,
    provenance: Mapping[str, Any],
) -> CandidateCheckpoint:
    if set(checkpoint) != _CHECKPOINT_KEYS:
        raise DevelopmentEvaluationError("training checkpoint record is invalid")
    filename = checkpoint["filename"]
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise DevelopmentEvaluationError("training checkpoint filename is unsafe")
    _require_sha256(checkpoint["sha256"], "training checkpoint")
    if (
        isinstance(checkpoint["bytes"], bool)
        or not isinstance(checkpoint["bytes"], int)
        or checkpoint["bytes"] <= 0
    ):
        raise DevelopmentEvaluationError("training checkpoint byte count is invalid")
    return CandidateCheckpoint(
        label=label,
        path=report_path.parent / filename,
        sha256=checkpoint["sha256"],
        provenance={**provenance, "artifact_bytes": checkpoint["bytes"]},
    )


def _training_source_hashes() -> dict[str, str]:
    files = {
        "data_generator": Path(state_span_data.__file__),
        "model": Path(__file__).with_name("model.py"),
        "state_adapter": Path(__file__).parents[1] / "adapters" / "state_span.py",
        "state_checkpoint": Path(__file__).parents[1]
        / "adapters"
        / "state_checkpoint.py",
        "training": Path(__file__).with_name("train_state_span.py"),
    }
    return {
        name: _sha256(_read_regular_file(path, role=f"training {name} source"))
        for name, path in sorted(files.items())
    }


def load_candidates_from_training_report(
    report_path: str | Path,
    *,
    expected_report_sha256: str,
    expected_manifest_sha256: str,
    expected_dev_sha256: str,
    expected_train_sha256: str | None = None,
) -> tuple[CandidateCheckpoint, ...]:
    """Authenticate a report before using its embedded checkpoint digests."""

    path = Path(report_path)
    snapshot = _read_verified_file(
        path,
        expected_report_sha256,
        role="state/span training report",
    )
    report = _parse_json(snapshot, role="state/span training report")
    if not isinstance(report, dict) or set(report) != _TRAINING_REPORT_KEYS:
        raise DevelopmentEvaluationError("training report schema is invalid")
    if (
        report["schema_version"] != TRAINING_REPORT_SCHEMA_VERSION
        or report["recipe"] != TRAINING_RECIPE_VERSION
        or report["status"] != "complete"
    ):
        raise DevelopmentEvaluationError("training report is not a complete v0 run")
    if (
        isinstance(report["seed"], bool)
        or not isinstance(report["seed"], int)
        or report["parameter_count"] != NANO_MODEL_CONFIG.parameter_count
        or report["architecture_identity"] != FROZEN_NANO_V01.architecture_identity
        or report["base_checkpoint_sha256"] != FROZEN_NANO_V01.checkpoint_sha256
        or report["tokenizer_sha256"] != FROZEN_NANO_V01.tokenizer_sha256
    ):
        raise DevelopmentEvaluationError("training report model identity is invalid")
    if report["dataset_manifest_sha256"] != _require_sha256(
        expected_manifest_sha256, "expected manifest"
    ):
        raise DevelopmentEvaluationError(
            "training report used another dataset manifest"
        )

    dataset = report["dataset"]
    if not isinstance(dataset, dict) or set(dataset) != {
        "schema_version",
        "target_grammar",
        "train_sha256",
        "dev_sha256",
        "train_records",
        "dev_records",
    }:
        raise DevelopmentEvaluationError("training report dataset record is invalid")
    train_sha256 = _require_sha256(dataset["train_sha256"], "training dataset")
    expected_dev_sha256 = _require_sha256(
        expected_dev_sha256, "expected development data"
    )
    if (
        dataset["schema_version"] != DATASET_SCHEMA_VERSION
        or dataset["target_grammar"] != TARGET_GRAMMAR_VERSION
        or dataset["train_records"] != TRAIN_WORLDS * 4
        or dataset["dev_records"] != DEV_WORLDS * 4
    ):
        raise DevelopmentEvaluationError("training report dataset recipe is invalid")
    if dataset["dev_sha256"] != expected_dev_sha256:
        raise DevelopmentEvaluationError("training report used another development set")
    if expected_train_sha256 is not None and train_sha256 != _require_sha256(
        expected_train_sha256, "expected training data"
    ):
        raise DevelopmentEvaluationError("training report used another training set")

    prompt = report["prompt"]
    instruction_sha256 = _sha256(STATE_SPAN_PROMPT_INSTRUCTION.encode("utf-8"))
    if not isinstance(prompt, dict) or prompt != {
        "template_id": STATE_PROMPT_TEMPLATE_ID,
        "instruction_sha256": instruction_sha256,
    }:
        raise DevelopmentEvaluationError("training report prompt identity changed")
    source_hashes = report["source_sha256"]
    if not isinstance(source_hashes, dict):
        raise DevelopmentEvaluationError("training report source hashes are missing")
    for name, digest in source_hashes.items():
        if not isinstance(name, str) or not name:
            raise DevelopmentEvaluationError("training report source name is invalid")
        _require_sha256(digest, f"training source {name}")
    if source_hashes != _training_source_hashes():
        raise DevelopmentEvaluationError(
            "training report source hashes do not match the executable recipe"
        )

    epochs = report["epochs"]
    if not isinstance(epochs, list) or len(epochs) != 3:
        raise DevelopmentEvaluationError("training report must contain three epochs")
    candidates: list[CandidateCheckpoint] = []
    seed = report["seed"]
    report_identity = _sha256(snapshot)
    for expected_epoch, epoch_row in enumerate(epochs, 1):
        if not isinstance(epoch_row, dict) or set(epoch_row) != {
            "epoch",
            "train_loss",
            "dev_loss",
            "seconds",
            "checkpoint",
        }:
            raise DevelopmentEvaluationError("training epoch record is invalid")
        if epoch_row["epoch"] != expected_epoch:
            raise DevelopmentEvaluationError("training epochs are not ordered")
        for metric in ("train_loss", "dev_loss", "seconds"):
            value = epoch_row[metric]
            if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
                raise DevelopmentEvaluationError(f"training epoch {metric} is invalid")
        provenance = {
            "kind": "verified_training_report",
            "training_report_sha256": report_identity,
            "training_recipe": report["recipe"],
            "training_seed": seed,
            "epoch": expected_epoch,
            "train_loss": epoch_row["train_loss"],
            "dev_loss": epoch_row["dev_loss"],
            "training_source_sha256": dict(sorted(source_hashes.items())),
        }
        candidates.append(
            _checkpoint_from_report(
                path,
                epoch_row["checkpoint"],
                label=f"seed-{seed}-epoch-{expected_epoch}",
                provenance=provenance,
            )
        )
    if report["candidate"] != epochs[-1]["checkpoint"]:
        raise DevelopmentEvaluationError(
            "training report candidate is not the final epoch checkpoint"
        )
    return tuple(candidates)


def _resolve_device(device: str) -> str:
    if device not in {"cpu", "mps", "cuda"}:
        raise DevelopmentEvaluationError("device must be cpu, mps, or cuda")
    if device == "mps" and not torch.backends.mps.is_available():
        raise DevelopmentEvaluationError("MPS was requested but is unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        raise DevelopmentEvaluationError("CUDA was requested but is unavailable")
    return device


def encode_chatml_prompts(
    tokenizer: Tokenizer,
    prompts: Sequence[str],
) -> tuple[tuple[int, ...], ...]:
    """Encode the exact user/assistant prefix shared with frozen training."""

    start_token = tokenizer.token_to_id("<|im_start|>")
    end_token = tokenizer.token_to_id("<|im_end|>")
    if start_token is None or end_token is None:
        raise DevelopmentEvaluationError("tokenizer is missing ChatML tokens")
    user_header = tokenizer.encode("user\n", add_special_tokens=False).ids
    assistant_header = tokenizer.encode("assistant\n", add_special_tokens=False).ids
    encoded: list[tuple[int, ...]] = []
    for prompt in prompts:
        if not isinstance(prompt, str) or not prompt.strip():
            raise DevelopmentEvaluationError("generation prompts must contain text")
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        prefix = tuple(
            [start_token]
            + user_header
            + prompt_ids
            + [end_token, start_token]
            + assistant_header
        )
        if len(prefix) >= NANO_MODEL_CONFIG.sequence_length:
            raise DevelopmentEvaluationError(
                "development prompt exceeds Nano's context window"
            )
        encoded.append(prefix)
    if not encoded:
        raise DevelopmentEvaluationError("generation requires at least one prompt")
    return tuple(encoded)


def batched_greedy_generate(
    model: NanoGPT,
    tokenizer: Tokenizer,
    prompt_token_ids: Sequence[Sequence[int]],
    *,
    device: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> tuple[str, ...]:
    """Greedily decode equal-prefix-length groups without padding contamination."""

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise DevelopmentEvaluationError("batch_size must be a positive integer")
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens < 1
    ):
        raise DevelopmentEvaluationError("max_new_tokens must be a positive integer")
    end_token = tokenizer.token_to_id("<|im_end|>")
    if end_token is None:
        raise DevelopmentEvaluationError("tokenizer is missing <|im_end|>")
    frozen_prompts = tuple(tuple(tokens) for tokens in prompt_token_ids)
    if not frozen_prompts or any(not tokens for tokens in frozen_prompts):
        raise DevelopmentEvaluationError("encoded prompts must be non-empty")
    if any(
        len(tokens) >= NANO_MODEL_CONFIG.sequence_length for tokens in frozen_prompts
    ):
        raise DevelopmentEvaluationError("encoded prompt exceeds Nano's context window")

    groups: dict[int, list[int]] = defaultdict(list)
    for index, tokens in enumerate(frozen_prompts):
        groups[len(tokens)].append(index)
    decoded: list[str | None] = [None] * len(frozen_prompts)
    model.eval()
    with torch.inference_mode():
        for prompt_length in sorted(groups):
            indices = groups[prompt_length]
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                generated = torch.tensor(
                    [frozen_prompts[index] for index in batch_indices],
                    dtype=torch.long,
                    device=device,
                )
                finished = torch.zeros(
                    len(batch_indices), dtype=torch.bool, device=device
                )
                output_ids: list[list[int]] = [[] for _ in batch_indices]
                available = NANO_MODEL_CONFIG.sequence_length - prompt_length
                for _ in range(min(max_new_tokens, available)):
                    next_tokens = model(generated)[:, -1].argmax(dim=-1)
                    next_tokens = torch.where(
                        finished,
                        torch.full_like(next_tokens, end_token),
                        next_tokens,
                    )
                    token_values = next_tokens.detach().cpu().tolist()
                    for row, token in enumerate(token_values):
                        if not bool(finished[row].item()) and token != end_token:
                            output_ids[row].append(token)
                    finished = finished | next_tokens.eq(end_token)
                    if bool(finished.all().item()):
                        break
                    generated = torch.cat([generated, next_tokens[:, None]], dim=1)
                for row, index in enumerate(batch_indices):
                    decoded[index] = tokenizer.decode(output_ids[row]).strip()
    if any(value is None for value in decoded):  # pragma: no cover - loop invariant.
        raise RuntimeError("generation did not produce every requested row")
    return tuple(value for value in decoded if value is not None)


def _load_model(
    checkpoint_path: Path,
    *,
    expected_sha256: str,
    device: str,
    frozen_base: bool,
) -> NanoGPT:
    state_dict = (
        load_frozen_nano_state_dict(checkpoint_path)
        if frozen_base
        else load_verified_nano_state_dict(
            checkpoint_path, expected_sha256=expected_sha256
        )
    )
    model = NanoGPT()
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def _fixture_cases(examples: Sequence[StateSpanExample]) -> tuple[FixtureCase, ...]:
    cases: list[FixtureCase] = []
    for example in examples:
        request = NanoInput(item_id=example.example_id, transcript=example.transcript)
        gold = NanoOutput(
            item_id=example.example_id,
            solver_id="native-state-span-dev-gold-v0",
            fields=_extract_fields(request),
        )
        cases.append(
            FixtureCase(
                case_id=example.example_id,
                partition=DEVELOPMENT_PARTITION_ID,
                request=request,
                gold=gold,
                provenance={
                    "role": "sealed_development_model_selection",
                    "world_id": example.world_id,
                    "variant": example.variant,
                    "target_field": example.target_field.value,
                    "target_state": (
                        None
                        if example.target_state is None
                        else example.target_state.value
                    ),
                },
            )
        )
    return tuple(cases)


def _prediction_map(
    examples: Sequence[StateSpanExample], summaries: Sequence[str]
) -> dict[str, str]:
    if len(examples) != len(summaries):
        raise DevelopmentEvaluationError("generation returned the wrong row count")
    predictions: dict[str, str] = {}
    for example, summary in zip(examples, summaries, strict=True):
        previous = predictions.setdefault(example.transcript, summary)
        if previous != summary:
            raise DevelopmentEvaluationError(
                "identical transcripts received inconsistent generated summaries"
            )
    return predictions


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _span_key(span: Any) -> tuple[int, int, str, str]:
    return (span.start, span.end, span.text, span.speaker)


def _proposal_exact(proposal: StateSpanProposal, gold: FieldOutput) -> bool:
    return proposal.state is gold.state and {
        _span_key(span) for span in proposal.spans
    } == {_span_key(span) for span in gold.evidence}


def _empty_raw_bucket() -> dict[str, int]:
    return {
        "total": 0,
        "parsed": 0,
        "state_correct": 0,
        "exact": 0,
        "presented": 0,
        "wrong_presented": 0,
    }


def _finish_raw_bucket(bucket: dict[str, int]) -> dict[str, int | float | None]:
    return {
        **bucket,
        "parse_rate": _rate(bucket["parsed"], bucket["total"]),
        "state_accuracy": _rate(bucket["state_correct"], bucket["total"]),
        "exact_accuracy": _rate(bucket["exact"], bucket["total"]),
        "wrong_presented_rate": _rate(bucket["wrong_presented"], bucket["presented"]),
    }


def raw_state_span_diagnostics(
    examples: Sequence[StateSpanExample],
    cases: Sequence[FixtureCase],
    summaries: Sequence[str],
) -> dict[str, Any]:
    """Score raw state/span syntax separately from verifier-gated outputs."""

    if not (len(examples) == len(cases) == len(summaries)):
        raise DevelopmentEvaluationError("raw diagnostics row counts do not match")
    aggregate = _empty_raw_bucket()
    by_field = {field.value: _empty_raw_bucket() for field in FIELD_ORDER}
    by_gold_state = {state.value: _empty_raw_bucket() for state in FieldState}
    target_challenge = {
        state.value: _empty_raw_bucket()
        for state in (
            FieldState.MISSING,
            FieldState.UNCERTAIN,
            FieldState.CONFLICTING,
        )
    }
    malformed_items = 0
    item_rows: list[dict[str, Any]] = []

    for example, case, summary in zip(examples, cases, summaries, strict=True):
        for field_name in FIELD_ORDER:
            gold = case.gold.field(field_name)
            aggregate["total"] += 1
            by_field[field_name.value]["total"] += 1
            by_gold_state[gold.state.value]["total"] += 1
        if example.target_state is not None:
            target_challenge[example.target_state.value]["total"] += 1
        try:
            proposals = parse_state_span_summary(summary, example.transcript)
        except StateSpanFormatError as exc:
            malformed_items += 1
            item_rows.append(
                {
                    "example_id": example.example_id,
                    "variant": example.variant,
                    "target_field": example.target_field.value,
                    "target_state": (
                        None
                        if example.target_state is None
                        else example.target_state.value
                    ),
                    "raw_summary": summary,
                    "parse_status": "malformed",
                    "parse_error": str(exc),
                    "fields": None,
                }
            )
            continue

        field_rows: list[dict[str, Any]] = []
        for proposal in proposals:
            gold = case.gold.field(proposal.field)
            exact = _proposal_exact(proposal, gold)
            state_correct = proposal.state is gold.state
            presented = proposal.state in _PRESENTED_STATES
            wrong_presented = presented and not exact
            buckets = (
                aggregate,
                by_field[proposal.field.value],
                by_gold_state[gold.state.value],
            )
            for bucket in buckets:
                bucket["parsed"] += 1
                bucket["state_correct"] += int(state_correct)
                bucket["exact"] += int(exact)
                bucket["presented"] += int(presented)
                bucket["wrong_presented"] += int(wrong_presented)
            if (
                example.target_state is not None
                and proposal.field is example.target_field
            ):
                target = target_challenge[example.target_state.value]
                target["parsed"] += 1
                target["state_correct"] += int(state_correct)
                target["exact"] += int(exact)
                target["presented"] += int(presented)
                target["wrong_presented"] += int(wrong_presented)
            field_rows.append(
                {
                    "field": proposal.field.value,
                    "raw_state": proposal.state.value,
                    "gold_state": gold.state.value,
                    "state_correct": state_correct,
                    "exact": exact,
                    "presented": presented,
                    "wrong_presented": wrong_presented,
                    "raw_spans": [span.to_dict() for span in proposal.spans],
                    "gold_spans": [span.to_dict() for span in gold.evidence],
                }
            )
        item_rows.append(
            {
                "example_id": example.example_id,
                "variant": example.variant,
                "target_field": example.target_field.value,
                "target_state": (
                    None if example.target_state is None else example.target_state.value
                ),
                "raw_summary": summary,
                "parse_status": "parsed",
                "parse_error": None,
                "fields": field_rows,
            }
        )
    return {
        "items": len(examples),
        "parsed_items": len(examples) - malformed_items,
        "malformed_items": malformed_items,
        "malformed_rate": _rate(malformed_items, len(examples)),
        "fields": _finish_raw_bucket(aggregate),
        "wrong_presented_field_count": aggregate["wrong_presented"],
        "by_field": {
            name: _finish_raw_bucket(bucket) for name, bucket in by_field.items()
        },
        "by_gold_state": {
            name: _finish_raw_bucket(bucket) for name, bucket in by_gold_state.items()
        },
        "target_challenge": {
            name: _finish_raw_bucket(bucket)
            for name, bucket in target_challenge.items()
        },
        "item_diagnostics": item_rows,
    }


def final_state_diagnostics(
    report: EvaluationReport,
    examples: Sequence[StateSpanExample],
) -> dict[str, Any]:
    """Aggregate final verifier-gated correctness by gold and target state."""

    example_by_id = {example.example_id: example for example in examples}
    by_gold_state = {
        state.value: {
            "total": 0,
            "state_correct": 0,
            "grounded_exact": 0,
            "presented": 0,
            "wrong_presented": 0,
        }
        for state in FieldState
    }
    target_challenge = {
        state.value: {
            "total": 0,
            "state_correct": 0,
            "grounded_exact": 0,
            "presented": 0,
            "wrong_presented": 0,
        }
        for state in (
            FieldState.MISSING,
            FieldState.UNCERTAIN,
            FieldState.CONFLICTING,
        )
    }
    for item in report.items:
        example = example_by_id[item["case_id"]]
        if item["status"] != "ok":
            request = NanoInput(
                item_id=example.example_id, transcript=example.transcript
            )
            gold_fields = _extract_fields(request)
            for gold in gold_fields:
                by_gold_state[gold.state.value]["total"] += 1
            if example.target_state is not None:
                target_challenge[example.target_state.value]["total"] += 1
            continue
        fields = item["field_results"]
        assert isinstance(fields, list)
        for row in fields:
            gold_state = row["gold_state"]
            bucket = by_gold_state[gold_state]
            presented = row["predicted_state"] in {
                FieldState.SUPPORTED.value,
                FieldState.ABSENT.value,
            }
            bucket["total"] += 1
            bucket["state_correct"] += int(row["state_correct"])
            bucket["grounded_exact"] += int(row["grounded_exact"])
            bucket["presented"] += int(presented)
            bucket["wrong_presented"] += int(presented and not row["grounded_exact"])
            if (
                example.target_state is not None
                and row["field"] == example.target_field.value
            ):
                target = target_challenge[example.target_state.value]
                target["total"] += 1
                target["state_correct"] += int(row["state_correct"])
                target["grounded_exact"] += int(row["grounded_exact"])
                target["presented"] += int(presented)
                target["wrong_presented"] += int(
                    presented and not row["grounded_exact"]
                )

    def finish(values: Mapping[str, int]) -> dict[str, int | float | None]:
        return {
            **values,
            "state_accuracy": _rate(values["state_correct"], values["total"]),
            "grounded_exact_accuracy": _rate(values["grounded_exact"], values["total"]),
            "wrong_presented_rate": _rate(
                values["wrong_presented"], values["presented"]
            ),
        }

    wrong_presented = report.quality["false_presented_count"]
    return {
        "wrong_presented_field_count": wrong_presented,
        "wrong_presented_safety_gate_pass": wrong_presented == 0,
        "by_gold_state": {
            state: finish(bucket) for state, bucket in by_gold_state.items()
        },
        "target_challenge": {
            state: finish(bucket) for state, bucket in target_challenge.items()
        },
    }


def acceptance_diagnostics(
    report: EvaluationReport,
    examples: Sequence[StateSpanExample],
) -> dict[str, Any]:
    """Pin each preregistered final-output gate to one exact denominator."""

    example_by_id = {example.example_id: example for example in examples}
    held_fields = {
        FieldName.CHIEF_COMPLAINT.value,
        FieldName.DURATION.value,
        FieldName.MEDICATION.value,
        FieldName.ALLERGY.value,
    }
    counts = {
        "overall": [0, len(examples) * len(FIELD_ORDER)],
        "held_value": [0, 0],
        "missing_target": [0, 0],
        "conflict_target": [0, 0],
        "absence": [0, 0],
    }
    for item in report.items:
        example = example_by_id[item["case_id"]]
        request = NanoInput(item_id=example.example_id, transcript=example.transcript)
        gold_by_field = {field.field.value: field for field in _extract_fields(request)}
        for field_name, gold in gold_by_field.items():
            if gold.state is FieldState.SUPPORTED and field_name in held_fields:
                counts["held_value"][1] += 1
            if gold.state is FieldState.ABSENT:
                counts["absence"][1] += 1
        if example.variant == "missing":
            counts["missing_target"][1] += 1
        if example.variant == "conflicting":
            counts["conflict_target"][1] += 1
        if item["status"] != "ok":
            continue
        field_results = item["field_results"]
        assert isinstance(field_results, list)
        for row in field_results:
            grounded = bool(row["grounded_exact"])
            counts["overall"][0] += int(grounded)
            if (
                row["gold_state"] == FieldState.SUPPORTED.value
                and row["field"] in held_fields
            ):
                counts["held_value"][0] += int(grounded)
            if row["gold_state"] == FieldState.ABSENT.value:
                counts["absence"][0] += int(grounded)
            if row["field"] == example.target_field.value:
                if example.variant == "missing":
                    counts["missing_target"][0] += int(grounded)
                elif example.variant == "conflicting":
                    counts["conflict_target"][0] += int(grounded)

    def metric(numerator: int, denominator: int) -> dict[str, int | float | None]:
        return {
            "numerator": numerator,
            "denominator": denominator,
            "rate": _rate(numerator, denominator),
        }

    metrics = {
        name: metric(numerator, denominator)
        for name, (numerator, denominator) in counts.items()
    }
    metrics["failures"] = metric(report.failures["count"], len(examples))
    metrics["false_presented"] = metric(
        report.quality["false_presented_count"],
        report.quality["presented_field_count"],
    )
    if metrics["overall"]["numerator"] != report.quality["grounded_exact_field_count"]:
        raise RuntimeError("acceptance overall metric disagrees with evaluator")
    return {
        "metrics": metrics,
        "definitions": {
            "overall": "grounded exact over every development field",
            "held_value": (
                "grounded exact on gold-supported chief complaint, duration, "
                "medication, and allergy fields; closed severity is excluded"
            ),
            "missing_target": (
                "grounded exact on the designated target field of missing variants"
            ),
            "conflict_target": (
                "grounded exact on the designated target field of conflicting variants"
            ),
            "absence": "grounded exact on every gold-absent development field",
            "failures": "item-level inference failures over all development items",
            "false_presented": (
                "standard evaluator false-presented fields over final presented fields"
            ),
        },
        "held_value_split_note": (
            "The development open-value lexicons are disjoint from training."
        ),
    }


def _source_hashes() -> dict[str, str]:
    files = {
        "development_evaluator": Path(__file__),
        "evaluation": Path(__file__).parents[1] / "evaluation.py",
        "model": Path(__file__).with_name("model.py"),
        "state_span_data": Path(state_span_data.__file__),
        "state_span_adapter": Path(__file__).parents[1] / "adapters" / "state_span.py",
        "state_checkpoint_adapter": Path(__file__).parents[1]
        / "adapters"
        / "state_checkpoint.py",
        "legacy_summary_adapter": Path(__file__).parents[1]
        / "adapters"
        / "legacy_summary.py",
    }
    return {
        name: _sha256(_read_regular_file(path, role=f"{name} source"))
        for name, path in sorted(files.items())
    }


def _evaluate_generated(
    *,
    examples: Sequence[StateSpanExample],
    cases: Sequence[FixtureCase],
    summaries: Sequence[str],
    solver: LegacySummarySolver | StateSpanSolver,
) -> EvaluationReport:
    predictions = _prediction_map(examples, summaries)
    solver._predict = predictions.__getitem__  # type: ignore[attr-defined]
    return evaluate_solver(solver, cases, measure_latency=False)


def _comparison_to_base(
    candidate_report: EvaluationReport,
    candidate_final: Mapping[str, Any],
    candidate_acceptance: Mapping[str, Any],
    base_report: EvaluationReport,
    base_final: Mapping[str, Any],
    base_acceptance: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_quality = candidate_report.quality
    base_quality = base_report.quality

    def delta(key: str) -> float:
        return float(candidate_quality[key]) - float(base_quality[key])

    challenge_delta: dict[str, float] = {}
    for state in ("missing", "uncertain", "conflicting"):
        candidate_rate = candidate_final["target_challenge"][state][
            "grounded_exact_accuracy"
        ]
        base_rate = base_final["target_challenge"][state]["grounded_exact_accuracy"]
        assert candidate_rate is not None and base_rate is not None
        challenge_delta[state] = candidate_rate - base_rate
    acceptance_comparison: dict[str, Any] = {}
    for name in (
        "overall",
        "held_value",
        "missing_target",
        "conflict_target",
        "absence",
        "failures",
        "false_presented",
    ):
        candidate_metric = candidate_acceptance["metrics"][name]
        base_metric = base_acceptance["metrics"][name]
        candidate_rate = candidate_metric["rate"]
        base_rate = base_metric["rate"]
        if candidate_rate is None or base_rate is None:
            rate_delta = None
        else:
            rate_delta = candidate_rate - base_rate
        acceptance_comparison[name] = {
            "candidate": candidate_metric,
            "frozen_base": base_metric,
            "candidate_minus_base_rate": rate_delta,
        }
    gates = {
        "overall_gain_at_least_5pp": (
            acceptance_comparison["overall"]["candidate_minus_base_rate"] >= 0.05
        ),
        "held_value_gain_at_least_10pp": (
            acceptance_comparison["held_value"]["candidate_minus_base_rate"] >= 0.10
        ),
        "missing_target_gain_at_least_50pp": (
            acceptance_comparison["missing_target"]["candidate_minus_base_rate"] >= 0.50
        ),
        "failure_rate_at_most_1pct": (
            candidate_acceptance["metrics"]["failures"]["rate"] <= 0.01
        ),
        "zero_false_presented": (
            candidate_acceptance["metrics"]["false_presented"]["numerator"] == 0
        ),
        "absence_regression_no_more_than_1pp": (
            acceptance_comparison["absence"]["candidate_minus_base_rate"] >= -0.01
        ),
        "conflict_target_regression_no_more_than_1pp": (
            acceptance_comparison["conflict_target"]["candidate_minus_base_rate"]
            >= -0.01
        ),
    }
    return {
        "grounded_exact_field_accuracy_delta": delta("grounded_exact_field_accuracy"),
        "state_accuracy_delta": delta("state_accuracy"),
        "inference_failure_rate_delta": (
            float(candidate_report.failures["rate"])
            - float(base_report.failures["rate"])
        ),
        "target_challenge_grounded_exact_accuracy_delta": challenge_delta,
        "wrong_presented_field_count_delta": (
            candidate_report.quality["false_presented_count"]
            - base_report.quality["false_presented_count"]
        ),
        "acceptance_metrics": acceptance_comparison,
        "acceptance_gates": {
            **gates,
            "all_measured_quality_gates_pass": all(gates.values()),
            "latency_gate_assessed": False,
            "fresh_v1_confirmation_assessed": False,
        },
    }


def _write_json_no_clobber(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise DevelopmentEvaluationError("evaluation output already exists") from exc


def evaluate_development(
    *,
    data_dir: str | Path,
    manifest_sha256: str,
    base_checkpoint: str | Path,
    tokenizer_path: str | Path,
    candidates: Sequence[CandidateCheckpoint],
    training_reports: Sequence[tuple[str | Path, str]] = (),
    output_path: str | Path,
    device: str = "cpu",
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> Mapping[str, Any]:
    """Compare frozen Nano and candidate epochs on sealed development only."""

    output = Path(output_path)
    if output.exists():
        raise DevelopmentEvaluationError("evaluation output already exists")
    if not output.parent.is_dir():
        raise DevelopmentEvaluationError("evaluation output parent must exist")
    resolved_device = _resolve_device(device)
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise DevelopmentEvaluationError("batch_size must be a positive integer")
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens < 1
    ):
        raise DevelopmentEvaluationError("max_new_tokens must be a positive integer")

    source_hashes = _source_hashes()
    bundle = load_development_bundle(data_dir, expected_manifest_sha256=manifest_sha256)
    tokenizer_snapshot = _read_verified_file(
        Path(tokenizer_path),
        FROZEN_NANO_V01.tokenizer_sha256,
        role="frozen Nano tokenizer",
    )
    try:
        tokenizer = Tokenizer.from_str(tokenizer_snapshot.decode("utf-8"))
    except Exception as exc:
        raise DevelopmentEvaluationError(
            "frozen Nano tokenizer could not be deserialized"
        ) from exc
    if (
        tokenizer.get_vocab_size(with_added_tokens=True)
        != NANO_MODEL_CONFIG.vocabulary_size
    ):
        raise DevelopmentEvaluationError("frozen tokenizer vocabulary size changed")

    resolved_candidates = list(candidates)
    for report_path, report_sha256 in training_reports:
        resolved_candidates.extend(
            load_candidates_from_training_report(
                report_path,
                expected_report_sha256=report_sha256,
                expected_manifest_sha256=bundle.manifest_sha256,
                expected_dev_sha256=bundle.dev_sha256,
                expected_train_sha256=bundle.manifest["train"]["sha256"],
            )
        )
    if not resolved_candidates:
        raise DevelopmentEvaluationError("at least one candidate is required")
    labels = [candidate.label for candidate in resolved_candidates]
    digests = [candidate.sha256 for candidate in resolved_candidates]
    if len(set(labels)) != len(labels):
        raise DevelopmentEvaluationError("candidate labels must be unique")
    if len(set(digests)) != len(digests):
        raise DevelopmentEvaluationError("candidate checkpoint digests must be unique")

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    examples = bundle.examples
    cases = _fixture_cases(examples)
    base_prompts = tuple(
        f"{example.transcript.rstrip()}\nSummarize the visit." for example in examples
    )
    candidate_prompts = tuple(
        build_state_span_prompt(example.transcript) for example in examples
    )
    base_tokens = encode_chatml_prompts(tokenizer, base_prompts)
    candidate_tokens = encode_chatml_prompts(tokenizer, candidate_prompts)

    base_model = _load_model(
        Path(base_checkpoint),
        expected_sha256=FROZEN_NANO_V01.checkpoint_sha256,
        device=resolved_device,
        frozen_base=True,
    )
    base_summaries = batched_greedy_generate(
        base_model,
        tokenizer,
        base_tokens,
        device=resolved_device,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    del base_model
    base_solver = LegacySummarySolver(
        lambda _transcript: "",
        solver_id=(
            "development/frozen-nano-v01/legacy-summary/"
            f"sha-{FROZEN_NANO_V01.checkpoint_sha256}"
        ),
        version="sealed-dev-v0",
        parameter_count=NANO_MODEL_CONFIG.parameter_count,
    )
    base_evaluation = _evaluate_generated(
        examples=examples,
        cases=cases,
        summaries=base_summaries,
        solver=base_solver,
    )
    base_final = final_state_diagnostics(base_evaluation, examples)
    base_acceptance = acceptance_diagnostics(base_evaluation, examples)

    candidate_rows: list[dict[str, Any]] = []
    for candidate in resolved_candidates:
        model = _load_model(
            candidate.path,
            expected_sha256=candidate.sha256,
            device=resolved_device,
            frozen_base=False,
        )
        summaries = batched_greedy_generate(
            model,
            tokenizer,
            candidate_tokens,
            device=resolved_device,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )
        del model
        solver = StateSpanSolver(
            lambda _transcript: "",
            solver_id=(
                f"development/native-state-span/{candidate.label}/"
                f"sha-{candidate.sha256}"
            ),
            version="sealed-dev-v0",
            parameter_count=NANO_MODEL_CONFIG.parameter_count,
        )
        evaluation = _evaluate_generated(
            examples=examples,
            cases=cases,
            summaries=summaries,
            solver=solver,
        )
        raw = raw_state_span_diagnostics(examples, cases, summaries)
        final = final_state_diagnostics(evaluation, examples)
        acceptance = acceptance_diagnostics(evaluation, examples)
        candidate_rows.append(
            {
                "label": candidate.label,
                "checkpoint_sha256": candidate.sha256,
                "provenance": dict(candidate.provenance),
                "evaluation": evaluation.to_dict(),
                "raw_state_span": raw,
                "final_state": final,
                "acceptance": acceptance,
                "comparison_to_frozen_base": _comparison_to_base(
                    evaluation,
                    final,
                    acceptance,
                    base_evaluation,
                    base_final,
                    base_acceptance,
                ),
            }
        )

    if _source_hashes() != source_hashes:
        raise DevelopmentEvaluationError("evaluation source changed during the run")
    result = {
        "schema_version": DEVELOPMENT_EVALUATION_SCHEMA_VERSION,
        "status": "complete",
        "partition": {
            "partition_id": DEVELOPMENT_PARTITION_ID,
            "role": "sealed_development_model_selection_only",
            "manifest_sha256": bundle.manifest_sha256,
            "development_sha256": bundle.dev_sha256,
            "records": len(examples),
            "worlds": len({example.world_id for example in examples}),
            "target_grammar": TARGET_GRAMMAR_VERSION,
            "historical_benchmark_read": False,
            "final_confirmation_required": "newly_sealed_fresh_v1",
        },
        "artifacts": {
            "frozen_base_checkpoint_sha256": FROZEN_NANO_V01.checkpoint_sha256,
            "tokenizer_sha256": FROZEN_NANO_V01.tokenizer_sha256,
            "architecture_identity": FROZEN_NANO_V01.architecture_identity,
            "parameter_count": NANO_MODEL_CONFIG.parameter_count,
        },
        "protocol": {
            "base_prompt_template_id": PROMPT_TEMPLATE_ID,
            "candidate_prompt_template_id": STATE_PROMPT_TEMPLATE_ID,
            "candidate_instruction_sha256": _sha256(
                STATE_SPAN_PROMPT_INSTRUCTION.encode("utf-8")
            ),
            "decode_policy_id": DECODE_POLICY_ID,
            "max_new_tokens": max_new_tokens,
            "batch_size": batch_size,
            "equal_prompt_length_batching": True,
            "latency_measured": False,
        },
        "runtime": {
            "device": resolved_device,
            "deterministic_algorithms": True,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "tokenizers": getattr(__import__("tokenizers"), "__version__", None),
        },
        "source_sha256": source_hashes,
        "frozen_base": {
            "checkpoint_sha256": FROZEN_NANO_V01.checkpoint_sha256,
            "evaluation": base_evaluation.to_dict(),
            "final_state": base_final,
            "acceptance": base_acceptance,
        },
        "candidates": candidate_rows,
        "selection_boundary": (
            "This report is development evidence only. It does not select, promote, "
            "or overwrite a checkpoint; final confirmation requires a newly sealed "
            "fresh-v1 partition."
        ),
    }
    _write_json_no_clobber(output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Compare Nano state/span candidates on sealed development"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=root / "checkpoints" / "anchors" / "nano_v01_scribe.pt",
    )
    parser.add_argument(
        "--tokenizer", type=Path, default=root / "sft" / "tokenizer.json"
    )
    parser.add_argument(
        "--candidate",
        nargs=3,
        action="append",
        metavar=("LABEL", "CHECKPOINT", "SHA256"),
        default=[],
        help="repeatable explicit label/path/digest candidate",
    )
    parser.add_argument(
        "--training-report",
        nargs=2,
        action="append",
        metavar=("REPORT", "SHA256"),
        default=[],
        help="repeatable digest-pinned report; evaluates all three epochs",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    direct_candidates = tuple(
        CandidateCheckpoint(label=label, path=Path(path), sha256=digest)
        for label, path, digest in args.candidate
    )
    evaluate_development(
        data_dir=args.data_dir,
        manifest_sha256=args.manifest_sha256,
        base_checkpoint=args.base_checkpoint,
        tokenizer_path=args.tokenizer,
        candidates=direct_candidates,
        training_reports=tuple(
            (Path(path), digest) for path, digest in args.training_report
        ),
        output_path=args.output,
        device=args.device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEVELOPMENT_EVALUATION_SCHEMA_VERSION",
    "CandidateCheckpoint",
    "DevelopmentBundle",
    "DevelopmentEvaluationError",
    "acceptance_diagnostics",
    "batched_greedy_generate",
    "encode_chatml_prompts",
    "evaluate_development",
    "final_state_diagnostics",
    "load_candidates_from_training_report",
    "load_development_bundle",
    "raw_state_span_diagnostics",
]
