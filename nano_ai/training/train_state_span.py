"""Frozen H1 continuation recipe for Nano's native state/span grammar.

This module trains only the existing 3,148,608-parameter Nano architecture.  It
hash-verifies the frozen base, tokenizer, generator, manifest, and dataset before
deserialization or optimization.  Development data may diagnose or compare the
two fixed seeds, but the historical fresh-v0 benchmark is never read here.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import platform
import random
import stat
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer
from torch import Tensor
from torch.nn import functional as torch_functional

from nano_ai.adapters.anchor_checkpoint import FROZEN_NANO_V01
from nano_ai.adapters.state_checkpoint import (
    STATE_PROMPT_TEMPLATE_ID,
    STATE_SPAN_PROMPT_INSTRUCTION,
    build_state_span_prompt,
)
from nano_ai.contract import FIELD_ORDER, FieldState
from nano_ai.training import state_span_data
from nano_ai.training.model import (
    NANO_MODEL_CONFIG,
    NanoGPT,
    load_frozen_nano_state_dict,
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
    build_manifest,
    canonical_json_bytes,
    generate_split,
    load_records,
)

TRAINING_RECIPE_VERSION = "nano-native-state-span-sft-v0"
TRAINING_SEEDS = (20260805, 20260806)
EPOCHS = 3
BATCH_SIZE = 32
PEAK_LEARNING_RATE = 1.5e-4
WARMUP_FRACTION = 0.03
COSINE_FLOOR = 0.1
WEIGHT_DECAY = 0.1
GRADIENT_CLIP = 1.0
ADAM_BETAS = (0.9, 0.95)
ADAM_EPSILON = 1e-8
LOG_EVERY_STEPS = 25


class TrainingInputError(ValueError):
    """A frozen training input or recipe invariant is invalid."""


@dataclass(frozen=True, slots=True)
class TrainingBundle:
    """Hash-verified state/span data and its exact manifest snapshot."""

    manifest: Mapping[str, Any]
    manifest_sha256: str
    train: tuple[StateSpanExample, ...]
    dev: tuple[StateSpanExample, ...]


@dataclass(frozen=True, slots=True)
class TokenizedSplit:
    """Fixed-length CPU tensors plus original unpadded sequence lengths."""

    token_ids: Tensor
    loss_mask: Tensor
    lengths: tuple[int, ...]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_regular_file(path: Path, *, role: str) -> bytes:
    try:
        with Path(path).open("rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise TrainingInputError(f"{role} is not a regular file")
            return handle.read()
    except TrainingInputError:
        raise
    except OSError as exc:
        raise TrainingInputError(f"{role} is unavailable") from exc


def _verify_file(path: Path, expected_sha256: str, *, role: str) -> bytes:
    payload = _read_regular_file(path, role=role)
    observed = _sha256(payload)
    if not hmac.compare_digest(observed, expected_sha256):
        raise TrainingInputError(
            f"{role} SHA-256 mismatch: expected {expected_sha256}, observed {observed}"
        )
    return payload


def _reject_json_constant(value: str) -> None:
    raise TrainingInputError(f"non-finite JSON constant is forbidden: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TrainingInputError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_manifest(snapshot: bytes) -> Mapping[str, Any]:
    try:
        value = json.loads(
            snapshot.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise TrainingInputError("training manifest is invalid JSON") from exc
    if not isinstance(value, dict):
        raise TrainingInputError("training manifest must be an object")
    return value


def _expected_quota(per_state_field: int) -> dict[str, int]:
    return {
        f"{state.value}:{field.value}": per_state_field
        for state in (
            FieldState.MISSING,
            FieldState.UNCERTAIN,
            FieldState.CONFLICTING,
        )
        for field in FIELD_ORDER
    }


def _validate_manifest_section(
    section: object,
    *,
    split: str,
    expected_seed: int,
    expected_records: int,
    expected_worlds: int,
    expected_quota: int,
) -> Mapping[str, Any]:
    if not isinstance(section, dict):
        raise TrainingInputError(f"manifest {split} section must be an object")
    expected_keys = {
        "seed",
        "records",
        "worlds",
        "sha256",
        "transcript_multiset_sha256",
        "state_field_quota",
    }
    if set(section) != expected_keys:
        raise TrainingInputError(f"manifest {split} section has unexpected keys")
    if section["seed"] != expected_seed:
        raise TrainingInputError(f"manifest {split} seed is not frozen")
    if section["records"] != expected_records:
        raise TrainingInputError(f"manifest {split} record count is not frozen")
    if section["worlds"] != expected_worlds:
        raise TrainingInputError(f"manifest {split} world count is not frozen")
    if not _valid_sha256(section["sha256"]) or not _valid_sha256(
        section["transcript_multiset_sha256"]
    ):
        raise TrainingInputError(f"manifest {split} digest is invalid")
    if section["state_field_quota"] != _expected_quota(expected_quota):
        raise TrainingInputError(f"manifest {split} state/field quota is not frozen")
    return section


def _validate_records(
    records: tuple[StateSpanExample, ...],
    *,
    split: str,
    expected_worlds: int,
) -> None:
    if any(record.split != split for record in records):
        raise TrainingInputError(f"{split} data contains a cross-split record")
    example_ids = [record.example_id for record in records]
    if len(set(example_ids)) != len(example_ids):
        raise TrainingInputError(f"{split} example IDs are not unique")
    groups: dict[str, list[StateSpanExample]] = defaultdict(list)
    for record in records:
        groups[record.world_id].append(record)
    if len(groups) != expected_worlds:
        raise TrainingInputError(f"{split} world cardinality is invalid")
    expected_variants = {"normal", "missing", "uncertain", "conflicting"}
    for world_id, examples in groups.items():
        if (
            len(examples) != 4
            or {item.variant for item in examples} != expected_variants
        ):
            raise TrainingInputError(f"{split} world {world_id} is not a complete pair")
        if len({item.target_field for item in examples}) != 1:
            raise TrainingInputError(f"{split} world {world_id} changes target field")


def load_training_bundle(data_dir: Path) -> TrainingBundle:
    """Verify the exact v0 data family before returning parsed records."""

    root = Path(data_dir)
    manifest_bytes = _read_regular_file(
        root / "manifest.json", role="training manifest"
    )
    manifest = _parse_manifest(manifest_bytes)
    expected_top_keys = {
        "schema_version",
        "target_grammar",
        "generator_sha256",
        "tokenizer_sha256",
        "base_checkpoint_sha256",
        "train",
        "dev",
        "isolation",
    }
    if set(manifest) != expected_top_keys:
        raise TrainingInputError("training manifest has unexpected keys")
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise TrainingInputError("training manifest schema is not supported")
    if manifest["target_grammar"] != TARGET_GRAMMAR_VERSION:
        raise TrainingInputError("training target grammar is not frozen")
    if manifest["tokenizer_sha256"] != FROZEN_NANO_V01.tokenizer_sha256:
        raise TrainingInputError("training tokenizer identity is not Nano v0.1")
    if manifest["base_checkpoint_sha256"] != FROZEN_NANO_V01.checkpoint_sha256:
        raise TrainingInputError("training base identity is not Nano v0.1")
    generator_digest = _sha256(Path(state_span_data.__file__).read_bytes())
    if manifest["generator_sha256"] != generator_digest:
        raise TrainingInputError("training generator source changed after data freeze")

    train_section = _validate_manifest_section(
        manifest["train"],
        split="train",
        expected_seed=TRAIN_SEED,
        expected_records=TRAIN_WORLDS * 4,
        expected_worlds=TRAIN_WORLDS,
        expected_quota=TRAIN_WORLDS // len(FIELD_ORDER),
    )
    dev_section = _validate_manifest_section(
        manifest["dev"],
        split="dev",
        expected_seed=DEV_SEED,
        expected_records=DEV_WORLDS * 4,
        expected_worlds=DEV_WORLDS,
        expected_quota=DEV_WORLDS // len(FIELD_ORDER),
    )
    train = load_records(root / "train.jsonl", expected_sha256=train_section["sha256"])
    dev = load_records(root / "dev.jsonl", expected_sha256=dev_section["sha256"])
    _validate_records(train, split="train", expected_worlds=TRAIN_WORLDS)
    _validate_records(dev, split="dev", expected_worlds=DEV_WORLDS)
    if {record.world_id for record in train} & {record.world_id for record in dev}:
        raise TrainingInputError("training and development worlds overlap")
    if train != generate_split("train") or dev != generate_split("dev"):
        raise TrainingInputError(
            "training records do not equal the deterministic frozen generation"
        )
    expected_manifest = build_manifest(
        train,
        dev,
        generator_sha256=generator_digest,
        tokenizer_sha256=FROZEN_NANO_V01.tokenizer_sha256,
        base_checkpoint_sha256=FROZEN_NANO_V01.checkpoint_sha256,
    )
    if manifest != expected_manifest:
        raise TrainingInputError("training manifest does not match the frozen data")
    return TrainingBundle(
        manifest=manifest,
        manifest_sha256=_sha256(manifest_bytes),
        train=train,
        dev=dev,
    )


def load_frozen_tokenizer(tokenizer_path: Path) -> Tokenizer:
    """Hash-check and parse the exact tokenizer snapshot used by Nano v0.1."""

    snapshot = _verify_file(
        Path(tokenizer_path),
        FROZEN_NANO_V01.tokenizer_sha256,
        role="frozen Nano tokenizer",
    )
    try:
        tokenizer = Tokenizer.from_str(snapshot.decode("utf-8"))
    except Exception as exc:
        raise TrainingInputError("frozen Nano tokenizer could not be parsed") from exc
    if (
        tokenizer.get_vocab_size(with_added_tokens=True)
        != NANO_MODEL_CONFIG.vocabulary_size
    ):
        raise TrainingInputError("frozen Nano tokenizer vocabulary size changed")
    for token in ("<|im_start|>", "<|im_end|>"):
        if tokenizer.token_to_id(token) is None:
            raise TrainingInputError(f"frozen Nano tokenizer is missing {token}")
    return tokenizer


def encode_training_example(
    tokenizer: Tokenizer,
    example: StateSpanExample,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Render one ChatML exchange and mask loss to the assistant target only."""

    start_token = tokenizer.token_to_id("<|im_start|>")
    end_token = tokenizer.token_to_id("<|im_end|>")
    if start_token is None or end_token is None:
        raise TrainingInputError("tokenizer is missing required ChatML tokens")
    user_header = tokenizer.encode("user\n", add_special_tokens=False).ids
    assistant_header = tokenizer.encode("assistant\n", add_special_tokens=False).ids
    prompt = build_state_span_prompt(example.transcript)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    target_ids = tokenizer.encode(example.target, add_special_tokens=False).ids
    prefix = (
        [start_token]
        + user_header
        + prompt_ids
        + [end_token, start_token]
        + assistant_header
    )
    token_ids = tuple(prefix + target_ids + [end_token])
    loss_mask = tuple([0] * len(prefix) + [1] * (len(target_ids) + 1))
    if len(token_ids) > NANO_MODEL_CONFIG.sequence_length:
        raise TrainingInputError(
            f"encoded example {example.example_id} exceeds Nano's context window"
        )
    if sum(loss_mask) < 2:
        raise TrainingInputError(f"encoded example {example.example_id} has no target")
    return token_ids, loss_mask


def tokenize_split(
    tokenizer: Tokenizer,
    examples: Sequence[StateSpanExample],
) -> TokenizedSplit:
    """Tokenize without dropping examples or changing the frozen quotas."""

    sequence_length = NANO_MODEL_CONFIG.sequence_length
    all_ids: list[list[int]] = []
    all_masks: list[list[int]] = []
    lengths: list[int] = []
    for example in examples:
        token_ids, loss_mask = encode_training_example(tokenizer, example)
        padding = sequence_length - len(token_ids)
        all_ids.append([*token_ids, *([0] * padding)])
        all_masks.append([*loss_mask, *([0] * padding)])
        lengths.append(len(token_ids))
    return TokenizedSplit(
        token_ids=torch.tensor(all_ids, dtype=torch.long),
        loss_mask=torch.tensor(all_masks, dtype=torch.bool),
        lengths=tuple(lengths),
    )


def grouped_batch_indices(
    examples: Sequence[StateSpanExample],
    *,
    batch_size: int,
    seed: int,
    epoch: int,
) -> tuple[tuple[int, ...], ...]:
    """Shuffle worlds while keeping each four-variant family in one batch."""

    if batch_size <= 0 or batch_size % 4:
        raise ValueError("batch_size must be a positive multiple of four")
    if epoch < 1:
        raise ValueError("epoch must be positive")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        grouped[example.world_id].append(index)
    if any(len(indices) != 4 for indices in grouped.values()):
        raise TrainingInputError("every training world must contain four variants")
    worlds = sorted(grouped)
    random.Random(seed + epoch * 1_000_003).shuffle(worlds)
    worlds_per_batch = batch_size // 4
    batches: list[tuple[int, ...]] = []
    for start in range(0, len(worlds), worlds_per_batch):
        selected = worlds[start : start + worlds_per_batch]
        batches.append(tuple(index for world in selected for index in grouped[world]))
    return tuple(batches)


def learning_rate_at(step: int, *, total_steps: int) -> float:
    """Historical linear-warmup/cosine schedule with a ten-percent floor."""

    if step < 1 or total_steps < 1 or step > total_steps:
        raise ValueError("step must be within the training schedule")
    warmup_steps = int(WARMUP_FRACTION * total_steps)
    if step < warmup_steps:
        return PEAK_LEARNING_RATE * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return PEAK_LEARNING_RATE * (
        COSINE_FLOOR + (1 - COSINE_FLOOR) * 0.5 * (1 + math.cos(math.pi * progress))
    )


def masked_cross_entropy(
    logits: Tensor,
    token_ids: Tensor,
    loss_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return normalized assistant-only loss and its supervised-token count."""

    targets = token_ids[:, 1:]
    target_mask = loss_mask[:, 1:]
    losses = torch_functional.cross_entropy(
        logits[:, :-1].reshape(-1, NANO_MODEL_CONFIG.vocabulary_size),
        targets.reshape(-1),
        reduction="none",
    ).reshape(targets.shape)
    supervised = target_mask.sum()
    if supervised.item() == 0:
        raise TrainingInputError("batch has no supervised assistant tokens")
    return (losses * target_mask).sum() / supervised, supervised


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise TrainingInputError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise TrainingInputError("MPS was requested but is unavailable")
    return requested


def _seed_training(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _optimizer(model: NanoGPT) -> torch.optim.AdamW:
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


def _cropped_batch(
    split: TokenizedSplit,
    indices: Sequence[int],
    *,
    device: str,
) -> tuple[Tensor, Tensor]:
    maximum = max(split.lengths[index] for index in indices)
    selected = torch.tensor(indices, dtype=torch.long)
    return (
        split.token_ids.index_select(0, selected)[:, :maximum].to(device),
        split.loss_mask.index_select(0, selected)[:, :maximum].to(device),
    )


def _development_loss(
    model: NanoGPT,
    split: TokenizedSplit,
    *,
    device: str,
) -> float:
    model.eval()
    loss_sum = 0.0
    token_count = 0
    with torch.no_grad():
        for start in range(0, len(split.lengths), BATCH_SIZE):
            indices = tuple(range(start, min(start + BATCH_SIZE, len(split.lengths))))
            token_ids, loss_mask = _cropped_batch(split, indices, device=device)
            loss, supervised = masked_cross_entropy(
                model(token_ids), token_ids, loss_mask
            )
            count = int(supervised.item())
            loss_sum += float(loss.item()) * count
            token_count += count
    model.train()
    return loss_sum / token_count


def _save_checkpoint(path: Path, model: NanoGPT) -> dict[str, Any]:
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
    payload = canonical_json_bytes(value)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _source_hashes() -> dict[str, str]:
    files = {
        "data_generator": Path(state_span_data.__file__),
        "model": Path(__file__).with_name("model.py"),
        "state_adapter": Path(__file__).parents[1] / "adapters" / "state_span.py",
        "state_checkpoint": Path(__file__).parents[1]
        / "adapters"
        / "state_checkpoint.py",
        "training": Path(__file__),
    }
    return {name: _sha256(path.read_bytes()) for name, path in sorted(files.items())}


def _require_unchanged_sources(expected: Mapping[str, str]) -> None:
    """Fail closed if executable recipe sources changed during a run."""

    if _source_hashes() != dict(expected):
        raise TrainingInputError("training source changed during execution")


def train_candidate(
    *,
    data_dir: Path,
    base_checkpoint: Path,
    tokenizer_path: Path,
    output_dir: Path,
    seed: int,
    device: str,
) -> Mapping[str, Any]:
    """Run one fixed H1 seed and create a new, never-overwritten candidate."""

    if seed not in TRAINING_SEEDS:
        raise TrainingInputError(
            f"seed must be one of the frozen seeds {TRAINING_SEEDS}"
        )
    output = Path(output_dir)
    if output.exists():
        raise TrainingInputError("candidate output directory must not exist")

    started = time.monotonic()
    source_hashes = _source_hashes()
    bundle = load_training_bundle(Path(data_dir))
    tokenizer = load_frozen_tokenizer(Path(tokenizer_path))
    state_dict = load_frozen_nano_state_dict(Path(base_checkpoint))
    train_tokens = tokenize_split(tokenizer, bundle.train)
    dev_tokens = tokenize_split(tokenizer, bundle.dev)
    resolved_device = _resolve_device(device)
    _seed_training(seed)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise TrainingInputError("candidate output directory must not exist") from exc

    model = NanoGPT()
    model.load_state_dict(state_dict, strict=True)
    model.to(resolved_device).train()
    optimizer = _optimizer(model)
    epoch_batches = grouped_batch_indices(
        bundle.train,
        batch_size=BATCH_SIZE,
        seed=seed,
        epoch=1,
    )
    if any(len(batch) != BATCH_SIZE for batch in epoch_batches):
        raise TrainingInputError("frozen training worlds do not fill exact batches")
    steps_per_epoch = len(epoch_batches)
    total_steps = steps_per_epoch * EPOCHS
    global_step = 0
    epoch_reports: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []

    print(
        json.dumps(
            {
                "event": "training_start",
                "recipe": TRAINING_RECIPE_VERSION,
                "seed": seed,
                "device": resolved_device,
                "train_records": len(bundle.train),
                "dev_records": len(bundle.dev),
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
        epoch_loss_sum = 0.0
        epoch_tokens = 0
        epoch_started = time.monotonic()
        for indices in batches:
            global_step += 1
            learning_rate = learning_rate_at(global_step, total_steps=total_steps)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            token_ids, loss_mask = _cropped_batch(
                train_tokens, indices, device=resolved_device
            )
            loss, supervised = masked_cross_entropy(
                model(token_ids), token_ids, loss_mask
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRADIENT_CLIP
            )
            optimizer.step()
            count = int(supervised.item())
            epoch_loss_sum += float(loss.item()) * count
            epoch_tokens += count
            if global_step == 1 or global_step % LOG_EVERY_STEPS == 0:
                print(
                    json.dumps(
                        {
                            "event": "training_step",
                            "epoch": epoch,
                            "step": global_step,
                            "steps": total_steps,
                            "loss": round(float(loss.item()), 6),
                            "gradient_norm": round(float(gradient_norm.item()), 6),
                            "learning_rate": learning_rate,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        dev_loss = _development_loss(model, dev_tokens, device=resolved_device)
        _require_unchanged_sources(source_hashes)
        checkpoint_name = f"epoch-{epoch}.pt" if epoch < EPOCHS else "candidate.pt"
        checkpoint = _save_checkpoint(output / checkpoint_name, model)
        checkpoints.append(checkpoint)
        epoch_report = {
            "epoch": epoch,
            "train_loss": epoch_loss_sum / epoch_tokens,
            "dev_loss": dev_loss,
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
        "schema_version": "nano.state-span-training-report.v0",
        "recipe": TRAINING_RECIPE_VERSION,
        "status": "complete",
        "seed": seed,
        "device": resolved_device,
        "parameter_count": NANO_MODEL_CONFIG.parameter_count,
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
        "prompt": {
            "template_id": STATE_PROMPT_TEMPLATE_ID,
            "instruction_sha256": _sha256(
                STATE_SPAN_PROMPT_INSTRUCTION.encode("utf-8")
            ),
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
            "assistant_only_loss": True,
            "world_grouped_batches": True,
        },
        "epochs": epoch_reports,
        "candidate": candidate,
        "source_sha256": source_hashes,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "seconds": time.monotonic() - started,
        },
        "selection_note": (
            "This seed is an unselected development candidate. Historical fresh-v0 "
            "was not read; final confirmation requires a newly sealed fresh-v1."
        ),
    }
    _require_unchanged_sources(source_hashes)
    _write_json_no_clobber(output / "training_report.json", report)
    print(
        json.dumps(
            {
                "event": "training_complete",
                "candidate_sha256": candidate["sha256"],
                "report": str(output / "training_report.json"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return report


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Train one frozen Nano native state/span candidate seed"
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
    train_candidate(
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
    "ADAM_BETAS",
    "ADAM_EPSILON",
    "BATCH_SIZE",
    "COSINE_FLOOR",
    "EPOCHS",
    "GRADIENT_CLIP",
    "PEAK_LEARNING_RATE",
    "TRAINING_RECIPE_VERSION",
    "TRAINING_SEEDS",
    "TokenizedSplit",
    "TrainingBundle",
    "TrainingInputError",
    "encode_training_example",
    "grouped_batch_indices",
    "learning_rate_at",
    "load_frozen_tokenizer",
    "load_training_bundle",
    "masked_cross_entropy",
    "tokenize_split",
    "train_candidate",
]
