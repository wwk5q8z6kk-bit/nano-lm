"""Candidate-blind construction for Nano's fresh-v1 confirmation partition.

The generator owns evaluator data only.  It reads source text solely to seal
the generator and to audit literal value/template isolation from the native
state/span data source.  It never imports native records, candidates, model
outputs, comparison results, or inference code.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
import stat
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

from nano_ai.contract import (
    CONTRACT_VERSION,
    FIELD_ORDER,
    EvidenceSpan,
    FieldName,
    FieldOutput,
    FieldState,
    NanoInput,
    NanoOutput,
)
from nano_ai.fixtures import FixtureCase

BENCHMARK_ID = "nano-fresh-v1-confirmation-20260803"
MANIFEST_SCHEMA_VERSION = "nano.fresh-confirmation-manifest.v1"
PARTITION_SCHEMA_VERSION = "nano.fresh-confirmation-partition.v1"
BENCHMARK_PROVENANCE_SCHEMA_VERSION = "nano.benchmark-case.v0"
GOLD_SOLVER_ID = "fresh-v1-evaluator-truth"
FRESH_SEED = 20260803

GENERATOR_PATH = "nano_ai/benchmarks/fresh_v1.py"
NATIVE_SOURCE_PATH = "nano_ai/training/state_span_data.py"
NATIVE_SOURCE_SHA256 = (
    "2d3fd33d694893aea5aa80b514e34256e9f6046f71b13f6410dbabec1278d707"
)

TOTAL_RECORDS = 220
TOTAL_FIELDS = TOTAL_RECORDS * len(FIELD_ORDER)
NORMAL_RECORDS = 160
NORMAL_RECORDS_PER_BAND = 80
CHALLENGE_RECORDS = 60
PAIRED_NORMAL_RECORDS = 60
STATE_CHALLENGE_RECORDS = 20
CELL_QUOTA = 2

VALUE_BANDS = ("pool_a", "pool_b")
TARGET_STATES = (
    FieldState.MISSING,
    FieldState.UNCERTAIN,
    FieldState.CONFLICTING,
)
PURPOSE = "final_confirmation_only"

_VALUE_POOLS: Mapping[str, Mapping[FieldName, tuple[str, ...]]] = {
    "pool_a": {
        FieldName.CHIEF_COMPLAINT: (
            "jaw hinge clicking",
            "upper arm prickling",
            "heel pulsation",
            "wrist buzzing",
            "rib edge stiffness",
            "lower back zapping",
            "outer ear flutter",
            "index finger locking",
            "ankle wobbliness",
            "scalp crawling",
            "elbow catching",
            "jawline quivering",
        ),
        FieldName.DURATION: (
            "6 hours",
            "12 hours",
            "18 hours",
            "24 hours",
            "2 months",
            "3 months",
            "4 months",
            "5 months",
        ),
        FieldName.SEVERITY: ("slight", "noticeable", "strong", "extreme"),
        FieldName.MEDICATION: (
            "acetaminophen gelcap",
            "naproxen caplet",
            "cetirizine chewable",
            "loratadine syrup",
            "saline nasal mist",
            "menthol balm",
            "zinc lozenge",
            "vitamin c gummy",
            "arnica rub",
            "heat wrap",
            "oral rehydration solution",
            "hydrocortisone lotion",
        ),
        FieldName.ALLERGY: (
            "coconut water",
            "latex gloves",
            "silver jewelry",
            "red food dye",
            "pineapple juice",
            "mosquito bites",
            "mold spores",
            "juniper pollen",
            "shellac resin",
            "laundry enzyme",
            "alpaca fiber",
            "chamomile tea",
        ),
    },
    "pool_b": {
        FieldName.CHIEF_COMPLAINT: (
            "voice box scratchiness",
            "hip socket clicking",
            "middle finger jolting",
            "upper back fluttering",
            "jaw corner stiffness",
            "heel pad buzzing",
            "ear canal fullness",
            "wrist joint catching",
            "lower rib quivering",
            "ankle joint locking",
            "scalp line prickling",
            "elbow tip pulsation",
        ),
        FieldName.DURATION: (
            "30 hours",
            "36 hours",
            "42 hours",
            "48 hours",
            "6 months",
            "7 months",
            "8 months",
            "9 months",
        ),
        FieldName.SEVERITY: ("faint", "marked", "intense", "overwhelming"),
        FieldName.MEDICATION: (
            "aspirin effervescent tablet",
            "fexofenadine liquid",
            "guaifenesin syrup",
            "ibuprofen suspension",
            "calamine lotion",
            "petrolatum ointment",
            "witch hazel pad",
            "eucalyptus vapor rub",
            "sodium bicarbonate rinse",
            "cooling eye mask",
            "electrolyte drink",
            "throat spray",
        ),
        FieldName.ALLERGY: (
            "blackberries",
            "duck eggs",
            "squid",
            "neoprene",
            "epoxy resin",
            "maple pollen",
            "mugwort pollen",
            "detergent fragrance",
            "cinnamon oil",
            "jellyfish stings",
            "silk fabric",
            "hemp seeds",
        ),
    },
}

_QUESTIONS: Mapping[str, Mapping[FieldName, tuple[str, ...]]] = {
    "pool_a": {
        FieldName.CHIEF_COMPLAINT: (
            "Start with the main concern: what are you noticing?",
            "Which symptom led you to arrange this visit?",
        ),
        FieldName.DURATION: (
            "Give me the timing: how long has it been present?",
            "What length of time has this continued?",
        ),
        FieldName.SEVERITY: (
            "How strong does the sensation feel overall?",
            "Which intensity word fits the sensation best?",
        ),
        FieldName.MEDICATION: (
            "What, if anything, have you used for relief?",
            "Name any remedy you have tried for this concern.",
        ),
        FieldName.ALLERGY: (
            "Name any substance that causes you a reaction.",
            "What known reaction trigger should be recorded?",
        ),
    },
    "pool_b": {
        FieldName.CHIEF_COMPLAINT: (
            "Tell me the single issue you most want checked.",
            "What physical change prompted today's conversation?",
        ),
        FieldName.DURATION: (
            "About how much time has passed since it began?",
            "State the approximate span of time involved.",
        ),
        FieldName.SEVERITY: (
            "How intense is the experience at the moment?",
            "Choose a word for the current strength of it.",
        ),
        FieldName.MEDICATION: (
            "Which treatment have you already used, if any?",
            "What have you taken or applied for the problem?",
        ),
        FieldName.ALLERGY: (
            "Which material or substance gives you a reaction?",
            "Identify any allergy trigger relevant to your care.",
        ),
    },
}

_ANSWERS: Mapping[str, Mapping[FieldName, tuple[str, ...]]] = {
    "pool_a": {
        FieldName.CHIEF_COMPLAINT: (
            "The main thing is {value}.",
            "I want help with {value}.",
        ),
        FieldName.DURATION: (
            "The timing is {value}.",
            "It has lasted {value} so far.",
        ),
        FieldName.SEVERITY: (
            "The best word is {value}.",
            "Overall it feels {value}.",
        ),
        FieldName.MEDICATION: (
            "For relief I used {value}.",
            "The remedy I tried was {value}.",
        ),
        FieldName.ALLERGY: (
            "A known trigger for me is {value}.",
            "I react badly to {value}.",
        ),
    },
    "pool_b": {
        FieldName.CHIEF_COMPLAINT: (
            "Please record {value} as the issue.",
            "What I have noticed is {value}.",
        ),
        FieldName.DURATION: (
            "The elapsed time is {value}.",
            "This has continued for {value}.",
        ),
        FieldName.SEVERITY: (
            "At present it is {value}.",
            "I would grade the feeling as {value}.",
        ),
        FieldName.MEDICATION: (
            "I attempted relief with {value}.",
            "So far I have used {value}.",
        ),
        FieldName.ALLERGY: (
            "My reaction trigger is {value}.",
            "The substance I cannot tolerate is {value}.",
        ),
    },
}

_DENIALS: Mapping[str, Mapping[FieldName, tuple[str, ...]]] = {
    "pool_a": {
        FieldName.MEDICATION: ("No medicine.",),
        FieldName.ALLERGY: ("No known allergy.",),
    },
    "pool_b": {
        FieldName.MEDICATION: ("I deny taking medication.",),
        FieldName.ALLERGY: ("I deny any allergy.",),
    },
}

_UNCERTAINTY_MARKERS: Mapping[str, tuple[str, ...]] = {
    "pool_a": ("Unclear.",),
    "pool_b": ("Unknown.",),
}

_NATIVE_LITERAL_POOLS = (
    "_TRAIN_CC_PARTS",
    "_TRAIN_CC_SIGNS",
    "_DEV_CC_PARTS",
    "_DEV_CC_SIGNS",
    "_TRAIN_MEDICATIONS",
    "_DEV_MEDICATIONS",
    "_TRAIN_ALLERGIES",
    "_DEV_ALLERGIES",
)
_NATIVE_TEMPLATE_DEFINITIONS = (
    "_QUESTIONS",
    "_ANSWERS",
    "_DENIALS",
    "_UNCERTAIN",
)

_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "benchmark_id",
        "contract_version",
        "status",
        "seed",
        "protocol",
        "generator",
        "native_source",
        "partition",
        "composition",
        "challenge_quota",
        "isolation",
    }
)
_PROTOCOL_KEYS = frozenset(
    {
        "purpose",
        "candidate_blind_generation",
        "candidate_inputs_used",
        "result_inputs_used",
        "inference_performed",
    }
)
_SOURCE_KEYS = frozenset({"path", "sha256"})
_PARTITION_SPEC_KEYS = frozenset({"path", "records", "fields", "sha256"})
_COMPOSITION_KEYS = frozenset(
    {
        "normal",
        "normal_pool_a",
        "normal_pool_b",
        "state_challenge",
        "missing",
        "uncertain",
        "conflicting",
        "paired_normal",
    }
)
_QUOTA_KEYS = frozenset(
    {"states", "fields", "value_bands", "per_state_field_band", "total"}
)
_ISOLATION_KEYS = frozenset(
    {
        "native_records_imported",
        "value_pools_disjoint",
        "surface_templates_disjoint",
        "fresh_value_inventory_sha256",
        "fresh_template_inventory_sha256",
        "native_value_inventory_sha256",
        "native_template_inventory_sha256",
    }
)
_PARTITION_KEYS = frozenset(
    {
        "schema_version",
        "benchmark_id",
        "seed",
        "generator_sha256",
        "native_source_sha256",
        "cases",
    }
)
_CASE_KEYS = frozenset(
    {"case_id", "transcript", "transcript_sha256", "gold", "benchmark"}
)
_BENCHMARK_KEYS = frozenset(
    {
        "schema_version",
        "family",
        "variant",
        "value_band",
        "target_state",
        "target_field",
        "pair_id",
    }
)


class ConfirmationIntegrityError(ValueError):
    """A source, generated partition, or seal violated the v1 contract."""


@dataclass(frozen=True, slots=True)
class SurfaceInventory:
    """Defined value pools and normalized surface-template signatures."""

    values: Mapping[FieldName, frozenset[str]]
    templates: frozenset[str]


@dataclass(frozen=True, slots=True)
class IsolationAudit:
    """Pinned source and inventory seals for the leakage audit."""

    native_source_sha256: str
    fresh_value_inventory_sha256: str
    fresh_template_inventory_sha256: str
    native_value_inventory_sha256: str
    native_template_inventory_sha256: str
    value_overlaps: Mapping[FieldName, frozenset[str]]
    template_overlaps: frozenset[str]

    def manifest_dict(self) -> dict[str, Any]:
        return {
            "native_records_imported": False,
            "value_pools_disjoint": not any(self.value_overlaps.values()),
            "surface_templates_disjoint": not self.template_overlaps,
            "fresh_value_inventory_sha256": self.fresh_value_inventory_sha256,
            "fresh_template_inventory_sha256": self.fresh_template_inventory_sha256,
            "native_value_inventory_sha256": self.native_value_inventory_sha256,
            "native_template_inventory_sha256": self.native_template_inventory_sha256,
        }


@dataclass(frozen=True, slots=True)
class FrozenConfirmationPartition:
    """One validated, externally sealed confirmation snapshot."""

    benchmark_id: str
    status: str
    purpose: str
    manifest_sha256: str
    partition_sha256: str
    cases: tuple[FixtureCase, ...]


@dataclass(frozen=True, slots=True)
class WrittenConfirmationPartition:
    """Paths and seals produced by one no-clobber write."""

    output_dir: Path
    manifest_path: Path
    partition_path: Path
    manifest_sha256: str
    partition_sha256: str
    manifest: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _NormalRecord:
    transcript: str
    fields: tuple[FieldOutput, ...]
    payload_ranges: Mapping[FieldName, tuple[int, int]]
    value_band: str
    pair_id: str | None = None
    target_state: FieldState | None = None
    target_field: FieldName | None = None


@dataclass(frozen=True, slots=True)
class _ChallengeRecord:
    transcript: str
    fields: tuple[FieldOutput, ...]
    value_band: str
    target_state: FieldState
    target_field: FieldName
    pair_id: str


def canonical_json_bytes(value: Any) -> bytes:
    """Return the sole accepted representation of a sealed JSON artifact."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _template_signature(value: str) -> str:
    return _normalized_text(value.replace("{value}", "{}"))


def _require_exact_keys(value: Any, keys: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or frozenset(value) != keys:
        raise ConfirmationIntegrityError(f"invalid {label} schema")
    return value


def _require_int(value: Any, label: str, *, expected: int | None = None) -> int:
    if type(value) is not int or (expected is not None and value != expected):
        suffix = "" if expected is None else f" equal to {expected}"
        raise ConfirmationIntegrityError(f"{label} must be an integer{suffix}")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ConfirmationIntegrityError(f"{label} must be a boolean")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ConfirmationIntegrityError(
            f"{label} must be a non-empty edge-trimmed string"
        )
    return value


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ConfirmationIntegrityError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConfirmationIntegrityError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ConfirmationIntegrityError(f"non-finite JSON value: {value}")


def _parse_json_snapshot(snapshot: bytes, label: str) -> Any:
    try:
        return json.loads(
            snapshot.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except ConfirmationIntegrityError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfirmationIntegrityError(f"cannot parse {label}: {exc}") from exc


def _read_regular_snapshot(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ConfirmationIntegrityError(f"cannot open {label}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ConfirmationIntegrityError(f"{label} is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    except OSError as exc:
        raise ConfirmationIntegrityError(f"cannot read {label}: {exc}") from exc
    finally:
        os.close(descriptor)


def _repo_file(repository_root: Path, relative: str) -> Path:
    relative_path = PurePosixPath(relative)
    if (
        relative_path.is_absolute()
        or not relative_path.parts
        or ".." in relative_path.parts
    ):
        raise ConfirmationIntegrityError(f"unsafe repository path: {relative}")
    try:
        root = Path(repository_root).resolve(strict=True)
    except OSError as exc:
        raise ConfirmationIntegrityError("repository root is unavailable") from exc
    if not root.is_dir():
        raise ConfirmationIntegrityError("repository root is not a directory")
    cursor = root
    for part in relative_path.parts:
        cursor /= part
        try:
            if stat.S_ISLNK(cursor.lstat().st_mode):
                raise ConfirmationIntegrityError(
                    f"repository path uses a symlink: {relative}"
                )
        except FileNotFoundError as exc:
            raise ConfirmationIntegrityError(
                f"repository file is unavailable: {relative}"
            ) from exc
    try:
        cursor.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise ConfirmationIntegrityError(
            f"repository path escapes root: {relative}"
        ) from exc
    return cursor


def _read_repo_snapshot(repository_root: Path, relative: str, label: str) -> bytes:
    return _read_regular_snapshot(_repo_file(repository_root, relative), label)


def _artifact_sibling(manifest_path: Path, relative: Any) -> Path:
    value = _require_string(relative, "partition path")
    relative_path = PurePosixPath(value)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ConfirmationIntegrityError("partition path must be a safe relative path")
    parent = manifest_path.parent.resolve(strict=True)
    candidate_path = parent.joinpath(*relative_path.parts)
    try:
        candidate_path.resolve(strict=True).relative_to(parent)
    except (OSError, ValueError) as exc:
        raise ConfirmationIntegrityError(
            "partition path escapes the manifest directory"
        ) from exc
    cursor = parent
    for part in relative_path.parts:
        cursor /= part
        if stat.S_ISLNK(cursor.lstat().st_mode):
            raise ConfirmationIntegrityError("partition path uses a symlink")
    return candidate_path


def fresh_value_pools() -> Mapping[str, Mapping[FieldName, frozenset[str]]]:
    """Return immutable copies of the two evaluator-owned value inventories."""

    return {
        band: {field: frozenset(values) for field, values in fields.items()}
        for band, fields in _VALUE_POOLS.items()
    }


def _fresh_surface_inventory() -> SurfaceInventory:
    values = {
        field: frozenset(
            value for band in VALUE_BANDS for value in _VALUE_POOLS[band][field]
        )
        for field in FIELD_ORDER
    }
    template_values = [
        template
        for band in VALUE_BANDS
        for field in FIELD_ORDER
        for template in (*_QUESTIONS[band][field], *_ANSWERS[band][field])
    ]
    template_values.extend(
        template
        for band in VALUE_BANDS
        for field in (FieldName.MEDICATION, FieldName.ALLERGY)
        for template in _DENIALS[band][field]
    )
    template_values.extend(
        marker for band in VALUE_BANDS for marker in _UNCERTAINTY_MARKERS[band]
    )
    return SurfaceInventory(
        values=values,
        templates=frozenset(_template_signature(item) for item in template_values),
    )


def _assignment_value(tree: ast.Module, name: str) -> ast.expr:
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == name:
                return node.value
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            if node.value is None:
                break
            return node.value
    raise ConfirmationIntegrityError(f"native source definition is missing: {name}")


def _literal_string_tuple(tree: ast.Module, name: str) -> tuple[str, ...]:
    try:
        value = ast.literal_eval(_assignment_value(tree, name))
    except (ValueError, TypeError, SyntaxError) as exc:
        raise ConfirmationIntegrityError(
            f"native source definition is not literal: {name}"
        ) from exc
    if (
        not isinstance(value, tuple)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ConfirmationIntegrityError(f"native source definition is invalid: {name}")
    return value


def load_native_surface_inventory(repository_root: Path) -> SurfaceInventory:
    """Parse pinned native literals without importing or generating records."""

    snapshot = _read_repo_snapshot(
        repository_root, NATIVE_SOURCE_PATH, "native state/span source"
    )
    if _sha256_bytes(snapshot) != NATIVE_SOURCE_SHA256:
        raise ConfirmationIntegrityError("native state/span source hash mismatch")
    try:
        tree = ast.parse(snapshot.decode("utf-8"), filename=NATIVE_SOURCE_PATH)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise ConfirmationIntegrityError(
            "cannot parse native state/span source"
        ) from exc

    literals = {
        name: _literal_string_tuple(tree, name) for name in _NATIVE_LITERAL_POOLS
    }
    train_complaints = {
        f"{part} {sign}"
        for part in literals["_TRAIN_CC_PARTS"]
        for sign in literals["_TRAIN_CC_SIGNS"]
    }
    dev_complaints = {
        f"{part} {sign}"
        for part in literals["_DEV_CC_PARTS"]
        for sign in literals["_DEV_CC_SIGNS"]
    }
    values = {
        FieldName.CHIEF_COMPLAINT: frozenset(train_complaints | dev_complaints),
        FieldName.DURATION: frozenset(
            {
                *(f"{number} days" for number in range(2, 20)),
                *(f"{number} weeks" for number in range(1, 9)),
            }
        ),
        FieldName.SEVERITY: frozenset({"mild", "moderate", "severe"}),
        FieldName.MEDICATION: frozenset(
            literals["_TRAIN_MEDICATIONS"] + literals["_DEV_MEDICATIONS"]
        ),
        FieldName.ALLERGY: frozenset(
            literals["_TRAIN_ALLERGIES"] + literals["_DEV_ALLERGIES"]
        ),
    }
    template_strings: set[str] = set()
    for name in _NATIVE_TEMPLATE_DEFINITIONS:
        node = _assignment_value(tree, name)
        template_strings.update(
            item.value
            for item in ast.walk(node)
            if isinstance(item, ast.Constant)
            and isinstance(item.value, str)
            and item.value not in {"train", "dev"}
        )
    if not template_strings:
        raise ConfirmationIntegrityError("native template inventory is empty")
    return SurfaceInventory(
        values=values,
        templates=frozenset(_template_signature(item) for item in template_strings),
    )


def _value_inventory_document(
    values: Mapping[FieldName, Iterable[str]],
) -> dict[str, list[str]]:
    return {field.value: sorted(values[field]) for field in FIELD_ORDER}


def _inventory_sha256(inventory: SurfaceInventory, *, values: bool) -> str:
    document: Any
    if values:
        document = _value_inventory_document(inventory.values)
    else:
        document = sorted(inventory.templates)
    return _sha256_bytes(canonical_json_bytes(document))


def audit_native_surface_disjointness(repository_root: Path) -> IsolationAudit:
    """Prove source-level value and surface-template disjointness."""

    fresh = _fresh_surface_inventory()
    native = load_native_surface_inventory(repository_root)
    overlaps = {
        field: frozenset(fresh.values[field] & native.values[field])
        for field in FIELD_ORDER
    }
    template_overlaps = frozenset(fresh.templates & native.templates)
    audit = IsolationAudit(
        native_source_sha256=NATIVE_SOURCE_SHA256,
        fresh_value_inventory_sha256=_inventory_sha256(fresh, values=True),
        fresh_template_inventory_sha256=_inventory_sha256(fresh, values=False),
        native_value_inventory_sha256=_inventory_sha256(native, values=True),
        native_template_inventory_sha256=_inventory_sha256(native, values=False),
        value_overlaps=overlaps,
        template_overlaps=template_overlaps,
    )
    if any(overlaps.values()):
        raise ConfirmationIntegrityError("fresh/native value pools overlap")
    if template_overlaps:
        raise ConfirmationIntegrityError("fresh/native surface templates overlap")
    return audit


def _sample_values(rng: random.Random, band: str) -> Mapping[FieldName, str | None]:
    pools = _VALUE_POOLS[band]
    return {
        FieldName.CHIEF_COMPLAINT: rng.choice(pools[FieldName.CHIEF_COMPLAINT]),
        FieldName.DURATION: rng.choice(pools[FieldName.DURATION]),
        FieldName.SEVERITY: rng.choice(pools[FieldName.SEVERITY]),
        FieldName.MEDICATION: (
            None if rng.random() < 0.25 else rng.choice(pools[FieldName.MEDICATION])
        ),
        FieldName.ALLERGY: (
            None if rng.random() < 0.25 else rng.choice(pools[FieldName.ALLERGY])
        ),
    }


def _normal_record(rng: random.Random, band: str) -> _NormalRecord:
    values = _sample_values(rng, band)
    lines: list[tuple[str, str, FieldName | None, str | None]] = []
    for field in FIELD_ORDER:
        lines.append(("Doctor", rng.choice(_QUESTIONS[band][field]), None, None))
        value = values[field]
        if value is None:
            answer = rng.choice(_DENIALS[band][field])
            evidence = answer
        else:
            answer = rng.choice(_ANSWERS[band][field]).format(value=value)
            evidence = value
        lines.append(("Patient", answer, field, evidence))

    transcript = "\n".join(f"{speaker}: {text}" for speaker, text, _, _ in lines)
    spans: dict[FieldName, EvidenceSpan] = {}
    payload_ranges: dict[FieldName, tuple[int, int]] = {}
    offset = 0
    for speaker, text, field, evidence in lines:
        rendered = f"{speaker}: {text}"
        if field is not None and evidence is not None:
            payload_start = offset + len("Patient: ")
            evidence_start = payload_start + text.index(evidence)
            spans[field] = EvidenceSpan(
                start=evidence_start,
                end=evidence_start + len(evidence),
                text=evidence,
            )
            payload_ranges[field] = (payload_start, payload_start + len(text))
        offset += len(rendered) + 1

    fields = tuple(
        FieldOutput(
            field=field,
            state=(
                FieldState.ABSENT if values[field] is None else FieldState.SUPPORTED
            ),
            value=values[field],
            evidence=(spans[field],),
        )
        for field in FIELD_ORDER
    )
    request = NanoInput(item_id="fresh-v1-generation-check", transcript=transcript)
    NanoOutput(
        item_id=request.item_id, solver_id=GOLD_SOLVER_ID, fields=fields
    ).validate_against(request)
    return _NormalRecord(
        transcript=transcript,
        fields=fields,
        payload_ranges=payload_ranges,
        value_band=band,
    )


def _alternate_value(band: str, field: FieldName, current: str | None) -> str:
    current_normalized = "" if current is None else _normalized_text(current)
    return next(
        value
        for value in _VALUE_POOLS[band][field]
        if _normalized_text(value) != current_normalized
    )


def _challenge_record(
    normal: _NormalRecord,
    state: FieldState,
    field: FieldName,
    pair_id: str,
) -> _ChallengeRecord:
    fields = list(normal.fields)
    field_index = FIELD_ORDER.index(field)
    original = fields[field_index]
    if state in {FieldState.MISSING, FieldState.UNCERTAIN}:
        start, end = normal.payload_ranges[field]
        if state is FieldState.MISSING:
            line_start = start - len("Patient: ")
            replacement = " " * (end - line_start)
            transcript = (
                normal.transcript[:line_start] + replacement + normal.transcript[end:]
            )
            target = FieldOutput(field=field, state=state)
        else:
            marker = _UNCERTAINTY_MARKERS[normal.value_band][0]
            if len(marker) > end - start:
                raise ConfirmationIntegrityError(
                    "target payload is too short for uncertainty marker"
                )
            replacement = marker + " " * (end - start - len(marker))
            transcript = (
                normal.transcript[:start] + replacement + normal.transcript[end:]
            )
            target = FieldOutput(
                field=field,
                state=state,
                evidence=(
                    EvidenceSpan(start=start, end=start + len(marker), text=marker),
                ),
            )
    else:
        alternate = _alternate_value(normal.value_band, field, original.value)
        question = _QUESTIONS[normal.value_band][field][0]
        answer = _ANSWERS[normal.value_band][field][0].format(value=alternate)
        addition = f"\nDoctor: {question}\nPatient: {answer}"
        transcript = normal.transcript + addition
        evidence_start = (
            len(normal.transcript)
            + len(f"\nDoctor: {question}\nPatient: ")
            + answer.index(alternate)
        )
        alternate_span = EvidenceSpan(
            start=evidence_start,
            end=evidence_start + len(alternate),
            text=alternate,
        )
        target = FieldOutput(
            field=field,
            state=state,
            evidence=original.evidence + (alternate_span,),
        )
    fields[field_index] = target
    request = NanoInput(item_id="fresh-v1-generation-check", transcript=transcript)
    result_fields = tuple(fields)
    NanoOutput(
        item_id=request.item_id,
        solver_id=GOLD_SOLVER_ID,
        fields=result_fields,
    ).validate_against(request)
    return _ChallengeRecord(
        transcript=transcript,
        fields=result_fields,
        value_band=normal.value_band,
        target_state=state,
        target_field=field,
        pair_id=pair_id,
    )


def _generate_records(
    seed: int = FRESH_SEED,
) -> tuple[_NormalRecord | _ChallengeRecord, ...]:
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    rng = random.Random(seed)
    normals: list[_NormalRecord] = []
    exact_hashes: set[str] = set()
    normalized_hashes: set[str] = set()
    for band in VALUE_BANDS:
        count = 0
        attempts = 0
        while count < NORMAL_RECORDS_PER_BAND:
            attempts += 1
            if attempts > 100_000:
                raise ConfirmationIntegrityError(
                    f"could not construct unique {band} records"
                )
            record = _normal_record(rng, band)
            exact = _sha256_text(record.transcript)
            normalized = _sha256_text(_normalized_text(record.transcript))
            if exact in exact_hashes or normalized in normalized_hashes:
                continue
            exact_hashes.add(exact)
            normalized_hashes.add(normalized)
            normals.append(record)
            count += 1

    challenges: list[_ChallengeRecord] = []
    pair_number = 0
    for band in VALUE_BANDS:
        available = [
            index for index, record in enumerate(normals) if record.value_band == band
        ]
        rng.shuffle(available)
        cursor = 0
        for state in TARGET_STATES:
            for field in FIELD_ORDER:
                accepted = 0
                while accepted < CELL_QUOTA:
                    if cursor >= len(available):
                        raise ConfirmationIntegrityError(
                            "could not construct unique challenge quota"
                        )
                    normal_index = available[cursor]
                    cursor += 1
                    pair_id = f"fresh-v1-pair-{pair_number:04d}"
                    challenge = _challenge_record(
                        normals[normal_index], state, field, pair_id
                    )
                    exact = _sha256_text(challenge.transcript)
                    normalized = _sha256_text(_normalized_text(challenge.transcript))
                    if exact in exact_hashes or normalized in normalized_hashes:
                        continue
                    normals[normal_index] = replace(
                        normals[normal_index],
                        pair_id=pair_id,
                        target_state=state,
                        target_field=field,
                    )
                    exact_hashes.add(exact)
                    normalized_hashes.add(normalized)
                    challenges.append(challenge)
                    pair_number += 1
                    accepted += 1

    records: list[_NormalRecord | _ChallengeRecord] = [*normals, *challenges]
    rng.shuffle(records)
    return tuple(records)


def _benchmark_metadata(record: _NormalRecord | _ChallengeRecord) -> dict[str, Any]:
    if isinstance(record, _ChallengeRecord):
        return {
            "schema_version": BENCHMARK_PROVENANCE_SCHEMA_VERSION,
            "family": "state_challenge",
            "variant": record.target_state.value,
            "value_band": record.value_band,
            "target_state": record.target_state.value,
            "target_field": record.target_field.value,
            "pair_id": record.pair_id,
        }
    return {
        "schema_version": BENCHMARK_PROVENANCE_SCHEMA_VERSION,
        "family": "normal",
        "variant": "normal",
        "value_band": record.value_band,
        "target_state": (
            record.target_state.value if record.target_state is not None else None
        ),
        "target_field": (
            record.target_field.value if record.target_field is not None else None
        ),
        "pair_id": record.pair_id,
    }


def _partition_document_from_records(
    records: Sequence[_NormalRecord | _ChallengeRecord],
    *,
    generator_sha256: str,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        case_id = f"nano-fresh-v1-{index:04d}"
        request = NanoInput(item_id=case_id, transcript=record.transcript)
        output = NanoOutput(
            item_id=case_id, solver_id=GOLD_SOLVER_ID, fields=record.fields
        )
        output.validate_against(request)
        cases.append(
            {
                "case_id": case_id,
                "transcript": record.transcript,
                "transcript_sha256": _sha256_text(record.transcript),
                "gold": output.to_dict(),
                "benchmark": _benchmark_metadata(record),
            }
        )
    return {
        "schema_version": PARTITION_SCHEMA_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "seed": FRESH_SEED,
        "generator_sha256": generator_sha256,
        "native_source_sha256": NATIVE_SOURCE_SHA256,
        "cases": cases,
    }


def _generator_sha256(repository_root: Path) -> str:
    return _sha256_bytes(
        _read_repo_snapshot(repository_root, GENERATOR_PATH, "fresh-v1 generator")
    )


def build_fresh_partition_document(repository_root: Path) -> dict[str, Any]:
    """Regenerate the complete evaluator partition in memory, without inference."""

    audit = audit_native_surface_disjointness(repository_root)
    generator_sha256 = _generator_sha256(repository_root)
    document = _partition_document_from_records(
        _generate_records(), generator_sha256=generator_sha256
    )
    _validate_partition(document, audit, generator_sha256)
    return document


def _composition() -> dict[str, int]:
    return {
        "normal": NORMAL_RECORDS,
        "normal_pool_a": NORMAL_RECORDS_PER_BAND,
        "normal_pool_b": NORMAL_RECORDS_PER_BAND,
        "state_challenge": CHALLENGE_RECORDS,
        "missing": STATE_CHALLENGE_RECORDS,
        "uncertain": STATE_CHALLENGE_RECORDS,
        "conflicting": STATE_CHALLENGE_RECORDS,
        "paired_normal": PAIRED_NORMAL_RECORDS,
    }


def _challenge_quota() -> dict[str, Any]:
    return {
        "states": [state.value for state in TARGET_STATES],
        "fields": [field.value for field in FIELD_ORDER],
        "value_bands": list(VALUE_BANDS),
        "per_state_field_band": CELL_QUOTA,
        "total": CHALLENGE_RECORDS,
    }


def _protocol() -> dict[str, Any]:
    return {
        "purpose": PURPOSE,
        "candidate_blind_generation": True,
        "candidate_inputs_used": False,
        "result_inputs_used": False,
        "inference_performed": False,
    }


def build_sealed_manifest(
    partition_document: Mapping[str, Any],
    *,
    partition_path: str,
    repository_root: Path,
) -> dict[str, Any]:
    """Build the canonical manifest for separately persisted partition bytes."""

    relative = PurePosixPath(partition_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ConfirmationIntegrityError("partition path must be a safe relative path")
    audit = audit_native_surface_disjointness(repository_root)
    generator_sha256 = _generator_sha256(repository_root)
    _validate_partition(dict(partition_document), audit, generator_sha256)
    partition_bytes = canonical_json_bytes(partition_document)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "contract_version": CONTRACT_VERSION,
        "status": "sealed_unmeasured",
        "seed": FRESH_SEED,
        "protocol": _protocol(),
        "generator": {"path": GENERATOR_PATH, "sha256": generator_sha256},
        "native_source": {
            "path": NATIVE_SOURCE_PATH,
            "sha256": NATIVE_SOURCE_SHA256,
        },
        "partition": {
            "path": partition_path,
            "records": TOTAL_RECORDS,
            "fields": TOTAL_FIELDS,
            "sha256": _sha256_bytes(partition_bytes),
        },
        "composition": _composition(),
        "challenge_quota": _challenge_quota(),
        "isolation": audit.manifest_dict(),
    }


def _load_case(row: Any, index: int) -> FixtureCase:
    case = _require_exact_keys(row, _CASE_KEYS, f"case {index}")
    case_id = _require_string(case["case_id"], f"case {index} ID")
    if case_id != f"nano-fresh-v1-{index:04d}":
        raise ConfirmationIntegrityError("case IDs are not neutral and sequential")
    transcript = case["transcript"]
    if not isinstance(transcript, str) or not transcript.strip():
        raise ConfirmationIntegrityError(f"invalid transcript: {case_id}")
    digest = _require_sha256(case["transcript_sha256"], f"transcript digest {case_id}")
    if _sha256_text(transcript) != digest:
        raise ConfirmationIntegrityError(f"transcript digest mismatch: {case_id}")
    benchmark = _require_exact_keys(
        case["benchmark"], _BENCHMARK_KEYS, f"benchmark metadata {case_id}"
    )
    if benchmark["schema_version"] != BENCHMARK_PROVENANCE_SCHEMA_VERSION:
        raise ConfirmationIntegrityError(
            f"benchmark provenance version mismatch: {case_id}"
        )
    try:
        request = NanoInput(item_id=case_id, transcript=transcript)
        output = NanoOutput.from_dict(case["gold"])
        if output.solver_id != GOLD_SOLVER_ID:
            raise ConfirmationIntegrityError(f"invalid gold solver ID: {case_id}")
        output.validate_against(request)
        return FixtureCase(
            case_id=case_id,
            partition=BENCHMARK_ID,
            request=request,
            gold=output,
            provenance={"benchmark": dict(benchmark), "transcript_sha256": digest},
        )
    except ConfirmationIntegrityError:
        raise
    except (TypeError, ValueError) as exc:
        raise ConfirmationIntegrityError(
            f"invalid case contract: {case_id}: {exc}"
        ) from exc


def _validate_normal_case(case: FixtureCase, band: str) -> None:
    pools = _VALUE_POOLS[band]
    lines = case.request.transcript.splitlines()
    if len(lines) != len(FIELD_ORDER) * 2:
        raise ConfirmationIntegrityError(
            "normal transcript must contain exactly five question/answer pairs"
        )
    for index, field in enumerate(FIELD_ORDER):
        output = case.gold.field(field)
        question = lines[index * 2]
        answer = lines[index * 2 + 1]
        if question not in {
            f"Doctor: {template}" for template in _QUESTIONS[band][field]
        }:
            raise ConfirmationIntegrityError(
                "normal question is outside its declared fresh templates"
            )
        if output.state is FieldState.SUPPORTED:
            if output.value not in pools[field]:
                raise ConfirmationIntegrityError(
                    "normal value is outside its declared fresh pool"
                )
            expected_answers = {
                f"Patient: {template.format(value=output.value)}"
                for template in _ANSWERS[band][field]
            }
            if answer not in expected_answers:
                raise ConfirmationIntegrityError(
                    "normal answer is outside its declared fresh templates"
                )
        elif output.state is not FieldState.ABSENT or field not in {
            FieldName.MEDICATION,
            FieldName.ALLERGY,
        }:
            raise ConfirmationIntegrityError("normal case has an invalid field state")
        elif answer not in {
            f"Patient: {template}" for template in _DENIALS[band][field]
        }:
            raise ConfirmationIntegrityError(
                "normal denial is outside its declared fresh templates"
            )


def _validate_partition(
    value: Any,
    audit: IsolationAudit,
    generator_sha256: str,
) -> tuple[FixtureCase, ...]:
    root = _require_exact_keys(value, _PARTITION_KEYS, "partition")
    if root["schema_version"] != PARTITION_SCHEMA_VERSION:
        raise ConfirmationIntegrityError("partition schema version mismatch")
    if root["benchmark_id"] != BENCHMARK_ID:
        raise ConfirmationIntegrityError("partition benchmark ID mismatch")
    _require_int(root["seed"], "partition seed", expected=FRESH_SEED)
    if root["generator_sha256"] != generator_sha256:
        raise ConfirmationIntegrityError("partition generator digest mismatch")
    if root["native_source_sha256"] != audit.native_source_sha256:
        raise ConfirmationIntegrityError("partition native-source digest mismatch")
    rows = root["cases"]
    if not isinstance(rows, list) or len(rows) != TOTAL_RECORDS:
        raise ConfirmationIntegrityError("partition must contain exactly 220 cases")
    cases = tuple(_load_case(row, index) for index, row in enumerate(rows))
    if sum(len(case.gold.fields) for case in cases) != TOTAL_FIELDS:
        raise ConfirmationIntegrityError("partition must contain exactly 1100 fields")

    exact_hashes: set[str] = set()
    normalized_hashes: set[str] = set()
    pair_members: dict[str, list[FixtureCase]] = {}
    normal_counts = {band: 0 for band in VALUE_BANDS}
    normal_count = challenge_count = paired_normal = 0
    state_counts = {state.value: 0 for state in TARGET_STATES}
    quota = {
        (state.value, field.value, band): 0
        for state in TARGET_STATES
        for field in FIELD_ORDER
        for band in VALUE_BANDS
    }
    for case in cases:
        transcript = case.request.transcript
        exact = _sha256_text(transcript)
        normalized = _sha256_text(_normalized_text(transcript))
        if exact in exact_hashes or normalized in normalized_hashes:
            raise ConfirmationIntegrityError(
                "partition transcripts are not internally unique"
            )
        exact_hashes.add(exact)
        normalized_hashes.add(normalized)
        metadata = case.provenance["benchmark"]
        family = metadata["family"]
        band = metadata["value_band"]
        pair_id = metadata["pair_id"]
        if band not in VALUE_BANDS:
            raise ConfirmationIntegrityError("invalid benchmark value band")
        if pair_id is not None:
            if not isinstance(pair_id, str) or not pair_id.startswith("fresh-v1-pair-"):
                raise ConfirmationIntegrityError("invalid pair ID")
            pair_members.setdefault(pair_id, []).append(case)
        if family == "normal":
            metadata_nulls = sum(
                item is None
                for item in (
                    metadata["target_state"],
                    metadata["target_field"],
                    pair_id,
                )
            )
            if metadata["variant"] != "normal" or metadata_nulls not in {0, 3}:
                raise ConfirmationIntegrityError("invalid normal-case metadata")
            if metadata_nulls == 0:
                try:
                    state = FieldState(metadata["target_state"])
                    FieldName(metadata["target_field"])
                except (TypeError, ValueError) as exc:
                    raise ConfirmationIntegrityError(
                        "invalid paired-normal target"
                    ) from exc
                if state not in TARGET_STATES:
                    raise ConfirmationIntegrityError(
                        "invalid paired-normal target state"
                    )
            _validate_normal_case(case, band)
            normal_count += 1
            normal_counts[band] += 1
            paired_normal += pair_id is not None
        elif family == "state_challenge":
            if pair_id is None:
                raise ConfirmationIntegrityError("challenge case lacks a pair")
            try:
                state = FieldState(metadata["target_state"])
                field = FieldName(metadata["target_field"])
            except (TypeError, ValueError) as exc:
                raise ConfirmationIntegrityError("invalid challenge target") from exc
            if (
                state not in TARGET_STATES
                or metadata["variant"] != state.value
                or case.gold.field(field).state is not state
            ):
                raise ConfirmationIntegrityError(
                    "challenge target state disagrees with gold"
                )
            if sum(item.state in TARGET_STATES for item in case.gold.fields) != 1:
                raise ConfirmationIntegrityError(
                    "challenge must change exactly one field state"
                )
            challenge_count += 1
            state_counts[state.value] += 1
            quota[(state.value, field.value, band)] += 1
        else:
            raise ConfirmationIntegrityError("invalid benchmark family")

    if (
        normal_count != NORMAL_RECORDS
        or any(count != NORMAL_RECORDS_PER_BAND for count in normal_counts.values())
        or challenge_count != CHALLENGE_RECORDS
        or paired_normal != PAIRED_NORMAL_RECORDS
        or any(count != STATE_CHALLENGE_RECORDS for count in state_counts.values())
        or any(count != CELL_QUOTA for count in quota.values())
    ):
        raise ConfirmationIntegrityError(
            "partition composition or challenge quotas are invalid"
        )
    if len(pair_members) != CHALLENGE_RECORDS or any(
        len(members) != 2 for members in pair_members.values()
    ):
        raise ConfirmationIntegrityError("partition pairs are incomplete or duplicated")

    for pair_id, members in pair_members.items():
        normal = next(
            (
                member
                for member in members
                if member.provenance["benchmark"]["family"] == "normal"
            ),
            None,
        )
        challenge = next(
            (
                member
                for member in members
                if member.provenance["benchmark"]["family"] == "state_challenge"
            ),
            None,
        )
        if normal is None or challenge is None:
            raise ConfirmationIntegrityError(
                f"pair lacks one normal and one challenge: {pair_id}"
            )
        normal_meta = normal.provenance["benchmark"]
        challenge_meta = challenge.provenance["benchmark"]
        if (
            normal_meta["value_band"] != challenge_meta["value_band"]
            or normal_meta["target_state"] != challenge_meta["target_state"]
            or normal_meta["target_field"] != challenge_meta["target_field"]
        ):
            raise ConfirmationIntegrityError(f"pair metadata disagrees: {pair_id}")
        target_field = FieldName(challenge_meta["target_field"])
        target_state = FieldState(challenge_meta["target_state"])
        for field in FIELD_ORDER:
            if field is not target_field and (
                normal.gold.field(field).to_dict()
                != challenge.gold.field(field).to_dict()
            ):
                raise ConfirmationIntegrityError(
                    f"challenge changed unaffected gold: {pair_id}"
                )
        if target_state in {FieldState.MISSING, FieldState.UNCERTAIN}:
            if len(normal.request.transcript) != len(challenge.request.transcript):
                raise ConfirmationIntegrityError(
                    f"replacement challenge changed transcript length: {pair_id}"
                )
            normal_spans = normal.gold.field(target_field).evidence
            if not normal_spans:
                raise ConfirmationIntegrityError(
                    f"normal target lacks evidence: {pair_id}"
                )
            first = min(span.start for span in normal_spans)
            line_start = normal.request.transcript.rfind("\n", 0, first) + 1
            line_end = normal.request.transcript.find("\n", first)
            if line_end < 0:
                line_end = len(normal.request.transcript)
            prefix = "Patient: "
            if not normal.request.transcript.startswith(prefix, line_start):
                raise ConfirmationIntegrityError(
                    f"target evidence is not in a patient answer: {pair_id}"
                )
            if target_state is FieldState.MISSING:
                replacement = " " * (line_end - line_start)
                expected_transcript = (
                    normal.request.transcript[:line_start]
                    + replacement
                    + normal.request.transcript[line_end:]
                )
            else:
                marker = _UNCERTAINTY_MARKERS[normal_meta["value_band"]][0]
                payload_start = line_start + len(prefix)
                replacement = marker + " " * (line_end - payload_start - len(marker))
                expected_transcript = (
                    normal.request.transcript[:payload_start]
                    + replacement
                    + normal.request.transcript[line_end:]
                )
                expected_evidence = (
                    EvidenceSpan(
                        start=payload_start,
                        end=payload_start + len(marker),
                        text=marker,
                    ),
                )
                if challenge.gold.field(target_field).evidence != expected_evidence:
                    raise ConfirmationIntegrityError(
                        f"uncertainty evidence is not canonical: {pair_id}"
                    )
            if challenge.request.transcript != expected_transcript:
                raise ConfirmationIntegrityError(
                    f"replacement challenge changed more than its target: {pair_id}"
                )
        else:
            if not challenge.request.transcript.startswith(
                normal.request.transcript + "\nDoctor: "
            ):
                raise ConfirmationIntegrityError(
                    f"conflict challenge is not append-only: {pair_id}"
                )
            appended = challenge.request.transcript[
                len(normal.request.transcript) + 1 :
            ]
            lines = appended.splitlines()
            if (
                len(lines) != 2
                or not lines[0].startswith("Doctor: ")
                or not lines[1].startswith("Patient: ")
            ):
                raise ConfirmationIntegrityError(
                    f"conflict challenge must append one question and answer: {pair_id}"
                )
            normal_target = normal.gold.field(target_field)
            challenge_target = challenge.gold.field(target_field)
            if (
                tuple(challenge_target.evidence[: len(normal_target.evidence)])
                != normal_target.evidence
            ):
                raise ConfirmationIntegrityError(
                    f"conflict challenge lost original evidence: {pair_id}"
                )
            if len(challenge_target.evidence) != len(normal_target.evidence) + 1:
                raise ConfirmationIntegrityError(
                    f"conflict challenge has non-canonical evidence: {pair_id}"
                )
            alternate = challenge_target.evidence[-1].text
            band = normal_meta["value_band"]
            if alternate not in _VALUE_POOLS[band][target_field]:
                raise ConfirmationIntegrityError(
                    f"conflict value is outside its declared fresh pool: {pair_id}"
                )
            expected_appendix = (
                f"Doctor: {_QUESTIONS[band][target_field][0]}\n"
                f"Patient: {_ANSWERS[band][target_field][0].format(value=alternate)}"
            )
            if appended != expected_appendix:
                raise ConfirmationIntegrityError(
                    f"conflict challenge uses a non-canonical appendix: {pair_id}"
                )
    return cases


def validate_partition_document(
    value: Any, *, repository_root: Path
) -> tuple[FixtureCase, ...]:
    """Strictly validate one in-memory partition against the pinned sources."""

    audit = audit_native_surface_disjointness(repository_root)
    return _validate_partition(value, audit, _generator_sha256(repository_root))


def _validate_manifest(
    value: Any,
    *,
    audit: IsolationAudit,
    generator_sha256: str,
) -> dict[str, Any]:
    root = _require_exact_keys(value, _MANIFEST_KEYS, "confirmation manifest")
    if root["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ConfirmationIntegrityError("manifest schema version mismatch")
    if (
        root["benchmark_id"] != BENCHMARK_ID
        or root["contract_version"] != CONTRACT_VERSION
    ):
        raise ConfirmationIntegrityError("manifest benchmark or contract mismatch")
    if root["status"] != "sealed_unmeasured":
        raise ConfirmationIntegrityError("manifest status is invalid")
    _require_int(root["seed"], "manifest seed", expected=FRESH_SEED)
    protocol = _require_exact_keys(root["protocol"], _PROTOCOL_KEYS, "protocol")
    _require_string(protocol["purpose"], "protocol purpose")
    for key in (
        "candidate_blind_generation",
        "candidate_inputs_used",
        "result_inputs_used",
        "inference_performed",
    ):
        _require_bool(protocol[key], f"protocol {key}")
    if protocol != _protocol():
        raise ConfirmationIntegrityError("manifest protocol mismatch")
    generator = _require_exact_keys(root["generator"], _SOURCE_KEYS, "generator")
    if generator != {"path": GENERATOR_PATH, "sha256": generator_sha256}:
        raise ConfirmationIntegrityError("manifest generator declaration mismatch")
    native = _require_exact_keys(root["native_source"], _SOURCE_KEYS, "native source")
    if native != {"path": NATIVE_SOURCE_PATH, "sha256": NATIVE_SOURCE_SHA256}:
        raise ConfirmationIntegrityError("manifest native-source declaration mismatch")
    partition = _require_exact_keys(
        root["partition"], _PARTITION_SPEC_KEYS, "partition declaration"
    )
    _require_string(partition["path"], "partition path")
    _require_int(partition["records"], "partition records", expected=TOTAL_RECORDS)
    _require_int(partition["fields"], "partition fields", expected=TOTAL_FIELDS)
    _require_sha256(partition["sha256"], "partition digest")
    composition = _require_exact_keys(
        root["composition"], _COMPOSITION_KEYS, "composition"
    )
    if composition != _composition() or any(
        type(count) is not int for count in composition.values()
    ):
        raise ConfirmationIntegrityError("manifest composition mismatch")
    quota = _require_exact_keys(root["challenge_quota"], _QUOTA_KEYS, "challenge quota")
    _require_int(
        quota["per_state_field_band"],
        "per-state/field/band quota",
        expected=CELL_QUOTA,
    )
    _require_int(quota["total"], "challenge quota total", expected=CHALLENGE_RECORDS)
    if quota != _challenge_quota():
        raise ConfirmationIntegrityError("manifest challenge quota mismatch")
    isolation = _require_exact_keys(root["isolation"], _ISOLATION_KEYS, "isolation")
    for key in (
        "native_records_imported",
        "value_pools_disjoint",
        "surface_templates_disjoint",
    ):
        _require_bool(isolation[key], f"isolation {key}")
    for key in (
        "fresh_value_inventory_sha256",
        "fresh_template_inventory_sha256",
        "native_value_inventory_sha256",
        "native_template_inventory_sha256",
    ):
        _require_sha256(isolation[key], f"isolation {key}")
    if isolation != audit.manifest_dict():
        raise ConfirmationIntegrityError("manifest isolation declaration mismatch")
    return root


def load_sealed_confirmation_partition(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    repository_root: Path,
) -> FrozenConfirmationPartition:
    """Load one canonical manifest/partition pair under an external seal."""

    expected = _require_sha256(expected_manifest_sha256, "expected manifest digest")
    manifest_path = Path(manifest_path)
    manifest_snapshot = _read_regular_snapshot(manifest_path, "confirmation manifest")
    manifest_sha256 = _sha256_bytes(manifest_snapshot)
    if manifest_sha256 != expected:
        raise ConfirmationIntegrityError("confirmation manifest seal mismatch")
    manifest_value = _parse_json_snapshot(manifest_snapshot, "confirmation manifest")
    if canonical_json_bytes(manifest_value) != manifest_snapshot:
        raise ConfirmationIntegrityError("confirmation manifest is not canonical JSON")

    audit = audit_native_surface_disjointness(repository_root)
    generator_sha256 = _generator_sha256(repository_root)
    manifest = _validate_manifest(
        manifest_value, audit=audit, generator_sha256=generator_sha256
    )
    partition_path = _artifact_sibling(manifest_path, manifest["partition"]["path"])
    partition_snapshot = _read_regular_snapshot(
        partition_path, "confirmation partition"
    )
    partition_sha256 = _sha256_bytes(partition_snapshot)
    if partition_sha256 != manifest["partition"]["sha256"]:
        raise ConfirmationIntegrityError("confirmation partition seal mismatch")
    partition_value = _parse_json_snapshot(partition_snapshot, "confirmation partition")
    if canonical_json_bytes(partition_value) != partition_snapshot:
        raise ConfirmationIntegrityError("confirmation partition is not canonical JSON")
    cases = _validate_partition(partition_value, audit, generator_sha256)
    return FrozenConfirmationPartition(
        benchmark_id=manifest["benchmark_id"],
        status=manifest["status"],
        purpose=manifest["protocol"]["purpose"],
        manifest_sha256=manifest_sha256,
        partition_sha256=partition_sha256,
        cases=cases,
    )


def _write_exclusive(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError:
        raise
    except OSError as exc:
        raise ConfirmationIntegrityError(f"cannot write {path.name}: {exc}") from exc


def write_fresh_confirmation_partition(
    output_dir: Path,
    *,
    repository_root: Path,
) -> WrittenConfirmationPartition:
    """Regenerate and create a sealed two-file family without clobbering."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    partition_document = build_fresh_partition_document(repository_root)
    partition_bytes = canonical_json_bytes(partition_document)
    manifest = build_sealed_manifest(
        partition_document,
        partition_path="partition.json",
        repository_root=repository_root,
    )
    manifest_bytes = canonical_json_bytes(manifest)
    output.mkdir(parents=True, exist_ok=False)
    partition_path = output / "partition.json"
    manifest_path = output / "manifest.json"
    _write_exclusive(partition_path, partition_bytes)
    _write_exclusive(manifest_path, manifest_bytes)
    return WrittenConfirmationPartition(
        output_dir=output,
        manifest_path=manifest_path,
        partition_path=partition_path,
        manifest_sha256=_sha256_bytes(manifest_bytes),
        partition_sha256=_sha256_bytes(partition_bytes),
        manifest=manifest,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Nano's candidate-blind fresh-v1 confirmation partition"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    written = write_fresh_confirmation_partition(
        args.output_dir, repository_root=args.repository_root
    )
    print(
        json.dumps(
            {
                "benchmark_id": BENCHMARK_ID,
                "manifest_path": str(written.manifest_path),
                "manifest_sha256": written.manifest_sha256,
                "partition_path": str(written.partition_path),
                "partition_sha256": written.partition_sha256,
                "purpose": PURPOSE,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BENCHMARK_ID",
    "BENCHMARK_PROVENANCE_SCHEMA_VERSION",
    "CELL_QUOTA",
    "CHALLENGE_RECORDS",
    "FRESH_SEED",
    "GENERATOR_PATH",
    "GOLD_SOLVER_ID",
    "MANIFEST_SCHEMA_VERSION",
    "NATIVE_SOURCE_PATH",
    "NATIVE_SOURCE_SHA256",
    "NORMAL_RECORDS",
    "NORMAL_RECORDS_PER_BAND",
    "PARTITION_SCHEMA_VERSION",
    "PURPOSE",
    "STATE_CHALLENGE_RECORDS",
    "TOTAL_FIELDS",
    "TOTAL_RECORDS",
    "VALUE_BANDS",
    "ConfirmationIntegrityError",
    "FrozenConfirmationPartition",
    "IsolationAudit",
    "SurfaceInventory",
    "WrittenConfirmationPartition",
    "audit_native_surface_disjointness",
    "build_fresh_partition_document",
    "build_sealed_manifest",
    "canonical_json_bytes",
    "fresh_value_pools",
    "load_native_surface_inventory",
    "load_sealed_confirmation_partition",
    "main",
    "validate_partition_document",
    "write_fresh_confirmation_partition",
]
