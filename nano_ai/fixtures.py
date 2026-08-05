"""Versioned, integrity-checked fixtures for Nano's AI contract.

The packaged contract-smoke cases are self-contained. Historical R-star files
are optional provenance sources and are inspected only when callers explicitly
request repository verification.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contract import (
    CONTRACT_VERSION,
    FIELD_ORDER,
    FieldOutput,
    FieldState,
    NanoInput,
    NanoOutput,
)

FIXTURE_SCHEMA_VERSION = "nano.fixtures.v0"
GOLD_SOLVER_ID = "fixture-gold-v0"
CONTRACT_SMOKE_SHA256 = (
    "7d0814f724afca78cf6eb50001bcad7be6d85b6f64928c2d5269e227d8cb9638"
)
_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = Path(__file__).with_name("fixtures") / "v0" / "manifest.json"

_PURPOSE = (
    "Contract conformance and historical provenance for Nano v0; this is not a "
    "fresh scientific benchmark."
)
_SCIENTIFIC_STATUS = {
    "fresh_test_partition": None,
    "historical_eval_is_fresh": False,
    "next_requirement": (
        "Create and seal a disjoint, previously unmeasured Nano v0 test partition "
        "before the next scientific benchmark."
    ),
}
_PARTITIONS: dict[str, dict[str, Any]] = {
    "rstar_train": {
        "path": "trajectory/e4/data/rstar_train.json",
        "records": 800,
        "sha256": ("0c0065c3f01d93a7850d93ab7cb29de8a100a386191401beaf04652ca84e4d13"),
        "role": "historical_training",
        "exposure": "previously_generated_and_used",
    },
    "rstar_dev": {
        "path": "trajectory/e4/data/rstar_dev.json",
        "records": 100,
        "sha256": ("57321d1b729a48944e575df6226ec9ce1a0a05a5c100069c05ee54a78fcb52e6"),
        "role": "historical_validation",
        "exposure": "previously_generated_and_used",
    },
    "rstar_eval": {
        "path": "trajectory/e4/data/rstar_eval.json",
        "records": 220,
        "sha256": ("37e96ff94bff41d25e5b06f22b4d941fc7a2c6067540148e88022779e8fc2ff7"),
        "role": "historical_regression",
        "exposure": "fully_measured_before_nano_v0",
    },
}
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "contract_version",
        "purpose",
        "scientific_status",
        "partitions",
        "contract_smoke",
    }
)
_PARTITION_KEYS = frozenset({"path", "records", "sha256", "role", "exposure"})
_CASE_KEYS = frozenset({"case_id", "partition", "purpose", "source", "gold"})
_GOLD_KEYS = frozenset({"field", "state", "value", "evidence"})
_EVIDENCE_KEYS = frozenset({"start", "end", "text", "speaker"})


class FixtureIntegrityError(ValueError):
    """Raised when a fixture or its provenance does not match the frozen policy."""


@dataclass(frozen=True, slots=True)
class FixtureCase:
    """One evaluator-owned input/truth pair."""

    case_id: str
    partition: str
    request: NanoInput
    gold: NanoOutput
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.case_id, str)
            or not self.case_id
            or self.case_id.strip() != self.case_id
        ):
            raise FixtureIntegrityError(
                "fixture case_id must be a non-empty, edge-trimmed string"
            )
        if (
            not isinstance(self.partition, str)
            or not self.partition
            or self.partition.strip() != self.partition
        ):
            raise FixtureIntegrityError(
                "fixture partition must be a non-empty, edge-trimmed string"
            )
        if not isinstance(self.request, NanoInput):
            raise FixtureIntegrityError("fixture request must be a NanoInput")
        if not isinstance(self.gold, NanoOutput):
            raise FixtureIntegrityError("fixture gold must be a NanoOutput")
        if self.request.item_id != self.case_id or self.gold.item_id != self.case_id:
            raise FixtureIntegrityError(
                "fixture case, request, and gold item IDs must match"
            )
        if not isinstance(self.provenance, dict) or any(
            not isinstance(key, str) for key in self.provenance
        ):
            raise FixtureIntegrityError(
                "fixture provenance must be a string-keyed object"
            )
        try:
            self.gold.validate_against(self.request)
        except ValueError as exc:
            raise FixtureIntegrityError(
                f"fixture gold violates the Nano contract for {self.case_id}: {exc}"
            ) from exc


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FixtureIntegrityError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise FixtureIntegrityError(f"non-finite JSON value: {value}")


def _read_bytes_snapshot(path: Path, *, description: str) -> bytes:
    """Read one immutable snapshot, rejecting non-regular sources."""

    try:
        with path.open("rb") as handle:
            mode = os.fstat(handle.fileno()).st_mode
            if not stat.S_ISREG(mode):
                raise FixtureIntegrityError(
                    f"{description} is not a regular file: {path}"
                )
            return handle.read()
    except FixtureIntegrityError:
        raise
    except OSError as exc:
        raise FixtureIntegrityError(f"cannot read {description} {path}: {exc}") from exc


def _parse_json_snapshot(snapshot: bytes, *, description: str) -> Any:
    try:
        text = snapshot.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except FixtureIntegrityError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FixtureIntegrityError(f"cannot parse {description}: {exc}") from exc


def _load_json(path: Path, *, description: str) -> Any:
    snapshot = _read_bytes_snapshot(path, description=description)
    return _parse_json_snapshot(snapshot, description=f"{description} {path}")


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_record_sha256(record: Any) -> str:
    return _canonical_sha256(record)


def _require_exact_keys(
    value: Any, expected: frozenset[str], label: str
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise FixtureIntegrityError(f"invalid {label} schema")
    return value


def _require_string(value: Any, label: str, *, edge_trimmed: bool = True) -> str:
    if not isinstance(value, str) or not value:
        raise FixtureIntegrityError(f"{label} must be a non-empty string")
    if edge_trimmed and value.strip() != value:
        raise FixtureIntegrityError(f"{label} must be edge-trimmed")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FixtureIntegrityError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_partition_declarations(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != set(_PARTITIONS):
        raise FixtureIntegrityError(
            "fixture partitions do not match the frozen R-star set"
        )
    for name, expected in _PARTITIONS.items():
        spec = _require_exact_keys(value[name], _PARTITION_KEYS, f"partition {name}")
        if spec != expected or type(spec["records"]) is not int:
            raise FixtureIntegrityError(
                f"partition declaration does not match frozen policy: {name}"
            )
    return value


def _validate_evidence_schema(value: Any, *, label: str) -> None:
    span = _require_exact_keys(value, _EVIDENCE_KEYS, f"{label} evidence")
    if (
        type(span["start"]) is not int
        or type(span["end"]) is not int
        or not isinstance(span["text"], str)
        or span["speaker"] != "patient"
    ):
        raise FixtureIntegrityError(f"invalid {label} evidence values")


def _validate_gold_schema(value: Any, *, case_id: str) -> None:
    if not isinstance(value, list) or len(value) != len(FIELD_ORDER):
        raise FixtureIntegrityError(
            f"gold annotations must contain exactly five fields: {case_id}"
        )
    for index, (annotation, expected_field) in enumerate(zip(value, FIELD_ORDER)):
        label = f"gold annotation {case_id}[{index}]"
        field = _require_exact_keys(annotation, _GOLD_KEYS, label)
        if field["field"] != expected_field.value:
            raise FixtureIntegrityError(f"gold fields are out of order: {case_id}")
        try:
            FieldState(field["state"])
        except (TypeError, ValueError) as exc:
            raise FixtureIntegrityError(f"invalid gold state: {case_id}") from exc
        if field["value"] is not None and not isinstance(field["value"], str):
            raise FixtureIntegrityError(f"invalid gold value: {case_id}")
        evidence = field["evidence"]
        if not isinstance(evidence, list):
            raise FixtureIntegrityError(f"gold evidence must be a list: {case_id}")
        for span_index, span in enumerate(evidence):
            _validate_evidence_schema(span, label=f"{label}[{span_index}]")


def _validate_source_schema(value: Any, *, case_id: str) -> None:
    if not isinstance(value, dict):
        raise FixtureIntegrityError(f"invalid smoke source schema: {case_id}")
    source_type = value.get("type")
    if source_type == "inline":
        source = _require_exact_keys(
            value,
            frozenset({"type", "transcript", "transcript_sha256"}),
            f"inline smoke source {case_id}",
        )
    elif source_type == "inline_snapshot":
        source = _require_exact_keys(
            value,
            frozenset({"type", "transcript", "transcript_sha256", "historical_source"}),
            f"historical snapshot source {case_id}",
        )
        historical = _require_exact_keys(
            source["historical_source"],
            frozenset({"partition", "index", "record_sha256"}),
            f"historical source {case_id}",
        )
        if historical["partition"] not in _PARTITIONS:
            raise FixtureIntegrityError(f"unknown historical partition: {case_id}")
        partition_records = _PARTITIONS[historical["partition"]]["records"]
        if (
            type(historical["index"]) is not int
            or historical["index"] < 0
            or historical["index"] >= partition_records
        ):
            raise FixtureIntegrityError(f"invalid historical record index: {case_id}")
        _require_sha256(
            historical["record_sha256"], f"historical record digest {case_id}"
        )
    else:
        raise FixtureIntegrityError(f"invalid smoke source type: {case_id}")

    transcript = _require_string(
        source["transcript"], f"source transcript {case_id}", edge_trimmed=False
    )
    expected = _require_sha256(
        source["transcript_sha256"], f"transcript digest {case_id}"
    )
    actual = hashlib.sha256(transcript.encode("utf-8")).hexdigest()
    if actual != expected:
        raise FixtureIntegrityError(f"inline transcript hash mismatch: {case_id}")


def _validate_contract_smoke(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise FixtureIntegrityError("contract_smoke must be a non-empty list")
    seen_ids: set[str] = set()
    for row_index, row_value in enumerate(value):
        row = _require_exact_keys(
            row_value, _CASE_KEYS, f"contract smoke case {row_index}"
        )
        case_id = _require_string(row["case_id"], "contract smoke case_id")
        if case_id in seen_ids:
            raise FixtureIntegrityError("contract smoke case IDs must be unique")
        seen_ids.add(case_id)
        if row["partition"] != "contract_smoke":
            raise FixtureIntegrityError(
                f"contract smoke case has an invalid partition: {case_id}"
            )
        _require_string(row["purpose"], f"contract smoke purpose {case_id}")
        _validate_source_schema(row["source"], case_id=case_id)
        _validate_gold_schema(row["gold"], case_id=case_id)

    if _canonical_sha256(value) != CONTRACT_SMOKE_SHA256:
        raise FixtureIntegrityError("contract smoke digest mismatch")
    return value


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    """Load and validate fixture policy without requiring repository datasets."""

    manifest = _load_json(Path(path), description="fixture manifest")
    root = _require_exact_keys(manifest, _MANIFEST_KEYS, "fixture manifest")
    if root["schema_version"] != FIXTURE_SCHEMA_VERSION:
        raise FixtureIntegrityError("unsupported fixture schema version")
    if root["contract_version"] != CONTRACT_VERSION:
        raise FixtureIntegrityError("fixture contract version does not match Nano")
    if root["purpose"] != _PURPOSE:
        raise FixtureIntegrityError("fixture purpose does not match frozen policy")
    scientific_status = root["scientific_status"]
    if (
        not isinstance(scientific_status, dict)
        or set(scientific_status) != set(_SCIENTIFIC_STATUS)
        or scientific_status != _SCIENTIFIC_STATUS
        or type(scientific_status["historical_eval_is_fresh"]) is not bool
    ):
        raise FixtureIntegrityError("scientific status does not match frozen policy")
    _validate_partition_declarations(root["partitions"])
    _validate_contract_smoke(root["contract_smoke"])
    return root


def _strip_historical_instruction(content: str) -> str:
    suffix = "\nSummarize the visit."
    if not content.endswith(suffix):
        raise FixtureIntegrityError(
            "historical fixture lacks the frozen instruction suffix"
        )
    return content[: -len(suffix)]


def _historical_transcript(record: Any) -> str:
    try:
        user_message = record["convo"][0]
        if user_message["role"] != "user":
            raise FixtureIntegrityError("fixture transcript is not the user message")
        content = user_message["content"]
        if not isinstance(content, str):
            raise FixtureIntegrityError("historical fixture content is not text")
        return _strip_historical_instruction(content)
    except FixtureIntegrityError:
        raise
    except (KeyError, IndexError, TypeError) as exc:
        raise FixtureIntegrityError(
            "historical fixture record has an invalid shape"
        ) from exc


def _repo_path(relative: str, repository_root: Path) -> Path:
    if not isinstance(relative, str) or not relative:
        raise FixtureIntegrityError("fixture source path must be a non-empty string")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise FixtureIntegrityError(f"fixture source escapes repository: {relative}")
    try:
        root = repository_root.resolve(strict=True)
    except OSError as exc:
        raise FixtureIntegrityError(
            f"repository root is unavailable: {repository_root}"
        ) from exc
    if not root.is_dir():
        raise FixtureIntegrityError(f"repository root is not a directory: {root}")
    cursor = root
    for part in relative_path.parts:
        cursor /= part
        if cursor.is_symlink():
            raise FixtureIntegrityError(f"fixture source uses a symlink: {relative}")
    try:
        candidate = (root / relative_path).resolve(strict=True)
    except OSError as exc:
        raise FixtureIntegrityError(
            f"fixture source is unavailable: {relative}"
        ) from exc
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise FixtureIntegrityError(
            f"fixture source escapes repository: {relative}"
        ) from exc
    if not candidate.is_file():
        raise FixtureIntegrityError(
            f"fixture source is not a regular in-repo file: {relative}"
        )
    return candidate


def verify_repository_partitions(
    path: Path = DEFAULT_MANIFEST_PATH,
    *,
    repository_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Verify optional R-star provenance from one byte snapshot per partition.

    This check is deliberately separate from :func:`load_contract_smoke` so an
    installed Nano package does not need the research repository checkout.
    """

    manifest = load_manifest(path)
    root = _REPO_ROOT if repository_root is None else Path(repository_root)
    records_by_partition: dict[str, list[Any]] = {}
    verified: dict[str, dict[str, Any]] = {}

    for name, spec in manifest["partitions"].items():
        source = _repo_path(spec["path"], root)
        snapshot = _read_bytes_snapshot(
            source, description=f"repository partition {name}"
        )
        actual_sha256 = hashlib.sha256(snapshot).hexdigest()
        if actual_sha256 != spec["sha256"]:
            raise FixtureIntegrityError(f"partition hash mismatch: {name}")
        records = _parse_json_snapshot(snapshot, description=f"partition {name}")
        if not isinstance(records, list) or len(records) != spec["records"]:
            raise FixtureIntegrityError(f"partition record-count mismatch: {name}")
        records_by_partition[name] = records
        verified[name] = {
            "path": spec["path"],
            "records": len(records),
            "sha256": actual_sha256,
        }

    for row in manifest["contract_smoke"]:
        source_spec = row["source"]
        if source_spec["type"] != "inline_snapshot":
            continue
        historical = source_spec["historical_source"]
        partition_name = historical["partition"]
        records = records_by_partition[partition_name]
        index = historical["index"]
        if index >= len(records):
            raise FixtureIntegrityError(
                f"historical record index is out of bounds: {row['case_id']}"
            )
        record = records[index]
        if _canonical_record_sha256(record) != historical["record_sha256"]:
            raise FixtureIntegrityError(
                f"historical record hash mismatch: {row['case_id']}"
            )
        if _historical_transcript(record) != source_spec["transcript"]:
            raise FixtureIntegrityError(
                f"historical transcript snapshot mismatch: {row['case_id']}"
            )

    return verified


def _transcript_from_source(source_spec: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    transcript = source_spec["transcript"]
    actual_sha256 = hashlib.sha256(transcript.encode("utf-8")).hexdigest()
    if actual_sha256 != source_spec["transcript_sha256"]:
        raise FixtureIntegrityError("inline transcript hash mismatch")
    provenance: dict[str, Any] = {
        "type": source_spec["type"],
        "transcript_sha256": actual_sha256,
    }
    if source_spec["type"] == "inline_snapshot":
        provenance["historical_source"] = dict(source_spec["historical_source"])
    return transcript, provenance


def _gold_output(case: dict[str, Any], request: NanoInput) -> NanoOutput:
    try:
        fields = tuple(
            FieldOutput.from_dict(annotation, path=f"$.gold[{index}]")
            for index, annotation in enumerate(case["gold"])
        )
        output = NanoOutput(
            item_id=request.item_id,
            solver_id=GOLD_SOLVER_ID,
            fields=fields,
        )
        output.validate_against(request)
    except (KeyError, TypeError, ValueError) as exc:
        raise FixtureIntegrityError(
            f"invalid gold annotation for {case.get('case_id')}"
        ) from exc
    return output


def load_contract_smoke(path: Path = DEFAULT_MANIFEST_PATH) -> tuple[FixtureCase, ...]:
    """Load the self-contained conformance suite without exposing truth to solvers."""

    manifest = load_manifest(path)
    cases: list[FixtureCase] = []
    for row in manifest["contract_smoke"]:
        case_id = row["case_id"]
        transcript, provenance = _transcript_from_source(row["source"])
        request = NanoInput(item_id=case_id, transcript=transcript)
        cases.append(
            FixtureCase(
                case_id=case_id,
                partition=row["partition"],
                request=request,
                gold=_gold_output(row, request),
                provenance={**provenance, "purpose": row["purpose"]},
            )
        )
    return tuple(cases)


__all__ = [
    "CONTRACT_SMOKE_SHA256",
    "DEFAULT_MANIFEST_PATH",
    "FIXTURE_SCHEMA_VERSION",
    "FixtureCase",
    "FixtureIntegrityError",
    "load_contract_smoke",
    "load_manifest",
    "verify_repository_partitions",
]
