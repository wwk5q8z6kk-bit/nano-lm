"""Versioned input and output contract for the Nano scribe AI.

The contract is intentionally independent of model frameworks, serving layers,
and research harnesses.  It represents only what Nano consumes and produces.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

CONTRACT_VERSION = "nano.scribe.v0"


class ContractValidationError(ValueError):
    """A machine-classifiable violation of the Nano contract."""

    def __init__(self, code: str, message: str, *, path: str = "$") -> None:
        self.code = code
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message} [{code}]")


class FieldName(str, Enum):
    CHIEF_COMPLAINT = "chief_complaint"
    DURATION = "duration"
    SEVERITY = "severity"
    MEDICATION = "medication"
    ALLERGY = "allergy"


FIELD_ORDER: tuple[FieldName, ...] = (
    FieldName.CHIEF_COMPLAINT,
    FieldName.DURATION,
    FieldName.SEVERITY,
    FieldName.MEDICATION,
    FieldName.ALLERGY,
)


class FieldState(str, Enum):
    SUPPORTED = "supported"
    ABSENT = "absent"
    MISSING = "missing"
    UNCERTAIN = "uncertain"
    CONFLICTING = "conflicting"


_DENIAL_PATTERNS = {
    FieldName.MEDICATION: re.compile(
        r"\s*(?:"
        r"no,?\s+nothing(?:\s+yet)?[.!]?|"
        r"nothing\s+at\s+all[.!]?|"
        r"i\s+haven['\N{RIGHT SINGLE QUOTATION MARK}]?t\s+taken\s+anything[.!]?|"
        r"no\s+(?:medication(?:s)?|medicine|meds?)[.!]?|"
        r"i\s+(?:deny|denied)\s+(?:taking\s+)?(?:medication(?:s)?|medicine|meds?)[.!]?"
        r")\s*",
        flags=re.IGNORECASE,
    ),
    FieldName.ALLERGY: re.compile(
        r"\s*(?:"
        r"no\s+(?:known\s+)?allerg(?:y|ies)[.!]?|"
        r"not\s+that\s+i\s+know\s+of[.!]?|"
        r"none\s+whatsoever[.!]?|"
        r"i\s+(?:deny|denied)\s+(?:any\s+)?allerg(?:y|ies)[.!]?"
        r")\s*",
        flags=re.IGNORECASE,
    ),
}
_PATIENT_LINE_PATTERN = re.compile(r"^\s*patient\s*:\s*", flags=re.IGNORECASE)
_SURROUNDING_PUNCTUATION = (
    " \t\r\n.,;:!?\"'`\N{LEFT SINGLE QUOTATION MARK}\N{RIGHT SINGLE QUOTATION MARK}"
)


def normalize_value(value: str) -> str:
    """Normalize superficial textual variation without changing semantics."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = " ".join(normalized.split()).strip(_SURROUNDING_PUNCTUATION)
    for article in ("a ", "an ", "the "):
        if normalized.startswith(article):
            return normalized[len(article) :].strip(_SURROUNDING_PUNCTUATION)
    return normalized


def _is_field_denial(field: FieldName, text: str) -> bool:
    """Recognize only v0 denials whose wording matches the asserted field."""

    pattern = _DENIAL_PATTERNS.get(field)
    return pattern is not None and pattern.fullmatch(text) is not None


def _fail(code: str, message: str, path: str) -> None:
    raise ContractValidationError(code, message, path=path)


def _require_mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("type_error", "expected an object", path)
    if any(not isinstance(key, str) for key in value):
        _fail("key_type", "object keys must be strings", path)
    return value


def _require_exact_keys(
    value: object,
    expected: frozenset[str],
    path: str,
) -> Mapping[str, Any]:
    mapping = _require_mapping(value, path)
    actual = frozenset(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details: list[str] = []
        if missing:
            details.append(f"missing keys: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected keys: {', '.join(extra)}")
        _fail("invalid_keys", "; ".join(details), path)
    return mapping


def _require_nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        _fail("invalid_string", "expected a non-empty, edge-trimmed string", path)
    return value


def _require_optional_nonempty_string(value: object, path: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty_string(value, path)


def _require_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("type_error", "expected an integer", path)
    return value


def _parse_enum(enum_type: type[Enum], value: object, path: str) -> Any:
    if not isinstance(value, str):
        _fail("type_error", "expected a string", path)
    try:
        return enum_type(value)
    except ValueError:
        allowed = ", ".join(member.value for member in enum_type)
        _fail("invalid_enum", f"expected one of: {allowed}", path)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate_key", f"duplicate JSON key: {key}", "$")
        result[key] = value
    return result


def _load_json(raw: str) -> Any:
    if not isinstance(raw, str):
        _fail("type_error", "expected JSON text", "$")
    try:
        return json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except ContractValidationError:
        raise
    except (json.JSONDecodeError, TypeError) as exc:
        _fail("invalid_json", str(exc), "$")


def _patient_content_ranges(transcript: str) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    offset = 0
    for line in transcript.splitlines(keepends=True):
        visible_line = line.rstrip("\r\n")
        match = _PATIENT_LINE_PATTERN.match(visible_line)
        if match is not None and match.end() < len(visible_line):
            ranges.append((offset + match.end(), offset + len(visible_line)))
        offset += len(line)
    return tuple(ranges)


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    """An exact, patient-authored character span in the supplied transcript."""

    start: int
    end: int
    text: str
    speaker: str = "patient"

    def __post_init__(self) -> None:
        _require_int(self.start, "$.start")
        _require_int(self.end, "$.end")
        if self.start < 0 or self.end <= self.start:
            _fail("invalid_span", "span must satisfy 0 <= start < end", "$")
        _require_nonempty_string(self.text, "$.text")
        if self.speaker != "patient":
            _fail("invalid_speaker", 'speaker must be exactly "patient"', "$.speaker")

    def validate_against(self, transcript: str, *, path: str = "$.evidence") -> None:
        if not isinstance(transcript, str):
            _fail("type_error", "transcript must be a string", "$.transcript")
        if self.end > len(transcript):
            _fail("evidence_bounds", "evidence extends beyond the transcript", path)
        if transcript[self.start : self.end] != self.text:
            _fail(
                "evidence_text_mismatch",
                "evidence text does not match its offsets",
                path,
            )
        if not any(
            content_start <= self.start and self.end <= content_end
            for content_start, content_end in _patient_content_ranges(transcript)
        ):
            _fail("evidence_not_patient", "evidence is not inside a Patient turn", path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "speaker": self.speaker,
        }

    @classmethod
    def from_dict(cls, data: object, *, path: str = "$") -> EvidenceSpan:
        mapping = _require_exact_keys(
            data,
            frozenset({"start", "end", "text", "speaker"}),
            path,
        )
        return cls(
            start=_require_int(mapping["start"], f"{path}.start"),
            end=_require_int(mapping["end"], f"{path}.end"),
            text=_require_nonempty_string(mapping["text"], f"{path}.text"),
            speaker=_require_nonempty_string(mapping["speaker"], f"{path}.speaker"),
        )


@dataclass(frozen=True, slots=True)
class FieldOutput:
    """One canonical scribe field plus its support/abstention state."""

    field: FieldName
    state: FieldState
    value: str | None = None
    evidence: tuple[EvidenceSpan, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.field, FieldName):
            _fail("type_error", "field must be a FieldName", "$.field")
        if not isinstance(self.state, FieldState):
            _fail("type_error", "state must be a FieldState", "$.state")
        object.__setattr__(self, "evidence", tuple(self.evidence))
        if any(not isinstance(span, EvidenceSpan) for span in self.evidence):
            _fail(
                "type_error", "evidence must contain EvidenceSpan values", "$.evidence"
            )

        if self.state is FieldState.SUPPORTED:
            value = _require_optional_nonempty_string(self.value, "$.value")
            if value is None:
                _fail(
                    "supported_without_value",
                    "supported fields require a value",
                    "$.value",
                )
            if not self.evidence:
                _fail(
                    "supported_without_evidence",
                    "supported fields require evidence",
                    "$.evidence",
                )
            normalized_value = normalize_value(value)
            if not normalized_value or not any(
                normalize_value(span.text) == normalized_value for span in self.evidence
            ):
                _fail(
                    "unsupported_value",
                    "at least one evidence span must normalize exactly to the value",
                    "$.evidence",
                )
            return

        if self.value is not None:
            _fail(
                "value_on_non_supported_state",
                "only supported fields may carry a value",
                "$.value",
            )

        if self.state is FieldState.ABSENT:
            if not self.evidence:
                _fail(
                    "absence_without_evidence",
                    "absent fields require denial evidence",
                    "$.evidence",
                )
            if not any(
                _is_field_denial(self.field, span.text) for span in self.evidence
            ):
                _fail(
                    "absence_without_denial",
                    "absent fields require an explicit denial for that field",
                    "$.evidence",
                )
        elif self.state is FieldState.MISSING and self.evidence:
            _fail(
                "evidence_on_missing",
                "missing fields cannot carry evidence",
                "$.evidence",
            )
        elif self.state is FieldState.CONFLICTING:
            distinct = {(span.start, span.end) for span in self.evidence}
            if len(distinct) < 2:
                _fail(
                    "insufficient_conflict_evidence",
                    "conflicting fields require two distinct evidence spans",
                    "$.evidence",
                )
            distinct_text = {normalize_value(span.text) for span in self.evidence}
            if len(distinct_text) < 2:
                _fail(
                    "duplicate_conflict_evidence",
                    "conflicting fields require two textually distinct observations",
                    "$.evidence",
                )

    @property
    def abstained(self) -> bool:
        return self.state in {
            FieldState.MISSING,
            FieldState.UNCERTAIN,
            FieldState.CONFLICTING,
        }

    @property
    def presented(self) -> bool:
        """Whether the field has a grounded positive or negative answer."""

        return not self.abstained

    def validate_against(self, transcript: str, *, path: str = "$") -> None:
        for index, span in enumerate(self.evidence):
            span.validate_against(transcript, path=f"{path}.evidence[{index}]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field.value,
            "state": self.state.value,
            "value": self.value,
            "evidence": [span.to_dict() for span in self.evidence],
        }

    @classmethod
    def from_dict(cls, data: object, *, path: str = "$") -> FieldOutput:
        mapping = _require_exact_keys(
            data,
            frozenset({"field", "state", "value", "evidence"}),
            path,
        )
        raw_evidence = mapping["evidence"]
        if not isinstance(raw_evidence, list):
            _fail("type_error", "evidence must be an array", f"{path}.evidence")
        value = mapping["value"]
        if value is not None and not isinstance(value, str):
            _fail("type_error", "value must be a string or null", f"{path}.value")
        return cls(
            field=_parse_enum(FieldName, mapping["field"], f"{path}.field"),
            state=_parse_enum(FieldState, mapping["state"], f"{path}.state"),
            value=value,
            evidence=tuple(
                EvidenceSpan.from_dict(span, path=f"{path}.evidence[{index}]")
                for index, span in enumerate(raw_evidence)
            ),
        )


@dataclass(frozen=True, slots=True)
class NanoInput:
    """The complete information available to a Nano solver."""

    item_id: str
    transcript: str
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_nonempty_string(self.item_id, "$.item_id")
        if not isinstance(self.transcript, str) or not self.transcript.strip():
            _fail("invalid_transcript", "transcript must contain text", "$.transcript")
        if self.schema_version != CONTRACT_VERSION:
            _fail(
                "schema_version",
                f"schema_version must be {CONTRACT_VERSION}",
                "$.schema_version",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "item_id": self.item_id,
            "transcript": self.transcript,
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> NanoInput:
        mapping = _require_exact_keys(
            data,
            frozenset({"schema_version", "item_id", "transcript"}),
            "$",
        )
        transcript = mapping["transcript"]
        if not isinstance(transcript, str):
            _fail("type_error", "transcript must be a string", "$.transcript")
        return cls(
            item_id=_require_nonempty_string(mapping["item_id"], "$.item_id"),
            transcript=transcript,
            schema_version=_require_nonempty_string(
                mapping["schema_version"],
                "$.schema_version",
            ),
        )

    @classmethod
    def from_json(cls, raw: str) -> NanoInput:
        return cls.from_dict(_load_json(raw))


@dataclass(frozen=True, slots=True)
class NanoOutput:
    """A complete, ordered five-field Nano inference result."""

    item_id: str
    solver_id: str
    fields: tuple[FieldOutput, ...]
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_nonempty_string(self.item_id, "$.item_id")
        _require_nonempty_string(self.solver_id, "$.solver_id")
        if self.schema_version != CONTRACT_VERSION:
            _fail(
                "schema_version",
                f"schema_version must be {CONTRACT_VERSION}",
                "$.schema_version",
            )
        object.__setattr__(self, "fields", tuple(self.fields))
        if any(not isinstance(field, FieldOutput) for field in self.fields):
            _fail("type_error", "fields must contain FieldOutput values", "$.fields")
        actual_order = tuple(field.field for field in self.fields)
        if actual_order != FIELD_ORDER:
            expected = ", ".join(field.value for field in FIELD_ORDER)
            _fail(
                "field_order",
                f"fields must contain exactly these fields in order: {expected}",
                "$.fields",
            )

    def validate_against(self, request: NanoInput) -> None:
        if not isinstance(request, NanoInput):
            _fail("type_error", "request must be a NanoInput", "$")
        if self.item_id != request.item_id:
            _fail(
                "item_id_mismatch", "output item_id does not match input", "$.item_id"
            )
        if self.schema_version != request.schema_version:
            _fail(
                "schema_version_mismatch",
                "output schema_version does not match input",
                "$.schema_version",
            )
        for index, field in enumerate(self.fields):
            field.validate_against(request.transcript, path=f"$.fields[{index}]")

    def field(self, name: FieldName) -> FieldOutput:
        if not isinstance(name, FieldName):
            _fail("type_error", "name must be a FieldName", "$.field")
        return self.fields[FIELD_ORDER.index(name)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "item_id": self.item_id,
            "solver_id": self.solver_id,
            "fields": [field.to_dict() for field in self.fields],
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> NanoOutput:
        mapping = _require_exact_keys(
            data,
            frozenset({"schema_version", "item_id", "solver_id", "fields"}),
            "$",
        )
        raw_fields = mapping["fields"]
        if not isinstance(raw_fields, list):
            _fail("type_error", "fields must be an array", "$.fields")
        return cls(
            item_id=_require_nonempty_string(mapping["item_id"], "$.item_id"),
            solver_id=_require_nonempty_string(mapping["solver_id"], "$.solver_id"),
            fields=tuple(
                FieldOutput.from_dict(field, path=f"$.fields[{index}]")
                for index, field in enumerate(raw_fields)
            ),
            schema_version=_require_nonempty_string(
                mapping["schema_version"],
                "$.schema_version",
            ),
        )

    @classmethod
    def from_json(cls, raw: str) -> NanoOutput:
        return cls.from_dict(_load_json(raw))
