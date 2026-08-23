"""Encounter Representation v0 — evidence-grounded truth object for P1 scribing.

The note is a later view. This module is software: exact offsets, typed states,
canonical JSON, and fail-closed invariants. It does not extract facts.
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

SCHEMA_VERSION = "nano.encounter.v0"
STRIP_ARTICLES_NFKC = "strip_articles_nfkc"
_SURROUNDING = " \t\r\n.,;:!?\"'`‘’"


class EncounterError(ValueError):
    """A machine-classifiable Encounter Representation violation."""

    def __init__(self, code: str, message: str, *, path: str = "$") -> None:
        self.code = code
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message} [{code}]")


class Speaker(str, Enum):
    PATIENT = "patient"
    CLINICIAN = "clinician"
    OTHER = "other"
    UNKNOWN = "unknown"


AUTHORITATIVE_SPEAKERS = frozenset({Speaker.PATIENT, Speaker.CLINICIAN})


class AssertionState(str, Enum):
    ASSERTED = "asserted"
    DENIED = "denied"
    UNCERTAIN = "uncertain"
    CONFLICTING = "conflicting"


class Temporality(str, Enum):
    CURRENT = "current"
    HISTORICAL = "historical"
    FUTURE = "future"
    HYPOTHETICAL = "hypothetical"
    UNKNOWN = "unknown"


class Experiencer(str, Enum):
    PATIENT = "patient"
    OTHER = "other"
    UNKNOWN = "unknown"


class Certainty(str, Enum):
    STATED = "stated"
    UNCERTAIN = "uncertain"
    UNKNOWN = "unknown"


class AtomType(str, Enum):
    SYMPTOM = "symptom"
    MEDICATION = "medication"
    ALLERGY = "allergy"
    HISTORY = "history"
    MEASUREMENT = "measurement"
    DIAGNOSIS_STATEMENT = "diagnosis_statement"
    ASSESSMENT = "assessment"
    PLAN = "plan"
    PROCEDURE = "procedure"
    INSTRUCTION = "instruction"


def _fail(code: str, message: str, path: str) -> None:
    raise EncounterError(code, message, path=path)


def _require_mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("type_error", "expected an object", path)
    if any(not isinstance(key, str) for key in value):
        _fail("key_type", "object keys must be strings", path)
    return value


def _require_exact_keys(value: object, expected: frozenset[str], path: str) -> Mapping[str, Any]:
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


def _optional_string(value: object, path: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty_string(value, path)


def _require_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("type_error", "expected an integer", path)
    return value


def _require_bool(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        _fail("type_error", "expected a boolean", path)
    return value


def _parse_enum(enum_type: type[Enum], value: object, path: str) -> Any:
    if not isinstance(value, str):
        _fail("type_error", "expected a string", path)
    try:
        return enum_type(value)
    except ValueError:
        allowed = ", ".join(member.value for member in enum_type)
        _fail("invalid_enum", f"expected one of: {allowed}", path)


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
    except EncounterError:
        raise
    except (json.JSONDecodeError, TypeError) as exc:
        _fail("invalid_json", str(exc), "$")


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_value(value: str) -> str:
    """Named superficial normalization. Does not change clinical meaning."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = " ".join(normalized.split()).strip(_SURROUNDING)
    for article in ("a ", "an ", "the "):
        if normalized.startswith(article):
            return normalized[len(article) :].strip(_SURROUNDING)
    return normalized


def apply_normalization(name: str, value: str) -> str:
    if name != STRIP_ARTICLES_NFKC:
        _fail("unknown_transform", f"unknown normalization transform: {name}", "$.normalization_transform")
    return normalize_value(value)


def _value_grounded_in_span(raw_value: str, span_text: str) -> bool:
    """Literal containment after only superficial normalization. Not semantic inference."""
    if raw_value in span_text:
        return True
    grounded = normalize_value(raw_value)
    return bool(grounded) and grounded in normalize_value(span_text)


def _require_id_tuple(value: object, path: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        _fail("type_error", "expected a sequence of identifiers", path)
    ids = tuple(value)
    if any(not isinstance(item, str) or not item or item.strip() != item for item in ids):
        _fail("invalid_string", "identifiers must be non-empty, edge-trimmed strings", path)
    return ids


def _require_typed_tuple(value: object, expected: type, path: str) -> tuple[Any, ...]:
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError:
        _fail("type_error", "expected a sequence", path)
    if any(not isinstance(item, expected) for item in items):
        _fail("type_error", f"expected {expected.__name__} values", path)
    return items


@dataclass(frozen=True, slots=True)
class TemporalState:
    kind: Temporality
    onset_raw: str | None = None
    duration_raw: str | None = None
    time_expression_raw: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, Temporality):
            _fail("type_error", "kind must be a Temporality", "$.kind")
        if self.onset_raw is not None:
            _require_nonempty_string(self.onset_raw, "$.onset_raw")
        if self.duration_raw is not None:
            _require_nonempty_string(self.duration_raw, "$.duration_raw")
        if self.time_expression_raw is not None:
            _require_nonempty_string(self.time_expression_raw, "$.time_expression_raw")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "onset_raw": self.onset_raw,
            "duration_raw": self.duration_raw,
            "time_expression_raw": self.time_expression_raw,
        }

    @classmethod
    def from_dict(cls, data: object, *, path: str = "$") -> TemporalState:
        mapping = _require_exact_keys(
            data,
            frozenset({"kind", "onset_raw", "duration_raw", "time_expression_raw"}),
            path,
        )
        return cls(
            kind=_parse_enum(Temporality, mapping["kind"], f"{path}.kind"),
            onset_raw=_optional_string(mapping["onset_raw"], f"{path}.onset_raw"),
            duration_raw=_optional_string(mapping["duration_raw"], f"{path}.duration_raw"),
            time_expression_raw=_optional_string(
                mapping["time_expression_raw"], f"{path}.time_expression_raw"
            ),
        )


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    evidence_id: str
    source_id: str
    turn_id: str
    speaker: Speaker
    start: int
    end: int
    text: str

    def __post_init__(self) -> None:
        _require_nonempty_string(self.evidence_id, "$.evidence_id")
        _require_nonempty_string(self.source_id, "$.source_id")
        _require_nonempty_string(self.turn_id, "$.turn_id")
        if not isinstance(self.speaker, Speaker):
            _fail("type_error", "speaker must be a Speaker", "$.speaker")
        _require_int(self.start, "$.start")
        _require_int(self.end, "$.end")
        if self.start < 0 or self.end <= self.start:
            _fail("invalid_span", "span must satisfy 0 <= start < end", "$")
        _require_nonempty_string(self.text, "$.text")

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source_id": self.source_id,
            "turn_id": self.turn_id,
            "speaker": self.speaker.value,
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: object, *, path: str = "$") -> EvidenceSpan:
        mapping = _require_exact_keys(
            data,
            frozenset({"evidence_id", "source_id", "turn_id", "speaker", "start", "end", "text"}),
            path,
        )
        return cls(
            evidence_id=_require_nonempty_string(mapping["evidence_id"], f"{path}.evidence_id"),
            source_id=_require_nonempty_string(mapping["source_id"], f"{path}.source_id"),
            turn_id=_require_nonempty_string(mapping["turn_id"], f"{path}.turn_id"),
            speaker=_parse_enum(Speaker, mapping["speaker"], f"{path}.speaker"),
            start=_require_int(mapping["start"], f"{path}.start"),
            end=_require_int(mapping["end"], f"{path}.end"),
            text=_require_nonempty_string(mapping["text"], f"{path}.text"),
        )


@dataclass(frozen=True, slots=True)
class Turn:
    turn_id: str
    source_id: str
    speaker: Speaker
    start: int
    end: int
    text: str

    def __post_init__(self) -> None:
        _require_nonempty_string(self.turn_id, "$.turn_id")
        _require_nonempty_string(self.source_id, "$.source_id")
        if not isinstance(self.speaker, Speaker):
            _fail("type_error", "speaker must be a Speaker", "$.speaker")
        _require_int(self.start, "$.start")
        _require_int(self.end, "$.end")
        if self.start < 0 or self.end <= self.start:
            _fail("invalid_span", "turn must satisfy 0 <= start < end", "$")
        _require_nonempty_string(self.text, "$.text")

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "source_id": self.source_id,
            "speaker": self.speaker.value,
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: object, *, path: str = "$") -> Turn:
        mapping = _require_exact_keys(
            data,
            frozenset({"turn_id", "source_id", "speaker", "start", "end", "text"}),
            path,
        )
        return cls(
            turn_id=_require_nonempty_string(mapping["turn_id"], f"{path}.turn_id"),
            source_id=_require_nonempty_string(mapping["source_id"], f"{path}.source_id"),
            speaker=_parse_enum(Speaker, mapping["speaker"], f"{path}.speaker"),
            start=_require_int(mapping["start"], f"{path}.start"),
            end=_require_int(mapping["end"], f"{path}.end"),
            text=_require_nonempty_string(mapping["text"], f"{path}.text"),
        )


@dataclass(frozen=True, slots=True)
class Source:
    source_id: str
    text: str
    turns: tuple[Turn, ...]

    def __post_init__(self) -> None:
        _require_nonempty_string(self.source_id, "$.source_id")
        if not isinstance(self.text, str) or not self.text:
            _fail("invalid_string", "source text must be a non-empty string", "$.text")
        try:
            object.__setattr__(self, "turns", tuple(self.turns))
        except TypeError:
            _fail("type_error", "turns must be a sequence of Turn values", "$.turns")
        if any(not isinstance(turn, Turn) for turn in self.turns):
            _fail("type_error", "turns must contain Turn values", "$.turns")

    def turn(self, turn_id: str) -> Turn:
        for turn in self.turns:
            if turn.turn_id == turn_id:
                return turn
        _fail("unknown_turn", f"unknown turn_id: {turn_id}", "$.turns")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "text": self.text,
            "turns": [turn.to_dict() for turn in self.turns],
        }

    @classmethod
    def from_dict(cls, data: object, *, path: str = "$") -> Source:
        mapping = _require_exact_keys(data, frozenset({"source_id", "text", "turns"}), path)
        raw_turns = mapping["turns"]
        if not isinstance(raw_turns, list):
            _fail("type_error", "turns must be an array", f"{path}.turns")
        text = mapping["text"]
        if not isinstance(text, str):
            _fail("type_error", "text must be a string", f"{path}.text")
        return cls(
            source_id=_require_nonempty_string(mapping["source_id"], f"{path}.source_id"),
            text=text,
            turns=tuple(
                Turn.from_dict(turn, path=f"{path}.turns[{index}]")
                for index, turn in enumerate(raw_turns)
            ),
        )


def assemble_source(
    source_id: str,
    turns: Sequence[tuple[Speaker, str]],
    *,
    sep: str = "\n",
) -> Source:
    """SOFTWARE helper: assign exact offsets for labeled turns. Not an extractor."""
    parts: list[str] = []
    built: list[Turn] = []
    offset = 0
    for index, (speaker, text) in enumerate(turns):
        if index:
            offset += len(sep)
        start = offset
        end = start + len(text)
        built.append(
            Turn(
                turn_id=f"{source_id}:t{index}",
                source_id=source_id,
                speaker=speaker,
                start=start,
                end=end,
                text=text,
            )
        )
        parts.append(text)
        offset = end
    return Source(source_id=source_id, text=sep.join(parts), turns=tuple(built))


@dataclass(frozen=True, slots=True)
class ClinicalAtom:
    atom_id: str
    atom_type: AtomType
    raw_value: str
    assertion_state: AssertionState
    speaker: Speaker
    experiencer: Experiencer
    temporality: TemporalState
    certainty: Certainty
    evidence_ids: tuple[str, ...]
    normalized_value: str | None = None
    normalization_transform: str | None = None
    review_required: bool = False

    def __post_init__(self) -> None:
        _require_nonempty_string(self.atom_id, "$.atom_id")
        if not isinstance(self.atom_type, AtomType):
            _fail("type_error", "atom_type must be an AtomType", "$.atom_type")
        _require_nonempty_string(self.raw_value, "$.raw_value")
        if not isinstance(self.assertion_state, AssertionState):
            _fail("type_error", "assertion_state must be an AssertionState", "$.assertion_state")
        if not isinstance(self.speaker, Speaker):
            _fail("type_error", "speaker must be a Speaker", "$.speaker")
        if not isinstance(self.experiencer, Experiencer):
            _fail("type_error", "experiencer must be an Experiencer", "$.experiencer")
        if not isinstance(self.temporality, TemporalState):
            _fail("type_error", "temporality must be a TemporalState", "$.temporality")
        if not isinstance(self.certainty, Certainty):
            _fail("type_error", "certainty must be a Certainty", "$.certainty")
        object.__setattr__(self, "evidence_ids", _require_id_tuple(self.evidence_ids, "$.evidence_ids"))
        _require_bool(self.review_required, "$.review_required")
        if (
            self.assertion_state is AssertionState.UNCERTAIN
            and self.certainty is not Certainty.UNCERTAIN
        ):
            _fail(
                "uncertain_certainty_mismatch",
                "UNCERTAIN atoms require Certainty.UNCERTAIN",
                "$.certainty",
            )
        if self.normalized_value is not None and self.normalization_transform is None:
            _fail(
                "normalized_without_transform",
                "normalized_value requires an explicit named transform",
                "$.normalization_transform",
            )
        if self.normalization_transform is not None:
            expected = apply_normalization(self.normalization_transform, self.raw_value)
            if self.normalized_value != expected:
                _fail(
                    "normalization_mismatch",
                    "normalized_value must equal the named transform of raw_value",
                    "$.normalized_value",
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "atom_id": self.atom_id,
            "atom_type": self.atom_type.value,
            "raw_value": self.raw_value,
            "normalized_value": self.normalized_value,
            "assertion_state": self.assertion_state.value,
            "speaker": self.speaker.value,
            "experiencer": self.experiencer.value,
            "temporality": self.temporality.to_dict(),
            "certainty": self.certainty.value,
            "evidence_ids": list(self.evidence_ids),
            "normalization_transform": self.normalization_transform,
            "review_required": self.review_required,
        }

    @classmethod
    def from_dict(cls, data: object, *, path: str = "$") -> ClinicalAtom:
        mapping = _require_exact_keys(
            data,
            frozenset(
                {
                    "atom_id",
                    "atom_type",
                    "raw_value",
                    "normalized_value",
                    "assertion_state",
                    "speaker",
                    "experiencer",
                    "temporality",
                    "certainty",
                    "evidence_ids",
                    "normalization_transform",
                    "review_required",
                }
            ),
            path,
        )
        raw_ids = mapping["evidence_ids"]
        if not isinstance(raw_ids, list) or any(not isinstance(item, str) for item in raw_ids):
            _fail("type_error", "evidence_ids must be an array of strings", f"{path}.evidence_ids")
        return cls(
            atom_id=_require_nonempty_string(mapping["atom_id"], f"{path}.atom_id"),
            atom_type=_parse_enum(AtomType, mapping["atom_type"], f"{path}.atom_type"),
            raw_value=_require_nonempty_string(mapping["raw_value"], f"{path}.raw_value"),
            assertion_state=_parse_enum(
                AssertionState, mapping["assertion_state"], f"{path}.assertion_state"
            ),
            speaker=_parse_enum(Speaker, mapping["speaker"], f"{path}.speaker"),
            experiencer=_parse_enum(Experiencer, mapping["experiencer"], f"{path}.experiencer"),
            temporality=TemporalState.from_dict(mapping["temporality"], path=f"{path}.temporality"),
            certainty=_parse_enum(Certainty, mapping["certainty"], f"{path}.certainty"),
            evidence_ids=tuple(raw_ids),
            normalized_value=_optional_string(mapping["normalized_value"], f"{path}.normalized_value"),
            normalization_transform=_optional_string(
                mapping["normalization_transform"], f"{path}.normalization_transform"
            ),
            review_required=_require_bool(mapping["review_required"], f"{path}.review_required"),
        )


@dataclass(frozen=True, slots=True)
class Conflict:
    conflict_id: str
    atom_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    note: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty_string(self.conflict_id, "$.conflict_id")
        object.__setattr__(self, "atom_ids", _require_id_tuple(self.atom_ids, "$.atom_ids"))
        object.__setattr__(self, "evidence_ids", _require_id_tuple(self.evidence_ids, "$.evidence_ids"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "atom_ids": list(self.atom_ids),
            "evidence_ids": list(self.evidence_ids),
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: object, *, path: str = "$") -> Conflict:
        mapping = _require_exact_keys(
            data, frozenset({"conflict_id", "atom_ids", "evidence_ids", "note"}), path
        )
        atom_ids = mapping["atom_ids"]
        evidence_ids = mapping["evidence_ids"]
        if not isinstance(atom_ids, list) or not isinstance(evidence_ids, list):
            _fail("type_error", "atom_ids and evidence_ids must be arrays", path)
        return cls(
            conflict_id=_require_nonempty_string(mapping["conflict_id"], f"{path}.conflict_id"),
            atom_ids=tuple(atom_ids),
            evidence_ids=tuple(evidence_ids),
            note=_optional_string(mapping["note"], f"{path}.note"),
        )


@dataclass(frozen=True, slots=True)
class UnresolvedItem:
    unresolved_id: str
    topic: str
    reason: str
    review_required: bool = True
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty_string(self.unresolved_id, "$.unresolved_id")
        _require_nonempty_string(self.topic, "$.topic")
        _require_nonempty_string(self.reason, "$.reason")
        _require_bool(self.review_required, "$.review_required")
        object.__setattr__(self, "evidence_ids", _require_id_tuple(self.evidence_ids, "$.evidence_ids"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "unresolved_id": self.unresolved_id,
            "topic": self.topic,
            "reason": self.reason,
            "review_required": self.review_required,
            "evidence_ids": list(self.evidence_ids),
        }

    @classmethod
    def from_dict(cls, data: object, *, path: str = "$") -> UnresolvedItem:
        mapping = _require_exact_keys(
            data,
            frozenset({"unresolved_id", "topic", "reason", "review_required", "evidence_ids"}),
            path,
        )
        evidence_ids = mapping["evidence_ids"]
        if not isinstance(evidence_ids, list):
            _fail("type_error", "evidence_ids must be an array", f"{path}.evidence_ids")
        return cls(
            unresolved_id=_require_nonempty_string(
                mapping["unresolved_id"], f"{path}.unresolved_id"
            ),
            topic=_require_nonempty_string(mapping["topic"], f"{path}.topic"),
            reason=_require_nonempty_string(mapping["reason"], f"{path}.reason"),
            review_required=_require_bool(mapping["review_required"], f"{path}.review_required"),
            evidence_ids=tuple(evidence_ids),
        )


@dataclass(frozen=True, slots=True)
class EncounterRecord:
    encounter_id: str
    sources: tuple[Source, ...]
    evidence: tuple[EvidenceSpan, ...]
    atoms: tuple[ClinicalAtom, ...]
    conflicts: tuple[Conflict, ...] = ()
    unresolved: tuple[UnresolvedItem, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_nonempty_string(self.encounter_id, "$.encounter_id")
        if self.schema_version != SCHEMA_VERSION:
            _fail("schema_version", f"schema_version must be {SCHEMA_VERSION}", "$.schema_version")
        object.__setattr__(self, "sources", _require_typed_tuple(self.sources, Source, "$.sources"))
        object.__setattr__(
            self, "evidence", _require_typed_tuple(self.evidence, EvidenceSpan, "$.evidence")
        )
        object.__setattr__(self, "atoms", _require_typed_tuple(self.atoms, ClinicalAtom, "$.atoms"))
        object.__setattr__(
            self, "conflicts", _require_typed_tuple(self.conflicts, Conflict, "$.conflicts")
        )
        object.__setattr__(
            self, "unresolved", _require_typed_tuple(self.unresolved, UnresolvedItem, "$.unresolved")
        )
        self.validate()

    def source(self, source_id: str) -> Source:
        for item in self.sources:
            if item.source_id == source_id:
                return item
        _fail("unknown_source", f"unknown source_id: {source_id}", "$.sources")

    def span(self, evidence_id: str) -> EvidenceSpan:
        for item in self.evidence:
            if item.evidence_id == evidence_id:
                return item
        _fail("unknown_evidence", f"unknown evidence_id: {evidence_id}", "$.evidence")

    def atom(self, atom_id: str) -> ClinicalAtom:
        for item in self.atoms:
            if item.atom_id == atom_id:
                return item
        _fail("unknown_atom", f"unknown atom_id: {atom_id}", "$.atoms")

    def validate(self) -> None:
        self._validate_unique_ids()
        self._validate_sources()
        for index, span in enumerate(self.evidence):
            self._validate_span(span, path=f"$.evidence[{index}]")
        for index, atom in enumerate(self.atoms):
            self._validate_atom(atom, path=f"$.atoms[{index}]")
        for index, conflict in enumerate(self.conflicts):
            self._validate_conflict(conflict, path=f"$.conflicts[{index}]")
        for index, item in enumerate(self.unresolved):
            self._validate_unresolved(item, path=f"$.unresolved[{index}]")

    def _all_ids(self) -> Iterable[str]:
        yield self.encounter_id
        for source in self.sources:
            yield source.source_id
            for turn in source.turns:
                yield turn.turn_id
        for span in self.evidence:
            yield span.evidence_id
        for atom in self.atoms:
            yield atom.atom_id
        for conflict in self.conflicts:
            yield conflict.conflict_id
        for item in self.unresolved:
            yield item.unresolved_id

    def _validate_unique_ids(self) -> None:
        seen: set[str] = set()
        for identifier in self._all_ids():
            if identifier in seen:
                _fail("duplicate_id", f"duplicate identifier: {identifier}", "$")
            seen.add(identifier)

    def _validate_sources(self) -> None:
        for source in self.sources:
            for turn in source.turns:
                if turn.source_id != source.source_id:
                    _fail("source_mismatch", "turn.source_id must match source", "$.turns")
                if turn.end > len(source.text) or source.text[turn.start : turn.end] != turn.text:
                    _fail("evidence_text_mismatch", "turn text does not match offsets", "$.turns")

    def _validate_span(self, span: EvidenceSpan, *, path: str) -> None:
        source = self.source(span.source_id)
        if span.end > len(source.text):
            _fail("evidence_bounds", "evidence extends beyond the source", path)
        if source.text[span.start : span.end] != span.text:
            _fail("evidence_text_mismatch", "evidence text does not match its offsets", path)
        turn = source.turn(span.turn_id)
        if not (turn.start <= span.start and span.end <= turn.end):
            _fail("evidence_crosses_turn", "span cannot silently cross turn boundaries", path)
        if span.speaker is not turn.speaker:
            _fail("evidence_speaker_mismatch", "speaker must match the containing turn", path)

    def _validate_atom(self, atom: ClinicalAtom, *, path: str) -> None:
        spans = []
        for evidence_id in atom.evidence_ids:
            span = self.span(evidence_id)
            if span.speaker is not atom.speaker:
                _fail(
                    "evidence_speaker_mismatch",
                    "atom speaker must match its evidence speaker",
                    path,
                )
            spans.append(span)
        if atom.assertion_state is AssertionState.DENIED and not atom.evidence_ids:
            _fail("denied_without_evidence", "DENIED requires explicit evidence, not silence", path)
        if atom.assertion_state is AssertionState.ASSERTED and not atom.evidence_ids:
            _fail("asserted_without_evidence", "ASSERTED requires evidence", path)
        if atom.assertion_state is AssertionState.UNCERTAIN and not atom.evidence_ids:
            _fail(
                "uncertain_without_evidence",
                "UNCERTAIN requires explicit uncertainty evidence, not silence",
                path,
            )
        if atom.assertion_state in (AssertionState.ASSERTED, AssertionState.UNCERTAIN):
            if not any(_value_grounded_in_span(atom.raw_value, span.text) for span in spans):
                _fail(
                    "value_not_grounded",
                    "raw_value must be contained in at least one referenced evidence span",
                    path,
                )
        if any(span.speaker not in AUTHORITATIVE_SPEAKERS for span in spans):
            if not atom.review_required:
                _fail(
                    "nonauthoritative_without_review",
                    "OTHER/UNKNOWN evidence cannot settle an atom without review",
                    path,
                )
        if atom.assertion_state is AssertionState.CONFLICTING:
            distinct = {normalize_value(span.text) for span in spans}
            if len(distinct) < 2:
                _fail(
                    "insufficient_conflict_evidence",
                    "CONFLICTING requires two textually distinct evidence spans",
                    path,
                )

    def _validate_conflict(self, conflict: Conflict, *, path: str) -> None:
        for atom_id in conflict.atom_ids:
            self.atom(atom_id)
        for evidence_id in conflict.evidence_ids:
            self.span(evidence_id)

    def _validate_unresolved(self, item: UnresolvedItem, *, path: str) -> None:
        for evidence_id in item.evidence_ids:
            self.span(evidence_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "encounter_id": self.encounter_id,
            "sources": [source.to_dict() for source in self.sources],
            "evidence": [span.to_dict() for span in self.evidence],
            "atoms": [atom.to_dict() for atom in self.atoms],
            "conflicts": [conflict.to_dict() for conflict in self.conflicts],
            "unresolved": [item.to_dict() for item in self.unresolved],
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> EncounterRecord:
        mapping = _require_exact_keys(
            data,
            frozenset(
                {
                    "schema_version",
                    "encounter_id",
                    "sources",
                    "evidence",
                    "atoms",
                    "conflicts",
                    "unresolved",
                }
            ),
            "$",
        )
        for key in ("sources", "evidence", "atoms", "conflicts", "unresolved"):
            if not isinstance(mapping[key], list):
                _fail("type_error", f"{key} must be an array", f"$.{key}")
        record = cls(
            encounter_id=_require_nonempty_string(mapping["encounter_id"], "$.encounter_id"),
            sources=tuple(
                Source.from_dict(item, path=f"$.sources[{index}]")
                for index, item in enumerate(mapping["sources"])
            ),
            evidence=tuple(
                EvidenceSpan.from_dict(item, path=f"$.evidence[{index}]")
                for index, item in enumerate(mapping["evidence"])
            ),
            atoms=tuple(
                ClinicalAtom.from_dict(item, path=f"$.atoms[{index}]")
                for index, item in enumerate(mapping["atoms"])
            ),
            conflicts=tuple(
                Conflict.from_dict(item, path=f"$.conflicts[{index}]")
                for index, item in enumerate(mapping["conflicts"])
            ),
            unresolved=tuple(
                UnresolvedItem.from_dict(item, path=f"$.unresolved[{index}]")
                for index, item in enumerate(mapping["unresolved"])
            ),
            schema_version=_require_nonempty_string(mapping["schema_version"], "$.schema_version"),
        )
        record.validate()
        return record

    @classmethod
    def from_json(cls, raw: str) -> EncounterRecord:
        return cls.from_dict(_load_json(raw))
