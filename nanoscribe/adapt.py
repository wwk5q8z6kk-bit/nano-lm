"""Model adapter — candidate proposals into PR2 PredictedEncounter.

The model proposes values/states/quotes. Software selects evidence.
EncounterRecord is never model output.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nanoscribe.encounter import EncounterRecord

from nanoscribe.encounter import (
    AssertionState,
    AtomType,
    Certainty,
    EncounterError,
    Experiencer,
    Speaker,
    Source,
    TemporalState,
    Temporality,
)
from nanoscribe.egress import ExternalEgressAuthorization
from nanoscribe.evaluate import PredictedAtom, PredictedEncounter
from nanoscribe.select import ConstrainedSelector

CANDIDATE_SCHEMA_VERSION = "nano.candidate.v0"

_LABEL_RE = re.compile(
    r"\b(STATED|ASSERTED|DENIED|UNCERTAIN|NOT_MENTIONED)\b(?:\s*:\s*(.*))?",
    re.IGNORECASE | re.DOTALL,
)
_QUOTED_RE = re.compile(r'"([^"]+)"')

_FORBIDDEN_CANDIDATE_KEYS = frozenset(
    {
        "start",
        "end",
        "evidence_id",
        "evidence_ids",
        "source_id",
        "turn_id",
        "spans",
        "normalized_value",
        "normalization_transform",
        "verifier_support",
        "support_relation",
    }
)

_LABEL_TO_ASSERTION = {
    "STATED": AssertionState.ASSERTED,
    "ASSERTED": AssertionState.ASSERTED,
    "DENIED": AssertionState.DENIED,
    "UNCERTAIN": AssertionState.UNCERTAIN,
}

_DEFAULT_SPEAKER = Speaker.PATIENT
_DEFAULT_EXPERIENCER = Experiencer.PATIENT
_DEFAULT_TEMPORALITY = TemporalState(kind=Temporality.CURRENT)
_DEFAULT_CERTAINTY = Certainty.STATED


class AdaptError(ValueError):
    """A machine-classifiable model-adapter violation."""

    def __init__(self, code: str, message: str, *, path: str = "$") -> None:
        self.code = code
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message} [{code}]")


def _fail(code: str, message: str, path: str) -> None:
    raise AdaptError(code, message, path=path)


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
    except AdaptError:
        raise
    except (json.JSONDecodeError, TypeError) as exc:
        _fail("invalid_json", str(exc), "$")


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


def _reject_forbidden_keys(mapping: Mapping[str, Any], path: str) -> None:
    forbidden = sorted(_FORBIDDEN_CANDIDATE_KEYS & frozenset(mapping))
    if forbidden:
        _fail(
            "forbidden_key",
            f"candidate must not include trusted-evidence keys: {', '.join(forbidden)}",
            path,
        )


def parse_label_and_quotes(raw: str) -> tuple[str | None, tuple[str, ...]]:
    """Parse Qwen span-port style one-line answers: STATED/DENIED/NOT_MENTIONED + quotes."""
    if not raw or not raw.strip():
        return None, ()
    matches = list(_LABEL_RE.finditer(raw))
    if not matches:
        return None, ()
    match = matches[-1]
    label = match.group(1).upper()
    rest = match.group(2) or ""
    quotes = tuple(part.strip() for part in _QUOTED_RE.findall(rest) if part.strip())
    return label, quotes


def extract_span_port_line(raw: str) -> str:
    """Pick the first line that looks like a span-port answer from model output."""
    stripped = raw.strip()
    if not stripped:
        return stripped
    for line in stripped.splitlines():
        candidate = line.strip()
        label, _ = parse_label_and_quotes(candidate)
        if label is not None:
            return candidate
    return stripped.splitlines()[0].strip()


def format_label_answer(label: str, quotes: Sequence[str] = ()) -> str:
    if not quotes:
        return label
    return f"{label}: " + " ".join(f'"{quote}"' for quote in quotes)


@dataclass(frozen=True, slots=True)
class ModelInput:
    """Encounter source plus model-facing context. No gold truth."""

    source: Source
    encounter_id: str
    prompt: str | None = None
    external_egress: ExternalEgressAuthorization | None = None

    def __post_init__(self) -> None:
        if self.external_egress is not None and not isinstance(
            self.external_egress, ExternalEgressAuthorization
        ):
            raise TypeError("external_egress must be an ExternalEgressAuthorization")


def _parse_candidate_temporality(value: object, path: str) -> TemporalState:
    mapping = _require_mapping(value, path)
    if "kind" not in mapping:
        _fail("missing_key", "temporality.kind is required", path)
    kind = _parse_enum(Temporality, mapping["kind"], f"{path}.kind")
    onset = mapping.get("onset_raw")
    duration = mapping.get("duration_raw")
    time_expr = mapping.get("time_expression_raw")
    if onset is not None and not isinstance(onset, str):
        _fail("type_error", "onset_raw must be a string or null", f"{path}.onset_raw")
    if duration is not None and not isinstance(duration, str):
        _fail("type_error", "duration_raw must be a string or null", f"{path}.duration_raw")
    if time_expr is not None and not isinstance(time_expr, str):
        _fail("type_error", "time_expression_raw must be a string or null", f"{path}.time_expression_raw")
    return TemporalState(
        kind=kind,
        onset_raw=onset,
        duration_raw=duration,
        time_expression_raw=time_expr,
    )


@dataclass(frozen=True, slots=True)
class CandidateEvidenceRequest:
    """A quote the model asks software to locate in the source."""

    quote: str


def evidence_requests(candidate: CandidateAtom) -> tuple[CandidateEvidenceRequest, ...]:
    """Lift model quotes into constrained-selector lookup requests."""
    return tuple(CandidateEvidenceRequest(quote=quote) for quote in candidate.quotes)


@dataclass(frozen=True, slots=True)
class CandidateAtom:
    """Model proposal — quotes only, never trusted offsets or evidence IDs."""

    atom_id: str
    atom_type: AtomType | None = None
    raw_value: str | None = None
    assertion_state: AssertionState | None = None
    speaker: Speaker | None = None
    experiencer: Experiencer | None = None
    temporality: TemporalState | None = None
    certainty: Certainty | None = None
    quotes: tuple[str, ...] = ()
    review_required: bool = False
    abstained: bool = False
    malformed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "atom_id": self.atom_id,
            "atom_type": self.atom_type.value if self.atom_type else None,
            "raw_value": self.raw_value,
            "assertion_state": self.assertion_state.value if self.assertion_state else None,
            "speaker": self.speaker.value if self.speaker else None,
            "experiencer": self.experiencer.value if self.experiencer else None,
            "temporality": self.temporality.to_dict() if self.temporality else None,
            "certainty": self.certainty.value if self.certainty else None,
            "quotes": list(self.quotes),
            "review_required": self.review_required,
            "abstained": self.abstained,
            "malformed": self.malformed,
        }

    @classmethod
    def from_dict(cls, data: object, *, path: str = "$") -> CandidateAtom:
        mapping = _require_mapping(data, path)
        _reject_forbidden_keys(mapping, path)
        if "schema_version" in mapping:
            _fail("forbidden_key", "candidate atoms must not carry schema_version", path)
        quotes = mapping.get("quotes")
        evidence_quote = mapping.get("evidence_quote")
        if evidence_quote is not None:
            if not isinstance(evidence_quote, str) or not evidence_quote:
                _fail("type_error", "evidence_quote must be a non-empty string", f"{path}.evidence_quote")
            quotes = [evidence_quote]
        if quotes is None:
            quotes = []
        elif not isinstance(quotes, list):
            _fail("type_error", "quotes must be an array of strings", f"{path}.quotes")
        if any(not isinstance(item, str) for item in quotes):
            _fail("type_error", "quotes must be an array of strings", f"{path}.quotes")
        temporality = mapping.get("temporality")
        return cls(
            atom_id=_require_nonempty_string(mapping["atom_id"], f"{path}.atom_id"),
            atom_type=(
                _parse_enum(AtomType, mapping["atom_type"], f"{path}.atom_type")
                if mapping.get("atom_type") is not None
                else None
            ),
            raw_value=(
                _require_nonempty_string(mapping["raw_value"], f"{path}.raw_value")
                if mapping.get("raw_value") is not None
                else None
            ),
            assertion_state=(
                _parse_enum(AssertionState, mapping["assertion_state"], f"{path}.assertion_state")
                if mapping.get("assertion_state") is not None
                else None
            ),
            speaker=(
                _parse_enum(Speaker, mapping["speaker"], f"{path}.speaker")
                if mapping.get("speaker") is not None
                else None
            ),
            experiencer=(
                _parse_enum(Experiencer, mapping["experiencer"], f"{path}.experiencer")
                if mapping.get("experiencer") is not None
                else None
            ),
            temporality=(
                _parse_candidate_temporality(temporality, f"{path}.temporality")
                if temporality is not None
                else None
            ),
            certainty=(
                _parse_enum(Certainty, mapping["certainty"], f"{path}.certainty")
                if mapping.get("certainty") is not None
                else None
            ),
            quotes=tuple(quotes),
            review_required=_require_bool(mapping.get("review_required", False), f"{path}.review_required"),
            abstained=_require_bool(mapping.get("abstained", False), f"{path}.abstained"),
            malformed=_require_bool(mapping.get("malformed", False), f"{path}.malformed"),
        )


class AdapterExecutionMode(str, Enum):
    """How an adapter batch was actually produced."""

    UNSPECIFIED = "unspecified"
    FIXTURE = "fixture"
    LOCAL_WEIGHTS = "local_weights"
    EXTERNAL_API = "external_api"


@dataclass(frozen=True, slots=True)
class ModelCandidate:
    """Bundle of model proposals for one encounter."""

    atoms: tuple[CandidateAtom, ...]
    schema_version: str = CANDIDATE_SCHEMA_VERSION
    latency_s: float = 0.0
    memory_bytes: int = 0
    execution_mode: AdapterExecutionMode = AdapterExecutionMode.UNSPECIFIED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "atoms": [atom.to_dict() for atom in self.atoms],
            "latency_s": self.latency_s,
            "memory_bytes": self.memory_bytes,
            "execution_mode": self.execution_mode.value,
        }

    @classmethod
    def from_dict(cls, data: object) -> ModelCandidate:
        mapping = _require_mapping(data, "$")
        if mapping.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
            _fail(
                "schema_version",
                f"schema_version must be {CANDIDATE_SCHEMA_VERSION}",
                "$.schema_version",
            )
        atoms = mapping.get("atoms", ())
        if not isinstance(atoms, list):
            _fail("type_error", "atoms must be an array", "$.atoms")
        latency = mapping.get("latency_s", 0.0)
        memory = mapping.get("memory_bytes", 0)
        execution_mode = mapping.get("execution_mode", AdapterExecutionMode.UNSPECIFIED.value)
        if not isinstance(latency, (int, float)) or isinstance(latency, bool):
            _fail("type_error", "latency_s must be a number", "$.latency_s")
        if not isinstance(memory, int) or isinstance(memory, bool):
            _fail("type_error", "memory_bytes must be an integer", "$.memory_bytes")
        execution = _parse_enum(
            AdapterExecutionMode,
            execution_mode,
            "$.execution_mode",
        )
        return cls(
            atoms=tuple(
                CandidateAtom.from_dict(atom, path=f"$.atoms[{index}]")
                for index, atom in enumerate(atoms)
            ),
            schema_version=CANDIDATE_SCHEMA_VERSION,
            latency_s=float(latency),
            memory_bytes=memory,
            execution_mode=execution,
        )

    @classmethod
    def from_json(cls, raw: str) -> ModelCandidate:
        return cls.from_dict(_load_json(raw))


# Mandate name for batched model proposals (alias kept for clarity in pipeline docs).
ModelCandidateBatch = ModelCandidate


def candidate_from_span_port_line(
    *,
    atom_id: str,
    atom_type: AtomType,
    raw_value: str,
    raw_line: str,
    speaker: Speaker = _DEFAULT_SPEAKER,
    experiencer: Experiencer = _DEFAULT_EXPERIENCER,
    temporality: TemporalState | None = None,
    certainty: Certainty | None = None,
) -> CandidateAtom:
    """Lift a Qwen span-port one-liner into a typed candidate (no evidence binding)."""
    label, quotes = parse_label_and_quotes(raw_line)
    if label is None:
        return CandidateAtom(atom_id=atom_id, malformed=True)
    if label == "NOT_MENTIONED":
        return CandidateAtom(atom_id=atom_id, abstained=True)
    assertion = _LABEL_TO_ASSERTION.get(label)
    if assertion is None:
        return CandidateAtom(atom_id=atom_id, malformed=True)
    if not quotes and raw_value:
        quotes = (raw_value,)
    certainty_value = certainty
    if assertion is AssertionState.UNCERTAIN:
        certainty_value = certainty or Certainty.UNCERTAIN
    elif assertion is AssertionState.DENIED:
        certainty_value = certainty or Certainty.STATED
    else:
        certainty_value = certainty or Certainty.STATED
    return CandidateAtom(
        atom_id=atom_id,
        atom_type=atom_type,
        raw_value=raw_value,
        assertion_state=assertion,
        speaker=speaker,
        experiencer=experiencer,
        temporality=temporality or _DEFAULT_TEMPORALITY,
        certainty=certainty_value or _DEFAULT_CERTAINTY,
        quotes=quotes,
    )


def adapt_candidate(
    source: Source,
    candidate: CandidateAtom,
    *,
    selector: ConstrainedSelector | None = None,
    evidence_id_prefix: str | None = None,
) -> PredictedAtom:
    """Bind one candidate through constrained evidence selection."""
    selector = selector or ConstrainedSelector()
    prefix = evidence_id_prefix or candidate.atom_id
    if candidate.malformed:
        return PredictedAtom(atom_id=candidate.atom_id, malformed=True)
    if candidate.abstained:
        return PredictedAtom(atom_id=candidate.atom_id, abstained=True)
    spans = []
    evidence_ids: list[str] = []
    for index, quote in enumerate(candidate.quotes):
        evidence_id = f"{prefix}:ev{index}"
        span = selector.select_quote(source, quote, evidence_id=evidence_id)
        if span is None:
            return PredictedAtom(
                atom_id=candidate.atom_id,
                atom_type=candidate.atom_type,
                raw_value=candidate.raw_value,
                assertion_state=candidate.assertion_state,
                speaker=candidate.speaker,
                experiencer=candidate.experiencer,
                temporality=candidate.temporality,
                certainty=candidate.certainty,
                review_required=candidate.review_required,
                abstained=True,
                quote=quote,
            )
        spans.append(span)
        evidence_ids.append(span.evidence_id)
    quote = candidate.quotes[0] if candidate.quotes else None
    return PredictedAtom(
        atom_id=candidate.atom_id,
        atom_type=candidate.atom_type,
        raw_value=candidate.raw_value,
        assertion_state=candidate.assertion_state,
        speaker=candidate.speaker,
        experiencer=candidate.experiencer,
        temporality=candidate.temporality,
        certainty=candidate.certainty,
        evidence_ids=tuple(evidence_ids),
        spans=tuple(spans),
        review_required=candidate.review_required,
        quote=quote,
    )


def adapt(
    model_input: ModelInput,
    candidate: ModelCandidate,
    *,
    selector: ConstrainedSelector | None = None,
) -> PredictedEncounter:
    """Convert model proposals into a PR2 PredictedEncounter."""
    selector = selector or ConstrainedSelector()
    atoms = tuple(
        adapt_candidate(
            model_input.source,
            atom,
            selector=selector,
            evidence_id_prefix=f"{model_input.encounter_id}:{atom.atom_id}",
        )
        for atom in candidate.atoms
    )
    return PredictedEncounter(
        atoms=atoms,
        latency_s=candidate.latency_s,
        memory_bytes=candidate.memory_bytes,
    )


def adapt_span_port_line(
    model_input: ModelInput,
    *,
    atom_id: str,
    atom_type: AtomType,
    raw_value: str,
    raw_line: str,
    selector: ConstrainedSelector | None = None,
    speaker: Speaker = _DEFAULT_SPEAKER,
    experiencer: Experiencer = _DEFAULT_EXPERIENCER,
    temporality: TemporalState | None = None,
    certainty: Certainty | None = None,
) -> PredictedAtom:
    """Convenience path for the historical Qwen one-line answer format."""
    candidate = candidate_from_span_port_line(
        atom_id=atom_id,
        atom_type=atom_type,
        raw_value=raw_value,
        raw_line=raw_line,
        speaker=speaker,
        experiencer=experiencer,
        temporality=temporality,
        certainty=certainty,
    )
    return adapt_candidate(model_input.source, candidate, selector=selector)


def adapt_json(raw: str, model_input: ModelInput, *, selector: ConstrainedSelector | None = None) -> PredictedEncounter:
    return adapt(model_input, ModelCandidate.from_json(raw), selector=selector)


def run_pipeline(
    model_input: ModelInput,
    batch: ModelCandidateBatch,
    *,
    selector: ConstrainedSelector | None = None,
    gold: EncounterRecord | None = None,
):
    """End-to-end smoke: candidates → PredictedEncounter → optional PR2 eval."""
    from nanoscribe.evaluate import EvalReport, evaluate

    predicted = adapt(model_input, batch, selector=selector)
    report = evaluate(gold, predicted) if gold is not None else None
    return predicted, report
