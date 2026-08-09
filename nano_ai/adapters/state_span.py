"""Strict native-state candidate grammar and evidence-gated adapter.

The model proposes a state and exact patient-authored span for each field.  The
closed-world binder independently checks those proposals before any supported or
absent value is exposed.  A well-formed but unverified proposal becomes
``uncertain``; malformed grammar or an unlocatable/ambiguous span fails the whole
inference so format failures remain measurable.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from nano_ai.contract import (
    FIELD_ORDER,
    EvidenceSpan,
    FieldName,
    FieldOutput,
    FieldState,
    NanoInput,
    NanoOutput,
    normalize_value,
)
from nano_ai.solver import SolverDescriptor, SolverKind

from .deterministic_v0 import _extract_fields

STATE_SPAN_DIAGNOSTICS_VERSION = "state-span-diagnostics-v0"

_FIELD_LABELS: dict[FieldName, str] = {
    FieldName.CHIEF_COMPLAINT: "CC",
    FieldName.DURATION: "DUR",
    FieldName.SEVERITY: "SEV",
    FieldName.MEDICATION: "MED",
    FieldName.ALLERGY: "ALG",
}
_STATE_BY_CODE: dict[str, FieldState] = {
    "S": FieldState.SUPPORTED,
    "A": FieldState.ABSENT,
    "M": FieldState.MISSING,
    "U": FieldState.UNCERTAIN,
    "C": FieldState.CONFLICTING,
}
_SLOT_RE = re.compile(r"(?P<code>[SAMUC])(?:\[(?P<payload>[^\[\]|\r\n]*)\])?\Z")
_PATIENT_PREFIX_RE = re.compile(r"^[ \t]*patient[ \t]*:[ \t]*", flags=re.IGNORECASE)


class StateSpanFormatError(ValueError):
    """The native-state output cannot be parsed without guessing."""


@dataclass(frozen=True, slots=True)
class StateSpanProposal:
    """One parsed raw model proposal with uniquely located evidence."""

    field: FieldName
    state_code: str
    state: FieldState
    spans: tuple[EvidenceSpan, ...]


def _patient_content_ranges(transcript: str) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    offset = 0
    for line in transcript.splitlines(keepends=True):
        visible = line.rstrip("\r\n")
        match = _PATIENT_PREFIX_RE.match(visible)
        if match is not None and match.end() < len(visible):
            ranges.append((offset + match.end(), offset + len(visible)))
        offset += len(line)
    return tuple(ranges)


def _locate_unique_patient_span(transcript: str, text: str) -> EvidenceSpan:
    matches: list[tuple[int, int]] = []
    for content_start, content_end in _patient_content_ranges(transcript):
        position = transcript.find(text, content_start, content_end)
        while position >= 0:
            matches.append((position, position + len(text)))
            position = transcript.find(text, position + 1, content_end)
    if not matches:
        raise StateSpanFormatError(
            "native-state evidence is not exact text inside a Patient turn"
        )
    if len(matches) != 1:
        raise StateSpanFormatError(
            "native-state evidence is ambiguous across Patient turns"
        )
    start, end = matches[0]
    span = EvidenceSpan(start=start, end=end, text=text, speaker="patient")
    span.validate_against(transcript)
    return span


def classify_patient_span_relocation(
    transcript: str, text: str
) -> str:
    """Classify route-(b) generated span text against the unmodified locator.

    Returns one of ``relocated``, ``no_match``, or ``ambiguous``. Used by the
    span-port falsifier (`papers/PREREG_SPAN_PORT_B.md`) so probe code does not
    re-implement or reinterpret the binder's refusal rules.
    """
    try:
        _locate_unique_patient_span(transcript, text)
    except StateSpanFormatError as exc:
        message = str(exc)
        if "ambiguous" in message:
            return "ambiguous"
        return "no_match"
    return "relocated"


def _payload_texts(code: str, payload: str | None) -> tuple[str, ...]:
    if code == "M":
        if payload is not None:
            raise StateSpanFormatError("M must not carry an evidence bracket")
        return ()
    if payload is None:
        raise StateSpanFormatError(f"{code} must carry an evidence bracket")
    if code == "U" and payload == "":
        return ()
    if not payload or payload.strip() != payload:
        raise StateSpanFormatError(
            f"{code} evidence must be non-empty and edge-trimmed"
        )
    if code == "C":
        parts = tuple(payload.split(";"))
        if len(parts) != 2 or any(not part or part.strip() != part for part in parts):
            raise StateSpanFormatError(
                "C must carry exactly two edge-trimmed evidence spans"
            )
        if parts[0] == parts[1]:
            raise StateSpanFormatError("C evidence spans must not be duplicates")
        return parts
    if ";" in payload:
        raise StateSpanFormatError(
            f"{code} must carry exactly one unambiguous evidence span"
        )
    return (payload,)


def parse_state_span_summary(
    raw_summary: object,
    transcript: str,
) -> tuple[StateSpanProposal, ...]:
    """Parse the five ordered slots and uniquely locate every proposed span."""

    if not isinstance(raw_summary, str):
        raise StateSpanFormatError("native-state summary must be text")
    pieces = raw_summary.split("|")
    if len(pieces) != len(FIELD_ORDER):
        raise StateSpanFormatError(
            "native-state summary must contain exactly five ordered slots"
        )

    proposals: list[StateSpanProposal] = []
    used_offsets: set[tuple[int, int]] = set()
    for field, piece in zip(FIELD_ORDER, pieces, strict=True):
        prefix = f"{_FIELD_LABELS[field]}:"
        if not piece.startswith(prefix):
            raise StateSpanFormatError(
                "native-state fields must be exactly ordered as CC, DUR, SEV, MED, ALG"
            )
        slot = _SLOT_RE.fullmatch(piece[len(prefix) :])
        if slot is None:
            raise StateSpanFormatError(
                f"{_FIELD_LABELS[field]} has malformed native-state syntax"
            )
        code = slot.group("code")
        texts = _payload_texts(code, slot.group("payload"))
        spans = tuple(_locate_unique_patient_span(transcript, text) for text in texts)
        offsets = {(span.start, span.end) for span in spans}
        if len(offsets) != len(spans):
            raise StateSpanFormatError("native-state evidence spans are duplicated")
        if used_offsets & offsets:
            raise StateSpanFormatError(
                "one exact evidence span cannot be duplicated across field proposals"
            )
        used_offsets.update(offsets)
        proposals.append(
            StateSpanProposal(
                field=field,
                state_code=code,
                state=_STATE_BY_CODE[code],
                spans=spans,
            )
        )
    return tuple(proposals)


def _span_key(span: EvidenceSpan) -> tuple[int, int, str]:
    return (span.start, span.end, span.text)


def _all_spans_are_bound(
    proposed: tuple[EvidenceSpan, ...], source: FieldOutput
) -> bool:
    source_keys = {_span_key(span) for span in source.evidence}
    return all(_span_key(span) in source_keys for span in proposed)


def _verify_proposal(
    proposal: StateSpanProposal,
    source: FieldOutput,
) -> tuple[bool, str]:
    if proposal.state is not source.state:
        return False, "independent_binder_state_mismatch"
    if proposal.state is FieldState.MISSING:
        return (not proposal.spans, "verified_missing_state")
    if proposal.state is FieldState.SUPPORTED:
        assert source.value is not None
        span = proposal.spans[0]
        if normalize_value(span.text) != normalize_value(source.value):
            return False, "normalized_value_mismatch"
        if not _all_spans_are_bound(proposal.spans, source):
            return False, "supported_evidence_mismatch"
        return True, "verified_supported_span"
    if proposal.state is FieldState.ABSENT:
        if not _all_spans_are_bound(proposal.spans, source):
            return False, "denial_evidence_mismatch"
        return True, "verified_denial_span"
    if proposal.state is FieldState.UNCERTAIN:
        if not proposal.spans:
            if source.evidence:
                return False, "uncertainty_evidence_omitted"
            return True, "verified_evidenceless_uncertainty"
        if not _all_spans_are_bound(proposal.spans, source):
            return False, "uncertainty_evidence_mismatch"
        return True, "verified_uncertainty_span"
    if not _all_spans_are_bound(proposal.spans, source):
        return False, "conflict_evidence_mismatch"
    return True, "verified_conflicting_spans"


def _accepted_field(
    proposal: StateSpanProposal,
    source: FieldOutput,
) -> FieldOutput:
    if proposal.state is FieldState.SUPPORTED:
        return FieldOutput(
            field=proposal.field,
            state=proposal.state,
            value=source.value,
            evidence=proposal.spans,
        )
    return FieldOutput(
        field=proposal.field,
        state=proposal.state,
        evidence=proposal.spans,
    )


def _rejected_field(proposal: StateSpanProposal) -> FieldOutput:
    # Never substitute the binder's missing/conflicting state for a wrong raw
    # proposal.  Keeping the final state uncertain makes learned state accuracy
    # observable while still blocking an unsupported presentation.
    return FieldOutput(
        field=proposal.field,
        state=FieldState.UNCERTAIN,
        evidence=proposal.spans,
    )


class StateSpanSolver:
    """Ground strict native-state proposals without exposing verifier guesses."""

    def __init__(
        self,
        predict: Callable[[str], object],
        *,
        solver_id: str = "candidate/native-state-span-v0",
        version: str = "0",
        parameter_count: int | None = None,
        artifact_bytes: int | None = None,
    ) -> None:
        if not callable(predict):
            raise TypeError("predict must be callable")
        self._predict = predict
        self.descriptor = SolverDescriptor(
            solver_id=solver_id,
            kind=SolverKind.LEGACY_ADAPTER,
            version=version,
            parameter_count=parameter_count,
            artifact_bytes=artifact_bytes,
        )
        self.solver_id = solver_id

    def _infer(self, item: NanoInput) -> tuple[NanoOutput, dict[str, object]]:
        raw = self._predict(item.transcript)
        proposals = parse_state_span_summary(raw, item.transcript)
        bound = {field.field: field for field in _extract_fields(item)}

        grounded: list[FieldOutput] = []
        traces: list[dict[str, object]] = []
        for proposal in proposals:
            source = bound[proposal.field]
            verified, reason = _verify_proposal(proposal, source)
            final = (
                _accepted_field(proposal, source)
                if verified
                else _rejected_field(proposal)
            )
            grounded.append(final)
            traces.append(
                {
                    "field": proposal.field.value,
                    "raw_state_code": proposal.state_code,
                    "raw_state": proposal.state.value,
                    "raw_spans": [span.to_dict() for span in proposal.spans],
                    "verifier_state": source.state.value,
                    "verified": verified,
                    "decision": "accepted" if verified else "rejected_to_uncertain",
                    "reason": reason,
                    "final_state": final.state.value,
                }
            )

        output = NanoOutput(
            item_id=item.item_id,
            solver_id=self.solver_id,
            fields=tuple(grounded),
        )
        output.validate_against(item)
        return output, {
            "protocol_version": STATE_SPAN_DIAGNOSTICS_VERSION,
            "raw_summary": raw,
            "fields": traces,
        }

    def infer(self, item: NanoInput) -> NanoOutput:
        output, _diagnostics = self._infer(item)
        return output

    def infer_with_state_diagnostics(
        self, item: NanoInput
    ) -> tuple[NanoOutput, dict[str, object]]:
        """Return the new trace explicitly; the legacy evaluator does not score it."""

        return self._infer(item)


__all__ = [
    "STATE_SPAN_DIAGNOSTICS_VERSION",
    "StateSpanFormatError",
    "StateSpanProposal",
    "StateSpanSolver",
    "parse_state_span_summary",
]
