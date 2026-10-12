"""Verification-gated adapter for Nano's direct state/pointer predictions.

The trained model proposes exactly five field states and bounded evidence spans.
This adapter validates those raw proposals, then asks the existing independent
closed-world binder to verify them.  It never receives evaluator truth and never
substitutes a binder guess for a wrong model proposal.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from nano_ai.contract import (
    FIELD_ORDER,
    EvidenceSpan,
    FieldOutput,
    FieldState,
    NanoInput,
    NanoOutput,
    normalize_value,
)
from nano_ai.solver import SolverDescriptor, SolverKind
from nano_ai.training.pointer_model import NANO_POINTER_PARAMETER_COUNT

from .deterministic_v0 import _extract_fields
from .state_span import (
    StateSpanFormatError,
    StateSpanProposal,
    _rejected_field,
    _verify_proposal,
)

POINTER_SPAN_DIAGNOSTICS_VERSION = "pointer-span-diagnostics-v0"
POINTER_SPAN_PIPELINE_VERSION = "nano-pointer-span-v0"

_STATE_CODE = {
    FieldState.SUPPORTED: "S",
    FieldState.ABSENT: "A",
    FieldState.MISSING: "M",
    FieldState.UNCERTAIN: "U",
    FieldState.CONFLICTING: "C",
}


def _required_span_count(state: FieldState) -> tuple[int, ...]:
    if state is FieldState.MISSING:
        return (0,)
    if state is FieldState.UNCERTAIN:
        return (0, 1)
    if state is FieldState.CONFLICTING:
        return (2,)
    return (1,)


def validate_pointer_proposals(
    proposals: object,
    transcript: str,
) -> tuple[StateSpanProposal, ...]:
    """Validate the complete raw pointer boundary without consulting gold."""

    if not isinstance(proposals, Sequence) or isinstance(
        proposals, (str, bytes, bytearray)
    ):
        raise StateSpanFormatError("pointer predictions must be a proposal sequence")
    frozen = tuple(proposals)
    if len(frozen) != len(FIELD_ORDER):
        raise StateSpanFormatError("pointer predictions must contain five fields")

    for expected_field, proposal in zip(FIELD_ORDER, frozen, strict=True):
        if not isinstance(proposal, StateSpanProposal):
            raise StateSpanFormatError(
                "pointer predictions must contain StateSpanProposal values"
            )
        if proposal.field is not expected_field:
            raise StateSpanFormatError(
                "pointer prediction fields must be unique and canonically ordered"
            )
        if proposal.state_code != _STATE_CODE.get(proposal.state):
            raise StateSpanFormatError("pointer state code and state disagree")
        allowed = _required_span_count(proposal.state)
        if len(proposal.spans) not in allowed:
            choices = " or ".join(str(value) for value in allowed)
            raise StateSpanFormatError(
                f"{proposal.state.value} requires {choices} evidence span(s)"
            )
        local_offsets: set[tuple[int, int]] = set()
        local_text: set[str] = set()
        for span in proposal.spans:
            if not isinstance(span, EvidenceSpan):
                raise StateSpanFormatError(
                    "pointer evidence must be EvidenceSpan values"
                )
            try:
                span.validate_against(transcript)
            except Exception as exc:
                raise StateSpanFormatError(
                    "pointer evidence must be exact text inside a Patient turn"
                ) from exc
            key = (span.start, span.end)
            if key in local_offsets:
                raise StateSpanFormatError("one field cannot repeat an evidence span")
            local_offsets.add(key)
            local_text.add(normalize_value(span.text))
        if proposal.state is FieldState.CONFLICTING and len(local_text) != 2:
            raise StateSpanFormatError(
                "conflicting pointer evidence must contain two distinct texts"
            )
    return frozen


def _model_owned_field(proposal: StateSpanProposal) -> FieldOutput:
    """Build an accepted field without importing a value from the verifier.

    A supported pointer proposal already owns one exact transcript span.  The
    contract's fixed surface normalizer turns that model-owned text into the
    canonical presented value; the deterministic binder may accept or reject
    the proposal, but it must never fill or correct the value.
    """

    if proposal.state is FieldState.SUPPORTED:
        return FieldOutput(
            field=proposal.field,
            state=proposal.state,
            value=normalize_value(proposal.spans[0].text),
            evidence=proposal.spans,
        )
    return FieldOutput(
        field=proposal.field,
        state=proposal.state,
        evidence=proposal.spans,
    )


class PointerSpanSolver:
    """Ground one direct state/pointer model without hiding raw model quality."""

    def __init__(
        self,
        predict: Callable[[str], Sequence[StateSpanProposal]],
        *,
        solver_id: str = "candidate/nano-pointer-span-v0",
        version: str = POINTER_SPAN_PIPELINE_VERSION,
        artifact_bytes: int | None = None,
    ) -> None:
        if not callable(predict):
            raise TypeError("predict must be callable")
        self._predict = predict
        self.descriptor = SolverDescriptor(
            solver_id=solver_id,
            kind=SolverKind.HYBRID,
            version=version,
            parameter_count=NANO_POINTER_PARAMETER_COUNT,
            artifact_bytes=artifact_bytes,
        )
        self.solver_id = solver_id

    def _infer(self, item: NanoInput) -> tuple[NanoOutput, dict[str, object]]:
        proposals = validate_pointer_proposals(
            self._predict(item.transcript), item.transcript
        )
        bound = {field.field: field for field in _extract_fields(item)}

        grounded = []
        traces: list[dict[str, object]] = []
        for proposal in proposals:
            source = bound[proposal.field]
            verified, reason = _verify_proposal(proposal, source)
            final = (
                _model_owned_field(proposal) if verified else _rejected_field(proposal)
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
            "protocol_version": POINTER_SPAN_DIAGNOSTICS_VERSION,
            "fields": traces,
        }

    def infer(self, item: NanoInput) -> NanoOutput:
        output, _diagnostics = self._infer(item)
        return output

    def infer_with_diagnostics(
        self, item: NanoInput
    ) -> tuple[NanoOutput, dict[str, object]]:
        """Return verifier-gated output plus auditable raw model proposals."""

        return self._infer(item)


__all__ = [
    "POINTER_SPAN_DIAGNOSTICS_VERSION",
    "POINTER_SPAN_PIPELINE_VERSION",
    "PointerSpanSolver",
    "validate_pointer_proposals",
]
