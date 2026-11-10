"""Development context adapter; preserved experimental adapters stay byte-identical.

Raw model decisions remain visible separately from source-preserving repairs.
This hybrid development policy is not the frozen learned-state instrument.
"""
from __future__ import annotations

from nano_ai.adapters.state_span import (
    StateSpanFormatError, StateSpanProposal, StateSpanSolver,
    parse_state_span_summary, _verify_proposal, _accepted_field, _FIELD_LABELS,
)
from nano_ai.contract import FIELD_ORDER, NanoInput, NanoOutput, FieldOutput, FieldState
from nanoscribe.context import extract_context_fields
from nanoscribe.check_note import inspect_context, observations_conflict, transcript_source

CONTEXT_DIAGNOSTICS_VERSION = "nanoscribe.context-state-diagnostics.v3"


def _parse_context_proposals(
    raw: object, transcript: str,
) -> tuple[list[str], list[StateSpanProposal | None], list[str | None], str | None]:
    """Recover independent slots without weakening the frozen exact parser.

    Only five correctly labelled, ordered slots establish field boundaries.
    Every isolated field still has to pass the original grammar and unique
    patient-span locator. No text repair or evidence disambiguation is allowed.
    """
    try:
        proposals = parse_state_span_summary(raw, transcript)
    except StateSpanFormatError as exc:
        strict_error = str(exc)
        if not isinstance(raw, str):
            raise
        pieces = raw.split("|")
        if len(pieces) != len(FIELD_ORDER) or any(
            not piece.startswith(f"{_FIELD_LABELS[field]}:")
            for field, piece in zip(FIELD_ORDER, pieces)
        ):
            raise
    else:
        return raw.split("|"), list(proposals), [None] * len(FIELD_ORDER), None

    recovered: list[StateSpanProposal | None] = []
    errors: list[str | None] = []
    for index, piece in enumerate(pieces):
        isolated = [f"{_FIELD_LABELS[field]}:M" for field in FIELD_ORDER]
        isolated[index] = piece
        try:
            recovered.append(parse_state_span_summary("|".join(isolated), transcript)[index])
            errors.append(None)
        except StateSpanFormatError as exc:
            recovered.append(None)
            errors.append(str(exc))

    # Isolation removes the frozen parser's cross-slot duplicate check. Restore
    # it symmetrically: both the first and every later owner must be rejected.
    owners: dict[tuple[int, int], set[int]] = {}
    for index, proposal in enumerate(recovered):
        if proposal is not None:
            for span in proposal.spans:
                owners.setdefault((span.start, span.end), set()).add(index)
    for indices in owners.values():
        if len(indices) > 1:
            for index in indices:
                errors[index] = "one exact evidence span cannot be duplicated across field proposals"
    return pieces, recovered, errors, strict_error


class ContextStateSpanSolver(StateSpanSolver):
    """Bind model proposals using the bounded development question grammar."""

    def _infer(self, item: NanoInput) -> tuple[NanoOutput, dict[str, object]]:
        raw = self._predict(item.transcript)
        slots, proposals, parse_errors, strict_error = _parse_context_proposals(raw, item.transcript)
        bound = {field.field: field for field in extract_context_fields(item)}
        document = transcript_source(item)
        observations, global_unsafe = inspect_context(document)

        grounded: list[FieldOutput] = []
        traces: list[dict[str, object]] = []
        for field, slot, proposal, parse_error in zip(FIELD_ORDER, slots, proposals, parse_errors, strict=True):
            source = bound[field]
            if parse_error is not None:
                verified, reason = False, "native_state_parse_failed"
            else:
                assert proposal is not None
                verified, reason = _verify_proposal(proposal, source)
            if (parse_error is None and not verified and not global_unsafe and proposal.spans
                    and proposal.state is source.state is FieldState.UNCERTAIN):
                # Aggregate uncertainty (e.g. coexisting medicine names) has
                # scalar source spans, while U syntax quotes the whole reply.
                # Accept only complete replies containing bound observations
                # of this field; unrelated or partial quotes stay rejected.
                full_replies = {
                    (turn.start, turn.end, turn.text) for turn in document.turns
                    if any(observation.field is proposal.field and observation.span is not None
                           and observation.span in source.evidence
                           and turn.start <= observation.span.start < observation.span.end <= turn.end
                           for observation in observations)
                }
                if all((span.start, span.end, span.text) in full_replies for span in proposal.spans):
                    verified, reason = True, "verified_uncertainty_reply"
            if verified and proposal.state is FieldState.CONFLICTING:
                selected = tuple(observation for observation in observations
                                 if observation.field is proposal.field and observation.span in proposal.spans)
                if not observations_conflict(proposal.field, selected):
                    verified, reason = False, "proposed_pair_does_not_establish_conflict"
            if source.state is FieldState.CONFLICTING:
                # Never discard a checked conflict or its third observation
                # because the model picked only a denial/value/two spans.
                final = source
                decision = "accepted_preserving_source_conflict" if verified else "rejected_preserving_source_conflict"
            elif verified:
                final = _accepted_field(proposal, source)
                # Keep repeated/uncertain observations even if the proposal
                # quoted only one. The model's original spans remain in trace.
                final = FieldOutput(final.field, final.state, final.value, source.evidence)
                decision = "accepted"
            else:
                evidence = tuple(dict.fromkeys((*source.evidence, *(proposal.spans if proposal else ()))))
                final = FieldOutput(field, FieldState.UNCERTAIN, evidence=evidence)
                decision = "rejected_to_uncertain"
            grounded.append(final)
            traces.append(
                {
                    "field": field.value,
                    "raw_slot": slot,
                    "raw_state_code": proposal.state_code if proposal else None,
                    "raw_state": proposal.state.value if proposal else None,
                    "raw_spans": [span.to_dict() for span in proposal.spans] if proposal else [],
                    "parse_error": parse_error,
                    "verifier_state": source.state.value,
                    "verified": verified,
                    "decision": decision,
                    "reason": reason,
                    "final_state": final.state.value,
                    "final_spans": [span.to_dict() for span in final.evidence],
                }
            )

        output = NanoOutput(
            item_id=item.item_id,
            solver_id=self.solver_id,
            fields=tuple(grounded),
        )
        output.validate_against(item)
        return output, {
            "protocol_version": CONTEXT_DIAGNOSTICS_VERSION,
            "raw_summary": raw,
            "strict_parse_error": strict_error,
            "recovered_field_count": sum(error is None for error in parse_errors) if strict_error else 0,
            "fields": traces,
        }

    def infer_with_diagnostics(self, item: NanoInput) -> tuple[NanoOutput, dict[str, object]]:
        """Expose raw/repaired decisions through run_inference without a rerun."""
        return self._infer(item)
