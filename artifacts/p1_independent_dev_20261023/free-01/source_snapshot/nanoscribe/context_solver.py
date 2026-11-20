"""Development context adapter; preserved experimental adapters stay byte-identical.

Raw model decisions remain visible separately from source-preserving repairs.
This hybrid development policy is not the frozen learned-state instrument.
"""
from nano_ai.adapters.state_span import (
    StateSpanSolver, parse_state_span_summary, _verify_proposal, _accepted_field,
)
from nano_ai.contract import NanoInput, NanoOutput, FieldOutput, FieldState
from nanoscribe.context import extract_context_fields
from nanoscribe.check_note import inspect_context, observations_conflict, transcript_source

CONTEXT_DIAGNOSTICS_VERSION = "nanoscribe.context-state-diagnostics.v2"


class ContextStateSpanSolver(StateSpanSolver):
    """Bind model proposals using the bounded development question grammar."""

    def _infer(self, item: NanoInput) -> tuple[NanoOutput, dict[str, object]]:
        raw = self._predict(item.transcript)
        proposals = parse_state_span_summary(raw, item.transcript)
        bound = {field.field: field for field in extract_context_fields(item)}
        document = transcript_source(item)
        observations, global_unsafe = inspect_context(document)

        grounded: list[FieldOutput] = []
        traces: list[dict[str, object]] = []
        for proposal in proposals:
            source = bound[proposal.field]
            verified, reason = _verify_proposal(proposal, source)
            if (not verified and not global_unsafe and proposal.spans
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
                evidence = tuple(dict.fromkeys((*source.evidence, *proposal.spans)))
                final = FieldOutput(proposal.field, FieldState.UNCERTAIN, evidence=evidence)
                decision = "rejected_to_uncertain"
            grounded.append(final)
            traces.append(
                {
                    "field": proposal.field.value,
                    "raw_state_code": proposal.state_code,
                    "raw_state": proposal.state.value,
                    "raw_spans": [span.to_dict() for span in proposal.spans],
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
            "fields": traces,
        }

    def infer_with_diagnostics(self, item: NanoInput) -> tuple[NanoOutput, dict[str, object]]:
        """Expose raw/repaired decisions through run_inference without a rerun."""
        return self._infer(item)
