"""Development context adapter; preserved experimental adapters stay byte-identical.

The inference trace intentionally mirrors the frozen state-span adapter. Only
its source binder changes; parsing and proposal acceptance reuse that adapter.
"""
from nano_ai.adapters.state_span import (
    STATE_SPAN_DIAGNOSTICS_VERSION, StateSpanSolver, parse_state_span_summary,
    _verify_proposal, _accepted_field, _rejected_field,
)
from nano_ai.contract import NanoInput, NanoOutput, FieldOutput
from nanoscribe.context import extract_context_fields


class ContextStateSpanSolver(StateSpanSolver):
    """Bind model proposals using the bounded development question grammar."""

    def _infer(self, item: NanoInput) -> tuple[NanoOutput, dict[str, object]]:
        raw = self._predict(item.transcript)
        proposals = parse_state_span_summary(raw, item.transcript)
        bound = {field.field: field for field in extract_context_fields(item)}

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
