"""Adapter for the repository's legacy pipe-delimited scribe summaries."""

from __future__ import annotations

import re
from collections.abc import Callable

from nano_ai.contract import (
    FIELD_ORDER,
    FieldName,
    FieldOutput,
    FieldState,
    NanoInput,
    NanoOutput,
)
from nano_ai.solver import SolverDescriptor, SolverKind

from .deterministic_v0 import _canonical_value, _extract_fields

_LEGACY_KEYS: dict[FieldName, str] = {
    FieldName.CHIEF_COMPLAINT: "cc",
    FieldName.DURATION: "dur",
    FieldName.SEVERITY: "sev",
    FieldName.MEDICATION: "med",
    FieldName.ALLERGY: "alg",
}

_SUMMARY_RE = re.compile(
    r"^\s*CC:\s*(?P<cc>.*?)\s*\|\s*"
    r"DUR:\s*(?P<dur>.*?)\s*\|\s*"
    r"SEV:\s*(?P<sev>.*?)\s*\|\s*"
    r"MED:\s*(?P<med>.*?)\s*\|\s*"
    r"ALG:\s*(?P<alg>.*?)\s*$",
    re.IGNORECASE | re.DOTALL,
)

LEGACY_DIAGNOSTICS_VERSION = "legacy-summary-diagnostics-v0"


class LegacySummaryFormatError(ValueError):
    """The legacy generator did not return the frozen five-field format."""


def _uncertain(field: FieldName, source: FieldOutput | None = None) -> FieldOutput:
    evidence = () if source is None else source.evidence
    return FieldOutput(field=field, state=FieldState.UNCERTAIN, evidence=evidence)


class LegacySummarySolver:
    """Ground a legacy model's five proposed values before exposing them.

    ``predict`` receives only the transcript string.  A proposed value is emitted
    as supported only when the closed-world v0 binding independently finds the
    same value in the corresponding patient reply.  A proposed ``none`` becomes
    absent only when that reply contains an explicit recognized denial.
    """

    def __init__(
        self,
        predict: Callable[[str], object],
        *,
        solver_id: str = "legacy/summary-v0",
        version: str = "0",
        parameter_count: int | None = None,
        artifact_bytes: int | None = None,
    ) -> None:
        self._predict = predict
        self.descriptor = SolverDescriptor(
            solver_id=solver_id,
            kind=SolverKind.LEGACY_ADAPTER,
            version=version,
            parameter_count=parameter_count,
            artifact_bytes=artifact_bytes,
        )
        self.solver_id = solver_id

    def infer_with_diagnostics(
        self, item: NanoInput
    ) -> tuple[NanoOutput, dict[str, object]]:
        """Return the verified contract output and a gold-free proposal trace."""

        raw = self._predict(item.transcript)
        if not isinstance(raw, str):
            raise LegacySummaryFormatError("legacy summary must be text")
        raw = str(raw)
        match = _SUMMARY_RE.fullmatch(raw)
        if match is None:
            raise LegacySummaryFormatError(
                "legacy summary must contain exactly CC, DUR, SEV, MED, and ALG"
            )

        bound = {field.field: field for field in _extract_fields(item)}
        grounded: list[FieldOutput] = []
        diagnostics: list[dict[str, object]] = []
        for field in FIELD_ORDER:
            source = bound[field]
            proposal = match.group(_LEGACY_KEYS[field]).strip()
            raw_proposal = proposal or None
            if not proposal:
                proposal_kind = "missing"
            elif proposal.casefold() == "none":
                proposal_kind = "absence"
            else:
                proposal_kind = "value"

            if source.state is FieldState.CONFLICTING:
                final = source
                decision = "preserved_conflict"
                reason = "conflicting_transcript_evidence"
            elif proposal_kind == "missing":
                final = _uncertain(field, source)
                decision = "native_abstention"
                reason = "no_proposal"
            elif proposal_kind == "absence":
                if source.state is FieldState.ABSENT:
                    final = source
                    decision = "accepted_absent"
                    reason = "verified_explicit_denial"
                else:
                    final = _uncertain(field, source)
                    decision = "rejected_unproven_absence"
                    reason = "explicit_denial_not_verified"
            elif (
                source.state is FieldState.SUPPORTED
                and _canonical_value(field, proposal) == source.value
            ):
                final = source
                decision = "accepted_supported"
                reason = "verified_value_match"
            else:
                final = _uncertain(field, source)
                decision = "rejected_ungrounded"
                reason = "proposal_value_not_grounded"

            grounded.append(final)
            diagnostics.append(
                {
                    "field": field.value,
                    "raw_proposal": raw_proposal,
                    "proposal_kind": proposal_kind,
                    "decision": decision,
                    "reason": reason,
                }
            )

        output = NanoOutput(
            item_id=item.item_id,
            solver_id=self.solver_id,
            fields=tuple(grounded),
        )
        output.validate_against(item)
        return output, {
            "protocol_version": LEGACY_DIAGNOSTICS_VERSION,
            "raw_summary": raw,
            "fields": diagnostics,
        }

    def infer(self, item: NanoInput) -> NanoOutput:
        output, _diagnostics = self.infer_with_diagnostics(item)
        return output


__all__ = [
    "LEGACY_DIAGNOSTICS_VERSION",
    "LegacySummaryFormatError",
    "LegacySummarySolver",
]
