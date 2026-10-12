"""Claim decomposition — map atoms + eval results into reviewable claim records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nanoscribe.decompose import classify_report
from nanoscribe.encounter import EncounterRecord
from nanoscribe.evaluate import EvalReport, SupportRelation, atom_result


@dataclass(frozen=True, slots=True)
class ClaimRecord:
    claim_id: str
    atom_type: str
    raw_value: str
    assertion_state: str
    evidence_ids: tuple[str, ...]
    support_relation: str | None
    exact_gold_span: bool
    review_required: bool
    failure_layers: tuple[str, ...]


def _atom_layers(item) -> tuple[str, ...]:
    layers: list[str] = []
    if item.invalid_span or item.wrong_source or item.wrong_mention:
        layers.append("transport")
    if item.support_relation in {
        SupportRelation.UNSUPPORTED,
        SupportRelation.CONTRADICTED,
        SupportRelation.REVIEW_REQUIRED,
    }:
        layers.append("support")
    if not item.assertion_state_correct and not item.omitted:
        layers.append("state")
    if item.omitted or item.abstained:
        layers.append("abstention")
    if item.spurious_atom:
        layers.append("commission")
    if item.malformed or item.critical_error:
        layers.append("malformed")
    return tuple(layers)


def decompose_claims(record: EncounterRecord, report: EvalReport) -> tuple[ClaimRecord, ...]:
    """One claim row per gold atom with per-atom failure layering."""
    rows: list[ClaimRecord] = []
    for atom in record.atoms:
        item = atom_result(report, atom.atom_id)
        relation = item.support_relation.value if item.support_relation else None
        rows.append(
            ClaimRecord(
                claim_id=atom.atom_id,
                atom_type=atom.atom_type.value,
                raw_value=atom.raw_value,
                assertion_state=atom.assertion_state.value,
                evidence_ids=atom.evidence_ids,
                support_relation=relation,
                exact_gold_span=bool(item.exact_gold_span),
                review_required=atom.review_required or relation == SupportRelation.REVIEW_REQUIRED.value,
                failure_layers=_atom_layers(item),
            )
        )
    return tuple(rows)


def claim_decomposition_payload(record: EncounterRecord, report: EvalReport) -> dict[str, Any]:
    """JSON-serializable bundle for dashboards and note rendering."""
    summary = classify_report(report)
    claims = [
        {
            "claim_id": row.claim_id,
            "atom_type": row.atom_type,
            "raw_value": row.raw_value,
            "assertion_state": row.assertion_state,
            "evidence_ids": list(row.evidence_ids),
            "support_relation": row.support_relation,
            "exact_gold_span": row.exact_gold_span,
            "review_required": row.review_required,
            "failure_layers": list(row.failure_layers),
        }
        for row in decompose_claims(record, report)
    ]
    return {
        "encounter_id": record.encounter_id,
        "claims": claims,
        "report_summary": summary,
    }
