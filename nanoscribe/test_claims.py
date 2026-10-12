"""Tests for claim decomposition (B3)."""

from __future__ import annotations

from nanoscribe.adapt import ModelCandidate, adapt, candidate_from_span_port_line
from nanoscribe.campaign_datasets import SMOKE_SUITE_REVISION, campaign_cases
from nanoscribe.claims import claim_decomposition_payload, decompose_claims
from nanoscribe.evaluate import PredictedEncounter, SupportRelation, evaluate
from nanoscribe.select import ConstrainedSelector, relocate


def _pred_from_gold(gold, atom_id: str):
    from nanoscribe.evaluate import PredictedAtom

    atom = gold.atom(atom_id)
    spans = tuple(gold.span(evidence_id) for evidence_id in atom.evidence_ids)
    return PredictedAtom(
        atom_id=atom.atom_id,
        atom_type=atom.atom_type,
        raw_value=atom.raw_value,
        assertion_state=atom.assertion_state,
        speaker=atom.speaker,
        experiencer=atom.experiencer,
        temporality=atom.temporality,
        certainty=atom.certainty,
        evidence_ids=atom.evidence_ids,
        spans=spans,
        review_required=atom.review_required,
    )


def test_decompose_claims_matches_gold_atoms() -> None:
    case = campaign_cases(SMOKE_SUITE_REVISION)[0]
    gold = case.gold
    preds = tuple(_pred_from_gold(gold, atom.atom_id) for atom in gold.atoms)
    report = evaluate(gold, PredictedEncounter(atoms=preds))
    rows = decompose_claims(gold, report)
    assert len(rows) == len(gold.atoms)
    assert {row.claim_id for row in rows} == {atom.atom_id for atom in gold.atoms}
    assert all(row.exact_gold_span for row in rows)
    assert all("transport" not in row.failure_layers for row in rows)
    assert all("malformed" not in row.failure_layers for row in rows)


def test_claim_payload_flags_transport_failure() -> None:
    case = campaign_cases(SMOKE_SUITE_REVISION)[0]
    spec = case.atom_specs[0]
    candidate = candidate_from_span_port_line(
        atom_id=spec.atom_id,
        atom_type=spec.atom_type,
        raw_value=spec.raw_value,
        raw_line='STATED: "NECK"',
    )
    predicted = adapt(
        case.model_input,
        ModelCandidate(atoms=(candidate,)),
        selector=_LegacyExactSelector(),
    )
    report = evaluate(case.gold, predicted)
    payload = claim_decomposition_payload(case.gold, report)
    neck = next(c for c in payload["claims"] if c["claim_id"] == spec.atom_id)
    assert "transport" in neck["failure_layers"] or neck["support_relation"] != SupportRelation.DIRECT_EXACT.value


class _LegacyExactSelector(ConstrainedSelector):
    def select_quote(self, source, quote: str, *, evidence_id: str, raw_value: str | None = None):
        del raw_value
        return relocate(source, quote, evidence_id=evidence_id)


def test_v2_selector_recovers_degraded_quote() -> None:
    case = campaign_cases(SMOKE_SUITE_REVISION)[0]
    spec = case.atom_specs[0]
    candidate = candidate_from_span_port_line(
        atom_id=spec.atom_id,
        atom_type=spec.atom_type,
        raw_value=spec.raw_value,
        raw_line='STATED: "NECK"',
    )
    predicted = adapt(
        case.model_input,
        ModelCandidate(atoms=(candidate,)),
        selector=ConstrainedSelector(),
    )
    report = evaluate(case.gold, predicted)
    neck = next(item for item in report.atom_results if item.atom_id == spec.atom_id)
    assert neck.exact_gold_span
