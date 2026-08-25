"""Smoke tests for verified record → note rendering v0."""

from __future__ import annotations

from nanoscribe.campaign_datasets import campaign_cases, SMOKE_SUITE_REVISION
from nanoscribe.encounter import (
    AssertionState,
    AtomType,
    ClinicalAtom,
)
from nanoscribe.evaluate import SupportRelation
from nanoscribe.render import (
    ClaimFlag,
    render_encounter_note,
    render_verified_note,
    verify_claim,
    verify_record,
)


def _smoke_cases():
    return campaign_cases(SMOKE_SUITE_REVISION)


def test_render_smoke_three_contract_encounters() -> None:
    cases = _smoke_cases()
    assert len(cases) == 3
    for case in cases:
        result = render_verified_note(case.gold)
        note = result.note
        assert case.encounter_id in note
        assert "# Encounter note" in note
        for atom in case.gold.atoms:
            assert f"[{atom.atom_id}]" in note
        assert result.unsupported_count == 0


def test_render_enc1_includes_evidence_and_claim_ids() -> None:
    case = _smoke_cases()[0]
    note = render_encounter_note(case.gold)
    assert "enc-1" in note
    assert "[atom-neck]" in note
    assert "neck" in note
    assert 'evidence: "neck"' in note


def test_render_denied_allergy() -> None:
    case = _smoke_cases()[0]
    note = render_encounter_note(case.gold)
    denied = next(atom for atom in case.gold.atoms if atom.atom_type is AtomType.ALLERGY)
    assert denied.assertion_state is AssertionState.DENIED
    assert "[atom-alg]" in note
    assert "Denies" in note


def test_render_uncertainty_case() -> None:
    case = _smoke_cases()[1]
    note = render_encounter_note(case.gold)
    assert "[atom-chest]" in note
    assert "Uncertain" in note
    assert "pressure" in note


def test_render_family_history_and_symptom() -> None:
    case = _smoke_cases()[2]
    note = render_encounter_note(case.gold)
    assert "## History" in note
    assert "[atom-fh]" in note
    assert "diabetes" in note
    assert "## Symptoms" in note
    assert "[atom-tired]" in note
    assert "tired" in note


def test_verifier_flags_unsupported_claim() -> None:
    case = _smoke_cases()[0]
    gold = case.gold
    template = gold.atoms[0]
    bad_atom = ClinicalAtom(
        atom_id="atom-fever",
        atom_type=AtomType.SYMPTOM,
        raw_value="fever",
        assertion_state=AssertionState.ASSERTED,
        speaker=template.speaker,
        experiencer=template.experiencer,
        temporality=template.temporality,
        certainty=template.certainty,
        evidence_ids=template.evidence_ids,
    )
    assert verify_claim(gold, bad_atom) is SupportRelation.UNSUPPORTED
    flags = verify_record(gold)
    assert flags == ()
    flagged = (
        ClaimFlag(
            atom_id="atom-fever",
            relation=SupportRelation.UNSUPPORTED,
            message="unsupported for 'fever'",
        ),
    )
    result = render_verified_note(gold)
    assert result.unsupported_count == 0
    assert "⚠ UNSUPPORTED" not in result.note
    assert flagged[0].relation is SupportRelation.UNSUPPORTED
