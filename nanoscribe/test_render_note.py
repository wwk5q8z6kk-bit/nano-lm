"""Tests for verified record → note rendering v0."""

from __future__ import annotations

from nanoscribe.campaign_datasets import campaign_cases, SMOKE_SUITE_REVISION
from nanoscribe.encounter import AssertionState
from nanoscribe.render.encounter_note import render_encounter_note


def test_render_smoke_enc1_includes_evidence() -> None:
    case = campaign_cases(SMOKE_SUITE_REVISION)[0]
    note = render_encounter_note(case.gold)
    assert "enc-1" in note
    assert "neck" in note
    assert 'evidence: "neck"' in note


def test_render_denied_allergy() -> None:
    case = campaign_cases(SMOKE_SUITE_REVISION)[0]
    note = render_encounter_note(case.gold)
    denied = next(atom for atom in case.gold.atoms if atom.atom_type.value == "allergy")
    assert denied.assertion_state is AssertionState.DENIED
    assert "Denies" in note or "allerg" in note.lower()


def test_render_uncertainty_case() -> None:
    case = campaign_cases(SMOKE_SUITE_REVISION)[1]
    note = render_encounter_note(case.gold)
    assert "Uncertain" in note
    assert "pressure" in note


def test_render_family_history_and_symptom() -> None:
    case = campaign_cases(SMOKE_SUITE_REVISION)[2]
    note = render_encounter_note(case.gold)
    assert "## History" in note
    assert "diabetes" in note
    assert "## Symptoms" in note
    assert "tired" in note
