"""NANO-CLIN-001 invariants (D-NANO-2026-08-25 §9).

These are the required invariants stated as executable tests. Each one is a
property the pipeline must have, not a score it should reach — the distinction
that `papers/METHODS_ADVERSARIAL_INSTRUMENTATION.md` argues is what separates a
control from a decoration.
"""

from __future__ import annotations

import pytest

from nano.contracts import (
    ClinicalAssertion, EpistemicStatus, EvidenceLedger, EvidenceSpanV2, Locator,
    Modality, SourceArtifact, TemporalExtent, TimePrecision,
)
from nano.fixtures import ALL_FIXTURES, BASIC, CONFLICTING, UNCERTAIN
from nano.pipeline import baseline_a, candidate_b


# --------------------------------------------------------------------------
# Contract-level invariants
# --------------------------------------------------------------------------

def test_assertion_without_evidence_is_rejected() -> None:
    """absence-never-from-silence, enforced in the type."""
    with pytest.raises(ValueError, match="needs evidence"):
        ClinicalAssertion(
            patient_id="p", subject="clinician", predicate="states", obj="x",
            original_wording="x", epistemic_status=EpistemicStatus.CLINICIAN_ASSERTED)


def test_absence_statuses_may_stand_without_evidence() -> None:
    """NOT_FOUND asserts that nothing was found; it cannot cite a span."""
    a = ClinicalAssertion(
        patient_id="p", subject="record", predicate="lacks", obj="colonoscopy",
        original_wording="no colonoscopy in record",
        epistemic_status=EpistemicStatus.NOT_FOUND)
    assert a.assertion_id


def test_approximate_time_cannot_carry_an_exact_date() -> None:
    """§6: do not invent exact dates from approximate language."""
    with pytest.raises(ValueError, match="refusing to manufacture precision"):
        TemporalExtent(precision=TimePrecision.APPROXIMATE, event_time="2021-06-01")


def test_locator_requires_exactly_one_modality_family() -> None:
    with pytest.raises(ValueError, match="exactly one family"):
        Locator(start=0, end=5, row=2)
    with pytest.raises(ValueError, match="exactly one family"):
        Locator()


def test_evidence_span_must_be_patient_scoped() -> None:
    """No unscoped evidence — the precondition for no cross-patient mixing."""
    src = SourceArtifact(patient_id="p", modality=Modality.TEXT,
                         document_type="t", content="hello")
    with pytest.raises(ValueError, match="patient-scoped"):
        EvidenceSpanV2(source_id=src.source_id, patient_id="", modality=Modality.TEXT,
                       locator=Locator(start=0, end=3), verbatim="hel")


# --------------------------------------------------------------------------
# Pipeline invariants
# --------------------------------------------------------------------------

@pytest.mark.parametrize("fx", ALL_FIXTURES, ids=lambda f: f.fixture_id)
def test_every_content_claim_cites_evidence(fx) -> None:
    """Content-asserting claims must cite a span. Absence claims are exempt by
    construction: there is no span for something that is not there."""
    r = candidate_b(fx)
    absence = {EpistemicStatus.NOT_FOUND.value, EpistemicStatus.UNAVAILABLE.value}
    content = [c for c in r["claims"] if c.get("epistemic_status") not in absence]
    assert content, "fixture produced no content claims"
    uncited = [c for c in content if not c["evidence_span_ids"]]
    assert not uncited, f"uncited content claims: {uncited}"


@pytest.mark.parametrize("fx", ALL_FIXTURES, ids=lambda f: f.fixture_id)
def test_original_wording_is_preserved(fx) -> None:
    r = candidate_b(fx)
    for a in r["assertions"]:
        assert a.original_wording
        assert a.original_wording in fx.transcript


@pytest.mark.parametrize("fx", ALL_FIXTURES, ids=lambda f: f.fixture_id)
def test_patient_report_is_not_promoted_to_clinician_confirmation(fx) -> None:
    """A patient statement must never be rendered as a clinician assessment."""
    r = candidate_b(fx)
    for a in r["assertions"]:
        if a.subject == "patient":
            assert a.epistemic_status != EpistemicStatus.CLINICIAN_ASSERTED
    for line in r["note"].split("\n"):
        if line.startswith("Clinician assessment:"):
            assert "patient:" not in line.lower()


@pytest.mark.parametrize("fx", ALL_FIXTURES, ids=lambda f: f.fixture_id)
def test_inference_is_never_rendered_as_documented(fx) -> None:
    r = candidate_b(fx)
    for c in r["claims"]:
        if c.get("epistemic_status") in {EpistemicStatus.INFERRED.value,
                                         EpistemicStatus.RECONSTRUCTED.value}:
            assert not c["text"].startswith("Documented:")


def test_conflicts_are_surfaced_and_never_silently_resolved() -> None:
    r = candidate_b(CONFLICTING)
    assert len(r["conflicts"]) == len(CONFLICTING.gold_conflicts)
    for c in r["conflicts"]:
        assert c.resolution_status == "unresolved"
        assert not c.human_disposition
    assert "CONFLICT (unresolved)" in r["note"]


def test_sequential_dates_are_not_reported_as_a_conflict() -> None:
    """Regression: 2019 metoprolol start and 2021 stop are a chronology, not a
    disagreement. An earlier detector flagged any two distinct years."""
    r = candidate_b(UNCERTAIN)
    assert r["conflicts"] == [], f"false-positive conflict: {r['conflicts']}"


def test_not_found_is_distinguished_from_absent() -> None:
    r = candidate_b(UNCERTAIN)
    assert len(r["gaps"]) == len(UNCERTAIN.gold_gaps)
    assert "NOT FOUND IN RECORD (not equivalent to absent)" in r["note"]
    for forbidden in UNCERTAIN.must_not_assert:
        assert forbidden.lower() not in r["note"].lower()


@pytest.mark.parametrize("fx", ALL_FIXTURES, ids=lambda f: f.fixture_id)
def test_forbidden_assertions_never_appear(fx) -> None:
    r = candidate_b(fx)
    for forbidden in fx.must_not_assert:
        assert forbidden.lower() not in r["note"].lower()


def test_negation_is_preserved() -> None:
    """'No fever' must not become a finding of fever."""
    r = candidate_b(BASIC)
    denied = [a for a in r["assertions"] if a.negated]
    assert any("fever" in a.original_wording.lower() for a in denied)
    assert "denies" in r["note"].lower()


@pytest.mark.parametrize("fx", ALL_FIXTURES, ids=lambda f: f.fixture_id)
def test_reprocessing_is_deterministic(fx) -> None:
    """Frozen inputs must produce equivalent structured outputs."""
    a, b = candidate_b(fx), candidate_b(fx)
    assert a["state"].snapshot_id == b["state"].snapshot_id
    assert a["ledger"].ledger_hash() == b["ledger"].ledger_hash()
    assert [x.assertion_id for x in a["assertions"]] == \
           [x.assertion_id for x in b["assertions"]]


@pytest.mark.parametrize("fx", ALL_FIXTURES, ids=lambda f: f.fixture_id)
def test_state_is_a_rebuildable_projection(fx) -> None:
    """State must be derivable from the ledger, not stored independently."""
    r = candidate_b(fx)
    assert r["state"].ledger_hash == r["ledger"].ledger_hash()
    assert r["state"].evidence_ledger_version == r["ledger"].version


def test_new_evidence_appends_and_does_not_overwrite() -> None:
    """§7: do not mutate history in place."""
    led = EvidenceLedger(patient_id="p")
    src = SourceArtifact(patient_id="p", modality=Modality.TEXT,
                         document_type="t", content="first")
    led.append(sources=[src])
    v1, h1 = led.version, led.ledger_hash()
    src2 = SourceArtifact(patient_id="p", modality=Modality.TEXT,
                          document_type="t", content="second")
    led.append(sources=[src2])
    assert led.version > v1
    assert led.ledger_hash() != h1
    assert src in led.sources, "earlier evidence was dropped"


@pytest.mark.parametrize("fx", ALL_FIXTURES, ids=lambda f: f.fixture_id)
def test_candidate_beats_baseline_on_provenance(fx) -> None:
    """The measured claim of the experiment, asserted as a test."""
    _, a_claims = baseline_a(fx)
    r = candidate_b(fx)
    cov = lambda cs: sum(1 for c in cs if c["evidence_span_ids"]) / max(1, len(cs))
    assert cov(a_claims) == 0.0
    assert cov(r["claims"]) > cov(a_claims)


@pytest.mark.parametrize("fx", ALL_FIXTURES, ids=lambda f: f.fixture_id)
def test_no_cross_patient_contamination(fx) -> None:
    r = candidate_b(fx)
    for coll in ("spans", "assertions", "events", "conflicts", "gaps"):
        for item in r[coll]:
            assert item.patient_id == fx.patient_id
