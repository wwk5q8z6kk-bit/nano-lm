"""Pins for StateDelta — the hinge from per-encounter to longitudinal state."""
from __future__ import annotations

import pytest

from nano.contracts import PatientStateSnapshot, StateDelta, diff_states


def _snap(v, ledger_hash, **kw):
    return PatientStateSnapshot(patient_id="p", evidence_ledger_version=v,
                                ledger_hash=ledger_hash, **kw)


def test_delta_reports_additions_with_their_kind() -> None:
    a = _snap(1, "h1", current_medications=("lisinopril",))
    b = _snap(2, "h2", current_medications=("lisinopril", "metformin"),
              active_conditions=("t2dm",))
    d = diff_states(a, b, evidence_span_ids=("ev_1",))
    assert "medications:metformin" in d.added
    assert "conditions:t2dm" in d.added
    assert d.removed == ()


def test_change_without_evidence_is_rejected() -> None:
    """absence-never-from-silence applies to change, not just to claims."""
    a = _snap(1, "h1")
    b = _snap(2, "h2", active_conditions=("t2dm",))
    with pytest.raises(ValueError, match="cites no evidence"):
        diff_states(a, b)


def test_a_delta_must_move_forward() -> None:
    a, b = _snap(2, "h2"), _snap(1, "h1")
    with pytest.raises(ValueError, match="must move forward"):
        diff_states(a, b, evidence_span_ids=("ev_1",))


def test_identical_snapshots_are_not_a_change() -> None:
    a = _snap(1, "h1", active_conditions=("x",))
    with pytest.raises(ValueError, match="identical snapshots"):
        StateDelta(patient_id="p", from_version=1, to_version=2,
                   from_snapshot_id=a.snapshot_id, to_snapshot_id=a.snapshot_id)


def test_superseded_is_not_removed() -> None:
    """A corrected fact and a fact that stopped being true are different things.

    Collapsing them loses the reason, which is what a reader actually needs.
    """
    a = _snap(1, "h1", current_medications=("metoprolol",))
    b = _snap(2, "h2")
    plain = diff_states(a, b, evidence_span_ids=("ev_1",))
    assert "medications:metoprolol" in plain.removed
    corrected = diff_states(a, b, evidence_span_ids=("ev_1",),
                            superseded=("medications:metoprolol",))
    assert corrected.removed == ()
    assert "medications:metoprolol" in corrected.superseded


def test_resolving_an_uncertainty_is_reported_as_newly_confirmed() -> None:
    a = _snap(1, "h1", uncertainties=("onset date",))
    b = _snap(2, "h2", active_conditions=("x",))
    d = diff_states(a, b, evidence_span_ids=("ev_1",))
    assert "onset date" in d.newly_confirmed


def test_refuses_to_diff_across_patients() -> None:
    a = PatientStateSnapshot(patient_id="p1", evidence_ledger_version=1, ledger_hash="h")
    b = PatientStateSnapshot(patient_id="p2", evidence_ledger_version=2, ledger_hash="h2")
    with pytest.raises(ValueError, match="across patients"):
        diff_states(a, b, evidence_span_ids=("ev_1",))


def test_delta_is_deterministic() -> None:
    a = _snap(1, "h1", current_medications=("x",))
    b = _snap(2, "h2", current_medications=("x", "y"))
    d1 = diff_states(a, b, evidence_span_ids=("ev_1",))
    d2 = diff_states(a, b, evidence_span_ids=("ev_1",))
    assert d1.delta_id == d2.delta_id


def test_summary_counts_every_change_category() -> None:
    a = _snap(1, "h1", current_medications=("x",), uncertainties=("u",))
    b = _snap(2, "h2", current_medications=("y",), active_conditions=("c",))
    d = diff_states(a, b, evidence_span_ids=("ev_1",))
    s = d.summary()
    assert set(s) == {"added", "removed", "modified", "newly_uncertain",
                      "newly_confirmed", "superseded", "conflicting"}
    assert s["added"] == 2 and s["removed"] == 1 and s["newly_confirmed"] == 1
