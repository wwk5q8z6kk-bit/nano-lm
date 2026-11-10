"""Pins for the cognitive-substrate interfaces (§36, §37, §43)."""
from __future__ import annotations

import pytest

from nano.kernel import (
    ArtifactIR, Entity, Identity, MemoryRecord, MemoryScale, SliceStatus, Tool,
    WorkSlice, WorldStateProjection, assert_same_subject,
)


def _id(sub="s1", tenant="default"):
    return Identity(subject_id=sub, kind="subject", tenant=tenant)


# --- Identity: semantic correctness includes identity (§6) ---

def test_identity_requires_a_subject() -> None:
    with pytest.raises(ValueError, match="requires a subject_id"):
        Identity(subject_id="", kind="subject")


def test_cross_subject_join_is_a_hard_failure() -> None:
    with pytest.raises(ValueError, match="cross-subject contamination"):
        assert_same_subject(_id("a"), _id("b"))


def test_cross_tenant_join_is_a_hard_failure() -> None:
    with pytest.raises(ValueError, match="cross-subject contamination"):
        assert_same_subject(_id("a", "t1"), _id("a", "t2"))


def test_same_subject_joins_cleanly() -> None:
    assert_same_subject(_id("a"), _id("a"), _id("a"))


# --- Entity: resolution is a decision that can be wrong ---

def test_entity_without_mentions_is_rejected() -> None:
    with pytest.raises(ValueError, match="invented"):
        Entity(subject=_id(), entity_type="med", canonical_name="x", mention_ids=())


def test_resolution_confidence_is_bounded() -> None:
    with pytest.raises(ValueError, match="\\[0,1\\]"):
        Entity(subject=_id(), entity_type="med", canonical_name="x",
               mention_ids=("m1",), resolution_confidence=1.4)


# --- WorldStateProjection: must state the time it projects ---

def test_projection_must_name_its_time_coordinate() -> None:
    with pytest.raises(ValueError, match="must state the time"):
        WorldStateProjection(subject=_id(), as_of="", ledger_version=1, ledger_hash="h")


# --- Tool: the fabric cannot route without declared cost (§17) ---

def test_tool_with_side_effects_must_declare_permissions() -> None:
    with pytest.raises(ValueError, match="must declare permissions"):
        Tool(name="rm", inputs=(), outputs=(), cost=1.0, latency_s=0.1,
             reliability=1.0, side_effects=("deletes files",))


def test_tool_reliability_is_bounded() -> None:
    with pytest.raises(ValueError, match="reliability"):
        Tool(name="t", inputs=(), outputs=(), cost=1.0, latency_s=0.1, reliability=2.0)


# --- WorkSlice: a unit that cannot stop is a loop (§11) ---

def test_workslice_requires_stop_conditions() -> None:
    with pytest.raises(ValueError, match="cannot stop"):
        WorkSlice(objective="o", subject=_id(), stop_conditions=(), compute_budget=1.0)


def test_workslice_requires_a_positive_budget() -> None:
    with pytest.raises(ValueError, match="compute_budget"):
        WorkSlice(objective="o", subject=_id(), stop_conditions=("done",),
                  compute_budget=0)


def test_exceeding_budget_stops_the_slice() -> None:
    w = WorkSlice(objective="o", subject=_id(), stop_conditions=("done",),
                  compute_budget=10.0)
    w.spend(4.0)
    assert w.status is SliceStatus.OPEN
    w.spend(7.0)
    assert w.status is SliceStatus.STOPPED
    assert w.exhausted


def test_child_slice_cannot_exceed_parent_remaining_budget() -> None:
    """Nested cognition must not escape the parent's resource envelope."""
    w = WorkSlice(objective="parent", subject=_id(), stop_conditions=("done",),
                  compute_budget=10.0)
    w.spend(8.0)
    with pytest.raises(ValueError, match="exceeds parent remaining"):
        w.spawn("child", budget=5.0, stop_conditions=("done",))
    child = w.spawn("child", budget=2.0, stop_conditions=("done",))
    assert child in w.children


# --- ArtifactIR: plan before rendering (§19) ---

def test_artifact_ir_requires_claims_and_evidence() -> None:
    with pytest.raises(ValueError, match="no required claims"):
        ArtifactIR(subject=_id(), purpose="p", audience="clinician",
                   required_claims=(), selected_evidence=("e1",))
    with pytest.raises(ValueError, match="select evidence before rendering"):
        ArtifactIR(subject=_id(), purpose="p", audience="clinician",
                   required_claims=("c1",), selected_evidence=())


def test_same_state_compiles_to_different_audiences() -> None:
    """One world state, many consistent artifacts -- not many summarisers."""
    a = ArtifactIR(subject=_id(), purpose="handoff", audience="clinician",
                   required_claims=("c1",), selected_evidence=("e1",))
    b = ArtifactIR(subject=_id(), purpose="handoff", audience="patient",
                   required_claims=("c1",), selected_evidence=("e1",))
    assert a.ir_id != b.ir_id
    assert a.required_claims == b.required_claims


# --- MemoryRecord: compression must preserve reconstructability (§10) ---

def test_compressed_memory_must_retain_its_source() -> None:
    with pytest.raises(ValueError, match="lossy rewrite"):
        MemoryRecord(subject=_id(), scale=MemoryScale.EPISODIC,
                     content="summary", compressed=True)


def test_uncompressed_memory_needs_no_source() -> None:
    m = MemoryRecord(subject=_id(), scale=MemoryScale.WORKING, content="note")
    assert m.record_id
