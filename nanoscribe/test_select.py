import pytest
from nanoscribe.select import snap_relocate
from nanoscribe.encounter import Source, Turn, Speaker

def test_snap_relocate_edge_cases():
    turn = Turn(
        source_id="s1",
        turn_id="t1",
        speaker=Speaker.CLINICIAN,
        start=0,
        end=30,
        text="this is a test. this is a test"
    )
    source = Source(
        source_id="s1",
        text="this is a test. this is a test",
        turns=(turn,)
    )

    # Test quote appears multiple times
    res = snap_relocate(source, "this is a test", evidence_id="e1")
    assert res is None, "Should return None if quote appears multiple times"

    # Test quote appears 0 times
    res = snap_relocate(source, "not in text", evidence_id="e2")
    assert res is None, "Should return None if quote appears 0 times"

def test_snap_relocate_success():
    turn = Turn(
        source_id="s1",
        turn_id="t1",
        speaker=Speaker.CLINICIAN,
        start=0,
        end=28,
        text="this is a unique string. no."
    )
    source = Source(
        source_id="s1",
        text="this is a unique string. no.",
        turns=(turn,)
    )

    # Test quote appears exactly 1 time
    res = snap_relocate(source, "unique string", evidence_id="e3")
    assert res is not None, "Should return EvidenceSpan if quote appears exactly 1 time"
    assert res.text == "unique string"
    assert res.start == 10
    assert res.end == 23
    assert res.evidence_id == "e3"
