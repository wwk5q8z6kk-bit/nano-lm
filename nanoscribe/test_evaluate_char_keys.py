import pytest
from nanoscribe.evaluate import char_keys
from nanoscribe.encounter import EvidenceSpan, Speaker

def test_char_keys_empty():
    assert char_keys([]) == frozenset()

def test_char_keys_single_span():
    span = EvidenceSpan(
        evidence_id="ev1",
        source_id="src1",
        turn_id="turn1",
        speaker=Speaker.PATIENT,
        start=5,
        end=10,
        text="hello"
    )
    result = char_keys([span])
    expected = frozenset({("src1", 5), ("src1", 6), ("src1", 7), ("src1", 8), ("src1", 9)})
    assert result == expected

def test_char_keys_multiple_spans_same_source():
    span1 = EvidenceSpan(
        evidence_id="ev1",
        source_id="src1",
        turn_id="turn1",
        speaker=Speaker.PATIENT,
        start=0,
        end=2,
        text="hi"
    )
    span2 = EvidenceSpan(
        evidence_id="ev2",
        source_id="src1",
        turn_id="turn2",
        speaker=Speaker.CLINICIAN,
        start=4,
        end=6,
        text="ok"
    )
    result = char_keys([span1, span2])
    expected = frozenset({("src1", 0), ("src1", 1), ("src1", 4), ("src1", 5)})
    assert result == expected

def test_char_keys_multiple_spans_different_sources():
    span1 = EvidenceSpan(
        evidence_id="ev1",
        source_id="src1",
        turn_id="turn1",
        speaker=Speaker.PATIENT,
        start=0,
        end=2,
        text="hi"
    )
    span2 = EvidenceSpan(
        evidence_id="ev2",
        source_id="src2",
        turn_id="turn2",
        speaker=Speaker.CLINICIAN,
        start=0,
        end=2,
        text="ok"
    )
    result = char_keys([span1, span2])
    expected = frozenset({("src1", 0), ("src1", 1), ("src2", 0), ("src2", 1)})
    assert result == expected

def test_char_keys_overlapping_spans():
    span1 = EvidenceSpan(
        evidence_id="ev1",
        source_id="src1",
        turn_id="turn1",
        speaker=Speaker.PATIENT,
        start=0,
        end=3,
        text="abc"
    )
    span2 = EvidenceSpan(
        evidence_id="ev2",
        source_id="src1",
        turn_id="turn2",
        speaker=Speaker.CLINICIAN,
        start=2,
        end=5,
        text="cde"
    )
    result = char_keys([span1, span2])
    # Overlapping indices 2 should only appear once
    expected = frozenset({("src1", 0), ("src1", 1), ("src1", 2), ("src1", 3), ("src1", 4)})
    assert result == expected

def test_char_keys_invalid_spans():
    span = EvidenceSpan(
        evidence_id="ev1",
        source_id="src1",
        turn_id="turn1",
        speaker=Speaker.PATIENT,
        start=0,
        end=2,
        text="hi"
    )
    result = char_keys([span, "not_a_span", None, 123])
    expected = frozenset({("src1", 0), ("src1", 1)})
    assert result == expected
