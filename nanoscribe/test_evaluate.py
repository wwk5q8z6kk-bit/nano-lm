import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoscribe.encounter import EvidenceSpan, Speaker
from nanoscribe.evaluate import span_character_f1

def test_span_character_f1_empty():
    assert span_character_f1([], []) == 0.0

def test_span_character_f1_empty_gold():
    pred = [
        EvidenceSpan(
            evidence_id="ev1",
            source_id="src1",
            turn_id="turn1",
            speaker=Speaker.PATIENT,
            start=0,
            end=5,
            text="hello"
        )
    ]
    assert span_character_f1([], pred) == 0.0

def test_span_character_f1_empty_pred():
    gold = [
        EvidenceSpan(
            evidence_id="ev1",
            source_id="src1",
            turn_id="turn1",
            speaker=Speaker.PATIENT,
            start=0,
            end=5,
            text="hello"
        )
    ]
    assert span_character_f1(gold, []) == 0.0

def test_span_character_f1_exact_match():
    gold = [
        EvidenceSpan(
            evidence_id="ev1",
            source_id="src1",
            turn_id="turn1",
            speaker=Speaker.PATIENT,
            start=0,
            end=5,
            text="hello"
        )
    ]
    pred = [
        EvidenceSpan(
            evidence_id="ev1",
            source_id="src1",
            turn_id="turn1",
            speaker=Speaker.PATIENT,
            start=0,
            end=5,
            text="hello"
        )
    ]
    assert span_character_f1(gold, pred) == 1.0

def test_span_character_f1_partial_overlap():
    # gold is 0 to 5 (5 chars)
    # pred is 3 to 8 (5 chars)
    # overlap is 3 to 5 (2 chars)
    # precision = 2/5 = 0.4
    # recall = 2/5 = 0.4
    # F1 = 2 * 0.4 * 0.4 / 0.8 = 0.4
    gold = [
        EvidenceSpan(
            evidence_id="ev1",
            source_id="src1",
            turn_id="turn1",
            speaker=Speaker.PATIENT,
            start=0,
            end=5,
            text="hello"
        )
    ]
    pred = [
        EvidenceSpan(
            evidence_id="ev2",
            source_id="src1",
            turn_id="turn1",
            speaker=Speaker.PATIENT,
            start=3,
            end=8,
            text="lo wo"
        )
    ]
    assert abs(span_character_f1(gold, pred) - 0.4) < 1e-6

def test_span_character_f1_no_overlap_offsets():
    gold = [
        EvidenceSpan(
            evidence_id="ev1",
            source_id="src1",
            turn_id="turn1",
            speaker=Speaker.PATIENT,
            start=0,
            end=5,
            text="hello"
        )
    ]
    pred = [
        EvidenceSpan(
            evidence_id="ev2",
            source_id="src1",
            turn_id="turn1",
            speaker=Speaker.PATIENT,
            start=5,
            end=10,
            text="world"
        )
    ]
    assert span_character_f1(gold, pred) == 0.0

def test_span_character_f1_no_overlap_source_id():
    gold = [
        EvidenceSpan(
            evidence_id="ev1",
            source_id="src1",
            turn_id="turn1",
            speaker=Speaker.PATIENT,
            start=0,
            end=5,
            text="hello"
        )
    ]
    pred = [
        EvidenceSpan(
            evidence_id="ev2",
            source_id="src2",
            turn_id="turn1",
            speaker=Speaker.PATIENT,
            start=0,
            end=5,
            text="hello"
        )
    ]
    assert span_character_f1(gold, pred) == 0.0

def test_span_character_f1_multiple_spans():
    gold = [
        EvidenceSpan(
            evidence_id="ev1",
            source_id="src1",
            turn_id="turn1",
            speaker=Speaker.PATIENT,
            start=0,
            end=5,
            text="hello"
        ),
        EvidenceSpan(
            evidence_id="ev2",
            source_id="src1",
            turn_id="turn1",
            speaker=Speaker.PATIENT,
            start=10,
            end=15,
            text="world"
        )
    ]
    pred = [
        EvidenceSpan(
            evidence_id="ev3",
            source_id="src1",
            turn_id="turn1",
            speaker=Speaker.PATIENT,
            start=0,
            end=15,
            text="hello     world"
        )
    ]
    # Overlap is 10 chars
    # Precision = 10 / 15
    # Recall = 10 / 10 = 1.0
    # F1 = 2 * (10/15) * 1 / (10/15 + 1) = (20/15) / (25/15) = 20/25 = 0.8
    assert abs(span_character_f1(gold, pred) - 0.8) < 1e-6
