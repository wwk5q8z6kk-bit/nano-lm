# Tests for select.py
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoscribe.encounter import EncounterError, Source, Speaker, Turn, assemble_source
from nanoscribe.select import (
    ConstrainedSelector,
    _exact_hits,
    _normalize_surface,
    _span_in_turn,
    _surface_hits,
    copy_span,
    match_count,
    relocate,
    snap_relocate,
)

def _source() -> Source:
    return assemble_source(
        "src-1",
        (
            (Speaker.CLINICIAN, "What brings you in today?"),
            (Speaker.PATIENT, "My neck has been hurting."),
            (Speaker.CLINICIAN, "How long?"),
            (Speaker.PATIENT, "About 6 days."),
            (Speaker.PATIENT, "My mother had breast cancer."),
            (Speaker.CLINICIAN, "I think this is cervical strain."),
            (Speaker.CLINICIAN, "Start ibuprofen and follow up in a week."),
            (Speaker.PATIENT, "I am not sure if I take anything."),
            (Speaker.PATIENT, "No allergies."),
            (Speaker.PATIENT, "I used to have migraines years ago."),
        ),
    )

def expect(code: str, fn) -> None:
    try:
        fn()
    except EncounterError as exc:
        assert exc.code == code, f"expected {code}, got {exc.code}: {exc}"
        return
    raise AssertionError(f"expected EncounterError {code}")

def test_normalize_surface() -> None:
    text = "Hello  World! “Smart Quotes”"
    norm, index_map = _normalize_surface(text)
    assert norm == 'hello world! "smart quotes"'
    # Check index_map length
    assert len(index_map) == len(norm)
    # Check first letter
    assert norm[0] == 'h'
    assert index_map[0] == 0
    # Check spaces are collapsed
    # 'Hello  World!' -> 'hello world!'
    assert norm.startswith('hello world!')

def test_span_in_turn_valid() -> None:
    source = _source()
    turn = source.turns[1] # "My neck has been hurting."
    text = "neck has been"
    start = source.text.find(text)
    end = start + len(text)

    span = _span_in_turn(source, start, end, text, "ev-1")
    assert span.text == text
    assert span.evidence_id == "ev-1"
    assert span.speaker == Speaker.PATIENT
    assert span.turn_id == turn.turn_id

def test_span_in_turn_out_of_bounds() -> None:
    source = _source()
    expect("evidence_bounds", lambda: _span_in_turn(source, -1, 5, "Hello", "ev-1"))
    expect("evidence_bounds", lambda: _span_in_turn(source, 5, 4, "Hello", "ev-1"))
    expect("evidence_bounds", lambda: _span_in_turn(source, len(source.text), len(source.text)+5, "Hello", "ev-1"))

def test_span_in_turn_text_mismatch() -> None:
    source = _source()
    text = "neck has been"
    start = source.text.find(text)
    end = start + len(text)
    expect("evidence_text_mismatch", lambda: _span_in_turn(source, start, end, "wrong text", "ev-1"))

def test_span_in_turn_crosses_turn() -> None:
    source = _source()
    text = "today?\nMy neck"
    start = source.text.find(text)
    end = start + len(text)
    # the text matches, but it crosses turns!
    expect("evidence_crosses_turn", lambda: _span_in_turn(source, start, end, text, "ev-1"))

def test_copy_span() -> None:
    source = _source()
    text = "breast cancer"
    start = source.text.find(text)
    end = start + len(text)

    span1 = copy_span(source, start, end, evidence_id="ev-1")
    assert span1.text == text

    span2 = copy_span(source, start, end, evidence_id="ev-1", text=text)
    assert span2.text == text

def test_exact_hits() -> None:
    source = _source()
    assert len(_exact_hits(source, "")) == 0
    assert len(_exact_hits(source, "neck has been")) == 1

    # Add a repetitive test string
    source2 = assemble_source("src-2", ((Speaker.PATIENT, "test test test"),))
    hits = _exact_hits(source2, "test")
    assert len(hits) == 3

def test_match_count() -> None:
    source = assemble_source("src-2", ((Speaker.PATIENT, "test test test"),))
    assert match_count(source, "test") == 3

def test_relocate() -> None:
    source = _source()
    span = relocate(source, "neck has been", evidence_id="ev-1")
    assert span is not None
    assert span.text == "neck has been"

    # ambiguous quote
    source2 = assemble_source("src-2", ((Speaker.PATIENT, "test test test"),))
    assert relocate(source2, "test", evidence_id="ev-1") is None

    # not found
    assert relocate(source, "not in source", evidence_id="ev-1") is None

def test_surface_hits() -> None:
    source = _source()
    # "My neck has been hurting." -> "my neck   has been hurting." (search term has extra space)
    # The normalizer will match them
    hits = _surface_hits(source, "my neck   has been")
    assert len(hits) == 1
    start, end = hits[0]
    assert source.text[start:end] == "My neck has been"

def test_snap_relocate() -> None:
    source = _source()
    span = snap_relocate(source, "my neck   has been", evidence_id="ev-1")
    assert span is not None
    assert span.text == "My neck has been"

    # multiple hits
    source2 = assemble_source("src-2", ((Speaker.PATIENT, "test test test"),))
    assert snap_relocate(source2, "test", evidence_id="ev-1") is None

def test_constrained_selector() -> None:
    selector = ConstrainedSelector()
    source = _source()

    # test copy_span
    start = source.text.find("breast cancer")
    end = start + len("breast cancer")
    span1 = selector.copy_span(source, start, end, evidence_id="ev-1")
    assert span1.text == "breast cancer"

    # test select_quote exact
    span2 = selector.select_quote(source, "breast cancer", evidence_id="ev-2")
    assert span2 is not None
    assert span2.text == "breast cancer"

    # test select_quote surface
    span3 = selector.select_quote(source, " breast cancer ", evidence_id="ev-3")
    assert span3 is not None
    assert span3.text == "breast cancer"

if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in fns:
        fn()
        print(f"  PASS {name}")
    print(f"select.py pins: {len(fns)}/{len(fns)} PASS")
