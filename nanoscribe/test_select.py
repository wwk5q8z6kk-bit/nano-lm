# Adversarial contract pins for span extraction and normalization.
# Run: python3 nanoscribe/test_select.py
# Stdlib only — no model, no pytest, no PHI.

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoscribe.encounter import EncounterError, Speaker, assemble_source
from nanoscribe.select import (
    _normalize_surface,
    _span_in_turn,
    copy_span,
    _exact_hits,
    match_count,
    relocate,
    _surface_hits,
    snap_relocate,
    ConstrainedSelector,
)


def expect(code: str, fn) -> None:
    try:
        fn()
    except EncounterError as exc:
        assert exc.code == code, f"expected {code}, got {exc.code}: {exc}"
        return
    raise AssertionError(f"expected EncounterError {code}")


def test_normalize_surface() -> None:
    text = "“Hello” — it’s a \u00a0 test."
    norm, indices = _normalize_surface(text)
    assert norm == '"hello" - it\'s a test.', norm

    # Check length and indices mapping
    assert len(norm) == len(indices)

    # Check empty string
    assert _normalize_surface("")[0] == ""

    # Check only spaces
    assert _normalize_surface("   ")[0] == ""

    # Check multiple spaces collapsing
    norm2, ind2 = _normalize_surface("a  b")
    assert norm2 == "a b", norm2
    assert ind2 == [0, 1, 3]  # 'a', space at 1, 'b' at 3


def _source():
    return assemble_source(
        "src-1",
        (
            (Speaker.CLINICIAN, "What brings you in today?"),
            (Speaker.PATIENT, "My neck has been hurting. And my back."),
            (Speaker.CLINICIAN, "I see. Yes, I see."),
        ),
    )


def test_span_in_turn() -> None:
    src = _source()
    turn0 = src.turns[0]

    # Valid
    span = _span_in_turn(
        src, turn0.start, turn0.end, "What brings you in today?", "ev-1"
    )
    assert span.text == "What brings you in today?"
    assert span.evidence_id == "ev-1"

    # Invalid bounds
    expect("evidence_bounds", lambda: _span_in_turn(src, -1, 5, "What ", "ev"))
    expect(
        "evidence_bounds",
        lambda: _span_in_turn(src, turn0.start, turn0.start, "", "ev"),
    )
    expect(
        "evidence_bounds",
        lambda: _span_in_turn(src, turn0.start, len(src.text) + 1, "...", "ev"),
    )

    # Text mismatch
    expect(
        "evidence_text_mismatch",
        lambda: _span_in_turn(src, turn0.start, turn0.start + 4, "WhaT", "ev"),
    )

    # Cross turn boundaries
    expect(
        "evidence_crosses_turn",
        lambda: _span_in_turn(
            src,
            turn0.start,
            src.turns[1].start + 2,
            "What brings you in today?\nMy",
            "ev",
        ),
    )

    # Text mismatch due to cross turn (not exact match to claimed text if different)
    expect(
        "evidence_text_mismatch",
        lambda: _span_in_turn(
            src,
            turn0.start,
            src.turns[1].start + 2,
            "What brings you in today?\\nMy",
            "ev",
        ),
    )


def test_copy_span() -> None:
    src = _source()
    turn0 = src.turns[0]
    span = copy_span(src, turn0.start, turn0.start + 4, evidence_id="ev-copy")
    assert span.text == "What"

    span2 = copy_span(
        src, turn0.start, turn0.start + 4, evidence_id="ev-copy2", text="What"
    )
    assert span2.text == "What"


def test_exact_hits_and_match_count() -> None:
    src = _source()
    assert match_count(src, "What brings") == 1
    assert match_count(src, "I see.") == 2
    assert match_count(src, "Not present") == 0
    assert match_count(src, "") == 0

    hits = _exact_hits(src, "I see.")
    assert len(hits) == 2


def test_relocate() -> None:
    src = _source()
    # 1 match
    span = relocate(src, "What brings", evidence_id="ev-1")
    assert span is not None
    assert span.text == "What brings"

    # 0 match
    assert relocate(src, "XYZ", evidence_id="ev-xyz") is None

    # >1 match
    assert relocate(src, "I see.", evidence_id="ev-mul") is None


def test_surface_hits() -> None:
    # Punctuation/spacing surface hit
    # Create a source with weird punctuation
    src_weird = assemble_source(
        "src-2",
        ((Speaker.CLINICIAN, "“Hello”  –  test!"),),
    )
    # The normalized version is '"hello" - test!'
    hits = _surface_hits(src_weird, '"Hello" - test!')
    assert len(hits) == 1
    # Check snap_relocate
    span = snap_relocate(src_weird, '"Hello" - test!', evidence_id="ev-snap")
    assert span is not None
    assert span.text == "“Hello”  –  test!"

    assert snap_relocate(src_weird, "Not there", evidence_id="ev") is None

    # Ambiguous snap
    src_ambig = assemble_source("src-ambig", ((Speaker.PATIENT, "A b. A  b."),))
    assert snap_relocate(src_ambig, "A b.", evidence_id="ev") is None


def test_constrained_selector() -> None:
    sel = ConstrainedSelector()
    src = _source()
    turn0 = src.turns[0]

    # copy_span
    span = sel.copy_span(src, turn0.start, turn0.start + 4, evidence_id="ev-sel")
    assert span.text == "What"

    # select_quote (exact)
    span2 = sel.select_quote(src, "What brings", evidence_id="ev-sel2")
    assert span2 is not None
    assert span2.text == "What brings"

    # select_quote (surface)
    src_weird = assemble_source(
        "src-3",
        ((Speaker.CLINICIAN, "“Hello”  –  test!"),),
    )
    span3 = sel.select_quote(src_weird, '"hello" - test!', evidence_id="ev-sel3")
    assert span3 is not None
    assert span3.text == "“Hello”  –  test!"


if __name__ == "__main__":
    test_normalize_surface()
    test_span_in_turn()
    test_copy_span()
    test_exact_hits_and_match_count()
    test_relocate()
    test_surface_hits()
    test_constrained_selector()
