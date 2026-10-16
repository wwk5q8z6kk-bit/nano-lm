import pytest
from nanoscribe.select import match_count
from nanoscribe.encounter import Source, Turn, Speaker

def test_match_count():
    text = "Hello world! This is a test. Hello again, world!"
    turns = (
        Turn(turn_id="t1", source_id="s1", speaker=Speaker.PATIENT, start=0, end=12, text="Hello world!"),
        Turn(turn_id="t2", source_id="s1", speaker=Speaker.CLINICIAN, start=13, end=28, text="This is a test."),
        Turn(turn_id="t3", source_id="s1", speaker=Speaker.PATIENT, start=29, end=48, text="Hello again, world!"),
    )
    source = Source(source_id="s1", text=text, turns=turns)

    assert match_count(source, "Hello") == 2
    assert match_count(source, "world") == 2
    assert match_count(source, "test") == 1
    assert match_count(source, "missing") == 0
    assert match_count(source, "Hello world!") == 1
    assert match_count(source, "") == 0

def test_match_count_overlap():
    text = "aaaaa"
    turns = (
        Turn(turn_id="t1", source_id="s1", speaker=Speaker.PATIENT, start=0, end=5, text="aaaaa"),
    )
    source = Source(source_id="s1", text=text, turns=turns)
    # The current implementation of `_exact_hits` uses `source.text.find(quote, found + 1, turn.end)`
    # Which finds overlapping matches.
    assert match_count(source, "aa") == 4

def test_match_count_cross_turn_boundary():
    # Matches should NOT cross turn boundaries
    text = "Hello world! How are you?"
    turns = (
        Turn(turn_id="t1", source_id="s1", speaker=Speaker.PATIENT, start=0, end=12, text="Hello world!"),
        Turn(turn_id="t2", source_id="s1", speaker=Speaker.CLINICIAN, start=13, end=25, text="How are you?"),
    )
    source = Source(source_id="s1", text=text, turns=turns)
    assert match_count(source, "world! How") == 0
    assert match_count(source, "world!") == 1
    assert match_count(source, "How") == 1

def test_match_count_multiple_exact_matches_same_turn():
    text = "test test test"
    turns = (
        Turn(turn_id="t1", source_id="s1", speaker=Speaker.PATIENT, start=0, end=14, text="test test test"),
    )
    source = Source(source_id="s1", text=text, turns=turns)
    assert match_count(source, "test") == 3

def test_match_count_multiple_turns_empty_quotes():
    text = "abc"
    turns = (
        Turn(turn_id="t1", source_id="s1", speaker=Speaker.PATIENT, start=0, end=3, text="abc"),
    )
    source = Source(source_id="s1", text=text, turns=turns)
    assert match_count(source, "") == 0
