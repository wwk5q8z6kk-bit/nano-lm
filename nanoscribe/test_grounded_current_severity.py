"""A current severity assertion must retain the time of its measurement."""
from dataclasses import replace

import pytest

from nano_ai.contract import EvidenceSpan, FieldName, FieldState, NanoInput
from nanoscribe.check_note import assess_fields, transcript_source
from nanoscribe.compact import compact_solver
from nanoscribe.grounded import build_grounded_english_menu, format_grounded_span_token, grounded_candidate_choices


@pytest.mark.parametrize("reply, quote, accepted", [
    ("Seven out of ten was the number just after the fall.", "Seven out of ten", False),
    ("Eight out of ten was the score at its worst.", "Eight out of ten", False),
    ("Four out of ten was my rating yesterday.", "Four out of ten", False),
    ("Six out of ten was the number before the visit.", "Six out of ten", False),
    ("Seven out of ten was my score when I checked just now.", "Seven out of ten", True),
    ("Six out of ten was the rating when I measured it right now.", "Six out of ten", True),
    ("Five out of ten was my score when I just checked it.", "Five out of ten", True),
    ("Three out of ten was my rating when I rated my pain a moment ago.", "Three out of ten", True),
    ("Seven out of ten was my score when you asked me just now.", "Seven out of ten", True),
    ("Six out of ten was my rating when you just asked me.", "Six out of ten", True),
    ("Five out of ten was my score when you asked me about my pain a moment ago.", "Five out of ten", True),
    ("Seven out of ten was my score when you asked my mother just now.", "Seven out of ten", False),
    ("Seven out of ten was my score when you asked me about her pain just now.", "Seven out of ten", False),
    ("Seven out of ten was my score when you asked me yesterday.", "Seven out of ten", False),
    ("Seven out of ten right now.", "Seven out of ten", True),
    ("Five out of ten at this moment.", "Five out of ten", True),
    ("Three out of ten was my score when I just checked it now.", "Three out of ten", True),
    ("Three out of ten was my score when you just asked me right now.", "Three out of ten", True),
    ("Four out of ten was my score when I just checked it yesterday.", "Four out of ten", False),
    ("Four out of ten was my score when I just checked it last week.", "Four out of ten", False),
    ("Seven out of ten right now for my bus ride.", "Seven out of ten", False),
    ("Seven out of ten was my score when I checked yesterday.", "Seven out of ten", False),
    ("Seven out of ten was my score when I first injured my wrist.", "Seven out of ten", False),
    ("Seven out of ten was my score when I checked before the fall.", "Seven out of ten", False),
    ("Seven out of ten was my score when I read yesterday's chart just now.", "Seven out of ten", False),
    ("Seven out of ten was my score when I checked my daughter's pain just now.", "Seven out of ten", False),
    ("Seven out of ten was my score when I did not check it just now.", "Seven out of ten", False),
    ("Seven out of ten was my score when I would have checked it just now.", "Seven out of ten", False),
    ("Seven out of ten was my score when I checked yesterday, but it is three out of ten now.", "Seven out of ten", False),
    ("Seven out of ten was my score when I checked yesterday, but it is three out of ten now.", "three out of ten", True),
    ("Right now it is three out of ten.", "three out of ten", True),
    ("Right now, two out of ten.", "two out of ten", True),
    ("At the moment it is five out of ten.", "five out of ten", True),
    ("Seven out of ten.", "Seven out of ten", True),
    ("My wrist pain is four out of ten after exercise.", "four out of ten", True),
    ("My bus ride is three out of ten.", "three out of ten", False),
])
def test_current_severity_source_paths(reply, quote, accepted):
    _check_rating("How bad is the pain now?", reply, quote, accepted)


@pytest.mark.parametrize("question, reply, quote, accepted", [
    ("How bad was the pain last Friday?", "7/10 was my score when you just asked me.", "7/10", False),
    ("How bad was the pain?", "7/10 was my score when you just asked me.", "7/10", False),
    ("How would you rate your pain yesterday?", "7/10 was my score when you asked me just now.", "7/10", False),
    ("What was your pain score last month?", "7/10 was my rating when you asked me just now.", "7/10", False),
    ("How bad was the pain last Friday?", "7/10.", "7/10", False),
    ("How bad was the pain last Friday?", "7/10 was my score when you just asked me, but it is 2/10 now.", "7/10", False),
    ("How bad was the pain last Friday?", "7/10 was my score when you just asked me, but it is 2/10 now.", "2/10", True),
    ("What was your pain score last month?", "Right now it is 2/10.", "2/10", True),
    ("How bad was your wrist pain yesterday?", "4/10 was my score when I checked my wrist pain just now.", "4/10", True),
    ("What is your pain score right now?", "4/10 was my score when you asked me just now.", "4/10", True),
    ("How severe is your wrist pain right now?", "4/10 was my score when I checked my wrist pain just now.", "4/10", True),
    ("How bad is the pain now?", "4/10 was my score when I assessed my left knee pain just now.", "4/10", True),
    ("How bad is the pain now?", "4/10 was my score when I checked my wrist pain diary just now.", "4/10", False),
    ("How bad is the pain now?", "4/10 was my score when I checked my sister's wrist pain just now.", "4/10", False),
    ("How bad is the pain now?", "4/10 was my score when I checked my wrist pain yesterday.", "4/10", False),
    ("How bad was your mother's pain last month?", "Right now it is 2/10.", "2/10", False),
    ("How bad was the pain just now?", "7/10 was my score when you just asked me.", "7/10", True),
    ("On a scale from zero to ten, how bad is your pain now?", "7/10 was my score when you just asked me.", "7/10", True),
    ("What was your pain score last month?", "My wrist pain right now is 2/10.", "2/10", True),
    ("What was your pain score last month?", "Now I remember my wrist pain was 7/10.", "7/10", False),
    ("Who drove you today?", "7/10 was my score when you just asked me.", "7/10", False),
])
def test_current_measurement_and_time_requested(question, reply, quote, accepted):
    _check_rating(question, reply, quote, accepted)


_PRESENTING_TIGHTNESS = "Doctor: Why did you schedule this visit?\nPatient: chest tightness"
_DURATION_FOLLOWUP = "\nDoctor: How long did the tightness last?\nPatient: about forty minutes"


@pytest.mark.parametrize("context, question, reply, quote, accepted", [
    (_PRESENTING_TIGHTNESS + _DURATION_FOLLOWUP, "How bad was it?", "severe", "severe", True),
    (_PRESENTING_TIGHTNESS, "How bad was the tightness?", "5/10.", "5/10", True),
    ("Patient: My wrist pain is bothering me.", "How severe was the pain?", "moderate", "moderate", True),
    (_PRESENTING_TIGHTNESS + _DURATION_FOLLOWUP, "How bad was it last Friday?", "severe", "severe", False),
    (_PRESENTING_TIGHTNESS + _DURATION_FOLLOWUP, "How bad was it?", "7/10 was my rating when you just asked me.", "7/10", False),
    (_PRESENTING_TIGHTNESS, "How bad was the headache?", "severe", "severe", False),
    (_PRESENTING_TIGHTNESS + "\nDoctor: How long did the train ride last?\nPatient: about forty minutes",
     "How bad was it?", "severe", "severe", False),
    ("Doctor: What problems did you have last Friday?\nPatient: chest tightness",
     "How bad was it?", "severe", "severe", False),
    ("Doctor: Last Friday, what symptoms did you have?\nPatient: chest tightness",
     "How bad was it?", "severe", "severe", False),
    ("Doctor: What medication did you take?\nPatient: Aspirin.", "How bad was it?", "severe", "severe", False),
])
def test_past_tense_rating_of_presenting_episode(context, question, reply, quote, accepted):
    _check_rating(question, reply, quote, accepted, context=context + "\n")


def _check_rating(question, reply, quote, accepted, *, context=""):
    field = FieldName.SEVERITY
    request = NanoInput("current-rating", f"{context}Doctor: {question}\nPatient: {reply}")
    source = transcript_source(request)
    start = request.transcript.index(quote)
    span = EvidenceSpan(start, start + len(quote), quote)
    token = format_grounded_span_token(source, span)
    raw = f"CC:M|DUR:M|SEV:S[{token}]|MED:M|ALG:M"
    result, _ = compact_solver(lambda _: raw, binder="grounded").infer_grounded(request)
    rating = next(entry for entry in result.fields if entry.field is field)
    assert (rating.state is FieldState.SUPPORTED) == accepted
    supplied = replace(rating, state=FieldState.SUPPORTED, value=quote, evidence=(span,))
    extraction = replace(result, fields=tuple(supplied if entry.field is field else entry for entry in result.fields))
    checked = next(entry for entry in assess_fields(request, extraction, context_policy="grounded") if entry.field is field)
    assert (checked.state is FieldState.SUPPORTED) == accepted
    menu = build_grounded_english_menu(source, field, set())
    assert any(code == "S" and span in spans for code, spans in menu.values()) == accepted
    assert (f":S[{quote}]" in grounded_candidate_choices(request.transcript, field, set())) == accepted
