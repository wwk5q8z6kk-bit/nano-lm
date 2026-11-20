"""Interaction checks for complete quantities, medication intervals and shared citations."""
import pytest

from nano_ai.contract import FieldName, NanoInput
from nanoscribe.check_note import transcript_source
from nanoscribe.compact import compact_solver
from nanoscribe.grounded import build_grounded_english_menu, grounded_candidate_choices
from nanoscribe.scribe import scribe


def draft(source, fields):
    raw = "|".join(label + ":" + fields.get(label, "M") for label in ("CC", "DUR", "SEV", "MED", "ALG"))
    result = scribe(source, solver=compact_solver(lambda _: raw, binder="grounded"), context_policy="grounded")
    assert result["verification"]["passed"]
    return {claim["field"]: claim for claim in result["note"]["claims"]}


@pytest.mark.parametrize("full,short", [
    ("twenty-four days", "four days"), ("sixty five minutes", "five minutes"),
    ("one hundred and six hours", "six hours"), ("half an hour", "an hour"),
    ("half a day", "a day"), ("one and a half weeks", "half weeks"),
    ("three quarters of an hour", "an hour"), ("a quarter of a day", "a day"),
    ("two and three quarter hours", "three quarter hours"),
    ("2 and a half weeks", "a half weeks"), ("an hour and a half", "an hour"),
    ("two days and three hours", "three hours"), ("two days, six hours", "two days"),
])
def test_complete_spoken_duration_across_free_and_constrained_routes(full, short):
    source = f"Patient: It began {full} ago."
    menu = grounded_candidate_choices(source, FieldName.DURATION, set())
    assert f":S[{full}]" in menu
    assert f":S[{short}]" not in menu
    assert draft(source, {"DUR": f"S[{full}]"})["duration"]["value"] == full
    assert draft(source, {"DUR": f"S[{short}]"})["duration"]["state"] == "uncertain"


@pytest.mark.parametrize("quantity,fragment", [
    ("two to four days", "four days"), ("between three and five days", "five days"),
    ("one point five days", "five days"), ("three or four days", "four days"),
    ("minus three days", "three days"), ("point five days", "five days"),
])
def test_unsupported_spoken_amount_cannot_be_silently_truncated(quantity, fragment):
    source = f"Patient: It lasted {quantity}."
    assert f":S[{fragment}]" not in grounded_candidate_choices(source, FieldName.DURATION, set())
    assert draft(source, {"DUR": f"S[{fragment}]"})["duration"]["state"] == "uncertain"


@pytest.mark.parametrize("positive,negative,conflict", [
    ("last night", "today", False), ("yesterday morning", "today", False),
    ("this morning", "this afternoon", False), ("today", "this morning", False),
    ("yesterday", "on the train", False), ("yesterday", "on 10/11", False),
    ("on Monday", "on Tuesday", False), ("before breakfast", "after breakfast", False),
    ("yesterday", "since yesterday evening", False), ("", "today", False),
    ("yesterday morning", "yesterday", True), ("this morning", "today", True),
    ("last night", "last night", True), ("today", "", True), ("", "", True),
])
@pytest.mark.parametrize("denial_first", [False, True])
def test_medication_denial_requires_coverage_of_the_positive_event(positive, negative, conflict, denial_first):
    taken = ("Patient: I took ibuprofen " + positive).rstrip() + "."
    denial = ("Patient: I haven't taken anything " + negative).rstrip() + "."
    source = "\n".join((denial, taken) if denial_first else (taken, denial))
    menu = grounded_candidate_choices(source, FieldName.MEDICATION, set())
    assert any(c.startswith(":C[") for c in menu) == conflict
    med = draft(source, {"MED": "S[ibuprofen]"})["medication"]
    assert med["state"] == ("conflicting" if conflict else "supported")
    if not conflict:
        assert med["value"] == "ibuprofen"
        quotation = denial.removeprefix("Patient: ")
        assert f":A[{quotation}]" not in menu
        assert draft(source, {"MED": "A[" + quotation + "]"})["medication"]["state"] == "uncertain"


@pytest.mark.parametrize("when,state", [("yesterday", "supported"), ("this morning", "conflicting")])
def test_question_time_can_scope_an_elliptical_medication_answer(when, state):
    source = (f"Doctor: What medication did you take {when}?\nPatient: Ibuprofen.\n"
              "Patient: I haven't taken anything today.")
    assert draft(source, {"MED": "S[Ibuprofen]"})["medication"]["state"] == state


def test_fraction_modifying_another_measurement_does_not_extend_duration():
    source = "Patient: It lasted two days and a half bottle was left."
    assert draft(source, {"DUR": "S[two days]"})["duration"]["state"] == "supported"
    assert ":S[two days and a half]" not in grounded_candidate_choices(source, FieldName.DURATION, set())
    assert draft(source, {"DUR": "S[two days and a half]"})["duration"]["state"] == "uncertain"


def test_constrained_nested_citations_match_free_grounded_fields_and_exclude_exact_reuse():
    source = "Patient: Severe headache for four days."
    doc = transcript_source(NanoInput("overlap", source))

    def menu(field, used):
        return build_grounded_english_menu(doc, field, used)

    complaint = menu(FieldName.CHIEF_COMPLAINT, set())[" REPORTED[Severe headache]"][1][0]
    reserved = {(complaint.start, complaint.end)}
    severity = menu(FieldName.SEVERITY, reserved)[" REPORTED[Severe]"][1][0]
    assert " REPORTED[Severe headache]" not in menu(FieldName.CHIEF_COMPLAINT, reserved)
    assert " REPORTED[Severe]" not in menu(FieldName.SEVERITY, {(severity.start, severity.end)})
    assert " REPORTED[Severe headache]" in menu(FieldName.CHIEF_COMPLAINT, {(severity.start, severity.end)})
    assert " REPORTED[four days]" in menu(FieldName.DURATION, reserved)
    claims = draft(source, {"CC": "S[Severe headache]", "SEV": "S[Severe]", "DUR": "S[four days]"})
    assert all(claims[field]["state"] == "supported" for field in ("chief_complaint", "severity", "duration"))


@pytest.mark.parametrize("question,reply,value,supported", [
    ("How long have you lived at the new address?", "For eight months.", "eight months", False),
    ("When was your last appointment?", "five weeks ago", "five weeks ago", False),
    ("When did you last have your prescription renewed?", "last Friday", "last Friday", False),
    ("What date was the scan for your headache?", "Last Tuesday.", "Last Tuesday", False),
    ("When did you move to this address after the headache began?", "Two months ago.", "Two months ago", False),
    ("When was your last appointment?", "It was three weeks ago.", "three weeks ago", False),
    ("When did you renew the prescription for your cough?", "Yesterday.", "Yesterday", False),
    ("How long have you lived at the new address?", "My headache began two days ago.", "two days ago", True),
    ("When was your last appointment?", "My cough has lasted three weeks.", "three weeks", True),
    ("How long have you been living in pain?", "For three weeks.", "three weeks", True),
    ("How long have you been living in a constant state of pain?", "Two months.", "Two months", True),
    ("How long did you live in the apartment with that cough?", "For three weeks.", "three weeks", False),
    ("How long ago did you move into your current house?", "Last Friday.", "Last Friday", False),
    ("How many weeks ago did you renew your prescription?", "Two weeks ago.", "Two weeks ago", False),
    ("How long ago was your last scan?", "Last Monday.", "Last Monday", False),
    ("How many days ago did the pain start after your move?", "Two days ago.", "Two days ago", True),
    ("When was your last appointment? Actually, when did the headache start?", "Three weeks ago.", "Three weeks ago", True),
    ("When did the pain start? Sorry, when was your last visit?", "Last Friday.", "Last Friday", False),
    ("When was your last appointment, I mean when did the pain start?", "Yesterday.", "Yesterday", True),
    ("When was your scan after you noticed when the pain started?", "Two weeks ago.", "Two weeks ago", False),
    ("When did you move into your current house?", "Last Friday.", "Last Friday", False),
    ("When did the pain start after your appointment?", "Two days ago.", "Two days ago", True),
    ("How long have you had a headache at your new address?", "Three weeks.", "Three weeks", True),
    ("What date did you notice the rash after your scan?", "Last Tuesday.", "Last Tuesday", True),
])
def test_duration_uses_the_event_the_question_dates(question, reply, value, supported):
    transcript = f"Doctor: {question}\nPatient: {reply}"
    source = transcript_source(NanoInput("duration-owner", transcript))
    choices = grounded_candidate_choices(transcript, FieldName.DURATION, set())
    menu = build_grounded_english_menu(source, FieldName.DURATION, set())
    assert (f":S[{value}]" in choices) is supported
    assert (f" REPORTED[{value}]" in menu) is supported
    claim = draft(transcript, {"DUR": f"S[{value}]"})["duration"]
    assert (claim["state"] == "supported") is supported
