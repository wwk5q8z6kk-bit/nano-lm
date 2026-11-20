"""Source-scope regressions for the grounded compiler, menus and independent note checker."""
import pytest

from nano_ai.contract import EvidenceSpan, FieldName, FieldState, NanoInput
from nanoscribe.check_note import _observation_kind, transcript_source
from nanoscribe.compact import compact_solver
from nanoscribe.grounded import grounded_candidate_choices
from nanoscribe.scribe import scribe


MED, ALG = FieldName.MEDICATION, FieldName.ALLERGY


def check(source, field, code, quote, expected, evidence=None):
    label = "MED" if field is MED else "ALG"
    raw = "|".join(key + (f":{code}[{quote}]" if key == label else ":M")
                   for key in ("CC", "DUR", "SEV", "MED", "ALG"))
    draft = scribe(source, solver=compact_solver(lambda _: raw, binder="grounded"), context_policy="grounded")
    assert draft["verification"]["passed"], draft["verification"]
    claim = next(c for c in draft["note"]["claims"] if c["field"] == field.value)
    assert claim["state"] == expected
    spans = {e["evidence_id"]: e for e in draft["encounter"]["evidence"]}
    refs = [spans[ref] for ref in claim["evidence_ids"]]
    assert all(source[s["start"]:s["end"]] == s["text"] for s in refs)
    if evidence is not None:
        assert [s["text"] for s in refs] == evidence
    doc = transcript_source(NanoInput("regression", source))
    if code in ("A", "S"):
        start = source.index(quote, source.index("Patient:"))
        span = EvidenceSpan(start, start + len(quote), quote)
        kind = _observation_kind(doc, field, span, context_policy="grounded")
        assert kind is (FieldState(expected) if expected in ("supported", "absent") else None)
    return grounded_candidate_choices(source, field, set())


@pytest.mark.parametrize("statement,field,quote", [
    ("Am I allergic to penicillin?", ALG, "penicillin"),
    ("Did I take ibuprofen?", MED, "ibuprofen"),
    ("Could I be allergic to nickel", ALG, "nickel"),
])
def test_patient_question_cannot_establish_observation(statement, field, quote):
    assert f":S[{quote}]" not in check("Patient: " + statement, field, "S", quote, "uncertain")


def test_patient_question_keeps_separate_current_fact_in_same_sentence():
    source = "Patient: I took ibuprofen, can you check whether I am allergic to nickel?"
    assert ":S[ibuprofen]" in check(source, MED, "S", "ibuprofen", "supported")
    check(source, ALG, "S", "nickel", "uncertain")


@pytest.mark.parametrize("statement,field", [
    ("It is not true that I have no allergies.", ALG),
    ("It is wrong to say I took no medications.", MED),
    ("I never said I have no allergies.", ALG),
    ("It is not the case that I have no allergies.", ALG),
    ("I cannot say I have no medication allergies.", ALG),
    ('My chart says "no allergies".', ALG),
    ("I have no allergies is not correct.", ALG),
])
def test_rejected_or_reported_denial_cannot_establish_absence(statement, field):
    menu = check("Patient: " + statement, field, "A", statement, "uncertain")
    assert not any(c.startswith(":A[") for c in menu)


@pytest.mark.parametrize("statement,field", [
    ("No, I have no allergies.", ALG), ("I haven't taken any medications.", MED),
])
def test_current_direct_denials_still_establish_absence(statement, field):
    assert f":A[{statement}]" in check("Patient: " + statement, field, "A", statement, "absent")


@pytest.mark.parametrize("separator", ["\nPatient: ", " "])
@pytest.mark.parametrize("tail,state", [
    ("My daily routine includes vitamin D.", "supported"),
    ("Ibuprofen is just my usual routine.", "uncertain"),
    ("That medication is just my usual routine.", "uncertain"),
])
def test_routine_disclaimer_targets_named_or_immediate_antecedent(tail, state, separator):
    source = "Patient: I took ibuprofen for the headache." + separator + tail
    choices = check(source, MED, "S", "ibuprofen", state)
    assert (":S[ibuprofen]" in choices) == (state == "supported")


def test_pronoun_refers_to_most_recent_medication():
    source = "Patient: I took aspirin. I took zinc. That is just my usual routine."
    assert ":S[aspirin]" in check(source, MED, "S", "aspirin", "supported")
    check(source, MED, "S", "zinc", "uncertain")


@pytest.mark.parametrize("field,first,second,pattern", [
    (MED, "ibuprofen", "acetaminophen", "I took {}."),
    (ALG, "nickel", "latex", "I am allergic to {}."),
])
@pytest.mark.parametrize("separator", [" ", "\r\nPatient: ", "; "])
def test_separate_active_substances_require_all_source_values(field, first, second, pattern, separator):
    first_report = pattern.format(first)
    if separator == "; ":
        first_report = first_report.removesuffix(".")
    source = "Patient: " + first_report + separator + pattern.format(second)
    menu = check(source, field, "S", first, "uncertain", [first, second])
    assert f":U[{first}||{second}]" in menu
    assert not any(c.startswith(":S[") for c in menu)
    check(source, field, "U", f"{first}||{second}", "uncertain", [first, second])


def test_single_clause_pair_and_separate_third_value_are_one_observation_set():
    source = "Patient: I took aspirin and zinc. I took ibuprofen."
    menu = check(source, MED, "S", "ibuprofen", "uncertain", ["aspirin", "zinc", "ibuprofen"])
    assert ":U[aspirin||zinc||ibuprofen]" in menu


def test_same_value_repeated_is_a_single_substance():
    source = "Patient: I took Aspirin for the headache. I took aspirin today."
    # Pick the case-sensitive distinct source occurrence explicitly.
    menu = check(source, MED, "S", "Aspirin", "supported")
    assert not any("||" in c for c in menu)


@pytest.mark.parametrize("statement", ["I will take ibuprofen tomorrow.", "I plan to use ibuprofen.",
                                       "I'll take ibuprofen later.", "I am going to start taking ibuprofen.",
                                       "I will be taking ibuprofen tomorrow."])
def test_prospective_medication_is_quarantined(statement):
    assert ":S[ibuprofen]" not in check("Patient: " + statement, MED, "S", "ibuprofen", "uncertain")


@pytest.mark.parametrize("statement", [
    "I took aspirin and I will take ibuprofen tomorrow.",
    "I will take ibuprofen tomorrow but I took aspirin today.",
])
def test_coordinated_plans_keep_the_actual_administration(statement):
    source = "Patient: " + statement
    assert ":S[aspirin]" in check(source, MED, "S", "aspirin", "supported")
    check(source, MED, "S", "ibuprofen", "uncertain")


@pytest.mark.parametrize("tail", ["Thank you.", "The headache disappeared."])
def test_nonanswer_after_medication_does_not_create_a_second_substance(tail):
    source = "Doctor: Which medicine did you try?\nPatient: I took aspirin. " + tail
    menu = check(source, MED, "S", "aspirin", "supported")
    assert not any("||" in choice for choice in menu)


def test_routine_reference_across_clarifying_question():
    source = "Patient: I took aspirin.\nDoctor: Was it for your headache?\nPatient: It was just my usual routine."
    check(source, MED, "S", "aspirin", "uncertain")


def test_coordinated_unrelated_routine_does_not_retract_actual_administration():
    source = "Patient: I took aspirin for the headache, but vitamin D is my daily routine."
    assert ":S[aspirin]" in check(source, MED, "S", "aspirin", "supported")
    check(source, MED, "S", "vitamin D", "uncertain")


@pytest.mark.parametrize("field,statement,first,second", [
    (MED, "I took aspirin and then I took zinc.", "aspirin", "zinc"),
    (ALG, "I am allergic to nickel. I am also allergic to latex.", "nickel", "latex"),
])
def test_coordinated_affirmative_substance_events(field, statement, first, second):
    menu = check("Patient: " + statement, field, "S", first, "uncertain", [first, second])
    assert f":U[{first}||{second}]" in menu
