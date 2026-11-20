"""Contrast cases at the public compiler, constrained menu and independent verifier."""
import pytest

from nano_ai.contract import EvidenceSpan, FieldName, FieldState, NanoInput
from nanoscribe.check_note import _observation_kind, transcript_source
from nanoscribe.compact import compact_solver
from nanoscribe.grounded import format_grounded_span_token, grounded_candidate_choices
from nanoscribe.scribe import scribe


def check(source, slot, quote, code="S", expected="supported"):
    field = {"SEV": FieldName.SEVERITY, "DUR": FieldName.DURATION,
             "MED": FieldName.MEDICATION, "ALG": FieldName.ALLERGY}[slot]
    doc = transcript_source(NanoInput("scope", source))
    pos = source.index(quote, source.index("Patient:"))
    span = EvidenceSpan(pos, pos + len(quote), quote)
    token = format_grounded_span_token(doc, span)
    choice = f":{code}[{token}]"
    raw = "|".join(k + (choice if k == slot else ":M") for k in ("CC", "DUR", "SEV", "MED", "ALG"))
    draft = scribe(source, solver=compact_solver(lambda _: raw, binder="grounded"), context_policy="grounded")
    assert draft["verification"]["passed"]
    claim = next(c for c in draft["note"]["claims"] if c["field"] == field.value)
    assert claim["state"] == expected
    if expected == "supported":
        assert claim["value"] == quote
    menu = grounded_candidate_choices(source, field, set())
    if expected != "conflicting":
        wanted = FieldState(expected) if expected in ("supported", "absent") else None
        assert _observation_kind(doc, field, span, context_policy="grounded") is wanted
        assert (choice in menu) == (wanted is not None)
    return menu


@pytest.mark.parametrize("first,second,quote,slot", [
    ("It was mild this morning.", "It is severe now.", "severe", "SEV"),
    ("It was moderate yesterday.", "It is mild today.", "mild", "SEV"),
    ("It is mild at rest.", "It is severe when walking.", "severe", "SEV"),
    ("It is 3/10.", "It is three out of ten.", "3/10", "SEV"),
    ("It is moderate.", "It is five out of ten.", "moderate", "SEV"),
    ("It began one week ago.", "It began seven days ago.", "one week", "DUR"),
    ("It started two weeks ago.", "It started 14 days ago.", "14 days", "DUR"),
    ("It has lasted half an hour.", "It has lasted 30 minutes.", "30 minutes", "DUR"),
    ("It began one year ago.", "It began twelve months ago.", "twelve months", "DUR"),
    ("It began about four days ago.", "It began three days ago.", "three days", "DUR"),
])
def test_distinct_renderings_or_scopes_do_not_force_conflict(first, second, quote, slot):
    menu = check(f"Patient: {first}\nPatient: {second}", slot, quote)
    assert not any(c.startswith(":C[") for c in menu)


@pytest.mark.parametrize("first,second,quote,slot", [
    ("It began one week ago.", "It began six days ago.", "one week", "DUR"),
    ("It is .5/10.", "It is 5/10.", ".5/10", "SEV"),
    ("It is 7.5/10 now.", "It is 6/10 now.", "6/10", "SEV"),
    ("It is mild now.", "It is severe now.", "mild", "SEV"),
])
def test_incompatible_values_at_same_scope_force_conflict(first, second, quote, slot):
    menu = check(f"Patient: {first}\nPatient: {second}", slot, quote, expected="conflicting")
    assert any(c.startswith(":C[") for c in menu)


@pytest.mark.parametrize("separator", ["\nPatient: ", " "])
@pytest.mark.parametrize("wrong", ["My mother was wrong about her ibuprofen.",
                                   "If I was mistaken about ibuprofen, I would check.",
                                   "Was I mistaken about ibuprofen?", "I was mistaken about a bus time.",
                                   "My mother was wrong about her tablets. I remember ibuprofen."])
def test_out_of_scope_corrections_keep_current_medication(separator, wrong):
    check("Patient: I took ibuprofen." + separator + wrong, "MED", "ibuprofen")


@pytest.mark.parametrize("correction", ["I was mistaken about ibuprofen.", "Actually I was wrong."])
def test_actual_patient_correction_retracts_original(correction):
    check("Patient: I took ibuprofen.\nPatient: " + correction, "MED", "ibuprofen", expected="uncertain")


@pytest.mark.parametrize("slot,quote,matching,targeted", [
    ("ALG", "None whatsoever.", "Do you have any allergies?", "Are you allergic to penicillin?"),
    ("MED", "Nothing at all.", "Have you taken anything for the pain?", "Have you taken any aspirin?"),
    ("MED", "No, nothing yet.", "Did you try any medicine?", "Have you tried aspirin?"),
    ("ALG", "None whatsoever.", "Are you allergic to anything?", "Any allergies to latex?"),
])
def test_elliptical_denial_requires_field_wide_question(slot, quote, matching, targeted):
    for question, state in [(matching, "absent"), (targeted, "uncertain"), ("Have you had chest pain?", "uncertain"),
                            ("Was the pain caused by your medicine?", "uncertain"), ("", "uncertain")]:
        prefix = f"Doctor: {question}\n" if question else ""
        check(prefix + "Patient: " + quote, slot, quote, "A", state)


@pytest.mark.parametrize("slot,quote", [("ALG", "No known allergies."), ("MED", "I haven't taken anything.")])
def test_explicit_field_denials_are_self_contained(slot, quote):
    check("Patient: " + quote, slot, quote, "A", "absent")


@pytest.mark.parametrize("sentence,slot,full,fragment", [
    ("The pain is 7.5/10.", "SEV", "7.5/10", "5/10"),
    ("It is 8.25 out of ten.", "SEV", "8.25 out of ten", "25 out of ten"),
    ("It started 12 days ago.", "DUR", "12 days", "2 days"),
    ("It started 1.5 days ago.", "DUR", "1.5 days", "5 days"),
    ("It began 1234 minutes ago.", "DUR", "1234 minutes", "234 minutes"),
    ("It is .5/10.", "SEV", ".5/10", "5/10"),
])
def test_numeric_boundaries_preserve_exact_value(sentence, slot, full, fragment):
    source = "Patient: " + sentence + " Thank you."
    check(source, slot, full)
    check(source, slot, fragment, expected="uncertain")


@pytest.mark.parametrize("sentence,slot,partial", [
    ("It was 7/100.", "SEV", "7/10"), ("It lasted 1e3 days.", "DUR", "3 days"), ("It lasted 1,200 minutes.", "DUR", "200 minutes"),
    ("It lasted 1/2 days.", "DUR", "2 days"), ("It was -3/10.", "SEV", "3/10"),
])
def test_numeric_tokens_cannot_be_reinterpreted_by_truncation(sentence, slot, partial):
    check("Patient: " + sentence, slot, partial, expected="uncertain")


def test_fractional_decimal_remains_distinct_in_encounter_and_grounded_contract():
    from nano_ai.contract import ContractValidationError
    from nanoscribe.encounter import normalize_value
    from nanoscribe.grounded import GroundedFieldOutput

    assert normalize_value("the '.5/10.'") == ".5/10"
    span = EvidenceSpan(0, 5, ".5/10")
    with pytest.raises(ContractValidationError, match="unsupported_value"):
        GroundedFieldOutput(FieldName.SEVERITY, FieldState.SUPPORTED, value="5/10", evidence=(span,))
