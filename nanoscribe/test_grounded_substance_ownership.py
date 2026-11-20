"""A supplied source quote needs a local medication or allergen relationship."""
import pytest

from nano_ai.contract import EvidenceSpan, FieldName, FieldState, NanoInput
from nanoscribe.check_note import _observation_kind, transcript_source
from nanoscribe.compact import compact_solver
from nanoscribe.grounded import build_grounded_english_menu, grounded_candidate_choices
from nanoscribe.scribe import scribe

MED, ALG = FieldName.MEDICATION, FieldName.ALLERGY


def observed(report, field, value):
    doc = transcript_source(NanoInput("ownership", report))
    key = "MED" if field is MED else "ALG"
    raw = "|".join(f"{slot}:S[{value}]" if slot == key else f"{slot}:M"
                   for slot in ("CC", "DUR", "SEV", "MED", "ALG"))
    draft = scribe(report, solver=compact_solver(lambda _: raw, binder="grounded"), context_policy="grounded")
    assert draft["verification"]["passed"]
    claim = next(c for c in draft["note"]["claims"] if c["field"] == field.value)
    start = report.index(value, report.index("Patient:"))
    direct = _observation_kind(doc, field, EvidenceSpan(start, start + len(value), value), context_policy="grounded")
    return claim, direct, grounded_candidate_choices(report, field, set()), build_grounded_english_menu(doc, field, set())


@pytest.mark.parametrize("field,question,reply,value", [
    (MED, "Tell me more.", "Wrist pain.", "Wrist pain"),
    (ALG, "Tell me more.", "Wrist pain.", "Wrist pain"),
    (MED, "Tell me more.", "Aspirin.", "Aspirin"),
    (ALG, "Tell me more.", "Nickel.", "Nickel"),
    (MED, "What brings you in?", "Wrist pain.\nPatient: No medications.", "Wrist pain"),
    (ALG, "What brings you in?", "Wrist pain.\nPatient: No allergies.", "Wrist pain"),
    (MED, "Tell me more.", "Cash for the parking meter.\nPatient: No medications.", "Cash"),
    (MED, "Any medicine?", "I took aspirin for my wrist pain.", "wrist pain"),
    (MED, "Any medicine?", "I am allergic to birch pollen.", "birch pollen"),
    (ALG, "Any allergies?", "I took a blue gel cap.", "blue gel cap"),
    (ALG, "Any allergies?", "Nickel gives me hives.", "hives"),
    (ALG, "Tell me more.", "I am allergic to nickel because it causes a rash.", "rash"),
    (MED, "Tell me more.", "My daily medicine is Xyzalorfen for my wrist pain.", "wrist pain"),
    (MED, "What brings you in?", "I am on edge.", "edge"),
    (MED, "Tell me more.", "I take the bus.", "bus"),
    (ALG, "Tell me more.", "I do not have a penicillin allergy.", "penicillin"),
    (ALG, "Tell me more.", "I had a penicillin allergy in childhood.", "penicillin"),
    (ALG, "Tell me more.", "If I have a penicillin allergy I will call.", "penicillin"),
    (ALG, "Tell me more.", "My sister has a penicillin allergy.", "penicillin"),
    (MED, "Tell me more.", "I am not currently taking aspirin.", "aspirin"),
    (MED, "Tell me more.", "I was previously taking aspirin.", "aspirin"),
    (MED, "Tell me more.", "If I am currently taking aspirin, I will check.", "aspirin"),
    (MED, "Tell me more.", "My sister is currently taking aspirin.", "aspirin"),
    (MED, "Tell me more.", "I am currently taking aspirin for my thumb pain.", "thumb pain"),
])
def test_unlicensed_supplied_span_cannot_become_a_substance(field, question, reply, value):
    claim, direct, compact, english = observed(f"Doctor: {question}\nPatient: {reply}", field, value)
    assert claim["state"] == "uncertain" and claim["value"] is None
    assert direct is None
    assert f":S[{value}]" not in compact and f" REPORTED[{value}]" not in english
    assert not any(choice.startswith(":C[") for choice in compact)
    assert not any(choice.startswith(" CONTRADICTED[") for choice in english)


@pytest.mark.parametrize("field,question,reply,value", [
    (ALG, "Any allergies?", "I would have to check with the pharmacy.", "I would have"),
    (ALG, "Any allergies?", "I'd have to ask about nickel.", "nickel"),
    (MED, "What medicine did you take?", "I need to check the aspirin bottle.", "aspirin"),
    (MED, "What medicine did you take?", "Let me check the aspirin bottle.", "aspirin"),
])
def test_deferring_a_substance_answer_offers_no_reported_value(field, question, reply, value):
    claim, direct, compact, english = observed(f"Doctor: {question}\nPatient: {reply}", field, value)
    assert claim["value"] is None and claim["state"] == "uncertain"
    assert direct is None
    assert not any(choice.startswith(":S[") for choice in compact)
    assert not any(choice.startswith(" REPORTED[") for choice in english)


@pytest.mark.parametrize("field,report,value", [
    (MED, "Patient: I took Xyzalorfen-420mg.", "Xyzalorfen-420mg"),
    (MED, "Patient: I am on warfarin.", "warfarin"),
    (MED, "Patient: My daily medicine is Xyzalorfen.", "Xyzalorfen"),
    (MED, "Patient: I take metformin.", "metformin"),
    (ALG, "Doctor: Tell me more.\nPatient: I have a penicillin allergy.", "penicillin"),
    (ALG, "Patient: I've got a known red cedar pollen allergy.", "red cedar pollen"),
    (ALG, "Patient: I have an almond allergy that causes hives.", "almond"),
    (ALG, "Patient: I have an allergy to nickel.", "nickel"),
    (MED, "Doctor: Tell me more.\nPatient: I am currently taking aspirin.", "aspirin"),
    (MED, "Patient: I'm still using cedar gel.", "cedar gel"),
    (MED, "Patient: I have also been regularly applying cedar gel.", "cedar gel"),
    (MED, "Patient: I use cedar gel for my rash.", "cedar gel"),
    (ALG, "Patient: My known allergen is cedar pollen.", "cedar pollen"),
    (ALG, "Patient: I am allergic to nickel because it causes a rash.", "nickel"),
    (MED, "Patient: My daily medicine is Xyzalorfen for my wrist pain.", "Xyzalorfen"),
    (MED, "Patient: Silver leaf drops for the headache.", "Silver leaf drops"),
    (ALG, "Patient: I am allergic to birch pollen.", "birch pollen"),
    (ALG, "Patient: Birch pollen gives me hives.", "Birch pollen"),
    (MED, "Doctor: What medication did you take?\nPatient: Blueberry gel caps.", "Blueberry gel caps"),
    (ALG, "Doctor: Any known allergies?\nPatient: Birch pollen.", "Birch pollen"),
])
def test_source_owned_substance_and_contrary_denial_survive(field, report, value):
    claim, direct, compact, english = observed(report, field, value)
    assert claim["state"] == "supported" and claim["value"] == value
    assert direct is FieldState.SUPPORTED
    assert f":S[{value}]" in compact and f" REPORTED[{value}]" in english

    denial = "No medications." if field is MED else "No allergies."
    claim, _, compact, english = observed(report + "\nPatient: " + denial, field, value)
    assert claim["state"] == "conflicting" and claim["value"] is None
    assert any(choice.startswith(":C[") for choice in compact)
    assert any(choice.startswith(" CONTRADICTED[") for choice in english)


def test_free_proposal_is_not_limited_to_the_menu_spelling():
    report = "Patient: I took Xyzalorfen-420mg because my cough returned."
    value = "Xyzalorfen-420mg"
    claim, direct, compact, english = observed(report, MED, value)
    assert claim["state"] == "supported" and claim["value"] == value
    assert direct is FieldState.SUPPORTED
    assert f":S[{value}]" not in compact and f" REPORTED[{value}]" not in english


def test_later_emphatic_confirmation_uses_its_own_source_offsets():
    report = "Patient: I took aspirin.\nPatient: I'm not sure whether I took it.\nPatient: I checked. I did take aspirin."
    doc = transcript_source(NanoInput("later-confirmation", report))
    start = report.rindex("aspirin")
    span = EvidenceSpan(start, start + len("aspirin"), "aspirin")
    tagged = f"@{start}..{span.end}:aspirin"
    raw = f"CC:M|DUR:M|SEV:M|MED:S[{tagged}]|ALG:M"
    draft = scribe(report, solver=compact_solver(lambda _: raw, binder="grounded"), context_policy="grounded")
    assert draft["verification"]["passed"]
    claim = next(c for c in draft["extraction"]["fields"] if c["field"] == "medication")
    assert claim["state"] == "supported" and claim["evidence"] == [span.to_dict()]
    assert _observation_kind(doc, MED, span, context_policy="grounded") is FieldState.SUPPORTED
    assert f":S[{tagged}]" in grounded_candidate_choices(report, MED, set())
    earlier = report.index("aspirin")
    menu = build_grounded_english_menu(doc, MED, {(earlier, earlier + len("aspirin"))})
    assert menu[" REPORTED[aspirin]"] == ("S", (span,))
