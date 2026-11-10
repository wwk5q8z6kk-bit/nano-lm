"""New injected development cases; no validation corpus or model outputs used."""
import pytest

from nanoscribe.compact import compact_solver
from nanoscribe.scribe import scribe

BASE = "Doctor: What symptoms brought you here?\nPatient: I've been having heel tenderness.\n"
RAW = "CC:S[heel tenderness]|DUR:M|SEV:M|MED:M|ALG:M"


def checked(source, raw=RAW):
    calls = []
    def predict(text):
        calls.append(text)
        return raw
    result = scribe(source, context_policy="context", solver=compact_solver(predict, binder="context"))
    assert calls == [source]
    assert result["verification"]["passed"], result
    for span in result["encounter"]["evidence"]:
        assert source[span["start"]:span["end"]] == span["text"]
    return result


@pytest.mark.parametrize("extra", [
    "Doctor: Was the parking lot crowded?\nPatient: There was plenty of room.\n",
    "Doctor: Which medicine did you try?\nPatient: I misplaced the packet and cannot identify the tablet.\n",
    "Doctor: Which medicines is your cousin taking?\nPatient: Just fexofenadine.\n",
    "Doctor: Which medicines were you taking last year?\nPatient: Just loratadine.\n",
    "Doctor:\nPatient: The map was hard to read.\n",
])
def test_unrelated_or_field_local_unknown_context_preserves_valid_complaint(extra):
    result = checked(BASE + extra)
    assert result["disposition"] == "review"
    assert result["note"]["claims"][0]["value"] == "heel tenderness"
    assert result["note"]["claims"][3]["state"] not in ("supported", "absent")


def test_uncertain_reply_cannot_hide_cross_field_correction():
    source = BASE + "Doctor: Which medicine did you try?\nPatient: Just cetirizine.\nDoctor: How bad would you say it is?\nPatient: I cannot rate it, and I did not take any medicine.\n"
    result = checked(source, RAW.replace("MED:M", "MED:S[cetirizine]"))
    assert result["disposition"] == "review"
    assert result["note"]["claims"][0]["state"] == "supported"
    assert result["note"]["claims"][3]["state"] == "uncertain"
    assert any("did not take" in e["text"] for e in result["encounter"]["evidence"])


@pytest.mark.parametrize("extra", [
    "Patient: Actually, those symptoms were my cousin's.\n",
    "Nurse: Everything the patient said was about her husband.\n",
])
def test_reassigned_source_cannot_retain_patient_complaint(extra):
    result = checked(BASE + extra)
    assert result["note"]["claims"][0]["state"] == "uncertain"
    assert result["disposition"] == "abstain"


def test_unattributed_named_correction_cannot_hide_behind_unknown_question():
    source = BASE + "Doctor: Which medicine did you try?\nPatient: Just cetirizine.\nDoctor: Anything to add before we finish?\nPatient: Cetirizine was not something I took.\n"
    result = checked(source, RAW.replace("MED:M", "MED:S[cetirizine]"))
    assert result["note"]["claims"][0]["state"] == "supported"
    assert result["note"]["claims"][3]["state"] == "uncertain"


def test_family_medication_question_does_not_erase_current_patient_medication():
    source = BASE + "Doctor: Which medicine did you try?\nPatient: Just cetirizine.\nDoctor: Which medicines is your cousin taking?\nPatient: Just loratadine.\n"
    result = checked(source, RAW.replace("MED:M", "MED:S[cetirizine]"))
    assert result["note"]["claims"][3]["state"] == "supported"
    assert result["note"]["claims"][3]["value"] == "cetirizine"


@pytest.mark.parametrize("question,reply,slot", [
    ("What symptoms brought you here?", "I have elbow stiffness.", "CC:S[elbow stiffness]"),
    ("What is the duration of your symptoms?", "Nine days.", "DUR:S[Nine days]"),
    ("How bad would you say it is?", "It feels moderate.", "SEV:S[moderate]"),
    ("Which medicine did you try?", "I took cetirizine.", "MED:S[cetirizine]"),
    ("Which substances are you allergic to?", "Nickel.", "ALG:S[Nickel]"),
])
def test_complete_simple_replies_use_exact_source_spans(question, reply, slot):
    label = slot.split(":")[0]
    raw = "|".join(slot if k == label else k + ":M" for k in ("CC", "DUR", "SEV", "MED", "ALG"))
    result = checked(f"Doctor: {question}\r\nPatient: {reply}\r\n", raw)
    assert result["disposition"] == "review"
    assert sum(c["state"] == "supported" for c in result["note"]["claims"]) == 1


@pytest.mark.parametrize("reply", ["I took no cetirizine.", "I took cetirizine and loratadine.", "I took my cousin's cetirizine.", "I might take cetirizine.", "I took something."])
def test_simple_reply_support_never_extracts_from_qualified_or_compound_use(reply):
    source = BASE + "Doctor: Which medicine did you try?\nPatient: " + reply
    result = checked(source, RAW.replace("MED:M", "MED:S[cetirizine]") if "cetirizine" in reply else RAW)
    assert result["disposition"] == "review"
    assert result["note"]["claims"][3]["state"] == "uncertain"


@pytest.mark.parametrize("question", [
    "What medicine did your cousin use, and did you take any?",
    "What medicine did you use last year, and what about now?",
    "What medicine did your cousin use?",
])
def test_mixed_other_scope_cannot_hide_a_patient_correction(question):
    source = BASE + "Doctor: Which medicine did you try?\nPatient: Just cetirizine.\nDoctor: " + question + "\nPatient: I did not take any medicine."
    result = checked(source, RAW.replace("MED:M", "MED:S[cetirizine]"))
    assert result["note"]["claims"][0]["state"] == "supported"
    assert result["note"]["claims"][3]["state"] == "uncertain"


@pytest.mark.parametrize("question,reply,label,index", [
    ("What medicine?", "Nope.", "MED", 3),
    ("Any allergies?", "Negative.", "ALG", 4),
    ("What medicine?", "Cetirizine unavailable.", "MED", 3),
    ("What medicine?", "Two tablets.", "MED", 3),
    ("What medicine?", "White pill.", "MED", 3),
    ("Any allergies?", "Rash.", "ALG", 4),
    ("What medicine?", "I took two tablets.", "MED", 3),
])
def test_bare_nonvalues_are_not_assertions(question, reply, label, index):
    source = BASE + "Doctor: " + question + "\nPatient: " + reply
    raw = RAW.replace(label + ":M", label + ":S[" + reply[:-1] + "]")
    result = checked(source, raw)
    assert result["note"]["claims"][index]["state"] == "uncertain"


@pytest.mark.parametrize("correction", ["Actually, that was my sister's experience.", "I made that up."])
def test_unbounded_anaphoric_reassignment_invalidates_source(correction):
    result = checked(BASE + "Patient: " + correction)
    assert result["note"]["claims"][0]["state"] == "uncertain"


@pytest.mark.parametrize("extra", [
    "Patient: I didn't use it after all.",
    "Patient: That was not true.",
    "Patient: Actually, I was mistaken.",
    "Patient: I was wrong.",
    "Doctor: When did you take it?\nPatient: It was before this episode.",
])
def test_unresolved_anaphoric_correction_quarantines_prior_field(extra):
    source = BASE + "Doctor: Which medicine did you try?\nPatient: Just cetirizine.\n" + extra
    result = checked(source, RAW.replace("MED:M", "MED:S[cetirizine]"))
    assert result["note"]["claims"][0]["state"] == "supported"
    assert result["note"]["claims"][3]["state"] == "uncertain"
