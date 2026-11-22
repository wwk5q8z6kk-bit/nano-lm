from nanoscribe.encounter import AtomType, Speaker
from nanoscribe.prompt import (
    _format_transcript,
    topic_for_spec,
    _answer_hint,
    build_span_port_prompt,
    span_port_system_prompt,
    _SPAN_PORT_SYSTEM
)
from dataclasses import dataclass
import pytest

@dataclass
class MockSpec:
    atom_id: str
    atom_type: AtomType
    raw_value: str
    speaker: Speaker

@dataclass
class MockTurn:
    speaker: Speaker
    text: str

@dataclass
class MockSource:
    turns: list[MockTurn]

def test_format_transcript():
    source = MockSource(turns=[
        MockTurn(speaker=Speaker.CLINICIAN, text="Any allergies?"),
        MockTurn(speaker=Speaker.PATIENT, text="No allergies.")
    ])
    expected = "clinician: Any allergies?\npatient: No allergies."
    assert _format_transcript(source) == expected

def test_topic_for_spec_allergy():
    spec = MockSpec(atom_id="1", atom_type=AtomType.ALLERGY, raw_value="", speaker=Speaker.PATIENT)
    assert topic_for_spec(spec) == "Does the patient mention or deny allergies?"

def test_topic_for_spec_medication():
    spec = MockSpec(atom_id="1", atom_type=AtomType.MEDICATION, raw_value="", speaker=Speaker.PATIENT)
    assert topic_for_spec(spec) == "Does the patient mention any medication they take?"

def test_topic_for_spec_history():
    spec = MockSpec(atom_id="1", atom_type=AtomType.HISTORY, raw_value="heart disease", speaker=Speaker.PATIENT)
    assert topic_for_spec(spec) == "Does the transcript mention 'heart disease' (patient history or family history)?"

def test_topic_for_spec_clinician():
    spec = MockSpec(atom_id="1", atom_type=AtomType.SYMPTOM, raw_value="cough", speaker=Speaker.CLINICIAN)
    assert topic_for_spec(spec) == "Does the clinician state 'cough'?"

def test_topic_for_spec_assessment():
    spec = MockSpec(atom_id="1", atom_type=AtomType.ASSESSMENT, raw_value="fever", speaker=Speaker.PATIENT)
    assert topic_for_spec(spec) == "Does the clinician's assessment include 'fever'?"

def test_topic_for_spec_symptom():
    spec = MockSpec(atom_id="1", atom_type=AtomType.SYMPTOM, raw_value="cough", speaker=Speaker.PATIENT)
    assert topic_for_spec(spec) == "Does the patient mention 'cough' (current or past)?"

def test_topic_for_spec_no_raw_value_symptom():
    spec = MockSpec(atom_id="1", atom_type=AtomType.SYMPTOM, raw_value="", speaker=Speaker.PATIENT)
    assert topic_for_spec(spec) == "Does the transcript mention any symptom or complaint?"

def test_topic_for_spec_no_raw_value_plan():
    spec = MockSpec(atom_id="1", atom_type=AtomType.PLAN, raw_value="", speaker=Speaker.PATIENT)
    assert topic_for_spec(spec) == "Does the transcript mention the care plan?"

class UnknownAtomType:
    value = "unknown_type"

def test_topic_for_spec_no_raw_value_other():
    # We can mock a completely unknown enum type to hit the final fallback line:
    spec = MockSpec(atom_id="1", atom_type=UnknownAtomType(), raw_value="", speaker=Speaker.PATIENT)
    assert topic_for_spec(spec) == "Does the transcript mention the unknown_type field?"

def test_answer_hint_allergy():
    spec = MockSpec(atom_id="1", atom_type=AtomType.ALLERGY, raw_value="", speaker=Speaker.PATIENT)
    assert 'DENIED: "No allergies."' in _answer_hint(spec)

def test_answer_hint_clinician():
    spec = MockSpec(atom_id="1", atom_type=AtomType.SYMPTOM, raw_value="cough", speaker=Speaker.CLINICIAN)
    assert 'STATED: "cough"' in _answer_hint(spec)
    assert "clinician lines" in _answer_hint(spec)

def test_answer_hint_raw_value():
    spec = MockSpec(atom_id="1", atom_type=AtomType.SYMPTOM, raw_value="cough", speaker=Speaker.PATIENT)
    assert 'STATED: "cough"' in _answer_hint(spec)
    assert "explicit denial" in _answer_hint(spec)

def test_answer_hint_empty():
    spec = MockSpec(atom_id="1", atom_type=UnknownAtomType(), raw_value="", speaker=Speaker.PATIENT)
    assert _answer_hint(spec) == ""

def test_build_span_port_prompt():
    source = MockSource(turns=[
        MockTurn(speaker=Speaker.CLINICIAN, text="Any allergies?"),
        MockTurn(speaker=Speaker.PATIENT, text="No allergies.")
    ])
    spec = MockSpec(atom_id="1", atom_type=AtomType.ALLERGY, raw_value="", speaker=Speaker.PATIENT)
    prompt = build_span_port_prompt(source, spec)
    assert "clinician: Any allergies?" in prompt
    assert "patient: No allergies." in prompt
    assert "Does the patient mention or deny allergies?" in prompt
    assert "Answer using only the patient's words." in prompt

def test_build_span_port_prompt_clinician():
    source = MockSource(turns=[
        MockTurn(speaker=Speaker.CLINICIAN, text="Patient has cough.")
    ])
    spec = MockSpec(atom_id="1", atom_type=AtomType.SYMPTOM, raw_value="cough", speaker=Speaker.CLINICIAN)
    prompt = build_span_port_prompt(source, spec)
    assert "clinician: Patient has cough." in prompt
    assert "Does the clinician state 'cough'?" in prompt
    assert "Answer using only the clinician's words." in prompt

def test_span_port_system_prompt():
    assert span_port_system_prompt() == _SPAN_PORT_SYSTEM

if __name__ == '__main__':
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in fns:
        fn()
        print(f"  PASS {name}")
    print(f"prompt pins: {len(fns)}/{len(fns)} PASS")
