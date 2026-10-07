import pytest
from nanoscribe.prompt import (
    topic_for_spec,
    _answer_hint,
    build_span_port_prompt,
    span_port_system_prompt,
)
from nanoscribe.encounter import AtomType, Speaker, Source, Turn

class MockSpec:
    def __init__(self, atom_id="1", atom_type=AtomType.SYMPTOM, raw_value="", speaker=Speaker.PATIENT):
        self.atom_id = atom_id
        self.atom_type = atom_type
        self.raw_value = raw_value
        self.speaker = speaker

def test_topic_for_spec_allergy():
    spec = MockSpec(atom_type=AtomType.ALLERGY)
    assert topic_for_spec(spec) == "Does the patient mention or deny allergies?"

def test_topic_for_spec_medication():
    spec = MockSpec(atom_type=AtomType.MEDICATION)
    assert topic_for_spec(spec) == "Does the patient mention any medication they take?"

def test_topic_for_spec_history():
    spec = MockSpec(atom_type=AtomType.HISTORY, raw_value="diabetes")
    assert topic_for_spec(spec) == "Does the transcript mention 'diabetes' (patient history or family history)?"

def test_topic_for_spec_clinician_speaker():
    spec = MockSpec(speaker=Speaker.CLINICIAN, raw_value="surgery")
    assert topic_for_spec(spec) == "Does the clinician state 'surgery'?"

def test_topic_for_spec_assessment():
    spec = MockSpec(atom_type=AtomType.ASSESSMENT, raw_value="hypertension")
    assert topic_for_spec(spec) == "Does the clinician's assessment include 'hypertension'?"

def test_topic_for_spec_raw_value_only():
    spec = MockSpec(raw_value="cough")
    assert topic_for_spec(spec) == "Does the patient mention 'cough' (current or past)?"

def test_topic_for_spec_symptom():
    spec = MockSpec(atom_type=AtomType.SYMPTOM)
    assert topic_for_spec(spec) == "Does the transcript mention any symptom or complaint?"

def test_topic_for_spec_diagnosis():
    spec = MockSpec(atom_type=AtomType.DIAGNOSIS_STATEMENT)
    assert topic_for_spec(spec) == "Does the transcript mention any diagnosis?"

def test_topic_for_spec_plan():
    spec = MockSpec(atom_type=AtomType.PLAN)
    assert topic_for_spec(spec) == "Does the transcript mention the care plan?"

def test_topic_for_spec_procedure():
    spec = MockSpec(atom_type=AtomType.PROCEDURE)
    assert topic_for_spec(spec) == "Does the transcript mention any procedure?"

def test_topic_for_spec_instruction():
    spec = MockSpec(atom_type=AtomType.INSTRUCTION)
    assert topic_for_spec(spec) == "Does the transcript mention patient instructions?"

def test_topic_for_spec_measurement():
    spec = MockSpec(atom_type=AtomType.MEASUREMENT)
    assert topic_for_spec(spec) == "Does the transcript mention any measurement or vital sign?"

def test_topic_for_spec_unknown():
    spec = MockSpec(atom_type=type("MockEnum", (), {"value": "unknown_type"})())
    assert topic_for_spec(spec) == "Does the transcript mention the unknown_type field?"
