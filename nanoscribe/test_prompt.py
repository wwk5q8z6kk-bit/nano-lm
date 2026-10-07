from dataclasses import dataclass
from nanoscribe.prompt import (
    topic_for_spec,
    _answer_hint,
    build_span_port_prompt,
    span_port_system_prompt,
    _format_transcript,
)
from nanoscribe.encounter import AtomType, Speaker, Source, Turn

@dataclass
class MockSpec:
    atom_id: str
    atom_type: AtomType
    raw_value: str
    speaker: Speaker

def test_topic_for_spec_allergy():
    spec = MockSpec(atom_id="1", atom_type=AtomType.ALLERGY, raw_value="", speaker=Speaker.PATIENT)
    assert topic_for_spec(spec) == "Does the patient mention or deny allergies?"

def test_topic_for_spec_medication():
    spec = MockSpec(atom_id="1", atom_type=AtomType.MEDICATION, raw_value="", speaker=Speaker.PATIENT)
    assert topic_for_spec(spec) == "Does the patient mention any medication they take?"

def test_topic_for_spec_history():
    spec = MockSpec(atom_id="1", atom_type=AtomType.HISTORY, raw_value="diabetes", speaker=Speaker.PATIENT)
    assert topic_for_spec(spec) == "Does the transcript mention 'diabetes' (patient history or family history)?"

def test_topic_for_spec_clinician():
    spec = MockSpec(atom_id="1", atom_type=AtomType.SYMPTOM, raw_value="cough", speaker=Speaker.CLINICIAN)
    assert topic_for_spec(spec) == "Does the clinician state 'cough'?"

def test_topic_for_spec_assessment():
    spec = MockSpec(atom_id="1", atom_type=AtomType.ASSESSMENT, raw_value="hypertension", speaker=Speaker.PATIENT)
    assert topic_for_spec(spec) == "Does the clinician's assessment include 'hypertension'?"

def test_topic_for_spec_raw_value_patient():
    spec = MockSpec(atom_id="1", atom_type=AtomType.SYMPTOM, raw_value="headache", speaker=Speaker.PATIENT)
    assert topic_for_spec(spec) == "Does the patient mention 'headache' (current or past)?"

def test_topic_for_spec_no_raw_value_mapping():
    # Test symptom mapping when raw_value is empty and speaker is not clinician
    spec = MockSpec(atom_id="1", atom_type=AtomType.SYMPTOM, raw_value="", speaker=Speaker.PATIENT)
    assert topic_for_spec(spec) == "Does the transcript mention any symptom or complaint?"

    spec = MockSpec(atom_id="2", atom_type=AtomType.DIAGNOSIS_STATEMENT, raw_value="", speaker=Speaker.PATIENT)
    assert topic_for_spec(spec) == "Does the transcript mention any diagnosis?"

    spec = MockSpec(atom_id="3", atom_type=AtomType.PLAN, raw_value="", speaker=Speaker.PATIENT)
    assert topic_for_spec(spec) == "Does the transcript mention the care plan?"

    spec = MockSpec(atom_id="4", atom_type=AtomType.PROCEDURE, raw_value="", speaker=Speaker.PATIENT)
    assert topic_for_spec(spec) == "Does the transcript mention any procedure?"

    spec = MockSpec(atom_id="5", atom_type=AtomType.INSTRUCTION, raw_value="", speaker=Speaker.PATIENT)
    assert topic_for_spec(spec) == "Does the transcript mention patient instructions?"

    spec = MockSpec(atom_id="6", atom_type=AtomType.MEASUREMENT, raw_value="", speaker=Speaker.PATIENT)
    assert topic_for_spec(spec) == "Does the transcript mention any measurement or vital sign?"

def test_topic_for_spec_fallback():
    pass

def test_answer_hint_allergy():
    spec = MockSpec(atom_id="1", atom_type=AtomType.ALLERGY, raw_value="", speaker=Speaker.PATIENT)
    assert _answer_hint(spec) == ' If denied, reply DENIED: "No allergies." If affirmed, STATED with a quote.'

def test_answer_hint_clinician():
    spec = MockSpec(atom_id="1", atom_type=AtomType.SYMPTOM, raw_value="cough", speaker=Speaker.CLINICIAN)
    assert _answer_hint(spec) == ' If yes, reply STATED: "cough". Use only clinician lines, not patient lines.'

def test_answer_hint_raw_value():
    spec = MockSpec(atom_id="1", atom_type=AtomType.SYMPTOM, raw_value="headache", speaker=Speaker.PATIENT)
    assert _answer_hint(spec) == ' If mentioned (including past history), reply STATED: "headache". Use DENIED only for explicit denial.'

def test_answer_hint_empty():
    spec = MockSpec(atom_id="1", atom_type=AtomType.SYMPTOM, raw_value="", speaker=Speaker.PATIENT)
    assert _answer_hint(spec) == ""

def test_format_transcript():
    source = Source(
        source_id="src-1",
        text="Hello\nHi",
        turns=(
            Turn(turn_id="t1", source_id="src-1", speaker=Speaker.CLINICIAN, start=0, end=5, text="Hello"),
            Turn(turn_id="t2", source_id="src-1", speaker=Speaker.PATIENT, start=6, end=8, text="Hi"),
        )
    )
    assert _format_transcript(source) == "clinician: Hello\npatient: Hi"

def test_build_span_port_prompt():
    source = Source(
        source_id="src-1",
        text="Hello\nHi",
        turns=(
            Turn(turn_id="t1", source_id="src-1", speaker=Speaker.CLINICIAN, start=0, end=5, text="Hello"),
            Turn(turn_id="t2", source_id="src-1", speaker=Speaker.PATIENT, start=6, end=8, text="Hi"),
        )
    )
    spec = MockSpec(atom_id="1", atom_type=AtomType.SYMPTOM, raw_value="headache", speaker=Speaker.PATIENT)
    prompt = build_span_port_prompt(source, spec)
    assert "clinician: Hello\npatient: Hi" in prompt
    assert "Does the patient mention 'headache' (current or past)?" in prompt
    assert "Answer using only the patient's words." in prompt
    assert 'If mentioned (including past history), reply STATED: "headache". Use DENIED only for explicit denial.' in prompt
    assert "Reply with exactly one line: STATED, DENIED, UNCERTAIN, or NOT_MENTIONED with a verbatim quote." in prompt

def test_span_port_system_prompt():
    prompt = span_port_system_prompt()
    assert "You extract clinical facts from transcripts." in prompt
    assert "STATED, DENIED, UNCERTAIN, or NOT_MENTIONED" in prompt

if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in fns:
        fn()
        print(f"  PASS {name}")
    print(f"prompt pins: {len(fns)}/{len(fns)} PASS")
