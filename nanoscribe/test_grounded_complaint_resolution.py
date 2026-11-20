"""Current complaint evidence retains its own resolution and recurrence context."""
from dataclasses import replace

import pytest

from nano_ai.contract import EvidenceSpan, FieldName, FieldState, NanoInput
from nanoscribe.check_note import assess_fields, transcript_source
from nanoscribe.compact import compact_solver
from nanoscribe.grounded import build_grounded_english_menu, format_grounded_span_token


@pytest.mark.parametrize("report,quote,occurrence,present", [
    ("My headache has resolved.", "headache", 0, False),
    ("My cough is gone.", "cough", 0, False),
    ("My rash has cleared up.", "rash", 0, False),
    ("My headache went away.", "headache", 0, False),
    ("My fever completely disappeared.", "fever", 0, False),
    ("My headache stopped me from sleeping.", "headache", 0, True),
    ("My cough has not stopped.", "cough", 0, True),
    ("My headache has stopped.", "headache", 0, False),
    ("My pain over my left eye has resolved.", "pain", 0, False),
    ("My headache yesterday has resolved.", "headache", 0, False),
    ("My pain is over my left eye.", "pain", 0, True),
    ("My headache has gone away briefly but has returned.", "headache", 0, True),
    ("My cough is over.", "cough", 0, False),
    ("My headache has not resolved.", "headache", 0, True),
    ("My headache hasn’t completely resolved.", "headache", 0, True),
    ("My headache isn't completely gone.", "headache", 0, True),
    ("My headache mostly resolved.", "headache", 0, True),
    ("My headache over my left eye is throbbing.", "headache", 0, True),
    ("My headache has resolved, but I still have nausea.", "headache", 0, False),
    ("My headache has resolved, but I still have nausea.", "nausea", 0, True),
    ("My headache has resolved but nausea persists.", "headache", 0, False),
    ("My headache has resolved but nausea persists.", "nausea", 0, True),
    ("My cough persists, but my rash has resolved.", "cough", 0, True),
    ("My cough persists, but my rash has resolved.", "rash", 0, False),
    ("My headache resolved but has returned.", "headache", 0, True),
    ("My headache has resolved. I now have a headache again.", "headache", 0, False),
    ("My headache has resolved. I now have a headache again.", "headache", 1, True),
    ("I had a headache. It has resolved.", "headache", 0, False),
    ("I had a headache.\nPatient: It's gone.", "headache", 0, False),
    ("I had a headache. The headache has resolved.", "headache", 0, False),
    ("I have left wrist pain. My right wrist pain has resolved.", "left wrist pain", 0, True),
    ("I have chest pain. It has not resolved.", "chest pain", 0, True),
    ("I have chest pain. The swelling went away.", "chest pain", 0, True),
    ("I have chest pain.\nDoctor: It has resolved.", "chest pain", 0, True),
    ("I have no headache.", "headache", 0, False),
])
def test_current_complaint_citation(report, quote, occurrence, present):
    field = FieldName.CHIEF_COMPLAINT
    transcript = f"Patient: {report}"
    item = NanoInput("complaint-course", transcript)
    source = transcript_source(item)
    offsets = [i for i in range(len(transcript)) if transcript.startswith(quote, i)]
    start = offsets[occurrence]
    span = EvidenceSpan(start, start + len(quote), quote)
    raw = f"CC:S[{format_grounded_span_token(source, span)}]|DUR:M|SEV:M|MED:M|ALG:M"
    result, _ = compact_solver(lambda _: raw, binder="grounded").infer_grounded(item)
    complaint = next(entry for entry in result.fields if entry.field is field)
    assert (complaint.state is FieldState.SUPPORTED) == present

    supplied = replace(complaint, state=FieldState.SUPPORTED, value=quote, evidence=(span,))
    forced = replace(result, fields=tuple(supplied if entry.field is field else entry for entry in result.fields))
    assessment = next(x for x in assess_fields(item, forced, context_policy="grounded") if x.field is field)
    assert (assessment.state is FieldState.SUPPORTED) == present
    menu = build_grounded_english_menu(source, field, set())
    assert any(s.start <= start and span.end <= s.end for code, spans in menu.values()
               if code == "S" for s in spans) == present
