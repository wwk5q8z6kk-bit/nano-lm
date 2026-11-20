"""A scalar medication observation distinguishes additional treatments from how they were taken."""
import pytest

from nano_ai.contract import EvidenceSpan, FieldName, FieldState, NanoInput
from nanoscribe.check_note import transcript_source
from nanoscribe.compact import compact_solver
from nanoscribe.grounded import build_grounded_english_menu, format_grounded_span_token, verify_grounded_span


@pytest.mark.parametrize("report,medicines", [
    ("I take aspirin and go to bed.", ("aspirin",)),
    ("I take aspirin and read a book.", ("aspirin",)),
    ("I take aspirin and feel better.", ("aspirin",)),
    ("I am taking aspirin and walk daily.", ("aspirin",)),
    ("I took aspirin and then went home.", ("aspirin",)),
    ("I take aspirin and pack my bag.", ("aspirin",)),
    ("I take aspirin, eat lunch and read the paper.", ("aspirin",)),
    ("I take aspirin and read a book with ibuprofen.", ("aspirin",)),
    ("I have taken aspirin and read a book for my headache.", ("aspirin",)),
    ("I take aspirin and ibuprofen and feel better.", ("aspirin", "ibuprofen")),
    ("I take aspirin and Xylora.", ("aspirin", "Xylora")),
    ("I take aspirin and Sleep-Eze.", ("aspirin", "Sleep-Eze")),
    ("I take aspirin and Sleep-Eze tablets.", ("aspirin", "Sleep-Eze tablets")),
    ("I take aspirin and sleep medication.", ("aspirin", "sleep medication")),
    ("I take aspirin and sleep aid for my headache.", ("aspirin", "sleep aid")),
    ("I take aspirin and sleep tablets.", ("aspirin", "sleep tablets")),
    ("I take aspirin and sleep.", ("aspirin",)),
    ("I take aspirin and go buy tablets.", ("aspirin",)),
    ("I take aspirin and read my medication list.", ("aspirin",)),
    ("I take aspirin and vitamin A.", ("aspirin", "vitamin A")),
    ("I take aspirin and milk thistle.", ("aspirin", "milk thistle")),
    ("I take aspirin and ibuprofen.", ("aspirin", "ibuprofen")),
    ("I currently take aspirin and ibuprofen.", ("aspirin", "ibuprofen")),
    ("I am taking aspirin and ibuprofen.", ("aspirin", "ibuprofen")),
    ("I’m currently taking aspirin and ibuprofen.", ("aspirin", "ibuprofen")),
    ("I have been taking aspirin and ibuprofen.", ("aspirin", "ibuprofen")),
    ("I still use aspirin and ibuprofen.", ("aspirin", "ibuprofen")),
    ("I am on aspirin and ibuprofen.", ("aspirin", "ibuprofen")),
    ("I took aspirin and ibuprofen.", ("aspirin", "ibuprofen")),
    ("I did take aspirin plus ibuprofen.", ("aspirin", "ibuprofen")),
    ("I have taken aspirin and ibuprofen.", ("aspirin", "ibuprofen")),
    ("I took aspirin with ibuprofen.", ("aspirin", "ibuprofen")),
    ("I took aspirin with co-trimoxazole.", ("aspirin", "co-trimoxazole")),
    ("I took aspirin with 5 mg of Zorina.", ("aspirin", "Zorina")),
    ("I took aspirin with calcium carbonate.", ("aspirin", "calcium carbonate")),
    ("I took aspirin with a nasal spray.", ("aspirin", "nasal spray")),
    ("I took aspirin with my other medication named Zorina.", ("aspirin", "Zorina")),
    ("I took aspirin and ibuprofen with breakfast.", ("aspirin", "ibuprofen")),
    ("I took aspirin with water and ibuprofen.", ("aspirin", "ibuprofen")),
    ("I take aspirin, ibuprofen and metformin.", ("aspirin", "ibuprofen", "metformin")),
    ("I take aspirin, ibuprofen, and metformin.", ("aspirin", "ibuprofen", "metformin")),
    ("I took aspirin, with food.", ("aspirin",)),
    ("I took aspirin and ibuprofen plus metformin.", ("aspirin", "ibuprofen", "metformin")),
    ("I took aspirin and co-trimoxazole for pain.", ("aspirin", "co-trimoxazole")),
    ("I took aspirin with water.", ("aspirin",)),
    ("I took aspirin with 250 ml of water.", ("aspirin",)),
    ("I took aspirin with a water pill.", ("aspirin", "water pill")),
    ("I took aspirin with some food and a napkin.", ("aspirin",)),
    ("I took aspirin and ibuprofen with water and a straw.", ("aspirin", "ibuprofen")),
    ("I took aspirin with some food.", ("aspirin",)),
    ("I take aspirin with a full glass of water.", ("aspirin",)),
    ("I am taking aspirin with breakfast.", ("aspirin",)),
    ("I took aspirin with my sister present.", ()),
    ("I took aspirin with encouragement.", ("aspirin",)),
    ("I took aspirin with a spray bottle.", ("aspirin",)),
    ("I took aspirin for pain and fever.", ("aspirin",)),
    ("I will take aspirin and ibuprofen.", ()),
    ("I took aspirin and aspirin.", ("aspirin",)),
])
def test_actual_coordinated_medication_evidence(report, medicines):
    field = FieldName.MEDICATION
    item = NanoInput("medication-source", "Patient: " + report)
    source = transcript_source(item)
    expected = tuple(
        EvidenceSpan(source.text.index(name), source.text.index(name) + len(name), name) for name in medicines
    )
    start = source.text.index("aspirin")
    proposed = EvidenceSpan(start, start + len("aspirin"), "aspirin")
    raw = f"CC:M|DUR:M|SEV:M|MED:S[{format_grounded_span_token(source, proposed)}]|ALG:M"
    result, _ = compact_solver(lambda _: raw, binder="grounded").infer_grounded(item)
    med = next(slot for slot in result.fields if slot.field is field)
    choices = build_grounded_english_menu(source, field, set()).values()
    single = len(medicines) == 1
    assert (med.state is FieldState.SUPPORTED) == single
    assert verify_grounded_span(source, field, FieldState.SUPPORTED, proposed)[0] == single
    assert any(code == "S" and proposed in spans for code, spans in choices) == single
    pairs = [tuple(spans) for code, spans in choices if code == "U" and len(spans) > 1]
    if len(medicines) > 1:
        assert med.state is FieldState.UNCERTAIN and med.evidence == expected
        assert expected in pairs
    else:
        assert not pairs


@pytest.mark.parametrize("companion", ["water", "food", "breakfast", "encouragement", "spray bottle"])
def test_administration_companion_cannot_be_a_medication_value(companion):
    item = NanoInput("admin-companion", f"Patient: I took aspirin with {companion}.")
    src = transcript_source(item)
    start = src.text.index(companion)
    span = EvidenceSpan(start, start + len(companion), companion)
    assert not verify_grounded_span(src, FieldName.MEDICATION, FieldState.SUPPORTED, span)[0]
    choices = build_grounded_english_menu(src, FieldName.MEDICATION, set()).values()
    assert not any(s.start <= start and span.end <= s.end for code, spans in choices
                   if code in {"S", "U"} for s in spans)
