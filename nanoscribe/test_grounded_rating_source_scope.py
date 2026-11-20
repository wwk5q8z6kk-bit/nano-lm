"""The conversational source and anatomical site determine which rating an utterance owns."""
import pytest
from nano_ai.contract import FieldName
from nanoscribe.test_grounded_followup import check


@pytest.mark.parametrize("source,quote,accepted", [
    ("Doctor: Read the sentence on the printed card aloud.\nPatient: My shaking is 5/10.", "5/10", False),
    ("Doctor: Please read the line on that card.\nPatient: My headache is 5/10.", "5/10", False),
    ("Doctor: Read the printed example silently, then tell me your actual headache rating.\nPatient: My headache is 5/10 right now.", "5/10", True),
    ("Doctor: Read the example aloud.\nPatient: I cannot see the example. My headache is 5/10 right now.", "5/10", True),
    ("Doctor: Read the example aloud.\nPatient: I can see the example. My headache is 5/10 right now.", "5/10", False),
    ("Doctor: Read the example aloud.\nPatient: My headache is 5/10 right now.", "5/10", False),
    ("Patient: My tremor is 5/10. That rating was yesterday afternoon. The tremor is gone now.", "5/10", False),
    ("Patient: My left wrist pain is 5/10. That was at work. Right now my right knee pain is 2/10.", "5/10", True),
    ("Patient: The headache is 5/10. That was at work. Right now my tremor is 2/10.", "5/10", True),
    ("Patient: My left wrist pain is 5/10. That was at work. Right now my left wrist pain is 2/10.", "5/10", False),
    ("Patient: My left wrist pain is 5/10. That was at work. Right now my right wrist pain is 2/10.", "5/10", True),
])
def test_local_statement_owner(source, quote, accepted):
    check(source, "SEV", FieldName.SEVERITY, quote, source.index(quote), accepted)
