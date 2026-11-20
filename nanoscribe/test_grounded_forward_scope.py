"""Only observations in the patient's source scope can support a patient field."""

import pytest

from nano_ai.contract import EvidenceSpan, FieldName, FieldState, NanoInput
from nanoscribe.check_note import _observation_kind, transcript_source
from nanoscribe.compact import compact_solver
from nanoscribe.grounded import grounded_candidate_choices
from nanoscribe.scribe import scribe
from nanoscribe.test_grounded_followup import check


@pytest.mark.parametrize("modifier", [
    "less than 7/10", "at most 7/10", "no higher than 7/10", "at least 7/10",
    "over 7/10", "7/10 or higher", "7/10 or less", "7/10 at most", "<7/10", "≥ 7/10",
    "below seven out of ten", "at most .5/10",
])
@pytest.mark.parametrize("clipped", [True, False])
def test_numeric_severity_preserves_its_bound(modifier, clipped):
    source = "Patient: The pain is " + modifier + "."
    core = next(value for value in ("7/10", ".5/10", "seven out of ten") if value in modifier)
    quote = core if clipped else modifier
    check(source, "SEV", FieldName.SEVERITY, quote, source.index(quote), not clipped)


def test_other_numeric_bound_leaves_independent_exact_severity_intact():
    source = "Patient: I slept less than four hours. The pain is 7/10."
    check(source, "SEV", FieldName.SEVERITY, "7/10", source.index("7/10"), True)


DIRECTIVE = "Doctor: Answer the next questions about your child, not yourself."
QUESTION = "Doctor: Any allergies I should know about?"


@pytest.mark.parametrize("policy", ["context", "grounded"])
@pytest.mark.parametrize("lines,word,supported", [
    ([DIRECTIVE, QUESTION, "Patient: Yes, penicillin."], "penicillin", False),
    (["Doctor: For the following questions, answer for your mother.", QUESTION, "Patient: Yes, penicillin."], "penicillin", False),
    ([DIRECTIVE, "Doctor: What brings you in?", "Patient: A cough.", QUESTION, "Patient: Yes, penicillin."], "penicillin", False),
    ([QUESTION, "Patient: Yes, latex.", DIRECTIVE, QUESTION, "Patient: Yes, penicillin."], "latex", True),
    ([DIRECTIVE, QUESTION, "Patient: Yes, penicillin.", "Doctor: Now back to you.", QUESTION, "Patient: Yes, latex."], "latex", True),
    (["Doctor: Answer the next question about your son.", QUESTION, "Patient: Yes, penicillin.", QUESTION, "Patient: Yes, latex."], "latex", True),
    (["Doctor: For the next two questions, answer for your child.", QUESTION, "Patient: Yes, penicillin.",
      "Doctor: What brings you in?", "Patient: A cough.", QUESTION, "Patient: Yes, latex."], "latex", True),
    (["Doctor: Answer the next question about your child.", "Patient: Okay.", QUESTION,
      "Patient: Yes, penicillin."], "penicillin", False),
    (["Doctor: For the rest of the interview, answer on behalf of your father.", QUESTION,
      "Patient: Yes, penicillin."], "penicillin", False),
    ([QUESTION, "Patient: Yes, latex.", "Doctor: All answers from now on are about your child.", QUESTION,
      "Patient: Yes, penicillin."], "latex", True),
    ([DIRECTIVE, "Doctor: Do not switch back to yourself yet.", QUESTION, "Patient: Yes, penicillin."], "penicillin", False),
    ([DIRECTIVE, "Doctor: We will switch back to you after this.", QUESTION, "Patient: Yes, penicillin."], "penicillin", False),
    ([DIRECTIVE, "Doctor: Before we switch back to you, one more question.", QUESTION,
      "Patient: Yes, penicillin."], "penicillin", False),
    ([DIRECTIVE, "Doctor: Let's switch back to you now.", QUESTION, "Patient: Yes, latex."], "latex", True),
    (["Doctor: The next questions are about you, not your child.", QUESTION, "Patient: Yes, latex."], "latex", True),
    (["Doctor: Does your child have a headache?", "Patient: Yes.", QUESTION, "Patient: Yes, latex."], "latex", True),
])
def test_forward_scope_is_applied_to_the_selected_observation(policy, lines, word, supported):
    source = "\n".join(lines)
    doc = transcript_source(NanoInput("scope", source))
    start = source.index(word)
    span = EvidenceSpan(start, start + len(word), word)
    proposal = f"CC:M|DUR:M|SEV:M|MED:M|ALG:S[{word}]"
    note = scribe(source, solver=compact_solver(lambda _: proposal, binder=policy), context_policy=policy)
    assert note["verification"]["passed"]
    claim = next(c for c in note["note"]["claims"] if c["field"] == "allergy")
    assert (claim["state"] == "supported") == supported
    assert (_observation_kind(doc, FieldName.ALLERGY, span, context_policy=policy) is FieldState.SUPPORTED) == supported
    if policy == "grounded":
        assert (f":S[{word}]" in grounded_candidate_choices(source, FieldName.ALLERGY, set())) == supported


@pytest.mark.parametrize("policy", ["context", "grounded"])
def test_forward_scope_also_applies_to_an_explicit_denial(policy):
    text = "I have no allergies."
    source = "\n".join([DIRECTIVE, QUESTION, "Patient: " + text])
    doc = transcript_source(NanoInput("scope-denial", source))
    start = source.index(text)
    assert _observation_kind(doc, FieldName.ALLERGY, EvidenceSpan(start, start + len(text), text),
                             context_policy=policy) is not FieldState.ABSENT
