"""Conservative development context binder, separate from the v0 reference.

Only question attribution changes. Reply grammars and conservative conflict
resolution are inherited from the bounded synthetic instrument, not a general
clinical language verifier. Unknown, compound, historical or third-party
questions block assertions and prevent a verified missing-state claim.
"""
import re

from nano_ai.adapters.deterministic_v0 import (
    _observe_reply, _question_field, _resolve, _turns,
)
from nano_ai.contract import FIELD_ORDER, FieldName, FieldOutput, FieldState

VERSION = "context-questions-v1"
_BLOCK = re.compile(r"\b(mother|father|sister|brother|family|child|wife|husband|partner|pet|previous|previously|last|past|historical|history|used to|if|hypothetical|repeat|say|ignore|pretend|not|never|no)\b")
_INTENTS = {
    FieldName.MEDICATION: r"\b(medicines?|medications?|drugs?|taken anything)\b",
    FieldName.ALLERGY: r"\b(allerg\w*)\b",
    FieldName.DURATION: r"\b(duration|how long|length of time|when.*start|since when|how many days|onset)\b",
    FieldName.SEVERITY: r"\b(sever\w*|intens\w*|how bad|mild|moderate)\b",
    FieldName.CHIEF_COMPLAINT: r"\b(brings|brought|bothering|trouble|issue|problem|symptoms|going on)\b",
}
_QUESTION_WORDS = frozenset("please what which how when since for any are do have did describe tell me about substances you your to is the of this it has been going on long many days start length time noticed symptoms symptom problem discomfort feels feel intense intensity severe severity mild moderate bad duration onset brings brought bothering trouble issue here that medicines medicine medications medication drugs drug taken anything taking used try allergies allergy allergic known know should today now".split())


def question_field(text):
    question = text.strip().casefold()
    legacy = _question_field(text)
    if legacy is not None:
        return legacy
    if _BLOCK.search(question):
        return None
    # A bounded vocabulary prevents a recognized field noun from authorizing
    # unknown subjects, temporal modifiers, negation or compound requests.
    if not set(re.findall(r"[\w']+", question)) <= _QUESTION_WORDS:
        return None
    if not re.match(r"(?:please\s+)?(?:what|which|how|when|since|for what|any|are|do|have|did|describe|tell)\b", question):
        return None
    found = {field for field, pattern in _INTENTS.items() if re.search(pattern, question)}
    # A generic problem/symptom mention is subordinate to a specific attribute.
    # E.g. duration of symptoms; medicine for this problem. Two specific intents
    # remain ambiguous rather than guessing which part the Patient answered.
    specific = found - {FieldName.CHIEF_COMPLAINT}
    if len(specific) == 1:
        return next(iter(specific))
    if not specific and found:
        return FieldName.CHIEF_COMPLAINT
    return None


def extract_context_fields(item):
    observed = {field: [] for field in FIELD_ORDER}
    turns = _turns(item.transcript)
    unknown = False
    # Do not silently skip unparsed role/content lines or orphan replies.
    if len(turns) != len([line for line in item.transcript.splitlines() if line.strip()]):
        unknown = True
    for index, turn in enumerate(turns):
        if turn.speaker != "doctor":
            if index == 0 or turns[index - 1].speaker != "doctor":
                unknown = True
            continue
        field = question_field(turn.text)
        if field is None:
            unknown = True
            continue
        reply = turns[index + 1] if index + 1 < len(turns) and turns[index + 1].speaker == "patient" else None
        observed[field].append(_observe_reply(field, reply))
    result = []
    for field in FIELD_ORDER:
        value = _resolve(field, observed[field])
        # Unknown context could contain a competing observation of any field.
        if unknown:
            value = FieldOutput(field, FieldState.UNCERTAIN, evidence=value.evidence)
        result.append(value)
    return tuple(result)
