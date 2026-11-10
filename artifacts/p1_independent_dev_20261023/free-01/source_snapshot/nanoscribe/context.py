"""Conservative development context binder, separate from the v0 reference.

Complete reply scope is checked before field-local failures are allowed.
This is a bounded development grammar, not a general clinical verifier.
"""
import re

from nano_ai.adapters.deterministic_v0 import (
    _Observation, _question_field, _resolve,
)
from nano_ai.contract import FIELD_ORDER, FieldName, FieldOutput, FieldState, normalize_value

VERSION = "context-source-scope-v2"
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
    from nanoscribe.check_note import inspect_context, transcript_source
    observed = {field: [] for field in FIELD_ORDER}
    observations, unknown = inspect_context(transcript_source(item))
    for observation in observations:
        observed[observation.field].append(_Observation(
            observation.state,
            normalize_value(observation.span.text) if observation.state is FieldState.SUPPORTED else None,
            (observation.span,) if observation.span else (),
        ))
    result = []
    for field in FIELD_ORDER:
        value = _resolve(field, observed[field])
        evidence = tuple(span for observation in observed[field] for span in observation.evidence)
        if value.state is FieldState.CONFLICTING:
            # Different medicines/allergens can coexist. The scalar schema
            # cannot represent a list, but that does not establish contradiction.
            has_denial = any(observation.state is FieldState.ABSENT for observation in observed[field])
            state = (FieldState.UNCERTAIN if field in (FieldName.MEDICATION, FieldName.ALLERGY) and not has_denial
                     else FieldState.CONFLICTING)
            value = FieldOutput(field, state, evidence=evidence)
        # Unknown context could contain a competing observation of any field.
        if unknown:
            value = FieldOutput(field, FieldState.UNCERTAIN, evidence=evidence)
        result.append(value)
    return tuple(result)
