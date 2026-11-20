"""Conservative development context binder, separate from the v0 reference.

Complete reply scope is checked before field-local failures are allowed.
This is a bounded development grammar, not a general clinical verifier.
"""
import re

from nano_ai.adapters.deterministic_v0 import (
    _Observation, _question_field, _resolve,
)
from nano_ai.contract import FIELD_ORDER, FieldName, FieldOutput, FieldState, normalize_value

VERSION = "context-source-scope-v4"
# Attribute a *request*, not a field noun appearing somewhere in a sentence.
# Noun phrases are bounded independently from the question operator, so an
# ordinary symptom modifier does not require adding every word to a global
# vocabulary. These are development rules, not general clinical entailment.
_BLOCK = re.compile(
    r"\b(mother|father|sister|brother|family|child|children|daughter|son|wife|husband|"
    r"partner|pet|uncle|aunt|cousin|grandmother|grandfather|friend|someone|anyone|"
    r"else|he|she|they|his|her|their|previous|previously|old|last|past|historical|"
    r"history|used to|if|hypothetical|repeat|ignore|pretend|not|never|no|"
    r"tomorrow|next|future|planning|planned|plan|will|recommend|recommended|suggest)\b"
    r"|\b\w+n't\b"
)
_MEDICINE = r"(?:medicines?|medications?|drugs?|tablets?|pills?|remedies|remedy)"
_SYMPTOM = (
    r"(?:symptoms?|problems?|concerns?|discomfort|pain|ache|headache|cough|"
    r"nausea|dizziness|vertigo|ringing|swelling|rash|itching|fever|fatigue|"
    r"soreness|numbness|tingling|vomiting|diarrhea|bleeding|breathlessness)"
)
_WORD = r"[^\W\d_]+(?:-[^\W\d_]+)*"
_NOUN_PHRASE = rf"(?:{_WORD}\s+){{0,4}}{_SYMPTOM}"
_REFERENCE = rf"(?:it|this|that|(?:the|your|this|that|these|those)\s+{_NOUN_PHRASE})"
_BODY_REFERENCE = rf"(?:the|your|this|that)\s+(?:{_WORD}\s+){{0,4}}{_WORD}"
_CURRENT_TIME = r"(?:\s+(?:today|now|at the moment|currently|so far))?"
_FOR_REFERENCE = rf"(?:\s+(?:for|to help with)\s+{_REFERENCE})?"
_RATINGS = r"(?:mild|moderate|severe)(?:,?\s+(?:or\s+)?(?:mild|moderate|severe))*"
_QUESTION_PATTERNS = {
    FieldName.CHIEF_COMPLAINT: (
        r"what (?:symptoms?|problem|trouble|issue)",
        r"what symptoms? are you (?:having|experiencing)" + _CURRENT_TIME,
        r"what is troubling you" + _CURRENT_TIME,
        r"what (?:symptoms? )?(?:brings|brought) you (?:in|here|to (?:the|our) (?:clinic|office))" + _CURRENT_TIME,
        r"what (?:is|has been) bothering you" + _CURRENT_TIME,
        r"what (?:is|seems to be) (?:the|your) (?:main )?(?:problem|trouble|issue|concern)" + _CURRENT_TIME,
        r"(?:describe|tell me about) (?:the|your) (?:main )?(?:symptoms?|problem|concern|issue)(?: you are having)?" + _CURRENT_TIME,
        r"tell me what (?:is|has been) (?:going on|bothering you)" + _CURRENT_TIME,
    ),
    FieldName.DURATION: (
        r"how long",
        rf"how long (?:has|have) {_REFERENCE} (?:hurt|ached)",
        rf"how long (?:has|have) {_REFERENCE} (?:been (?:going on|there|present)|lasted|persisted)",
        rf"how long (?:has|have) {_BODY_REFERENCE} been (?:hurting|aching|swollen|sore)",
        rf"how long have you (?:had|experienced) {_REFERENCE}",
        r"how long have you (?:felt this way|been feeling this way)",
        rf"when did {_REFERENCE} (?:first )?(?:start|begin|appear)",
        rf"when did you (?:first )?(?:notice|develop) {_REFERENCE}",
        rf"how many (?:days|weeks) (?:has|have) {_REFERENCE} (?:lasted|persisted|been present)",
        rf"what is the (?:duration|length of time) of {_REFERENCE}",
        rf"since when have you had {_REFERENCE}",
    ),
    FieldName.SEVERITY: (
        r"how (?:bad|severe|intense)",
        rf"how (?:painful|strong|troublesome) (?:is|are) {_REFERENCE}" + _CURRENT_TIME,
        rf"how (?:bad|severe|intense) (?:is|are) {_REFERENCE}" + _CURRENT_TIME,
        rf"how (?:bad|severe|intense) (?:would you say|does) {_REFERENCE} (?:is|feel)",
        rf"how would you (?:rate|describe) {_REFERENCE}(?: in severity)?",
        rf"would you (?:rate|describe|call) {_REFERENCE} (?:as )?{_RATINGS}",
        rf"is {_REFERENCE} {_RATINGS}",
        rf"how much does {_BODY_REFERENCE} hurt",
        rf"(?:describe|tell me) how (?:severe|intense) {_REFERENCE} is",
        rf"what is the (?:severity|intensity) of {_REFERENCE}",
    ),
    FieldName.MEDICATION: (
        rf"any {_MEDICINE}" + _FOR_REFERENCE + _CURRENT_TIME,
        r"what are you (?:taking|using)" + _FOR_REFERENCE + _CURRENT_TIME,
        rf"(?:what|which) {_MEDICINE}" + _FOR_REFERENCE + _CURRENT_TIME,
        rf"(?:what|which) {_MEDICINE} (?:are you (?:taking|using)|have you (?:taken|used|tried)|did you (?:take|use|try))" + _FOR_REFERENCE + _CURRENT_TIME,
        rf"(?:have you|did you) (?:taken|take|used|use|tried|try) (?:anything|any {_MEDICINE})" + _FOR_REFERENCE + _CURRENT_TIME,
        rf"are you (?:taking|using|on) (?:anything|any {_MEDICINE}|{_MEDICINE})" + _FOR_REFERENCE + _CURRENT_TIME,
        rf"what (?:have you (?:taken|used|tried)|did you (?:take|use|try))" + _FOR_REFERENCE + _CURRENT_TIME,
        rf"(?:what|which) {_MEDICINE} (?:do you take|do you use)" + _FOR_REFERENCE + _CURRENT_TIME,
    ),
    FieldName.ALLERGY: (
        rf"(?:do you have |are there )?any (?:known )?(?:{_MEDICINE} )?allerg(?:y|ies)(?: (?:i|we) should know about)?",
        rf"(?:do you have |are there )?any (?:known )?allerg(?:y|ies) to (?:any )?{_MEDICINE}",
        rf"are you allergic to (?:anything|(?:any )?{_MEDICINE})",
        rf"(?:are there|is there) any {_MEDICINE} (?:that )?you are allergic to",
        rf"(?:what|which) (?:substances|{_MEDICINE}) are you allergic to",
        r"(?:what|which) allergies (?:do you have|should i know about)",
        r"(?:describe|tell me about) your (?:known )?allergies",
    ),
}


def _normalized_question(text):
    question = " ".join(text.strip().casefold().replace("’", "'").split())
    question = re.sub(r"^what's been\b", "what has been", question)
    question = re.sub(r"^what's\b", "what is", question)
    question = re.sub(r"^(?:hello|hi|good morning)[, ]+", "", question)
    question = re.sub(r"^(?:please\s+)?(?:can|could) you (?=describe|tell me)", "", question).strip()
    question = re.sub(r"^please\s+", "", question)
    return question.rstrip("?.!").strip()


def question_targets(text):
    """Possible requested fields for quarantine only, never positive evidence.

    Unlike question_field, this intentionally retains targets of third-person,
    hypothetical, or compound requests. A medicine noun modifying an allergy
    request does not independently request medication-use information.
    """
    question = _normalized_question(text)
    legacy = _question_field(text)
    if legacy is not None:
        return frozenset({legacy})
    found = set()
    if re.search(r"\ballerg\w*\b", question):
        found.add(FieldName.ALLERGY)
    medicine = re.search(rf"\b{_MEDICINE}\b", question)
    medicine_use = re.search(r"\b(?:taken|taking|take|used|using|use|tried|try)\b", question)
    coordinated_medicine = re.search(
        rf"\b(?:{_MEDICINE}\s+(?:and|or)\s+(?:(?:what|which|your|any)\s+)?allerg\w*|"
        rf"allerg\w*\s+(?:and|or)\s+(?:(?:what|which|your|any)\s+)?{_MEDICINE})\b", question)
    if (medicine and (FieldName.ALLERGY not in found or medicine_use or coordinated_medicine)
            or re.search(r"\b(?:taken|taking|used|using|tried|try) anything\b", question)):
        found.add(FieldName.MEDICATION)
    if re.search(r"\b(?:duration|how long|length of time|when did|since when|how many (?:days|weeks)|onset)\b", question):
        found.add(FieldName.DURATION)
    if re.search(r"\b(?:sever\w*|intens\w*|how bad|how much.*hurt|mild|moderate|rate)\b", question):
        found.add(FieldName.SEVERITY)
    complaint = re.search(r"\b(?:brings|brought|bothering|trouble|issue|problem|symptoms?|concern|going on)\b", question)
    coordinated_complaint = re.search(
        rf"\b(?:symptoms?|problems?|concerns?)\s+(?:and|or)\s+(?:(?:what|which|your|any)\s+)?(?:{_MEDICINE}|allerg\w*|duration|severity)\b", question)
    if complaint and (not found or coordinated_complaint):
        found.add(FieldName.CHIEF_COMPLAINT)
    # Structural requests such as 'What have you used for the ache?' can lack
    # an explicit field noun. Their full request pattern supplies the target.
    found.update(field for field, patterns in _QUESTION_PATTERNS.items()
                 if any(re.fullmatch(pattern, question) for pattern in patterns))
    return frozenset(found)


def question_field(text):
    legacy = _question_field(text)
    if legacy is not None:
        return legacy
    question = _normalized_question(text)
    if _BLOCK.search(question) or re.search(r"[?.!;:]|\b(?:and|but|then)\b", question):
        return None
    # Full request structure, rather than every mentioned field word, decides
    # attribution. For example, 'mild swelling' is the symptom whose onset is
    # requested; its adjective does not also request a severity answer.
    targets = {field for field, patterns in _QUESTION_PATTERNS.items()
               if any(re.fullmatch(pattern, question) for pattern in patterns)}
    if len(targets) != 1:
        return None
    return next(iter(targets))


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
        # Explicit unbounded source reassignment invalidates all patient claims.
        if unknown:
            value = FieldOutput(field, FieldState.UNCERTAIN, evidence=evidence)
        result.append(value)
    return tuple(result)
