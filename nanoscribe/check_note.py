"""Independent, deliberately narrow source and draft checks for the P1 bridge.

This is a deterministic source-context allowlist, not clinical entailment or a
learned verifier. It never calls a solver or the note renderer. Passing means
the bounded representation/quotation checks passed; human review still applies.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from nano_ai.contract import (
    FIELD_ORDER, EvidenceSpan as CandidateSpan, FieldName, FieldState,
    NanoInput, NanoOutput,
)
from nanoscribe.encounter import (
    AssertionState, AtomType, Certainty, EncounterRecord, Experiencer,
    Source, Speaker, TemporalState, Temporality, Turn, normalize_value,
)


# Intentionally narrower than the reference extractor's synthetic grammar.
# New phrasing must be reviewed, rather than silently broadening this policy.
QUESTIONS = {
    FieldName.CHIEF_COMPLAINT: {"good morning, what brings you in today?", "what brings you to the clinic today?"},
    FieldName.DURATION: {"how long has this been going on?", "when did it start?"},
    FieldName.SEVERITY: {"how bad would you say it is?", "is it mild, moderate, or severe?"},
    FieldName.MEDICATION: {"have you taken anything for it?", "did you try any medicine?"},
    FieldName.ALLERGY: {"any allergies i should know about?", "do you have any known allergies?"},
}
_POSITIVE = {
    FieldName.CHIEF_COMPLAINT: ("I've been having {}.", "I came in because of {}."),
    FieldName.DURATION: ("For about {} now.", "For about {}.", "It started {} ago."),
    FieldName.SEVERITY: ("I'd call it {}.", "It's {}, I would say.", "Definitely {}."),
    FieldName.MEDICATION: ("I've been taking {}.", "Just {}.", "Only {} so far."),
    FieldName.ALLERGY: ("I'm allergic to {}.", "Yes, {}.", "Just {}."),
}
_NEGATIVE = {
    FieldName.MEDICATION: {"no, nothing yet.", "i haven't taken anything.", "nothing at all."},
    FieldName.ALLERGY: {"no allergies.", "none whatsoever."},
}
_UNCERTAINTY = {
    FieldName.DURATION: r"I cannot (?:remember how long it has been|recall when it started)\.",
    FieldName.SEVERITY: r"I (?:cannot rate it|am not sure how intense it is)\.",
    FieldName.MEDICATION: r"I (?:cannot remember what I took|am not sure whether I took anything)\.",
    FieldName.ALLERGY: r"Not that I know of\.",
}
_CONTEXT_POSITIVE = {
    **_POSITIVE,
    FieldName.CHIEF_COMPLAINT: (*_POSITIVE[FieldName.CHIEF_COMPLAINT], "I have {}.", "I've got {}.", "I'm dealing with {}.", "I am here because of {}."),
    FieldName.DURATION: (*_POSITIVE[FieldName.DURATION], "For {}.", "About {}.", "{} ago.", "It began {} ago.", "It's been {}.", "It has been {}.", "I'd say {}."),
    FieldName.SEVERITY: (*_POSITIVE[FieldName.SEVERITY], "It feels {}.", "It's {}.", "It is {}.", "I'd say {}.", "I would call it {}."),
    FieldName.MEDICATION: (*_POSITIVE[FieldName.MEDICATION], "Some {}, but it barely helps.", "I took {}.", "I take {}.", "I am taking {}."),
    FieldName.ALLERGY: (*_POSITIVE[FieldName.ALLERGY], "I have an allergy to {}.", "I am allergic to {}."),
}
_QUALIFIERS = frozenset("no not none nothing neither nor never zero free without denies denied uncertain unknown unsure maybe perhaps possibly possible probably apparently reportedly suspected likely unlikely might may could would should if unless or but except although because that which who before after previously past history resolved stopped tomorrow yesterday next last used formerly my his her their mother father child daughter son wife husband partner think guess believe seem seems sometimes intermittent occasionally almost barely".split())
# A positive reply wrapper does not turn non-use, contemplated use, or an
# epistemic modifier into a scalar name. Exclude these contextual words even
# after an article; this remains a lexical guard, not general entailment.
_CONTEXTUAL_VALUE_WORDS = frozenset(
    "avoid avoided avoiding avoidance skip skipped skipping withhold withheld withholding "
    "stop stopping discontinue discontinued discontinuing consider considered considering "
    "potential potentially".split()
)
# This bridge cannot reconcile a complaint that also asserts an allergy or
# medication relationship. Quarantine it instead of treating its field label as
# proof that it cannot contradict another field (for example, an allergy denial).
_COMPLAINT_RELATION_WORDS = frozenset(
    "allergy allergies allergic medication medications medicine medicines drug drugs "
    "effect effects reaction reactions".split()
)
_NONVALUE_WORDS = frozenset(
    "i you we it this these those something anything everything someone somebody "
    "yes okay ok sure don't do does did take took taking have has had am is are was were "
    "can cannot can't remember know recall identify cousin neighbor neighbour and with for plus versus vs "
    "nope negative nil nah clear unavailable available untried untaken recommended suggested prescribed only just".split()
)
_UNNAMED_TREATMENT_WORDS = frozenset(
    "pill pills tablet tablets capsule capsules dose doses medication medications medicine medicines "
    "drug drugs syrup liquid cream ointment treatment treatments remedy remedies painkiller painkillers "
    "white blue red yellow green pink brown round oval one two three four five six seven eight nine ten".split()
)
_REACTION_ONLY_WORDS = frozenset("rash hives itching swelling nausea reaction reactions".split())
LABELS = dict(zip(FIELD_ORDER, ("Chief complaint", "Duration", "Severity", "Medication", "Allergy")))
HEADER = "Patient-reported encounter draft — human review required"
SCOPE_REASON = "Five patient-reported fields only; source checks are bounded and human review is required. No diagnosis, plan, or full encounter completeness is established."
UNHANDLED_REASON = "Content outside the checked five-field path; inspect for omissions or corrections."
CONFLICT_NOTE = "Conflicting observations retained; no value selected."


def transcript_source(request: NanoInput) -> Source:
    """Keep original characters/offsets, including CRLF; never reassemble text."""
    turns = []
    offset = 0
    for index, line in enumerate(request.transcript.splitlines(keepends=True)):
        text = line.rstrip("\r\n")
        match = re.match(r"\s*(Doctor|Patient)\s*:[ \t]*", text, re.IGNORECASE)
        content_start = match.end() if match else 0
        start = content_start + len(text[content_start:]) - len(text[content_start:].lstrip())
        end = len(text.rstrip())
        if start < end:
            speaker = (Speaker.PATIENT if match.group(1).casefold() == "patient" else Speaker.CLINICIAN) if match else Speaker.UNKNOWN
            turns.append(Turn(f"source:t{index}", "source", speaker, offset + start, offset + end, text[start:end]))
        elif text.strip():
            # Do not erase an empty role-labelled turn and accidentally attach
            # the following reply to an earlier question across that boundary.
            start = len(text) - len(text.lstrip())
            turns.append(Turn(f"source:t{index}", "source", Speaker.UNKNOWN, offset + start, offset + end, text[start:end]))
        offset += len(line)
    return Source("source", request.transcript, tuple(turns))


def _turn_index(source: Source, start: int, end: int) -> int | None:
    return next((i for i, turn in enumerate(source.turns) if turn.start <= start < end <= turn.end), None)


def uncovered_turns(source: Source, output: NanoOutput, *, context_policy: str = "legacy") -> tuple[Turn, ...]:
    """Unrepresented context from any speaker can change a field's meaning."""
    covered = {_turn_index(source, span.start, span.end) for field in output.fields for span in field.evidence}
    known_questions = set().union(*QUESTIONS.values())
    if context_policy in ("context", "grounded"):
        from nanoscribe.context import question_field
        known_questions |= {turn.text.casefold() for turn in source.turns
                            if turn.speaker is Speaker.CLINICIAN and question_field(turn.text) is not None}
        if context_policy == "grounded":
            for idx in list(covered):
                if idx is not None and idx > 0 and source.turns[idx - 1].speaker is Speaker.CLINICIAN:
                    prev_text = source.turns[idx - 1].text
                    if not _outside_current_scope(prev_text) and not _source_reassignment(prev_text):
                        covered.add(idx - 1)
        # A quoted value does not cover the rest of a multi-clause reply.
        # Retain such turns for review even when one clause supplies a fact.
        covered -= {i for i, turn in enumerate(source.turns)
                    if turn.speaker is Speaker.PATIENT and len(_reply_clauses(turn)) > 1}
    return tuple(turn for i, turn in enumerate(source.turns)
                 if i not in covered and (turn.speaker is not Speaker.CLINICIAN or turn.text.casefold() not in known_questions))


def _plain_value(field: FieldName, text: str) -> bool:
    if field is FieldName.DURATION:
        return re.fullmatch(r"(?:[1-9]\d{0,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve) (?:hours?|days?|weeks?|months?)", text.casefold()) is not None
    if field is FieldName.SEVERITY:
        return text.casefold() in {"mild", "moderate", "severe"}
    tokens = re.findall(r"\w+", text.casefold())
    return (0 < len(tokens) <= 5 and len(text) <= 80
            and re.fullmatch(r"[^\W_]+(?:[ -][^\W_]+)*", text, re.UNICODE) is not None
            and not _QUALIFIERS.intersection(tokens)
            and not _CONTEXTUAL_VALUE_WORDS.intersection(tokens)
            and not _NONVALUE_WORDS.intersection(tokens)
            and not (field is FieldName.MEDICATION and _UNNAMED_TREATMENT_WORDS.intersection(tokens))
            and not (field is FieldName.ALLERGY and _REACTION_ONLY_WORDS.intersection(tokens))
            and not (field in (FieldName.MEDICATION, FieldName.ALLERGY) and _COMPLAINT_RELATION_WORDS.intersection(tokens))
            and not (field is FieldName.CHIEF_COMPLAINT and _COMPLAINT_RELATION_WORDS.intersection(tokens)))


def bounded_observation(field: FieldName, turn: Turn) -> tuple[FieldState, CandidateSpan] | None:
    """Recognize a whole reply, never just the model's selected substring.

    The caller must establish patient/question attribution. These development patterns
    are deliberately bounded, not a general semantic or clinical verifier.
    """
    if turn.speaker is not Speaker.PATIENT:
        return None
    whole = CandidateSpan(turn.start, turn.end, turn.text)
    # Complete blanket medication denials only. A named drug or a restricted
    # scope such as 'no regular medication' cannot contradict all medication use.
    if field is FieldName.MEDICATION and re.fullmatch(
        r"(?:no (?:medications?|medicine|meds?)|i (?:deny|denied) (?:taking )?(?:medications?|medicine|meds?))[.!]?",
        turn.text, re.IGNORECASE,
    ):
        return FieldState.ABSENT, whole
    if turn.text.casefold() in _NEGATIVE.get(field, ()):
        return FieldState.ABSENT, whole
    if field in _UNCERTAINTY and re.fullmatch(_UNCERTAINTY[field], turn.text, re.IGNORECASE):
        return FieldState.UNCERTAIN, whole
    # Bounded adjuncts describe an actual administration or explicit allergy
    # relation. They are parsed as part of the whole clause, never stripped
    # from an arbitrary sentence containing a model-selected medicine name.
    patterns = {
        FieldName.MEDICATION: (
            r"I (?:took|used|have taken|have used|am taking|am using) (?P<value>.+?)"
            r"(?: (?:today|this morning|this evening|last night))?[.]?",
        ),
        FieldName.ALLERGY: (
            r"(?P<value>.+?) (?:gives me|causes me to develop) (?:hives|a rash|itching|swelling)[.]?",
        ),
        FieldName.SEVERITY: (r"(?:It(?:'s| is| feels)|The pain is) (?P<value>mild|moderate|severe)(?:,? really)?[.]?",),
    }
    for pattern in patterns.get(field, ()):
        match = re.fullmatch(pattern, turn.text, re.IGNORECASE)
        if match and _plain_value(field, match.group('value')):
            return FieldState.SUPPORTED, CandidateSpan(
                turn.start + match.start('value'), turn.start + match.end('value'), match.group('value'))
    for template in _CONTEXT_POSITIVE[field]:
        before, after = template.split("{}")
        match = re.fullmatch(re.escape(before) + r"(.+?)" + re.escape(after), turn.text, re.IGNORECASE)
        if match and _plain_value(field, match.group(1)):
            if field in (FieldName.MEDICATION, FieldName.ALLERGY):
                words = set(re.findall(r"\w+", match.group(1).casefold()))
                if words.intersection({"and", "with", "for", "plus", "versus", "vs"} | _COMPLAINT_RELATION_WORDS):
                    continue
            return FieldState.SUPPORTED, CandidateSpan(
                turn.start + match.start(1), turn.start + match.end(1), match.group(1))
    # Short complete replies are common after an explicit field question. Do
    # not mine a substring of a qualified sentence or guess an implicit value.
    bare = re.fullmatch(r"(.+?)[.]?", turn.text)
    if bare and _plain_value(field, bare.group(1)) and (
        field is not FieldName.CHIEF_COMPLAINT or re.search(
            r"\b(?:pain|ache|headache|cough|nausea|dizziness|rash|itching|fever|fatigue|"
            r"soreness|numbness|tingling|vomiting|diarrhea|bleeding|sore|swelling|stiffness)\b", bare.group(1), re.IGNORECASE)
    ):
        if not (field in (FieldName.MEDICATION, FieldName.ALLERGY)
                and _COMPLAINT_RELATION_WORDS.intersection(re.findall(r"\w+", bare.group(1).casefold()))):
            return FieldState.SUPPORTED, CandidateSpan(
                turn.start + bare.start(1), turn.start + bare.end(1), bare.group(1))
    return None


@dataclass(frozen=True)
class SourceObservation:
    field: FieldName
    state: FieldState
    span: CandidateSpan | None


def _reply_clauses(turn: Turn) -> tuple[Turn, ...]:
    """Split sentence/semicolon boundaries while preserving original offsets."""
    return tuple(Turn(turn.turn_id, turn.source_id, turn.speaker,
                      turn.start + match.start(), turn.start + match.end(), match.group())
                 for match in re.finditer(
                     r"(?:[^\s.!?;]|(?<!\w)\.(?=\d))(?:[^.!?;]|(?:(?<=\d)|(?<!\w))\.(?=\d))*[.!?;]*",
                     turn.text)
                 if match.group().strip())


def scoped_reply_observations(field: FieldName | None, turn: Turn,
                              previous_observations=(), antecedent_fields=()) -> tuple[SourceObservation, ...]:
    """Bind complete clauses; unknown clauses cannot disappear after a match.

    Only the first clause can inherit the requested field for a bare answer.
    Later clauses require an explicit patient assertion or a local uncertainty
    target. Corrections and source reassignment never become positive evidence.
    """
    if turn.speaker is not Speaker.PATIENT:
        return ()
    direct = bounded_observation(field, turn) if field is not None else None
    if direct is not None:
        return (SourceObservation(field, *direct),)
    clauses = _reply_clauses(turn)
    result = []
    for index, clause in enumerate(clauses):
        # Removing only delimiter/outer whitespace is safe for grammar matching;
        # every evidence span still refers to the untouched source bytes.
        content = clause.text.rstrip(';').rstrip()
        clause = Turn(clause.turn_id, clause.source_id, clause.speaker,
                      clause.start, clause.start + len(content), content)
        words = clause.text.casefold()
        correction = bool(re.search(r"\b(?:actually|correction|meant|mistaken|wrong|incorrect|"
                                    r"rather|instead|sorry|not really|after all)\b", words))
        explicit = []
        if not correction:
            for target in FIELD_ORDER:
                # Bare scalar replies are attributed only to a known request.
                # A first-person sentence or explicit reaction relation can
                # establish its own field, including in a compound answer.
                if ((index == 0 and target is field)
                        or re.match(r"(?:i\b|i['’]|it(?:'s| is| has been| began| started| feels)\b|the pain is\b)", words)
                        or (target is FieldName.ALLERGY and re.search(r"\b(?:gives me|causes me to develop)\b", words))):
                    observation = bounded_observation(target, clause)
                    if observation is not None:
                        explicit.append(SourceObservation(target, *observation))
        # An assertion with ambiguous field semantics must stay unresolved.
        if len(explicit) == 1:
            result.extend(explicit)
            continue
        # Discourse-only clauses remain reviewable source, without making an
        # otherwise concrete observation uncertain through a pronoun match.
        if re.fullmatch(r"(?:thank you|thanks|that's all|that is all|okay|ok|"
                        r"i remember (?:the date|that clearly))[.!]?", words):
            continue
        targets = _mentioned_fields(clause.text, (*previous_observations, *result))
        if not targets and (correction or re.search(r"\b(?:it|that|this)\b", words)):
            if result:
                targets.add(result[-1].field)
            elif field is not None:
                targets.add(field)
            else:
                targets.update(antecedent_fields)
        # Discourse-only clauses neither assert nor contradict a field. Their
        # text is still retained as uncovered source for review.
        if not targets and field is not None:
            targets.add(field)
        span = CandidateSpan(clause.start, clause.end, clause.text)
        result.extend(SourceObservation(target, FieldState.UNCERTAIN, span) for target in targets)
    if field is not None and not any(item.field is field for item in result):
        result.append(SourceObservation(field, FieldState.UNCERTAIN,
                                        CandidateSpan(turn.start, turn.end, turn.text)))
    return tuple(result)


def observations_conflict(field: FieldName, observations: tuple[SourceObservation, ...]) -> bool:
    """Check a concrete pair without interpreting two coexisting names as C."""
    supported = [item for item in observations if item.field is field and item.state is FieldState.SUPPORTED]
    absent = [item for item in observations if item.field is field and item.state is FieldState.ABSENT]
    if supported and absent:
        return True
    return (field not in (FieldName.MEDICATION, FieldName.ALLERGY)
            and len({normalize_value(item.span.text) for item in supported}) > 1)


def _outside_current_scope(text: str) -> bool:
    """Explicit other-person/old-episode questions cannot update patient fields."""
    if re.search(r"\b(?:and|but|now|today|currently)\b|[;]", text.casefold()):
        return False  # Mixed current/other scope needs field-local quarantine.
    return bool(re.search(
        r"\b(?:your|his|her|their) (?:mother|father|sister|brother|child|daughter|son|wife|husband|partner|cousin|friend|neighbor|neighbour)\b"
        r"|\b(?:last year|last month|previous episode|previous visit|old episode|used to|in childhood)\b",
        text.casefold()))


_QUESTION_SERIES_RE = re.compile(
    r"\b(?:next|following|remaining|subsequent)\s+(?:(?P<count>\d{1,4}|few|several|one|two|three|four|five|six|seven|eight|nine|ten)\s+)?"
    r"(?P<unit>questions?|answers?|responses?)\b", re.I,
)
_OPEN_SERIES_RE = re.compile(
    r"\b(?:from (?:now|here) on|for the rest of (?:the|this|our) (?:visit|interview|questions|conversation))\b", re.I,
)
_COUNT_WORDS = "zero one two three four five six seven eight nine ten".split()
_PROXY_TARGET_RE = re.compile(
    r"\b(?:about|for|as|on behalf of)\s+(?:(?:your|the|his|her|their)\s+)?"
    r"(?:child(?:ren)?|son|daughter|mother|father|sibling|brother|sister|parent|partner|husband|wife|friend|"
    r"grandmother|grandfather|caregiver|him|her|them)\b", re.I,
)
# A proposed, delayed or negated switch is not an actual change of subject.
_SELF_SCOPE_RE = re.compile(
    r"(?:^|[.!;]\s*)(?:(?:okay|thanks|all right)[, ]+)?(?:now[ ,]+)?"
    r"(?:(?:(?:let['’]s|let us|we are|we['’]re)\s+)?(?:now\s+)?"
    r"(?:back to|(?:switch|return)(?:ing)? (?:back )?to|talk about|focus on)"
    r"|(?:the |these )?(?:(?:next|following|remaining) )?questions? (?:are |is )?about)"
    r"\s+(?:you(?:rself)?\b|your own\s+[\w -]+?)(?:\s+(?:now|again))?"
    r"(?:,\s*(?:not|rather than|instead of)\s+[^.!?;]+)?\s*(?=[.!?;]|$)", re.I,
)


def _proxy_question_count(text: str) -> int:
    """Zero means no directive, and -1 means an open-ended series."""
    if not _PROXY_TARGET_RE.search(text) or _SELF_SCOPE_RE.search(text):
        return 0
    series = _QUESTION_SERIES_RE.search(text)
    if series is None:
        return -1 if _OPEN_SERIES_RE.search(text) else 0
    count = (series["count"] or "").casefold()
    if count.isdigit():
        return int(count)
    if count in _COUNT_WORDS:
        return _COUNT_WORDS.index(count)
    return -1 if series["unit"].casefold().endswith("s") else 1


def _forward_proxy_turns(source: Source) -> set[int]:
    """Return patient turns attributed by an explicit forward clinician directive.

    A finite question ends at the next question after its answer, so adjacent
    patient continuations remain with the same experiencer. A neutral question
    leaves an open-ended series in effect until an explicit switch to the patient.
    """
    hidden = set()
    remaining = 0
    in_answer = False
    pending_question = False
    for index, turn in enumerate(source.turns):
        if turn.speaker is Speaker.CLINICIAN:
            declared = _proxy_question_count(turn.text)
            if declared:
                remaining = declared
                in_answer = False
                pending_question = turn.text.rstrip().endswith("?")
            elif _SELF_SCOPE_RE.search(turn.text):
                remaining = 0
                in_answer = pending_question = False
            elif turn.text.rstrip().endswith("?") or re.match(r"(?:tell me|list|describe)\\b", turn.text, re.I):
                in_answer = False
                pending_question = True
        elif turn.speaker is Speaker.PATIENT and (remaining or in_answer):
            hidden.add(index)
            if pending_question:
                if remaining > 0:
                    remaining -= 1
                in_answer = True
                pending_question = False
    return hidden


def _source_reassignment(text: str) -> bool:
    return bool(re.search(
        r"\b(?:those answers|these answers|all (?:my |the )?answers|everything (?:i|the patient) said|all of that)\b"
        r".*\b(?:about|belong|belonged|were|was|incorrect|wrong|untrue)\b"
        r"|\b(?:that|this|it) was (?:about |actually )?(?:my|his|her|their) \w+['’]s (?:experience|story|answer)\b"
        r"|\bi made (?:that|it|those answers) up\b", text.casefold()))


def _mentioned_fields(text: str, observations) -> set[FieldName]:
    """Conservative quarantine cues; these never authorize a positive claim."""
    from nanoscribe.context import question_targets
    targets = set(question_targets(text))
    words = text.casefold()
    if re.search(r"\b(?:take|took|taken|taking|medicine|medication|tablet|pill)s?\b"
                 r"|\b(?:i|you) (?:didn't |did not |never )?use (?:it|that|anything|any)\b", words):
        targets.add(FieldName.MEDICATION)
    if re.search(r"\b(?:days?|weeks?|months?|hours?)\b", words):
        targets.add(FieldName.DURATION)
    if (not targets.intersection({FieldName.DURATION, FieldName.SEVERITY})
            and re.search(r"\b(?:symptoms?|complaint|pain|ache|tenderness|stiffness)\b", words)):
        targets.add(FieldName.CHIEF_COMPLAINT)
    for item in observations:
        if item.state is FieldState.SUPPORTED and item.span:
            value = normalize_value(item.span.text)
            if value and re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", words):
                targets.add(item.field)
    return targets


def _effect_only_reply(question: str, reply: str) -> bool:
    """A bounded efficacy answer neither names nor retracts medication use."""
    return bool(re.fullmatch(
        r"(?:has (?:the |your )?(?:medication|medicine) helped|did (?:it|the medication|the medicine) help)[?]?",
        question.casefold()) and re.fullmatch(
        r"(?:it (?:helped|did not help|didn't help|has helped|hasn't helped)|"
        r"i (?:felt better|could finally sleep))[.!]?", reply.casefold()))


def inspect_context(source: Source) -> tuple[tuple[SourceObservation, ...], bool]:
    """Scan all original turns; return observations and a global scope hazard.

    Omitted or mislabelled model evidence cannot change source scope. A known
    question with no answer affects that field. Unknown replies quarantine
    their requested/mentioned fields; unrelated context remains uncovered for
    review. Only explicit unbounded source reassignment invalidates all fields.
    """
    from nanoscribe.context import question_field, _BLOCK
    observations = []
    global_unsafe = False
    proxy_turns = _forward_proxy_turns(source)
    for index, turn in enumerate(source.turns):
        if index in proxy_turns:
            continue
        if turn.speaker is not Speaker.CLINICIAN or not _proxy_question_count(turn.text):
            global_unsafe |= _source_reassignment(turn.text)
        if turn.speaker is Speaker.CLINICIAN:
            field = question_field(turn.text)
            if field is not None and (index + 1 == len(source.turns) or source.turns[index + 1].speaker is not Speaker.PATIENT):
                observations.append(SourceObservation(field, FieldState.MISSING, None))
            elif field is None and not _outside_current_scope(turn.text) and not turn.text.rstrip().endswith("?"):
                for target in _mentioned_fields(turn.text, observations):
                    observations.append(SourceObservation(target, FieldState.UNCERTAIN, None))
            continue
        previous = source.turns[index - 1] if index else None
        field = question_field(previous.text) if previous and previous.speaker is Speaker.CLINICIAN else None
        if (previous and previous.speaker is Speaker.CLINICIAN
                and _effect_only_reply(previous.text, turn.text)):
            continue  # Still uncovered for review; this is no medication claim.
        if (previous and previous.speaker is Speaker.CLINICIAN and _outside_current_scope(previous.text)
                and not re.search(r"\b(?:i|my|me|but|actually)\b", turn.text.casefold())):
            continue
        # A self-contained first-person clause can identify its own field, but
        # an explicit non-current/other-person/hypothetical prompt cannot grant
        # that attribution. Unknown speakers also break the question boundary.
        scope_safe = (previous is None or previous.speaker is Speaker.PATIENT
                      or (previous.speaker is Speaker.CLINICIAN and field is not None
                          and not _BLOCK.search(previous.text.casefold())))
        antecedents = {item.field for item in observations
                       if previous and previous.speaker is Speaker.PATIENT and item.span
                       and previous.start <= item.span.start < item.span.end <= previous.end}
        scoped = scoped_reply_observations(field, turn, observations, antecedents) if scope_safe else ()
        if scoped:
            observations.extend(scoped)
            continue
        targets = _mentioned_fields(turn.text, observations)
        if field is not None:
            targets.add(field)
        elif previous and previous.speaker is Speaker.CLINICIAN:
            targets.update(_mentioned_fields(previous.text, observations))
        elif (previous and previous.speaker is Speaker.PATIENT and index >= 2
              and source.turns[index - 2].speaker is Speaker.CLINICIAN
              and (re.fullmatch(r"(?:actually,? )?i was (?:mistaken|wrong|incorrect)[.!]?", turn.text.casefold())
                   or (re.search(r"\b(?:it|that)\b", turn.text.casefold())
                       and re.search(r"\b(?:not|never|wrong|incorrect|actually|after all)\b|n't\b", turn.text.casefold())))):
            prior = question_field(source.turns[index - 2].text)
            if prior is not None:
                targets.add(prior)
        span = CandidateSpan(turn.start, turn.end, turn.text) if turn.speaker is Speaker.PATIENT else None
        for target in targets:
            observations.append(SourceObservation(target, FieldState.UNCERTAIN, span))
    return tuple(observations), global_unsafe


def _observation_kind(source: Source, field: FieldName, span: CandidateSpan, *, context_policy: str = "legacy") -> FieldState | None:
    """Recognize a complete observation independently of the proposed state."""
    index = _turn_index(source, span.start, span.end)
    if index is None or source.turns[index].speaker is not Speaker.PATIENT:
        return None
    turn = source.turns[index]
    previous = source.turns[index - 1] if index else None
    if context_policy == "grounded":
        from nanoscribe.grounded import verify_grounded_span
        if verify_grounded_span(source, field, FieldState.ABSENT, span)[0]:
            return FieldState.ABSENT
        if verify_grounded_span(source, field, FieldState.SUPPORTED, span)[0]:
            return FieldState.SUPPORTED
        return None
    if context_policy == "context":
        observations, _ = inspect_context(source)
        matches = {item.state for item in observations if item.field is field and item.span == span}
        return next(iter(matches)) if len(matches) == 1 else None
    if previous is None or previous.speaker is not Speaker.CLINICIAN:
        return None
    if previous.text.casefold() not in QUESTIONS[field]:
        return None
    if span.text == turn.text and turn.text.casefold() in _NEGATIVE.get(field, ()):
        return FieldState.ABSENT
    if _plain_value(field, span.text) and any(
        template.format(span.text).casefold() == turn.text.casefold()
        for template in _POSITIVE[field]
    ):
        return FieldState.SUPPORTED
    return None


@dataclass(frozen=True)
class FieldAssessment:
    field: FieldName
    state: FieldState
    reasons: tuple[str, ...] = ()


def assess_fields(request: NanoInput, output: NanoOutput, *, context_policy: str = "legacy") -> tuple[FieldAssessment, ...]:
    """Check full reply + preceding question, not just a matching value span.

    A rejected candidate is quarantined as UNCERTAIN. Original inference is
    retained separately by the caller; this check does not rewrite its result.
    """
    output.validate_against(request)
    source = transcript_source(request)
    observations = ()
    # A clinician can establish a different experiencer or correct an earlier
    # answer. Only the exact allowlisted questions are safe to leave uncovered.
    unhandled = bool(uncovered_turns(source, output, context_policy=context_policy))
    # Evidence under an uncertain/wrong field may contain a correction to a
    # different field. Merely being referenced does not make that turn handled.
    unverified = any(_observation_kind(source, field.field, span, context_policy=context_policy) is None
                     for field in output.fields for span in field.evidence)
    if context_policy == "context":
        observations, unhandled = inspect_context(source)
        # Unknown source scope is global; a bad proposed span is local and is
        # checked below. Uncovered source remains in the encounter for review.
        unverified = False
    elif context_policy == "grounded":
        from nanoscribe.grounded import detect_grounded_conflicts, has_global_reassignment, recognized_conflict
        unhandled = has_global_reassignment(source)
        unverified = False
    assessments = []
    for field in output.fields:
        reasons = []
        if field.presented:
            if unhandled:
                reasons.append("unhandled_source_context")
            if unverified:
                reasons.append("unverified_source_observation")
            if context_policy == "context":
                for observation in observations:
                    if observation.field is not field.field:
                        continue
                    if observation.state is not field.state:
                        reasons.append("competing_or_unresolved_source_observation")
                    elif field.state is FieldState.SUPPORTED and normalize_value(observation.span.text) != normalize_value(field.value):
                        reasons.append("inconsistent_supported_observations")
            elif context_policy == "grounded":
                if detect_grounded_conflicts(source, field.field):
                    reasons.append("competing_or_unresolved_source_observation")
            for span in field.evidence:
                if _observation_kind(source, field.field, span, context_policy=context_policy) is not field.state:
                    reasons.append("observation_state_not_verified")
                if field.state is FieldState.SUPPORTED and normalize_value(span.text) != normalize_value(field.value):
                    reasons.append("inconsistent_supported_observations")
            state = FieldState.UNCERTAIN if reasons else field.state
        elif context_policy in ("context", "grounded") and field.state is FieldState.CONFLICTING:
            if unhandled:
                reasons.append("unhandled_source_context")
            if context_policy == "context":
                retained = tuple(item for item in observations if item.field is field.field and item.span in field.evidence)
                if not observations_conflict(field.field, retained):
                    reasons.append("conflict_not_verified")
                if any(_observation_kind(source, field.field, span, context_policy=context_policy) is None for span in field.evidence):
                    reasons.append("unverified_source_observation")
            else:
                if not recognized_conflict(field.evidence, detect_grounded_conflicts(source, field.field)):
                    reasons.append("conflict_not_verified")
            state = FieldState.UNCERTAIN if reasons else field.state
        else:
            state = field.state
        assessments.append(FieldAssessment(field.field, state, tuple(dict.fromkeys(reasons))))
    return tuple(assessments)


def check_note(request: NanoInput, output: NanoOutput, record: EncounterRecord, note: dict, *, context_policy: str = "legacy") -> dict:
    """Check an untrusted rendered note against the original source and record.

    Covers exact source identity, patient attribution, state/value/reference
    conservation, all five fields, and all visible text. A pass is not a claim
    that the extractor understood unrestricted clinical language.
    """
    errors = []
    try:
        output.validate_against(request)
        record.validate()
        assessments = assess_fields(request, output, context_policy=context_policy)
        if record.encounter_id != request.item_id or record.sources != (transcript_source(request),):
            errors.append("source_identity_mismatch")
        claims = note.get("claims")
        if not isinstance(claims, list) or len(claims) != len(FIELD_ORDER):
            return {"passed": False, "errors": ["claim_coverage_mismatch"], "scope": "bounded_source_and_representation"}
        lines = [HEADER]
        expected_atoms = set()
        expected_evidence = set()
        expected_conflicts = set()
        expected_unresolved = {"u:scope": {
            "unresolved_id": "u:scope", "topic": "scope", "reason": SCOPE_REASON,
            "review_required": True, "evidence_ids": [],
        }}
        for field, assessment, claim in zip(output.fields, assessments, claims):
            name = field.field.value
            refs = [f"e:{name}:{i}" for i in range(len(field.evidence))]
            expected_evidence.update(refs)
            state = assessment.state
            value = field.value if state is FieldState.SUPPORTED else None
            if claim != {"field": name, "state": state.value, "value": value, "evidence_ids": refs}:
                errors.append(f"claim_mismatch:{name}")
            for ref, span in zip(refs, field.evidence):
                saved = record.span(ref)
                if (saved.start, saved.end, saved.text, saved.speaker) != (span.start, span.end, span.text, Speaker.PATIENT):
                    errors.append(f"evidence_mismatch:{name}")
            if field.evidence:
                atom_id = f"a:{name}"
                expected_atoms.add(atom_id)
                atom = record.atom(atom_id)
                expected_state = {FieldState.SUPPORTED: AssertionState.ASSERTED, FieldState.ABSENT: AssertionState.DENIED,
                                  FieldState.UNCERTAIN: AssertionState.UNCERTAIN, FieldState.CONFLICTING: AssertionState.CONFLICTING}[state]
                expected_type = {FieldName.MEDICATION: AtomType.MEDICATION, FieldName.ALLERGY: AtomType.ALLERGY}.get(field.field, AtomType.SYMPTOM)
                expected_experiencer = Experiencer.PATIENT if state in (FieldState.SUPPORTED, FieldState.ABSENT) else Experiencer.UNKNOWN
                expected_certainty = Certainty.UNCERTAIN if state is FieldState.UNCERTAIN else Certainty.STATED if state in (FieldState.SUPPORTED, FieldState.ABSENT) else Certainty.UNKNOWN
                if (atom.assertion_state is not expected_state or atom.raw_value != (value or field.evidence[0].text)
                        or atom.evidence_ids != tuple(refs) or atom.speaker is not Speaker.PATIENT or not atom.review_required
                        or atom.atom_type is not expected_type or atom.experiencer is not expected_experiencer
                        or atom.certainty is not expected_certainty or atom.temporality != TemporalState(Temporality.UNKNOWN)
                        or atom.normalized_value is not None or atom.normalization_transform is not None):
                    errors.append(f"atom_mismatch:{name}")
            if state is FieldState.CONFLICTING:
                expected_conflicts.add(f"c:{name}")
                conflict = next((item for item in record.conflicts if item.conflict_id == f"c:{name}"), None)
                if (conflict is None or conflict.atom_ids != (f"a:{name}",)
                        or conflict.evidence_ids != tuple(refs) or conflict.note != CONFLICT_NOTE):
                    errors.append(f"conflict_mismatch:{name}")
            if state not in (FieldState.SUPPORTED, FieldState.ABSENT):
                expected_unresolved[f"u:{name}"] = {
                    "unresolved_id": f"u:{name}", "topic": name,
                    "reason": "; ".join(assessment.reasons) or state.value,
                    "review_required": True, "evidence_ids": refs,
                }
            # Independently construct the allowed line from checked source data;
            # never use the renderer's text or its claimed verification status.
            prefix = f"{LABELS[field.field]}: "
            if state is FieldState.SUPPORTED:
                content = "patient reports " + json.dumps(value, ensure_ascii=False)
            elif state is FieldState.ABSENT:
                content = "patient denial " + json.dumps(field.evidence[0].text, ensure_ascii=False)
            else:
                content = f"not asserted ({state.value}); review required"
            citation = " [" + ", ".join(refs) + "]" if refs else ""
            lines.append(prefix + content + citation)
        if {atom.atom_id for atom in record.atoms} != expected_atoms:
            errors.append("atom_coverage_mismatch")
        if {item.conflict_id for item in record.conflicts} != expected_conflicts:
            errors.append("conflict_coverage_mismatch")
        if note.get("text") != "\n".join(lines):
            errors.append("visible_text_mismatch")
        if set(note) != {"claims", "text", "review_required"} or note.get("review_required") is not True:
            errors.append("note_envelope_mismatch")
        # Review items cannot be removed from the structured record to make a
        # draft look complete. Missing does not mean a clinical negative.
        for turn in uncovered_turns(record.sources[0], output, context_policy=context_policy):
            ref = f"e:{turn.turn_id}"
            expected_evidence.add(ref)
            saved = record.span(ref)
            if (saved.start, saved.end, saved.text, saved.speaker) != (turn.start, turn.end, turn.text, turn.speaker):
                errors.append(f"unhandled_turn_evidence_mismatch:{turn.turn_id}")
            expected_unresolved[f"u:{turn.turn_id}"] = {
                "unresolved_id": f"u:{turn.turn_id}", "topic": "unhandled_source_turn",
                "reason": UNHANDLED_REASON, "review_required": True, "evidence_ids": [ref],
            }
        if {span.evidence_id for span in record.evidence} != expected_evidence:
            errors.append("evidence_coverage_mismatch")
        if {item.unresolved_id: item.to_dict() for item in record.unresolved} != expected_unresolved:
            errors.append("unresolved_projection_mismatch")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError) as exc:
        errors.append(f"invalid_boundary:{type(exc).__name__}")
    return {"passed": not errors, "errors": errors, "scope": "bounded_source_and_representation"}
