"""Closed-world deterministic reference for the v0 synthetic scribe grammar.

This solver exists as a performance floor and a contract diagnostic.  It only
recognizes the question and answer templates used by the original synthetic
five-field task; it is not Nano and it is not a general clinical extractor.
It receives a transcript only and never receives fixture targets or tuples.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from nano_ai.contract import (
    FIELD_ORDER,
    EvidenceSpan,
    FieldName,
    FieldOutput,
    FieldState,
    NanoInput,
    NanoOutput,
)
from nano_ai.solver import SolverDescriptor, SolverKind

_QUESTIONS: dict[FieldName, frozenset[str]] = {
    FieldName.CHIEF_COMPLAINT: frozenset(
        {
            "good morning, what brings you in today?",
            "hello, what can i do for you?",
            "hi there, what seems to be the trouble?",
            "so, tell me what's going on.",
            "what brings you to the clinic today?",
            "morning — what's been bothering you?",
            "come in, have a seat. what's the issue today?",
        }
    ),
    FieldName.DURATION: frozenset(
        {
            "how long has this been going on?",
            "when did it start?",
            "how many days has it been?",
            "since when have you had it?",
        }
    ),
    FieldName.SEVERITY: frozenset(
        {
            "how bad would you say it is?",
            "is it mild, moderate, or severe?",
            "on a scale from mild to severe, where is it?",
        }
    ),
    FieldName.MEDICATION: frozenset(
        {
            "have you taken anything for it?",
            "are you on any medication for this?",
            "did you try any medicine?",
        }
    ),
    FieldName.ALLERGY: frozenset(
        {
            "any allergies i should know about?",
            "are you allergic to anything?",
            "do you have any known allergies?",
        }
    ),
}

_ANSWER_TEMPLATES: dict[FieldName, tuple[tuple[str, str], ...]] = {
    FieldName.CHIEF_COMPLAINT: (
        ("I've been having {value}.", r".+?"),
        ("I came in because of {value}.", r".+?"),
        ("It's {value}, doctor.", r".+?"),
        ("Well, I've got {value} that won't go away.", r".+?"),
        ("I'm dealing with {value}.", r".+?"),
        ("Honestly, {value} has been troubling me.", r".+?"),
        ("It started as {value} and hasn't stopped.", r".+?"),
    ),
    FieldName.DURATION: (
        ("For about {value} now.", r"\d+\s+(?:days?|weeks?)"),
        ("For about {value}.", r"\d+\s+(?:days?|weeks?)"),
        ("It started {value} ago.", r"\d+\s+(?:days?|weeks?)"),
        ("Around {value}.", r"\d+\s+(?:days?|weeks?)"),
        ("I'd say it's been {value}.", r"\d+\s+(?:days?|weeks?)"),
        ("Since about {value} back.", r"\d+\s+(?:days?|weeks?)"),
        ("Started roughly {value} prior.", r"\d+\s+(?:days?|weeks?)"),
        ("On and off for maybe {value}.", r"\d+\s+(?:days?|weeks?)"),
        ("Coming up on {value} now.", r"\d+\s+(?:days?|weeks?)"),
    ),
    FieldName.SEVERITY: (
        ("I'd call it {value}.", r"mild|moderate|severe"),
        ("It's {value}, I would say.", r"mild|moderate|severe"),
        ("Pretty {value}.", r"mild|moderate|severe"),
        ("Definitely {value}.", r"mild|moderate|severe"),
    ),
    FieldName.MEDICATION: (
        ("I've been taking {value}.", r".+?"),
        ("Just {value}.", r".+?"),
        ("Some {value}, but it barely helps.", r".+?"),
        ("Only {value} so far.", r".+?"),
    ),
    FieldName.ALLERGY: (
        ("I'm allergic to {value}.", r".+?"),
        ("Yes, {value}.", r".+?"),
        ("Just {value}.", r".+?"),
        ("I do — {value}.", r".+?"),
    ),
}

_DENIALS: dict[FieldName, frozenset[str]] = {
    FieldName.MEDICATION: frozenset(
        {"no, nothing yet.", "i haven't taken anything.", "nothing at all."}
    ),
    FieldName.ALLERGY: frozenset(
        {"no allergies.", "not that i know of.", "none whatsoever."}
    ),
}


def _template_pattern(template: str, value_pattern: str) -> re.Pattern[str]:
    before, after = template.split("{value}")
    return re.compile(
        rf"{re.escape(before)}(?P<value>{value_pattern}){re.escape(after)}",
        re.IGNORECASE,
    )


_ANSWER_PATTERNS = {
    field: tuple(_template_pattern(template, value) for template, value in templates)
    for field, templates in _ANSWER_TEMPLATES.items()
}


@dataclass(frozen=True)
class _Turn:
    speaker: str
    text: str
    start: int


@dataclass(frozen=True)
class _Observation:
    state: FieldState
    value: str | None = None
    evidence: tuple[EvidenceSpan, ...] = ()


def _turns(transcript: str) -> tuple[_Turn, ...]:
    """Parse role-prefixed lines without changing source character offsets."""

    parsed: list[_Turn] = []
    offset = 0
    for line in transcript.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        match = re.fullmatch(r"(Doctor|Patient):[ \t]*(.*)", content)
        if match is not None:
            parsed.append(
                _Turn(
                    speaker=match.group(1).casefold(),
                    text=match.group(2),
                    start=offset + match.start(2),
                )
            )
        offset += len(line)
    return tuple(parsed)


def _question_field(text: str) -> FieldName | None:
    question = text.strip().casefold()
    for field, candidates in _QUESTIONS.items():
        if question in candidates:
            return field
    return None


def _canonical_value(field: FieldName, value: str) -> str:
    canonical = " ".join(value.strip().split()).casefold()
    if field is FieldName.CHIEF_COMPLAINT:
        canonical = re.sub(r"^(?:a|an|the)\s+", "", canonical)
    return canonical


def _reply_evidence(
    turn: _Turn,
    start: int | None = None,
    end: int | None = None,
) -> EvidenceSpan:
    begin = len(turn.text) - len(turn.text.lstrip()) if start is None else start
    finish = len(turn.text.rstrip()) if end is None else end
    text = turn.text[begin:finish]
    return EvidenceSpan(
        start=turn.start + begin,
        end=turn.start + finish,
        text=text,
        speaker="patient",
    )


def _observe_reply(field: FieldName, reply: _Turn | None) -> _Observation:
    if reply is None:
        return _Observation(FieldState.MISSING)
    if not reply.text.strip():
        return _Observation(FieldState.UNCERTAIN)

    if reply.text.strip().casefold() in _DENIALS.get(field, ()):
        return _Observation(FieldState.ABSENT, evidence=(_reply_evidence(reply),))

    for pattern in _ANSWER_PATTERNS[field]:
        match = pattern.fullmatch(reply.text.strip())
        if match is None:
            continue
        # Account for leading whitespace removed by strip() while retaining exact
        # offsets into the original transcript.
        leading = len(reply.text) - len(reply.text.lstrip())
        start = leading + match.start("value")
        end = leading + match.end("value")
        value = _canonical_value(field, reply.text[start:end])
        if not value:
            return _Observation(
                FieldState.UNCERTAIN, evidence=(_reply_evidence(reply),)
            )
        return _Observation(
            FieldState.SUPPORTED,
            value=value,
            evidence=(_reply_evidence(reply, start, end),),
        )

    return _Observation(FieldState.UNCERTAIN, evidence=(_reply_evidence(reply),))


def _observations(transcript: str) -> dict[FieldName, list[_Observation]]:
    result: dict[FieldName, list[_Observation]] = {field: [] for field in FIELD_ORDER}
    turns = _turns(transcript)
    for index, turn in enumerate(turns):
        if turn.speaker != "doctor":
            continue
        field = _question_field(turn.text)
        if field is None:
            continue
        reply = None
        for candidate in turns[index + 1 :]:
            if candidate.speaker == "doctor":
                break
            if candidate.speaker == "patient":
                reply = candidate
                break
        result[field].append(_observe_reply(field, reply))
    return result


def _flatten_evidence(observations: Iterable[_Observation]) -> tuple[EvidenceSpan, ...]:
    return tuple(span for observation in observations for span in observation.evidence)


def _resolve(field: FieldName, observations: list[_Observation]) -> FieldOutput:
    if not observations:
        return FieldOutput(field=field, state=FieldState.MISSING)

    supported = [item for item in observations if item.state is FieldState.SUPPORTED]
    absent = [item for item in observations if item.state is FieldState.ABSENT]
    uncertain = [item for item in observations if item.state is FieldState.UNCERTAIN]
    missing = [item for item in observations if item.state is FieldState.MISSING]

    concrete_values = {item.value for item in supported}
    has_conflict = len(concrete_values) > 1 or (bool(supported) and bool(absent))
    if has_conflict:
        concrete = (*supported, *absent)
        return FieldOutput(
            field=field,
            state=FieldState.CONFLICTING,
            evidence=_flatten_evidence(concrete),
        )

    # An unrecognized repeated answer prevents this conservative diagnostic from
    # asserting a value even if another reply happened to match the closed grammar.
    if uncertain or (missing and (supported or absent)):
        return FieldOutput(
            field=field,
            state=FieldState.UNCERTAIN,
            evidence=_flatten_evidence((*supported, *absent, *uncertain)),
        )

    if supported:
        return FieldOutput(
            field=field,
            state=FieldState.SUPPORTED,
            value=supported[0].value,
            evidence=_flatten_evidence(supported),
        )
    if absent:
        return FieldOutput(
            field=field,
            state=FieldState.ABSENT,
            evidence=_flatten_evidence(absent),
        )
    return FieldOutput(field=field, state=FieldState.MISSING)


def _extract_fields(item: NanoInput) -> tuple[FieldOutput, ...]:
    observed = _observations(item.transcript)
    return tuple(_resolve(field, observed[field]) for field in FIELD_ORDER)


class DeterministicV0Solver:
    """Rules-perfect diagnostic for recognized v0 templates, never a trained AI."""

    descriptor = SolverDescriptor(
        solver_id="reference/deterministic-v0",
        kind=SolverKind.REFERENCE,
        version="0",
        parameter_count=0,
        artifact_bytes=0,
    )
    solver_id = descriptor.solver_id

    def infer(self, item: NanoInput) -> NanoOutput:
        output = NanoOutput(
            item_id=item.item_id,
            solver_id=self.solver_id,
            fields=_extract_fields(item),
        )
        output.validate_against(item)
        return output


__all__ = ["DeterministicV0Solver"]
