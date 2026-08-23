"""Span-port prompt templates for P1 model adapters."""

from __future__ import annotations

from typing import Protocol

from nanoscribe.encounter import AtomType, Speaker, Source


class _AtomSpecLike(Protocol):
    atom_id: str
    atom_type: AtomType
    raw_value: str
    speaker: Speaker

_SPAN_PORT_SYSTEM = (
    "You extract clinical facts from transcripts. "
    "Reply with exactly one line: STATED (or ASSERTED), DENIED, UNCERTAIN, or NOT_MENTIONED. "
    "For STATED/ASSERTED, DENIED, or UNCERTAIN include a verbatim quote in double quotes.\n"
    'Example: STATED: "neck"\n'
    'Example: DENIED: "No allergies."\n'
    "Example: UNCERTAIN: \"Maybe a little pressure sometimes.\"\n"
    "Example: NOT_MENTIONED"
)

_SPAN_PORT_SUFFIX = """Answer on one line:
- STATED or ASSERTED: "exact words that name it"
- DENIED: "exact words that deny it"
- UNCERTAIN: "exact words showing uncertainty"
- NOT_MENTIONED
Quotes must copy source words exactly. If absent, write only NOT_MENTIONED."""

_FIELD_QUESTION: dict[AtomType, str] = {
    AtomType.MEDICATION: "any medication the patient is taking",
    AtomType.ALLERGY: "any allergy the patient has",
    AtomType.SYMPTOM: "the symptom or complaint described",
    AtomType.HISTORY: "relevant medical history mentioned",
    AtomType.ASSESSMENT: "the clinician's assessment or impression",
    AtomType.DIAGNOSIS_STATEMENT: "any diagnosis stated",
    AtomType.PLAN: "the care plan discussed",
    AtomType.PROCEDURE: "any procedure mentioned",
    AtomType.INSTRUCTION: "patient instructions given",
    AtomType.MEASUREMENT: "any measurement or vital sign",
}


def _format_transcript(source: Source) -> str:
    lines: list[str] = []
    for turn in source.turns:
        lines.append(f"{turn.speaker.value}: {turn.text}")
    return "\n".join(lines)


def topic_for_spec(spec: _AtomSpecLike) -> str:
    """Human-readable question topic for one atom slot."""
    if spec.raw_value:
        if spec.speaker is Speaker.CLINICIAN:
            return f"whether the transcript mentions {spec.raw_value!r}"
        return f"whether the patient mentions {spec.raw_value!r}"
    question = _FIELD_QUESTION.get(spec.atom_type)
    if question:
        return question
    return f"the {spec.atom_type.value} field"


def build_span_port_prompt(source: Source, spec: _AtomSpecLike) -> str:
    """Build a single-atom span-port probe prompt from encounter source."""
    transcript = _format_transcript(source)
    topic = topic_for_spec(spec)
    if spec.speaker is Speaker.CLINICIAN:
        who = "the clinician's own words"
        choices = "STATED for clinician assertions, NOT_MENTIONED if absent"
    else:
        who = "the patient's own words"
        choices = (
            "STATED - names a specific one; DENIED - denies having any; "
            "NOT_MENTIONED - topic never comes up"
        )
    return (
        f"Transcript:\n{transcript}\n\n"
        f"Question: regarding {topic}, which of these do {who} do?\n\n"
        f"{choices}\n\n"
        f"{_SPAN_PORT_SUFFIX}"
    )


def span_port_system_prompt() -> str:
    return _SPAN_PORT_SYSTEM
