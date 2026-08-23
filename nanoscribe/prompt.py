"""Span-port prompt templates for P1 model adapters."""

from __future__ import annotations

from typing import Protocol

from nanoscribe.encounter import AtomType, Speaker, Source


class _AtomSpecLike(Protocol):
    atom_id: str
    atom_type: AtomType
    raw_value: str
    speaker: Speaker

_SPAN_PORT_INSTRUCTIONS = """Answer on one line:
- STATED: "exact words that name it"
- DENIED: "exact words that deny it"
- NOT_MENTIONED
Quotes must copy the source words exactly. If the topic never comes up, write
only NOT_MENTIONED with no quotes."""

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
        perspective = "the clinician's own words"
        label_hint = "STATED for clinician assertions, NOT_MENTIONED if absent"
    else:
        perspective = "the patient's own words"
        label_hint = (
            "STATED - names a specific one; DENIED - denies having any; "
            "NOT_MENTIONED - topic never comes up"
        )
    return (
        "You are reading a clinic transcript.\n\n"
        f"{transcript}\n\n"
        f"Question: regarding {topic}, which of these do {perspective} do?\n\n"
        f"{label_hint}\n\n"
        f"{_SPAN_PORT_INSTRUCTIONS}"
    )
