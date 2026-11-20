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
    "Reply with exactly one line: STATED, DENIED, UNCERTAIN, or NOT_MENTIONED. "
    "For STATED, DENIED, or UNCERTAIN include a verbatim quote in double quotes.\n"
    'Example: STATED: "neck"\n'
    'Example: DENIED: "No allergies."\n'
    'Example: UNCERTAIN: "pressure"\n'
    "Example: NOT_MENTIONED"
)


def format_transcript(source: Source) -> str:
    """Serialize the exact transcript portion included in a span-port prompt."""
    lines: list[str] = []
    for turn in source.turns:
        lines.append(f"{turn.speaker.value}: {turn.text}")
    return "\n".join(lines)


def topic_for_spec(spec: _AtomSpecLike) -> str:
    """Human-readable extraction task for one atom slot."""
    if spec.atom_type is AtomType.ALLERGY:
        return "Does the patient mention or deny allergies?"
    if spec.atom_type is AtomType.MEDICATION:
        return "Does the patient mention any medication they take?"
    if spec.atom_type is AtomType.HISTORY:
        return (
            f"Does the transcript mention {spec.raw_value!r} "
            "(patient history or family history)?"
        )
    if spec.speaker is Speaker.CLINICIAN:
        return f"Does the clinician state {spec.raw_value!r}?"
    if spec.atom_type is AtomType.ASSESSMENT:
        return f"Does the clinician's assessment include {spec.raw_value!r}?"
    if spec.raw_value:
        return f"Does the patient mention {spec.raw_value!r} (current or past)?"
    question = {
        AtomType.SYMPTOM: "any symptom or complaint",
        AtomType.DIAGNOSIS_STATEMENT: "any diagnosis",
        AtomType.PLAN: "the care plan",
        AtomType.PROCEDURE: "any procedure",
        AtomType.INSTRUCTION: "patient instructions",
        AtomType.MEASUREMENT: "any measurement or vital sign",
    }.get(spec.atom_type)
    if question:
        return f"Does the transcript mention {question}?"
    return f"Does the transcript mention the {spec.atom_type.value} field?"


def _answer_hint(spec: _AtomSpecLike) -> str:
    if spec.atom_type is AtomType.ALLERGY:
        return ' If denied, reply DENIED: "No allergies." If affirmed, STATED with a quote.'
    if spec.speaker is Speaker.CLINICIAN:
        return (
            f' If yes, reply STATED: "{spec.raw_value}". '
            "Use only clinician lines, not patient lines."
        )
    if spec.raw_value:
        return (
            f' If mentioned (including past history), reply STATED: "{spec.raw_value}". '
            "Use DENIED only for explicit denial."
        )
    return ""


def build_span_port_prompt(source: Source, spec: _AtomSpecLike) -> str:
    """Build a single-atom span-port probe prompt from encounter source."""
    transcript = format_transcript(source)
    task = topic_for_spec(spec)
    who = "clinician" if spec.speaker is Speaker.CLINICIAN else "patient"
    hint = _answer_hint(spec)
    return (
        f"{transcript}\n\n"
        f"{task} Answer using only the {who}'s words.{hint} "
        "Reply with exactly one line: "
        "STATED, DENIED, UNCERTAIN, or NOT_MENTIONED with a verbatim quote."
    )


def span_port_system_prompt() -> str:
    return _SPAN_PORT_SYSTEM
