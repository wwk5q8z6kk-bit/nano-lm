"""Span-port prompt templates for P1 model adapters."""

from __future__ import annotations

from typing import Protocol

from nanoscribe.encounter import AtomType, Speaker, Source
from nanoscribe import leakage


class _AtomSpecLike(Protocol):
    atom_id: str
    atom_type: AtomType
    raw_value: str
    speaker: Speaker
    concept_label: str


_SPAN_PORT_PREAMBLE = (
    "You extract clinical facts from transcripts. "
    "Reply with exactly one line: STATED, DENIED, UNCERTAIN, or NOT_MENTIONED. "
    "For STATED, DENIED, or UNCERTAIN include a verbatim quote in double quotes.\n"
)

# The shipped format examples are themselves gold answers in this suite:
# "neck" (enc-1/atom-neck), "No allergies." (enc-1/atom-alg), "pressure"
# (enc-2/atom-chest). That makes the system prompt a gold-value channel too, so
# it is gated by C1 along with the user prompt.
_SPAN_PORT_SYSTEM = _SPAN_PORT_PREAMBLE + (
    'Example: STATED: "neck"\n'
    'Example: DENIED: "No allergies."\n'
    'Example: UNCERTAIN: "pressure"\n'
    "Example: NOT_MENTIONED"
)

# Format-only examples whose values occur in no suite encounter.
_SPAN_PORT_SYSTEM_NEUTRAL = _SPAN_PORT_PREAMBLE + (
    'Example: STATED: "ankle"\n'
    'Example: DENIED: "No prior surgery."\n'
    'Example: UNCERTAIN: "lightheaded"\n'
    "Example: NOT_MENTIONED"
)


def _format_transcript(source: Source) -> str:
    lines: list[str] = []
    for turn in source.turns:
        lines.append(f"{turn.speaker.value}: {turn.text}")
    return "\n".join(lines)


_SLOT_QUESTION = {
    AtomType.SYMPTOM: "any symptom or complaint",
    AtomType.MEDICATION: "any medication the patient takes",
    AtomType.ALLERGY: "any allergy",
    AtomType.HISTORY: "any past or family medical history",
    AtomType.DIAGNOSIS_STATEMENT: "any diagnosis",
    AtomType.ASSESSMENT: "the clinician's assessment",
    AtomType.PLAN: "the care plan",
    AtomType.PROCEDURE: "any procedure",
    AtomType.INSTRUCTION: "patient instructions",
    AtomType.MEASUREMENT: "any measurement or vital sign",
}


def slot_topic_for_spec(spec: _AtomSpecLike) -> str:
    """Task phrased by slot type only (Q off).

    Underdetermined by construction: two slots of the same atom_type get the
    same question. Diagnostic cell only — see leakage.py.
    """
    question = _SLOT_QUESTION.get(spec.atom_type)
    if question is None:
        return f"Does the transcript mention the {spec.atom_type.value} field?"
    return f"Does the transcript mention {question}?"


def label_topic_for_spec(spec: _AtomSpecLike) -> str:
    """Task phrased by role label — identifies the slot, names no surface form."""
    label = getattr(spec, "concept_label", "") or ""
    if not label:
        # No label authored: fall back to slot-type phrasing rather than
        # silently reintroducing the surface string.
        return slot_topic_for_spec(spec)
    return f"What is {label}?"


def topic_for_spec(spec: _AtomSpecLike) -> str:
    """Human-readable extraction task for one atom slot."""
    if not leakage.PROMPT_QUESTION_NAMES_CONCEPT:
        return slot_topic_for_spec(spec)
    if not leakage.PROMPT_QUESTION_USES_GOLD_SURFACE:
        return label_topic_for_spec(spec)
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


def answer_hint_for_spec(spec: _AtomSpecLike) -> str:
    if not leakage.PROMPT_ANSWER_TEMPLATE_GOLD_VALUE:
        # Keep the task/format guidance, drop every gold-value-bearing clause.
        return " Use DENIED only for explicit denial."
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
    transcript = _format_transcript(source)
    task = topic_for_spec(spec)
    who = "clinician" if spec.speaker is Speaker.CLINICIAN else "patient"
    hint = answer_hint_for_spec(spec)
    return (
        f"{transcript}\n\n"
        f"{task} Answer using only the {who}'s words.{hint} "
        "Reply with exactly one line: "
        "STATED, DENIED, UNCERTAIN, or NOT_MENTIONED with a verbatim quote."
    )


def span_port_system_prompt() -> str:
    if not leakage.PROMPT_ANSWER_TEMPLATE_GOLD_VALUE:
        return _SPAN_PORT_SYSTEM_NEUTRAL
    return _SPAN_PORT_SYSTEM
