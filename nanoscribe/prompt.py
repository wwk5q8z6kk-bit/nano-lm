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


_STRUCTURED_SYSTEM = (
    "You extract clinical facts from transcripts into structured candidate atoms. "
    "Reply with JSON only — no markdown fences.\n"
    "Schema:\n"
    '{"schema_version":"nano.candidate.v0","atoms":[{"atom_id":"...","atom_type":"symptom",'
    '"raw_value":"...","assertion_state":"asserted|denied|uncertain","speaker":"patient|clinician",'
    '"experiencer":"patient|clinician|other","temporality":{"kind":"current|historical|future"},'
    '"certainty":"stated|uncertain","evidence_quote":"verbatim substring from transcript",'
    '"review_required":false,"abstained":false,"malformed":false}]}\n'
    "Do not emit offsets, evidence_id, source_id, or normalized_value. "
    "evidence_quote must copy source words exactly. "
    "If absent, set abstained=true and omit evidence_quote."
)


def build_structured_candidate_prompt(source: Source, specs: tuple[_AtomSpecLike, ...]) -> str:
    """Batch structured CandidateAtom probe for one encounter."""
    transcript = _format_transcript(source)
    slots: list[str] = []
    for spec in specs:
        topic = topic_for_spec(spec)
        slots.append(
            f"- atom_id={spec.atom_id!r} atom_type={spec.atom_type.value!r} "
            f"raw_value={spec.raw_value!r} speaker={spec.speaker.value!r}; "
            f"question: {topic}"
        )
    return (
        f"Transcript:\n{transcript}\n\n"
        "For each atom slot below, emit one object in atoms[] with matching atom_id.\n"
        + "\n".join(slots)
        + "\n\nReturn JSON with schema_version nano.candidate.v0 and atoms for every slot."
    )


def structured_candidate_system_prompt() -> str:
    return _STRUCTURED_SYSTEM


_TOOL_SYSTEM = (
    "You extract clinical facts from transcripts into structured candidate atoms. "
    "You MUST call the submit_candidate_atoms tool exactly once with all atom slots filled. "
    "Do not reply with free text or markdown. "
    "Quote-only evidence — never emit offsets, evidence_id, source_id, or normalized_value. "
    "evidence_quote must copy source words exactly. "
    "If absent, set abstained=true and omit evidence_quote."
)


def tool_candidate_system_prompt() -> str:
    return _TOOL_SYSTEM
