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


def _canonical_topic_for_spec(spec: _AtomSpecLike) -> str:
    """Topic phrasing for the native training/eval prompt surface."""
    if spec.raw_value:
        if spec.speaker is Speaker.CLINICIAN:
            return f"whether the transcript mentions {spec.raw_value!r}"
        return f"whether the patient mentions {spec.raw_value!r}"
    question = _FIELD_QUESTION.get(spec.atom_type)
    if question:
        return question
    return f"the {spec.atom_type.value} field"


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
    """Compact span-port probe for Qwen adapter / campaign measurement."""
    transcript = _format_transcript(source)
    task = topic_for_spec(spec)
    who = "clinician" if spec.speaker is Speaker.CLINICIAN else "patient"
    hint = _answer_hint(spec)
    return (
        f"{transcript}\n\n"
        f"{task} Answer using only the {who}'s words.{hint} "
        "Reply with exactly one line: "
        "STATED, DENIED, UNCERTAIN, or NOT_MENTIONED with a verbatim quote."
    )


def build_canonical_span_port_prompt(source: Source, spec: _AtomSpecLike) -> str:
    """Canonical Transcript/Question surface for native training and screening eval.

    Kept separate from `build_span_port_prompt` so master's compact Qwen prompts
  are not regressed while the native corpus factory stays aligned with the
    evaluator regex and left-truncation budgets.
    """
    transcript = _format_transcript(source)
    topic = _canonical_topic_for_spec(spec)
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
