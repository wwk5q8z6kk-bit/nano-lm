"""Span-port prompt templates for P1 model adapters."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Protocol

from nanoscribe.encounter import AtomType, Speaker, Source
from nanoscribe import delimit, leakage


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


IDENTIFIER_PLACEHOLDER = "\u0000IDENT\u0000"


def identifier_for_spec(spec: _AtomSpecLike) -> str:
    """The string that tells the model WHICH slot is being asked about.

    This is the only thing C3 varies. On, it is the slot's gold surface form —
    which is also the answer. Off, it is a role label that shares no surface
    form with any value in the instance.
    """
    if leakage.PROMPT_QUESTION_USES_GOLD_SURFACE:
        return repr(spec.raw_value)
    label = getattr(spec, "concept_label", "") or ""
    return label or f"the {spec.atom_type.value} field"


def question_template(spec: _AtomSpecLike, identifier: str) -> str:
    """One question form for BOTH C3 arms — wh-extraction over the same answer space.

    The first version of this varied the FORM as well as the identifier: C3-on
    asked a yes/no question ("Does the patient mention 'migraines'?") while
    C3-off asked a wh-question ("What is a condition ...?"). That confounded the
    contrast and voided the arm — under the yes/no form the model answers the
    yes/no question, which the harness reads as NOT_MENTIONED, so the leakier
    cells scored WORSE. Same form, same answer space, same response mode; only
    `identifier` differs. `test_form_equivalence_invariant` asserts it.
    """
    who = "clinician" if spec.speaker is Speaker.CLINICIAN else "patient"
    return f"What does the {who} say about {identifier}?"


def label_topic_for_spec(spec: _AtomSpecLike) -> str:
    """Task phrased by role label — identifies the slot, names no surface form."""
    return question_template(spec, identifier_for_spec(spec))


def topic_with_placeholder(spec: _AtomSpecLike) -> str:
    """The question with the identifier removed — used to prove form equivalence."""
    return question_template(spec, IDENTIFIER_PLACEHOLDER)


def topic_for_spec(spec: _AtomSpecLike) -> str:
    """Human-readable extraction task for one atom slot."""
    if not leakage.PROMPT_QUESTION_NAMES_CONCEPT:
        return slot_topic_for_spec(spec)
    return question_template(spec, identifier_for_spec(spec))


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


def question_for_spec(spec: _AtomSpecLike) -> str:
    """The question, and nothing else. Byte-identical across E-DELIMIT arms.

    R1 requires the arms to vary only the output-format module. This function
    is what R1 protects, and `question_template_hash` is what proves it held.
    """
    who = "clinician" if spec.speaker is Speaker.CLINICIAN else "patient"
    return (
        f"{topic_for_spec(spec)} Answer using only the {who}'s words."
        f"{answer_hint_for_spec(spec)}"
    )


def build_span_port_prompt(source: Source, spec: _AtomSpecLike) -> str:
    """Build a single-atom span-port probe prompt from encounter source."""
    return (
        f"{delimit.transcript_block(source)}\n\n"
        f"{question_for_spec(spec)} "
        f"{delimit.format_instruction(source, spec.atom_id)}"
    )


def question_template_hash(specs: Iterable[_AtomSpecLike]) -> str:
    """Digest of the realized questions for a slot set.

    R2 says a contrast is legal iff the prompt-template hashes are equal. The
    naive whole-module hash cannot express that here: the arms differ in the
    output-format module by construction, so every contrast would read illegal.
    The invariant R2 actually protects is *same instrument* — so hash the
    questions, which must match, and hash the format separately
    (`delimit.output_format_hash`), which must differ.
    """
    digest = hashlib.sha256()
    for spec in sorted(specs, key=lambda s: s.atom_id):
        digest.update(spec.atom_id.encode())
        digest.update(b"\x00")
        digest.update(question_for_spec(spec).encode())
        digest.update(b"\x00")
    return digest.hexdigest()[:16]


def span_port_system_prompt() -> str:
    if not leakage.PROMPT_ANSWER_TEMPLATE_GOLD_VALUE:
        return _SPAN_PORT_SYSTEM_NEUTRAL
    return _SPAN_PORT_SYSTEM
