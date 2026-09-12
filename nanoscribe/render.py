"""Verified encounter record → clinical note rendering (view layer, not truth).

The note is a deterministic VIEW of EncounterRecord. EncounterRecord remains
the primary truth object; this module never invents clinical content.
"""

from __future__ import annotations

from dataclasses import dataclass

from nanoscribe.encounter import (
    AssertionState,
    AtomType,
    ClinicalAtom,
    EncounterRecord,
    UnresolvedItem,
    normalize_value,
)
from nanoscribe.evaluate import SupportRelation

_SECTION_ORDER: tuple[tuple[str, frozenset[AtomType]], ...] = (
    ("Symptoms", frozenset({AtomType.SYMPTOM, AtomType.MEASUREMENT})),
    ("Allergies", frozenset({AtomType.ALLERGY})),
    ("Medications", frozenset({AtomType.MEDICATION})),
    ("History", frozenset({AtomType.HISTORY})),
    (
        "Assessment",
        frozenset({AtomType.ASSESSMENT, AtomType.DIAGNOSIS_STATEMENT}),
    ),
    (
        "Plan",
        frozenset({AtomType.PLAN, AtomType.PROCEDURE, AtomType.INSTRUCTION}),
    ),
)

_FLAGGED_RELATIONS = frozenset(
    {
        SupportRelation.UNSUPPORTED,
        SupportRelation.CONTRADICTED,
        SupportRelation.REVIEW_REQUIRED,
    }
)

_SEMANTIC_STATES = frozenset({AssertionState.DENIED, AssertionState.CONFLICTING})


@dataclass(frozen=True, slots=True)
class ClaimFlag:
    """Verifier outcome for a single rendered claim line."""

    atom_id: str
    relation: SupportRelation
    message: str


@dataclass(frozen=True, slots=True)
class NoteRenderResult:
    """Sectioned note plus per-claim verification flags."""

    note: str
    flags: tuple[ClaimFlag, ...]

    @property
    def unsupported_count(self) -> int:
        return sum(1 for flag in self.flags if flag.relation in _FLAGGED_RELATIONS)


def _atom_spans(record: EncounterRecord, atom: ClinicalAtom):
    return tuple(record.span(evidence_id) for evidence_id in atom.evidence_ids)


def _mechanical_support(raw_value: str, spans) -> SupportRelation | None:
    if any(raw_value in span.text for span in spans):
        return SupportRelation.DIRECT_EXACT
    grounded = normalize_value(raw_value)
    if grounded and any(grounded in normalize_value(span.text) for span in spans):
        return SupportRelation.NORMALIZED
    return None


def verify_claim(record: EncounterRecord, atom: ClinicalAtom) -> SupportRelation:
    """Mechanical verifier for a single atom — mirrors evaluate.py support layer."""
    spans = _atom_spans(record, atom)
    if not spans:
        return SupportRelation.UNSUPPORTED
    if atom.assertion_state in _SEMANTIC_STATES:
        mechanical = _mechanical_support(atom.raw_value, spans)
        if mechanical is not None:
            return mechanical
        return SupportRelation.REVIEW_REQUIRED
    mechanical = _mechanical_support(atom.raw_value, spans)
    if mechanical is not None:
        return mechanical
    return SupportRelation.UNSUPPORTED


def verify_record(record: EncounterRecord) -> tuple[ClaimFlag, ...]:
    """Verify every atom in a record; flag unsupported or review-required claims."""
    flags: list[ClaimFlag] = []
    for atom in record.atoms:
        relation = verify_claim(record, atom)
        if relation in _FLAGGED_RELATIONS:
            flags.append(
                ClaimFlag(
                    atom_id=atom.atom_id,
                    relation=relation,
                    message=f"{relation.value} for {atom.raw_value!r}",
                )
            )
    return tuple(flags)


def _evidence_citation(record: EncounterRecord, atom: ClinicalAtom) -> str:
    quotes = [record.span(evidence_id).text.strip() for evidence_id in atom.evidence_ids]
    if not quotes:
        return ""
    joined = "; ".join(quotes)
    return f' (evidence: "{joined}")'


def _claim_prefix(atom_id: str, relation: SupportRelation) -> str:
    if relation in _FLAGGED_RELATIONS:
        return f"- [{atom_id}] ⚠ {relation.value.upper()}:"
    return f"- [{atom_id}]"


def _format_atom_line(record: EncounterRecord, atom: ClinicalAtom, relation: SupportRelation) -> str:
    cite = _evidence_citation(record, atom)
    value = atom.raw_value
    prefix = _claim_prefix(atom.atom_id, relation)
    if atom.assertion_state is AssertionState.DENIED:
        return f"{prefix} Denies {value}{cite}"
    if atom.assertion_state is AssertionState.UNCERTAIN:
        return f"{prefix} Uncertain: {value}{cite}"
    if atom.assertion_state is AssertionState.CONFLICTING:
        return f"{prefix} Conflicting reports: {value}{cite}"
    return f"{prefix} {value}{cite}"


def _atoms_for_section(record: EncounterRecord, types: frozenset[AtomType]) -> tuple[ClinicalAtom, ...]:
    return tuple(atom for atom in record.atoms if atom.atom_type in types)


def _render_section(title: str, lines: list[str]) -> str | None:
    if not lines:
        return None
    return f"## {title}\n\n" + "\n".join(lines)


def _render_unresolved(items: tuple[UnresolvedItem, ...]) -> str | None:
    if not items:
        return None
    lines = [f"- [{item.unresolved_id}] {item.topic}: {item.reason}" for item in items]
    return "## Review required\n\n" + "\n".join(lines)


def _render_verification_flags(flags: tuple[ClaimFlag, ...]) -> str | None:
    if not flags:
        return None
    lines = [f"- [{flag.atom_id}] {flag.relation.value}: {flag.message}" for flag in flags]
    return "## Verification flags\n\n" + "\n".join(lines)


def render_verified_note(record: EncounterRecord) -> NoteRenderResult:
    """Render a sectioned note with claim IDs and verifier flags."""
    flags = verify_record(record)
    flag_by_id = {flag.atom_id: flag.relation for flag in flags}
    sections: list[str] = [f"# Encounter note — {record.encounter_id}"]

    for title, atom_types in _SECTION_ORDER:
        atoms = _atoms_for_section(record, atom_types)
        if not atoms:
            continue
        lines = [
            _format_atom_line(record, atom, flag_by_id.get(atom.atom_id, SupportRelation.DIRECT_EXACT))
            for atom in atoms
        ]
        block = _render_section(title, lines)
        if block:
            sections.append(block)

    other_atoms = tuple(
        atom
        for atom in record.atoms
        if not any(atom.atom_type in types for _, types in _SECTION_ORDER)
    )
    if other_atoms:
        lines = [
            _format_atom_line(record, atom, flag_by_id.get(atom.atom_id, SupportRelation.DIRECT_EXACT))
            for atom in other_atoms
        ]
        block = _render_section("Other", lines)
        if block:
            sections.append(block)

    unresolved = _render_unresolved(record.unresolved)
    if unresolved:
        sections.append(unresolved)

    verification = _render_verification_flags(flags)
    if verification:
        sections.append(verification)

    note = "\n\n".join(sections) + "\n"
    return NoteRenderResult(note=note, flags=flags)


def render_encounter_note(record: EncounterRecord) -> str:
    """Render a deterministic markdown note from a verified encounter record."""
    return render_verified_note(record).note
