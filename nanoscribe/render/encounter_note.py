"""Verified encounter record → clinical note rendering (view layer, not truth)."""

from __future__ import annotations

from nanoscribe.encounter import (
    AssertionState,
    AtomType,
    ClinicalAtom,
    EncounterRecord,
    UnresolvedItem,
)

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


def _evidence_citation(record: EncounterRecord, atom: ClinicalAtom) -> str:
    quotes: list[str] = []
    for evidence_id in atom.evidence_ids:
        span = record.span(evidence_id)
        quotes.append(span.text.strip())
    if not quotes:
        return ""
    joined = "; ".join(quotes)
    return f' (evidence: "{joined}")'


def _format_atom_line(record: EncounterRecord, atom: ClinicalAtom) -> str:
    cite = _evidence_citation(record, atom)
    value = atom.raw_value
    if atom.assertion_state is AssertionState.DENIED:
        return f"- Denies {value}{cite}"
    if atom.assertion_state is AssertionState.UNCERTAIN:
        return f"- Uncertain: {value}{cite}"
    if atom.assertion_state is AssertionState.CONFLICTING:
        return f"- Conflicting reports: {value}{cite}"
    return f"- {value}{cite}"


def _atoms_for_section(record: EncounterRecord, types: frozenset[AtomType]) -> tuple[ClinicalAtom, ...]:
    return tuple(atom for atom in record.atoms if atom.atom_type in types)


def _render_section(title: str, lines: list[str]) -> str | None:
    if not lines:
        return None
    body = "\n".join(lines)
    return f"## {title}\n\n{body}"


def _render_unresolved(items: tuple[UnresolvedItem, ...]) -> str | None:
    if not items:
        return None
    lines = [f"- {item.topic}: {item.reason}" for item in items]
    return "## Review required\n\n" + "\n".join(lines)


def render_encounter_note(record: EncounterRecord) -> str:
    """Render a deterministic markdown note from a verified encounter record."""
    sections: list[str] = [f"# Encounter note — {record.encounter_id}"]
    for title, atom_types in _SECTION_ORDER:
        atoms = _atoms_for_section(record, atom_types)
        if not atoms:
            continue
        lines = [_format_atom_line(record, atom) for atom in atoms]
        block = _render_section(title, lines)
        if block:
            sections.append(block)

    other_atoms = tuple(
        atom
        for atom in record.atoms
        if not any(atom.atom_type in types for _, types in _SECTION_ORDER)
    )
    if other_atoms:
        lines = [_format_atom_line(record, atom) for atom in other_atoms]
        block = _render_section("Other", lines)
        if block:
            sections.append(block)

    unresolved = _render_unresolved(record.unresolved)
    if unresolved:
        sections.append(unresolved)

    return "\n\n".join(sections) + "\n"
