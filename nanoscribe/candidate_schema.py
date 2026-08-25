"""JSON Schema for CandidateAtom — generated from the canonical adapt contract."""

from __future__ import annotations

from typing import Any

from nanoscribe.adapt import CANDIDATE_SCHEMA_VERSION, CandidateAtom
from nanoscribe.encounter import (
    AssertionState,
    AtomType,
    Certainty,
    Experiencer,
    Speaker,
    Temporality,
)


def _enum_values(enum_type: type) -> list[str]:
    return [member.value for member in enum_type]


def _temporality_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": _enum_values(Temporality)},
            "onset_raw": {"type": ["string", "null"]},
            "duration_raw": {"type": ["string", "null"]},
            "time_expression_raw": {"type": ["string", "null"]},
        },
        "required": ["kind"],
        "additionalProperties": False,
    }


def candidate_atom_json_schema() -> dict[str, Any]:
    """OpenAI function-parameter item schema derived from CandidateAtom fields."""
    # Keys mirror CandidateAtom.to_dict() minus trusted-evidence fields rejected in from_dict.
    _ = CandidateAtom  # anchor contract — schema must stay aligned with from_dict validation
    return {
        "type": "object",
        "properties": {
            "atom_id": {"type": "string", "description": "Slot id from the prompt"},
            "atom_type": {"type": "string", "enum": _enum_values(AtomType)},
            "raw_value": {"type": "string"},
            "assertion_state": {
                "type": "string",
                "enum": _enum_values(AssertionState),
            },
            "speaker": {"type": "string", "enum": _enum_values(Speaker)},
            "experiencer": {"type": "string", "enum": _enum_values(Experiencer)},
            "temporality": _temporality_schema(),
            "certainty": {"type": "string", "enum": _enum_values(Certainty)},
            "evidence_quote": {
                "type": "string",
                "description": "Verbatim substring from transcript; omit when abstained",
            },
            "quotes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional multiple verbatim quotes",
            },
            "review_required": {"type": "boolean"},
            "abstained": {"type": "boolean"},
            "malformed": {"type": "boolean"},
        },
        "required": ["atom_id"],
        "additionalProperties": False,
    }


def candidate_batch_parameters_schema() -> dict[str, Any]:
    """Parameters object for submit_candidate_atoms tool calls."""
    return {
        "type": "object",
        "properties": {
            "schema_version": {
                "type": "string",
                "enum": [CANDIDATE_SCHEMA_VERSION],
            },
            "atoms": {
                "type": "array",
                "items": candidate_atom_json_schema(),
            },
        },
        "required": ["schema_version", "atoms"],
        "additionalProperties": False,
    }
