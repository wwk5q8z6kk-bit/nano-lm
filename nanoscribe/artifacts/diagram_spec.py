"""Diagram artifact stub — nano.diagram.v0 (validation only, no renderer)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nanoscribe.artifacts.errors import artifact_fail
from nanoscribe.artifacts.validation import require_mapping, require_nonempty_string, require_schema_version

DIAGRAM_SCHEMA_VERSION = "nano.diagram.v0"
ALLOWED_DIAGRAM_KINDS = frozenset({"flowchart", "timeline", "entity", "stub"})


@dataclass(frozen=True, slots=True)
class DiagramSpec:
    """Minimal diagram contract — extend when diagram capability ships."""

    title: str
    diagram_kind: str
    notation: str = "mermaid"
    source_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DIAGRAM_SCHEMA_VERSION,
            "title": self.title,
            "diagram_kind": self.diagram_kind,
            "notation": self.notation,
            "source_text": self.source_text,
        }

    @classmethod
    def from_dict(cls, data: object, *, path: str = "$") -> DiagramSpec:
        mapping = require_mapping(data, path)
        require_schema_version(
            mapping.get("schema_version"),
            DIAGRAM_SCHEMA_VERSION,
            f"{path}.schema_version",
        )
        diagram_kind = mapping.get("diagram_kind", "stub")
        if not isinstance(diagram_kind, str) or diagram_kind not in ALLOWED_DIAGRAM_KINDS:
            allowed = ", ".join(sorted(ALLOWED_DIAGRAM_KINDS))
            artifact_fail("invalid_enum", f"diagram_kind must be one of: {allowed}", f"{path}.diagram_kind")
        notation = mapping.get("notation", "mermaid")
        if not isinstance(notation, str):
            artifact_fail("type_error", "notation must be a string", f"{path}.notation")
        source_text = mapping.get("source_text", "")
        if not isinstance(source_text, str):
            artifact_fail("type_error", "source_text must be a string", f"{path}.source_text")
        return cls(
            title=require_nonempty_string(mapping.get("title", ""), f"{path}.title"),
            diagram_kind=diagram_kind,
            notation=notation,
            source_text=source_text,
        )
