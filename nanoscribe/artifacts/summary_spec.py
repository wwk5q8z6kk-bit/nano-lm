"""Structured summarization schema — nano.summary.v0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nanoscribe.artifacts.errors import artifact_fail
from nanoscribe.artifacts.validation import (
    require_mapping,
    require_nonempty_string,
    require_schema_version,
    require_string_list,
)

SUMMARY_SCHEMA_VERSION = "nano.summary.v0"


@dataclass(frozen=True, slots=True)
class SummarySection:
    heading: str
    bullets: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"heading": self.heading, "bullets": list(self.bullets)}

    @classmethod
    def from_dict(cls, data: object, *, path: str) -> SummarySection:
        mapping = require_mapping(data, path)
        bullets = mapping.get("bullets", [])
        if not isinstance(bullets, list):
            artifact_fail("type_error", "bullets must be an array", f"{path}.bullets")
        if not bullets:
            artifact_fail("empty_bullets", "bullets must be non-empty", f"{path}.bullets")
        return cls(
            heading=require_nonempty_string(mapping.get("heading", ""), f"{path}.heading"),
            bullets=require_string_list(bullets, f"{path}.bullets"),
        )


@dataclass(frozen=True, slots=True)
class SummarySpec:
    title: str
    sections: tuple[SummarySection, ...]
    source_atom_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "title": self.title,
            "sections": [section.to_dict() for section in self.sections],
            "source_atom_ids": list(self.source_atom_ids),
        }

    @classmethod
    def from_dict(cls, data: object, *, path: str = "$") -> SummarySpec:
        mapping = require_mapping(data, path)
        require_schema_version(
            mapping.get("schema_version"),
            SUMMARY_SCHEMA_VERSION,
            f"{path}.schema_version",
        )
        sections_raw = mapping.get("sections")
        if not isinstance(sections_raw, list) or not sections_raw:
            artifact_fail("type_error", "sections must be a non-empty array", f"{path}.sections")
        sections = tuple(
            SummarySection.from_dict(section, path=f"{path}.sections[{index}]")
            for index, section in enumerate(sections_raw)
        )
        source_atom_ids = mapping.get("source_atom_ids", [])
        if not isinstance(source_atom_ids, list):
            artifact_fail("type_error", "source_atom_ids must be an array", f"{path}.source_atom_ids")
        if any(not isinstance(item, str) for item in source_atom_ids):
            artifact_fail("type_error", "source_atom_ids must be strings", f"{path}.source_atom_ids")
        return cls(
            title=require_nonempty_string(mapping.get("title", ""), f"{path}.title"),
            sections=sections,
            source_atom_ids=tuple(source_atom_ids),
        )
