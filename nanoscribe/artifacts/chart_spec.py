"""Chart artifact stub — nano.chart.v0 (validation only, no renderer)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nanoscribe.artifacts.errors import artifact_fail
from nanoscribe.artifacts.validation import require_mapping, require_nonempty_string, require_schema_version

CHART_SCHEMA_VERSION = "nano.chart.v0"
ALLOWED_CHART_KINDS = frozenset({"line", "bar", "scatter", "pie", "stub"})


@dataclass(frozen=True, slots=True)
class ChartSpec:
    """Minimal chart contract — extend when chart capability ships."""

    title: str
    chart_kind: str
    x_label: str | None = None
    y_label: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CHART_SCHEMA_VERSION,
            "title": self.title,
            "chart_kind": self.chart_kind,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: object, *, path: str = "$") -> ChartSpec:
        mapping = require_mapping(data, path)
        require_schema_version(
            mapping.get("schema_version"),
            CHART_SCHEMA_VERSION,
            f"{path}.schema_version",
        )
        chart_kind = mapping.get("chart_kind", "stub")
        if not isinstance(chart_kind, str) or chart_kind not in ALLOWED_CHART_KINDS:
            allowed = ", ".join(sorted(ALLOWED_CHART_KINDS))
            artifact_fail("invalid_enum", f"chart_kind must be one of: {allowed}", f"{path}.chart_kind")
        x_label = mapping.get("x_label")
        y_label = mapping.get("y_label")
        if x_label is not None and not isinstance(x_label, str):
            artifact_fail("type_error", "x_label must be a string or null", f"{path}.x_label")
        if y_label is not None and not isinstance(y_label, str):
            artifact_fail("type_error", "y_label must be a string or null", f"{path}.y_label")
        notes = mapping.get("notes", "")
        if not isinstance(notes, str):
            artifact_fail("type_error", "notes must be a string", f"{path}.notes")
        return cls(
            title=require_nonempty_string(mapping.get("title", ""), f"{path}.title"),
            chart_kind=chart_kind,
            x_label=x_label,
            y_label=y_label,
            notes=notes,
        )
