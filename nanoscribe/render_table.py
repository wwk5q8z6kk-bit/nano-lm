"""Markdown renderer stub for canonical tables."""

from __future__ import annotations

from nanoscribe.artifacts import TableSpec


def render_table_markdown(spec: TableSpec) -> str:
    """Render a TableSpec as a GitHub-flavored markdown table."""
    if not spec.columns:
        return ""
    header = "| " + " | ".join(column.label for column in spec.columns) + " |"
    separator = "| " + " | ".join("---" for _ in spec.columns) + " |"
    rows = [
        "| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |"
        for row in spec.rows
    ]
    lines = [header, separator, *rows]
    if spec.title:
        return f"## {spec.title}\n\n" + "\n".join(lines)
    return "\n".join(lines)
