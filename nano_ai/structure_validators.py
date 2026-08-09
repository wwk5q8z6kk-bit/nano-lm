"""Structure validators for synthetic Markdown tables and Mermaid diagrams.

`papers/PLAN_20260805_SURFACE_ROBUSTNESS.md` P4 and its predecessor
`ENHANCED_PLAN_20260805.md` P2: cheap syntactic checks that serve three roles
at once -- evaluator (score whether a model's structured output actually
parses), synthetic-data filter (reject malformed generated examples before
they enter a training pool), and RLVR reward (a binary verifiable signal,
`StructureValidation.reward`). "Charts and diagrams" belongs here, as
validation, not as a training corpus.

These are deliberately syntax-only, not full renderers: no Mermaid compiler is
vendored (none was found open-licensed and dependency-free; see
`data/external/` for the project's provenance discipline on vendored tools),
and a full CommonMark table parser is more than this needs. A binary
pass/fail over well-defined structural invariants (row/column consistency,
known diagram-type keyword, balanced brackets) is the smallest tool that is
mathematically honest for a verifiable reward -- no calibration, threshold,
or statistical estimate is involved, so the math-toolkit formula-documentation
template does not apply.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SEPARATOR_CELL = re.compile(r"^:?-{3,}:?$")
_MERMAID_DIAGRAM_TYPES = (
    "graph",
    "flowchart",
    "sequenceDiagram",
    "classDiagram",
    "stateDiagram-v2",
    "stateDiagram",
    "erDiagram",
    "gantt",
    "pie",
    "journey",
    "mindmap",
    "gitGraph",
    "quadrantChart",
    "timeline",
)


@dataclass(frozen=True)
class StructureValidation:
    valid: bool
    reason: str

    @property
    def reward(self) -> float:
        """RLVR reward: 1.0 if structurally valid, else 0.0."""
        return 1.0 if self.valid else 0.0


def _table_row_cells(row: str) -> list[str]:
    stripped = row.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def validate_markdown_table(text: str) -> StructureValidation:
    """Check that `text` is a syntactically valid GFM-style Markdown table.

    Requires: a header row, a separator row of `---`/`:---`/`---:`/`:---:`
    cells whose count matches the header, and every data row matching that
    same column count.
    """
    rows = [line for line in text.strip("\n").splitlines() if line.strip()]
    if len(rows) < 2:
        return StructureValidation(False, "fewer than 2 non-blank lines")
    if "|" not in rows[0]:
        return StructureValidation(False, "header row has no '|' delimiter")
    header_cells = _table_row_cells(rows[0])
    sep_cells = _table_row_cells(rows[1])
    if len(sep_cells) != len(header_cells):
        return StructureValidation(
            False, "separator column count != header column count"
        )
    if not all(_SEPARATOR_CELL.match(cell) for cell in sep_cells):
        return StructureValidation(False, "row 2 is not a valid separator row")
    for lineno, row in enumerate(rows[2:], start=3):
        if len(_table_row_cells(row)) != len(header_cells):
            return StructureValidation(
                False, f"row {lineno} column count != header column count"
            )
    return StructureValidation(True, "ok")


def validate_mermaid(text: str) -> StructureValidation:
    """Check that `text` opens with a known Mermaid diagram-type declaration,
    has a non-empty body, and has balanced `()`/`[]`/`{}` brackets throughout.
    """
    lines = [
        line.strip()
        for line in text.strip().splitlines()
        if line.strip() and not line.strip().startswith("%%")
    ]
    if not lines:
        return StructureValidation(False, "empty diagram")
    first = lines[0]
    if not any(
        first == kw or first.startswith(kw + " ") or first.startswith(kw + ";")
        for kw in _MERMAID_DIAGRAM_TYPES
    ):
        return StructureValidation(
            False, f"first line {first!r} is not a known diagram type declaration"
        )
    if len(lines) < 2:
        return StructureValidation(False, "diagram declaration has no body")
    body = "\n".join(lines[1:])
    for open_ch, close_ch in (("(", ")"), ("[", "]"), ("{", "}")):
        if body.count(open_ch) != body.count(close_ch):
            return StructureValidation(
                False, f"unbalanced '{open_ch}{close_ch}' in diagram body"
            )
    return StructureValidation(True, "ok")
