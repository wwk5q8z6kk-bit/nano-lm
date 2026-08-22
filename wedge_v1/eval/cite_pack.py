"""Dense citation packing for verified claims (product UX; not Layer-1)."""
from __future__ import annotations

from typing import Any


def pack_evidence(evidence: list[dict] | None, *, max_spans: int = 3) -> list[dict[str, Any]]:
    """Collapse evidence dicts into compact citation records."""
    packed: list[dict[str, Any]] = []
    for ev in (evidence or [])[:max_spans]:
        if not isinstance(ev, dict):
            continue
        text = (ev.get("text") or ev.get("line") or "").strip()
        if not text:
            continue
        cite = {
            "doc_id": ev.get("doc_id"),
            "start": ev.get("start"),
            "end": ev.get("end"),
            "quote": text if len(text) <= 160 else text[:157] + "...",
        }
        if ev.get("context"):
            ctx = str(ev["context"]).strip()
            cite["context"] = ctx if len(ctx) <= 120 else ctx[:117] + "..."
        packed.append(cite)
    return packed


def pack_claim(claim: dict[str, Any], *, max_spans: int = 3) -> dict[str, Any]:
    """Pack one claim dict into a citation-first card."""
    value = claim.get("value")
    if isinstance(value, dict):
        short = value.get("answer") or value.get("term") or value.get("relation")
        if short is None and "all_values" in value:
            short = value.get("all_values")
        if short is None and value.get("field"):
            vals = value.get("values") or {}
            short = f"{value.get('field')}: {vals}" if vals else value.get("field")
        display = short if short is not None else value
    else:
        display = value
    return {
        "task_id": claim.get("task_id"),
        "status": claim.get("status"),
        "doc_id": claim.get("doc_id"),
        "value": display,
        "citations": pack_evidence(claim.get("evidence"), max_spans=max_spans),
        "notes": claim.get("notes"),
    }


def pack_claims(claims: list[Any] | None, *, max_spans: int = 3) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for c in claims or []:
        if isinstance(c, dict):
            out.append(pack_claim(c, max_spans=max_spans))
    return out


def format_citation_md(cite: dict[str, Any]) -> str:
    doc = cite.get("doc_id") or "?"
    start, end = cite.get("start"), cite.get("end")
    loc = f":{start}-{end}" if start is not None and end is not None else ""
    quote = cite.get("quote") or ""
    return f'[`{doc}{loc}`] "{quote}"'


def format_packed_claims_md(claims: list[dict[str, Any]]) -> list[str]:
    """Markdown lines for packed claim cards."""
    lines: list[str] = []
    if not claims:
        lines.append("_No claims._")
        return lines
    lines.append(f"## Claims ({len(claims)})")
    lines.append("")
    for i, c in enumerate(claims, 1):
        tid = c.get("task_id") or "?"
        st = c.get("status") or "?"
        lines.append(f"{i}. **`{tid}`** `{st}` — `{c.get('value')}`")
        doc = c.get("doc_id")
        if doc:
            lines.append(f"   - doc: `{doc}`")
        for cite in c.get("citations") or []:
            lines.append(f"   - cite: {format_citation_md(cite)}")
        notes = c.get("notes")
        if notes:
            lines.append(f"   - _{notes}_")
        lines.append("")
    return lines
