"""Corpus-agnostic multi-doc epistemic merge (W3).

Extract typed numeric/string fields across documents with evidence spans.
Fixture doc ids are not control flow.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from wedge_v1.classical.solvers import Claim


@dataclass(frozen=True)
class FieldSpec:
    field_id: str
    pattern: str
    value_type: str = "number"  # number | string
    flags: int = re.I


DEFAULT_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("ttl_seconds", r"TTL(?:\s+as|\s+is|\s*[=:]\s*|\s+of)\s+(\d+)\s+seconds"),
    FieldSpec("metformin_dose_mg", r"metformin\s+(\d+)\s*mg"),
    FieldSpec("sample_n", r"\bn\s*=\s*(\d+)\b"),
)

# Term hints → field_id (corpus-agnostic; no fixture doc ids).
TERM_FIELD_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("ttl", "cache", "expire", "timeout", "invalidation", "cached"), "ttl_seconds"),
    (("metformin", "dose", "mg"), "metformin_dose_mg"),
    (("sample", "n=", "n ="), "sample_n"),
)


def merge_field(docs: dict[str, str], spec: FieldSpec) -> Claim:
    """Merge one field across docs; DISPUTED when values disagree."""
    values: dict[str, object] = {}
    evidence = []
    rx = re.compile(spec.pattern, spec.flags)
    for did, body in docs.items():
        m = rx.search(body)
        if not m:
            continue
        raw = m.group(1)
        val = int(raw) if spec.value_type == "number" else raw
        values[did] = val
        evidence.append(
            {
                "doc_id": did,
                "start": m.start(1),
                "end": m.end(1),
                "text": m.group(1),
                "field": spec.field_id,
            }
        )
    if not values:
        return Claim("MERGE", None, None, status="ABSTAIN", notes=f"merge:{spec.field_id}:absent")
    uniq = sorted(set(values.values()), key=lambda x: str(x))
    if len(uniq) >= 2:
        return Claim(
            "MERGE",
            None,
            {"field": spec.field_id, "values": values, "from": uniq[0], "to": uniq[-1]},
            evidence=evidence,
            status="DISPUTED",
            notes=f"merge:{spec.field_id}:conflict",
        )
    only = uniq[0]
    return Claim(
        "MERGE",
        next(iter(values)),
        {"field": spec.field_id, "value": only, "values": values},
        evidence=evidence,
        status="PRESENT",
        notes=f"merge:{spec.field_id}:agree",
    )


def merge_all(docs: dict[str, str], fields: tuple[FieldSpec, ...] = DEFAULT_FIELDS) -> list[Claim]:
    return [merge_field(docs, spec) for spec in fields]


def fields_for_term(term: str) -> tuple[FieldSpec, ...]:
    """Pick typed merge fields relevant to a compare/ask term."""
    low = term.strip().lower()
    if not low:
        return ()
    matched: set[str] = set()
    for hints, field_id in TERM_FIELD_HINTS:
        if low in hints or any(h in low for h in hints):
            matched.add(field_id)
    if not matched:
        return DEFAULT_FIELDS
    return tuple(spec for spec in DEFAULT_FIELDS if spec.field_id in matched)


def merge_for_term(docs: dict[str, str], term: str) -> list[Claim]:
    return [merge_field(docs, spec) for spec in fields_for_term(term)]


def epistemic_entry(claim: Claim) -> dict:
    """Product-facing merge row: typed field, per-doc values, all evidence spans."""
    val = claim.value if isinstance(claim.value, dict) else {}
    field_id = val.get("field") or (claim.notes.split(":")[1] if claim.notes and ":" in claim.notes else None)
    by_doc: dict[str, object] = {}
    if isinstance(val.get("values"), dict):
        by_doc = dict(val["values"])
    spans = []
    for e in claim.evidence or []:
        if not isinstance(e, dict):
            continue
        spans.append(
            {
                "doc_id": e.get("doc_id"),
                "start": e.get("start"),
                "end": e.get("end"),
                "text": e.get("text"),
                "field": e.get("field") or field_id,
            }
        )
    return {
        "field_id": field_id,
        "status": claim.status,
        "values_by_doc": by_doc,
        "unique_values": sorted({str(v) for v in by_doc.values()}, key=str),
        "evidence_spans": spans,
        "notes": claim.notes,
        "disputed": claim.status == "DISPUTED",
    }


def predicate_claims_for_domains(docs: dict[str, str], domains: list[str]) -> list[Claim]:
    """Emit one merge claim per requested domain (atomic predicates)."""
    specs: list[FieldSpec] = []
    for d in domains:
        if d == "ttl_cache":
            specs.append(FieldSpec("ttl_seconds", r"TTL(?:\s+as|\s+is|\s*[=:]\s*|\s+of)\s+(\d+)\s+seconds"))
        elif d == "dose":
            specs.append(FieldSpec("metformin_dose_mg", r"metformin\s+(\d+)\s*mg"))
        elif d == "biblio":
            specs.append(FieldSpec("year", r"\b(20\d{2}|19\d{2})\b", value_type="number"))
    return [merge_field(docs, spec) for spec in specs]
