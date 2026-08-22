"""Corpus-agnostic multi-doc epistemic merge (W3).

Extract typed numeric/string fields across documents with evidence spans.
Fixture doc ids are not control flow. Field inventory lives in
``field_registry.json`` (defaults) and can be extended without editing this
module's hard-coded triples.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from wedge_v1.classical.solvers import Claim

REGISTRY_PATH = Path(__file__).with_name("field_registry.json")


@dataclass(frozen=True)
class FieldSpec:
    field_id: str
    pattern: str
    value_type: str = "number"  # number | string
    flags: int = re.I
    term_hints: tuple[str, ...] = ()


def _parse_value(raw: str, value_type: str) -> object:
    if value_type != "number":
        return raw
    if "." in raw:
        return float(raw)
    return int(raw)


def _specs_from_payload(payload: dict) -> tuple[FieldSpec, ...]:
    out: list[FieldSpec] = []
    for row in payload.get("fields") or []:
        if not isinstance(row, dict):
            continue
        fid = str(row.get("field_id") or "").strip()
        pat = str(row.get("pattern") or "").strip()
        if not fid or not pat:
            continue
        hints = row.get("term_hints") or []
        if not isinstance(hints, list):
            hints = []
        out.append(
            FieldSpec(
                field_id=fid,
                pattern=pat,
                value_type=str(row.get("value_type") or "number"),
                term_hints=tuple(str(h) for h in hints),
            )
        )
    return tuple(out)


def load_field_registry(path: Path | None = None) -> tuple[FieldSpec, ...]:
    """Load typed-field specs from JSON; empty if missing/invalid."""
    p = path or REGISTRY_PATH
    if not p.exists():
        return ()
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    if not isinstance(payload, dict):
        return ()
    return _specs_from_payload(payload)


# Built-in fallback if registry file is absent (keeps imports resilient).
_FALLBACK_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec(
        "ttl_seconds",
        r"TTL(?:\s+as|\s+is|\s*[=:]\s*|\s+of)\s+(\d+)\s+seconds",
        term_hints=("ttl", "cache", "expire", "timeout", "invalidation", "cached"),
    ),
    FieldSpec(
        "metformin_dose_mg",
        r"metformin\s+(\d+)\s*mg",
        term_hints=("metformin", "dose", "mg"),
    ),
    FieldSpec(
        "sample_n",
        r"\bn\s*=\s*(\d+)\b",
        term_hints=("sample", "n=", "n =", "sample size"),
    ),
    FieldSpec(
        "peak_qps",
        r"(?:peak\s+)?QPS\s+(?:is|remains|=|:)?\s*(\d+)",
        term_hints=("qps", "throughput", "queries per second", "peak qps"),
    ),
)


def active_fields(extra: tuple[FieldSpec, ...] = ()) -> tuple[FieldSpec, ...]:
    """Registry ∪ fallback ∪ caller extras; first field_id wins."""
    seen: set[str] = set()
    out: list[FieldSpec] = []
    for spec in (*load_field_registry(), *_FALLBACK_FIELDS, *extra):
        if spec.field_id in seen:
            continue
        seen.add(spec.field_id)
        out.append(spec)
    return tuple(out)


# Public default inventory (resolved at import; reload via active_fields()).
DEFAULT_FIELDS: tuple[FieldSpec, ...] = active_fields()

# Backward-compatible hint table derived from specs.
TERM_FIELD_HINTS: tuple[tuple[tuple[str, ...], str], ...] = tuple(
    (spec.term_hints, spec.field_id) for spec in DEFAULT_FIELDS if spec.term_hints
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
        val = _parse_value(raw, spec.value_type)
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


def merge_all(docs: dict[str, str], fields: tuple[FieldSpec, ...] | None = None) -> list[Claim]:
    return [merge_field(docs, spec) for spec in (fields if fields is not None else active_fields())]


def fields_for_term(term: str, fields: tuple[FieldSpec, ...] | None = None) -> tuple[FieldSpec, ...]:
    """Pick typed merge fields relevant to a compare/ask term."""
    low = term.strip().lower()
    if not low:
        return ()
    inventory = fields if fields is not None else active_fields()
    matched: set[str] = set()
    for spec in inventory:
        hints = spec.term_hints or (spec.field_id.replace("_", " "),)
        if low == spec.field_id or low in hints or any(h in low for h in hints):
            matched.add(spec.field_id)
        elif any(low == h or low in h for h in hints):
            matched.add(spec.field_id)
    if not matched:
        return ()
    return tuple(spec for spec in inventory if spec.field_id in matched)


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
    by_id = {s.field_id: s for s in active_fields()}
    specs: list[FieldSpec] = []
    for d in domains:
        if d in {"ttl_cache", "ttl"}:
            specs.append(
                by_id.get("ttl_seconds")
                or FieldSpec("ttl_seconds", r"TTL(?:\s+as|\s+is|\s*[=:]\s*|\s+of)\s+(\d+)\s+seconds")
            )
        elif d == "dose":
            specs.append(
                by_id.get("metformin_dose_mg")
                or FieldSpec("metformin_dose_mg", r"metformin\s+(\d+)\s*mg")
            )
        elif d in {"biblio", "year"}:
            specs.append(FieldSpec("year", r"\b(20\d{2}|19\d{2})\b", value_type="number", term_hints=("year",)))
        elif d in {"throughput", "qps"}:
            specs.append(
                by_id.get("peak_qps")
                or FieldSpec("peak_qps", r"(?:peak\s+)?QPS\s+(?:is|remains|=|:)?\s*(\d+)")
            )
    return [merge_field(docs, spec) for spec in specs]
