"""Atomic predicate decomposition for compound verified-ask queries (W3 + CoE).

Splits conjunctions into minimal support conditions. A compound answer is only
PRESENT/CONTRADICTED when every atomic predicate is evidenced.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


_SPLIT = re.compile(r"\s+and\s+|\s*;\s*", re.I)

DOMAIN_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("ttl_cache", ("ttl", "cache", "expire", "cached", "timeout", "seconds")),
    ("dose", ("dose", "metformin", "mg", "dosage")),
    ("biblio", ("author", "title", "year", "doi")),
    ("entity", ("binding", "coref", "antecedent", "pronoun")),
]


@dataclass
class AtomicPredicate:
    pred_id: str
    domain: str
    surface: str
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PredicateSupport:
    pred_id: str
    domain: str
    surface: str
    supported: bool
    support_kind: str  # claim | lexical_only | none
    evidence_hint: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _classify_domain(text: str) -> tuple[str, list[str]]:
    ql = text.lower()
    for domain, keys in DOMAIN_RULES:
        hit = [k for k in keys if k in ql]
        if hit:
            return domain, hit
    toks = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text)
    stop = {"what", "the", "is", "are", "for", "how", "long", "before", "stated", "please", "and"}
    keys = [t.lower() for t in toks if t.lower() not in stop][:6]
    return "open", keys


def decompose(query: str) -> list[AtomicPredicate]:
    q = (query or "").strip()
    if not q:
        return []
    parts = [p.strip() for p in _SPLIT.split(q) if p.strip()]
    domain_map = dict(DOMAIN_RULES)
    if len(parts) == 1:
        ql = q.lower()
        domains = [d for d, keys in DOMAIN_RULES if any(k in ql for k in keys)]
        if len(domains) >= 2:
            return [
                AtomicPredicate(f"p{i+1}", d, q, [k for k in domain_map[d] if k in ql])
                for i, d in enumerate(domains)
            ]
        domain, keys = _classify_domain(q)
        return [AtomicPredicate("p1", domain, q, keys)]

    out: list[AtomicPredicate] = []
    for i, part in enumerate(parts, 1):
        domain, keys = _classify_domain(part)
        out.append(AtomicPredicate(f"p{i}", domain, part, keys))
    return out


def _claim_blob(claims: list[Any]) -> str:
    parts = []
    for c in claims:
        if hasattr(c, "value"):
            parts.append(f"{c.value} {c.notes} {c.task_id} {c.doc_id}")
            for e in c.evidence or []:
                if isinstance(e, dict):
                    parts.append(str(e.get("text") or e.get("line") or ""))
        elif isinstance(c, dict):
            parts.append(str(c))
    return " ".join(parts).lower()


def _lexical_support(docs: dict[str, str], keywords: list[str]) -> tuple[bool, str]:
    if not keywords:
        return False, ""
    for did, body in docs.items():
        low = body.lower()
        hits = [k for k in keywords if len(k) >= 3 and k.lower() in low]
        need = 1 if len(keywords) <= 2 else max(1, (len(keywords) + 1) // 2)
        if len(hits) >= need:
            return True, f"{did}:{','.join(hits[:4])}"
    return False, ""


def evaluate_predicates(
    preds: list[AtomicPredicate],
    docs: dict[str, str],
    claims: list[Any],
) -> list[PredicateSupport]:
    blob = _claim_blob(claims)
    out: list[PredicateSupport] = []
    for p in preds:
        supported = False
        kind = "none"
        hint = ""
        if p.domain == "ttl_cache" and any(
            k in blob for k in ("ttl", "seconds", "cache", "300", "timeout")
        ):
            supported, kind, hint = True, "claim", "ttl_cache"
        elif p.domain == "dose" and any(
            k in blob for k in ("metformin", "mg", "dose", "500", "850")
        ):
            supported, kind, hint = True, "claim", "dose"
        elif p.domain == "biblio" and any(k in blob for k in ("author", "title", "year", "doi")):
            supported, kind, hint = True, "claim", "biblio"
        elif p.domain == "entity" and any(k in blob for k in ("binding", "coref", "antecedent")):
            supported, kind, hint = True, "claim", "entity"
        else:
            ok, h = _lexical_support(docs, p.keywords)
            if ok and any(k in blob for k in p.keywords if len(k) >= 3):
                supported, kind, hint = True, "claim", h
            elif ok:
                supported, kind, hint = False, "lexical_only", h
            else:
                supported, kind, hint = False, "none", ""
        out.append(
            PredicateSupport(
                pred_id=p.pred_id,
                domain=p.domain,
                surface=p.surface,
                supported=supported,
                support_kind=kind,
                evidence_hint=hint,
            )
        )
    return out


def incomplete_conjunction(supports: list[PredicateSupport]) -> bool:
    return len(supports) >= 2 and any(not s.supported for s in supports)
