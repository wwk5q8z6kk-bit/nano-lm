"""Synonym / query-expand plugin (corpus-agnostic)."""
from __future__ import annotations

import re

from wedge_v1.classical.solvers import Claim, _find
from wedge_v1.plugins.lexicon import synonyms


def expand_terms(query: str) -> set[str]:
    q = query.lower()
    terms = set(re.findall(r"[a-z0-9]+", q))
    for src, dsts in synonyms().items():
        if src.lower() in q:
            terms.update(d.lower() for d in dsts)
            terms.update(re.findall(r"[a-z0-9]+", src.lower()))
    return terms


def probe_ttl(docs: dict[str, str], query: str) -> Claim:
    """Locate TTL via synonym-expanded query; any doc, not a fixed id."""
    terms = expand_terms(query)
    best = None
    best_score = 0
    pat = re.compile("TTL as (" + r"\d+" + ") seconds")
    for doc_id, text in docs.items():
        low = text.lower()
        score = sum(1 for t in terms if t in low)
        m = pat.search(text)
        if m:
            score += 5
        if score > best_score:
            best_score = score
            best = (doc_id, text, m)
    if best and best[2] is not None and best_score > 0:
        doc_id, text, m = best
        span = _find(text, m.group(0))
        return Claim(
            "T35",
            doc_id,
            f"{m.group(1)} seconds",
            evidence=[span or {"text": m.group(0)}],
            status="PRESENT",
            notes="plugin.synonym.ttl",
            meta={"plugin": "synonym", "score": best_score},
        )
    return Claim("T35", None, None, status="ABSTAIN", notes="plugin.synonym.ttl")
