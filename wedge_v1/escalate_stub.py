"""Opt-in hybrid stub after classical ABSTAIN — never invents unsupported text."""
from __future__ import annotations

import re
import time
from typing import Any

from wedge_v1.classical import solvers as S

_TTL = re.compile(r"TTL\s+(?:as\s+)?(\d+)\s+seconds", re.I)
_NUM_UNIT = re.compile(r"(\d+(?:\.\d+)?)\s*(seconds|second|ms|qps|%)\b", re.I)

_STUB_EXPAND = {
    "invalidation": ["ttl", "cache", "seconds"],
    "invalidate": ["ttl", "cache", "seconds"],
    "memoized": ["cache", "cached", "ttl"],
    "window": ["ttl", "seconds"],
    "period": ["ttl", "seconds"],
    "duration": ["ttl", "seconds"],
    "timeout": ["ttl", "seconds"],
    "expire": ["ttl", "seconds", "invalidation"],
    "expiry": ["ttl", "seconds"],
}


def _expanded_terms(query: str) -> set[str]:
    terms = {t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2}
    for src, dsts in _STUB_EXPAND.items():
        if src in query.lower():
            terms.update(dsts)
    return terms


def _top_paragraphs(docs: dict[str, str], query: str, *, k: int = 4) -> list[dict[str, Any]]:
    terms = _expanded_terms(query)
    scored: list[tuple[int, str, str]] = []
    for did, text in docs.items():
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if len(para) < 20:
                continue
            low = para.lower()
            score = sum(1 for t in terms if t in low)
            if score > 0:
                scored.append((score, did, para))
    scored.sort(key=lambda x: (-x[0], x[1]))
    out: list[dict[str, Any]] = []
    for score, did, para in scored[:k]:
        out.append({"doc_id": did, "text": para, "score": score, "promote": score >= 2})
    return out


def _claim_from_match(did: str, text: str, match: re.Match[str], *, method: str) -> dict[str, Any]:
    span_text = match.group(0)
    local = text.find(span_text)
    evidence = []
    if local >= 0:
        evidence = [
            {
                "start": local,
                "end": local + len(span_text),
                "text": span_text,
                "line": text[max(0, text.rfind("\n", 0, local) + 1) : text.find("\n", local)].strip()[:240],
            }
        ]
    c = S.Claim(
        "ASK_STUB",
        did,
        {"answer": match.group(1) if match.lastindex else span_text, "span": span_text},
        evidence=evidence,
        status="PRESENT",
        notes=method,
    )
    return {
        "task_id": c.task_id,
        "doc_id": c.doc_id,
        "value": c.value,
        "evidence": c.evidence,
        "status": c.status,
        "notes": c.notes,
    }


def escalate_stub_ask(query: str, docs: dict[str, str]) -> dict[str, Any]:
    """Constructive span stub for classical ABSTAIN. Default path remains fail-closed."""
    t0 = time.perf_counter()
    q_low = query.lower()
    ttl_query = any(
        k in q_low
        for k in ("ttl", "expire", "expiry", "timeout", "cache", "invalidation", "memoized", "seconds")
    )

    if ttl_query:
        for did, text in docs.items():
            m = _TTL.search(text)
            if m:
                return {
                    "query": query,
                    "answer_status": "SUPPORTED",
                    "claims": [_claim_from_match(did, text, m, method="hybrid_stub_ttl_scan")],
                    "solver_path": ["hybrid_stub", "ttl_scan"],
                    "lm_invoked": False,
                    "escalation": "stub_ttl_scan",
                    "latency_s": round(time.perf_counter() - t0, 4),
                }

    q_tokens = _expanded_terms(query)
    for hit in _top_paragraphs(docs, query):
        if not hit.get("promote"):
            continue
        text = hit["text"]
        did = hit["doc_id"]
        hit_tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
        if q_tokens and len(q_tokens & hit_tokens) < max(1, min(2, len(q_tokens))):
            continue
        m = _TTL.search(text) if ttl_query else None
        if m is None and ttl_query:
            m = _NUM_UNIT.search(text)
        if m is None:
            continue
        return {
            "query": query,
            "answer_status": "SUPPORTED",
            "claims": [_claim_from_match(did, text, m, method="hybrid_stub_paragraph")],
            "solver_path": ["hybrid_stub", "paragraph_promote"],
            "lm_invoked": False,
            "escalation": "stub_paragraph",
            "latency_s": round(time.perf_counter() - t0, 4),
        }

    return {
        "query": query,
        "answer_status": "ABSTAIN",
        "claims": [],
        "solver_path": ["hybrid_stub"],
        "lm_invoked": False,
        "escalation": "stub_no_recovery",
        "latency_s": round(time.perf_counter() - t0, 4),
    }
