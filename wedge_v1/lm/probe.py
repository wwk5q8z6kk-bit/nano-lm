"""Constructive-faithfulness LM probe harness (stub backend default).

Allowlisted E-class tasks only (T35/T36/T39). A–D stay classical-only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from wedge_v1.classical.bm25 import top_paragraphs
from wedge_v1.classical.solvers import Claim, _find
from wedge_v1.classical.verifier import verify_claim
from wedge_v1.plugins import synonym as synonym_plugin

ALLOWLIST_TASKS = frozenset({"T35", "T36", "T39"})

T35_QUERY = "How long before cached entries expire?"
TTL_PAT = re.compile(r"TTL\s+(?:as|is)\s+(\d+)\s+seconds", re.I)
DOSE_PAT = re.compile(r"metformin\s+(\d+)\s*mg", re.I)


class LMBackend(Protocol):
    name: str

    def propose(self, task_id: str, docs: dict[str, str], *, query: str = "") -> Claim: ...


@dataclass
class StubLMBackend:
    """Span-first stub: retrieve → lock substring → verify. No generation."""

    name: str = "stub_constructive"

    def propose(self, task_id: str, docs: dict[str, str], *, query: str = "") -> Claim:
        if task_id not in ALLOWLIST_TASKS:
            return Claim(task_id, None, None, status="ABSTAIN", notes="lm_not_allowlisted")
        if task_id == "T35":
            return self._t35(docs, query or T35_QUERY)
        if task_id == "T36":
            return self._t36(docs)
        return self._t39(docs)

    def _t35(self, docs: dict[str, str], query: str) -> Claim:
        plug = synonym_plugin.probe_ttl(docs, query)
        if plug.status == "PRESENT" and plug.evidence:
            c = Claim(
                "T35",
                plug.doc_id,
                {"query": query, "answer": plug.value, "method": self.name},
                evidence=plug.evidence,
                status="PRESENT",
                notes="lm_stub_span_locked",
            )
            return verify_claim(c)
        hits = top_paragraphs(docs, query, k=3)
        for hit in hits:
            if not hit.get("promote"):
                continue
            text = hit["text"]
            m = TTL_PAT.search(text)
            if not m:
                continue
            span_text = m.group(0)
            i = text.find(span_text)
            if i < 0:
                continue
            ev = [{
                "start": hit["start"] + i,
                "end": hit["start"] + i + len(span_text),
                "text": span_text,
            }]
            c = Claim(
                "T35",
                hit["doc_id"],
                {"query": query, "answer": f"{m.group(1)} seconds", "method": self.name},
                evidence=ev,
                status="PRESENT",
                notes="lm_stub_span_locked",
            )
            return verify_claim(c)
        return Claim("T35", None, {"query": query, "method": self.name}, status="ABSTAIN", notes="lm_stub_no_span")

    def _t36(self, docs: dict[str, str]) -> Claim:
        dose: dict[str, int] = {}
        evidence = []
        for doc_id, text in docs.items():
            m = DOSE_PAT.search(text)
            if m:
                dose[doc_id] = int(m.group(1))
                span = _find(text, m.group(0))
                if span:
                    evidence.append(span)
        vals = sorted(set(dose.values()))
        if len(vals) >= 2 and evidence:
            c = Claim(
                "T36",
                None,
                {"relation": "implies_dose_change", "from": vals[0], "to": vals[-1], "values": dose, "method": self.name},
                evidence=evidence,
                status="PRESENT",
                notes="lm_stub_symbolic_lock",
            )
            return verify_claim(c)
        return Claim("T36", None, {"method": self.name}, status="ABSTAIN", notes="lm_stub_no_dose_change")

    def _t39(self, docs: dict[str, str]) -> Claim:
        from wedge_v1.plugins import coref as coref_plugin

        found = coref_plugin.probe_docs(docs)
        if found and found[0].status == "PRESENT":
            c = found[0]
            c.task_id = "T39"
            c.notes = "lm_stub_coref"
            c.meta["method"] = self.name
            return verify_claim(c)
        return Claim("T39", None, {"method": self.name}, status="ABSTAIN", notes="lm_stub_coref_miss")


def probe_eclass_task(
    task_id: str,
    docs: dict[str, str],
    *,
    backend: LMBackend | None = None,
    classical_status: str = "ABSTAIN",
    query: str = "",
) -> dict[str, Any]:
    """Run an LM probe on one allowlisted task after classical abstention."""
    backend = backend or StubLMBackend()
    if task_id not in ALLOWLIST_TASKS:
        return {
            "task_id": task_id,
            "skipped": True,
            "reason": "not_allowlisted",
            "lm_invoked": False,
        }
    if classical_status not in {"ABSTAIN", "MISSING", "REJECTED"}:
        return {
            "task_id": task_id,
            "skipped": True,
            "reason": "classical_already_resolved",
            "classical_status": classical_status,
            "lm_invoked": False,
        }
    claim = backend.propose(task_id, docs, query=query)
    return {
        "task_id": task_id,
        "backend": backend.name,
        "lm_invoked": True,
        "claim_status": claim.status,
        "value": claim.value,
        "evidence_n": len(claim.evidence or []),
        "verify": claim.meta.get("verify"),
        "notes": claim.notes,
        "constructive_faithfulness": bool(claim.evidence) or claim.status == "ABSTAIN",
    }


def ablation_fails_support(claim: Claim, docs: dict[str, str]) -> bool:
    """Remove cited offsets — support must fail (B13 constructive faithfulness)."""
    if claim.status != "PRESENT" or not claim.evidence:
        return True
    for ev in claim.evidence:
        did = claim.doc_id or ev.get("doc_id")
        body = docs.get(did or "", "")
        start, end = ev.get("start"), ev.get("end")
        if body and start is not None and end is not None:
            if body[int(start):int(end)] != (ev.get("text") or ""):
                return True
    # Simulate ablation: empty evidence → should not stay PRESENT
    stripped = Claim(claim.task_id, claim.doc_id, claim.value, evidence=[], status="PRESENT")
    v = verify_claim(stripped)
    return v.status == "REJECTED" or v.meta.get("verify") == "fail_no_evidence"
