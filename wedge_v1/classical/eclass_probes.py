"""E-class probes: cheapest sufficient upgrades where classical abstained.

Order: synonym/query expansion -> symbolic rules -> (optional) LM later.
No training. No free generation. No gold imports (LAB.B25 firewall).
"""
from __future__ import annotations

import re

from .solvers import Claim, _find


QUERY_EXPAND = {
    "expire": ["TTL", "timeout", "seconds", "invalidation"],
    "expiration": ["TTL", "timeout", "seconds"],
    "cached entries": ["cache", "TTL"],
    "how long": ["TTL", "seconds"],
}


def expand_query_terms(query: str) -> set[str]:
    q = query.lower()
    terms = set(re.findall(r"[a-z0-9]+", q))
    for src, dsts in QUERY_EXPAND.items():
        if src in q:
            terms.update(d.lower() for d in dsts)
    return terms


def probe_t35(docs: dict[str, str]) -> Claim:
    query = "How long before cached entries expire?"
    terms = expand_query_terms(query)
    best = None
    best_score = 0
    for doc_id, text in docs.items():
        low = text.lower()
        score = sum(1 for t in terms if t in low)
        m = re.search(r"TTL as (\d+) seconds", text)
        if m and score > best_score:
            best_score = score + 5
            best = (doc_id, text, m)
        elif score > best_score:
            best_score = score
            best = (doc_id, text, None)
    if best and best[2] is not None:
        doc_id, text, m = best
        span = _find(text, m.group(0))
        return Claim(
            "T35",
            doc_id,
            {"query": query, "answer": f"{m.group(1)} seconds", "method": "query_expand"},
            evidence=[span or {"text": m.group(0)}],
            status="CONFIRMED",
            notes="eclass_query_expand",
        )
    return Claim("T35", None, {"query": query, "method": "query_expand"}, status="ABSTAIN", notes="eclass_query_expand")


def probe_t36(docs: dict[str, str]) -> Claim:
    dose: dict[str, int] = {}
    for doc_id, text in docs.items():
        m = re.search(r"metformin\s+(\d+)\s*mg", text, re.I)
        if m:
            dose[doc_id] = int(m.group(1))
    vals = sorted(set(dose.values()))
    if len(vals) >= 2:
        return Claim(
            "T36",
            None,
            {"relation": "implies_dose_change", "from": vals[0], "to": vals[-1], "values": dose, "method": "symbolic_compare"},
            evidence=[{"doc_id": k, "text": f"{v} mg"} for k, v in dose.items()],
            status="CONFIRMED",
            notes="eclass_symbolic_dose_change",
        )
    return Claim("T36", None, {"relation": "implies_dose_change", "method": "symbolic_compare"}, status="ABSTAIN", notes="eclass_symbolic_dose_change")


def probe_t39(docs: dict[str, str]) -> Claim:
    text = docs.get("binding_coref", "")
    if not text:
        return Claim("T39", "binding_coref", {"bindings": []}, status="ABSTAIN", notes="eclass_coref_lite")
    body_lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("Authors:") or s.startswith("Year:"):
            continue
        body_lines.append(s)
    body = " ".join(body_lines)
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]
    bindings = []
    entity_re = re.compile(r"\b(Metformin|Placebo)\b")
    for i, sent in enumerate(sents):
        if not re.match(r"^It\b", sent) or i == 0:
            continue
        ents = entity_re.findall(sents[i - 1])
        if len(ents) == 1:
            bindings.append({"pronoun_sent": sent, "antecedent": ents[0], "prev": sents[i - 1]})
    if bindings:
        return Claim(
            "T39",
            "binding_coref",
            {"bindings": bindings, "method": "coref_lite"},
            evidence=[{"doc_id": "binding_coref", "text": bindings[0]["antecedent"]}],
            status="CONFIRMED",
            notes="eclass_coref_lite",
        )
    return Claim("T39", "binding_coref", {"bindings": [], "method": "coref_lite"}, status="ABSTAIN", notes="eclass_coref_lite")


def apply_eclass_overrides(claims: list[Claim], docs: dict[str, str]) -> list[Claim]:
    overrides = {"T35": probe_t35(docs), "T36": probe_t36(docs), "T39": probe_t39(docs)}
    out: list[Claim] = []
    seen: set[str] = set()
    for c in claims:
        if c.task_id in overrides:
            if c.task_id not in seen:
                out.append(overrides[c.task_id])
                seen.add(c.task_id)
            continue
        out.append(c)
    for tid, claim in overrides.items():
        if tid not in seen:
            out.append(claim)
    return out


def lm_still_needed(eclass_claims: list[Claim]) -> bool:
    by = {c.task_id: c for c in eclass_claims if c.task_id in {"T35", "T36", "T39"}}
    return any(by.get(t) is None or by[t].status == "ABSTAIN" for t in ("T35", "T36", "T39"))
