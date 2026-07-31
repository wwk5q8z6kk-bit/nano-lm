"""Ordered classical plugin cascade (W4).

synonym → ocr → coref → (callers may add merge/symbolic).
No LM. No fixture doc-id switches.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from wedge_v1.classical.solvers import Claim
from wedge_v1.plugins import coref, ocr, synonym


@dataclass
class CascadeResult:
    claims: list[Claim] = field(default_factory=list)
    modules_run: list[str] = field(default_factory=list)

    def to_meta(self) -> dict:
        return {"modules_run": list(self.modules_run), "n_claims": len(self.claims)}


def run_cascade(docs: dict[str, str], query: str = "", *, want: set[str] | None = None) -> CascadeResult:
    """Run selected plugins. want ⊆ {synonym, ocr, coref}; default all."""
    want = want or {"synonym", "ocr", "coref"}
    out = CascadeResult()
    ql = (query or "").lower()

    if "synonym" in want and (
        not query
        or any(k in ql for k in ("expire", "ttl", "cached", "cache", "timeout", "long"))
    ):
        out.modules_run.append("synonym")
        if query:
            c = synonym.probe_ttl(docs, query)
            if c.status != "ABSTAIN":
                out.claims.append(c)
        else:
            # scan-style: expand using a canonical paraphrastic query
            c = synonym.probe_ttl(docs, "How long before cached entries expire?")
            if c.status != "ABSTAIN":
                out.claims.append(c)

    if "ocr" in want:
        out.modules_run.append("ocr")
        out.claims.extend(ocr.probe_docs(docs))

    if "coref" in want and (
        not query
        or any(k in ql for k in ("binding", "coref", "antecedent", "pronoun"))
        or re_search_it(ql)
    ):
        out.modules_run.append("coref")
        out.claims.extend(coref.probe_docs(docs))

    return out


def re_search_it(ql: str) -> bool:
    import re

    return bool(re.search(r"it", ql))
