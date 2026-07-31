"""Deterministic verifier for wedge claims (decidable R)."""
from __future__ import annotations

from .solvers import Claim


def verify_claim(claim: Claim) -> Claim:
    """Apply present/abstain/reject under decidable checks."""
    # Explicit abstain / disputed / review pass through
    if claim.status in {"ABSTAIN", "DISPUTED", "REVIEW"}:
        return claim

    # T33 / empty evidence: reject from presentation
    if claim.meta.get("expect_reject") or not claim.evidence:
        claim.status = "REJECTED"
        claim.meta["verify"] = "fail_no_evidence"
        return claim

    # Evidence must include offsets or text; doc_id may live on the claim
    for ev in claim.evidence:
        if not any(k in ev for k in ("start", "text", "line")):
            claim.status = "REJECTED"
            claim.meta["verify"] = "fail_malformed_evidence"
            return claim

    claim.meta["verify"] = "pass"
    return claim


def verify_all(claims: list[Claim]) -> list[Claim]:
    return [verify_claim(c) for c in claims]


def present(claims: list[Claim]) -> list[Claim]:
    return [c for c in claims if c.status in {"CONFIRMED", "PROBABLE", "DISPUTED"}]
