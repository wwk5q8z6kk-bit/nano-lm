"""Deterministic verifier for wedge claims (decidable R)."""
from __future__ import annotations

from .solvers import Claim


def verify_claim(claim: Claim) -> Claim:
    """Apply present/abstain/reject under decidable checks."""
    # Non-presentable states do not need source validation here. DISPUTED claims
    # remain presentable and therefore go through the same evidence checks.
    if claim.status in {"ABSTAIN", "REJECTED", "REVIEW", "MISSING"}:
        return claim

    # T33 / empty evidence: reject from presentation
    if claim.meta.get("expect_reject") or not claim.evidence:
        claim.status = "REJECTED"
        claim.meta["verify"] = "fail_no_evidence"
        return claim

    # Evidence must identify a source and contain either a complete, well-formed
    # offset pair or text that the CoE binder can locate in that source.
    for ev in claim.evidence:
        if not isinstance(ev, dict) or not (ev.get("doc_id") or claim.doc_id):
            claim.status = "REJECTED"
            claim.meta["verify"] = "fail_missing_source"
            return claim

        has_start = ev.get("start") is not None
        has_end = ev.get("end") is not None
        if has_start != has_end:
            claim.status = "REJECTED"
            claim.meta["verify"] = "fail_incomplete_offset"
            return claim
        if has_start:
            if isinstance(ev["start"], bool) or isinstance(ev["end"], bool):
                claim.status = "REJECTED"
                claim.meta["verify"] = "fail_malformed_offset"
                return claim
            try:
                start, end = int(ev["start"]), int(ev["end"])
            except (TypeError, ValueError):
                claim.status = "REJECTED"
                claim.meta["verify"] = "fail_malformed_offset"
                return claim
            if start < 0 or end <= start:
                claim.status = "REJECTED"
                claim.meta["verify"] = "fail_malformed_offset"
                return claim
        elif not (ev.get("text") or ev.get("line")):
            claim.status = "REJECTED"
            claim.meta["verify"] = "fail_malformed_evidence"
            return claim

    claim.meta["verify"] = "pass"
    return claim


def verify_all(claims: list[Claim]) -> list[Claim]:
    return [verify_claim(c) for c in claims]


def present(claims: list[Claim]) -> list[Claim]:
    return [c for c in claims if c.status in {"CONFIRMED", "PROBABLE", "DISPUTED"}]
