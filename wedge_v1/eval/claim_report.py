"""Claim-level reporting layer (LAB.B26) — instrument health ≠ decision utility."""
from __future__ import annotations

from collections import Counter
from typing import Any


def claim_level_report(claims: list[Any]) -> dict:
    statuses = Counter(getattr(c, "status", "?") for c in claims)
    presented = [c for c in claims if getattr(c, "status", None) in {"PRESENT", "CONFIRMED"}]
    with_evidence = [
        c for c in presented
        if getattr(c, "evidence", None)
    ]
    return {
        "n_claims": len(claims),
        "by_status": dict(statuses),
        "n_presented": len(presented),
        "n_presented_with_evidence": len(with_evidence),
        "n_abstain": statuses.get("ABSTAIN", 0),
        "n_disputed": statuses.get("DISPUTED", 0),
        "n_missing": statuses.get("MISSING", 0),
        "note": "Instrument layer; official U remains DRAFT until AUTHORIZE_WEDGE_V1_U_FREEZE",
    }
