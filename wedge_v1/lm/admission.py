"""Decide when a marginal model probe is worth running."""
from __future__ import annotations

from typing import Any

PREREQ_WORKSTREAMS = ("W1", "W2", "W3", "W4", "W5")

IRREDUCIBLE_BUCKETS = frozenset({
    "over_abstention",
    "over_abstain",
    "over_abstain_or_unsupported",
    "retrieval_miss",
    "evidence_absent",
    "wrong_span_retrieval",
})

INGEST_FIRST_BUCKETS = frozenset({
    "ingestion_layout_failure",
    "no_corpus",
})


def _tally(gallery: dict[str, Any], *keys: str) -> int:
    tallies = dict(gallery.get("tallies") or {})
    fine = dict(gallery.get("fine_counts") or gallery.get("fine_tallies") or {})
    for k in gallery.get("fine_buckets") or {}:
        if isinstance(gallery["fine_buckets"][k], list):
            fine.setdefault(k, len(gallery["fine_buckets"][k]))
    total = 0
    for key in keys:
        total += int(tallies.get(key) or fine.get(key) or 0)
    return total


def evaluate_admission(
    gallery: dict[str, Any] | None = None,
    *,
    eclass_lm_still_needed: bool | None = None,
    min_irreducible: int = 2,
    owner_corpus_contact: bool = False,
) -> dict[str, Any]:
    """Return admission verdict for W6 marginal LM probe."""
    gallery = gallery or {}
    irreducible = _tally(gallery, *IRREDUCIBLE_BUCKETS)
    ingest = _tally(gallery, *INGEST_FIRST_BUCKETS)
    correct_abstain = _tally(gallery, "correct_abstention", "ok_abstain")
    over_only = _tally(gallery, "over_abstention", "over_abstain", "over_abstain_or_unsupported")

    gates: dict[str, bool] = {
        "w1_w5_prereq_assumed": True,
        "ingest_not_dominant": ingest <= irreducible,
        "irreducible_abstain_present": irreducible >= 1,
        "irreducible_meets_threshold": irreducible >= min_irreducible,
        "owner_corpus_contact": owner_corpus_contact,
    }

    reasons: list[str] = []
    if eclass_lm_still_needed is False:
        gates["eclass_residual_open"] = False
        reasons.append("E-class closed by non-LM probes (T35/T36/T39 CONFIRMED)")
    elif eclass_lm_still_needed is True:
        gates["eclass_residual_open"] = True
    else:
        gates["eclass_residual_open"] = None

    if ingest > irreducible and ingest > 0:
        reasons.append(f"ingest failures ({ingest}) dominate — finish W5 first")
    if irreducible < min_irreducible:
        reasons.append(f"irreducible_abstain={irreducible} < threshold={min_irreducible}")
    if over_only == 0 and irreducible > 0:
        reasons.append("abstain mix is retrieval/evidence — classical fixes before LM")

    indicated = (
        gates["w1_w5_prereq_assumed"]
        and gates["ingest_not_dominant"]
        and gates["irreducible_meets_threshold"]
        and gates.get("eclass_residual_open") is not False
    )

    verdict = "LM_PROBE_INDICATED" if indicated else "LM_PROBE_NOT_INDICATED"

    return {
        "schema": "nano-lm.wedge_v1.lm_admission.v1",
        "workstream": "W6",
        "verdict": verdict,
        "lm_probe_indicated": indicated,
        "irreducible_abstain_count": irreducible,
        "over_abstention_count": over_only,
        "correct_abstention_count": correct_abstain,
        "ingest_failure_count": ingest,
        "gates": gates,
        "reasons": reasons,
        "prereq_workstreams": list(PREREQ_WORKSTREAMS),
        "min_irreducible": min_irreducible,
        "note": (
            "Synthetic clean track may be NOT_INDICATED even when "
            "owner-corpus over_abstention warrants a stub probe later."
        ),
    }
