"""Typed failure codes for Nano Runtime / Active Frontier (product architecture).

Not Layer-1 Evidence Core. Codes are stable strings for traces, gallery, and tests.
"""
from __future__ import annotations

from enum import Enum


class FailureCode(str, Enum):
    # Ingest / identity
    NO_CORPUS = "NO_CORPUS"
    INGESTION_LAYOUT_FAILURE = "INGESTION_LAYOUT_FAILURE"
    # Retrieval
    RETRIEVAL_MISS = "RETRIEVAL_MISS"
    WRONG_SPAN_RETRIEVAL = "WRONG_SPAN_RETRIEVAL"
    LOW_MARGIN_RETRIEVAL = "LOW_MARGIN_RETRIEVAL"
    # Evidence / verify
    EVIDENCE_ABSENT = "EVIDENCE_ABSENT"
    EMPTY_EVIDENCE_REJECTED = "EMPTY_EVIDENCE_REJECTED"
    VERIFIER_REJECTION = "VERIFIER_REJECTION"
    # Abstention
    CORRECT_ABSTENTION = "CORRECT_ABSTENTION"
    OVER_ABSTENTION = "OVER_ABSTENTION"
    UNSAFE_ANSWER_BLOCKED = "UNSAFE_ANSWER_BLOCKED"
    # Composition / entities / contradictions
    UNSUPPORTED_COMPOSITION = "UNSUPPORTED_COMPOSITION"
    ENTITY_TYPE_COLLISION = "ENTITY_TYPE_COLLISION"
    MULTI_DOC_CONTRADICTION = "MULTI_DOC_CONTRADICTION"
    NUMERIC_CONTRADICTION = "NUMERIC_CONTRADICTION"
    TEMPORAL_CONTRADICTION = "TEMPORAL_CONTRADICTION"
    # Product / routing
    RULE_BRITTLENESS = "RULE_BRITTLENESS"
    FIXTURE_TIED_SOLVER = "FIXTURE_TIED_SOLVER"
    HIGH_REVIEW_BURDEN = "HIGH_REVIEW_BURDEN"
    # Chain-of-Evidence (Science One principles adapted; local verified intelligence)
    COE_MISSING_SOURCE = "COE_MISSING_SOURCE"
    COE_STALE_SOURCE_VERSION = "COE_STALE_SOURCE_VERSION"
    COE_INVALID_OFFSET = "COE_INVALID_OFFSET"
    COE_UNSUPPORTED_PREDICATE = "COE_UNSUPPORTED_PREDICATE"
    COE_INCOMPLETE_CONJUNCTION = "COE_INCOMPLETE_CONJUNCTION"
    COE_METHOD_TRACE_MISMATCH = "COE_METHOD_TRACE_MISMATCH"
    COE_NONREPRODUCIBLE_RESULT = "COE_NONREPRODUCIBLE_RESULT"
    COE_POSTHOC_CITATION = "COE_POSTHOC_CITATION"
    COE_UNREGISTERED_SOLVER = "COE_UNREGISTERED_SOLVER"
    COE_CONFIG_MISSING = "COE_CONFIG_MISSING"
    COE_CONTRADICTION_IGNORED = "COE_CONTRADICTION_IGNORED"
    COE_DERIVATION_UNKNOWN = "COE_DERIVATION_UNKNOWN"
    COE_USER_CORRECTION_NOT_PROPAGATED = "COE_USER_CORRECTION_NOT_PROPAGATED"
    # Catch-all
    OTHER = "OTHER"


# Historical → code (documentation for registry; not auto-fired)
HISTORICAL_LESSONS: dict[str, list[FailureCode]] = {
    "open_vocab_emission_gap": [FailureCode.OVER_ABSTENTION, FailureCode.WRONG_SPAN_RETRIEVAL],
    "exact_presence_not_source_conditioned": [FailureCode.WRONG_SPAN_RETRIEVAL],
    "scale_confounded_with_pretrain": [FailureCode.OTHER],
    "exact_match_not_faithfulness": [FailureCode.VERIFIER_REJECTION, FailureCode.WRONG_SPAN_RETRIEVAL],
    "classical_beats_gen_closed_task": [FailureCode.OTHER],
    "single_instance_conceals_failure": [FailureCode.OTHER],
    "post_rationalized_citations": [FailureCode.WRONG_SPAN_RETRIEVAL, FailureCode.UNSUPPORTED_COMPOSITION],
    "generator_isomorphic_benchmark": [FailureCode.FIXTURE_TIED_SOLVER],
}


FINE_BUCKET_TO_CODE: dict[str, FailureCode] = {
    "evidence_absent": FailureCode.EVIDENCE_ABSENT,
    "retrieval_miss": FailureCode.RETRIEVAL_MISS,
    "wrong_span_retrieval": FailureCode.WRONG_SPAN_RETRIEVAL,
    "verifier_rejection": FailureCode.VERIFIER_REJECTION,
    "correct_abstention": FailureCode.CORRECT_ABSTENTION,
    "over_abstention": FailureCode.OVER_ABSTENTION,
    "entity_type_collision": FailureCode.ENTITY_TYPE_COLLISION,
    "multi_document_contradiction": FailureCode.MULTI_DOC_CONTRADICTION,
    "unsupported_composition": FailureCode.UNSUPPORTED_COMPOSITION,
    "ingestion_layout_failure": FailureCode.INGESTION_LAYOUT_FAILURE,
    "low_margin_review": FailureCode.LOW_MARGIN_RETRIEVAL,
}
