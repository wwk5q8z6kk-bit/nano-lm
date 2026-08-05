"""Structured ask/find/compare traces for observability and failure localization."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from wedge_v1.arch.failure_codes import FailureCode


@dataclass
class TraceEvent:
    stage: str
    detail: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"stage": self.stage, "detail": self.detail, "meta": self.meta}


@dataclass
class AskTrace:
    """Replayable decision trace attached to ask()/find()/compare() payloads."""

    query: str
    corpus_dir: str
    op: str = "ask"
    events: list[TraceEvent] = field(default_factory=list)
    solvers: list[str] = field(default_factory=list)
    failure_codes: list[FailureCode] = field(default_factory=list)
    n_docs: int = 0
    n_claims_raw: int = 0
    n_claims_presented: int = 0
    n_empty_evidence_rejected: int = 0
    n_bm25_review: int = 0
    answer_status: str = ""
    t0: float = field(default_factory=time.perf_counter)

    def event(self, stage: str, detail: str = "", **meta: Any) -> None:
        self.events.append(TraceEvent(stage=stage, detail=detail, meta=meta))

    def add_solver(self, name: str) -> None:
        self.solvers.append(name)

    def add_failure(self, code: FailureCode) -> None:
        if code not in self.failure_codes:
            self.failure_codes.append(code)

    def finalize(self, answer_status: str) -> dict:
        self.answer_status = answer_status
        latency_ms = int(round((time.perf_counter() - self.t0) * 1000))
        return {
            "schema": "nano-lm.wedge_v1.ask_trace.v1",
            "op": self.op,
            "query": self.query,
            "corpus_dir": self.corpus_dir,
            "answer_status": answer_status,
            "n_docs": self.n_docs,
            "n_claims_raw": self.n_claims_raw,
            "n_claims_presented": self.n_claims_presented,
            "n_empty_evidence_rejected": self.n_empty_evidence_rejected,
            "n_bm25_review": self.n_bm25_review,
            "solvers": list(self.solvers),
            "failure_codes": [c.value for c in self.failure_codes],
            "events": [e.to_dict() for e in self.events],
            "latency_ms": latency_ms,
            "layers_touched": _infer_layers(self.solvers, self.failure_codes),
        }


def _infer_layers(solvers: list[str], codes: list[FailureCode]) -> list[str]:
    layers = {"L1", "L3"}
    blob = " ".join(solvers).lower()
    if "bm25" in blob:
        layers.add("L4")
    if "eclass" in blob or "keyword" in blob or "numeric" in blob:
        layers.add("L5")
    if any(c in codes for c in (FailureCode.EMPTY_EVIDENCE_REJECTED, FailureCode.VERIFIER_REJECTION)):
        layers.update({"L7", "L8"})
    if any(
        c in codes
        for c in (
            FailureCode.MULTI_DOC_CONTRADICTION,
            FailureCode.NUMERIC_CONTRADICTION,
            FailureCode.UNSUPPORTED_COMPOSITION,
        )
    ):
        layers.add("L8")
    if any(
        c in codes
        for c in (
            FailureCode.CORRECT_ABSTENTION,
            FailureCode.OVER_ABSTENTION,
            FailureCode.LOW_MARGIN_RETRIEVAL,
        )
    ):
        layers.add("L9")
    layers.add("L10")
    return sorted(layers)


def classify_abstain_failures(
    *,
    has_lexical_hit: bool,
    bm25_review_n: int,
    empty_rejected: int,
    oos_expected: bool,
    composition_blocked: bool,
) -> list[FailureCode]:
    codes: list[FailureCode] = []
    if composition_blocked:
        codes.append(FailureCode.UNSUPPORTED_COMPOSITION)
        return codes
    if oos_expected or (not has_lexical_hit and bm25_review_n == 0):
        codes.append(FailureCode.CORRECT_ABSTENTION)
        if not has_lexical_hit:
            codes.append(FailureCode.EVIDENCE_ABSENT)
        return codes
    if bm25_review_n > 0 and not has_lexical_hit:
        codes.append(FailureCode.LOW_MARGIN_RETRIEVAL)
        codes.append(FailureCode.RETRIEVAL_MISS)
        return codes
    if has_lexical_hit and empty_rejected > 0:
        codes.append(FailureCode.EMPTY_EVIDENCE_REJECTED)
    if has_lexical_hit:
        # Evidence somewhere in corpus but cascade did not present — over-abstention risk
        codes.append(FailureCode.OVER_ABSTENTION)
    else:
        codes.append(FailureCode.RETRIEVAL_MISS)
    return codes
