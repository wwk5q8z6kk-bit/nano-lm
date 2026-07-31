"""Typed Chain-of-Evidence claim / evidence / derivation schema."""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class DerivationKind(str, Enum):
    EXACT_QUOTE = "EXACT_QUOTE"
    DETERMINISTIC_NORMALIZATION = "DETERMINISTIC_NORMALIZATION"
    SCHEMA_PROJECTION = "SCHEMA_PROJECTION"
    NUMERIC_COMPARISON = "NUMERIC_COMPARISON"
    SYMBOLIC_MERGE = "SYMBOLIC_MERGE"
    BM25_PARAGRAPH = "BM25_PARAGRAPH"
    KEYWORD_LOCATE = "KEYWORD_LOCATE"
    UNKNOWN = "UNKNOWN"


class EvidenceRelation(str, Enum):
    EXACTLY_STATED = "EXACTLY_STATED"
    NORMALIZED_EQUIVALENT = "NORMALIZED_EQUIVALENT"
    NUMERICALLY_DERIVED = "NUMERICALLY_DERIVED"
    STRUCTURALLY_INFERRED = "STRUCTURALLY_INFERRED"
    TEMPORALLY_SUPERSEDED = "TEMPORALLY_SUPERSEDED"
    CONTRADICTED = "CONTRADICTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass
class EvidenceAtom:
    atom_id: str
    doc_id: str | None
    doc_digest: str | None
    start: int | None
    end: int | None
    text: str
    relation: EvidenceRelation = EvidenceRelation.EXACTLY_STATED

    def to_dict(self) -> dict:
        d = asdict(self)
        d["relation"] = self.relation.value
        return d


@dataclass
class VerificationRecord:
    verifier_id: str
    independent: bool
    outcome: str  # pass | fail | abstain
    checks: list[str] = field(default_factory=list)
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TypedClaim:
    claim_id: str
    claim_type: str
    proposition: str
    status: str
    derivation: DerivationKind
    solver_id: str
    solver_version: str
    source_doc_ids: list[str]
    corpus_digest: str | None
    evidence_atoms: list[EvidenceAtom]
    verification: list[VerificationRecord]
    contradiction_ids: list[str] = field(default_factory=list)
    confidence: float | None = None
    abstention_state: str | None = None
    execution_event_ids: list[str] = field(default_factory=list)
    reproducibility: dict[str, Any] = field(default_factory=dict)
    raw_value: Any = None

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "claim_type": self.claim_type,
            "proposition": self.proposition,
            "status": self.status,
            "derivation": self.derivation.value,
            "solver_id": self.solver_id,
            "solver_version": self.solver_version,
            "source_doc_ids": self.source_doc_ids,
            "corpus_digest": self.corpus_digest,
            "evidence_atoms": [a.to_dict() for a in self.evidence_atoms],
            "verification": [v.to_dict() for v in self.verification],
            "contradiction_ids": self.contradiction_ids,
            "confidence": self.confidence,
            "abstention_state": self.abstention_state,
            "execution_event_ids": self.execution_event_ids,
            "reproducibility": self.reproducibility,
            "raw_value": self.raw_value,
        }


SOLVER_VERSION = "wedge_v1.coe.v1"


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def digest_docs(docs: dict[str, str]) -> str:
    h = hashlib.sha256()
    for k in sorted(docs):
        h.update(k.encode())
        h.update(bytes([0]))
        h.update(docs[k].encode("utf-8", errors="replace"))
        h.update(bytes([0]))
    return h.hexdigest()[:16]


def infer_derivation(task_id: str, notes: str) -> DerivationKind:
    n = (notes or "").lower()
    tid = (task_id or "").upper()
    if "bm25" in n or tid == "T19":
        return DerivationKind.BM25_PARAGRAPH
    if "symbolic" in n or tid in {"T36", "T29", "T30"}:
        return DerivationKind.SYMBOLIC_MERGE
    if "numeric" in n or tid == "FIND":
        return DerivationKind.KEYWORD_LOCATE
    if "exact" in n or "quote" in n:
        return DerivationKind.EXACT_QUOTE
    if tid.startswith("T0") or tid in {"T01", "T02", "T03"}:
        return DerivationKind.SCHEMA_PROJECTION
    if "compare" in n or "union" in n:
        return DerivationKind.NUMERIC_COMPARISON
    return DerivationKind.DETERMINISTIC_NORMALIZATION
