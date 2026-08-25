"""Nano Core contracts — source, evidence, assertion, event, state, verification.

Extends the conventions already established in `fabric/schemas.py` (frozen
dataclasses, content-addressed ids via `_cid`, validation in `__post_init__`)
rather than introducing a second schema system — see D-NANO-2026-08-25 §5,
"Do not create duplicate schema systems."

Relationship to fabric:
    fabric.EvidenceSpan   -> EvidenceSpanV2 (adds modality locators, bitemporal
                             times, content hash, access class, extraction ver)
    fabric.Claim          -> ClinicalAssertion (adds original wording, normalized
                             concept, epistemic status, temporal extent)
    fabric.VerificationResult is reused as-is inside VerificationReceipt.

Architectural law (§3): the evidence ledger is authoritative; PatientStateSnapshot
is a rebuildable *projection* over it. Nothing here mutates history.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional

from fabric.schemas import _cid  # reuse the existing content-addressed id helper


# --------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------

class Modality(str, Enum):
    TEXT = "text"
    AUDIO = "audio"
    TABLE = "table"
    IMAGE = "image"
    SIGNAL = "signal"


class EpistemicStatus(str, Enum):
    """§5: confidence must be decomposed. Never collapse these to one number."""
    DIRECT_MEASUREMENT = "direct_measurement"
    DIRECT_OBSERVATION = "direct_observation"
    DIRECT_DOCUMENTATION = "direct_documentation"
    PATIENT_REPORTED = "patient_reported"
    CAREGIVER_REPORTED = "caregiver_reported"
    CLINICIAN_ASSERTED = "clinician_asserted"
    INFERRED = "inferred"
    RECONSTRUCTED = "reconstructed"
    CONFLICTING = "conflicting"
    UNCERTAIN = "uncertain"
    NOT_FOUND = "not_found"
    UNAVAILABLE = "unavailable"
    OUTDATED = "outdated"
    SUPERSEDED = "superseded"


#: Statuses that may never be rendered as though directly documented (§16).
_INFERRED_STATUSES = frozenset({
    EpistemicStatus.INFERRED,
    EpistemicStatus.RECONSTRUCTED,
    EpistemicStatus.UNCERTAIN,
    EpistemicStatus.CONFLICTING,
})

#: Statuses that assert absence of information, distinct from absence of fact.
_ABSENCE_STATUSES = frozenset({
    EpistemicStatus.NOT_FOUND,
    EpistemicStatus.UNAVAILABLE,
})


class TimePrecision(str, Enum):
    """§6: a year, a month and an exact timestamp are not equivalent."""
    EXACT = "exact"
    DAY = "day"
    MONTH = "month"
    YEAR = "year"
    APPROXIMATE = "approximate"
    RELATIVE = "relative"
    UNKNOWN = "unknown"


class ConflictType(str, Enum):
    DATE_DISAGREEMENT = "date_disagreement"
    PRESENCE_DISAGREEMENT = "presence_disagreement"
    VALUE_DISAGREEMENT = "value_disagreement"


class GapKind(str, Enum):
    NOT_FOUND = "not_found"              # searched, absent from the record
    UNAVAILABLE = "unavailable"          # known to exist, not accessible
    NEVER_PERFORMED = "never_performed"  # positively documented as not done
    AMBIGUOUS = "ambiguous"


# --------------------------------------------------------------------------
# Source and evidence
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceArtifact:
    """The immutable input. Never rewritten; corrections arrive as new artifacts."""
    patient_id: str
    modality: Modality
    document_type: str
    content: str
    author_or_device: str = ""
    encounter_id: str = ""
    organization_id: str = ""
    source_system: str = ""
    creation_time: str = ""      # when the source was authored
    received_time: str = ""      # when this system learned of it (system time)
    security_classification: str = "synthetic_non_phi"
    consent_scope: str = "research_synthetic"
    content_hash: str = ""
    source_id: str = ""

    def __post_init__(self):
        if not self.patient_id:
            raise ValueError("patient_id required (no unscoped evidence)")
        if not self.content:
            raise ValueError("empty source content")
        object.__setattr__(self, "content_hash",
                           hashlib.sha256(self.content.encode()).hexdigest())
        object.__setattr__(self, "source_id", _cid(
            {"p": self.patient_id, "m": self.modality.value,
             "d": self.document_type, "h": self.content_hash}, "src"))


@dataclass(frozen=True)
class Locator:
    """Modality-specific pointer into a source. Exactly one family must be set."""
    # text
    start: Optional[int] = None
    end: Optional[int] = None
    line: Optional[int] = None
    section: str = ""
    # audio / signal
    t_start: Optional[float] = None
    t_end: Optional[float] = None
    channel: str = ""
    # table
    table: str = ""
    row: Optional[int] = None
    column: str = ""
    # image
    image_id: str = ""
    bbox: Optional[tuple] = None

    def families(self) -> list[str]:
        fam = []
        if self.start is not None and self.end is not None:
            fam.append("text")
        if self.t_start is not None and self.t_end is not None:
            fam.append("interval")
        if self.row is not None:
            fam.append("table")
        if self.bbox is not None:
            fam.append("image")
        return fam

    def __post_init__(self):
        fam = self.families()
        if len(fam) != 1:
            raise ValueError(f"locator must set exactly one family, got {fam}")
        if "text" in fam and not (0 <= self.start < self.end):
            raise ValueError(f"invalid text offsets [{self.start},{self.end})")
        if "interval" in fam and not (0 <= self.t_start < self.t_end):
            raise ValueError(f"invalid interval [{self.t_start},{self.t_end})")


@dataclass(frozen=True)
class EvidenceSpanV2:
    """Modality-independent evidence locator. Successor to fabric.EvidenceSpan."""
    source_id: str
    patient_id: str
    modality: Modality
    locator: Locator
    verbatim: str
    speaker: str = ""
    encounter_id: str = ""
    section: str = ""
    documentation_time: str = ""   # when it was recorded
    candidate_event_time: str = ""  # when the described thing may have occurred
    access_label: str = "synthetic_non_phi"
    extraction_version: str = "nano-clin-001"
    content_hash: str = ""
    evidence_span_id: str = ""

    def __post_init__(self):
        if not self.verbatim:
            raise ValueError("evidence span must carry verbatim content")
        if not self.patient_id:
            raise ValueError("evidence span must be patient-scoped")
        object.__setattr__(self, "content_hash",
                           hashlib.sha256(self.verbatim.encode()).hexdigest())
        object.__setattr__(self, "evidence_span_id", _cid(
            {"s": self.source_id, "l": asdict(self.locator),
             "h": self.content_hash, "k": self.speaker}, "ev"))


# --------------------------------------------------------------------------
# Time
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TemporalExtent:
    """Bitemporal: when it happened vs when the system learned it (§6)."""
    event_time: str = ""
    onset_time: str = ""
    discovery_time: str = ""
    documentation_time: str = ""
    start_time: str = ""
    end_time: str = ""
    duration: str = ""
    relative_time: str = ""
    precision: TimePrecision = TimePrecision.UNKNOWN
    uncertainty: str = ""
    system_recorded_time: str = ""

    def __post_init__(self):
        # §6: do not invent exact dates from approximate language.
        if self.precision == TimePrecision.EXACT and self.relative_time:
            raise ValueError("EXACT precision cannot be derived from relative_time")
        if self.precision in (TimePrecision.APPROXIMATE, TimePrecision.RELATIVE) \
                and len(self.event_time) == 10 and self.event_time.count("-") == 2:
            raise ValueError(
                f"precision={self.precision.value} but event_time={self.event_time!r} "
                "is a full date — refusing to manufacture precision")


# --------------------------------------------------------------------------
# Assertions, events, conflicts, gaps
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ClinicalAssertion:
    """Successor to fabric.Claim: keeps original wording AND normalized concept."""
    patient_id: str
    subject: str
    predicate: str
    obj: str
    original_wording: str
    epistemic_status: EpistemicStatus
    evidence_span_ids: tuple = ()
    normalized_concept: str = ""
    negated: bool = False
    temporal: TemporalExtent = field(default_factory=TemporalExtent)
    author: str = ""
    extractor: str = "nano-clin-001"
    original_units: str = ""
    normalized_units: str = ""
    assertion_id: str = ""

    def __post_init__(self):
        if not self.original_wording:
            raise ValueError("original wording must be preserved")
        # An assertion that claims support must point at evidence. Absence
        # statuses are the sole exception: they assert that nothing was found.
        if not self.evidence_span_ids and self.epistemic_status not in _ABSENCE_STATUSES:
            raise ValueError(
                f"assertion with status {self.epistemic_status.value} needs evidence "
                "(absence-never-from-silence)")
        object.__setattr__(self, "assertion_id", _cid(
            {"p": self.patient_id, "s": self.subject, "pr": self.predicate,
             "o": self.obj, "e": self.epistemic_status.value,
             "n": self.negated, "ev": list(self.evidence_span_ids)}, "asrt"))

    @property
    def is_inferred(self) -> bool:
        return self.epistemic_status in _INFERRED_STATUSES


@dataclass(frozen=True)
class ClinicalEvent:
    patient_id: str
    event_type: str
    temporal: TemporalExtent
    assertion_ids: tuple = ()
    participants: tuple = ()
    entities: tuple = ()
    encounter_id: str = ""
    status: str = "recorded"
    event_id: str = ""

    def __post_init__(self):
        if not self.assertion_ids:
            raise ValueError("event must derive from at least one assertion")
        object.__setattr__(self, "event_id", _cid(
            {"p": self.patient_id, "t": self.event_type,
             "a": list(self.assertion_ids)}, "evt"))


@dataclass(frozen=True)
class ConflictRecord:
    """§16: no silent conflict resolution. Conflicts stay inspectable."""
    patient_id: str
    conflict_type: ConflictType
    claim_set: tuple
    supporting_evidence: tuple = ()
    contradictory_evidence: tuple = ()
    clinical_importance: str = "unknown"
    resolution_status: str = "unresolved"
    human_disposition: str = ""
    conflict_id: str = ""

    def __post_init__(self):
        if len(self.claim_set) < 2:
            raise ValueError("a conflict needs at least two claims")
        if self.resolution_status == "resolved" and not self.human_disposition:
            raise ValueError("conflicts may only be resolved by a human disposition")
        object.__setattr__(self, "conflict_id", _cid(
            {"p": self.patient_id, "t": self.conflict_type.value,
             "c": sorted(self.claim_set)}, "cfl"))


@dataclass(frozen=True)
class KnowledgeGap:
    """'Not mentioned' is not 'absent'. This type exists to keep them apart."""
    patient_id: str
    expected_information: str
    kind: GapKind
    why_expected: str = ""
    search_scope: str = ""
    clinical_importance: str = "unknown"
    gap_id: str = ""

    def __post_init__(self):
        object.__setattr__(self, "gap_id", _cid(
            {"p": self.patient_id, "e": self.expected_information,
             "k": self.kind.value}, "gap"))


# --------------------------------------------------------------------------
# Ledger and state projection
# --------------------------------------------------------------------------

@dataclass
class EvidenceLedger:
    """Append-only. L_{v+1} = L_v (+) e_{v+1}. History is never mutated (§7)."""
    patient_id: str
    sources: list = field(default_factory=list)
    spans: list = field(default_factory=list)
    assertions: list = field(default_factory=list)
    events: list = field(default_factory=list)
    conflicts: list = field(default_factory=list)
    gaps: list = field(default_factory=list)
    version: int = 0

    def append(self, **items) -> "EvidenceLedger":
        for key, values in items.items():
            bucket = getattr(self, key)
            for v in values:
                if v not in bucket:
                    bucket.append(v)
        self.version += 1
        return self

    def ledger_hash(self) -> str:
        payload = {
            "sources": [s.source_id for s in self.sources],
            "spans": [s.evidence_span_id for s in self.spans],
            "assertions": [a.assertion_id for a in self.assertions],
            "events": [e.event_id for e in self.events],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class PatientStateSnapshot:
    """A projection: S_v = Pi(L_v). Rebuildable, never authoritative."""
    patient_id: str
    evidence_ledger_version: int
    ledger_hash: str
    active_conditions: tuple = ()
    current_medications: tuple = ()
    laboratory_state: tuple = ()
    functional_state: tuple = ()
    uncertainties: tuple = ()
    conflicts: tuple = ()
    unresolved_questions: tuple = ()
    projection_version: str = "nano-clin-001"
    snapshot_id: str = ""

    def __post_init__(self):
        object.__setattr__(self, "snapshot_id", _cid(
            {"p": self.patient_id, "v": self.evidence_ledger_version,
             "h": self.ledger_hash, "pv": self.projection_version}, "state"))


# --------------------------------------------------------------------------
# Derived artifacts and verification
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DerivedArtifact:
    patient_id: str
    artifact_type: str
    task: str
    content: str
    patient_state_version: str
    evidence_ledger_version: int
    supporting_evidence: tuple = ()
    generation_method: str = ""
    model_version: str = "nano-clin-001"
    verification_status: str = "unverified"
    freshness_status: str = "current"
    artifact_id: str = ""

    def __post_init__(self):
        object.__setattr__(self, "artifact_id", _cid(
            {"p": self.patient_id, "t": self.artifact_type,
             "s": self.patient_state_version, "c": self.content}, "art"))


@dataclass(frozen=True)
class VerificationReceipt:
    """Claim-level, not artifact-level. Every factual sentence is checked."""
    artifact_id: str
    claim_results: tuple = ()      # tuple[dict]
    coverage_status: str = ""
    verifier_version: str = "nano-clin-001"
    receipt_id: str = ""

    @property
    def unsupported_count(self) -> int:
        return sum(1 for r in self.claim_results if not r.get("supported"))

    @property
    def provenance_coverage(self) -> float:
        if not self.claim_results:
            return 0.0
        cited = sum(1 for r in self.claim_results if r.get("evidence_span_ids"))
        return cited / len(self.claim_results)

    def __post_init__(self):
        object.__setattr__(self, "receipt_id", _cid(
            {"a": self.artifact_id, "n": len(self.claim_results)}, "vrfy"))
