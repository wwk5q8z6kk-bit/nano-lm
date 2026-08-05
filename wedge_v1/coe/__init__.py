"""Chain-of-Evidence support for the internal Wedge v1 validation pipeline.

Invariant: EVIDENCE MUST BE CREATED WITH THE CLAIM, NOT RECONSTRUCTED AFTER.
Not a Science One clone; no autonomous paper agents.
"""
from wedge_v1.coe.schema import (
    DerivationKind,
    EvidenceAtom,
    EvidenceRelation,
    TypedClaim,
    VerificationRecord,
)
from wedge_v1.coe.record import EvidenceRecord
from wedge_v1.coe.bind import bind_ask_payload
from wedge_v1.coe.audit import audit_payload, audit_record
from wedge_v1.coe.replay import replay_ask

__all__ = [
    "DerivationKind",
    "EvidenceAtom",
    "EvidenceRelation",
    "TypedClaim",
    "VerificationRecord",
    "EvidenceRecord",
    "bind_ask_payload",
    "audit_payload",
    "audit_record",
    "replay_ask",
]
