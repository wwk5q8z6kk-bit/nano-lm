# NanoScribe fabric — Stage V1: typed claim/evidence packets (Master Plan Phase 1).
# The cognitive fabric's data layer. Enforces the invariants in code:
#   (3) every claim carries provenance + uncertainty fields
#   (4) contradiction is a first-class verification state
#   (7) decisions are traceable (immutable content-addressed IDs, full lineage)
#   hard rule: absence must never be inferred from lack of evidence alone
#     (an ABSENT-kind claim cannot be VERIFIED by an empty evidence set).
# V1 gates (PREREG'd in NANOSCRIBE_VNEXT/MASTER_PLAN): 100% schema validity, 100%
# source-span traceability for presented non-absence claims, no presented-error
# increase, <5% serialization failures. This module is the schema+validation half;
# the ledger/verifier wiring is the next slice artifact.
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


def _cid(payload: dict, prefix: str) -> str:
    """Immutable content-addressed ID (invariant 7: traceability)."""
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return f"{prefix}_{hashlib.sha256(raw.encode()).hexdigest()[:16]}"


class VerificationState(str, Enum):
    VERIFIED = "verified"
    CONTRADICTED = "contradicted"       # invariant 4: first-class
    UNSUPPORTED = "unsupported"
    AMBIGUOUS = "ambiguous"
    UNVERIFIABLE = "unverifiable"


class ClaimKind(str, Enum):
    VALUE = "value"          # a field carries this value (copyable from source)
    ABSENT = "absent"        # a field is absent/none — needs positive absence evidence


class DecisionAction(str, Enum):
    PRESENT = "present"
    QUALIFY = "qualify"
    ABSTAIN = "abstain"
    REVIEW = "review"


@dataclass(frozen=True)
class EvidenceSpan:
    """A provenance-bearing span in a source document."""
    source_id: str                       # e.g. dialogue id / doc id
    source_role: str                     # e.g. "patient" | "doctor" — role-aware grounding
    start: int                           # char offset in source text
    end: int
    text: str
    span_id: str = ""

    def __post_init__(self):
        if not (0 <= self.start < self.end):
            raise ValueError(f"invalid span offsets [{self.start},{self.end})")
        if not self.text:
            raise ValueError("empty span text")
        object.__setattr__(self, "span_id", _cid(
            {"s": self.source_id, "r": self.source_role, "a": self.start,
             "b": self.end, "t": self.text}, "span"))


@dataclass(frozen=True)
class Claim:
    """Atomic claim c=(subject, predicate, object, scope, source-role constraint)."""
    subject: str                         # e.g. "patient"
    predicate: str                       # e.g. "reports_cc" | "takes_med" | "allergic_to"
    obj: str                             # the value ("neck pain") or "none" for ABSENT
    kind: ClaimKind
    slot: str                            # cc|dur|sev|med|alg (slot-aware: foundation #1)
    source_constraint: str = "patient"   # which role may ground it
    uncertainty: Optional[float] = None  # invariant 3: uncertainty slot (None=unscored)
    generator: str = ""                  # which capability produced it (provenance)
    claim_id: str = ""

    def __post_init__(self):
        if self.kind == ClaimKind.ABSENT and self.obj != "none":
            raise ValueError("ABSENT claims must carry obj='none'")
        if self.kind == ClaimKind.VALUE and self.obj == "none":
            raise ValueError("VALUE claims cannot carry obj='none'")
        if not self.slot:
            raise ValueError("slot required (invariant: slot-aware verification)")
        object.__setattr__(self, "claim_id", _cid(
            {"s": self.subject, "p": self.predicate, "o": self.obj,
             "k": self.kind.value, "f": self.slot, "c": self.source_constraint}, "claim"))


@dataclass(frozen=True)
class VerificationResult:
    claim_id: str
    state: VerificationState
    evidence: tuple = ()                 # tuple[EvidenceSpan]; empty allowed only per rule below
    verifier: str = ""                   # which verifier produced it (provenance)
    note: str = ""
    result_id: str = ""

    def __post_init__(self):
        if self.state == VerificationState.VERIFIED and not self.evidence:
            raise ValueError("VERIFIED requires >=1 evidence span (traceability gate)")
        object.__setattr__(self, "result_id", _cid(
            {"c": self.claim_id, "st": self.state.value,
             "e": [e.span_id for e in self.evidence], "v": self.verifier}, "vres"))

    @staticmethod
    def verify_absence(claim: Claim, positive_absence_evidence: tuple,
                       verifier: str) -> "VerificationResult":
        """HARD RULE: ¬Found(x) ⇏ ¬x. An ABSENT claim is VERIFIED only with POSITIVE
        absence evidence (e.g. an explicit denial span); no evidence → UNVERIFIABLE."""
        if claim.kind != ClaimKind.ABSENT:
            raise ValueError("verify_absence only applies to ABSENT claims")
        if positive_absence_evidence:
            return VerificationResult(claim.claim_id, VerificationState.VERIFIED,
                                      tuple(positive_absence_evidence), verifier,
                                      "positive absence evidence")
        return VerificationResult(claim.claim_id, VerificationState.UNVERIFIABLE, (),
                                  verifier, "no positive absence evidence; "
                                  "absence NOT inferred from lack of findings")


@dataclass(frozen=True)
class Decision:
    claim_id: str
    result_id: str
    action: DecisionAction
    rationale: str
    decider: str = ""                    # capability/policy that decided (provenance)
    decision_id: str = ""

    def __post_init__(self):
        object.__setattr__(self, "decision_id", _cid(
            {"c": self.claim_id, "r": self.result_id, "a": self.action.value,
             "d": self.decider}, "dec"))

    def check_presentation_gate(self, result: VerificationResult, claim: Claim) -> None:
        """V1 gate: a PRESENTed non-absence claim must be VERIFIED with spans."""
        if self.action == DecisionAction.PRESENT:
            if result.state != VerificationState.VERIFIED:
                raise ValueError("cannot PRESENT a non-verified claim")
            if claim.kind == ClaimKind.VALUE and not result.evidence:
                raise ValueError("presented VALUE claim lacks source spans")


def to_json(obj) -> str:
    d = asdict(obj)
    for k, v in list(d.items()):
        if isinstance(v, Enum):
            d[k] = v.value
    return json.dumps(d, sort_keys=True, ensure_ascii=False)


if __name__ == "__main__":
    # self-test: schema validity, ID stability, and the invariant enforcement paths
    sp = EvidenceSpan("dlg0", "patient", 10, 19, "neck pain")
    c = Claim("patient", "reports_cc", "neck pain", ClaimKind.VALUE, "cc", generator="nano-lm")
    v = VerificationResult(c.claim_id, VerificationState.VERIFIED, (sp,), "grounding.v1")
    d = Decision(c.claim_id, v.result_id, DecisionAction.PRESENT, "grounded", "risk.v1")
    d.check_presentation_gate(v, c)
    assert c.claim_id == Claim("patient", "reports_cc", "neck pain", ClaimKind.VALUE, "cc",
                               generator="nano-lm").claim_id, "IDs must be content-stable"
    # absence rule: no positive evidence -> UNVERIFIABLE, never VERIFIED
    a = Claim("patient", "takes_med", "none", ClaimKind.ABSENT, "med")
    ra = VerificationResult.verify_absence(a, (), "absence.v1")
    assert ra.state == VerificationState.UNVERIFIABLE
    deny = EvidenceSpan("dlg0", "patient", 40, 62, "No, nothing at the moment")
    rb = VerificationResult.verify_absence(a, (deny,), "absence.v1")
    assert rb.state == VerificationState.VERIFIED
    # presentation gate must reject unverified presents
    try:
        bad = Decision(c.claim_id, ra.result_id, DecisionAction.PRESENT, "x", "risk.v1")
        bad.check_presentation_gate(ra, a)
        raise SystemExit("GATE FAILED TO FIRE")
    except ValueError:
        pass
    # serialization round-trip
    for o in (sp, c, v, d):
        json.loads(to_json(o))
    print("V1 schema self-test: PASS (IDs stable; absence rule enforced; gate fires; JSON clean)")
