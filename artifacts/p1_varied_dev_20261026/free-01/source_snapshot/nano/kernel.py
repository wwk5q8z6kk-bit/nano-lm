"""Cognitive-substrate interfaces — §36 interface-first, §43 principles.

The seven interfaces the §36 check found missing: Identity, Entity,
WorldStateProjection, WorkSlice, Tool, ArtifactIR, MemoryRecord.

These are *interfaces with enforced invariants*, not stubs. §44 forbids empty
architecture packages, so every type here either constrains something or is not
worth having. Where a type cannot yet be exercised end-to-end, its invariants
are still executable and tested.

Control/intelligence boundary (§13): everything in this module belongs to the
**deterministic substrate**. Nothing here calls a model. The substrate governs;
the learned system proposes; specialised computation calculates; verification
adjudicates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from fabric.schemas import _cid


# ---------------------------------------------------------------------------
# Identity & Authority (Layer III) — §6: semantic correctness includes identity
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Identity:
    """Who/what a piece of state belongs to, and under what authority.

    §6: a correct fact attached to the wrong subject is an INCORRECT STATE, not
    an infrastructure inconvenience. Identity is therefore checked before any
    world-state update, not at the storage layer afterwards.
    """
    subject_id: str
    kind: str                      # subject | actor | source | tenant
    authority: str = ""            # who vouches for this identity
    access_scope: str = "synthetic_non_phi"
    consent_scope: str = "research_synthetic"
    tenant: str = "default"
    identity_id: str = ""

    def __post_init__(self):
        if not self.subject_id:
            raise ValueError("identity requires a subject_id — unscoped state is a defect")
        if self.kind not in {"subject", "actor", "source", "tenant"}:
            raise ValueError(f"unknown identity kind: {self.kind}")
        object.__setattr__(self, "identity_id", _cid(
            {"s": self.subject_id, "k": self.kind, "t": self.tenant}, "id"))

    def may_join(self, other: "Identity") -> bool:
        """Cross-subject and cross-tenant joins are hard failures (§37)."""
        return self.tenant == other.tenant and self.subject_id == other.subject_id


def assert_same_subject(*identities: Identity) -> None:
    """Guard for any operation combining state from multiple sources."""
    if not identities:
        raise ValueError("no identities supplied")
    first = identities[0]
    for other in identities[1:]:
        if not first.may_join(other):
            raise ValueError(
                f"cross-subject contamination: {first.subject_id}@{first.tenant} "
                f"vs {other.subject_id}@{other.tenant}")


# ---------------------------------------------------------------------------
# Entity (Layer VI) — identity across mentions, which is a decision, not a fact
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Entity:
    """A persistent thing referenced by many mentions.

    `mention_ids` is deliberately required: an entity with no mentions was
    invented rather than observed. Resolution confidence is kept explicit
    because entity resolution is a decision that can be wrong, and the mentions
    must survive independently of it (see ontology: Mention splits from Entity).
    """
    subject: Identity
    entity_type: str
    canonical_name: str
    mention_ids: tuple
    normalized_concept: str = ""
    resolution_confidence: float | None = None
    entity_id: str = ""

    def __post_init__(self):
        if not self.mention_ids:
            raise ValueError(
                f"{self.canonical_name}: entity with no mentions was invented, "
                "not observed")
        if self.resolution_confidence is not None and not 0.0 <= self.resolution_confidence <= 1.0:
            raise ValueError("resolution_confidence must be in [0,1]")
        object.__setattr__(self, "entity_id", _cid(
            {"s": self.subject.subject_id, "t": self.entity_type,
             "n": self.canonical_name}, "ent"))


# ---------------------------------------------------------------------------
# WorldStateProjection (Layer VI) — WorldState(t), never authoritative
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WorldStateProjection:
    """A view of what appears true at a coordinate in time (§8).

    Carries the ledger version it projected from so it can be rebuilt and so a
    stale projection is detectable rather than merely suspected.
    """
    subject: Identity
    as_of: str                      # the time coordinate this projects
    ledger_version: int
    ledger_hash: str
    facts: tuple = ()
    uncertainties: tuple = ()
    projection_id: str = ""

    def __post_init__(self):
        if not self.as_of:
            raise ValueError("a projection must state the time it projects")
        object.__setattr__(self, "projection_id", _cid(
            {"s": self.subject.subject_id, "t": self.as_of,
             "v": self.ledger_version, "h": self.ledger_hash}, "proj"))


# ---------------------------------------------------------------------------
# Tool & capability binding (Layer X) — §17 capability fabric
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Tool:
    """An invocable deterministic capability that advertises its own cost.

    §17: the controller solves "smallest sufficient capability combination".
    It cannot do that unless each tool declares cost, reliability, permissions
    and side effects, so those are required rather than optional.
    """
    name: str
    inputs: tuple
    outputs: tuple
    cost: float                    # relative, unitless
    latency_s: float
    reliability: float             # [0,1]
    permissions: tuple = ()
    side_effects: tuple = ()
    emits_evidence: bool = False
    tool_id: str = ""

    def __post_init__(self):
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError(f"{self.name}: reliability must be in [0,1]")
        if self.cost < 0 or self.latency_s < 0:
            raise ValueError(f"{self.name}: negative cost/latency")
        if self.side_effects and not self.permissions:
            raise ValueError(
                f"{self.name}: a tool with side effects must declare permissions")
        object.__setattr__(self, "tool_id", _cid({"n": self.name}, "tool"))


# ---------------------------------------------------------------------------
# WorkSlice (Layer IX) — §11, §43: the fundamental unit of cognition
# ---------------------------------------------------------------------------

class SliceStatus(str, Enum):
    OPEN = "open"
    RUNNING = "running"
    VERIFIED = "verified"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class WorkSlice:
    """A unit of cognitive work: serializable, inspectable, resumable.

    §12: reasoning is state transformation under constraints with verification,
    not `prompt -> long token sequence -> answer`. The slice is what makes that
    concrete — budgets and stop conditions are required, because a cognitive
    unit that cannot stop is a loop, not a unit.
    """
    objective: str
    subject: Identity
    stop_conditions: tuple
    compute_budget: float
    assumptions: tuple = ()
    unknowns: tuple = ()
    hypotheses: tuple = ()
    constraints: tuple = ()
    evidence_requirements: tuple = ()
    candidate_actions: tuple = ()
    tools: tuple = ()
    children: list = field(default_factory=list)
    observations: list = field(default_factory=list)
    decisions: list = field(default_factory=list)
    artifacts: list = field(default_factory=list)
    spent: float = 0.0
    status: SliceStatus = SliceStatus.OPEN
    slice_id: str = ""

    def __post_init__(self):
        if not self.objective:
            raise ValueError("a work slice without an objective cannot terminate")
        if not self.stop_conditions:
            raise ValueError(
                "stop_conditions required — a cognitive unit that cannot stop "
                "is a loop, not a unit")
        if self.compute_budget <= 0:
            raise ValueError("compute_budget must be positive")
        self.slice_id = _cid({"o": self.objective,
                              "s": self.subject.subject_id}, "slice")

    def spend(self, amount: float) -> None:
        if amount < 0:
            raise ValueError("cannot spend negative compute")
        self.spent += amount
        if self.spent > self.compute_budget:
            self.status = SliceStatus.STOPPED

    @property
    def exhausted(self) -> bool:
        return self.spent >= self.compute_budget

    def spawn(self, objective: str, *, budget: float,
              stop_conditions: tuple) -> "WorkSlice":
        """A nested slice draws from the parent's remaining budget (§11)."""
        remaining = self.compute_budget - self.spent
        if budget > remaining:
            raise ValueError(
                f"child budget {budget} exceeds parent remaining {remaining}")
        child = WorkSlice(objective=objective, subject=self.subject,
                          stop_conditions=stop_conditions, compute_budget=budget)
        self.children.append(child)
        return child


# ---------------------------------------------------------------------------
# ArtifactIR (Layer XI) — §19: generation is compilation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArtifactIR:
    """The plan for an artifact, produced BEFORE any rendering (§19.1).

    Separating the semantic plan from the representation is what lets one world
    state compile into many mutually consistent artifacts rather than being
    summarised independently many times.
    """
    subject: Identity
    purpose: str
    audience: str
    required_claims: tuple
    selected_evidence: tuple
    uncertainty_disclosure: tuple = ()
    ordering: tuple = ()
    representation: str = "prose"
    format_constraints: tuple = ()
    state_version: str = ""
    ir_id: str = ""

    def __post_init__(self):
        if not self.required_claims:
            raise ValueError("an artifact with no required claims has no purpose")
        # Every claim the artifact must make has to be backed before rendering.
        if not self.selected_evidence:
            raise ValueError(
                "ArtifactIR must select evidence before rendering — verification "
                "operates before AND after compilation (§19.2)")
        object.__setattr__(self, "ir_id", _cid(
            {"s": self.subject.subject_id, "p": self.purpose,
             "a": self.audience, "c": list(self.required_claims)}, "ir"))


# ---------------------------------------------------------------------------
# MemoryRecord (Layer VIII) — §10: memory is managed, not appended
# ---------------------------------------------------------------------------

class MemoryScale(str, Enum):
    WORKING = "working"
    WORKSLICE = "workslice"
    EPISODIC = "episodic"
    PERSISTENT_STATE = "persistent_state"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    EVIDENCE_DECISION = "evidence_decision"


@dataclass(frozen=True)
class MemoryRecord:
    """A retained item at a named scale, with a path back to its source.

    §10: "compression must preserve reconstructability". A compressed episode
    that cannot point at the evidence which produced it is a lossy rewrite, so
    `derived_from` is required whenever the record is a compression.
    """
    subject: Identity
    scale: MemoryScale
    content: str
    derived_from: tuple = ()
    compressed: bool = False
    provenance: str = ""
    record_id: str = ""

    def __post_init__(self):
        if not self.content:
            raise ValueError("empty memory record")
        if self.compressed and not self.derived_from:
            raise ValueError(
                "a compressed record must retain what it was derived from — "
                "compression without reconstructability is a lossy rewrite")
        object.__setattr__(self, "record_id", _cid(
            {"s": self.subject.subject_id, "sc": self.scale.value,
             "c": self.content}, "mem"))
