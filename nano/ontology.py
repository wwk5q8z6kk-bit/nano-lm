"""Nano ontology — an EXPANDING vocabulary of primitives, not a reductive set.

The governing rule is expansion:

    A new primitive is admitted when something the system must represent does
    not fit an existing one. Nothing is forced to collapse into a smaller set.

**This registry is OPEN and illustrative, never a closed specification.** The
entries below demonstrate how the primitives integrate with what is actually
built; they are not a boundary on what Nano may represent. Adding a primitive is
ordinary work and needs no justification beyond "this did not fit". Removing one
to make the set tidier is not.

This is deliberately the opposite of a minimal-ontology discipline. The failure
mode here is not proliferation — it is **overloading**: one primitive quietly
doing two jobs, which is how a distinction gets lost and can never be recovered
downstream. `Evidence` doing the work of `Claim`, or `State` doing the work of
`Belief`, destroys information that no later stage can rebuild.

So the registry records, for each primitive:
  - what it is, and what it must NOT be conflated with
  - whether it exists in code yet
  - what caused it to be admitted or split

`splits_from` is load-bearing history: when a primitive is found to be carrying
two jobs, it splits, and the record says which one it came from. That trail is
how a future reader knows a distinction was earned rather than invented.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Presence(str, Enum):
    IN_CODE = "IN_CODE"        # a type exists in this repository
    PARTIAL = "PARTIAL"        # a field or stub exists, not the primitive
    NAMED_ONLY = "NAMED_ONLY"  # admitted to the vocabulary, not built


@dataclass(frozen=True)
class Primitive:
    name: str
    plane: str
    definition: str
    must_not_conflate_with: tuple
    presence: Presence
    implementation: str = ""
    admitted_because: str = ""
    splits_from: str = ""

    def __post_init__(self):
        if not self.must_not_conflate_with:
            raise ValueError(
                f"{self.name}: must_not_conflate_with is required — a primitive "
                "that is not distinguished from anything is not a primitive")
        if self.presence in (Presence.IN_CODE, Presence.PARTIAL) and not self.implementation:
            raise ValueError(f"{self.name}: {self.presence.value} needs an implementation pointer")
        if self.presence is Presence.NAMED_ONLY and self.implementation:
            raise ValueError(f"{self.name}: NAMED_ONLY must not cite an implementation")


def _p(**kw) -> Primitive:
    return Primitive(**kw)


# ---------------------------------------------------------------------------
# Current vocabulary. Append freely; do not collapse.
# ---------------------------------------------------------------------------

PRIMITIVES: tuple[Primitive, ...] = (

    _p(name="Observation", plane="Observation",
       definition="A raw arrival from the world through any channel, before meaning is assigned.",
       must_not_conflate_with=("Source", "Evidence"),
       presence=Presence.PARTIAL, implementation="nano/contracts.py::SourceArtifact.content",
       admitted_because="the system receives arrivals it has not yet interpreted"),

    _p(name="Source", plane="Observation",
       definition="The artifact an observation arrived in, with authorship and access class.",
       must_not_conflate_with=("Observation", "Evidence", "Author"),
       presence=Presence.IN_CODE, implementation="nano/contracts.py::SourceArtifact",
       admitted_because="who produced it is separable from what arrived"),

    _p(name="Evidence", plane="Evidence",
       definition="A located, verbatim pointer into a source. Answers 'where did this come from'.",
       must_not_conflate_with=("Claim", "Source", "Assertion"),
       presence=Presence.IN_CODE, implementation="nano/contracts.py::EvidenceSpanV2",
       admitted_because="a claim and its location are different objects"),

    _p(name="Locator", plane="Evidence",
       definition="The modality-specific address inside a source (offsets, interval, cell, region).",
       must_not_conflate_with=("Evidence",),
       presence=Presence.IN_CODE, implementation="nano/contracts.py::Locator",
       admitted_because="text offsets and audio intervals are not the same address space",
       splits_from="Evidence"),

    _p(name="Claim", plane="World model",
       definition="A proposition asserted about the world, bound to evidence.",
       must_not_conflate_with=("Evidence", "Belief", "Fact"),
       presence=Presence.IN_CODE, implementation="nano/contracts.py::ClinicalAssertion",
       admitted_because="what is asserted differs from what supports it"),

    _p(name="Entity", plane="World model",
       definition="A persistent thing with identity across observations.",
       must_not_conflate_with=("Mention", "Concept"),
       presence=Presence.NAMED_ONLY,
       admitted_because="the same medication named three ways is one entity"),

    _p(name="Mention", plane="World model",
       definition="A surface reference to an entity in one place.",
       must_not_conflate_with=("Entity",),
       presence=Presence.NAMED_ONLY,
       admitted_because="entity resolution is a decision that can be wrong; the "
                        "mention must survive independently of the resolution",
       splits_from="Entity"),

    _p(name="Relation", plane="World model",
       definition="A typed, evidence-bearing edge between entities or events.",
       must_not_conflate_with=("Event", "Causal hypothesis"),
       presence=Presence.NAMED_ONLY,
       admitted_because="'treatment then response' is not 'treatment caused response'"),

    _p(name="Event", plane="World model",
       definition="Something that happened, with temporal extent and participants.",
       must_not_conflate_with=("Claim", "Episode"),
       presence=Presence.IN_CODE, implementation="nano/contracts.py::ClinicalEvent",
       admitted_because="an assertion about an event is not the event"),

    _p(name="Episode", plane="World model",
       definition="A bounded group of related events.",
       must_not_conflate_with=("Event", "Trajectory"),
       presence=Presence.NAMED_ONLY,
       admitted_because="retrieval needs a unit larger than an event and smaller "
                        "than a lifetime",
       splits_from="Event"),

    _p(name="Trajectory", plane="World model",
       definition="An ordered series of episodes or measurements for one problem.",
       must_not_conflate_with=("Episode", "State"),
       presence=Presence.NAMED_ONLY,
       admitted_because="change over time is its own object, not a list of states",
       splits_from="Episode"),

    _p(name="Time", plane="World model",
       definition="Bitemporal extent with precision: when it happened vs when it was learned.",
       must_not_conflate_with=("Documentation time", "Record time"),
       presence=Presence.IN_CODE, implementation="nano/contracts.py::TemporalExtent",
       admitted_because="a 2024 note about a 2018 symptom must not place it in 2024"),

    _p(name="State", plane="World model",
       definition="A projection of what appears true now, rebuildable from the ledger.",
       must_not_conflate_with=("Ledger", "Belief", "Artifact"),
       presence=Presence.IN_CODE, implementation="nano/contracts.py::PatientStateSnapshot",
       admitted_because="the current picture is derived, never authoritative"),

    _p(name="Ledger", plane="Evidence",
       definition="The append-only authoritative record everything else projects from.",
       must_not_conflate_with=("State", "Memory"),
       presence=Presence.IN_CODE, implementation="nano/contracts.py::EvidenceLedger",
       admitted_because="history must survive correction"),

    _p(name="Uncertainty", plane="Metacognition",
       definition="Decomposed doubt: source, extraction, temporal, retrieval, coverage, reasoning.",
       must_not_conflate_with=("Confidence scalar", "Conflict"),
       presence=Presence.PARTIAL,
       implementation="nano/contracts.py::EpistemicStatus (provenance axis only)",
       admitted_because="one number cannot say why something is doubted"),

    _p(name="Conflict", plane="World model",
       definition="Two or more claims that cannot both hold, held open.",
       must_not_conflate_with=("Uncertainty", "Gap"),
       presence=Presence.IN_CODE, implementation="nano/contracts.py::ConflictRecord",
       admitted_because="disagreement is not the same as doubt"),

    _p(name="Gap", plane="Metacognition",
       definition="Information expected but absent, with the reason it was expected.",
       must_not_conflate_with=("Conflict", "Absence"),
       presence=Presence.IN_CODE, implementation="nano/contracts.py::KnowledgeGap",
       admitted_because="'not found' is not 'not present'"),

    _p(name="Belief", plane="Metacognition",
       definition="What the system currently holds, with the reason it holds it.",
       must_not_conflate_with=("Claim", "State"),
       presence=Presence.NAMED_ONLY,
       admitted_because="the system must be able to say why it thinks something, "
                        "separately from what it thinks"),

    _p(name="Hypothesis", plane="Cognition",
       definition="A candidate explanation under test, never reported as observed.",
       must_not_conflate_with=("Claim", "Belief", "Prediction"),
       presence=Presence.NAMED_ONLY,
       admitted_because="explanation under test must be distinguishable from finding"),

    _p(name="Goal", plane="Cognition",
       definition="What the system is currently trying to achieve.",
       must_not_conflate_with=("Task", "Constraint"),
       presence=Presence.NAMED_ONLY,
       admitted_because="retrieval and compute allocation are goal-conditioned"),

    _p(name="Constraint", plane="Cognition",
       definition="A restriction the answer must satisfy (budget, safety, schema, scope).",
       must_not_conflate_with=("Goal",),
       presence=Presence.NAMED_ONLY,
       admitted_because="what must hold is not what is wanted"),

    _p(name="Action", plane="Tool",
       definition="Something the system does that changes the world or its own state.",
       must_not_conflate_with=("Tool", "Decision"),
       presence=Presence.NAMED_ONLY,
       admitted_because="the act is separable from the instrument"),

    _p(name="Tool", plane="Tool",
       definition="An external deterministic capability invoked rather than emulated.",
       must_not_conflate_with=("Action", "Specialist model"),
       presence=Presence.NAMED_ONLY,
       admitted_because="arithmetic should be computed, not generated"),

    _p(name="Memory", plane="Memory",
       definition="Retained material at a named scale (working/episodic/semantic/procedural).",
       must_not_conflate_with=("Ledger", "State", "Context window"),
       presence=Presence.NAMED_ONLY,
       admitted_because="a context window is not memory"),

    _p(name="Decision", plane="Cognition",
       definition="A recorded choice with its alternatives and rationale.",
       must_not_conflate_with=("Action", "Artifact"),
       presence=Presence.PARTIAL, implementation="fabric/schemas.py::Decision",
       admitted_because="why something was chosen must outlive the choice"),

    _p(name="Provenance", plane="Evidence",
       definition="The chain from an output back to the sources that produced it.",
       must_not_conflate_with=("Evidence", "Citation"),
       presence=Presence.IN_CODE,
       implementation="nano/contracts.py::DerivedArtifact (version chain)",
       admitted_because="a bibliography is not claim-level traceability"),

    _p(name="Artifact", plane="Artifact",
       definition="A compiled output derived from state, carrying its input versions.",
       must_not_conflate_with=("State", "Claim"),
       presence=Presence.IN_CODE, implementation="nano/contracts.py::DerivedArtifact",
       admitted_because="the note is compiled, never the truth"),

    _p(name="Verification", plane="Verification",
       definition="A claim-level check of an artifact against the ledger.",
       must_not_conflate_with=("Confidence", "Self-critique"),
       presence=Presence.IN_CODE, implementation="nano/contracts.py::VerificationReceipt",
       admitted_because="asking the generator whether it was right is not verification"),

    _p(name="WorkSlice", plane="Cognition",
       definition="A unit of cognitive work: objective, known state, unknowns, "
                  "candidate actions, budget, stop conditions.",
       must_not_conflate_with=("Goal", "Action", "Plan"),
       presence=Presence.NAMED_ONLY,
       admitted_because="adaptive compute needs an addressable unit to budget"),

    _p(name="StateDelta", plane="World model",
       definition="The change between two state projections, with its evidence.",
       must_not_conflate_with=("State", "Event"),
       presence=Presence.IN_CODE, implementation="nano/contracts.py::StateDelta",
       admitted_because="what changed is the longitudinal question; a new snapshot "
                        "does not answer it",
       splits_from="State"),

    _p(name="Supersession", plane="World model",
       definition="A fact replaced by a correction, as opposed to one that stopped being true.",
       must_not_conflate_with=("Removal", "StateDelta"),
       presence=Presence.IN_CODE,
       implementation="nano/contracts.py::StateDelta.superseded",
       admitted_because="a corrected date and a discontinued medication both "
                        "disappear from state; collapsing them loses the reason",
       splits_from="StateDelta"),

    _p(name="Identity", plane="Identity & Authority",
       definition="The subject/actor/source a piece of information belongs to.",
       must_not_conflate_with=("Entity", "Provenance"),
       presence=Presence.PARTIAL,
       implementation="nano/contracts.py (patient_id required on every object)",
       admitted_because="a correct fact attached to the wrong subject is a "
                        "semantic failure, not an infrastructure one"),

    _p(name="Dependency", plane="Dependency & Invalidation",
       definition="The lineage edge from a derived object back to what produced it.",
       must_not_conflate_with=("Provenance", "Relation"),
       presence=Presence.IN_CODE, implementation="nano/dependency.py::Dependency",
       admitted_because="when evidence is corrected, everything downstream must "
                        "become inspectably stale rather than silently wrong"),

    _p(name="Staleness", plane="Dependency & Invalidation",
       definition="A derived object whose inputs changed after it was produced.",
       must_not_conflate_with=("Supersession", "Uncertainty"),
       presence=Presence.IN_CODE, implementation="nano/dependency.py::Freshness",
       admitted_because="an artifact can be internally correct and still out of date",
       splits_from="Dependency"),

    _p(name="DerivationMode", plane="Metacognition",
       definition="How a statement came to be: observed, derived, inferred, "
                  "hypothesised, predicted, simulated.",
       must_not_conflate_with=("EpistemicStatus",),
       presence=Presence.IN_CODE, implementation="nano/contracts.py::DerivationMode",
       admitted_because="epistemic status says who reported it; derivation mode "
                        "says how the system produced it — orthogonal axes that "
                        "were being carried by one enum",
       splits_from="Uncertainty"),
)


PLANES = ("Identity & Authority", "Observation", "Evidence", "World model",
          "Memory", "Cognition", "Tool", "Artifact", "Verification",
          "Metacognition", "Dependency & Invalidation")


def by_presence() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {p.value: [] for p in Presence}
    for x in PRIMITIVES:
        out[x.presence.value].append(x.name)
    return out


def overloading_candidates() -> list[tuple[str, str]]:
    """Primitives whose implementation pointer is shared with another primitive.

    A shared pointer means one type is carrying two jobs — the expansion
    discipline's failure mode. Reported, not auto-resolved: splitting is a
    design decision.
    """
    seen: dict[str, str] = {}
    out = []
    for x in PRIMITIVES:
        if not x.implementation:
            continue
        key = x.implementation.split("::")[-1]
        if key in seen and seen[key] != x.name:
            out.append((seen[key], x.name))
        else:
            seen[key] = x.name
    return out


def summary() -> dict:
    return {
        "total": len(PRIMITIVES),
        "by_presence": {k: len(v) for k, v in by_presence().items()},
        "planes": len({x.plane for x in PRIMITIVES}),
        "splits_recorded": [x.name for x in PRIMITIVES if x.splits_from],
        "overloading": overloading_candidates(),
    }
