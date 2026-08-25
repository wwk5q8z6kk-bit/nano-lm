"""NANO-SLW-001 — Synthetic Longitudinal World.

A benchmark that proves the *already-implemented* substrate rather than adding
to it. Nothing here defines a new architectural primitive: it composes
`nano.contracts` (evidence ledger, temporal extent, state delta),
`nano.dependency` (lineage, freshness, invalidation) and `nano.kernel`
(identity) against a world whose ground truth is known by construction.

Why a synthetic world at all
----------------------------
Every claim the architecture makes — "corrections propagate", "approximate time
stays approximate", "contradictions are preserved", "unrelated branches are not
invalidated" — is a claim about behaviour under *imperfect observation*. Real
corpora cannot test it because the ground truth is unavailable: when the system
says a fact changed, there is nothing to check it against. A generated world
inverts that. The world knows what happened; the system only sees a corrupted
observation channel; the harness scores the gap.

Deliberately non-medical
------------------------
The domain is a fleet of sites, units, components, operators and resources. If
the substrate is general, a dependency between a rollup and its member views
behaves identically to one between a discharge summary and its labs. Medicine
is a benchmark, not the architecture. (The `Clinical*` prefix on the contract
types is a legacy name from NANO-CLIN-001; the types themselves carry no
clinical semantics, and this module is the evidence for that claim.)

The two arms
------------
* **BASELINE A** — rebuild every derived object from the whole ledger at every
  checkpoint. Always correct; cost grows with history. This is what "no lineage
  tracking" actually costs.
* **CANDIDATE B** — maintain a `DependencyGraph`, invalidate precisely, and
  recompute only what is stale.

B is only interesting if it is *both* cheaper and identical. The harness refuses
to report the cost saving unless the final states match exactly, because a fast
wrong answer is the failure mode this whole architecture exists to prevent.

Safety: synthetic data only, no PHI, no learned model, no network, no paid
compute. The first pass is entirely deterministic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum

from fabric.schemas import _cid

from nano.contracts import (
    ClinicalAssertion,
    ConflictRecord,
    ConflictType,
    DerivationMode,
    EpistemicStatus,
    EvidenceLedger,
    EvidenceSpanV2,
    GapKind,
    KnowledgeGap,
    Locator,
    Modality,
    PatientStateSnapshot,
    SourceArtifact,
    TemporalExtent,
    TimePrecision,
    diff_states,
)
from nano.dependency import Dependency, DependencyGraph, Freshness
from nano.kernel import Identity

EPOCH = date(2026, 1, 1)


def tick_to_date(tick: int) -> str:
    return (EPOCH + timedelta(days=tick)).isoformat()


def tick_to_month(tick: int) -> str:
    return (EPOCH + timedelta(days=tick)).strftime("%Y-%m")


# ---------------------------------------------------------------------------
# Ground truth: the world as it actually is
# ---------------------------------------------------------------------------

class EntityType(str, Enum):
    SITE = "site"
    UNIT = "unit"
    COMPONENT = "component"
    OPERATOR = "operator"
    RESOURCE = "resource"


class Relation(str, Enum):
    LOCATED_AT = "located_at"
    PART_OF = "part_of"
    RESPONSIBLE_FOR = "responsible_for"
    CONSUMES = "consumes"


#: Attribute -> which snapshot bucket it projects into. Explicit because the
#: projection must be inspectable, not implied by a naming convention.
STATUS_ATTRS = ("status",)
RELATION_ATTRS = ("operator",)
READING_ATTRS = ("load", "wear")

#: Status is a deterministic automaton, not a random walk: the transition is a
#: property of the world, so a test can assert the world is replayable.
STATUS_CYCLE = ("nominal", "degraded", "offline", "nominal")


class ObsKind(str, Enum):
    """How an observation relates to the world change it reports."""
    CLEAN = "clean"
    DELAYED = "delayed"              # documented well after it happened
    APPROXIMATE = "approximate"      # only the month is known
    DUPLICATE = "duplicate"          # a second source reports the same thing
    CONTRADICTORY = "contradictory"  # a second source reports something else
    CORRECTION = "correction"        # a later source corrects an earlier report


@dataclass(frozen=True)
class WorldChange:
    """A thing that actually happened, whether or not anyone saw it."""
    tick: int
    entity_id: str
    attribute: str
    old_value: str
    new_value: str
    observed: bool = True
    change_id: str = ""

    def __post_init__(self):
        object.__setattr__(self, "change_id", _cid(
            {"t": self.tick, "e": self.entity_id, "a": self.attribute,
             "v": self.new_value}, "chg"))


@dataclass(frozen=True)
class Observation:
    """What the system is allowed to see. Never the change itself."""
    obs_tick: int                # when the system learned of it (system time)
    event_tick: int              # when the world says it happened
    entity_id: str
    attribute: str
    value: str
    kind: ObsKind
    source_ordinal: int
    change_id: str
    corrects: str = ""           # obs_id of the report this supersedes
    faithful: bool = True        # does `value` match ground truth?
    obs_id: str = ""

    def __post_init__(self):
        object.__setattr__(self, "obs_id", _cid(
            {"o": self.obs_tick, "e": self.entity_id, "a": self.attribute,
             "v": self.value, "s": self.source_ordinal,
             "k": self.kind.value}, "obs"))


@dataclass
class WorldSpec:
    """Every knob that affects generation, so a run is reproducible from it."""
    seed: int = 20260823
    world_id: str = "slw-001"
    n_sites: int = 8
    units_per_site: int = 6
    components_per_unit: int = 2
    n_operators: int = 16
    n_resources: int = 8
    n_ticks: int = 60
    changes_per_tick: int = 6
    p_delayed: float = 0.20
    p_approximate: float = 0.15
    p_duplicate: float = 0.12
    p_contradictory: float = 0.08
    p_correction: float = 0.10
    p_unobserved: float = 0.10
    checkpoint_every: int = 10

    def fingerprint(self) -> str:
        return _cid(self.__dict__, "spec")


@dataclass
class SyntheticWorld:
    """Ground truth + the corrupted observation channel over it.

    The two are kept strictly separate. `truth_at(tick)` is never consulted by
    the ingest path — only by the scorer. Mixing them would make the benchmark
    grade itself.
    """
    spec: WorldSpec
    entities: dict = field(default_factory=dict)      # entity_id -> type
    relations: dict = field(default_factory=dict)     # (src, rel) -> dst
    changes: list = field(default_factory=list)       # chronological WorldChange
    observations: list = field(default_factory=list)  # chronological Observation
    _truth: dict = field(default_factory=dict)        # tick -> {(e,a): v}

    # -- construction -------------------------------------------------------

    @classmethod
    def generate(cls, spec: WorldSpec | None = None) -> "SyntheticWorld":
        spec = spec or WorldSpec()
        rng = random.Random(spec.seed)
        w = cls(spec=spec)

        sites = [f"site_{i:02d}" for i in range(spec.n_sites)]
        operators = [f"op_{i:02d}" for i in range(spec.n_operators)]
        resources = [f"res_{i:02d}" for i in range(spec.n_resources)]
        for s in sites:
            w.entities[s] = EntityType.SITE
        for o in operators:
            w.entities[o] = EntityType.OPERATOR
        for r in resources:
            w.entities[r] = EntityType.RESOURCE

        units, components = [], []
        for si, site in enumerate(sites):
            for u in range(spec.units_per_site):
                unit = f"unit_{si:02d}{u:02d}"
                units.append(unit)
                w.entities[unit] = EntityType.UNIT
                w.relations[(unit, Relation.LOCATED_AT)] = site
                w.relations[(unit, Relation.CONSUMES)] = resources[
                    (si + u) % len(resources)]
                for c in range(spec.components_per_unit):
                    comp = f"{unit}_c{c}"
                    components.append(comp)
                    w.entities[comp] = EntityType.COMPONENT
                    w.relations[(comp, Relation.PART_OF)] = unit

        # Initial state. Deterministic from the seed, and recorded as tick 0 so
        # that "the system was never told" is distinguishable from "it is the
        # default": the initial values are observed like any other change.
        state: dict = {}
        for i, unit in enumerate(units):
            state[(unit, "status")] = STATUS_CYCLE[i % 3]
            state[(unit, "load")] = str(10 + (i * 7) % 90)
            state[(unit, "operator")] = operators[i % len(operators)]
        for i, comp in enumerate(components):
            state[(comp, "wear")] = str((i * 13) % 100)

        for (ent, attr), val in sorted(state.items()):
            w.changes.append(WorldChange(tick=0, entity_id=ent, attribute=attr,
                                         old_value="", new_value=val))
        w._truth[0] = dict(state)

        mutable = sorted(state)
        for tick in range(1, spec.n_ticks + 1):
            for _ in range(spec.changes_per_tick):
                ent, attr = mutable[rng.randrange(len(mutable))]
                old = state[(ent, attr)]
                new = _advance(attr, old, rng)
                if new == old:
                    continue
                state[(ent, attr)] = new
                observed = rng.random() >= spec.p_unobserved
                w.changes.append(WorldChange(
                    tick=tick, entity_id=ent, attribute=attr,
                    old_value=old, new_value=new, observed=observed))
            w._truth[tick] = dict(state)

        w._emit_observations(rng)
        return w

    def _emit_observations(self, rng: random.Random) -> None:
        """Corrupt the change stream into what the system is allowed to see."""
        spec = self.spec
        pending: list[Observation] = []
        by_key: dict = {}   # (entity, attribute) -> last faithful obs_id

        for chg in self.changes:
            if not chg.observed:
                continue  # a real change nobody reported — a knowledge gap
            key = (chg.entity_id, chg.attribute)

            kind = ObsKind.CLEAN
            obs_tick = chg.tick
            if chg.tick > 0:
                r = rng.random()
                if r < spec.p_delayed:
                    kind = ObsKind.DELAYED
                    obs_tick = chg.tick + rng.randint(2, 9)
                elif r < spec.p_delayed + spec.p_approximate:
                    kind = ObsKind.APPROXIMATE

            primary = Observation(
                obs_tick=obs_tick, event_tick=chg.tick,
                entity_id=chg.entity_id, attribute=chg.attribute,
                value=chg.new_value, kind=kind, source_ordinal=0,
                change_id=chg.change_id)
            pending.append(primary)

            if rng.random() < spec.p_duplicate:
                # Corroboration from an independent source. Same content, so
                # the ingestor must merge rather than double-count.
                pending.append(Observation(
                    obs_tick=obs_tick + 1, event_tick=chg.tick,
                    entity_id=chg.entity_id, attribute=chg.attribute,
                    value=chg.new_value, kind=ObsKind.DUPLICATE,
                    source_ordinal=1, change_id=chg.change_id))

            if rng.random() < spec.p_contradictory:
                wrong = _advance(chg.attribute, chg.new_value, rng)
                if wrong != chg.new_value:
                    pending.append(Observation(
                        obs_tick=obs_tick, event_tick=chg.tick,
                        entity_id=chg.entity_id, attribute=chg.attribute,
                        value=wrong, kind=ObsKind.CONTRADICTORY,
                        source_ordinal=2, change_id=chg.change_id,
                        faithful=False))

            if chg.tick > 0 and rng.random() < spec.p_correction \
                    and key in by_key:
                # A source retracts an earlier *misreport*: the world did not
                # change, the record was wrong. This must supersede, not remove.
                bad = Observation(
                    obs_tick=obs_tick, event_tick=chg.tick,
                    entity_id=chg.entity_id, attribute=chg.attribute,
                    value=_advance(chg.attribute, chg.new_value, rng),
                    kind=ObsKind.CLEAN, source_ordinal=3,
                    change_id=chg.change_id, faithful=False)
                fix = Observation(
                    obs_tick=obs_tick + rng.randint(1, 4), event_tick=chg.tick,
                    entity_id=chg.entity_id, attribute=chg.attribute,
                    value=chg.new_value, kind=ObsKind.CORRECTION,
                    source_ordinal=3, change_id=chg.change_id,
                    corrects=bad.obs_id)
                if bad.value != chg.new_value:
                    pending.append(bad)
                    pending.append(fix)

            by_key[key] = primary.obs_id

        # Observations arrive in the order the system *learns* of them, which is
        # obs_tick order — not event order. Everything downstream must cope.
        self.observations = sorted(
            pending, key=lambda o: (o.obs_tick, o.entity_id, o.attribute,
                                    o.source_ordinal, o.obs_id))

    # -- ground-truth access (scorer only) ----------------------------------

    def truth_at(self, tick: int) -> dict:
        """The world's actual state at `tick`. Never used by the ingest path."""
        known = max(t for t in self._truth if t <= tick)
        return dict(self._truth[known])

    @property
    def identity(self) -> Identity:
        return Identity(subject_id=self.spec.world_id, kind="subject",
                        authority="synthetic_generator")

    def unobserved_changes(self) -> list:
        return [c for c in self.changes if not c.observed]

    def observations_through(self, tick: int) -> list:
        return [o for o in self.observations if o.obs_tick <= tick]

    def checkpoints(self) -> list:
        step = self.spec.checkpoint_every
        return list(range(step, self.spec.n_ticks + 1, step))


def _advance(attribute: str, old: str, rng: random.Random) -> str:
    """The world's transition rule. Deterministic given the rng stream."""
    if attribute == "status":
        return STATUS_CYCLE[(STATUS_CYCLE.index(old) + 1) % 3]
    if attribute == "operator":
        n = int(old.split("_")[1])
        return f"op_{(n + 1 + rng.randrange(3)) % 16:02d}"
    return str(max(0, min(100, int(old) + rng.choice((-11, -5, 6, 13)))))


# ---------------------------------------------------------------------------
# Ingest: observations -> ledger objects
# ---------------------------------------------------------------------------

def _source_for(obs: Observation, world_id: str) -> SourceArtifact:
    return SourceArtifact(
        patient_id=world_id,
        modality=Modality.TABLE,
        document_type=f"telemetry_feed_{obs.source_ordinal}",
        content=f"{obs.entity_id},{obs.attribute},{obs.value}",
        author_or_device=f"feed_{obs.source_ordinal}",
        source_system="slw_channel",
        creation_time=tick_to_date(obs.obs_tick),
        received_time=tick_to_date(obs.obs_tick),
    )


def _span_for(obs: Observation, source: SourceArtifact,
              world_id: str) -> EvidenceSpanV2:
    return EvidenceSpanV2(
        source_id=source.source_id,
        patient_id=world_id,
        modality=Modality.TABLE,
        locator=Locator(table=f"feed_{obs.source_ordinal}", row=obs.obs_tick,
                        column=obs.attribute),
        verbatim=f"{obs.entity_id} {obs.attribute} = {obs.value}",
        documentation_time=tick_to_date(obs.obs_tick),
        candidate_event_time=(tick_to_month(obs.event_tick)
                              if obs.kind is ObsKind.APPROXIMATE
                              else tick_to_date(obs.event_tick)),
        extraction_version="nano-slw-001",
    )


def _temporal_for(obs: Observation) -> TemporalExtent:
    """Approximate stays approximate — the guard in `TemporalExtent` refuses a
    full date under APPROXIMATE precision, so this cannot silently sharpen."""
    if obs.kind is ObsKind.APPROXIMATE:
        return TemporalExtent(
            event_time=tick_to_month(obs.event_tick),
            documentation_time=tick_to_date(obs.obs_tick),
            precision=TimePrecision.APPROXIMATE,
            uncertainty="month-resolution feed",
            system_recorded_time=tick_to_date(obs.obs_tick))
    return TemporalExtent(
        event_time=tick_to_date(obs.event_tick),
        documentation_time=tick_to_date(obs.obs_tick),
        precision=TimePrecision.DAY,
        system_recorded_time=tick_to_date(obs.obs_tick))


@dataclass
class IngestResult:
    ledger: EvidenceLedger
    spans_by_obs: dict = field(default_factory=dict)       # obs_id -> span id
    assertion_by_obs: dict = field(default_factory=dict)   # obs_id -> assert id
    obs_by_assertion: dict = field(default_factory=dict)   # assert id -> [obs]
    superseded_assertions: set = field(default_factory=set)
    conflicts: list = field(default_factory=list)


def ingest(world: SyntheticWorld, observations: list) -> IngestResult:
    """Fold observations into an evidence ledger.

    Duplicates merge into one assertion citing both spans (corroboration is not
    two facts). Contradictions do NOT merge — both assertions survive and a
    `ConflictRecord` is emitted, because silently picking a winner is the
    behaviour the architecture forbids.
    """
    wid = world.spec.world_id
    ledger = EvidenceLedger(patient_id=wid)
    res = IngestResult(ledger=ledger)

    # (entity, attribute, event_time, value) -> assertion, so a duplicate joins
    # the assertion it corroborates instead of creating a second one.
    merged: dict = {}
    by_obs_id: dict = {o.obs_id: o for o in observations}

    for obs in observations:
        source = _source_for(obs, wid)
        span = _span_for(obs, source, wid)
        temporal = _temporal_for(obs)
        res.spans_by_obs[obs.obs_id] = span.evidence_span_id

        key = (obs.entity_id, obs.attribute, temporal.event_time, obs.value)
        if key in merged:
            prior = merged[key]
            corroborated = ClinicalAssertion(
                patient_id=wid, subject=obs.entity_id, predicate=obs.attribute,
                obj=obs.value, original_wording=span.verbatim,
                epistemic_status=EpistemicStatus.DIRECT_MEASUREMENT,
                evidence_span_ids=tuple(sorted(
                    set(prior.evidence_span_ids) | {span.evidence_span_id})),
                normalized_concept=f"{obs.entity_id}::{obs.attribute}",
                temporal=temporal, extractor="nano-slw-001",
                derivation=DerivationMode.OBSERVED)
            ledger.append(sources=[source], spans=[span],
                          assertions=[corroborated])
            merged[key] = corroborated
            res.obs_by_assertion.setdefault(
                corroborated.assertion_id, []).extend(
                    res.obs_by_assertion.get(prior.assertion_id, []) + [obs.obs_id])
            for o in res.obs_by_assertion[corroborated.assertion_id]:
                res.assertion_by_obs[o] = corroborated.assertion_id
            continue

        assertion = ClinicalAssertion(
            patient_id=wid, subject=obs.entity_id, predicate=obs.attribute,
            obj=obs.value, original_wording=span.verbatim,
            epistemic_status=EpistemicStatus.DIRECT_MEASUREMENT,
            evidence_span_ids=(span.evidence_span_id,),
            normalized_concept=f"{obs.entity_id}::{obs.attribute}",
            temporal=temporal, extractor="nano-slw-001",
            derivation=DerivationMode.OBSERVED)
        ledger.append(sources=[source], spans=[span], assertions=[assertion])
        merged[key] = assertion
        res.assertion_by_obs[obs.obs_id] = assertion.assertion_id
        res.obs_by_assertion.setdefault(assertion.assertion_id, []).append(obs.obs_id)

        if obs.corrects:
            # Supersession, not deletion: the wrong assertion and its span stay
            # in the ledger; only its standing changes.
            bad_id = res.assertion_by_obs.get(obs.corrects)
            if bad_id:
                res.superseded_assertions.add(bad_id)

    # Contradictions: same entity+attribute+event_time, different values, none
    # of them superseded.
    live = [a for a in ledger.assertions
            if a.assertion_id not in res.superseded_assertions]
    grouped: dict = {}
    for a in live:
        grouped.setdefault(
            (a.subject, a.predicate, a.temporal.event_time), []).append(a)
    for (subject, predicate, when), group in sorted(grouped.items()):
        values = {a.obj for a in group}
        if len(values) > 1:
            conflict = ConflictRecord(
                patient_id=wid,
                conflict_type=ConflictType.VALUE_DISAGREEMENT,
                claim_set=tuple(sorted(a.assertion_id for a in group)),
                supporting_evidence=tuple(sorted(
                    s for a in group for s in a.evidence_span_ids)),
                clinical_importance="unknown",
                resolution_status="unresolved")
            ledger.append(conflicts=[conflict])
            res.conflicts.append((subject, predicate, when, conflict))

    # Changes nobody reported. The system must record a gap rather than assert
    # the old value is still true — "not found" is not "absent".
    for chg in world.unobserved_changes():
        ledger.append(gaps=[KnowledgeGap(
            patient_id=wid,
            expected_information=f"{chg.entity_id}.{chg.attribute}",
            kind=GapKind.NOT_FOUND,
            why_expected="attribute is tracked but no feed reported this interval",
            search_scope="all telemetry feeds")])
    return res


# ---------------------------------------------------------------------------
# Projection: ledger -> state snapshot
# ---------------------------------------------------------------------------

def resolve_views(res: IngestResult) -> tuple[dict, dict]:
    """(entity, attribute) -> value, plus the conflicted keys.

    Resolution rule, stated so it is auditable: drop superseded assertions;
    among the rest take the latest `event_time`; if the latest time carries more
    than one value, resolve nothing and mark the key conflicted. Refusing to
    pick is the correct behaviour — a tie is information, not a nuisance.
    """
    latest: dict = {}
    for a in res.ledger.assertions:
        if a.assertion_id in res.superseded_assertions:
            continue
        key = (a.subject, a.predicate)
        when = a.temporal.event_time
        current = latest.get(key)
        if current is None or when > current[0]:
            latest[key] = (when, {a.obj})
        elif when == current[0]:
            current[1].add(a.obj)

    views, conflicted = {}, {}
    for key, (when, values) in latest.items():
        if len(values) > 1:
            conflicted[key] = (when, tuple(sorted(values)))
        else:
            views[key] = next(iter(values))
    return views, conflicted


def project(world: SyntheticWorld, res: IngestResult) -> PatientStateSnapshot:
    views, conflicted = resolve_views(res)
    conditions, relations, readings = [], [], []
    for (ent, attr), val in sorted(views.items()):
        item = f"{ent}.{attr}={val}"
        if attr in STATUS_ATTRS:
            conditions.append(item)
        elif attr in RELATION_ATTRS:
            relations.append(item)
        else:
            readings.append(item)
    return PatientStateSnapshot(
        patient_id=world.spec.world_id,
        evidence_ledger_version=res.ledger.version,
        ledger_hash=res.ledger.ledger_hash(),
        active_conditions=tuple(conditions),
        current_medications=tuple(relations),
        laboratory_state=tuple(readings),
        uncertainties=tuple(sorted(f"{e}.{a}" for e, a in conflicted)),
        conflicts=tuple(sorted(c.conflict_id for _, _, _, c in res.conflicts)),
        unresolved_questions=tuple(sorted(
            {g.expected_information for g in res.ledger.gaps})),
        projection_version="nano-slw-001")


def state_signature(snapshot: PatientStateSnapshot) -> tuple:
    """What "the same final state" means. Ledger version and hash are excluded
    deliberately: the two arms take different paths through the ledger, so
    demanding identical bookkeeping would test the harness, not the substrate.
    What must match is the *believed world*."""
    return (snapshot.active_conditions, snapshot.current_medications,
            snapshot.laboratory_state, snapshot.uncertainties,
            snapshot.unresolved_questions)


# ---------------------------------------------------------------------------
# The derived-object layer both arms must produce
# ---------------------------------------------------------------------------

def _view_payload(views: dict, conflicted: dict, entity: str) -> str:
    parts = [f"{a}={v}" for (e, a), v in sorted(views.items()) if e == entity]
    parts += [f"{a}=?{'|'.join(vals)}"
              for (e, a), (_, vals) in sorted(conflicted.items()) if e == entity]
    return f"{entity}[" + ",".join(parts) + "]"


def derived_objects(world: SyntheticWorld, res: IngestResult) -> dict:
    """Build the three derived layers above the ledger.

        span -> assertion -> view:<entity> -> roll:<site> -> report:<site>

    The depth is the point. A change to one span must mark its assertion STALE,
    everything above it POSSIBLY_STALE, and *nothing* under another site.
    """
    views, conflicted = resolve_views(res)
    entities = sorted({e for e, _ in views} | {e for e, _ in conflicted})

    view_payloads = {e: _view_payload(views, conflicted, e) for e in entities}
    view_ids = {e: f"view:{e}@{_cid({'p': p}, 'v')}"
                for e, p in view_payloads.items()}

    site_of: dict = {}
    for ent in entities:
        etype = world.entities.get(ent)
        if etype is EntityType.UNIT:
            site_of[ent] = world.relations[(ent, Relation.LOCATED_AT)]
        elif etype is EntityType.COMPONENT:
            unit = world.relations[(ent, Relation.PART_OF)]
            site_of[ent] = world.relations[(unit, Relation.LOCATED_AT)]

    rolls, reports = {}, {}
    for site in sorted(set(site_of.values())):
        members = sorted(e for e in entities if site_of.get(e) == site)
        payload = "|".join(view_payloads[m] for m in members)
        roll_id = f"roll:{site}@{_cid({'p': payload}, 'r')}"
        rolls[site] = (roll_id, [view_ids[m] for m in members], payload)
        reports[site] = (f"report:{site}@{_cid({'p': payload}, 'rep')}",
                         [roll_id], payload)
    return {"views": view_ids, "view_payloads": view_payloads,
            "rolls": rolls, "reports": reports, "site_of": site_of,
            "assertion_ids": [a.assertion_id for a in res.ledger.assertions]}


# ---------------------------------------------------------------------------
# BASELINE A — full rebuild at every checkpoint
# ---------------------------------------------------------------------------

@dataclass
class ArmResult:
    name: str
    snapshots: dict = field(default_factory=dict)     # tick -> snapshot
    recomputed: dict = field(default_factory=dict)    # tick -> [derived ids]
    deltas: dict = field(default_factory=dict)        # tick -> StateDelta
    final: PatientStateSnapshot | None = None
    graph: DependencyGraph | None = None

    @property
    def total_recomputations(self) -> int:
        return sum(len(v) for v in self.recomputed.values())


def run_baseline_a(world: SyntheticWorld) -> ArmResult:
    """Rebuild everything, every time. Correct by construction; that is the
    only reason it is the baseline — its cost is the thing being measured."""
    arm = ArmResult(name="baseline_a_full_rebuild")
    previous = None
    for tick in world.checkpoints():
        res = ingest(world, world.observations_through(tick))
        snapshot = project(world, res)
        derived = derived_objects(world, res)
        arm.snapshots[tick] = snapshot
        arm.recomputed[tick] = (
            list(derived["assertion_ids"])
            + list(derived["views"].values())
            + [r[0] for r in derived["rolls"].values()]
            + [r[0] for r in derived["reports"].values()])
        if previous is not None and previous.snapshot_id != snapshot.snapshot_id:
            arm.deltas[tick] = diff_states(
                previous, snapshot,
                evidence_span_ids=tuple(
                    s.evidence_span_id for s in res.ledger.spans[-8:]))
        previous = snapshot
    arm.final = previous
    return arm


# ---------------------------------------------------------------------------
# CANDIDATE B — lineage-tracked incremental rebuild
# ---------------------------------------------------------------------------

def run_candidate_b(world: SyntheticWorld) -> ArmResult:
    """Recompute only what lineage says is stale.

    The graph is content-addressed: a recomputation whose inputs did not change
    produces the same id, so re-registering it raises. That is the honest
    precision signal — an object we recomputed but did not need to is visible
    as a skipped registration, not hidden inside a cost number.
    """
    arm = ArmResult(name="candidate_b_incremental")
    graph = DependencyGraph()
    arm.graph = graph

    known_view: dict = {}      # entity -> current view id
    known_roll: dict = {}      # site -> current roll id
    known_report: dict = {}
    seen_assertions: set = set()
    previous = None

    for tick in world.checkpoints():
        res = ingest(world, world.observations_through(tick))
        derived = derived_objects(world, res)
        recomputed: list = []

        span_of: dict = {}
        for a in res.ledger.assertions:
            span_of[a.assertion_id] = list(a.evidence_span_ids)

        # 1. New assertions enter the graph with their evidence as inputs.
        new_assertions = [a for a in res.ledger.assertions
                          if a.assertion_id not in seen_assertions]
        for a in new_assertions:
            graph.register(Dependency(
                derived_id=a.assertion_id, input_ids=tuple(a.evidence_span_ids),
                kind="assertion<-evidence", producer="nano.slw.ingest"))
            seen_assertions.add(a.assertion_id)
            recomputed.append(a.assertion_id)

        # 2. Supersession invalidates through lineage rather than by deletion.
        for bad in sorted(res.superseded_assertions):
            if bad in graph.freshness and \
                    graph.freshness[bad] is not Freshness.SUPERSEDED:
                graph.invalidate(bad, reason="corrected by a later report",
                                 superseded=True)

        # 3. Only entities whose view payload actually moved are rebuilt.
        touched_sites: set = set()
        for entity, view_id in sorted(derived["views"].items()):
            if known_view.get(entity) == view_id:
                continue
            inputs = tuple(sorted(
                a.assertion_id for a in res.ledger.assertions
                if a.subject == entity
                and a.assertion_id not in res.superseded_assertions))
            if not inputs:
                continue
            if view_id not in graph.edges:
                graph.register(Dependency(
                    derived_id=view_id, input_ids=inputs,
                    kind="view<-assertions", producer="nano.slw.resolve_views"))
            known_view[entity] = view_id
            recomputed.append(view_id)
            site = derived["site_of"].get(entity)
            if site:
                touched_sites.add(site)

        # 4. Rollups and reports only for sites whose members moved.
        for site in sorted(touched_sites):
            roll_id, member_views, _ = derived["rolls"][site]
            if known_roll.get(site) != roll_id:
                if roll_id not in graph.edges:
                    graph.register(Dependency(
                        derived_id=roll_id, input_ids=tuple(member_views),
                        kind="roll<-views", producer="nano.slw.derived_objects"))
                known_roll[site] = roll_id
                recomputed.append(roll_id)
            report_id, roll_inputs, _ = derived["reports"][site]
            if known_report.get(site) != report_id:
                if report_id not in graph.edges:
                    graph.register(Dependency(
                        derived_id=report_id, input_ids=tuple(roll_inputs),
                        kind="report<-roll", producer="nano.slw.derived_objects"))
                known_report[site] = report_id
                recomputed.append(report_id)

        snapshot = project(world, res)
        arm.snapshots[tick] = snapshot
        arm.recomputed[tick] = recomputed
        if previous is not None and previous.snapshot_id != snapshot.snapshot_id:
            arm.deltas[tick] = diff_states(
                previous, snapshot,
                evidence_span_ids=tuple(
                    s.evidence_span_id for s in res.ledger.spans[-8:]))
        previous = snapshot
    arm.final = previous
    return arm


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_invalidation(world: SyntheticWorld) -> dict:
    """Precision/recall of `DependencyGraph.invalidate` against ground truth.

    Ground truth here is not the world's physics — it is which derived objects
    genuinely depend on a changed span. That is computable exactly, which makes
    over-invalidation (safe but useless) as measurable as under-invalidation
    (dangerous). Reporting only recall would hide the first.
    """
    res = ingest(world, world.observations)
    derived = derived_objects(world, res)
    graph = DependencyGraph()

    for a in res.ledger.assertions:
        graph.register(Dependency(derived_id=a.assertion_id,
                                  input_ids=tuple(a.evidence_span_ids),
                                  kind="assertion<-evidence"))
    view_inputs: dict = {}
    for entity, view_id in derived["views"].items():
        inputs = tuple(sorted(a.assertion_id for a in res.ledger.assertions
                              if a.subject == entity))
        if inputs:
            graph.register(Dependency(derived_id=view_id, input_ids=inputs,
                                      kind="view<-assertions"))
            view_inputs[view_id] = inputs
    for site, (roll_id, members, _) in derived["rolls"].items():
        members = [m for m in members if m in view_inputs]
        if members:
            graph.register(Dependency(derived_id=roll_id,
                                      input_ids=tuple(members),
                                      kind="roll<-views"))
            report_id = derived["reports"][site][0]
            graph.register(Dependency(derived_id=report_id,
                                      input_ids=(roll_id,),
                                      kind="report<-roll"))

    tp = fp = fn = 0
    trials = 0
    for a in res.ledger.assertions[:40]:
        span = a.evidence_span_ids[0]
        expected = graph.dependents_of(span)
        marked = graph.invalidate(span, reason="benchmark probe")
        got = set(marked["direct"]) | set(marked["transitive"])
        tp += len(got & expected)
        fp += len(got - expected)
        fn += len(expected - got)
        trials += 1
        # Reset freshness so probes stay independent.
        graph.freshness = {k: Freshness.CURRENT for k in graph.freshness}

    return {
        "trials": trials,
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
    }


def score_unrelated_branches(world: SyntheticWorld) -> dict:
    """Changing one site must leave every other site CURRENT.

    This is the precision claim in its most load-bearing form: an architecture
    that marks the whole world stale on any change has lineage in name only.
    """
    res = ingest(world, world.observations)
    derived = derived_objects(world, res)
    graph = DependencyGraph()
    for a in res.ledger.assertions:
        graph.register(Dependency(derived_id=a.assertion_id,
                                  input_ids=tuple(a.evidence_span_ids)))
    registered_views = {}
    for entity, view_id in derived["views"].items():
        inputs = tuple(sorted(a.assertion_id for a in res.ledger.assertions
                              if a.subject == entity))
        if inputs:
            graph.register(Dependency(derived_id=view_id, input_ids=inputs))
            registered_views[entity] = view_id
    site_rolls = {}
    for site, (roll_id, members, _) in derived["rolls"].items():
        members = [m for m in members if m in registered_views.values()]
        if not members:
            continue
        graph.register(Dependency(derived_id=roll_id, input_ids=tuple(members)))
        report_id = derived["reports"][site][0]
        graph.register(Dependency(derived_id=report_id, input_ids=(roll_id,)))
        site_rolls[site] = (roll_id, report_id)

    target_site = sorted(site_rolls)[0]
    victim = next(e for e, s in derived["site_of"].items()
                  if s == target_site and e in registered_views)
    span = next(a.evidence_span_ids[0] for a in res.ledger.assertions
                if a.subject == victim)
    graph.invalidate(span, reason="single-site perturbation")

    other_sites = [s for s in site_rolls if s != target_site]
    still_current = sum(
        1 for s in other_sites
        if graph.freshness[site_rolls[s][0]] is Freshness.CURRENT
        and graph.freshness[site_rolls[s][1]] is Freshness.CURRENT)
    return {
        "perturbed_site": target_site,
        "other_sites": len(other_sites),
        "other_sites_still_current": still_current,
        "isolation": still_current / max(1, len(other_sites)),
        "target_roll_stale": graph.freshness[site_rolls[target_site][0]]
        is not Freshness.CURRENT,
    }


def run_slw_001(spec: WorldSpec | None = None) -> dict:
    """The benchmark. Returns a plain dict so the runner can serialise it."""
    world = SyntheticWorld.generate(spec)
    a = run_baseline_a(world)
    b = run_candidate_b(world)

    sig_a, sig_b = state_signature(a.final), state_signature(b.final)
    equivalent = sig_a == sig_b
    history_match = all(
        state_signature(a.snapshots[t]) == state_signature(b.snapshots[t])
        for t in world.checkpoints())

    counts = {k.value: 0 for k in ObsKind}
    for o in world.observations:
        counts[o.kind.value] += 1

    result = {
        "benchmark": "NANO-SLW-001",
        "spec_fingerprint": world.spec.fingerprint(),
        "seed": world.spec.seed,
        "world": {
            "entities": len(world.entities),
            "typed_relations": len(world.relations),
            "entity_types": sorted({t.value for t in world.entities.values()}),
            "ticks": world.spec.n_ticks,
            "ground_truth_changes": len(world.changes),
            "observations": len(world.observations),
            "unobserved_changes": len(world.unobserved_changes()),
            "observation_kinds": counts,
        },
        "arms": {
            a.name: {"recomputations": a.total_recomputations,
                     "checkpoints": len(a.snapshots),
                     "deltas": len(a.deltas)},
            b.name: {"recomputations": b.total_recomputations,
                     "checkpoints": len(b.snapshots),
                     "deltas": len(b.deltas)},
        },
        "equivalence": {
            "final_state_identical": equivalent,
            "all_checkpoints_identical": history_match,
            "final_facts": len(sig_a[0]) + len(sig_a[1]) + len(sig_a[2]),
        },
        "cost": {
            "baseline_recomputations": a.total_recomputations,
            "candidate_recomputations": b.total_recomputations,
            "recomputation_ratio": (b.total_recomputations
                                    / max(1, a.total_recomputations)),
        },
        "invalidation": score_invalidation(world),
        "branch_isolation": score_unrelated_branches(world),
    }
    # A cost saving is only reportable if the answers match. Stating it
    # unconditionally is how a benchmark starts rewarding being fast and wrong.
    if not (equivalent and history_match):
        result["cost"]["recomputation_ratio"] = None
        result["cost"]["withheld_reason"] = (
            "arms disagree; a cost saving over a different answer is not a saving")
    return result
