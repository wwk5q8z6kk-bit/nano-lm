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
* **BASELINE A** — throw the builder away and refold the entire observation
  history at every checkpoint. Always correct; cost grows with history. This is
  what "no lineage tracking" actually costs.
* **CANDIDATE B** — keep one builder, feed it only the observations that have
  arrived since the last checkpoint, and recompute only the derived objects
  lineage says are stale.

The arms share `LedgerBuilder` on purpose: if B had its own fold, a divergence
would be ambiguous between "incrementality is wrong" and "the two folds are
different programs". They differ in *what they are fed and what they rebuild*,
which is exactly the property under test.

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
    """Static structure. Relations that *change* (which operator is assigned to
    a unit) are carried as attributes instead, because a changing relation has
    to flow through the same evidence/supersession path as any other fact."""
    LOCATED_AT = "located_at"
    PART_OF = "part_of"
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
        return _cid(dict(sorted(self.__dict__.items())), "spec")


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
    observations: list = field(default_factory=list)  # arrival-ordered
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
        # that "the system was never told" stays distinguishable from "it is the
        # default": initial values are observed like any other change.
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
        seen_key: set = set()

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

            pending.append(Observation(
                obs_tick=obs_tick, event_tick=chg.tick,
                entity_id=chg.entity_id, attribute=chg.attribute,
                value=chg.new_value, kind=kind, source_ordinal=0,
                change_id=chg.change_id))

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

            if chg.tick > 0 and key in seen_key \
                    and rng.random() < spec.p_correction:
                # A source retracts an earlier *misreport*: the world did not
                # change, the record was wrong. This must supersede, not remove.
                bad = Observation(
                    obs_tick=obs_tick, event_tick=chg.tick,
                    entity_id=chg.entity_id, attribute=chg.attribute,
                    value=_advance(chg.attribute, chg.new_value, rng),
                    kind=ObsKind.CLEAN, source_ordinal=3,
                    change_id=chg.change_id, faithful=False)
                if bad.value != chg.new_value:
                    pending.append(bad)
                    pending.append(Observation(
                        obs_tick=obs_tick + rng.randint(1, 4),
                        event_tick=chg.tick,
                        entity_id=chg.entity_id, attribute=chg.attribute,
                        value=chg.new_value, kind=ObsKind.CORRECTION,
                        source_ordinal=3, change_id=chg.change_id,
                        corrects=bad.obs_id))

            seen_key.add(key)

        # Observations arrive in the order the system *learns* of them, which is
        # obs_tick order — not event order. Everything downstream must cope with
        # a correction landing before the report it corrects is even old.
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

    def observations_between(self, lo: int, hi: int) -> list:
        return [o for o in self.observations if lo < o.obs_tick <= hi]

    def checkpoints(self) -> list:
        step = self.spec.checkpoint_every
        return list(range(step, self.spec.n_ticks + 1, step))

    def site_of(self, entity: str) -> str:
        etype = self.entities.get(entity)
        if etype is EntityType.UNIT:
            return self.relations[(entity, Relation.LOCATED_AT)]
        if etype is EntityType.COMPONENT:
            unit = self.relations[(entity, Relation.PART_OF)]
            return self.relations[(unit, Relation.LOCATED_AT)]
        return ""


def _advance(attribute: str, old: str, rng: random.Random) -> str:
    """The world's transition rule. Deterministic given the rng stream."""
    if attribute == "status":
        return STATUS_CYCLE[(STATUS_CYCLE.index(old) + 1) % 3]
    if attribute == "operator":
        n = int(old.split("_")[1])
        return f"op_{(n + 1 + rng.randrange(3)) % 16:02d}"
    return str(max(0, min(100, int(old) + rng.choice((-11, -5, 6, 13)))))


# ---------------------------------------------------------------------------
# Ingest — one stateful fold, shared by both arms
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
        received_time=tick_to_date(obs.obs_tick))


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
        extraction_version="nano-slw-001")


def _temporal_for(obs: Observation) -> TemporalExtent:
    """Approximate stays approximate — `TemporalExtent` refuses a full date
    under APPROXIMATE precision, so this path cannot silently sharpen."""
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


def time_range(t: str) -> tuple:
    """A month-precision timestamp is an interval, not a point.

    `"2026-01"` means some day in January. Treating it as the string it happens
    to be makes it sort before `"2026-01-05"`, which asserts that a report about
    late January is older than one about the 5th. That is a fact about ASCII,
    not about time.
    """
    if len(t) == 7 and t[4] == "-":
        y, m = int(t[:4]), int(t[5:7])
        nxt = date(y + (m == 12), (m % 12) + 1, 1)
        return (f"{t}-01", (nxt - timedelta(days=1)).isoformat())
    return (t, t)


def strictly_after(a: str, b: str) -> bool:
    """True only when `a` *cannot* precede `b`.

    A day inside a month is incomparable to that month. Refusing to order them
    is the correct answer: the resolver then reports the key as uncertain rather
    than picking a winner by lexicographic accident.
    """
    return time_range(a)[0] > time_range(b)[1]


@dataclass
class LedgerBuilder:
    """Folds observations into an evidence ledger, one at a time.

    Every operation is local to the key an observation touches, so feeding the
    builder the whole history and feeding it only new arrivals must produce the
    same result. That equality *is* the incrementality claim; it is not asserted
    anywhere, it is measured by running both.

    Duplicates merge into one assertion citing both spans — corroboration is not
    two facts. Contradictions never merge: both assertions survive and the key
    is reported conflicted, because silently picking a winner is precisely the
    behaviour the architecture forbids.
    """
    world_id: str
    ledger: EvidenceLedger = None
    merged: dict = field(default_factory=dict)          # content key -> assertion
    by_key: dict = field(default_factory=dict)          # (ent,attr) -> [assertion]
    by_group: dict = field(default_factory=dict)        # (ent,attr,when) -> [a]
    assertion_by_obs: dict = field(default_factory=dict)
    obs_by_assertion: dict = field(default_factory=dict)
    superseded: set = field(default_factory=set)
    resolved: dict = field(default_factory=dict)        # (ent,attr) -> value
    conflicted: dict = field(default_factory=dict)      # (ent,attr) -> (when, vals)
    conflict_groups: dict = field(default_factory=dict)  # group key -> values
    gap_keys: set = field(default_factory=set)
    touched: set = field(default_factory=set)           # entities since last drain
    admitted: int = 0

    def __post_init__(self):
        if self.ledger is None:
            self.ledger = EvidenceLedger(patient_id=self.world_id)

    # -- folding ------------------------------------------------------------

    def admit(self, obs: Observation) -> None:
        source = _source_for(obs, self.world_id)
        span = _span_for(obs, source, self.world_id)
        temporal = _temporal_for(obs)
        key = (obs.entity_id, obs.attribute)
        group = (obs.entity_id, obs.attribute, temporal.event_time)
        content_key = group + (obs.value,)
        self.admitted += 1

        prior = self.merged.get(content_key)
        span_ids = ((span.evidence_span_id,) if prior is None
                    else tuple(sorted(set(prior.evidence_span_ids)
                                      | {span.evidence_span_id})))
        assertion = ClinicalAssertion(
            patient_id=self.world_id, subject=obs.entity_id,
            predicate=obs.attribute, obj=obs.value,
            original_wording=span.verbatim,
            epistemic_status=EpistemicStatus.DIRECT_MEASUREMENT,
            evidence_span_ids=span_ids,
            normalized_concept=f"{obs.entity_id}::{obs.attribute}",
            temporal=temporal, extractor="nano-slw-001",
            derivation=DerivationMode.OBSERVED)
        self.ledger.append(sources=[source], spans=[span],
                           assertions=[assertion])

        if prior is not None:
            # The corroborated assertion replaces the single-source one in the
            # live set. The original stays in the ledger: history is not edited.
            self.by_key[key] = [a for a in self.by_key.get(key, ())
                                if a.assertion_id != prior.assertion_id]
            self.by_group[group] = [a for a in self.by_group.get(group, ())
                                    if a.assertion_id != prior.assertion_id]
            carried = self.obs_by_assertion.pop(prior.assertion_id, [])
        else:
            carried = []

        self.merged[content_key] = assertion
        self.by_key.setdefault(key, []).append(assertion)
        self.by_group.setdefault(group, []).append(assertion)
        self.obs_by_assertion[assertion.assertion_id] = carried + [obs.obs_id]
        for o in self.obs_by_assertion[assertion.assertion_id]:
            self.assertion_by_obs[o] = assertion.assertion_id

        if obs.corrects:
            bad = self.assertion_by_obs.get(obs.corrects)
            if bad:
                self.supersede(bad)

        self._recompute_key(key)
        self._recompute_group(group)
        self.touched.add(obs.entity_id)

    def supersede(self, assertion_id: str) -> None:
        """Standing changes; the record does not. Supersession is not removal."""
        if assertion_id in self.superseded:
            return
        self.superseded.add(assertion_id)
        target = next((a for a in self.ledger.assertions
                       if a.assertion_id == assertion_id), None)
        if target is None:
            return
        self._recompute_key((target.subject, target.predicate))
        self._recompute_group((target.subject, target.predicate,
                               target.temporal.event_time))
        self.touched.add(target.subject)

    def admit_gap(self, change: WorldChange) -> None:
        """A tracked attribute changed and no feed reported it.

        Recorded as NOT_FOUND, never as the old value continuing to hold: the
        absence of a report is not evidence about the world.
        """
        gk = (change.entity_id, change.attribute)
        if gk in self.gap_keys:
            return
        self.gap_keys.add(gk)
        self.ledger.append(gaps=[KnowledgeGap(
            patient_id=self.world_id,
            expected_information=f"{change.entity_id}.{change.attribute}",
            kind=GapKind.NOT_FOUND,
            why_expected="attribute is tracked but no feed reported this interval",
            search_scope="all telemetry feeds")])

    # -- local resolution ---------------------------------------------------

    def _live(self, bucket) -> list:
        return [a for a in bucket if a.assertion_id not in self.superseded]

    def _recompute_key(self, key) -> None:
        """Resolution rule, stated so it is auditable.

        Keep the live assertions no other live assertion is *strictly after*.
        If that maximal set carries more than one value, resolve nothing and
        mark the key conflicted. Refusing to pick is the correct behaviour — a
        tie is information, not a nuisance.

        The maximal set, rather than `max(event_time)`, is what makes mixed
        precision safe: a month-precision report and a day-precision report
        inside that month are incomparable, so both stay maximal and the key is
        declared uncertain. An earlier version took the string max and silently
        preferred whichever sorted higher, which is where every undeclared error
        in the first full run came from.
        """
        live = self._live(self.by_key.get(key, ()))
        self.resolved.pop(key, None)
        self.conflicted.pop(key, None)
        if not live:
            return
        maximal = [a for a in live
                   if not any(strictly_after(b.temporal.event_time,
                                             a.temporal.event_time)
                              for b in live)]
        values = {a.obj for a in maximal}
        if len(values) > 1:
            latest = max(a.temporal.event_time for a in maximal)
            self.conflicted[key] = (latest, tuple(sorted(values)))
        else:
            self.resolved[key] = next(iter(values))

    def resolve_silently(self) -> dict:
        """Control arm: the same fold with the uncertainty machinery removed.

        Wherever the real resolver declares a key uncertain, this one picks a
        winner — the highest-sorting value, which is as defensible as any other
        arbitrary rule. It exists so the benchmark can answer "does declaring
        uncertainty actually reduce confident error, or does it just look
        careful?" with a number instead of an argument.
        """
        out = dict(self.resolved)
        for key, (_, values) in self.conflicted.items():
            out[key] = sorted(values)[-1]
        return out

    def _recompute_group(self, group) -> None:
        """A contradiction at an *older* time is still a contradiction. Tracked
        per (entity, attribute, event_time) so it survives being superseded by
        later observations rather than being quietly forgotten."""
        live = self._live(self.by_group.get(group, ()))
        values = {a.obj for a in live}
        if len(values) > 1:
            self.conflict_groups[group] = tuple(sorted(values))
        else:
            self.conflict_groups.pop(group, None)

    # -- outputs ------------------------------------------------------------

    def conflict_records(self) -> list:
        out = []
        for group in sorted(self.conflict_groups):
            live = self._live(self.by_group[group])
            out.append(ConflictRecord(
                patient_id=self.world_id,
                conflict_type=ConflictType.VALUE_DISAGREEMENT,
                claim_set=tuple(sorted(a.assertion_id for a in live)),
                supporting_evidence=tuple(sorted(
                    s for a in live for s in a.evidence_span_ids)),
                clinical_importance="unknown",
                resolution_status="unresolved"))
        return out

    def assertions_for(self, entity: str) -> tuple:
        return tuple(sorted(
            a.assertion_id
            for (ent, _), bucket in self.by_key.items() if ent == entity
            for a in self._live(bucket)))

    def drain_touched(self) -> set:
        out, self.touched = self.touched, set()
        return out

    def view_payload(self, entity: str) -> str:
        parts = [f"{a}={v}" for (e, a), v in sorted(self.resolved.items())
                 if e == entity]
        parts += [f"{a}=?{'|'.join(vals)}"
                  for (e, a), (_, vals) in sorted(self.conflicted.items())
                  if e == entity]
        return f"{entity}[" + ",".join(parts) + "]"

    def entities(self) -> list:
        return sorted({e for e, _ in self.resolved} | {e for e, _ in self.conflicted})


def build_full(world: SyntheticWorld, observations: list,
               through_tick: int) -> LedgerBuilder:
    """The whole-history fold. Baseline A calls this once per checkpoint."""
    b = LedgerBuilder(world_id=world.spec.world_id)
    for obs in observations:
        b.admit(obs)
    for chg in world.unobserved_changes():
        if chg.tick <= through_tick:
            b.admit_gap(chg)
    return b


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def project(world: SyntheticWorld, b: LedgerBuilder) -> PatientStateSnapshot:
    conditions, relations, readings = [], [], []
    for (ent, attr), val in sorted(b.resolved.items()):
        item = f"{ent}.{attr}={val}"
        if attr in STATUS_ATTRS:
            conditions.append(item)
        elif attr in RELATION_ATTRS:
            relations.append(item)
        elif attr in READING_ATTRS:
            readings.append(item)
        else:
            # An `else: readings.append(...)` would route an unclassified
            # attribute somewhere plausible and say nothing. Silent routing is
            # how a projection stops being inspectable.
            raise ValueError(
                f"attribute {attr!r} is not assigned to a snapshot bucket — "
                "add it to STATUS_ATTRS, RELATION_ATTRS or READING_ATTRS")
    return PatientStateSnapshot(
        patient_id=world.spec.world_id,
        evidence_ledger_version=b.ledger.version,
        ledger_hash=b.ledger.ledger_hash(),
        active_conditions=tuple(conditions),
        current_medications=tuple(relations),
        laboratory_state=tuple(readings),
        uncertainties=tuple(sorted(f"{e}.{a}" for e, a in b.conflicted)),
        conflicts=tuple(sorted(c.conflict_id for c in b.conflict_records())),
        unresolved_questions=tuple(sorted(
            f"{e}.{a}" for e, a in b.gap_keys)),
        projection_version="nano-slw-001")


def state_signature(snapshot: PatientStateSnapshot) -> tuple:
    """What "the same believed world" means, independent of bookkeeping.

    This is the *semantic* comparison: the facts held, plus what the system
    admits it does not know. It deliberately excludes ledger version and hash so
    that a legitimate difference in traversal would not be reported as a
    disagreement about the world.

    As it turns out the arms do not differ in traversal at all — both drive the
    same `LedgerBuilder.admit` in the same arrival order — so `snapshot_id`
    itself matches at every checkpoint. That stronger equality is measured
    separately (`identical_snapshot_ids`) rather than assumed here; keeping the
    weaker comparison as the gate means a future arm that legitimately reorders
    the fold is still judged on the world it believes.
    """
    return (snapshot.active_conditions, snapshot.current_medications,
            snapshot.laboratory_state, snapshot.uncertainties,
            snapshot.unresolved_questions)


# ---------------------------------------------------------------------------
# The derived-object layer both arms must produce
# ---------------------------------------------------------------------------
#
#     span -> assertion -> view:<entity> -> roll:<site> -> report:<site>
#
# The depth is the point. A change to one span must mark its assertion STALE,
# everything above it POSSIBLY_STALE, and *nothing* under another site.

def view_id_for(entity: str, payload: str) -> str:
    return f"view:{entity}@{_cid({'p': payload}, 'v')}"


def roll_id_for(site: str, payload: str) -> str:
    return f"roll:{site}@{_cid({'p': payload}, 'r')}"


def report_id_for(site: str, payload: str) -> str:
    return f"report:{site}@{_cid({'p': payload}, 'rep')}"


def derived_objects(world: SyntheticWorld, b: LedgerBuilder) -> dict:
    """Compute the full derived layer from scratch. This is Baseline A's job
    and the reference the incremental arm is scored against."""
    entities = b.entities()
    payloads = {e: b.view_payload(e) for e in entities}
    views = {e: view_id_for(e, p) for e, p in payloads.items()}
    site_of = {e: world.site_of(e) for e in entities}

    rolls, reports, members_of = {}, {}, {}
    for site in sorted({s for s in site_of.values() if s}):
        members = sorted(e for e in entities if site_of.get(e) == site)
        payload = "|".join(payloads[m] for m in members)
        members_of[site] = members
        rolls[site] = roll_id_for(site, payload)
        reports[site] = report_id_for(site, payload)
    return {"payloads": payloads, "views": views, "rolls": rolls,
            "reports": reports, "site_of": site_of, "members_of": members_of,
            "assertion_ids": [a.assertion_id for a in b.ledger.assertions]}


# ---------------------------------------------------------------------------
# BASELINE A — refold the whole history at every checkpoint
# ---------------------------------------------------------------------------

@dataclass
class ArmResult:
    name: str
    snapshots: dict = field(default_factory=dict)     # tick -> snapshot
    recomputed: dict = field(default_factory=dict)    # tick -> [derived ids]
    observations_folded: dict = field(default_factory=dict)
    deltas: dict = field(default_factory=dict)        # tick -> StateDelta
    conflict_keys: set = field(default_factory=set)
    lineage_obligations: dict = field(default_factory=dict)   # tick -> [id]
    unhonoured_obligations: list = field(default_factory=list)
    confirmed_unaffected: list = field(default_factory=list)
    final: PatientStateSnapshot | None = None
    graph: DependencyGraph | None = None

    @property
    def total_recomputations(self) -> int:
        return sum(len(v) for v in self.recomputed.values())

    @property
    def total_observations_folded(self) -> int:
        return sum(self.observations_folded.values())


def _delta(previous, snapshot, builder, superseded_keys):
    if previous is None or previous.snapshot_id == snapshot.snapshot_id:
        return None
    spans = tuple(s.evidence_span_id for s in builder.ledger.spans[-16:])
    return diff_states(previous, snapshot, evidence_span_ids=spans,
                       superseded=tuple(superseded_keys))


def run_baseline_a(world: SyntheticWorld) -> ArmResult:
    """Rebuild everything, every time. Correct by construction; that is the only
    reason it is the baseline — its cost is the thing being measured."""
    arm = ArmResult(name="baseline_a_full_rebuild")
    previous = None
    for tick in world.checkpoints():
        obs = world.observations_through(tick)
        b = build_full(world, obs, tick)
        snapshot = project(world, b)
        d = derived_objects(world, b)
        arm.snapshots[tick] = snapshot
        arm.observations_folded[tick] = len(obs)
        arm.recomputed[tick] = (list(d["assertion_ids"])
                                + sorted(d["views"].values())
                                + sorted(d["rolls"].values())
                                + sorted(d["reports"].values()))
        delta = _delta(previous, snapshot, b, ())
        if delta is not None:
            arm.deltas[tick] = delta
        previous = snapshot
        arm.conflict_keys = set(b.conflict_groups)
    arm.final = previous
    return arm


# ---------------------------------------------------------------------------
# CANDIDATE B — one builder, new observations only, lineage-gated recompute
# ---------------------------------------------------------------------------

def _retire(graph: DependencyGraph, old_id: str | None, new_id: str) -> None:
    """Mark a replaced node SUPERSEDED once its successor exists.

    Content-addressed recomputation produces a new id and leaves the old one in
    the graph — correctly, since history is not edited. But nothing was marking
    the old node as *retired*, so it stayed STALE forever and `recompute_order()`
    kept demanding work on an object that had already been rebuilt. An
    ever-growing list of obligations nobody can discharge is how an invalidation
    system stops being believed.
    """
    if old_id and old_id != new_id and old_id in graph.freshness:
        graph.freshness[old_id] = Freshness.SUPERSEDED
        graph.reasons[old_id] = f"replaced by {new_id}"


def _recompute_id(stem: str, world: SyntheticWorld, b: LedgerBuilder,
                  known_view: dict) -> str | None:
    """Recompute a derived object's content-addressed id from current state.

    Returns None for an assertion stem: assertions are immutable facts about a
    source, not recomputable projections, so an ancestor change cannot alter one
    in place — it can only supersede it.
    """
    kind, _, name = stem.partition(":")
    if kind == "view":
        return view_id_for(name, b.view_payload(name))
    if kind in ("roll", "report"):
        members = sorted(e for e in b.entities() if world.site_of(e) == name)
        payload = "|".join(b.view_payload(m) for m in members)
        return (roll_id_for(name, payload) if kind == "roll"
                else report_id_for(name, payload))
    return None


def run_candidate_b(world: SyntheticWorld) -> ArmResult:
    """Fold only what arrived; recompute only what lineage says is stale.

    The graph is content-addressed: a recomputation whose inputs did not change
    produces the same id. Those are *not* counted as recomputations, because
    counting work that produced nothing would inflate exactly the number the
    benchmark is trying to measure honestly.
    """
    arm = ArmResult(name="candidate_b_incremental")
    graph = DependencyGraph()
    arm.graph = graph
    b = LedgerBuilder(world_id=world.spec.world_id)

    known_view: dict = {}
    known_roll: dict = {}
    known_report: dict = {}
    seen_assertions: set = set()
    seen_superseded: set = set()
    # -1, not 0: `observations_between` is half-open at the low end, and the
    # world's initial state is observed at tick 0. Starting at 0 silently drops
    # every founding observation — which is what the equivalence gate caught on
    # the first run of this arm.
    previous, last_tick = None, -1

    for tick in world.checkpoints():
        arriving = world.observations_between(last_tick, tick)
        for obs in arriving:
            b.admit(obs)
        for chg in world.unobserved_changes():
            if last_tick < chg.tick <= tick:
                b.admit_gap(chg)
        arm.observations_folded[tick] = len(arriving)
        recomputed: list = []

        # 1. New assertions enter the graph with their evidence as inputs.
        for a in b.ledger.assertions:
            if a.assertion_id in seen_assertions:
                continue
            graph.register(Dependency(
                derived_id=a.assertion_id, input_ids=tuple(a.evidence_span_ids),
                kind="assertion<-evidence", producer="nano.slw.LedgerBuilder.admit"))
            seen_assertions.add(a.assertion_id)
            recomputed.append(a.assertion_id)

        # 2. Supersession invalidates through lineage rather than by deletion.
        corrections = sorted(b.superseded - seen_superseded)
        for bad in corrections:
            seen_superseded.add(bad)
            if bad in graph.freshness:
                graph.invalidate(bad, reason="corrected by a later report",
                                 superseded=True)

        # 2b. Ask lineage what a correction obliges us to rebuild, and in what
        #     order. This is the step that makes LRN-CORRECTION more than a
        #     graph nobody consults: the work list is *derived* from the graph
        #     rather than asserted alongside it, and step 5 checks that the
        #     content-driven rebuild below actually covered it.
        obliged = [o for o in graph.recompute_order()
                   if graph.freshness[o] is not Freshness.SUPERSEDED]
        arm.lineage_obligations[tick] = obliged

        # 3. Rebuild views only for entities the fold actually touched, and only
        #    when the payload moved.
        touched_sites: set = set()
        for entity in sorted(b.drain_touched()):
            payload = b.view_payload(entity)
            vid = view_id_for(entity, payload)
            if known_view.get(entity) == vid:
                continue
            inputs = b.assertions_for(entity)
            if not inputs:
                continue
            if vid not in graph.edges:
                graph.register(Dependency(
                    derived_id=vid, input_ids=inputs, kind="view<-assertions",
                    producer="nano.slw.LedgerBuilder.view_payload"))
            _retire(graph, known_view.get(entity), vid)
            known_view[entity] = vid
            recomputed.append(vid)
            site = world.site_of(entity)
            if site:
                touched_sites.add(site)

        # 4. Rollups and reports only for sites whose members moved.
        for site in sorted(touched_sites):
            members = sorted(e for e in b.entities() if world.site_of(e) == site)
            payload = "|".join(b.view_payload(m) for m in members)
            rid = roll_id_for(site, payload)
            if known_roll.get(site) != rid:
                member_views = tuple(known_view[m] for m in members
                                     if m in known_view)
                if rid not in graph.edges and member_views:
                    graph.register(Dependency(
                        derived_id=rid, input_ids=member_views,
                        kind="roll<-views", producer="nano.slw.run_candidate_b"))
                _retire(graph, known_roll.get(site), rid)
                known_roll[site] = rid
                recomputed.append(rid)
            pid = report_id_for(site, payload)
            if known_report.get(site) != pid:
                if pid not in graph.edges:
                    graph.register(Dependency(
                        derived_id=pid, input_ids=(rid,), kind="report<-roll",
                        producer="nano.slw.run_candidate_b"))
                _retire(graph, known_report.get(site), pid)
                known_report[site] = pid
                recomputed.append(pid)

        # 5. Discharge what lineage demanded.
        #
        #    POSSIBLY_STALE means "an ancestor changed, effect unconfirmed" —
        #    and *unconfirmed* is a state something has to resolve. Leaving it
        #    is the under-invalidation failure wearing a cautious label: the
        #    artifact is still being served and nobody has checked it.
        #
        #    So every obligation the content-driven pass did not already rebuild
        #    gets recomputed and compared. Identical content means the ancestor's
        #    change did not reach here — confirmed CURRENT. Different content
        #    that nobody rebuilt is a genuine miss. The confirmation costs real
        #    work, so it is counted as real work.
        rebuilt = set(recomputed)
        for obj in obliged:
            if obj in rebuilt:
                continue
            # Views/rolls/reports are rebuilt under a NEW content-addressed id,
            # so "honoured" means a successor was produced, not that the stale
            # id reappeared.
            stem = obj.split("@")[0]
            if any(r.split("@")[0] == stem for r in rebuilt):
                continue
            current = _recompute_id(stem, world, b, known_view)
            recomputed.append(f"confirm:{stem}")
            if current == obj:
                graph.freshness[obj] = Freshness.CURRENT
                graph.reasons[obj] = (
                    "recomputed after an ancestor changed; content unchanged, "
                    "so the change did not reach this object")
                arm.confirmed_unaffected.append((tick, obj))
            elif current is None:
                arm.confirmed_unaffected.append((tick, obj))
            else:
                arm.unhonoured_obligations.append((tick, obj))

        snapshot = project(world, b)
        arm.snapshots[tick] = snapshot
        arm.recomputed[tick] = recomputed
        delta = _delta(previous, snapshot, b, ())
        if delta is not None:
            arm.deltas[tick] = delta
        previous, last_tick = snapshot, tick
        arm.conflict_keys = set(b.conflict_groups)
    arm.final = previous
    return arm


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _reference_graph(world: SyntheticWorld, b: LedgerBuilder) -> tuple:
    """Build the lineage graph plus an *independent* structural index.

    The index is deliberately not derived from `DependencyGraph`. Scoring
    invalidation against `dependents_of` would be the function grading itself:
    `invalidate` calls that very method, so precision and recall would be 1.0 by
    construction whatever the traversal did. Ground truth here is recomputed
    from the derivation structure — who cites whom — so a broken traversal is
    visible.
    """
    d = derived_objects(world, b)
    graph = DependencyGraph()

    assertions_of_span: dict = {}
    for a in b.ledger.assertions:
        graph.register(Dependency(derived_id=a.assertion_id,
                                  input_ids=tuple(a.evidence_span_ids),
                                  kind="assertion<-evidence"))
        for s in a.evidence_span_ids:
            assertions_of_span.setdefault(s, set()).add(a.assertion_id)

    view_of_assertion: dict = {}
    registered_views: dict = {}
    for entity, vid in sorted(d["views"].items()):
        inputs = tuple(sorted({a.assertion_id for a in b.ledger.assertions
                               if a.subject == entity}))
        if not inputs or vid in graph.edges:
            continue
        graph.register(Dependency(derived_id=vid, input_ids=inputs,
                                  kind="view<-assertions"))
        registered_views[entity] = vid
        for aid in inputs:
            view_of_assertion.setdefault(aid, set()).add(vid)

    roll_of_view: dict = {}
    report_of_roll: dict = {}
    for site, members in sorted(d["members_of"].items()):
        member_views = tuple(registered_views[m] for m in members
                             if m in registered_views)
        if not member_views:
            continue
        rid, pid = d["rolls"][site], d["reports"][site]
        if rid in graph.edges:
            continue
        graph.register(Dependency(derived_id=rid, input_ids=member_views,
                                  kind="roll<-views"))
        graph.register(Dependency(derived_id=pid, input_ids=(rid,),
                                  kind="report<-roll"))
        for v in member_views:
            roll_of_view.setdefault(v, set()).add(rid)
        report_of_roll[rid] = pid

    def expected_dependents(span_id: str) -> set:
        """Walk the citation index, not the graph."""
        out = set(assertions_of_span.get(span_id, ()))
        views = {v for aid in out for v in view_of_assertion.get(aid, ())}
        rolls = {r for v in views for r in roll_of_view.get(v, ())}
        reports = {report_of_roll[r] for r in rolls if r in report_of_roll}
        return out | views | rolls | reports

    return graph, d, expected_dependents


def score_invalidation(world: SyntheticWorld, trials: int = 60) -> dict:
    """Precision/recall of `DependencyGraph.invalidate` against an index built
    independently of the graph.

    Over-invalidation (safe but useless) is as measurable as under-invalidation
    (dangerous). Reporting only recall would hide the first, and an architecture
    that marks the whole world stale on any change scores recall 1.0.
    """
    b = build_full(world, world.observations, world.spec.n_ticks)
    graph, _, expected_dependents = _reference_graph(world, b)

    spans = sorted({s for a in b.ledger.assertions
                    for s in a.evidence_span_ids})
    step = max(1, len(spans) // trials)
    probes = spans[::step][:trials]

    tp = fp = fn = 0
    depth_direct_ok = depth_transitive_ok = 0
    for span in probes:
        expected = expected_dependents(span)
        marked = graph.invalidate(span, reason="benchmark probe")
        got = set(marked["direct"]) | set(marked["transitive"])
        tp += len(got & expected)
        fp += len(got - expected)
        fn += len(expected - got)
        # An assertion cites the span directly, so it must be STALE, while a
        # view two hops up must be POSSIBLY_STALE. Collapsing the two would
        # still score precision 1.0, so the distinction is scored separately.
        if all(graph.freshness[a] is Freshness.STALE
               for a in marked["direct"]):
            depth_direct_ok += 1
        if all(graph.freshness[t] is Freshness.POSSIBLY_STALE
               for t in marked["transitive"]):
            depth_transitive_ok += 1
        graph.freshness = {k: Freshness.CURRENT for k in graph.freshness}

    return {
        "trials": len(probes),
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "direct_marked_stale": depth_direct_ok / max(1, len(probes)),
        "transitive_marked_possibly_stale":
            depth_transitive_ok / max(1, len(probes)),
        "graph_nodes": len(graph.freshness),
    }


def score_unrelated_branches(world: SyntheticWorld) -> dict:
    """Changing one site must leave every other site CURRENT.

    This is the precision claim in its most load-bearing form: an architecture
    that marks the whole world stale on any change has lineage in name only.
    """
    b = build_full(world, world.observations, world.spec.n_ticks)
    graph, d, _ = _reference_graph(world, b)

    sites = sorted(s for s in d["rolls"] if d["rolls"][s] in graph.edges)
    target = sites[0]
    victim = next(e for e in d["members_of"][target] if e in d["views"])
    span = next(a.evidence_span_ids[0] for a in b.ledger.assertions
                if a.subject == victim)
    graph.invalidate(span, reason="single-site perturbation")

    others = [s for s in sites if s != target]
    clean = sum(1 for s in others
                if graph.freshness[d["rolls"][s]] is Freshness.CURRENT
                and graph.freshness[d["reports"][s]] is Freshness.CURRENT)
    return {
        "perturbed_site": target,
        "perturbed_entity": victim,
        "other_sites": len(others),
        "other_sites_still_current": clean,
        "isolation": clean / max(1, len(others)),
        "target_roll_invalidated":
            graph.freshness[d["rolls"][target]] is not Freshness.CURRENT,
        "every_stale_object_has_a_reason": all(
            graph.explain(k)["reason"]
            for k, v in graph.freshness.items() if v is not Freshness.CURRENT),
    }


def score_faithfulness(world: SyntheticWorld) -> dict:
    """How close the resolved state gets to ground truth, and — more usefully —
    whether the misses are *declared*.

    A system that is wrong and says so is behaving correctly under a lying
    channel; one that is wrong and confident is not. So the headline is not
    accuracy, it is `undeclared_error`: keys the system asserted confidently and
    got wrong. That is the number an operator would actually be harmed by.

    An absolute threshold on that rate would be a tuned constant — it moves with
    the corruption rates, not with the substrate. So the score is reported
    against a **silent-resolution control**: the identical fold with the
    uncertainty machinery removed, forced to name a winner every time. If
    declaring uncertainty is doing real work, the control must be confidently
    wrong more often. That comparison holds at any world size; a threshold would
    not.
    """
    b = build_full(world, world.observations, world.spec.n_ticks)
    truth = world.truth_at(world.spec.n_ticks)
    declared = set(b.conflicted) | set(b.gap_keys)

    def tally(resolution: dict, declares: bool) -> dict:
        correct = wrong = undeclared = 0
        for key, val in resolution.items():
            if key not in truth:
                continue
            if val == truth[key]:
                correct += 1
                continue
            wrong += 1
            if not declares or key not in declared:
                undeclared += 1
        return {"resolved_keys": len(resolution), "correct": correct,
                "incorrect": wrong,
                "accuracy": correct / max(1, correct + wrong),
                "undeclared_error": undeclared,
                "undeclared_error_rate": undeclared / max(1, len(resolution))}

    nano = tally(b.resolved, declares=True)
    control = tally(b.resolve_silently(), declares=False)
    return {
        "ground_truth_keys": len(truth),
        "declared_unknown": len(declared),
        "nano": nano,
        "silent_resolution_control": control,
        "undeclared_error_avoided":
            control["undeclared_error"] - nano["undeclared_error"],
        "control_is_worse":
            control["undeclared_error"] > nano["undeclared_error"],
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
    # Stronger than the signature: content-addressed snapshot identity, which
    # also pins ledger version and hash. Measured, not assumed.
    identical_ids = all(a.snapshots[t].snapshot_id == b.snapshots[t].snapshot_id
                        for t in world.checkpoints())
    conflicts_match = a.conflict_keys == b.conflict_keys

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
            "relation_types": sorted({r.value for _, r in world.relations}),
            "ticks": world.spec.n_ticks,
            "ground_truth_changes": len(world.changes),
            "observations": len(world.observations),
            "unobserved_changes": len(world.unobserved_changes()),
            "observation_kinds": counts,
        },
        "arms": {
            a.name: {"recomputations": a.total_recomputations,
                     "observations_folded": a.total_observations_folded,
                     "checkpoints": len(a.snapshots), "deltas": len(a.deltas)},
            b.name: {"recomputations": b.total_recomputations,
                     "observations_folded": b.total_observations_folded,
                     "checkpoints": len(b.snapshots), "deltas": len(b.deltas)},
        },
        "equivalence": {
            "final_state_identical": equivalent,
            "all_checkpoints_identical": history_match,
            "conflict_sets_identical": conflicts_match,
            "identical_snapshot_ids": identical_ids,
            "final_facts": sum(len(x) for x in sig_a[:3]),
            "final_declared_unknown": len(sig_a[3]) + len(sig_a[4]),
        },
        "cost": {
            "baseline_recomputations": a.total_recomputations,
            "candidate_recomputations": b.total_recomputations,
            "baseline_observations_folded": a.total_observations_folded,
            "candidate_observations_folded": b.total_observations_folded,
        },
        "correction_absorption": {
            "corrections_applied": len(
                {o for obs in b.lineage_obligations.values() for o in obs}),
            "lineage_obligations": sum(
                len(v) for v in b.lineage_obligations.values()),
            "confirmed_unaffected": len(b.confirmed_unaffected),
            "unhonoured_obligations": len(b.unhonoured_obligations),
            "all_obligations_honoured": not b.unhonoured_obligations,
        },
        "invalidation": score_invalidation(world),
        "branch_isolation": score_unrelated_branches(world),
        "faithfulness": score_faithfulness(world),
    }
    # A cost saving is only reportable if the answers match. Stating it
    # unconditionally is how a benchmark starts rewarding being fast and wrong.
    if equivalent and history_match and conflicts_match:
        result["cost"]["recomputation_ratio"] = (
            b.total_recomputations / max(1, a.total_recomputations))
        result["cost"]["fold_ratio"] = (
            b.total_observations_folded / max(1, a.total_observations_folded))
    else:
        result["cost"]["recomputation_ratio"] = None
        result["cost"]["fold_ratio"] = None
        result["cost"]["withheld_reason"] = (
            "arms disagree; a saving measured against a different answer is "
            "not a saving")
    return result
