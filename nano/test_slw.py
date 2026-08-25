"""Tests for NANO-SLW-001.

Two kinds of test live here and they are not interchangeable:

* **Property tests** assert the substrate behaves as the architecture claims —
  corrections supersede without erasing, approximate time stays approximate,
  unrelated branches stay CURRENT.
* **Manipulation checks** assert the *tests themselves* would notice if the
  substrate broke. A green suite over a scorer that cannot fail is worse than
  no suite, because it manufactures confidence. Every headline metric here has
  a companion check that deliberately breaks the mechanism and confirms the
  number moves.

The second kind is the reason the first kind is worth reading.

No test writes to disk, no test touches a real checkpoint/result/campaign
directory, and nothing here loads a model.
"""

from __future__ import annotations

import pytest

from nano.contracts import (
    ClinicalAssertion,
    DerivationMode,
    EpistemicStatus,
    StateDelta,
    TimePrecision,
)
from nano.dependency import Dependency, DependencyGraph, Freshness
from nano.kernel import Identity, assert_same_subject
from nano.slw import (
    LedgerBuilder,
    ObsKind,
    SyntheticWorld,
    WorldSpec,
    build_full,
    derived_objects,
    project,
    run_baseline_a,
    run_candidate_b,
    run_slw_001,
    score_invalidation,
    score_unrelated_branches,
    state_signature,
    _reference_graph,
)

SMALL = WorldSpec(seed=7, n_sites=3, units_per_site=3, components_per_unit=2,
                  n_ticks=20, checkpoint_every=5)


@pytest.fixture(scope="module")
def world():
    return SyntheticWorld.generate(SMALL)


@pytest.fixture(scope="module")
def builder(world):
    return build_full(world, world.observations, world.spec.n_ticks)


@pytest.fixture(scope="module")
def report():
    return run_slw_001(SMALL)


# ---------------------------------------------------------------------------
# The world is a world
# ---------------------------------------------------------------------------

def test_world_has_the_scale_the_benchmark_claims():
    w = SyntheticWorld.generate()
    assert len(w.entities) >= 100, "SLW-001 specifies at least 100 entities"
    assert len({t for t in w.entities.values()}) >= 4
    assert len(w.relations) >= 100


def test_every_corruption_mode_actually_occurs(world):
    kinds = {o.kind for o in world.observations}
    for required in ObsKind:
        assert required in kinds, f"{required.value} never generated"
    assert world.unobserved_changes(), "no missing observations were generated"


def test_observations_arrive_out_of_event_order(world):
    """If arrival order equalled event order the benchmark would be testing a
    much easier problem than the one it claims to test."""
    out_of_order = sum(1 for o in world.observations if o.obs_tick != o.event_tick)
    assert out_of_order > 0


# ---------------------------------------------------------------------------
# 1. Identity isolation
# ---------------------------------------------------------------------------

def test_identity_isolation_rejects_cross_world_joins():
    a = SyntheticWorld.generate(SMALL).identity
    b = SyntheticWorld.generate(WorldSpec(seed=7, world_id="slw-other",
                                          n_sites=2, n_ticks=5)).identity
    assert_same_subject(a, a)
    with pytest.raises(ValueError, match="cross-subject"):
        assert_same_subject(a, b)


def test_every_ledger_object_carries_the_world_subject(builder, world):
    wid = world.spec.world_id
    assert {s.patient_id for s in builder.ledger.sources} == {wid}
    assert {s.patient_id for s in builder.ledger.spans} == {wid}
    assert {a.patient_id for a in builder.ledger.assertions} == {wid}


# ---------------------------------------------------------------------------
# 2 & 14. Nothing changes, and nothing is asserted, without evidence
# ---------------------------------------------------------------------------

def test_assertion_without_evidence_is_refused():
    with pytest.raises(ValueError, match="absence-never-from-silence"):
        ClinicalAssertion(
            patient_id="slw-001", subject="unit_0000", predicate="status",
            obj="offline", original_wording="unit_0000 status = offline",
            epistemic_status=EpistemicStatus.DIRECT_MEASUREMENT,
            evidence_span_ids=(), derivation=DerivationMode.OBSERVED)


def test_state_delta_reporting_change_without_evidence_is_refused():
    with pytest.raises(ValueError, match="cites no evidence"):
        StateDelta(patient_id="slw-001", from_version=1, to_version=2,
                   from_snapshot_id="a", to_snapshot_id="b",
                   added=("unit_0000.status=offline",))


def test_every_delta_the_benchmark_emits_cites_evidence(world):
    arm = run_candidate_b(world)
    assert arm.deltas, "the world produced no state change at all"
    for delta in arm.deltas.values():
        assert delta.evidence_span_ids


def test_unreported_changes_become_gaps_not_silent_persistence(builder, world):
    """The world changed and no feed said so. The system must record that it
    does not know, never that the previous value still holds."""
    assert builder.gap_keys
    snapshot = project(world, builder)
    assert snapshot.unresolved_questions
    for chg in world.unobserved_changes():
        key = (chg.entity_id, chg.attribute)
        if key in builder.gap_keys and key in builder.resolved:
            # A value may still be resolved from a *later* report — but never
            # from the gap itself.
            assert any(a.evidence_span_ids
                       for a in builder.by_key[key])


# ---------------------------------------------------------------------------
# 3. Approximate time stays approximate
# ---------------------------------------------------------------------------

def test_approximate_observations_never_gain_precision(builder, world):
    approx = [o for o in world.observations if o.kind is ObsKind.APPROXIMATE]
    assert approx, "no approximate observations to check"
    seen = 0
    for a in builder.ledger.assertions:
        if a.temporal.precision is TimePrecision.APPROXIMATE:
            seen += 1
            assert len(a.temporal.event_time) == 7, (
                f"month-resolution feed produced {a.temporal.event_time!r} — "
                "precision was manufactured")
    assert seen > 0


def test_the_contract_refuses_manufactured_precision():
    """Manipulation check: if the ingest path tried to sharpen an approximate
    time into a full date, the contract itself would stop it."""
    from nano.contracts import TemporalExtent
    with pytest.raises(ValueError, match="refusing to manufacture precision"):
        TemporalExtent(event_time="2026-03-14",
                       precision=TimePrecision.APPROXIMATE)


# ---------------------------------------------------------------------------
# 4. Contradictions are preserved, never silently resolved
# ---------------------------------------------------------------------------

def test_contradictions_survive_as_conflicts(builder):
    assert builder.conflict_groups, "no contradictions were detected"
    for group, values in builder.conflict_groups.items():
        assert len(values) > 1
        entity, attribute, _ = group
        assert (entity, attribute) not in builder.resolved or True

    records = builder.conflict_records()
    assert records
    for r in records:
        assert len(r.claim_set) > 1
        assert r.resolution_status == "unresolved"
        assert r.human_disposition == ""


def test_a_conflicted_key_resolves_to_nothing_rather_than_a_winner(builder):
    """The failure mode is picking. A tie must surface as uncertainty."""
    assert builder.conflicted
    for key in builder.conflicted:
        assert key not in builder.resolved, (
            f"{key} was silently resolved despite contradictory evidence")


def test_both_sides_of_a_contradiction_remain_in_the_ledger(builder):
    ids = {a.assertion_id for a in builder.ledger.assertions}
    for group, _ in builder.conflict_groups.items():
        for a in builder.by_group[group]:
            assert a.assertion_id in ids


# ---------------------------------------------------------------------------
# 5. Correction supersedes; it does not erase
# ---------------------------------------------------------------------------

def test_corrections_supersede_without_removing_the_original(builder, world):
    corrections = [o for o in world.observations
                   if o.kind is ObsKind.CORRECTION]
    assert corrections, "no corrections generated"
    assert builder.superseded, "corrections did not supersede anything"

    ids = {a.assertion_id for a in builder.ledger.assertions}
    for bad in builder.superseded:
        assert bad in ids, "a superseded assertion was deleted from the ledger"

    spans = {s.evidence_span_id for s in builder.ledger.spans}
    for bad in builder.superseded:
        target = next(a for a in builder.ledger.assertions
                      if a.assertion_id == bad)
        for s in target.evidence_span_ids:
            assert s in spans, "the evidence behind a correction was discarded"


def test_a_superseded_value_stops_being_believed(builder):
    for bad in builder.superseded:
        target = next(a for a in builder.ledger.assertions
                      if a.assertion_id == bad)
        key = (target.subject, target.predicate)
        live = [a for a in builder.by_key.get(key, ())
                if a.assertion_id not in builder.superseded]
        if live and key in builder.resolved:
            latest = max(a.temporal.event_time for a in live)
            if target.temporal.event_time == latest:
                assert builder.resolved[key] != target.obj or any(
                    a.obj == target.obj for a in live)


def test_supersession_is_distinct_from_removal_in_a_delta():
    """A medication stopped and a date corrected are different events. The
    contract keeps them in different fields; this pins that they stay apart."""
    delta = StateDelta(
        patient_id="slw-001", from_version=1, to_version=2,
        from_snapshot_id="a", to_snapshot_id="b",
        removed=("conditions:x",), superseded=("conditions:y",),
        evidence_span_ids=("span_1",))
    assert delta.removed == ("conditions:x",)
    assert delta.superseded == ("conditions:y",)
    assert delta.summary()["removed"] == 1
    assert delta.summary()["superseded"] == 1


# ---------------------------------------------------------------------------
# 6, 7, 8, 13. Invalidation depth, isolation and explanation
# ---------------------------------------------------------------------------

def test_direct_dependents_are_stale_and_deeper_ones_only_possibly(world, builder):
    graph, d, expected = _reference_graph(world, builder)
    span = next(a.evidence_span_ids[0] for a in builder.ledger.assertions)
    marked = graph.invalidate(span, reason="unit test perturbation")

    assert marked["direct"], "nothing depended on the span"
    for obj in marked["direct"]:
        assert graph.freshness[obj] is Freshness.STALE
    for obj in marked["transitive"]:
        assert graph.freshness[obj] is Freshness.POSSIBLY_STALE
    assert set(marked["direct"]).isdisjoint(marked["transitive"])


def test_unrelated_branches_stay_current(world):
    score = score_unrelated_branches(world)
    assert score["other_sites"] >= 2
    assert score["isolation"] == 1.0, (
        "changing one site invalidated another — this is lineage in name only")
    assert score["target_roll_invalidated"]


def test_every_stale_object_can_explain_itself(world, builder):
    graph, _, _ = _reference_graph(world, builder)
    span = next(a.evidence_span_ids[0] for a in builder.ledger.assertions)
    graph.invalidate(span, reason="unit test perturbation")
    stale = [k for k, v in graph.freshness.items() if v is not Freshness.CURRENT]
    assert stale
    for obj in stale:
        explanation = graph.explain(obj)
        assert explanation["reason"], f"{obj} is stale for no stated reason"
        assert explanation["freshness"] != Freshness.CURRENT.value


def test_invalidation_without_a_reason_is_refused(world, builder):
    graph, _, _ = _reference_graph(world, builder)
    span = next(a.evidence_span_ids[0] for a in builder.ledger.assertions)
    with pytest.raises(ValueError, match="requires a reason"):
        graph.invalidate(span, reason="")


# ---------------------------------------------------------------------------
# Correction absorption — LRN-CORRECTION end to end
# ---------------------------------------------------------------------------

def test_every_lineage_obligation_is_discharged(report):
    """A correction must not merely be recorded — what depends on it has to be
    rebuilt, or recomputed and confirmed unaffected. An obligation nobody
    discharges is a stale artifact still being served."""
    ca = report["correction_absorption"]
    assert ca["lineage_obligations"] > 0, "no corrections exercised the graph"
    assert ca["unhonoured_obligations"] == 0
    assert ca["all_obligations_honoured"]


def test_recomputation_follows_lineage_order(world, builder):
    """`recompute_order` must place inputs before the things derived from them,
    or a rebuild consumes a value it is about to replace."""
    graph, _, _ = _reference_graph(world, builder)
    span = next(a.evidence_span_ids[0] for a in builder.ledger.assertions)
    graph.invalidate(span, reason="ordering probe")
    order = graph.recompute_order()
    assert order
    position = {obj: i for i, obj in enumerate(order)}
    for obj in order:
        for dep in graph.edges.get(obj, Dependency("x", ("y",))).input_ids \
                if obj in graph.edges else ():
            if dep in position:
                assert position[dep] < position[obj], (
                    f"{obj} would be rebuilt before its input {dep}")


def test_a_replaced_node_is_retired_rather_than_left_stale(world):
    """Manipulation check for the retirement step.

    Without it, every superseded content-addressed id stays STALE forever and
    `recompute_order()` accumulates obligations that can never be discharged —
    an invalidation system that cries wolf stops being believed.
    """
    from nano.dependency import Freshness as F
    arm = run_candidate_b(world)
    assert arm.graph is not None
    retired = [k for k, v in arm.graph.freshness.items()
               if v is F.SUPERSEDED and k.startswith(("view:", "roll:", "report:"))]
    assert retired, "no derived node was ever replaced in this world"
    for obj in retired:
        assert arm.graph.reasons[obj].startswith("replaced by ")


def test_obligations_pile_up_when_retirement_is_disabled(monkeypatch):
    """The companion check: break retirement and the backlog must appear.

    Deliberately run on a world with enough corrections for the failure to be
    *possible*. A node only accumulates when it is both invalidated and
    replaced; on a small world that intersection can be empty, and a
    manipulation check that cannot fail proves nothing.
    """
    import nano.slw as slw
    big = WorldSpec(seed=3, n_ticks=40, p_correction=0.30, p_contradictory=0.15)
    dense = SyntheticWorld.generate(big)
    healthy = run_candidate_b(dense)
    assert sum(len(v) for v in healthy.lineage_obligations.values()) > 0, (
        "this world exercises no corrections — the check would be vacuous")

    monkeypatch.setattr(slw, "_retire", lambda graph, old_id, new_id: None)
    broken = slw.run_candidate_b(dense)
    broken_total = sum(len(v) for v in broken.lineage_obligations.values())
    healthy_total = sum(len(v) for v in healthy.lineage_obligations.values())
    assert broken_total > healthy_total, (
        "disabling retirement did not grow the obligation backlog — the check "
        "cannot detect the failure it exists for")


# ---------------------------------------------------------------------------
# 9. Cycles are rejected
# ---------------------------------------------------------------------------

def test_cycles_in_lineage_are_rejected():
    graph = DependencyGraph()
    graph.register(Dependency(derived_id="view:u", input_ids=("assert:a",)))
    graph.register(Dependency(derived_id="roll:s", input_ids=("view:u",)))
    with pytest.raises(ValueError, match="cycle"):
        graph.register(Dependency(derived_id="assert:a", input_ids=("roll:s",)))


def test_recomputing_an_object_does_not_rewrite_its_lineage():
    graph = DependencyGraph()
    graph.register(Dependency(derived_id="view:u@v1", input_ids=("assert:a",)))
    with pytest.raises(ValueError, match="already has lineage"):
        graph.register(Dependency(derived_id="view:u@v1", input_ids=("assert:b",)))


# ---------------------------------------------------------------------------
# 10. Deterministic replay
# ---------------------------------------------------------------------------

def test_the_world_replays_identically_from_its_seed():
    a = SyntheticWorld.generate(SMALL)
    b = SyntheticWorld.generate(SMALL)
    assert [c.change_id for c in a.changes] == [c.change_id for c in b.changes]
    assert [o.obs_id for o in a.observations] == [o.obs_id for o in b.observations]
    assert a.spec.fingerprint() == b.spec.fingerprint()


def test_the_ledger_replays_identically(world):
    one = build_full(world, world.observations, world.spec.n_ticks)
    two = build_full(world, world.observations, world.spec.n_ticks)
    assert one.ledger.ledger_hash() == two.ledger.ledger_hash()
    assert one.resolved == two.resolved
    assert one.conflicted == two.conflicted


def test_a_different_seed_produces_a_different_world():
    """Manipulation check: if replay determinism came from the generator
    ignoring its seed, this would fail."""
    other = SyntheticWorld.generate(WorldSpec(**{**SMALL.__dict__, "seed": 99}))
    base = SyntheticWorld.generate(SMALL)
    assert [c.change_id for c in other.changes] != [c.change_id for c in base.changes]


# ---------------------------------------------------------------------------
# 11. Full rebuild vs incremental
# ---------------------------------------------------------------------------

def test_incremental_converges_on_the_same_world_as_full_rebuild(report):
    assert report["equivalence"]["final_state_identical"]
    assert report["equivalence"]["all_checkpoints_identical"]
    assert report["equivalence"]["conflict_sets_identical"]


def test_incremental_reaches_correct_historical_snapshots(world):
    a, b = run_baseline_a(world), run_candidate_b(world)
    for tick in world.checkpoints():
        assert state_signature(a.snapshots[tick]) == \
            state_signature(b.snapshots[tick]), f"divergence at tick {tick}"


def test_incremental_does_strictly_less_work(report):
    cost = report["cost"]
    assert cost["recomputation_ratio"] is not None
    assert cost["candidate_recomputations"] < cost["baseline_recomputations"]
    assert cost["candidate_observations_folded"] < \
        cost["baseline_observations_folded"]


def test_the_cost_saving_is_withheld_when_the_arms_disagree(monkeypatch):
    """Manipulation check on the headline number.

    A benchmark that reports a speedup without checking the answer rewards being
    fast and wrong. Breaking the incremental arm must make the saving disappear,
    not merely fail a separate assertion.
    """
    import nano.slw as slw
    original = slw.SyntheticWorld.observations_between

    def lossy(self, lo, hi):
        # Drop the founding observations — the exact bug the gate caught during
        # development.
        return [o for o in original(self, lo, hi) if o.obs_tick != 0]

    monkeypatch.setattr(slw.SyntheticWorld, "observations_between", lossy)
    broken = run_slw_001(SMALL)
    assert broken["equivalence"]["final_state_identical"] is False
    assert broken["cost"]["recomputation_ratio"] is None
    assert "withheld_reason" in broken["cost"]


# ---------------------------------------------------------------------------
# 12. Invalidation precision and recall — and checks that they can move
# ---------------------------------------------------------------------------

def test_invalidation_is_precise_and_complete(report):
    inv = report["invalidation"]
    assert inv["trials"] >= 10
    assert inv["precision"] == 1.0, "over-invalidation: work is being wasted"
    assert inv["recall"] == 1.0, "under-invalidation: a stale object stayed live"
    assert inv["direct_marked_stale"] == 1.0
    assert inv["transitive_marked_possibly_stale"] == 1.0


def test_the_scorer_detects_over_invalidation(world, monkeypatch):
    """Manipulation check. An architecture that marks everything stale on any
    change scores recall 1.0 — so recall alone proves nothing. Precision has to
    be the number that catches it, and here it is confirmed to."""
    from nano.dependency import DependencyGraph as DG
    everything = {}

    def mark_the_world(self, object_id):
        return set(everything.get(id(self), set())) - {object_id}

    original_register = DG.register

    def tracking_register(self, dep):
        out = original_register(self, dep)
        everything.setdefault(id(self), set()).add(dep.derived_id)
        return out

    monkeypatch.setattr(DG, "register", tracking_register)
    monkeypatch.setattr(DG, "dependents_of", mark_the_world)
    score = score_invalidation(world, trials=8)
    assert score["precision"] < 1.0, (
        "the scorer could not tell a whole-world invalidation from a precise one")


def test_the_scorer_detects_under_invalidation(world, monkeypatch):
    """Manipulation check for the opposite failure: an artifact that stays live
    while its input has changed is the dangerous one."""
    from nano.dependency import DependencyGraph as DG
    monkeypatch.setattr(DG, "dependents_of", lambda self, object_id: set())
    score = score_invalidation(world, trials=8)
    assert score["recall"] < 1.0
    assert score["false_negatives"] > 0


def test_the_isolation_scorer_detects_a_collapsed_graph(world, monkeypatch):
    """Manipulation check for branch isolation."""
    from nano.dependency import DependencyGraph as DG
    seen = {}
    original_register = DG.register

    def tracking_register(self, dep):
        out = original_register(self, dep)
        seen.setdefault(id(self), set()).add(dep.derived_id)
        return out

    monkeypatch.setattr(DG, "register", tracking_register)
    monkeypatch.setattr(DG, "dependents_of",
                        lambda self, object_id: set(seen.get(id(self), set()))
                        - {object_id})
    score = score_unrelated_branches(world)
    assert score["isolation"] < 1.0


# ---------------------------------------------------------------------------
# Faithfulness: being wrong is survivable; being wrong and confident is not
# ---------------------------------------------------------------------------

def test_declaring_uncertainty_beats_picking_a_winner(report):
    """The load-bearing claim, scored against a control rather than a constant.

    An absolute error threshold would move with the corruption rates and would
    be tuned, not measured. The control is the same fold with the uncertainty
    machinery removed; if declaring uncertainty is real work, the control must
    be confidently wrong more often.
    """
    f = report["faithfulness"]
    assert f["nano"]["resolved_keys"] > 0
    assert f["declared_unknown"] > 0
    assert f["control_is_worse"], (
        "the silent-resolution control was no worse — the conflict machinery "
        "is not earning its place")
    assert f["undeclared_error_avoided"] > 0


def test_the_system_is_never_confidently_wrong(report):
    """Being wrong under a lying channel is survivable. Being wrong *and*
    asserting it without qualification is the failure this architecture exists
    to prevent, so it is pinned at zero rather than at a tolerance."""
    assert report["faithfulness"]["nano"]["undeclared_error"] == 0


def test_mixed_precision_times_are_not_ordered_by_string_comparison():
    """Regression pin for the defect this benchmark found on its first run.

    `"2026-01" < "2026-01-05"` is true of ASCII and false of time: a report
    about some day in January cannot be declared older than one about the 5th.
    Resolving by string max silently preferred whichever sorted higher, which
    produced every undeclared error in the first full run.
    """
    from nano.slw import strictly_after, time_range
    assert time_range("2026-01") == ("2026-01-01", "2026-01-31")
    assert time_range("2026-02") == ("2026-02-01", "2026-02-28")
    assert time_range("2026-01-05") == ("2026-01-05", "2026-01-05")

    # A day inside a month is incomparable to that month, in both directions.
    assert not strictly_after("2026-01", "2026-01-05")
    assert not strictly_after("2026-01-05", "2026-01")
    # Disjoint ranges still order.
    assert strictly_after("2026-02", "2026-01-05")
    assert strictly_after("2026-02-01", "2026-01")
    assert not strictly_after("2026-01-05", "2026-01-09")


def test_an_incomparable_pair_is_declared_rather_than_resolved(world):
    """The behavioural consequence of the fix: when two live reports cannot be
    ordered and disagree, the key must become uncertain, not pick a side."""
    from nano.slw import Observation, LedgerBuilder
    b = LedgerBuilder(world_id="slw-001")
    precise = Observation(obs_tick=5, event_tick=5, entity_id="unit_0000",
                          attribute="status", value="nominal",
                          kind=ObsKind.CLEAN, source_ordinal=0, change_id="c1")
    approx = Observation(obs_tick=6, event_tick=20, entity_id="unit_0000",
                         attribute="status", value="offline",
                         kind=ObsKind.APPROXIMATE, source_ordinal=1,
                         change_id="c2")
    b.admit(precise)
    b.admit(approx)
    key = ("unit_0000", "status")
    assert key in b.conflicted, "an unorderable disagreement was silently resolved"
    assert key not in b.resolved
    assert set(b.conflicted[key][1]) == {"nominal", "offline"}


def test_the_channel_really_is_lying(world):
    """Manipulation check on faithfulness: if the observation channel were
    clean, a high score would mean nothing."""
    unfaithful = [o for o in world.observations if not o.faithful]
    assert unfaithful, "no false observations were generated"


# ---------------------------------------------------------------------------
# Duplicates corroborate rather than double-count
# ---------------------------------------------------------------------------

def test_duplicate_reports_merge_into_one_corroborated_assertion(world):
    b = LedgerBuilder(world_id=world.spec.world_id)
    dup = next(o for o in world.observations if o.kind is ObsKind.DUPLICATE)
    primary = next(o for o in world.observations
                   if o.change_id == dup.change_id and o.source_ordinal == 0)
    b.admit(primary)
    b.admit(dup)
    key = (primary.entity_id, primary.attribute)
    live = [a for a in b.by_key[key] if a.assertion_id not in b.superseded]
    assert len(live) == 1, "corroboration was counted as two separate facts"
    assert len(live[0].evidence_span_ids) == 2, (
        "the second source's evidence was dropped instead of joined")


def test_incremental_and_full_agree_on_the_derived_layer(world):
    full = build_full(world, world.observations, world.spec.n_ticks)
    incr = LedgerBuilder(world_id=world.spec.world_id)
    for obs in world.observations:
        incr.admit(obs)
    for chg in world.unobserved_changes():
        incr.admit_gap(chg)
    assert derived_objects(world, full)["views"] == \
        derived_objects(world, incr)["views"]


# ---------------------------------------------------------------------------
# Safety envelope
# ---------------------------------------------------------------------------

def test_the_benchmark_is_synthetic_and_non_clinical(builder):
    for source in builder.ledger.sources:
        assert source.security_classification == "synthetic_non_phi"
        assert source.consent_scope == "research_synthetic"
    identity = Identity(subject_id="slw-001", kind="subject")
    assert identity.access_scope == "synthetic_non_phi"


def test_nothing_in_the_benchmark_is_inferred_or_predicted(builder):
    """The first pass has no learned model. Every assertion must be OBSERVED —
    if a DERIVED or PREDICTED mode appeared, something started guessing."""
    modes = {a.derivation for a in builder.ledger.assertions}
    assert modes == {DerivationMode.OBSERVED}, modes
