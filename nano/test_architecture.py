"""Pins for the five-specification separation and the extensibility guarantee."""
from __future__ import annotations

import nano.architecture as arch
from nano.architecture import (
    EXTENSIBLE_SETS, INVARIANT_SETS, ExperimentalObject, IntegrationMaturity,
    Layer, LearningLevel, ProvingStage, Spec,
)


def test_five_specifications_exist_and_are_named() -> None:
    assert {s.value for s in Spec} == {
        "ontology", "cognitive", "neural", "learning", "progression"}


def test_layers_and_proving_stages_do_not_share_members() -> None:
    """Spec 2 (architecture) and spec 5 (progression) must not be one field.

    This is the overload the registry previously carried: a single `Stage`
    enum holding both CORE (a layer) and A/B/C (a proving order).
    """
    assert not ({l.value for l in Layer} & {p.value for p in ProvingStage})


def test_progression_is_not_the_architecture() -> None:
    """No proving stage may be named as though it were a layer, or vice versa."""
    layer_words = {"ontology", "evidence", "memory", "verification"}
    for p in ProvingStage:
        stem = p.value.split("_", 1)[1]
        assert stem not in {l.value for l in Layer}, (
            f"{p.value} collides with a Layer name — progression is not architecture")


def test_almost_nothing_is_invariant() -> None:
    """An invariant is a promise. Promises are costly, so they stay few.

    If this ratio inverts, someone has been promoting current understanding
    into permanent commitments.
    """
    assert len(INVARIANT_SETS) < len(EXTENSIBLE_SETS)
    assert len(INVARIANT_SETS) <= 3, (
        "invariant set grew — each addition must be a deliberate promise")


def test_neural_mechanisms_are_hypotheses_not_commitments() -> None:
    """Spec 3 must never appear in spec 2. No layer may name a mechanism."""
    mechanisms = {"transformer", "moe", "mamba", "ssm", "recurrent", "attention"}
    for l in Layer:
        hits = [m for m in mechanisms if m in l.value.lower()]
        assert not hits, f"Layer {l.value} names a neural mechanism {hits}"
    assert "NEURAL_CANDIDATES" in EXTENSIBLE_SETS


def test_extensibility_declaration_is_present_and_explicit() -> None:
    d = arch.EXTENSIBILITY_DECLARATION.lower()
    assert "illustrative extensible sets" in d
    assert "not to bound" in d


def test_integration_maturity_is_documented_as_not_modules() -> None:
    doc = IntegrationMaturity.__doc__ or ""
    assert "not module boundaries" in doc.lower()
    assert len(list(IntegrationMaturity)) == 9


def test_three_experimental_object_kinds_are_distinguished() -> None:
    """Conflating them is how a mechanism result is misread as a capability result."""
    assert len(list(ExperimentalObject)) == 3
    note = arch.MECHANISM_MODEL_NOTE.lower()
    assert "category error" in note
    assert "below the task capability" in note


def test_efficiency_is_measured_over_the_system() -> None:
    e = arch.EFFICIENCY_OBJECTIVE.lower()
    assert "active compute" in e and "human review" in e
    assert "parameter count" not in e


def test_canonical_chain_direction_is_invariant() -> None:
    chain = arch.CANONICAL_CHAIN
    assert chain[0] == "WORLD" and chain[-1] == "ARTIFACT"
    assert "CANONICAL_CHAIN" in INVARIANT_SETS
