"""Pins for the capability registry.

The registry's value is that "is a capability missing?" and "is a status claim
backed?" are mechanical. These tests are that mechanism.
"""

from __future__ import annotations

import pytest

from nano.capabilities import (
    CAPABILITIES, REQUIRED_DOMAINS, Capability, Stage, Status, by_domain, coverage,
)


def test_all_required_domains_are_covered() -> None:
    missing = coverage()["missing_domains"]
    assert not missing, f"capability domains with no entry: {missing}"


def test_capability_ids_are_unique() -> None:
    ids = [c.capability_id for c in CAPABILITIES]
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f"duplicate capability ids: {dupes}"


def test_every_capability_names_failure_modes() -> None:
    """A capability with no named failure mode cannot be evaluated."""
    for c in CAPABILITIES:
        assert c.failure_modes, c.capability_id


def test_every_capability_names_a_benchmark() -> None:
    for c in CAPABILITIES:
        assert c.evaluation_benchmark, c.capability_id


def test_status_claims_are_backed_by_evidence() -> None:
    """IMPLEMENTED and PARTIAL require a pointer; PROPOSED must not have one."""
    for c in CAPABILITIES:
        if c.status in (Status.IMPLEMENTED, Status.PARTIAL):
            assert c.evidence, f"{c.capability_id} claims {c.status.value} unbacked"
        if c.status is Status.PROPOSED:
            assert not c.evidence, (
                f"{c.capability_id} is PROPOSED but cites evidence — citing "
                "evidence for unbuilt work is how a plan becomes a claim")


def test_unbacked_status_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="requires an evidence pointer"):
        Capability(
            capability_id="X", domain="Perception", capability="c",
            internal_representation="r", module="m", inputs=(), outputs=(),
            memory_requirements="", tools=(), training_objective="",
            evaluation_benchmark="b", failure_modes=("f",),
            implementation_stage=Stage.CORE, status=Status.IMPLEMENTED)


def test_capability_without_failure_modes_is_rejected() -> None:
    with pytest.raises(ValueError, match="failure_modes required"):
        Capability(
            capability_id="Y", domain="Perception", capability="c",
            internal_representation="r", module="m", inputs=(), outputs=(),
            memory_requirements="", tools=(), training_objective="",
            evaluation_benchmark="b", failure_modes=(),
            implementation_stage=Stage.CORE, status=Status.PROPOSED)


def test_implemented_capabilities_point_at_real_paths() -> None:
    """An IMPLEMENTED pointer must name a file that exists in this repo."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    for c in CAPABILITIES:
        if c.status is not Status.IMPLEMENTED:
            continue
        # Strip the ::symbol suffix FIRST, then keep path-looking tokens.
        # An earlier version tested for "/" and "::" in one expression, whose
        # precedence excluded every `pkg/mod.py::Symbol` pointer in the registry.
        paths = []
        for tok in c.evidence.replace(";", " ").replace(",", " ").split():
            head = tok.split("::")[0].strip("().,;")
            if "/" in head and head.endswith(".py") or head.endswith(".json"):
                paths.append(head)
        assert paths, f"{c.capability_id}: evidence names no path: {c.evidence!r}"
        assert any((root / p).exists() for p in paths), (
            f"{c.capability_id}: no existing path among {paths}")


def test_core_stage_capabilities_are_domain_independent() -> None:
    """Stage CORE must not name a clinical concept — the capability/knowledge
    boundary applied to the registry itself."""
    clinical = ("patient", "clinical", "medical", "diagnos", "medication")
    for c in CAPABILITIES:
        if c.implementation_stage is not Stage.CORE:
            continue
        blob = f"{c.capability} {c.internal_representation}".lower()
        hits = [w for w in clinical if w in blob]
        assert not hits, (
            f"{c.capability_id} is CORE but its definition names {hits} — "
            "move it to a clinical stage or restate it domain-independently")


def test_registry_is_honest_about_how_little_is_built() -> None:
    """Guard against status inflation: most of this is not built, and the
    registry must keep saying so until code changes."""
    cv = coverage()["by_status"]
    built = cv["IMPLEMENTED"]
    assert built < cv["ABSENT"] + cv["PROPOSED"], (
        "IMPLEMENTED now outnumbers unbuilt capabilities — if that is real, "
        "update this test deliberately rather than letting it pass silently")
