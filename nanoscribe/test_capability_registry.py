# Capability registry tests.
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoscribe.capabilities import CapabilityId, CapabilityStatus, get_capability, list_capabilities
from nanoscribe.capabilities.registry import capability_for_tool, SUBMIT_CANDIDATE_ATOMS_TOOL


def test_registry_contains_core_capabilities() -> None:
    scribe = get_capability(CapabilityId.SCRIBE)
    summarize = get_capability(CapabilityId.SUMMARIZE)
    table = get_capability(CapabilityId.TABLE)
    assert scribe.status is CapabilityStatus.ACTIVE
    assert summarize.submit_tool_name == "submit_summary"
    assert table.schema_version == "nano.table.v0"


def test_tool_name_lookup() -> None:
    spec = capability_for_tool(SUBMIT_CANDIDATE_ATOMS_TOOL)
    assert spec is not None
    assert spec.capability_id is CapabilityId.SCRIBE


def test_list_active_capabilities() -> None:
    active = list_capabilities(status=CapabilityStatus.ACTIVE)
    ids = {spec.capability_id for spec in active}
    assert CapabilityId.SCRIBE in ids
    assert CapabilityId.SUMMARIZE in ids
    assert CapabilityId.TABLE in ids
    assert CapabilityId.CHART not in ids
