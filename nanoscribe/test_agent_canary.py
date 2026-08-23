# Agent canary suite pins.
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoscribe.agent_canary import (
    AGENT_CANARY_REVISION,
    TaskFamily,
    agent_canary_manifest,
    build_agent_canary_tasks,
    export_agent_canary_json,
    load_agent_canary_tasks,
)


def test_agent_canary_task_count() -> None:
    tasks = build_agent_canary_tasks()
    assert 32 <= len(tasks) <= 64
    assert len(tasks) == 48


def test_all_families_represented() -> None:
    tasks = build_agent_canary_tasks()
    families = {t.family for t in tasks}
    assert families == set(TaskFamily)
    assert all(sum(1 for t in tasks if t.family == f) == 4 for f in TaskFamily)


def test_export_and_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / f"{AGENT_CANARY_REVISION}.json"
    export_agent_canary_json(path)
    loaded = load_agent_canary_tasks(path)
    assert len(loaded) == 48
    assert loaded[0].task_id.startswith("agent-")


def test_manifest_schema() -> None:
    manifest = agent_canary_manifest()
    assert manifest["schema"] == "nano.campaign.agent_canary.v1"
    assert manifest["revision"] == AGENT_CANARY_REVISION
    assert manifest["n_tasks"] == 48
    assert "tool_selection" in manifest["score_axes"]
