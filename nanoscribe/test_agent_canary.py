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


def test_axis_polarity_covers_every_score_axis() -> None:
    """Ranking donors per capability requires knowing each axis's direction of merit.

    unnecessary_call_rate / steps_to_resolution / cost are penalties where LOWER
    wins; a winner-picker that maximises uniformly would crown the worst teacher
    on those three.
    """
    from nanoscribe.agent_canary import AXIS_POLARITY, SCORE_AXES, better_of

    assert set(AXIS_POLARITY) == set(SCORE_AXES)
    assert AXIS_POLARITY["unnecessary_call_rate"] == "lower_better"
    assert AXIS_POLARITY["tool_selection"] == "higher_better"
    assert better_of("tool_selection", 0.2, 0.8) == 0.8
    assert better_of("unnecessary_call_rate", 0.2, 0.8) == 0.2


def test_parse_action_marks_unrecognised_output_unparsed() -> None:
    """A response with no action token must be UNPARSED, not a wrong action.

    _parse_action previously fell back to the first whitespace token, so a model
    emitting JSON scored the action '{' and was graded as a wrong answer. That
    conflates transport failure with capability — the exact defect that made the
    first canary scoreboard meaningless.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from agent_canary_bench import _parse_action, strip_thinking

    assert _parse_action("SEARCH") == "SEARCH"
    assert _parse_action('{\n  "foo": 1\n}') == ""
    assert _parse_action("gibberish here") == ""
    assert _parse_action("") == ""
    # inline reasoning must be stripped before parsing (Qwen3/GLM thinking mode)
    assert _parse_action("<think>deliberating</think>\nSTOP") == "STOP"
    assert strip_thinking("<think>cut off mid-thought") == ""


def test_long_horizon_prompts_state_execution_state() -> None:
    """Regression pin for the retracted 'no viable long-horizon donor' finding.

    v1 prompts named a trajectory ('read then verify span') while the harness asks
    for a single next action with no state, so the gold presumed hidden state and
    both teachers scored 0/12. Prompts for these families must describe what has
    already happened.
    """
    tasks = load_agent_canary_tasks()
    long_horizon = {TaskFamily.MULTI_TOOL, TaskFamily.PREMATURE_STOP, TaskFamily.REPEATED_CALLS}
    stateful = [t for t in tasks if t.family in long_horizon]
    assert len(stateful) == 12
    for task in stateful:
        assert len(task.prompt) > 60, f"{task.task_id} prompt too terse to determine gold"
        assert any(
            marker in task.prompt.lower()
            for marker in ("already", "you have", "have not", "no tools are available")
        ), f"{task.task_id} does not state prior execution state"
