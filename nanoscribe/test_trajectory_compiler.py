# Trajectory compiler pins.
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoscribe.trajectory_compiler import (
    BestStepSelector,
    PerStepEvaluator,
    TeacherRollout,
    TrajectoryCompiler,
    TrajectoryNormalizer,
    TrajectoryStep,
)


def _rollout(task_id: str, teacher: str, action: str) -> TeacherRollout:
    step = TrajectoryStep(
        step_index=0,
        state={"encounter_id": task_id},
        action=action,
        tool_name=None,
        tool_args={},
        observation="ok",
        teacher_id=teacher,
        cost_usd=0.01,
    )
    return TeacherRollout(task_id=task_id, teacher_id=teacher, steps=(step,), total_cost_usd=0.01)


def test_normalizer_hashes_state() -> None:
    norm = TrajectoryNormalizer().normalize(_rollout("t1", "teacher_a", "SEARCH"))
    assert norm.steps[0].state_hash
    assert norm.steps[0].action == "SEARCH"


def test_compiler_selects_best_teacher_step() -> None:
    task = {"task_id": "t1", "gold_action": "SEARCH", "expect_abstain": False}
    rollouts = [
        _rollout("t1", "teacher_a", "TOOL_NONE"),
        _rollout("t1", "teacher_b", "SEARCH"),
    ]
    examples = TrajectoryCompiler().compile(rollouts, {"t1": task})
    assert len(examples) == 1
    assert examples[0].selected_step.teacher_id == "teacher_b"
    assert examples[0].transfer_type == "behavior"


def test_per_step_evaluator_abstain_gate() -> None:
    step = TrajectoryNormalizer().normalize(_rollout("t2", "teacher_c", "ABSTAIN")).steps[0]
    task = {"gold_action": "ABSTAIN", "expect_abstain": True}
    scores = PerStepEvaluator().evaluate(
        TrajectoryNormalizer().normalize(_rollout("t2", "teacher_c", "ABSTAIN")),
        task,
    )
    assert scores[0].passed is True


def test_best_step_selector_empty_when_all_fail() -> None:
    traj = TrajectoryNormalizer().normalize(_rollout("t3", "teacher_x", "TOOL_NONE"))
    picked = BestStepSelector().select([traj], {"teacher_x": []})
    assert picked == []
