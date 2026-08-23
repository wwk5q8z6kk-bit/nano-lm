"""Trajectory Compiler — multi-teacher rollouts to verified training examples."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    step_index: int
    state: Mapping[str, Any]
    action: str
    tool_name: str | None
    tool_args: Mapping[str, Any]
    observation: str | None
    teacher_id: str
    cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "state": dict(self.state),
            "action": self.action,
            "tool_name": self.tool_name,
            "tool_args": dict(self.tool_args),
            "observation": self.observation,
            "teacher_id": self.teacher_id,
            "cost_usd": self.cost_usd,
        }


@dataclass(frozen=True, slots=True)
class TeacherRollout:
    task_id: str
    teacher_id: str
    steps: tuple[TrajectoryStep, ...]
    final_outcome: str | None = None
    total_cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "teacher_id": self.teacher_id,
            "steps": [s.to_dict() for s in self.steps],
            "final_outcome": self.final_outcome,
            "total_cost_usd": self.total_cost_usd,
        }


@dataclass(frozen=True, slots=True)
class NormalizedStep:
    step_index: int
    state_hash: str
    action: str
    tool_name: str | None
    tool_args: Mapping[str, Any]
    observation: str | None
    teacher_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "state_hash": self.state_hash,
            "action": self.action,
            "tool_name": self.tool_name,
            "tool_args": dict(self.tool_args),
            "observation": self.observation,
            "teacher_id": self.teacher_id,
        }


@dataclass(frozen=True, slots=True)
class NormalizedTrajectory:
    task_id: str
    teacher_id: str
    steps: tuple[NormalizedStep, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "teacher_id": self.teacher_id,
            "steps": [s.to_dict() for s in self.steps],
        }


@dataclass(frozen=True, slots=True)
class StepScore:
    step_index: int
    teacher_id: str
    scores: Mapping[str, float]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "teacher_id": self.teacher_id,
            "scores": dict(self.scores),
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class VerifiedExample:
    task_id: str
    selected_step: NormalizedStep
    step_score: StepScore
    transfer_type: str
    ablated_teachers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "selected_step": self.selected_step.to_dict(),
            "step_score": self.step_score.to_dict(),
            "transfer_type": self.transfer_type,
            "ablated_teachers": list(self.ablated_teachers),
        }


StepEvalHook = Callable[[NormalizedStep, Mapping[str, Any]], StepScore]


def _state_hash(state: Mapping[str, Any]) -> str:
    items = sorted((str(k), str(v)) for k, v in state.items())
    return str(hash(tuple(items)))


class TrajectoryNormalizer:
    """Stage 1: normalize heterogeneous teacher rollouts to comparable steps."""

    def normalize(self, rollout: TeacherRollout) -> NormalizedTrajectory:
        steps = tuple(
            NormalizedStep(
                step_index=step.step_index,
                state_hash=_state_hash(step.state),
                action=step.action,
                tool_name=step.tool_name,
                tool_args=step.tool_args,
                observation=step.observation,
                teacher_id=step.teacher_id,
            )
            for step in rollout.steps
        )
        return NormalizedTrajectory(
            task_id=rollout.task_id,
            teacher_id=rollout.teacher_id,
            steps=steps,
        )

    def normalize_many(self, rollouts: Sequence[TeacherRollout]) -> list[NormalizedTrajectory]:
        return [self.normalize(r) for r in rollouts]


class PerStepEvaluator:
    """Stage 2: score each step against task gold + capability axes."""

    def __init__(self, hooks: Sequence[StepEvalHook] | None = None) -> None:
        self._hooks = list(hooks or [])

    def evaluate(
        self,
        trajectory: NormalizedTrajectory,
        task: Mapping[str, Any],
        *,
        default_hook: StepEvalHook | None = None,
    ) -> list[StepScore]:
        hook = default_hook or self._default_hook
        hooks = [*self._hooks, hook]
        scores: list[StepScore] = []
        for step in trajectory.steps:
            merged: dict[str, float] = {}
            passed = True
            for h in hooks:
                result = h(step, task)
                merged.update(result.scores)
                passed = passed and result.passed
            scores.append(
                StepScore(
                    step_index=step.step_index,
                    teacher_id=step.teacher_id,
                    scores=merged,
                    passed=passed,
                )
            )
        return scores

    @staticmethod
    def _default_hook(step: NormalizedStep, task: Mapping[str, Any]) -> StepScore:
        gold = str(task.get("gold_action", ""))
        action_ok = step.action == gold or gold == "RECOVERY"
        tool_ok = not task.get("expect_abstain") or step.action == "ABSTAIN"
        outcome = 1.0 if action_ok and tool_ok else 0.0
        return StepScore(
            step_index=step.step_index,
            teacher_id=step.teacher_id,
            scores={"outcome": outcome, "stop_accuracy": outcome},
            passed=outcome >= 1.0,
        )


class BestStepSelector:
    """Stage 3: pick best verified step per task across teachers."""

    def select(
        self,
        trajectories: Sequence[NormalizedTrajectory],
        scores_by_teacher: Mapping[str, Sequence[StepScore]],
        *,
        transfer_type: str = "behavior",
    ) -> list[VerifiedExample]:
        by_task: dict[str, list[tuple[NormalizedStep, StepScore]]] = {}
        for traj in trajectories:
            teacher_scores = scores_by_teacher.get(traj.teacher_id, ())
            score_map = {s.step_index: s for s in teacher_scores}
            for step in traj.steps:
                score = score_map.get(step.step_index)
                if score is None or not score.passed:
                    continue
                by_task.setdefault(traj.task_id, []).append((step, score))

        examples: list[VerifiedExample] = []
        for task_id, candidates in by_task.items():
            if not candidates:
                continue
            best_step, best_score = max(
                candidates,
                key=lambda pair: sum(pair[1].scores.values()),
            )
            ablated = tuple(
                sorted({s.teacher_id for s, _ in candidates if s.teacher_id != best_step.teacher_id})
            )
            examples.append(
                VerifiedExample(
                    task_id=task_id,
                    selected_step=best_step,
                    step_score=best_score,
                    transfer_type=transfer_type,
                    ablated_teachers=ablated,
                )
            )
        return examples


@dataclass
class TrajectoryCompiler:
    """End-to-end: rollouts → normalize → per-step eval → best-step selection."""

    normalizer: TrajectoryNormalizer = field(default_factory=TrajectoryNormalizer)
    evaluator: PerStepEvaluator = field(default_factory=PerStepEvaluator)
    selector: BestStepSelector = field(default_factory=BestStepSelector)

    def compile(
        self,
        rollouts: Sequence[TeacherRollout],
        tasks_by_id: Mapping[str, Mapping[str, Any]],
        *,
        transfer_type: str = "behavior",
    ) -> list[VerifiedExample]:
        normalized = self.normalizer.normalize_many(rollouts)
        scores_by_teacher: dict[str, list[StepScore]] = {}
        for traj in normalized:
            task = tasks_by_id.get(traj.task_id, {})
            scores = self.evaluator.evaluate(traj, task)
            scores_by_teacher.setdefault(traj.teacher_id, []).extend(scores)
        return self.selector.select(
            normalized,
            scores_by_teacher,
            transfer_type=transfer_type,
        )
