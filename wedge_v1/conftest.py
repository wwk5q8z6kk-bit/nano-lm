"""Pytest isolation for local runtime artifacts."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_private_runtime_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise persistence without reading or writing owner runtime state."""
    import wedge_v1.coe.bind as bind_module
    import wedge_v1.cli as cli_module
    import wedge_v1.habit as habit_module
    import wedge_v1.owner_ready as owner_ready_module
    import wedge_v1.private_output as private_output_module
    import wedge_v1.study_lite as study_module

    root = Path(__file__).resolve().parent
    protected_names = {
        "results_owner_ready.json",
        "results_habit_session.json",
        "results_owner_habit.json",
        "results_habit_session.json",
        "results_review_state.json",
        "results_saved_questions.json",
    }
    protected_coe_root = (root / ".coe_runs").resolve()
    protected_private_root = (root / ".private").resolve()
    protected_owner_corpus = (root / "data" / "owner_corpus").resolve()
    protected_owner_tasks = (root / "data" / "owner_tasks").resolve()
    original_open = Path.open

    def protected_open(path: Path, mode: str = "r", *args, **kwargs):
        resolved = path.resolve()
        writes = any(flag in mode for flag in ("w", "a", "x", "+"))
        owner_output = (
            resolved.parent == root
            and (
                resolved.name.startswith("results_owner_")
                or resolved.name in protected_names
            )
        )
        try:
            owner_corpus_output = resolved.is_relative_to(protected_owner_corpus)
        except AttributeError:  # pragma: no cover - Python < 3.9 compatibility
            owner_corpus_output = protected_owner_corpus in resolved.parents
        try:
            owner_task_output = resolved.is_relative_to(protected_owner_tasks)
        except AttributeError:  # pragma: no cover - Python < 3.9 compatibility
            owner_task_output = protected_owner_tasks in resolved.parents
        try:
            coe_output = resolved.is_relative_to(protected_coe_root)
        except AttributeError:  # pragma: no cover - Python < 3.9 compatibility
            coe_output = protected_coe_root in resolved.parents or resolved == protected_coe_root
        try:
            private_output = resolved.is_relative_to(protected_private_root)
        except AttributeError:  # pragma: no cover - Python < 3.9 compatibility
            private_output = (
                protected_private_root in resolved.parents
                or resolved == protected_private_root
            )
        if writes and (
            owner_output
            or owner_corpus_output
            or owner_task_output
            or coe_output
            or private_output
        ):
            raise AssertionError(f"test attempted to write protected owner state: {resolved}")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", protected_open)
    monkeypatch.setattr(bind_module, "DEFAULT_RECORD_DIR", tmp_path / "coe_runs")

    task_root = tmp_path / "owner_tasks"
    monkeypatch.setattr(
        private_output_module, "PRIVATE_EXPORT_ROOT", tmp_path / "private_exports"
    )
    monkeypatch.setattr(private_output_module, "PRIVATE_TASK_ROOT", task_root)
    if hasattr(study_module, "PRIVATE_TASK_ROOT"):
        monkeypatch.setattr(study_module, "PRIVATE_TASK_ROOT", task_root)
    # Keep study_lite DEFAULT_TASKS pointing at tracked questions-v1.json
    if hasattr(owner_ready_module, "PRIVATE_TASK_ROOT"):
        monkeypatch.setattr(owner_ready_module, "PRIVATE_TASK_ROOT", task_root)
    if hasattr(owner_ready_module, "DEFAULT_PRIVATE_TASKS"):
        monkeypatch.setattr(
            owner_ready_module, "DEFAULT_PRIVATE_TASKS", task_root / "questions-v1.json"
        )

    monkeypatch.setattr(habit_module, "HABIT_PATH", tmp_path / "results_owner_habit.json")
    if hasattr(habit_module, "SAVED_QUESTIONS"):
        monkeypatch.setattr(habit_module, "SAVED_QUESTIONS", tmp_path / "results_saved_questions.json")
    monkeypatch.setattr(habit_module, "SESSION_PATH", tmp_path / "results_habit_session.json")
    if hasattr(habit_module, "REVIEW_PATH"):
        monkeypatch.setattr(habit_module, "REVIEW_PATH", tmp_path / "results_review_state.json")
    monkeypatch.setattr(owner_ready_module, "READY_OUT", tmp_path / "results_owner_ready.json")
    if hasattr(owner_ready_module, "REVIEW_PATH"):
        monkeypatch.setattr(owner_ready_module, "REVIEW_PATH", tmp_path / "results_review_state.json")

    habit_path = tmp_path / "results_owner_habit.json"

    def record_test_habit(kind: str, *, note: str = "") -> dict:
        return habit_module.record(kind, note=note, path=habit_path)

    # cli.py imports record directly, so changing HABIT_PATH would not affect
    # its bound default argument. Replace the imported callable instead.
    monkeypatch.setattr(cli_module, "habit_record", record_test_habit)
