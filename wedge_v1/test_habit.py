"""Habit stub pins."""
from __future__ import annotations

from pathlib import Path

from wedge_v1.habit import record, weekly_summary


def test_habit_record_and_summary(tmp_path: Path):
    path = tmp_path / "habit.json"
    record("ask", path=path)
    record("compare", path=path)
    s = weekly_summary(path=path)
    assert s["n_events"] == 2
    assert s["by_kind"]["ask"] == 1
    assert s["by_kind"]["compare"] == 1
