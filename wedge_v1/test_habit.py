"""Habit stub pins."""
from __future__ import annotations

from pathlib import Path

from wedge_v1.habit import (
    format_saved_list_md,
    record,
    save_question,
    saved_question_status,
    weekly_summary,
)


def test_habit_record_and_summary(tmp_path: Path):
    path = tmp_path / "habit.json"
    record("ask", path=path)
    record("compare", path=path)
    s = weekly_summary(path=path)
    assert s["n_events"] == 2
    assert s["by_kind"]["ask"] == 1
    assert s["by_kind"]["compare"] == 1


def test_saved_list_includes_scope(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note_a.md").write_text("Cache TTL as 300 seconds.\n", encoding="utf-8")
    (corpus / "note_b.md").write_text("unrelated\n", encoding="utf-8")
    saved_path = tmp_path / "saved.json"
    save_question(
        "How long before cached entries expire?",
        corpus=corpus,
        doc_ids=["note_a"],
        path=saved_path,
    )
    rows = saved_question_status(corpus, path=saved_path)
    assert len(rows) == 1
    assert rows[0]["doc_ids"] == ["note_a"]
    assert rows[0]["task_id"]
    md = format_saved_list_md(rows)
    assert "note_a" in md
    assert rows[0]["task_id"] in md
