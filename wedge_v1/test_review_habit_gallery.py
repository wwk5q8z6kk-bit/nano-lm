"""P3 review / habit / gallery pins (fixture corpora only)."""
from __future__ import annotations

import io
from pathlib import Path

from wedge_v1.failure_gallery import FINE_BUCKETS, build_gallery, classify_fine, gallery_to_markdown
from wedge_v1.habit import format_session_md, save_question, session
from wedge_v1.owner_ready import check
from wedge_v1.review import (
    LABELS,
    batch_label,
    build_card,
    cards_from_task_pack,
    interactive_review,
    load_state,
)
from wedge_v1.run_owner_dogfood import DEFAULT_TASKS, FIXTURE_CORPUS


def test_labels_stable():
    assert "USEFUL" in LABELS
    assert "OVER_ABSTENTION" in LABELS
    assert len(LABELS) == 9


def test_build_card_and_batch_label(tmp_path: Path):
    import wedge_v1.review as R

    R.REVIEW_PATH = tmp_path / "review.json"
    card = build_card(
        "How long before cached entries expire?",
        corpus=FIXTURE_CORPUS,
        mode="ask",
        task_id="T_TTL",
    )
    assert card["query"]
    assert card["answer_status"] in {"SUPPORTED", "CONTRADICTED", "ABSTAIN"}
    state = {"schema": "nano-lm.wedge_v1.review.v1", "cards": {}, "labels": {}}
    batch_label(state, [card], ["T_TTL:USEFUL"])
    assert state["labels"][card["id"]] == "USEFUL"
    assert R.REVIEW_PATH.is_file()


def test_interactive_review_stdin(tmp_path: Path):
    import wedge_v1.review as R

    R.REVIEW_PATH = tmp_path / "review.json"
    cards = cards_from_task_pack(DEFAULT_TASKS, FIXTURE_CORPUS, limit=2)
    state = load_state()
    stdin = io.StringIO("u\nq\n")
    stdout = io.StringIO()
    interactive_review(cards, state, stdin=stdin, stdout=stdout)
    assert state.get("labels") or "labeled" in stdout.getvalue()


def test_habit_session_fixture(tmp_path: Path):
    habit_path = tmp_path / "habit.json"
    saved_path = tmp_path / "saved.json"
    session_path = tmp_path / "session.json"
    sess = session(
        FIXTURE_CORPUS,
        habit_path=habit_path,
        saved_path=saved_path,
        session_path=session_path,
    )
    assert sess["ingest"]["n_docs"] >= 1
    assert sess["next_action"]
    assert sess.get("recent_documents") is not None
    md = format_session_md(sess)
    assert "habit session" in md
    save_question("How long before cached entries expire?", path=saved_path)


def test_gallery_fine_buckets_always_listed():
    g = build_gallery(
        dogfood={
            "n_tasks": 2,
            "n_ok": 1,
            "accuracy": 0.5,
            "corpus": str(FIXTURE_CORPUS),
            "rows": [
                {
                    "id": "a",
                    "ok": True,
                    "got_status": "SUPPORTED",
                    "expect_status": ["SUPPORTED"],
                    "query": "x",
                },
                {
                    "id": "b",
                    "ok": False,
                    "got_status": "ABSTAIN",
                    "expect_status": ["SUPPORTED"],
                    "ok_status": False,
                    "ok_needles": True,
                    "fail_kind": "over_abstain",
                    "query": "y",
                },
            ],
        },
        path=Path("synthetic"),
    )
    for b in FINE_BUCKETS:
        assert b in g["fine_buckets"]
        assert b in g["fine_counts"]
    assert g["fine_counts"]["over_abstention"] >= 1
    assert g["fine_counts"]["ok_supported"] >= 1
    assert "unobserved" in gallery_to_markdown(g)


def test_classify_fine_contradiction():
    assert (
        classify_fine(
            {"ok": True, "got_status": "CONTRADICTED", "expect_status": ["CONTRADICTED"]}
        )
        == "multi_document_contradiction"
    )


def test_owner_ready_demo():
    rep = check(demo=True)
    assert rep["exists"]
    assert rep["n_docs"] >= 1
    assert "canonical_command" in rep
    assert "OWNER_CORPUS_PENDING" in rep["blockers"]
    assert rep["ready_for_private_run"] is False
