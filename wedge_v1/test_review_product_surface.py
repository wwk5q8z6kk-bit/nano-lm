"""Review-card completeness, durability, correction, and privacy regressions."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

import wedge_v1.review as review
from wedge_v1.cli import main as cli_main
from wedge_v1.private_output import require_private_output


def _result() -> dict:
    return {
        "answer_status": "SUPPORTED",
        "claims": [
            {
                "task_id": "FIRST",
                "doc_id": "alpha",
                "value": "complete answer value",
                "status": "PRESENT",
                "evidence": [
                    {
                        "doc_id": "alpha",
                        "start": 1,
                        "end": 8,
                        "text": "first evidence row",
                        "relation": "EXACTLY_STATED",
                    },
                    {
                        "doc_id": "alpha",
                        "start": 20,
                        "end": 28,
                        "text": "second evidence row with an untruncated ending ZXQ",
                        "relation": "EXACTLY_STATED",
                    },
                ],
            },
            {
                "task_id": "SECOND",
                "doc_id": "beta",
                "value": {"field": "ttl", "values": {"alpha": 300, "beta": 600}},
                "status": "DISPUTED",
                "evidence": [
                    {
                        "doc_id": "beta",
                        "start": 4,
                        "end": 7,
                        "text": "600",
                        "relation": "EXACTLY_STATED",
                    }
                ],
            },
        ],
        "hits": [
            {"doc_id": "alpha", "start": 1, "end": 8, "text": "hit one"},
            {"doc_id": "beta", "start": 4, "end": 7, "text": "hit two"},
        ],
        "solver_path": ["fixture"],
        "latency_s": 0.01,
        "coe_audit": {"ok": True},
    }


def test_card_persists_and_renders_every_claim_evidence_and_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "alpha.md").write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(review, "ask", lambda *args, **kwargs: _result())

    card = review.build_card("question", corpus=corpus, persist_coe=False)
    rendered = review.format_card(card)

    assert len(card["claims"]) == 2
    assert [len(claim["evidence"]) for claim in card["claims"]] == [2, 1]
    assert len(card["hits"]) == 2
    assert "claims (2):" in rendered
    assert "claim 1:" in rendered and "claim 2:" in rendered
    assert "first evidence row" in rendered
    assert "second evidence row with an untruncated ending ZXQ" in rendered
    assert '"alpha": 300' in rendered and '"beta": 600' in rendered
    assert "hits (2):" in rendered
    assert "hit one" in rendered and "hit two" in rendered


def test_find_card_materializes_and_renders_every_exact_hit(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "notes.md").write_text(
        "needle before needle after needle", encoding="utf-8"
    )

    card = review.build_card(
        "needle", corpus=corpus, mode="find", persist_coe=False
    )
    rendered = review.format_card(card)

    assert card["n_claims"] == 3
    assert card["n_hits"] == 3
    assert len(card["hits"]) == 3
    assert "claims (3):" in rendered
    assert "hits (3):" in rendered
    assert rendered.count("  hit ") == 3


class _InterruptAfterOneLabel:
    def __init__(self) -> None:
        self.calls = 0

    def readline(self) -> str:
        self.calls += 1
        if self.calls == 1:
            return "u\n"
        raise KeyboardInterrupt


def test_interactive_review_saves_each_label_for_resume(tmp_path: Path):
    cards = [
        {**review.build_card("alpha", corpus=tmp_path, persist_coe=False), "task_id": "A"},
        {**review.build_card("beta", corpus=tmp_path, persist_coe=False), "task_id": "B"},
    ]
    state = {"schema": "nano-lm.wedge_v1.review.v1", "cards": {}, "labels": {}}
    state_path = tmp_path / "review.json"

    with pytest.raises(KeyboardInterrupt):
        review.interactive_review(
            cards,
            state,
            stdin=_InterruptAfterOneLabel(),
            stdout=io.StringIO(),
            state_path=state_path,
            reviewer_kind="owner",
        )

    persisted = review.load_state(state_path)
    assert persisted["labels"] == {cards[0]["id"]: "USEFUL"}
    assert persisted["active_card_ids"] == [card["id"] for card in cards]


def test_relabel_and_undo_are_append_only_audited(tmp_path: Path):
    card = review.build_card("question", corpus=tmp_path, persist_coe=False)
    state = {"schema": "nano-lm.wedge_v1.review.v1", "cards": {}, "labels": {}}
    review.snapshot_review_cards(state, [card])

    review.apply_label(state, card, "USEFUL", reviewer_kind="owner", review_elapsed_s=1)
    review.apply_label(
        state,
        card,
        "NOT_USEFUL",
        failure_reason="answer is incomplete",
        suggested_correction="include the second source",
        reviewer_kind="owner",
        review_elapsed_s=2,
    )
    review.undo_label(state, [card], card["task_id"], reviewer_kind="owner")

    assert card["id"] not in state["labels"]
    assert [event["action"] for event in state["audit_log"]] == [
        "LABEL",
        "RELABEL",
        "UNDO",
    ]
    assert state["audit_log"][1]["from_label"] == "USEFUL"
    assert state["audit_log"][2]["from_label"] == "NOT_USEFUL"


def test_private_tasks_need_corpus_and_local_state(tmp_path: Path):
    tasks = tmp_path / "tasks.json"
    tasks.write_text('{"tasks": []}', encoding="utf-8")

    with pytest.raises(SystemExit):
        cli_main(["review", "--tasks", str(tasks), "--interactive"])
    with pytest.raises(SystemExit):
        cli_main(["review", "--demo", "--tasks", str(tasks), "--interactive"])

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    with pytest.raises(SystemExit):
        cli_main(
            [
                "review",
                "--corpus",
                str(corpus),
                "--tasks",
                str(tasks),
                "--interactive",
            ]
        )


def test_review_state_rejects_commit_visible_and_symlinked_paths(tmp_path: Path):
    state = {"schema": "nano-lm.wedge_v1.review.v1", "cards": {}, "labels": {}}
    unsafe = review.ROOT.parent / "private-review-leak-test.json"
    assert not unsafe.exists()
    with pytest.raises(ValueError, match="ignored private namespace"):
        review.save_state(state, unsafe)
    assert not unsafe.exists()

    repo_link = tmp_path / "repo-link"
    repo_link.symlink_to(review.ROOT.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="ignored private namespace"):
        review.save_state(state, repo_link / "private-review-leak-test.json")

    misleading = review.ROOT / "results_owner_private-review-leak-test.txt"
    assert not misleading.exists()
    with pytest.raises(ValueError, match="ignored private namespace"):
        require_private_output(misleading, purpose="review state")

    owner_input = review.ROOT / "data" / "owner_corpus" / "results_owner_input.json"
    with pytest.raises(ValueError, match="ignored private namespace"):
        require_private_output(owner_input, purpose="review state")


def test_summary_and_next_read_snapshot_without_running_solver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    card = {
        "id": "card-1",
        "task_id": "TASK-1",
        "query": "stored question",
        "task_class": "ask",
        "answer_status": "SUPPORTED",
        "claims": [],
        "hits": [],
        "usefulness_label": None,
    }
    state_path = tmp_path / "review.json"
    state = {"schema": "nano-lm.wedge_v1.review.v1", "cards": {}, "labels": {}}
    review.snapshot_review_cards(state, [card])
    review.save_state(state, state_path)
    monkeypatch.setattr(
        review,
        "cards_from_task_pack",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("solver ran")),
    )

    assert cli_main(["review", "--state", str(state_path), "--summary"]) == 0
    assert cli_main(["review", "--state", str(state_path), "--next"]) == 0
    output = capsys.readouterr().out
    assert '"n_labeled": 0' in output
    assert "stored question" in output


@pytest.mark.parametrize(
    "payload",
    [
        "[]",
        '{"cards": [], "labels": {}}',
        '{"cards": {"x": []}, "labels": {}}',
        '{"cards": {}, "labels": {"x": []}}',
        '{"cards": {}, "labels": {"x": "USEFUL"}}',
        "not json",
    ],
)
def test_load_state_rejects_malformed_shapes_without_raising(
    tmp_path: Path, payload: str
):
    path = tmp_path / "review.json"
    path.write_text(payload, encoding="utf-8")

    state = review.load_state(path)

    assert state["cards"] == {}
    assert state["labels"] == {}
    assert state["load_errors"]
    assert review.merge_prior_labels([], state) == []
