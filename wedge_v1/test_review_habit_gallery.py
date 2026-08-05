"""P3 review / habit / gallery pins (fixture corpora only)."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from wedge_v1.failure_gallery import (
    FINE_BUCKETS,
    build_gallery,
    classify_fine,
    gallery_to_markdown,
)
from wedge_v1.habit import (
    format_session_md,
    review_queue_summary,
    save_question,
    session,
)
from wedge_v1.owner_ready import check
from wedge_v1.review import (
    LABELS,
    apply_label,
    batch_label,
    build_card,
    cards_from_task_pack,
    format_card,
    interactive_review,
    label_summary,
    load_state,
    merge_prior_labels,
)
from wedge_v1.run_owner_dogfood import DEFAULT_TASKS, FIXTURE_CORPUS


def test_labels_stable():
    assert "USEFUL" in LABELS
    assert "OVER_ABSTENTION" in LABELS
    assert len(LABELS) == 9


def test_build_card_and_batch_label(tmp_path: Path, monkeypatch):
    import wedge_v1.review as R

    review_path = tmp_path / "review.json"
    monkeypatch.setattr(R, "REVIEW_PATH", review_path)
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
    assert review_path.is_file()


def test_compare_review_card_surfaces_numeric_values_by_source(tmp_path: Path):
    tasks = tmp_path / "tasks.json"
    tasks.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "id": "O02",
                        "mode": "compare",
                        "query": "metformin",
                        "expect_status": ["CONTRADICTED"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    card = cards_from_task_pack(tasks, FIXTURE_CORPUS)[0]
    rendered = format_card(card)

    assert card["comparison_values"] == [
        {"doc_id": "protocol_metformin", "values": ["500"]},
        {"doc_id": "protocol_metformin_alt", "values": ["850"]},
    ]
    assert "values: protocol_metformin=500; protocol_metformin_alt=850" in rendered
    assert "doc: —" not in rendered
    assert "span: metformin" not in rendered


def test_interactive_review_stdin(tmp_path: Path, monkeypatch):
    import wedge_v1.review as R

    review_path = tmp_path / "review.json"
    monkeypatch.setattr(R, "REVIEW_PATH", review_path)
    cards = cards_from_task_pack(DEFAULT_TASKS, FIXTURE_CORPUS, limit=2)
    state = load_state()
    stdin = io.StringIO("u\nq\n")
    stdout = io.StringIO()
    interactive_review(cards, state, stdin=stdin, stdout=stdout)
    assert state.get("labels") or "labeled" in stdout.getvalue()


def test_interactive_summary_counts_only_current_cards(tmp_path: Path, monkeypatch, capsys):
    import wedge_v1.cli as C
    import wedge_v1.habit as H
    import wedge_v1.review as R

    review_path = tmp_path / "review.json"
    habit_path = tmp_path / "habit.json"
    review_path.write_text(
        json.dumps(
            {
                "schema": "nano-lm.wedge_v1.review.v1",
                "cards": {
                    f"legacy-{i}": {
                        "id": f"legacy-{i}",
                        "task_id": f"OLD-{i}",
                        "usefulness_label": "USEFUL",
                    }
                    for i in range(3)
                },
                "labels": {f"legacy-{i}": "USEFUL" for i in range(3)},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(R, "REVIEW_PATH", review_path)
    monkeypatch.setattr(C.sys, "stdin", io.StringIO("u\n"))
    monkeypatch.setattr(
        C,
        "habit_record",
        lambda kind, **kwargs: H.record(kind, path=habit_path, **kwargs),
    )

    assert C.main(["review", "--demo", "--interactive", "--limit", "1"]) == 0

    stdout = capsys.readouterr().out
    summary = json.loads(stdout[stdout.rfind('{\n  "n_labeled"') :])
    persisted = json.loads(review_path.read_text(encoding="utf-8"))
    assert summary["n_labeled"] == 1
    assert summary["by_label"] == {"USEFUL": 1}
    assert len(persisted["labels"]) == 4
    habit = json.loads(habit_path.read_text(encoding="utf-8"))
    assert [event["kind"] for event in habit["events"]] == ["review_interactive"]

    assert C.main(["review", "--demo", "--summary", "--limit", "1"]) == 0
    summary_only = json.loads(capsys.readouterr().out)
    assert summary_only["n_labeled"] == 1
    assert summary_only["by_label"] == {"USEFUL": 1}


def test_label_summary_scopes_history_to_current_cards():
    cards = cards_from_task_pack(DEFAULT_TASKS, FIXTURE_CORPUS, limit=2)
    state = {"schema": "nano-lm.wedge_v1.review.v1", "cards": {}, "labels": {}}
    for card in cards:
        apply_label(state, card, "USEFUL")
    state["cards"]["legacy"] = {"id": "legacy", "usefulness_label": "NOT_USEFUL"}
    state["labels"]["legacy"] = "NOT_USEFUL"

    summary = label_summary(state, cards=cards)

    assert summary["n_labeled"] == 2
    assert summary["by_label"] == {"USEFUL": 2}
    assert len(state["labels"]) == 3


def _state_with_label(card: dict) -> dict:
    state = {"schema": "nano-lm.wedge_v1.review.v1", "cards": {}, "labels": {}}
    apply_label(state, card, "USEFUL")
    return state


def test_review_restores_label_only_for_matching_provenance(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "cache.md").write_text("Cache TTL is 300 seconds.\n", encoding="utf-8")
    prior = build_card("What is the cache TTL?", corpus=corpus, task_id="T_TTL")
    state = _state_with_label(prior)

    current = build_card("What is the cache TTL?", corpus=corpus, task_id="T_TTL")
    merged = merge_prior_labels([current], state)[0]

    assert merged["usefulness_label"] == "USEFUL"
    assert merged["prior_label_status"] == "RESTORED"


def test_review_invalidates_label_when_generated_result_changes(tmp_path: Path, monkeypatch):
    import wedge_v1.review as R

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "cache.md").write_text("Cache TTL is 300 seconds.\n", encoding="utf-8")
    prior = build_card("What is the cache TTL?", corpus=corpus, task_id="T_TTL")
    state = _state_with_label(prior)

    monkeypatch.setattr(
        R,
        "ask",
        lambda *_args, **_kwargs: {
            "answer_status": "ABSTAIN",
            "claims": [],
            "solver_path": ["changed-test-result"],
            "note": "no supported answer",
        },
    )
    current = build_card("What is the cache TTL?", corpus=corpus, task_id="T_TTL")
    merged = merge_prior_labels([current], state)[0]

    assert (
        prior["provenance"]["result_fingerprint"]
        != current["provenance"]["result_fingerprint"]
    )
    assert merged["usefulness_label"] is None
    assert merged["prior_label_status"] == "IGNORED_RESULT_CHANGED"


def test_review_invalidates_legacy_label_without_result_fingerprint(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "cache.md").write_text("Cache TTL is 300 seconds.\n", encoding="utf-8")
    prior = build_card("What is the cache TTL?", corpus=corpus, task_id="T_TTL")
    state = _state_with_label(prior)
    state["cards"][prior["id"]]["provenance"].pop("result_fingerprint")

    current = build_card("What is the cache TTL?", corpus=corpus, task_id="T_TTL")
    merged = merge_prior_labels([current], state)[0]

    assert merged["usefulness_label"] is None
    assert (
        merged["prior_label_status"]
        == "IGNORED_LEGACY_MISSING_RESULT_FINGERPRINT"
    )


def test_review_drops_label_when_source_changes_at_same_path(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    note = corpus / "cache.md"
    note.write_text("Cache TTL is 300 seconds.\n", encoding="utf-8")
    prior = build_card("What is the cache TTL?", corpus=corpus, task_id="T_TTL")
    state = _state_with_label(prior)
    note.write_text("Cache TTL is 600 seconds.\n", encoding="utf-8")

    current = build_card("What is the cache TTL?", corpus=corpus, task_id="T_TTL")
    merged = merge_prior_labels([current], state)[0]

    assert merged["usefulness_label"] is None
    assert merged["prior_label_status"] == "IGNORED_CORPUS_CHANGED"


def test_review_drops_label_when_query_or_task_id_changes(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "cache.md").write_text("Cache TTL is 300 seconds.\n", encoding="utf-8")
    prior = build_card("What is the cache TTL?", corpus=corpus, task_id="T_TTL")
    state = _state_with_label(prior)

    changed_query = build_card("How long is the cache TTL?", corpus=corpus, task_id="T_TTL")
    changed_id = build_card("What is the cache TTL?", corpus=corpus, task_id="T_TTL_V2")

    for current in (changed_query, changed_id):
        merged = merge_prior_labels([current], state)[0]
        assert merged["usefulness_label"] is None
        assert merged["prior_label_status"] == "IGNORED_TASK_CHANGED"


def test_review_legacy_label_is_reported_and_state_is_not_rewritten(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "cache.md").write_text("Cache TTL is 300 seconds.\n", encoding="utf-8")
    current = build_card("What is the cache TTL?", corpus=corpus, task_id="T_TTL")
    legacy = dict(current)
    legacy.pop("provenance")
    legacy["usefulness_label"] = "USEFUL"
    state = {
        "schema": "nano-lm.wedge_v1.review.v1",
        "cards": {current["id"]: legacy},
        "labels": {current["id"]: "USEFUL"},
    }
    before = json.dumps(state, sort_keys=True)

    merged = merge_prior_labels([current], state)[0]

    assert merged["usefulness_label"] is None
    assert merged["prior_label_status"] == "IGNORED_LEGACY_MISSING_PROVENANCE"
    assert json.dumps(state, sort_keys=True) == before


def test_review_next_does_not_create_or_rewrite_state(tmp_path: Path, monkeypatch):
    import wedge_v1.cli as C
    import wedge_v1.review as R

    review_path = tmp_path / "review.json"
    monkeypatch.setattr(R, "REVIEW_PATH", review_path)

    assert C.main(["review", "--demo", "--next", "--limit", "1"]) == 0
    assert not review_path.exists()


def test_habit_session_fixture(tmp_path: Path):
    sess = session(
        FIXTURE_CORPUS,
        habit_path=tmp_path / "habit.json",
        saved_path=tmp_path / "saved.json",
        session_path=tmp_path / "session.json",
    )
    assert sess["ingest"]["n_docs"] >= 1
    assert sess["next_action"]
    assert sess.get("recent_documents") is not None
    md = format_session_md(sess)
    assert "habit session" in md
    save_question("How long before cached entries expire?", path=tmp_path / "saved.json")


def test_habit_routes_invalidated_review_label_back_to_review(tmp_path: Path, monkeypatch):
    import wedge_v1.habit as H

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    note = corpus / "cache.md"
    note.write_text("Cache TTL is 300 seconds.\n", encoding="utf-8")
    prior = build_card(
        "How long before cached entries expire?",
        corpus=corpus,
        task_id="O01",
        expect_status=["SUPPORTED", "CONTRADICTED"],
    )
    state = _state_with_label(prior)
    monkeypatch.setattr(H, "load_review", lambda: state)

    note.write_text("Cache TTL is 600 seconds.\n", encoding="utf-8")
    sess = session(
        corpus,
        habit_path=tmp_path / "habit.json",
        saved_path=tmp_path / "saved.json",
        session_path=tmp_path / "session.json",
    )

    assert sess["review_queue"]["n_labeled"] == 0
    assert sess["review_queue"]["n_invalidated_labels"] == 1
    assert sess["next_action"] == f"python -m wedge_v1 review --corpus {corpus} --interactive"


def test_review_queue_recomputes_cached_task_fingerprint(tmp_path: Path, monkeypatch):
    import wedge_v1.habit as H

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "cache.md").write_text("Cache TTL is 300 seconds.\n", encoding="utf-8")
    prior = build_card("What is the cache TTL?", corpus=corpus, task_id="T_TTL")
    state = _state_with_label(prior)
    monkeypatch.setattr(H, "load_review", lambda: state)

    summary = review_queue_summary(
        corpus,
        tasks=[
            {
                "id": "T_TTL",
                "mode": "ask",
                "query": "How long is the cache TTL?",
            }
        ],
    )

    assert summary["n_labeled"] == 0
    assert summary["n_invalidated_labels"] == 1
    assert summary["invalidated_by_reason"] == {"TASK_CHANGED": 1}


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


def test_owner_gallery_defaults_never_write_shared_outputs(tmp_path: Path, monkeypatch):
    import wedge_v1.failure_gallery as gallery_module

    source = tmp_path / "private-input.json"
    source.write_text(
        json.dumps(
            {
                "schema": "nano-lm.wedge_v1.owner_dogfood_result.v1",
                "corpus": "/private/corpus",
                "n_tasks": 1,
                "n_ok": 0,
                "accuracy": 0.0,
                "rows": [
                    {
                        "id": "PRIVATE_TASK",
                        "query": "PRIVATE_QUERY",
                        "got_status": "ABSTAIN",
                        "expect_status": ["SUPPORTED"],
                        "ok": False,
                        "fail_kind": "over_abstain",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    owner_json = tmp_path / "owner" / "results_owner_failure_gallery.json"
    owner_md = tmp_path / "owner" / "results_owner_failure_gallery.md"
    public_json = tmp_path / "public" / "results_wedge_v1_failure_gallery.json"
    public_md = tmp_path / "public" / "results_wedge_v1_failure_gallery.md"
    monkeypatch.setattr(gallery_module, "OWNER_GALLERY_JSON", owner_json)
    monkeypatch.setattr(gallery_module, "OWNER_GALLERY_MD", owner_md)
    monkeypatch.setattr(gallery_module, "PUBLIC_GALLERY_JSON", public_json)
    monkeypatch.setattr(gallery_module, "PUBLIC_GALLERY_MD", public_md)

    gallery_module.write_gallery(source)

    assert owner_json.is_file()
    assert owner_md.is_file()
    assert not public_json.exists()
    assert not public_md.exists()


def test_custom_owner_gallery_preserves_private_source_lineage(
    tmp_path: Path, monkeypatch
):
    import wedge_v1.failure_gallery as gallery_module

    owner_result = tmp_path / "custom-result.json"
    owner_result.write_text(
        json.dumps(
            {
                "schema": "nano-lm.wedge_v1.owner_dogfood_result.v1",
                "corpus_class": "OWNER_PRIVATE",
                "n_tasks": 1,
                "n_ok": 0,
                "accuracy": 0.0,
                "rows": [],
            }
        ),
        encoding="utf-8",
    )
    custom_gallery = tmp_path / "custom-gallery.json"
    custom_gallery.write_text(
        json.dumps(
            {
                "schema": "nano-lm.wedge_v1.failure_gallery.v1",
                "source": str(owner_result),
                "n_tasks": 1,
                "rows": [],
            }
        ),
        encoding="utf-8",
    )
    owner_json = tmp_path / "private" / "results_owner_failure_gallery.json"
    owner_md = tmp_path / "private" / "results_owner_failure_gallery.md"
    public_json = tmp_path / "public" / "results_wedge_v1_failure_gallery.json"
    public_md = tmp_path / "public" / "results_wedge_v1_failure_gallery.md"
    monkeypatch.setattr(gallery_module, "OWNER_GALLERY_JSON", owner_json)
    monkeypatch.setattr(gallery_module, "OWNER_GALLERY_MD", owner_md)
    monkeypatch.setattr(gallery_module, "PUBLIC_GALLERY_JSON", public_json)
    monkeypatch.setattr(gallery_module, "PUBLIC_GALLERY_MD", public_md)

    gallery_module.write_gallery(custom_gallery)

    assert owner_json.is_file()
    assert owner_md.is_file()
    assert not public_json.exists()
    assert not public_md.exists()


def test_private_gallery_rejects_commit_visible_explicit_output(tmp_path: Path):
    import wedge_v1.failure_gallery as gallery_module

    source = tmp_path / "private.json"
    source.write_text(
        json.dumps(
            {
                "schema": "nano-lm.wedge_v1.owner_dogfood_result.v1",
                "corpus_class": "OWNER_PRIVATE",
                "n_tasks": 0,
                "rows": [],
            }
        ),
        encoding="utf-8",
    )
    unsafe = gallery_module.ROOT.parent / "private-gallery-leak-test.md"
    assert not unsafe.exists()

    with pytest.raises(ValueError, match="ignored private namespace"):
        gallery_module.write_gallery(source, markdown_output=unsafe)

    assert not unsafe.exists()


def test_owner_ready_demo():
    rep = check(demo=True)
    assert rep["corpus_exists"]
    assert rep["corpus"]["n_documents"] >= 1
    assert rep["smoke_ready"] is True
    assert rep["real_private_smoke_ready"] is False
    assert rep["representative_ready"] is False
    assert "canonical_command" in rep
    assert "OWNER_CORPUS_PENDING" in rep["blockers"]
    assert rep["ready_for_private_run"] is False
