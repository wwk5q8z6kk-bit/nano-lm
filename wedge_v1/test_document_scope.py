"""Exact document-scope regression pins."""
from __future__ import annotations

from pathlib import Path

import pytest

from wedge_v1.coe.audit import audit_payload
from wedge_v1.coe.schema import digest_docs
from wedge_v1.habit import recall_saved, save_question
from wedge_v1.review import (
    apply_label,
    build_card,
    corpus_content_digest,
    merge_prior_labels,
)
from wedge_v1.runtime import ask, compare, find_spans, scan

FIXTURE_CORPUS = Path(__file__).resolve().parent / "fixtures" / "owner_corpus"
QUERY = "How long before cached entries expire?"


def _assert_auditable_scope_abstention(result: dict, missing: list[str]) -> None:
    failure_code = "UNKNOWN_DOCUMENT_ID" if missing else "EMPTY_DOCUMENT_SCOPE"
    assert result["answer_status"] == "ABSTAIN"
    assert result["claims"] == []
    assert result["coe_claims"] == []
    assert result["selected_doc_ids"] == []
    assert result["missing_doc_ids"] == missing
    assert result["solver_path"] == ["document_scope"]
    assert result["n_docs"] == 0
    assert result["n_claims_presented"] == 0
    assert result["failure_codes"] == [failure_code]
    assert result["coe"]["corpus_digest"] == digest_docs({})
    assert result["coe"]["n_typed_claims"] == 0
    assert result["coe"]["completeness"] is True
    assert result["coe_audit"]["ok"] is True
    assert audit_payload(result, {})["ok"] is True


@pytest.mark.parametrize(
    ("operation", "query"),
    [
        (ask, QUERY),
        (find_spans, "TTL"),
        (compare, "TTL"),
    ],
)
def test_public_query_operations_reject_unknown_and_empty_scope(operation, query):
    for scope, missing in ((["does-not-exist"], ["does-not-exist"]), ([], [])):
        result = operation(
            query,
            FIXTURE_CORPUS,
            doc_ids=scope,
            persist_coe=False,
        )

        _assert_auditable_scope_abstention(result, missing)


def test_scan_rejects_unknown_and_empty_scope():
    for scope, missing in ((["does-not-exist"], ["does-not-exist"]), ([], [])):
        result = scan(FIXTURE_CORPUS, doc_ids=scope, persist_coe=False)

        _assert_auditable_scope_abstention(result, missing)


def test_exact_scope_changes_answer_without_source_leakage():
    unscoped = ask(QUERY, FIXTURE_CORPUS, persist_coe=False)
    scoped = ask(
        QUERY,
        FIXTURE_CORPUS,
        doc_ids=["note_cache_policy"],
        persist_coe=False,
    )
    paired = compare(
        "TTL",
        FIXTURE_CORPUS,
        doc_ids=["note_cache_policy_v2", "note_cache_policy"],
        persist_coe=False,
    )

    assert unscoped["answer_status"] == "CONTRADICTED"
    assert scoped["answer_status"] == "SUPPORTED"
    assert scoped["selected_doc_ids"] == ["note_cache_policy"]
    assert scoped["missing_doc_ids"] == []
    assert scoped["coe"]["selected_doc_ids"] == ["note_cache_policy"]
    assert scoped["coe_audit"]["ok"] is True
    assert {
        claim["doc_id"]
        for claim in scoped["claims"]
        if claim.get("doc_id")
    } == {"note_cache_policy"}
    assert paired["answer_status"] == "CONTRADICTED"
    assert paired["selected_doc_ids"] == [
        "note_cache_policy",
        "note_cache_policy_v2",
    ]


def test_nested_document_ids_are_exact(tmp_path: Path):
    corpus = tmp_path / "corpus"
    nested = corpus / "alpha"
    nested.mkdir(parents=True)
    (nested / "report.md").write_text(
        "Cache policy: TTL as 300 seconds.\n",
        encoding="utf-8",
    )
    (corpus / "report.md").write_text(
        "Cache policy: TTL as 600 seconds.\n",
        encoding="utf-8",
    )

    result = find_spans(
        "TTL",
        corpus,
        doc_ids=["alpha/report"],
        persist_coe=False,
    )

    assert result["answer_status"] == "SUPPORTED"
    assert result["selected_doc_ids"] == ["alpha/report"]
    assert {claim["doc_id"] for claim in result["claims"]} == {"alpha/report"}


def test_scoped_digest_ignores_unrelated_documents(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    selected = corpus / "selected.md"
    selected.write_text("Cache TTL is 300 seconds.\n", encoding="utf-8")
    unrelated = corpus / "unrelated.md"
    unrelated.write_text("Unrelated v1.\n", encoding="utf-8")

    first = corpus_content_digest(corpus, doc_ids=["selected"])
    unrelated.write_text("Unrelated v2.\n", encoding="utf-8")
    assert corpus_content_digest(corpus, doc_ids=["selected"]) == first

    selected.write_text("Cache TTL is 600 seconds.\n", encoding="utf-8")
    assert corpus_content_digest(corpus, doc_ids=["selected"]) != first


def test_review_labels_are_scope_specific(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("Cache TTL is 300 seconds.\n", encoding="utf-8")
    (corpus / "b.md").write_text("Cache TTL is 600 seconds.\n", encoding="utf-8")

    card_a = build_card(QUERY, corpus=corpus, task_id="TTL", doc_ids=["a"])
    state = {"schema": "nano-lm.wedge_v1.review.v1", "cards": {}, "labels": {}}
    apply_label(state, card_a, "USEFUL")

    same_scope = build_card(QUERY, corpus=corpus, task_id="TTL", doc_ids=["a"])
    other_scope = build_card(QUERY, corpus=corpus, task_id="TTL", doc_ids=["b"])
    restored = merge_prior_labels([same_scope], state)[0]
    rejected = merge_prior_labels([other_scope], state)[0]

    assert card_a["id"] != other_scope["id"]
    assert restored["usefulness_label"] == "USEFUL"
    assert rejected["usefulness_label"] is None
    assert rejected["prior_label_status"] == "IGNORED_TASK_CHANGED"


def test_scoped_recall_refreshes_once_then_uses_audited_cache(
    tmp_path: Path,
    monkeypatch,
):
    from wedge_v1 import habit

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text(
        "Cache policy: TTL as 300 seconds.\n",
        encoding="utf-8",
    )
    (corpus / "b.md").write_text(
        "Cache policy: TTL as 600 seconds.\n",
        encoding="utf-8",
    )
    saved_path = tmp_path / "saved.json"
    saved = save_question(
        QUERY,
        task_id="TTL",
        corpus=corpus,
        doc_ids=["a", "a"],
        path=saved_path,
    )
    task_id = saved["questions"][0]["task_id"]
    calls = {"n": 0}

    def counted_ask(query: str, *, corpus_dir: Path, doc_ids=None):
        calls["n"] += 1
        return ask(
            query,
            corpus_dir=corpus_dir,
            doc_ids=doc_ids,
            persist_coe=False,
        )

    monkeypatch.setattr(habit, "ask", counted_ask)

    first = recall_saved(task_id, corpus, path=saved_path)
    assert first["recall_state"] == "REFRESHED"
    assert calls["n"] == 1

    calls["n"] = 0
    second = recall_saved(task_id, corpus, path=saved_path)
    assert second["recall_state"] == "CACHE_HIT"
    assert second["selected_doc_ids"] == ["a"]
    assert second["aggregate"]["avoided_solver_runs"] == 1
    assert calls["n"] == 0

    (corpus / "unrelated.md").write_text("new unrelated document\n", encoding="utf-8")
    third = recall_saved(task_id, corpus, path=saved_path)
    assert third["recall_state"] == "CACHE_HIT"
    assert calls["n"] == 0

    changed_scope = recall_saved(task_id, corpus, doc_ids=["b"], path=saved_path)
    assert changed_scope["recall_state"] == "REFRESHED"
    assert changed_scope["refresh_reason"] == "SCOPE_CHANGED"
    assert changed_scope["selected_doc_ids"] == ["b"]
    assert calls["n"] == 1

    calls["n"] = 0
    persisted_scope = recall_saved(task_id, corpus, path=saved_path)
    assert persisted_scope["recall_state"] == "CACHE_HIT"
    assert persisted_scope["selected_doc_ids"] == ["b"]
    assert calls["n"] == 0


def test_failed_scope_override_never_replaces_verified_snapshot(
    tmp_path: Path,
    monkeypatch,
):
    from wedge_v1 import habit

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text(
        "Cache policy: TTL as 300 seconds.\n",
        encoding="utf-8",
    )
    saved_path = tmp_path / "saved.json"
    saved = save_question(
        QUERY,
        task_id="TTL",
        corpus=corpus,
        doc_ids=["a"],
        path=saved_path,
    )
    task_id = saved["questions"][0]["task_id"]

    def real_ask(query: str, *, corpus_dir: Path, doc_ids=None):
        return ask(
            query,
            corpus_dir=corpus_dir,
            doc_ids=doc_ids,
            persist_coe=False,
        )

    monkeypatch.setattr(habit, "ask", real_ask)
    assert recall_saved(task_id, corpus, path=saved_path)["recall_state"] == "REFRESHED"
    before = saved_path.read_bytes()

    failed = recall_saved(
        task_id,
        corpus,
        doc_ids=["missing"],
        path=saved_path,
    )

    assert failed["recall_state"] == "REFRESH_FAILED"
    assert failed["answer"]["answer_status"] == "ABSTAIN"
    assert failed["answer"]["claims"] == []
    assert saved_path.read_bytes() == before
