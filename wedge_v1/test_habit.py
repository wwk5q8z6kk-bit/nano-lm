"""Habit stub pins."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from wedge_v1.habit import (
    recall_saved,
    record,
    rerun_saved,
    save_question,
    saved_question_status,
    solver_implementation_fingerprint,
    weekly_summary,
)
from wedge_v1.runtime import ask as runtime_ask


QUERY = "How long before cached entries expire?"


def _seed_recall(tmp_path: Path, monkeypatch):
    import wedge_v1.habit as H

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    note = corpus / "cache.md"
    note.write_text("Cache policy: TTL as 300 seconds.\n", encoding="utf-8")
    saved = tmp_path / "saved.json"
    state = save_question(QUERY, task_id="T_TTL", corpus=corpus, path=saved)
    task_id = state["questions"][0]["task_id"]
    response = {"value": runtime_ask(QUERY, corpus_dir=corpus, persist_coe=False)}
    calls = {"n": 0}

    def fake_ask(query: str, *, corpus_dir: Path):
        calls["n"] += 1
        value = response["value"]
        if isinstance(value, BaseException):
            raise value
        return deepcopy(value)

    monkeypatch.setattr(H, "ask", fake_ask)
    first = recall_saved(task_id, corpus, path=saved)
    assert first["recall_state"] == "REFRESHED"
    assert calls["n"] == 1
    calls["n"] = 0
    return corpus, note, saved, task_id, response, calls


def _saved_data(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_saved(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def test_habit_record_and_summary(tmp_path: Path):
    path = tmp_path / "habit.json"
    record("ask", path=path)
    record("compare", path=path)
    s = weekly_summary(path=path)
    assert s["n_events"] == 2
    assert s["by_kind"]["ask"] == 1
    assert s["by_kind"]["compare"] == 1


def test_saved_question_freshness_and_reverification(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    note = corpus / "cache.md"
    note.write_text("Cache policy: TTL as 300 seconds.\n", encoding="utf-8")
    saved = tmp_path / "saved.json"

    save_question(
        "How long before cached entries expire?",
        corpus=corpus,
        path=saved,
    )
    assert saved_question_status(corpus, path=saved)[0]["state"] == "UNVERIFIED"

    first = rerun_saved(corpus, path=saved)
    assert first[0]["verification_state"] == "VERIFIED"
    persisted = _saved_data(saved)["questions"][0]
    assert persisted["verified_answer"]["answer_status"] in {
        "SUPPORTED",
        "CONTRADICTED",
        "ABSTAIN",
    }
    assert persisted["last_result_digest"]
    assert persisted["last_verified_provenance"]["corpus_digest"]
    assert persisted["solver_version_fingerprint"]
    assert saved_question_status(corpus, path=saved)[0]["state"] == "CURRENT"

    note.write_text("Cache policy: TTL as 600 seconds.\n", encoding="utf-8")
    stale = saved_question_status(corpus, path=saved)[0]
    assert stale["state"] == "STALE"
    assert stale["reason"] == "CORPUS_CHANGED"
    second = rerun_saved(corpus, path=saved)
    assert second[0]["verification_state"] == "REVERIFIED"
    assert saved_question_status(corpus, path=saved)[0]["state"] == "CURRENT"


def test_saved_question_query_and_id_changes_are_stale(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "cache.md").write_text("Cache TTL is 300 seconds.\n", encoding="utf-8")
    saved = tmp_path / "saved.json"
    save_question("What is the cache TTL?", task_id="T_TTL", corpus=corpus, path=saved)
    rerun_saved(corpus, path=saved)

    data = json.loads(saved.read_text(encoding="utf-8"))
    data["questions"][0]["query"] = "How long is the cache TTL?"
    saved.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    before = saved.read_text(encoding="utf-8")
    changed_query = saved_question_status(corpus, path=saved)[0]
    assert changed_query["state"] == "STALE"
    assert changed_query["reason"] == "TASK_CHANGED"
    assert saved.read_text(encoding="utf-8") == before

    data["questions"][0]["query"] = "What is the cache TTL?"
    data["questions"][0]["task_id"] = "T_TTL_V2"
    saved.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    changed_id = saved_question_status(corpus, path=saved)[0]
    assert changed_id["state"] == "STALE"
    assert changed_id["reason"] == "TASK_CHANGED"


def test_legacy_saved_verification_is_not_restored_until_explicit_rerun(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "cache.md").write_text("Cache TTL is 300 seconds.\n", encoding="utf-8")
    saved = tmp_path / "saved.json"
    saved.write_text(
        json.dumps(
            {
                "schema": "nano-lm.wedge_v1.saved_questions.v1",
                "questions": [
                    {
                        "query": "What is the cache TTL?",
                        "mode": "ask",
                        "last_corpus_digest": "legacy-digest",
                        "last_result_digest": "legacy-result",
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    before = saved.read_text(encoding="utf-8")

    status = saved_question_status(corpus, path=saved)[0]
    assert status["state"] == "LEGACY"
    assert status["reason"] == "MISSING_PROVENANCE"
    assert saved.read_text(encoding="utf-8") == before

    rerun = rerun_saved(corpus, path=saved)
    assert rerun[0]["verification_state"] == "LEGACY_REVERIFIED"
    assert saved_question_status(corpus, path=saved)[0]["state"] == "CURRENT"


def test_solver_fingerprint_covers_code_and_plugin_data_deterministically():
    import wedge_v1.habit as H

    sources = H._solver_source_bytes()
    assert {
        "runtime.py",
        "classical/verifier.py",
        "coe/audit.py",
        "coe/bind.py",
        "coe/schema.py",
        "plugins/data/coref_entities.json",
        "plugins/data/ocr_substitutions.json",
        "plugins/data/synonyms.json",
    } <= set(sources)
    assert H.CURRENT_SOLVER_FINGERPRINT == solver_implementation_fingerprint(sources)
    original = {"runtime.py": b"runtime-v1", "coe/audit.py": b"audit-v1"}
    reordered = {"coe/audit.py": b"audit-v1", "runtime.py": b"runtime-v1"}
    changed = {"runtime.py": b"runtime-v2", "coe/audit.py": b"audit-v1"}

    assert solver_implementation_fingerprint(original) == solver_implementation_fingerprint(
        reordered
    )
    assert solver_implementation_fingerprint(original) != solver_implementation_fingerprint(
        changed
    )

    changed_plugin_data = dict(sources)
    changed_plugin_data["plugins/data/synonyms.json"] += b"\n"
    assert solver_implementation_fingerprint(sources) != solver_implementation_fingerprint(
        changed_plugin_data
    )


def test_recall_cache_hit_audits_live_corpus_without_ask(tmp_path: Path, monkeypatch):
    import wedge_v1.habit as H

    corpus, _note, saved, task_id, _response, calls = _seed_recall(
        tmp_path, monkeypatch
    )
    real_audit = H.audit_payload
    audit_calls = {"n": 0}

    def counted_audit(payload: dict, docs: dict[str, str]):
        audit_calls["n"] += 1
        return real_audit(payload, docs)

    monkeypatch.setattr(H, "audit_payload", counted_audit)
    before = saved.read_text(encoding="utf-8")
    out = recall_saved(task_id, corpus, path=saved)

    assert out["recall_state"] == "CACHE_HIT"
    assert out["answer"] == _saved_data(saved)["questions"][0]["verified_answer"]
    assert out["aggregate"]["cache_hits"] == 1
    assert out["aggregate"]["forced_refreshes"] == 0
    assert out["aggregate"]["avoided_solver_runs"] == 1
    assert out["aggregate"]["solver_runs"] == 0
    assert out["aggregate"]["latency_ms"] >= 0
    assert calls["n"] == 0
    assert audit_calls["n"] == 1
    assert saved.read_text(encoding="utf-8") == before


def test_recall_task_mismatch_forces_exactly_one_refresh(tmp_path: Path, monkeypatch):
    corpus, _note, saved, _task_id, _response, calls = _seed_recall(
        tmp_path, monkeypatch
    )
    data = _saved_data(saved)
    data["questions"][0]["task_id"] = "T_TTL_V2"
    _write_saved(saved, data)

    out = recall_saved("T_TTL_V2", corpus, path=saved)

    assert out["recall_state"] == "REFRESHED"
    assert out["refresh_reason"] == "TASK_CHANGED"
    assert out["aggregate"]["forced_refreshes"] == 1
    assert calls["n"] == 1
    assert _saved_data(saved)["questions"][0]["task_id"] == "T_TTL_V2"


def test_recall_corpus_mismatch_forces_exactly_one_refresh(tmp_path: Path, monkeypatch):
    import wedge_v1.habit as H

    corpus, note, saved, task_id, response, calls = _seed_recall(tmp_path, monkeypatch)
    note.write_text("Cache policy: TTL as 600 seconds.\n", encoding="utf-8")
    response["value"] = runtime_ask(QUERY, corpus_dir=corpus, persist_coe=False)

    out = recall_saved(task_id, corpus, path=saved)

    assert out["recall_state"] == "REFRESHED"
    assert out["refresh_reason"] == "CORPUS_CHANGED"
    assert calls["n"] == 1
    assert (
        _saved_data(saved)["questions"][0]["last_verified_provenance"]["corpus_digest"]
        == H.corpus_digest(corpus)
    )


def test_recall_solver_mismatch_forces_exactly_one_refresh(tmp_path: Path, monkeypatch):
    import wedge_v1.habit as H

    corpus, _note, saved, task_id, _response, calls = _seed_recall(
        tmp_path, monkeypatch
    )
    data = _saved_data(saved)
    data["questions"][0]["solver_version_fingerprint"] = "older-solver"
    _write_saved(saved, data)

    out = recall_saved(task_id, corpus, path=saved)

    assert out["recall_state"] == "REFRESHED"
    assert out["refresh_reason"] == "SOLVER_CHANGED"
    assert calls["n"] == 1
    assert (
        _saved_data(saved)["questions"][0]["solver_version_fingerprint"]
        == H.CURRENT_SOLVER_FINGERPRINT
    )


def test_recall_digest_tamper_forces_exactly_one_refresh(tmp_path: Path, monkeypatch):
    corpus, _note, saved, task_id, _response, calls = _seed_recall(
        tmp_path, monkeypatch
    )
    data = _saved_data(saved)
    data["questions"][0]["last_result_digest"] = "0" * 64
    _write_saved(saved, data)

    out = recall_saved(task_id, corpus, path=saved)

    assert out["recall_state"] == "REFRESHED"
    assert out["refresh_reason"] == "RESULT_DIGEST_MISMATCH"
    assert calls["n"] == 1


def test_recall_audit_failure_forces_exactly_one_refresh(tmp_path: Path, monkeypatch):
    import wedge_v1.habit as H

    corpus, _note, saved, task_id, _response, calls = _seed_recall(
        tmp_path, monkeypatch
    )
    data = _saved_data(saved)
    question = data["questions"][0]
    atom = question["verified_answer"]["coe_claims"][0]["evidence_atoms"][0]
    atom["text"] = "tampered evidence"
    question["last_result_digest"] = H._result_digest(question["verified_answer"])
    _write_saved(saved, data)

    out = recall_saved(task_id, corpus, path=saved)

    assert out["recall_state"] == "REFRESHED"
    assert out["refresh_reason"] == "AUDIT_FAILED"
    assert calls["n"] == 1


def test_recall_failed_refresh_never_serves_or_persists_stale_answer(
    tmp_path: Path, monkeypatch
):
    corpus, _note, saved, task_id, response, calls = _seed_recall(tmp_path, monkeypatch)
    data = _saved_data(saved)
    data["questions"][0]["last_result_digest"] = "tampered"
    _write_saved(saved, data)
    before = saved.read_text(encoding="utf-8")
    response["value"] = {
        "query": QUERY,
        "answer_status": "SUPPORTED",
        "claims": [],
    }

    out = recall_saved(task_id, corpus, path=saved)

    assert out["recall_state"] == "REFRESH_FAILED"
    assert out["refresh_reason"] == "RESULT_DIGEST_MISMATCH"
    assert out["answer"]["answer_status"] == "ABSTAIN"
    assert out["answer"]["claims"] == []
    assert out["aggregate"]["forced_refreshes"] == 1
    assert out["aggregate"]["solver_runs"] == 1
    assert calls["n"] == 1
    assert saved.read_text(encoding="utf-8") == before


def test_habit_save_cli_surfaces_task_id(monkeypatch, capsys):
    import wedge_v1.cli as C

    monkeypatch.setattr(
        C,
        "save_question",
        lambda query, corpus=None: {
            "schema": "nano-lm.wedge_v1.saved_questions.v1",
            "questions": [{"task_id": "task-123", "query": query}],
        },
    )
    monkeypatch.setattr(C, "habit_record", lambda *_args, **_kwargs: None)

    assert C.main(["habit", "--save", "What", "changed?"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out == {"saved": "What changed?", "task_id": "task-123"}


def test_habit_recall_cli_routes_task_id(monkeypatch, capsys, tmp_path: Path):
    import wedge_v1.cli as C

    corpus = tmp_path / "corpus"
    called = {}
    expected = {
        "schema": "nano-lm.wedge_v1.saved_answer_recall.v1",
        "task_id": "task-123",
        "recall_state": "CACHE_HIT",
        "answer": {"answer_status": "ABSTAIN", "claims": []},
        "aggregate": {
            "cache_hits": 1,
            "forced_refreshes": 0,
            "avoided_solver_runs": 1,
            "solver_runs": 0,
            "latency_ms": 1.0,
        },
    }

    monkeypatch.setattr(C, "resolve_session_corpus", lambda explicit: corpus)

    def fake_recall(task_id: str, actual_corpus: Path):
        called.update(task_id=task_id, corpus=actual_corpus)
        return expected

    monkeypatch.setattr(C, "recall_saved", fake_recall)

    assert C.main(["habit", "--recall", "task-123"]) == 0
    assert json.loads(capsys.readouterr().out) == expected
    assert called == {"task_id": "task-123", "corpus": corpus}
