"""Representative-usefulness study lifecycle and privacy boundaries."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import wedge_v1.habit as habit_module
import wedge_v1.study as study_module
from wedge_v1.cli import main as cli_main
from wedge_v1.coe.schema import digest_text
from wedge_v1.review import apply_label, load_state, save_state
from wedge_v1.study import (
    EXAMPLE_TASK_PACK,
    assess_inputs,
    review_study,
    run_study,
    summarize_study,
    verify_study,
)


def _representative_inputs(
    tmp_path: Path, *, include_recall: bool = True
) -> tuple[Path, Path]:
    corpus = tmp_path / "private-corpus"
    corpus.mkdir()
    tasks = []
    modes = ("ask", "find", "compare")
    for index in range(10):
        doc_id = f"private_note_{index}"
        marker = f"PRIVATE_MARKER_{index}_ZXQ"
        field = f"TTL_{index}"
        (corpus / f"{doc_id}.md").write_text(
            f"Project record {marker}. {field} is {300 + index} seconds.\n",
            encoding="utf-8",
        )
        mode = "recall" if include_recall and index == 0 else modes[index % len(modes)]
        query = (
            f"project record {marker}"
            if mode in {"ask", "recall"}
            else marker
            if mode == "find"
            else field
        )
        tasks.append(
            {
                "id": f"PRIVATE_TASK_{index}",
                "mode": mode,
                "query": query,
                "doc_ids": (
                    [doc_id, f"private_note_{(index + 1) % 10}"]
                    if mode == "compare"
                    else [doc_id]
                ),
                "expect_status": ["SUPPORTED", "CONTRADICTED", "ABSTAIN"],
                "manual_baseline_seconds": 60 + index,
            }
        )
    task_path = tmp_path / "private-questions.json"
    task_path.write_text(json.dumps({"tasks": tasks}), encoding="utf-8")
    return corpus, task_path


def _agent_applied_pilot_inputs(tmp_path: Path) -> tuple[Path, Path]:
    corpus, tasks = _representative_inputs(tmp_path)
    payload = json.loads(tasks.read_text(encoding="utf-8"))
    payload.update(
        {
            "schema": study_module.TASK_PACK_SCHEMA,
            "storage_class": study_module.OWNER_PRIVATE,
            "study_class": study_module.AGENT_APPLIED_SCOPED_PILOT,
        }
    )
    for task in payload["tasks"]:
        task.pop("manual_baseline_seconds")
    tasks.write_text(json.dumps(payload), encoding="utf-8")
    return corpus, tasks


def _coherently_rewrite_cards(study_dir: Path, cards: list[object]) -> None:
    cards_path = study_dir / "cards.json"
    cards_payload = json.loads(cards_path.read_text(encoding="utf-8"))
    cards_payload["cards"] = cards
    cards_path.write_text(
        json.dumps(cards_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    recall_cards = [
        card
        for card in cards
        if isinstance(card, dict) and card.get("task_class") == "recall"
    ]
    manifest_path = study_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["result"].update(
        {
            "digest": hashlib.sha256(cards_path.read_bytes()).hexdigest(),
            "n_cards": len(cards),
            "all_coe_audits_ok": bool(cards)
            and all(
                isinstance(card, dict) and card.get("coe_audit_ok") is True
                for card in cards
            ),
            "n_repeat_recall_cards": len(recall_cards),
            "all_repeat_recall_invariants_ok": bool(recall_cards),
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _coherently_rewrite_audited_results(
    study_dir: Path, audited_payload: dict
) -> None:
    audited_path = study_dir / "audited_results.json"
    audited_path.write_text(
        json.dumps(audited_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = study_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["result"]["audited_results_digest"] = hashlib.sha256(
        audited_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _reproject_forged_card(task: dict, result: dict, corpus: Path) -> dict:
    mode = str(task.get("mode") or "ask").strip().lower()
    scope = study_module.normalize_doc_ids(task.get("doc_ids")) or []
    corpus_digest = study_module.corpus_content_digest(corpus, doc_ids=scope)
    card = study_module.card_from_result(
        task["query"],
        result,
        corpus=corpus,
        mode=mode,
        task_id=task["id"],
        expect_status=task.get("expect_status"),
        corpus_digest=corpus_digest,
        doc_ids=scope,
        manual_baseline_seconds=task.get("manual_baseline_seconds"),
    )
    if mode == "recall":
        card["repeat_recall"] = study_module._expected_repeat_recall_metadata()
        card["provenance"] = study_module.provenance_record(
            task["query"],
            corpus=corpus,
            corpus_digest=corpus_digest,
            mode="recall",
            task_id=task["id"],
            expect_status=task.get("expect_status"),
            result_fingerprint=study_module.result_output_fingerprint(result, card),
            doc_ids=scope,
        )
    return card


def _forge_first_claim(payload: dict, forged_value: str) -> None:
    claim = payload["claims"][0]
    claim["value"] = forged_value
    claim["evidence"][0]["text"] = forged_value
    if isinstance(payload.get("evidence"), dict):
        payload["evidence"]["text"] = forged_value
        payload["evidence"]["value"] = forged_value
    if "evidence_span" in payload:
        payload["evidence_span"] = forged_value
    if payload.get("hits"):
        payload["hits"][0]["text"] = forged_value
        payload["hits"][0]["value"] = forged_value
    if payload.get("coe_claims"):
        typed_claim = payload["coe_claims"][0]
        typed_claim["raw_value"] = forged_value
        typed_claim["evidence_atoms"][0]["text"] = forged_value


def test_representative_check_requires_real_scoped_inputs(tmp_path: Path):
    corpus, tasks = _representative_inputs(tmp_path)

    report = assess_inputs(corpus, tasks)

    assert report["smoke_ready"] is True
    assert report["representative_ready"] is True
    assert report["blockers"] == []
    assert report["corpus"]["n_documents"] == 10
    assert report["tasks"]["n_tasks"] == 10
    assert report["tasks"]["n_exactly_scoped"] == 10
    assert report["tasks"]["n_unique_scoped_documents"] == 10
    assert report["tasks"]["mode_counts"]["recall"] >= 1
    assert set(report["identity"]) == {
        "corpus_digest",
        "task_pack_digest",
        "solver_fingerprint",
        "instrument_fingerprint",
        "pdf_extractor",
    }


def test_representative_check_requires_repeat_recall(tmp_path: Path):
    corpus, tasks = _representative_inputs(tmp_path, include_recall=False)

    report = assess_inputs(corpus, tasks)

    assert report["representative_ready"] is False
    assert "REPEAT_RECALL_REQUIRED" in report["blockers"]
    assert report["tasks"]["mode_counts"].get("recall", 0) == 0


@pytest.mark.parametrize(
    ("missing_field", "blocker"),
    (
        ("manual_baseline_seconds", "MANUAL_BASELINE_INCOMPLETE"),
        ("expect_status", "EXPECTED_STATUS_INCOMPLETE"),
    ),
)
def test_canonical_task_pack_requires_complete_measurement_inputs(
    tmp_path: Path, missing_field: str, blocker: str
) -> None:
    corpus, tasks = _representative_inputs(tmp_path)
    payload = json.loads(tasks.read_text(encoding="utf-8"))
    payload.update(
        {
            "schema": study_module.TASK_PACK_SCHEMA,
            "storage_class": study_module.OWNER_PRIVATE,
        }
    )
    payload["tasks"][0].pop(missing_field)
    tasks.write_text(json.dumps(payload), encoding="utf-8")

    report = assess_inputs(corpus, tasks)

    assert report["representative_ready"] is False
    assert blocker in report["blockers"]
    assert blocker not in report["warnings"]


def test_explicit_agent_applied_pilot_is_study_ready_not_representative_ready(
    tmp_path: Path,
) -> None:
    corpus, tasks = _agent_applied_pilot_inputs(tmp_path)

    report = assess_inputs(corpus, tasks)

    assert report["study_class"] == study_module.AGENT_APPLIED_SCOPED_PILOT
    assert report["study_ready"] is True
    assert report["representative_ready"] is False
    assert report["blockers"] == []
    assert report["warnings"] == ["MANUAL_BASELINE_INCOMPLETE"]
    assert report["required_reviewer_kind"] == "agent_applied"
    assert report["manual_time_comparison_enabled"] is False
    assert report["identity"]["task_pack_digest"] == hashlib.sha256(
        tasks.read_bytes()
    ).hexdigest()
    assert "not representative-use evidence" in report["claim_boundary"]
    assert "no time-saved claim" in report["claim_boundary"]


@pytest.mark.parametrize("study_class", ("UNKNOWN", 42, None, {}, []))
def test_unknown_or_malformed_study_class_fails_closed(
    tmp_path: Path, study_class: object
) -> None:
    corpus, tasks = _representative_inputs(tmp_path)
    payload = json.loads(tasks.read_text(encoding="utf-8"))
    payload.update(
        {
            "schema": study_module.TASK_PACK_SCHEMA,
            "storage_class": study_module.OWNER_PRIVATE,
            "study_class": study_class,
        }
    )
    tasks.write_text(json.dumps(payload), encoding="utf-8")

    report = assess_inputs(corpus, tasks)

    assert report["study_ready"] is False
    assert report["representative_ready"] is False
    assert report["blockers"] == ["TASK_PACK_STUDY_CLASS_INVALID"]


def test_pilot_class_requires_canonical_private_task_pack(tmp_path: Path) -> None:
    corpus, tasks = _representative_inputs(tmp_path)
    payload = json.loads(tasks.read_text(encoding="utf-8"))
    payload["study_class"] = study_module.AGENT_APPLIED_SCOPED_PILOT
    tasks.write_text(json.dumps(payload), encoding="utf-8")

    report = assess_inputs(corpus, tasks)

    assert report["study_ready"] is False
    assert report["representative_ready"] is False
    assert report["blockers"] == ["TASK_PACK_STUDY_CLASS_INVALID"]


def test_pilot_run_and_summary_preserve_agent_and_no_time_claim_boundaries(
    tmp_path: Path, capsys,
) -> None:
    corpus, tasks = _agent_applied_pilot_inputs(tmp_path)
    study_dir = tmp_path / "pilot-study"

    assert cli_main(
        [
            "study",
            "check",
            "--corpus",
            str(corpus),
            "--tasks",
            str(tasks),
            "--dir",
            str(study_dir),
        ]
    ) == 0
    check_output = json.loads(capsys.readouterr().out)
    assert check_output["study_ready"] is True
    assert check_output["representative_ready"] is False

    result = run_study(corpus, tasks, study_dir)
    manifest, cards_payload, errors = verify_study(study_dir)

    assert result["status"] == "COMPLETE"
    assert result["decision"] == "REVIEW_REQUIRED"
    assert result["study_class"] == study_module.AGENT_APPLIED_SCOPED_PILOT
    assert result["representative_ready"] is False
    assert result["required_reviewer_kind"] == "agent_applied"
    assert result["time_saved_claim_supported"] is False
    assert "not representative-use evidence" in result["claim_boundary"]
    assert "no time-saved claim" in result["claim_boundary"]
    assert errors == []
    assert manifest["study_class"] == study_module.AGENT_APPLIED_SCOPED_PILOT
    assert manifest["required_reviewer_kind"] == "agent_applied"
    assert manifest["manual_time_comparison_enabled"] is False
    assert len(cards_payload["cards"]) == 10

    wrong_reviewer = review_study(study_dir, reviewer_kind="owner")
    assert wrong_reviewer["decision"] == "INCOMPLETE"
    assert wrong_reviewer["blockers"] == ["PILOT_REVIEWER_KIND_REQUIRED"]

    review_path = study_dir / "review.json"
    state = load_state(review_path)
    for index, card in enumerate(cards_payload["cards"]):
        apply_label(
            state,
            card,
            "USEFUL",
            reviewer_kind="agent_applied",
            review_elapsed_s=1 + index,
        )
    save_state(state, review_path)

    summary = summarize_study(study_dir)

    assert summary["status"] == "COMPLETE"
    assert summary["study_ready"] is True
    assert summary["study_class"] == study_module.AGENT_APPLIED_SCOPED_PILOT
    assert summary["representative_ready"] is False
    assert summary["review_evidence_kind"] == "AGENT_APPLIED_RUBRIC"
    assert summary["manual_baseline"] == {
        "n_tasks_with_baseline": 0,
        "comparison": None,
        "time_saved_claim_supported": False,
    }
    assert "not representative-use evidence" in summary["claim_boundary"]
    assert "no time-saved claim" in summary["claim_boundary"]


def test_example_pack_and_fixture_can_never_be_representative():
    fixture = Path(__file__).resolve().parent / "fixtures" / "owner_corpus"

    report = assess_inputs(fixture, EXAMPLE_TASK_PACK, demo=True)

    assert report["smoke_ready"] is True
    assert report["representative_ready"] is False
    assert "DEMO_OR_FIXTURE_CORPUS" in report["blockers"]
    assert "EXAMPLE_TASK_PACK" in report["blockers"]


def test_pytest_guard_blocks_owner_artifact_writes():
    protected = Path(__file__).resolve().parent / "results_owner_guard_probe.json"

    with pytest.raises(AssertionError, match="protected owner state"):
        protected.write_text("must not be written", encoding="utf-8")

    assert not protected.exists()


def test_study_run_is_scoped_audited_and_does_not_persist_coe(
    tmp_path: Path, monkeypatch
):
    corpus, tasks = _representative_inputs(tmp_path)
    study_dir = tmp_path / "study"
    coe_dir = Path(study_module.ROOT) / ".coe_runs"
    coe_before = {path.resolve() for path in coe_dir.glob("*.jsonl")}
    global_saved = Path(habit_module.SAVED_QUESTIONS)
    global_saved_before = global_saved.read_bytes() if global_saved.is_file() else None
    original_recall_solver = habit_module.ask
    recall_solver_calls = []

    def _tracked_recall_solver(*args, **kwargs):
        recall_solver_calls.append((args, kwargs))
        return original_recall_solver(*args, **kwargs)

    monkeypatch.setattr(habit_module, "ask", _tracked_recall_solver)

    result = run_study(corpus, tasks, study_dir)
    manifest, cards_payload, errors = verify_study(study_dir)

    assert result["status"] == "COMPLETE"
    assert result["decision"] == "REVIEW_REQUIRED"
    assert errors == []
    assert manifest["result"]["all_coe_audits_ok"] is True
    assert manifest["result"]["digest"] == hashlib.sha256(
        (study_dir / "cards.json").read_bytes()
    ).hexdigest()
    assert len(cards_payload["cards"]) == 10
    for card in cards_payload["cards"]:
        assert card["selected_doc_ids"] == card["provenance"]["doc_ids"]
        assert card["missing_doc_ids"] == []
        assert card["coe_audit_ok"] is True
    recall_cards = [
        card for card in cards_payload["cards"] if card["task_class"] == "recall"
    ]
    assert recall_cards
    for card in recall_cards:
        repeat = card["repeat_recall"]
        assert repeat["first"] == {
            "audit_ok": True,
            "avoided_solver_runs": 0,
            "cache_hits": 0,
            "forced_refreshes": 1,
            "recall_state": "REFRESHED",
            "solver_runs": 1,
        }
        assert repeat["second"] == {
            "audit_ok": True,
            "avoided_solver_runs": 1,
            "cache_hits": 1,
            "forced_refreshes": 0,
            "recall_state": "CACHE_HIT",
            "solver_runs": 0,
        }
        assert repeat["answer_fingerprint_match"] is True
        assert card["selected_doc_ids"] == card["provenance"]["doc_ids"]
    assert len(recall_solver_calls) == 1
    assert recall_solver_calls[0][1]["persist_coe"] is False
    local_saved = study_dir / "saved_questions.json"
    assert local_saved.is_file()
    assert manifest["artifacts"]["saved_questions"] == "saved_questions.json"
    assert manifest["result"]["saved_questions_digest"] == hashlib.sha256(
        local_saved.read_bytes()
    ).hexdigest()
    audited_path = study_dir / "audited_results.json"
    audited_payload = json.loads(audited_path.read_text(encoding="utf-8"))
    assert manifest["artifacts"]["audited_results"] == "audited_results.json"
    assert manifest["result"]["audited_results_digest"] == hashlib.sha256(
        audited_path.read_bytes()
    ).hexdigest()
    assert audited_payload["schema"] == study_module.AUDITED_RESULTS_SCHEMA
    assert audited_payload["study_id"] == manifest["study_id"]
    assert len(audited_payload["results"]) == 10
    assert all(isinstance(row.get("result"), dict) for row in audited_payload["results"])
    assert {path.resolve() for path in coe_dir.glob("*.jsonl")} == coe_before
    assert global_saved.is_file() is (global_saved_before is not None)
    if global_saved_before is not None:
        assert global_saved.read_bytes() == global_saved_before


def test_study_verification_rederives_once_per_task_without_persistence(
    tmp_path: Path, monkeypatch
):
    corpus, tasks = _representative_inputs(tmp_path)
    study_dir = tmp_path / "study"
    run_study(corpus, tasks, study_dir)
    coe_dir = Path(study_module.ROOT) / ".coe_runs"

    def _file_snapshot(paths) -> dict[str, str]:
        return {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
            if path.is_file()
        }

    coe_before = _file_snapshot(coe_dir.rglob("*"))
    owner_paths = list(Path(study_module.ROOT).glob("results_owner*"))
    owner_before = _file_snapshot(owner_paths)
    global_saved = Path(habit_module.SAVED_QUESTIONS)
    global_saved_before = global_saved.read_bytes() if global_saved.is_file() else None
    study_before = _file_snapshot(study_dir.rglob("*"))
    task_rows = json.loads(tasks.read_text(encoding="utf-8"))["tasks"]
    expected_calls = [
        (
            (
                "ask"
                if row["mode"] in {"ask", "recall"}
                else "find_spans"
                if row["mode"] == "find"
                else row["mode"]
            ),
            row["query"],
            tuple(study_module.normalize_doc_ids(row["doc_ids"]) or []),
        )
        for row in task_rows
    ]
    solver_calls = []

    def _tracked_solver(name, original):
        def _run(query, *args, **kwargs):
            assert kwargs.get("persist_coe") is False
            assert Path(kwargs["corpus_dir"]).resolve() == corpus.resolve()
            solver_calls.append((name, query, tuple(kwargs.get("doc_ids") or [])))
            return original(query, *args, **kwargs)

        return _run

    for name in ("ask", "compare", "find_spans"):
        monkeypatch.setattr(
            study_module,
            name,
            _tracked_solver(name, getattr(study_module, name)),
        )

    def _recall_must_not_run(*_args, **_kwargs):
        raise AssertionError("verification must rederive recall through ask")

    monkeypatch.setattr(study_module, "recall_saved", _recall_must_not_run)

    _manifest, _cards, errors = verify_study(study_dir)

    assert errors == []
    assert solver_calls == expected_calls
    assert len(solver_calls) == len({row["id"] for row in task_rows})
    assert _file_snapshot(coe_dir.rglob("*")) == coe_before
    assert _file_snapshot(owner_paths) == owner_before
    assert _file_snapshot(study_dir.rglob("*")) == study_before
    assert global_saved.is_file() is (global_saved_before is not None)
    if global_saved_before is not None:
        assert global_saved.read_bytes() == global_saved_before


def test_study_normalizes_task_identity_and_query_consistently(tmp_path: Path):
    corpus, tasks = _representative_inputs(tmp_path)
    task_payload = json.loads(tasks.read_text(encoding="utf-8"))
    task_payload["tasks"][0]["id"] = f"  {task_payload['tasks'][0]['id']}  "
    task_payload["tasks"][0]["query"] = f"  {task_payload['tasks'][0]['query']}  "
    tasks.write_text(json.dumps(task_payload), encoding="utf-8")
    study_dir = tmp_path / "study"

    result = run_study(corpus, tasks, study_dir)
    _manifest, cards_payload, errors = verify_study(study_dir)

    assert result["status"] == "COMPLETE"
    assert errors == []
    recall_card = next(
        card for card in cards_payload["cards"] if card["task_class"] == "recall"
    )
    assert recall_card["task_id"] == task_payload["tasks"][0]["id"].strip()
    assert recall_card["query"] == task_payload["tasks"][0]["query"].strip()


def test_safe_summary_selects_repeated_failure_without_private_content(tmp_path: Path):
    corpus, tasks = _representative_inputs(tmp_path)
    study_dir = tmp_path / "study"
    run_study(corpus, tasks, study_dir)
    cards = json.loads((study_dir / "cards.json").read_text(encoding="utf-8"))["cards"]
    review_path = study_dir / "review.json"
    state = load_state(review_path)
    for index, card in enumerate(cards):
        apply_label(
            state,
            card,
            "WRONG_EVIDENCE" if index < 2 else "USEFUL",
            failure_reason="PRIVATE_REASON_ZXQ",
            suggested_correction="PRIVATE_CORRECTION_ZXQ",
            reviewer_kind="agent_applied",
            review_elapsed_s=2 + index,
        )
    save_state(state, review_path)

    summary = summarize_study(study_dir)
    serialized = json.dumps(summary, sort_keys=True)

    assert summary["status"] == "COMPLETE"
    assert summary["decision"] == "FIX_REPEATED_FAILURE"
    assert summary["outcomes"]["first_repeated_failure_class"] == "WRONG_EVIDENCE"
    assert summary["review_evidence_kind"] == "AGENT_APPLIED_RUBRIC"
    assert summary["identity"]["review_digest"] == hashlib.sha256(
        review_path.read_bytes()
    ).hexdigest()
    assert summary["manual_baseline"]["comparison"] is not None
    assert summary["coverage"]["repeat_recall"] == {
        "n_first_refreshed": 1,
        "n_second_cache_hits": 1,
        "n_solver_runs": 1,
        "n_tasks": 1,
        "n_avoided_solver_runs": 1,
    }
    for private_value in (
        "PRIVATE_MARKER",
        "PRIVATE_TASK",
        "private_note",
        "PRIVATE_REASON_ZXQ",
        "PRIVATE_CORRECTION_ZXQ",
        str(corpus),
        str(tasks),
    ):
        assert private_value not in serialized


def test_complete_review_records_no_repeated_failure(tmp_path: Path):
    corpus, tasks = _representative_inputs(tmp_path)
    study_dir = tmp_path / "study"
    run_study(corpus, tasks, study_dir)
    cards = json.loads((study_dir / "cards.json").read_text(encoding="utf-8"))["cards"]
    review_path = study_dir / "review.json"
    state = load_state(review_path)
    for card in cards:
        apply_label(
            state,
            card,
            "USEFUL",
            reviewer_kind="owner",
            review_elapsed_s=1,
        )
    save_state(state, review_path)

    summary = summarize_study(study_dir)

    assert summary["status"] == "COMPLETE"
    assert summary["decision"] == "NO_REPEATED_FAILURE"
    assert summary["outcomes"]["first_repeated_failure_class"] is None
    assert "none repeated" in summary["next_action"].lower()


def test_safe_summary_accepts_canonical_only_correction_reasons(tmp_path: Path):
    corpus, tasks = _representative_inputs(tmp_path)
    study_dir = tmp_path / "study"
    run_study(corpus, tasks, study_dir)
    cards = json.loads((study_dir / "cards.json").read_text(encoding="utf-8"))["cards"]
    review_path = study_dir / "review.json"
    state = load_state(review_path)
    for index, card in enumerate(cards):
        apply_label(
            state,
            card,
            "WRONG_EVIDENCE" if index < 2 else "USEFUL",
            correction_reason=("literal evidence was wrong" if index < 2 else ""),
            suggested_correction=("select the scoped source" if index < 2 else ""),
            reviewer_kind="owner",
            review_elapsed_s=1 + index,
        )
    save_state(state, review_path)

    summary = summarize_study(study_dir)

    assert summary["status"] == "COMPLETE"
    assert summary["decision"] == "FIX_REPEATED_FAILURE"
    assert summary["blockers"] == []
    assert summary["outcomes"]["first_repeated_failure_class"] == "WRONG_EVIDENCE"
    assert summary["outcomes"]["by_failure_class"] == {"WRONG_EVIDENCE": 2}


def test_successful_labels_cannot_manufacture_a_repeated_failure(tmp_path: Path):
    corpus, tasks = _representative_inputs(tmp_path)
    study_dir = tmp_path / "study"
    run_study(corpus, tasks, study_dir)
    card = json.loads((study_dir / "cards.json").read_text(encoding="utf-8"))["cards"][0]

    with pytest.raises(ValueError, match="successful label"):
        apply_label(
            load_state(study_dir / "review.json"),
            card,
            "USEFUL",
            failure_class="WRONG_EVIDENCE",
            reviewer_kind="owner",
            review_seconds=1,
        )


def test_incomplete_review_cannot_select_product_action(tmp_path: Path):
    corpus, tasks = _representative_inputs(tmp_path)
    study_dir = tmp_path / "study"
    run_study(corpus, tasks, study_dir)

    summary = summarize_study(study_dir)

    assert summary["decision"] == "INCOMPLETE"
    assert "REVIEW_INCOMPLETE" in summary["blockers"]
    assert "REVIEW_TIMING_INCOMPLETE" in summary["blockers"]


def test_frozen_study_detects_input_change(tmp_path: Path):
    corpus, tasks = _representative_inputs(tmp_path)
    study_dir = tmp_path / "study"
    run_study(corpus, tasks, study_dir)
    payload = json.loads(tasks.read_text(encoding="utf-8"))
    payload["tasks"][0]["query"] = "changed after freeze"
    tasks.write_text(json.dumps(payload), encoding="utf-8")

    summary = summarize_study(study_dir)

    assert summary["decision"] == "INCOMPLETE"
    assert "INPUT_OR_SOLVER_IDENTITY_CHANGED" in summary["blockers"]


def test_readiness_rejects_unreadable_docs_malformed_tasks_and_narrow_coverage(
    tmp_path: Path,
):
    corpus, tasks = _representative_inputs(tmp_path)
    (corpus / "private_note_9.md").write_text("   \n", encoding="utf-8")
    payload = json.loads(tasks.read_text(encoding="utf-8"))
    payload["tasks"][0]["query"] = 42
    for task in payload["tasks"]:
        task["doc_ids"] = ["private_note_0"]
    tasks.write_text(json.dumps(payload), encoding="utf-8")

    report = assess_inputs(corpus, tasks)

    assert report["representative_ready"] is False
    assert "UNREADABLE_OR_EMPTY_DOCUMENTS" in report["blockers"]
    assert "INVALID_TASK_DEFINITIONS" in report["blockers"]
    assert "TOO_FEW_SCOPED_DOCUMENTS" in report["blockers"]
    assert "COMPARE_SCOPE_TOO_SMALL" in report["blockers"]
    assert report["corpus"]["n_documents"] == 9
    assert report["tasks"]["n_unique_scoped_documents"] == 1


def test_frozen_study_detects_exact_card_and_manifest_tampering(tmp_path: Path):
    corpus, tasks = _representative_inputs(tmp_path)
    study_dir = tmp_path / "study"
    run_study(corpus, tasks, study_dir)
    cards_path = study_dir / "cards.json"
    payload = json.loads(cards_path.read_text(encoding="utf-8"))
    payload["cards"][0]["latency_s"] = 999.0
    cards_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _manifest, _cards, errors = verify_study(study_dir)
    summary = summarize_study(study_dir)

    assert "RESULT_ARTIFACT_CHANGED" in errors
    assert summary["decision"] == "INCOMPLETE"
    assert "coverage" not in summary


def test_frozen_study_detects_repeat_recall_state_tampering(tmp_path: Path):
    corpus, tasks = _representative_inputs(tmp_path)
    study_dir = tmp_path / "study"
    run_study(corpus, tasks, study_dir)
    saved_path = study_dir / "saved_questions.json"
    payload = json.loads(saved_path.read_text(encoding="utf-8"))
    payload["questions"][0]["query"] = "PRIVATE_TAMPER_ZXQ"
    saved_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    _manifest, _cards, errors = verify_study(study_dir)
    summary = summarize_study(study_dir)

    assert "RECALL_ARTIFACT_CHANGED" in errors
    assert summary["decision"] == "INCOMPLETE"
    assert "PRIVATE_TAMPER_ZXQ" not in json.dumps(summary, sort_keys=True)


def test_frozen_study_reaudits_coherently_forged_non_recall_claim(tmp_path: Path):
    corpus, tasks = _representative_inputs(tmp_path)
    study_dir = tmp_path / "study"
    run_study(corpus, tasks, study_dir)
    cards_path = study_dir / "cards.json"
    cards = json.loads(cards_path.read_text(encoding="utf-8"))["cards"]
    original_card = next(
        row
        for row in cards
        if row["task_class"] != "recall" and row.get("claims")
    )
    task_pack = json.loads(tasks.read_text(encoding="utf-8"))["tasks"]
    task = next(row for row in task_pack if row["id"] == original_card["task_id"])
    audited_path = study_dir / "audited_results.json"
    audited = json.loads(audited_path.read_text(encoding="utf-8"))
    raw_row = next(
        row for row in audited["results"] if row["task_id"] == task["id"]
    )
    forged_value = "PRIVATE_FORGED_NON_RECALL_/secret/claim"
    _forge_first_claim(raw_row["result"], forged_value)
    forged_card = _reproject_forged_card(task, raw_row["result"], corpus)
    cards[cards.index(original_card)] = forged_card
    _coherently_rewrite_audited_results(study_dir, audited)
    _coherently_rewrite_cards(study_dir, cards)

    _manifest, _cards, errors = verify_study(study_dir)
    summary = summarize_study(study_dir)
    serialized = json.dumps({"errors": errors, "summary": summary}, sort_keys=True)

    assert errors == ["FROZEN_RESULT_REAUDIT_FAILED"]
    assert summary["decision"] == "INCOMPLETE"
    assert forged_value not in serialized
    assert "/secret/claim" not in serialized


def test_frozen_study_reaudits_coherently_forged_recall_claim(tmp_path: Path):
    corpus, tasks = _representative_inputs(tmp_path)
    study_dir = tmp_path / "study"
    run_study(corpus, tasks, study_dir)
    cards_path = study_dir / "cards.json"
    cards = json.loads(cards_path.read_text(encoding="utf-8"))["cards"]
    original_card = next(row for row in cards if row["task_class"] == "recall")
    task_pack = json.loads(tasks.read_text(encoding="utf-8"))["tasks"]
    task = next(row for row in task_pack if row["id"] == original_card["task_id"])
    audited_path = study_dir / "audited_results.json"
    audited = json.loads(audited_path.read_text(encoding="utf-8"))
    raw_row = next(
        row for row in audited["results"] if row["task_id"] == task["id"]
    )
    forged_value = "PRIVATE_FORGED_RECALL_/secret/claim"
    _forge_first_claim(raw_row["result"], forged_value)
    card = _reproject_forged_card(task, raw_row["result"], corpus)
    cards[cards.index(original_card)] = card

    saved_path = study_dir / "saved_questions.json"
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    question = next(
        row for row in saved["questions"] if row["task_id"] == card["task_id"]
    )
    answer = question["verified_answer"]
    _forge_first_claim(answer, forged_value)
    question["last_result_digest"] = study_module.canonical_result_fingerprint(answer)
    card["provenance"]["result_fingerprint"] = (
        study_module.result_output_fingerprint(answer, card)
    )
    saved_path.write_text(
        json.dumps(saved, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _coherently_rewrite_audited_results(study_dir, audited)
    _coherently_rewrite_cards(study_dir, cards)
    manifest_path = study_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["result"]["saved_questions_digest"] = hashlib.sha256(
        saved_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    _manifest, _cards, errors = verify_study(study_dir)
    summary = summarize_study(study_dir)
    serialized = json.dumps({"errors": errors, "summary": summary}, sort_keys=True)

    assert errors == ["FROZEN_RESULT_REAUDIT_FAILED"]
    assert summary["decision"] == "INCOMPLETE"
    assert forged_value not in serialized
    assert "/secret/claim" not in serialized


def test_frozen_study_rejects_audit_valid_result_from_different_non_recall_query(
    tmp_path: Path,
):
    corpus, tasks = _representative_inputs(tmp_path)
    study_dir = tmp_path / "study"
    run_study(corpus, tasks, study_dir)
    task_pack = json.loads(tasks.read_text(encoding="utf-8"))["tasks"]
    task = next(row for row in task_pack if row["mode"] == "ask")
    scope = study_module.normalize_doc_ids(task["doc_ids"]) or []
    suffix = task["id"].rsplit("_", 1)[1]
    alternate = study_module.ask(
        f"TTL_{suffix}",
        corpus_dir=corpus,
        doc_ids=scope,
        persist_coe=False,
    )
    alternate["query"] = task["query"]
    live_docs = study_module.load_corpus(corpus)
    audit_input = json.loads(json.dumps(alternate))
    audit_input.pop("coe_audit", None)
    assert study_module.audit_payload(
        audit_input, {doc_id: live_docs[doc_id] for doc_id in scope}
    )["ok"] is True

    audited_path = study_dir / "audited_results.json"
    audited = json.loads(audited_path.read_text(encoding="utf-8"))
    raw_row = next(row for row in audited["results"] if row["task_id"] == task["id"])
    raw_row["result"] = alternate
    cards_path = study_dir / "cards.json"
    cards = json.loads(cards_path.read_text(encoding="utf-8"))["cards"]
    old_card = next(row for row in cards if row["task_id"] == task["id"])
    cards[cards.index(old_card)] = _reproject_forged_card(task, alternate, corpus)
    _coherently_rewrite_audited_results(study_dir, audited)
    _coherently_rewrite_cards(study_dir, cards)

    _manifest, _cards, errors = verify_study(study_dir)
    summary = summarize_study(study_dir)

    assert errors == ["FROZEN_RESULT_REAUDIT_FAILED"]
    assert summary["decision"] == "INCOMPLETE"
    assert "TTL_" not in json.dumps(summary, sort_keys=True)


def test_frozen_study_rejects_audit_valid_result_from_different_recall_query(
    tmp_path: Path,
):
    corpus, tasks = _representative_inputs(tmp_path)
    study_dir = tmp_path / "study"
    run_study(corpus, tasks, study_dir)
    task_pack = json.loads(tasks.read_text(encoding="utf-8"))["tasks"]
    task = next(row for row in task_pack if row["mode"] == "recall")
    scope = study_module.normalize_doc_ids(task["doc_ids"]) or []
    suffix = task["id"].rsplit("_", 1)[1]
    alternate = study_module.ask(
        f"TTL_{suffix}",
        corpus_dir=corpus,
        doc_ids=scope,
        persist_coe=False,
    )
    alternate["query"] = task["query"]
    live_docs = study_module.load_corpus(corpus)
    audit_input = json.loads(json.dumps(alternate))
    audit_input.pop("coe_audit", None)
    assert study_module.audit_payload(
        audit_input, {doc_id: live_docs[doc_id] for doc_id in scope}
    )["ok"] is True

    audited_path = study_dir / "audited_results.json"
    audited = json.loads(audited_path.read_text(encoding="utf-8"))
    raw_row = next(row for row in audited["results"] if row["task_id"] == task["id"])
    raw_row["result"] = alternate
    cards_path = study_dir / "cards.json"
    cards = json.loads(cards_path.read_text(encoding="utf-8"))["cards"]
    old_card = next(row for row in cards if row["task_id"] == task["id"])
    forged_card = _reproject_forged_card(task, alternate, corpus)
    cards[cards.index(old_card)] = forged_card

    saved_path = study_dir / "saved_questions.json"
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    question = next(row for row in saved["questions"] if row["task_id"] == task["id"])
    question["verified_answer"] = json.loads(json.dumps(alternate))
    question["last_result_digest"] = study_module.canonical_result_fingerprint(
        alternate
    )
    saved_path.write_text(
        json.dumps(saved, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _coherently_rewrite_audited_results(study_dir, audited)
    _coherently_rewrite_cards(study_dir, cards)
    manifest_path = study_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["result"]["saved_questions_digest"] = hashlib.sha256(
        saved_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    _manifest, _cards, errors = verify_study(study_dir)
    summary = summarize_study(study_dir)

    assert errors == ["FROZEN_RESULT_REAUDIT_FAILED"]
    assert summary["decision"] == "INCOMPLETE"
    assert "TTL_" not in json.dumps(summary, sort_keys=True)


def test_frozen_study_rejects_recall_saved_answer_sidecar_split_brain(
    tmp_path: Path,
):
    corpus, tasks = _representative_inputs(tmp_path)
    study_dir = tmp_path / "study"
    run_study(corpus, tasks, study_dir)
    task_pack = json.loads(tasks.read_text(encoding="utf-8"))["tasks"]
    task = next(row for row in task_pack if row["mode"] == "recall")
    cards_path = study_dir / "cards.json"
    cards = json.loads(cards_path.read_text(encoding="utf-8"))["cards"]
    original_card = next(row for row in cards if row["task_id"] == task["id"])
    saved_path = study_dir / "saved_questions.json"
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    question = next(row for row in saved["questions"] if row["task_id"] == task["id"])
    answer = question["verified_answer"]
    forged_value = "PRIVATE_SPLIT_BRAIN_/secret/saved"
    _forge_first_claim(answer, forged_value)
    question["last_result_digest"] = study_module.canonical_result_fingerprint(answer)
    forged_card = _reproject_forged_card(task, answer, corpus)
    cards[cards.index(original_card)] = forged_card
    saved_path.write_text(
        json.dumps(saved, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _coherently_rewrite_cards(study_dir, cards)
    manifest_path = study_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["result"]["saved_questions_digest"] = hashlib.sha256(
        saved_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    _manifest, _cards, errors = verify_study(study_dir)
    summary = summarize_study(study_dir)
    serialized = json.dumps({"errors": errors, "summary": summary}, sort_keys=True)

    assert errors == ["FROZEN_RESULT_REAUDIT_FAILED"]
    assert summary["decision"] == "INCOMPLETE"
    assert forged_value not in serialized
    assert "/secret/saved" not in serialized


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "invalid_json",
        "wrong_schema",
        "wrong_study_id",
        "duplicate_task",
        "missing_task",
        "extra_task",
        "wrong_task_class",
        "non_dict_result",
    ],
)
def test_frozen_study_fails_closed_for_malformed_audited_results(
    tmp_path: Path, case: str
):
    corpus, tasks = _representative_inputs(tmp_path)
    study_dir = tmp_path / "study"
    run_study(corpus, tasks, study_dir)
    audited_path = study_dir / "audited_results.json"
    audited = json.loads(audited_path.read_text(encoding="utf-8"))
    private_marker = "PRIVATE_BAD_AUDITED_/secret/raw"

    if case == "missing":
        audited_path.unlink()
    elif case == "invalid_json":
        audited_path.write_text("{" + private_marker, encoding="utf-8")
    else:
        if case == "wrong_schema":
            audited["schema"] = private_marker
        elif case == "wrong_study_id":
            audited["study_id"] = private_marker
        elif case == "duplicate_task":
            audited["results"].append(dict(audited["results"][0]))
        elif case == "missing_task":
            audited["results"].pop()
        elif case == "extra_task":
            audited["results"].append(
                {
                    "task_id": private_marker,
                    "task_class": "ask",
                    "result": {},
                }
            )
        elif case == "wrong_task_class":
            audited["results"][0]["task_class"] = "compare"
        elif case == "non_dict_result":
            audited["results"][0]["result"] = private_marker
        _coherently_rewrite_audited_results(study_dir, audited)

    if audited_path.is_file() and case == "invalid_json":
        manifest_path = study_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["result"]["audited_results_digest"] = hashlib.sha256(
            audited_path.read_bytes()
        ).hexdigest()
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    _manifest, _cards, errors = verify_study(study_dir)
    summary = summarize_study(study_dir)
    serialized = json.dumps({"errors": errors, "summary": summary}, sort_keys=True)

    assert "FROZEN_RESULT_REAUDIT_FAILED" in errors
    assert summary["decision"] == "INCOMPLETE"
    assert private_marker not in serialized
    assert "/secret/raw" not in serialized


def test_frozen_study_rejects_valid_evidence_from_out_of_scope_document(
    tmp_path: Path,
):
    corpus, tasks = _representative_inputs(tmp_path)
    study_dir = tmp_path / "study"
    run_study(corpus, tasks, study_dir)
    task_pack = json.loads(tasks.read_text(encoding="utf-8"))["tasks"]
    audited_path = study_dir / "audited_results.json"
    audited = json.loads(audited_path.read_text(encoding="utf-8"))
    raw_row = next(
        row
        for row in audited["results"]
        if row["task_class"] == "ask" and row["result"].get("claims")
    )
    task = next(row for row in task_pack if row["id"] == raw_row["task_id"])
    out_doc_id = next(
        f"private_note_{index}"
        for index in range(10)
        if f"private_note_{index}" not in task["doc_ids"]
    )
    out_text = (corpus / f"{out_doc_id}.md").read_text(encoding="utf-8")
    out_value = f"PRIVATE_MARKER_{out_doc_id.rsplit('_', 1)[1]}_ZXQ"
    start = out_text.index(out_value)
    end = start + len(out_value)
    result = raw_row["result"]
    public_claim = result["claims"][0]
    typed_claim = result["coe_claims"][0]
    atom = typed_claim["evidence_atoms"][0]
    atom.update(
        {
            "doc_id": out_doc_id,
            "doc_digest": digest_text(out_text),
            "start": start,
            "end": end,
            "text": out_value,
        }
    )
    typed_claim["raw_value"] = out_value
    typed_claim["source_doc_ids"] = [out_doc_id]
    public_claim["value"] = out_value
    public_claim["doc_id"] = out_doc_id
    public_claim["evidence"][0].update(
        {
            "atom_id": atom["atom_id"],
            "doc_id": out_doc_id,
            "doc_digest": atom["doc_digest"],
            "relation": atom["relation"],
            "start": start,
            "end": end,
            "text": out_value,
        }
    )
    cards_path = study_dir / "cards.json"
    cards = json.loads(cards_path.read_text(encoding="utf-8"))["cards"]
    original_card = next(row for row in cards if row["task_id"] == task["id"])
    cards[cards.index(original_card)] = _reproject_forged_card(task, result, corpus)
    _coherently_rewrite_audited_results(study_dir, audited)
    _coherently_rewrite_cards(study_dir, cards)

    _manifest, _cards, errors = verify_study(study_dir)
    summary = summarize_study(study_dir)

    assert errors == ["FROZEN_RESULT_REAUDIT_FAILED"]
    assert summary["decision"] == "INCOMPLETE"
    assert out_value not in json.dumps(summary, sort_keys=True)


def test_frozen_study_rejects_coherently_rewritten_non_dict_cards(tmp_path: Path):
    corpus, tasks = _representative_inputs(tmp_path)
    study_dir = tmp_path / "study"
    run_study(corpus, tasks, study_dir)
    _coherently_rewrite_cards(study_dir, [42])

    _manifest, _cards, errors = verify_study(study_dir)
    summary = summarize_study(study_dir)

    assert "CARDS_INVALID_SHAPE" in errors
    assert "RESULT_CARD_COUNT_MISMATCH" in errors
    assert "RESULT_AUDIT_INVARIANT_FAILED" in errors
    assert summary["decision"] == "INCOMPLETE"
    assert "PRIVATE_MARKER" not in json.dumps(summary, sort_keys=True)


def test_frozen_study_rejects_coherent_repeat_recall_removal(tmp_path: Path):
    corpus, tasks = _representative_inputs(tmp_path)
    study_dir = tmp_path / "study"
    run_study(corpus, tasks, study_dir)
    cards_path = study_dir / "cards.json"
    cards = json.loads(cards_path.read_text(encoding="utf-8"))["cards"]
    recall_card = next(card for card in cards if card["task_class"] == "recall")
    recall_card["task_class"] = "ask"
    _coherently_rewrite_cards(study_dir, cards)

    _manifest, _cards, errors = verify_study(study_dir)
    summary = summarize_study(study_dir)

    assert "RECALL_CARD_COUNT_MISMATCH" in errors
    assert "RECALL_INVARIANT_FAILED" in errors
    assert "CARD_MODE_COUNT_MISMATCH" in errors
    assert summary["decision"] == "INCOMPLETE"
    assert "PRIVATE_MARKER" not in json.dumps(summary, sort_keys=True)


def test_frozen_study_rejects_coherently_emptied_saved_recall_state(tmp_path: Path):
    corpus, tasks = _representative_inputs(tmp_path)
    study_dir = tmp_path / "study"
    run_study(corpus, tasks, study_dir)
    saved_path = study_dir / "saved_questions.json"
    saved_path.write_text(
        json.dumps(
            {"schema": "nano-lm.wedge_v1.saved_questions.v1", "questions": []},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = study_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["result"]["saved_questions_digest"] = hashlib.sha256(
        saved_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    _manifest, _cards, errors = verify_study(study_dir)
    summary = summarize_study(study_dir)

    assert "RECALL_ARTIFACT_CONTENT_MISMATCH" in errors
    assert summary["decision"] == "INCOMPLETE"
    assert "PRIVATE_MARKER" not in json.dumps(summary, sort_keys=True)


def test_manifest_summary_and_instrument_are_bound(tmp_path: Path, monkeypatch):
    corpus, tasks = _representative_inputs(tmp_path)
    study_dir = tmp_path / "study"
    run_study(corpus, tasks, study_dir)
    manifest_path = study_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["input_summary"]["corpus"]["n_documents"] = 999
    manifest["study_class"] = study_module.AGENT_APPLIED_SCOPED_PILOT
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(study_module, "_instrument_fingerprint", lambda: "f" * 64)

    _manifest, _cards, errors = verify_study(study_dir)

    assert "INPUT_SUMMARY_CHANGED" in errors
    assert "INPUT_OR_SOLVER_IDENTITY_CHANGED" in errors
    assert "STUDY_CONTRACT_CHANGED" in errors


def test_review_study_redacts_malformed_private_study_id_on_verify_error(
    tmp_path: Path,
):
    corpus, tasks = _representative_inputs(tmp_path)
    study_dir = tmp_path / "study"
    run_study(corpus, tasks, study_dir)
    manifest_path = study_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    private_id = "PRIVATE_LOCAL_PATH_/secret/doc.md"
    manifest["study_id"] = private_id
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    result = review_study(study_dir, reviewer_kind="owner")
    serialized = json.dumps(result, sort_keys=True)

    assert "STUDY_ID_LINK_MISMATCH" in result["blockers"]
    assert result["decision"] == "INCOMPLETE"
    assert result["study_id"] is None
    assert private_id not in serialized
    assert "/secret/doc.md" not in serialized


def test_private_malformed_input_pointers_fail_closed_without_leaking(tmp_path: Path):
    corpus, tasks = _representative_inputs(tmp_path)
    study_dir = tmp_path / "study"
    run_study(corpus, tasks, study_dir)
    manifest_path = study_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["local_inputs"] = {
        "corpus": "\x00PRIVATE_CORPUS_/secret/corpus",
        "tasks": "\x00PRIVATE_TASKS_/secret/tasks.json",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    _manifest, _cards, errors = verify_study(study_dir)
    summary = summarize_study(study_dir)
    serialized = json.dumps({"errors": errors, "summary": summary}, sort_keys=True)

    assert "INPUT_REASSESSMENT_FAILED" in errors
    assert summary["decision"] == "INCOMPLETE"
    assert "INPUT_REASSESSMENT_FAILED" in summary["blockers"]
    assert "PRIVATE_CORPUS" not in serialized
    assert "PRIVATE_TASKS" not in serialized
    assert "/secret/" not in serialized


def test_frozen_study_detects_solver_implementation_drift(tmp_path: Path, monkeypatch):
    corpus, tasks = _representative_inputs(tmp_path)
    study_dir = tmp_path / "study"
    run_study(corpus, tasks, study_dir)
    monkeypatch.setattr(
        study_module,
        "solver_implementation_fingerprint",
        lambda: "e" * 64,
    )

    _manifest, _cards, errors = verify_study(study_dir)

    assert "INPUT_OR_SOLVER_IDENTITY_CHANGED" in errors


def test_rerun_cannot_overwrite_frozen_check(tmp_path: Path):
    corpus, tasks = _representative_inputs(tmp_path)
    study_dir = tmp_path / "study"
    run_study(corpus, tasks, study_dir)
    check_before = (study_dir / "check.json").read_bytes()
    payload = json.loads(tasks.read_text(encoding="utf-8"))
    payload["tasks"][0]["query"] = "changed after freeze"
    tasks.write_text(json.dumps(payload), encoding="utf-8")

    result = run_study(corpus, tasks, study_dir)

    assert result["blockers"] == ["STUDY_ALREADY_FROZEN_USE_NEW_DIRECTORY"]
    assert (study_dir / "check.json").read_bytes() == check_before


def test_summary_rejects_tracked_study_directory_without_writing():
    invalid = Path(__file__).resolve().parent / "tracked-study-probe"
    assert not invalid.exists()

    summary = summarize_study(invalid)

    assert summary["blockers"] == ["STUDY_DIR_NOT_PRIVATE_LOCATION"]
    assert not invalid.exists()


def test_failure_labels_require_reason_and_correction(tmp_path: Path):
    corpus, tasks = _representative_inputs(tmp_path)
    study_dir = tmp_path / "study"
    run_study(corpus, tasks, study_dir)
    cards = json.loads((study_dir / "cards.json").read_text(encoding="utf-8"))["cards"]
    review_path = study_dir / "review.json"
    state = load_state(review_path)
    for index, card in enumerate(cards):
        apply_label(
            state,
            card,
            "WRONG_EVIDENCE" if index < 2 else "USEFUL",
            reviewer_kind="owner",
            review_elapsed_s=1,
        )
    save_state(state, review_path)

    summary = summarize_study(study_dir)

    assert summary["decision"] == "INCOMPLETE"
    assert "FAILURE_DETAIL_INCOMPLETE" in summary["blockers"]
    assert summary["outcomes"]["first_repeated_failure_class"] is None


@pytest.mark.parametrize(
    ("review_bytes", "blocker"),
    [
        (b"{not-json\n", "REVIEW_STATE_INVALID_SHAPE"),
        (b'{"cards": [], "labels": {}}\n', "REVIEW_STATE_INVALID_SHAPE"),
    ],
)
def test_malformed_review_state_blocks_without_overwrite(
    tmp_path: Path, review_bytes: bytes, blocker: str
):
    corpus, tasks = _representative_inputs(tmp_path)
    study_dir = tmp_path / "study"
    run_study(corpus, tasks, study_dir)
    review_path = study_dir / "review.json"
    review_path.write_bytes(review_bytes)

    summary = summarize_study(study_dir)
    review_result = review_study(study_dir, reviewer_kind="owner")

    assert blocker in summary["blockers"]
    assert blocker in review_result["blockers"]
    assert review_path.read_bytes() == review_bytes


@pytest.mark.parametrize(
    "injected_load_errors",
    [
        ["PRIVATE_REVIEW_MARKER_/secret/review.json"],
        [{"PRIVATE_REVIEW_MARKER": ["/secret/review.json"]}],
    ],
)
def test_untrusted_review_load_errors_are_redacted_and_cannot_crash(
    tmp_path: Path, injected_load_errors: object
):
    corpus, tasks = _representative_inputs(tmp_path)
    study_dir = tmp_path / "study"
    run_study(corpus, tasks, study_dir)
    review_path = study_dir / "review.json"
    review_path.write_text(
        json.dumps(
            {
                "schema": "nano-lm.wedge_v1.review.v1",
                "cards": {},
                "labels": {},
                "load_errors": injected_load_errors,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_study(study_dir)
    review_result = review_study(study_dir, reviewer_kind="owner")

    for result in (summary, review_result):
        serialized = json.dumps(result, sort_keys=True)
        assert result["blockers"] == ["REVIEW_STATE_INVALID_SHAPE"]
        assert "PRIVATE_REVIEW_MARKER" not in serialized
        assert "/secret/review.json" not in serialized


def test_study_cli_check_and_run_use_explicit_directory(tmp_path: Path):
    corpus, tasks = _representative_inputs(tmp_path)
    study_dir = tmp_path / "cli-study"

    assert cli_main(
        [
            "study",
            "check",
            "--corpus",
            str(corpus),
            "--tasks",
            str(tasks),
            "--dir",
            str(study_dir),
        ]
    ) == 0
    assert (study_dir / "check.json").is_file()
    assert cli_main(
        [
            "study",
            "run",
            "--corpus",
            str(corpus),
            "--tasks",
            str(tasks),
            "--dir",
            str(study_dir),
        ]
    ) == 0
    assert (study_dir / "manifest.json").is_file()
    assert (study_dir / "cards.json").is_file()
