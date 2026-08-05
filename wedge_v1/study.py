"""Isolated usefulness studies for the internal Wedge v1 pipeline.

Private inputs and per-question artifacts remain inside an explicit local study
directory. The exported summary is content-free and authorizes only the next
Wedge-development step; it is not scientific or Nano-capability evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wedge_v1.coe.audit import audit_payload
from wedge_v1.coe.canonical import canonical_result_fingerprint
from wedge_v1.habit import (
    RECALL_SCHEMA,
    recall_saved,
    save_question,
    solver_implementation_fingerprint,
)
from wedge_v1.ingest import SUPPORTED_SUFFIXES, document_id, load_corpus
from wedge_v1.review import (
    FAILURE_CLASSES,
    FAILURE_LABELS,
    PROVENANCE_SCHEMA,
    REVIEWER_KINDS,
    card_from_result,
    corpus_content_digest,
    interactive_review,
    load_state,
    merge_prior_labels,
    provenance_record,
    result_output_fingerprint,
    task_fingerprint,
)
from wedge_v1.runtime import ask, compare, find_spans, normalize_doc_ids
from wedge_v1.private_output import PRIVATE_TASK_ROOT, private_task_pack_allowed
from wedge_v1.study_capture import OWNER_PRIVATE, TASK_PACK_SCHEMA

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
PRIVATE_STUDY_ROOT = ROOT / ".studies"
PRIVATE_CORPUS_ROOT = ROOT / "data" / "owner_corpus"
EXAMPLE_TASK_PACK = ROOT / "data" / "owner_dogfood_tasks.example.json"

MIN_DOCUMENTS = 10
MAX_DOCUMENTS = 50
MAX_EXTRACTED_BYTES = 5_000_000
MIN_TASKS = 10
MAX_TASKS = 20
ALLOWED_MODES = ("ask", "find", "compare", "recall")

REPRESENTATIVE_USE = "REPRESENTATIVE_USE"
AGENT_APPLIED_SCOPED_PILOT = "AGENT_APPLIED_SCOPED_PILOT"
ALLOWED_STUDY_CLASSES = frozenset(
    {REPRESENTATIVE_USE, AGENT_APPLIED_SCOPED_PILOT}
)

CHECK_SCHEMA = "nano-lm.wedge_v1.study_check.v1"
MANIFEST_SCHEMA = "nano-lm.wedge_v1.study_manifest.v1"
CARDS_SCHEMA = "nano-lm.wedge_v1.study_cards.v1"
AUDITED_RESULTS_SCHEMA = "nano-lm.wedge_v1.study_audited_results.v1"
SUMMARY_SCHEMA = "nano-lm.wedge_v1.study_summary.v1"
SAVED_QUESTIONS_SCHEMA = "nano-lm.wedge_v1.saved_questions.v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_digest(path: Path) -> str | None:
    return _sha256_bytes(path.read_bytes()) if path.is_file() else None


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _is_demo_corpus(path: Path) -> bool:
    resolved = path.resolve()
    known = (
        ROOT / "fixtures" / "owner_corpus",
        ROOT / "data" / "corpus",
        ROOT / "data" / "fixtures",
    )
    return any(resolved == item.resolve() for item in known if item.exists())


def _private_task_location(path: Path) -> bool:
    return private_task_pack_allowed(path)


def _private_corpus_location(path: Path) -> bool:
    resolved = path.resolve()
    return not _inside(resolved, REPO_ROOT) or _inside(resolved, PRIVATE_CORPUS_ROOT)


def valid_study_directory(path: Path) -> bool:
    """Allow outside-repo directories or the dedicated ignored study root."""
    resolved = path.resolve()
    return not _inside(resolved, REPO_ROOT) or _inside(resolved, PRIVATE_STUDY_ROOT)


def _visible_files(corpus: Path) -> list[Path]:
    if not corpus.is_dir():
        return []
    return [
        path
        for path in corpus.rglob("*")
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(corpus).parts)
    ]


def _instrument_fingerprint() -> str:
    """Bind a study to the exact lifecycle and review-instrument bytes."""
    digest = hashlib.sha256()
    for path in (Path(__file__), ROOT / "review.py"):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _pdf_extractor_identity(n_pdf: int) -> str:
    if not n_pdf:
        return "NOT_USED"
    try:
        import pypdf
    except Exception:
        return "UNAVAILABLE"
    version = str(getattr(pypdf, "__version__", "UNKNOWN")).strip() or "UNKNOWN"
    return f"pypdf:{version}"


def _blocked_summary(blockers: object, *, study_id: object = None) -> dict:
    if isinstance(blockers, (list, tuple, set, frozenset)):
        raw_blockers = list(blockers)
        safe_blockers = {
            value for value in raw_blockers if isinstance(value, str) and value
        }
        if any(not isinstance(value, str) or not value for value in raw_blockers):
            safe_blockers.add("BLOCKER_METADATA_INVALID")
    else:
        safe_blockers = {"BLOCKER_METADATA_INVALID"}
    safe_id = (
        study_id
        if isinstance(study_id, str)
        and len(study_id) == 64
        and all(char in "0123456789abcdef" for char in study_id.lower())
        else None
    )
    return {
        "schema": SUMMARY_SCHEMA,
        "study_id": safe_id,
        "status": "INCOMPLETE",
        "decision": "INCOMPLETE",
        "blockers": sorted(safe_blockers),
        "next_action": "repair the frozen study or create a new isolated study directory",
        "claim_boundary": (
            "This blocked aggregate exposes no study content and authorizes no "
            "Nano capability or evidence claim."
        ),
    }


def assess_inputs(
    corpus: Path,
    tasks: Path,
    *,
    demo: bool = False,
    solver_fingerprint: str | None = None,
    instrument_fingerprint: str | None = None,
) -> dict:
    """Return a content-free readiness report for one proposed study."""
    corpus = Path(corpus).expanduser()
    tasks = Path(tasks).expanduser()
    solver_fingerprint = solver_fingerprint or solver_implementation_fingerprint()
    instrument_fingerprint = instrument_fingerprint or _instrument_fingerprint()
    blockers: list[str] = []
    warnings: list[str] = []

    visible = _visible_files(corpus)
    supported = [path for path in visible if path.suffix.lower() in SUPPORTED_SUFFIXES]
    unsupported = [path for path in visible if path.suffix.lower() not in SUPPORTED_SUFFIXES]
    suffix_counts = Counter(
        "markdown"
        if path.suffix.lower() in {".md", ".markdown"}
        else "txt"
        if path.suffix.lower() == ".txt"
        else "pdf"
        if path.suffix.lower() == ".pdf"
        else "other"
        for path in visible
    )

    docs: dict[str, str] = {}
    corpus_load_ok = corpus.is_dir()
    if not corpus_load_ok:
        blockers.append("CORPUS_PATH_MISSING")
    else:
        try:
            docs = load_corpus(corpus)
        except Exception:
            corpus_load_ok = False
            blockers.append("CORPUS_LOAD_ERROR")

    readable_docs = {
        doc_id: text
        for doc_id, text in docs.items()
        if isinstance(text, str) and text.strip()
    }
    expected_doc_ids = {document_id(path, corpus) for path in supported}
    unreadable_or_empty = expected_doc_ids - set(readable_docs)
    n_docs = len(readable_docs)
    extracted_bytes = sum(len(value.encode("utf-8")) for value in readable_docs.values())
    n_pdf = suffix_counts.get("pdf", 0)
    pdf_extractor = _pdf_extractor_identity(n_pdf)
    try:
        import pypdf  # noqa: F401

        pypdf_available = True
    except Exception:
        pypdf_available = False

    smoke_ready = bool(corpus_load_ok and n_docs)
    if not docs and corpus.is_dir() and "CORPUS_LOAD_ERROR" not in blockers:
        blockers.append("CORPUS_EMPTY_OR_UNREADABLE")
    if unsupported:
        blockers.append("UNSUPPORTED_CORPUS_FILES")
    if unreadable_or_empty:
        blockers.append("UNREADABLE_OR_EMPTY_DOCUMENTS")
    if n_pdf and not pypdf_available:
        blockers.append("PDF_DEPENDENCY_MISSING")
    if n_docs < MIN_DOCUMENTS:
        blockers.append("TOO_FEW_DOCUMENTS")
    if n_docs > MAX_DOCUMENTS:
        blockers.append("TOO_MANY_DOCUMENTS")
    if extracted_bytes > MAX_EXTRACTED_BYTES:
        blockers.append("CORPUS_TEXT_TOO_LARGE")
    if demo or _is_demo_corpus(corpus):
        blockers.append("DEMO_OR_FIXTURE_CORPUS")
    if not _private_corpus_location(corpus):
        blockers.append("CORPUS_NOT_PRIVATE_LOCATION")

    task_rows: list[dict] = []
    canonical_task_pack = False
    study_class: str | None = REPRESENTATIVE_USE
    task_pack_ok = tasks.is_file()
    if not task_pack_ok:
        blockers.append("TASK_PACK_MISSING")
    else:
        try:
            loaded = json.loads(tasks.read_text(encoding="utf-8"))
            raw_rows = loaded.get("tasks") if isinstance(loaded, dict) else None
            if not isinstance(raw_rows, list):
                raise ValueError("tasks must be a list")
            canonical_task_pack = loaded.get("schema") == TASK_PACK_SCHEMA
            if canonical_task_pack and loaded.get("storage_class") != OWNER_PRIVATE:
                blockers.append("TASK_PACK_STORAGE_CLASS_INVALID")
            if "study_class" in loaded:
                declared_study_class = loaded.get("study_class")
                if (
                    not canonical_task_pack
                    or not isinstance(declared_study_class, str)
                    or declared_study_class not in ALLOWED_STUDY_CLASSES
                ):
                    study_class = None
                    blockers.append("TASK_PACK_STUDY_CLASS_INVALID")
                else:
                    study_class = declared_study_class
            task_rows = [row for row in raw_rows if isinstance(row, dict)]
            if len(task_rows) != len(raw_rows):
                blockers.append("INVALID_TASK_ROWS")
        except Exception:
            task_pack_ok = False
            blockers.append("TASK_PACK_INVALID_JSON")

    if tasks.resolve() == EXAMPLE_TASK_PACK.resolve():
        blockers.append("EXAMPLE_TASK_PACK")
    if not _private_task_location(tasks):
        blockers.append("TASK_PACK_NOT_PRIVATE_LOCATION")

    n_tasks = len(task_rows)
    if n_tasks < MIN_TASKS:
        blockers.append("TOO_FEW_TASKS")
    if n_tasks > MAX_TASKS:
        blockers.append("TOO_MANY_TASKS")

    known_doc_ids = set(readable_docs)
    mode_counts: Counter[str] = Counter()
    ids: list[str] = []
    definition_digests: list[str] = []
    scoped_count = 0
    missing_scope_refs = 0
    manual_baseline_count = 0
    expected_status_count = 0
    invalid_rows = 0
    scoped_doc_ids: set[str] = set()
    for task in task_rows:
        raw_task_id = task.get("id")
        raw_query = task.get("query")
        raw_mode = task.get("mode", "ask")
        task_id = raw_task_id.strip() if isinstance(raw_task_id, str) else ""
        query = raw_query.strip() if isinstance(raw_query, str) else ""
        mode = raw_mode.strip().lower() if isinstance(raw_mode, str) else ""
        raw_scope = task.get("doc_ids")
        if not task_id or not query or mode not in ALLOWED_MODES:
            invalid_rows += 1
        ids.append(task_id)
        mode_counts[mode] += 1
        if not isinstance(raw_scope, list) or not raw_scope:
            invalid_rows += 1
            scope = []
        elif any(not isinstance(value, str) or not value.strip() for value in raw_scope):
            invalid_rows += 1
            scope = []
        else:
            scope = normalize_doc_ids(raw_scope) or []
            scoped_count += 1
            missing_scope_refs += sum(1 for value in scope if value not in known_doc_ids)
            scoped_doc_ids.update(value for value in scope if value in known_doc_ids)
            if mode == "compare" and len(scope) < 2:
                blockers.append("COMPARE_SCOPE_TOO_SMALL")
        projection = {
            "mode": mode,
            "query": query,
            "doc_ids": scope,
        }
        definition_digests.append(
            _sha256_bytes(
                json.dumps(projection, sort_keys=True, separators=(",", ":")).encode("utf-8")
            )
        )
        baseline = task.get("manual_baseline_seconds")
        if (
            isinstance(baseline, (int, float))
            and not isinstance(baseline, bool)
            and math.isfinite(float(baseline))
            and baseline > 0
        ):
            manual_baseline_count += 1
        if isinstance(task.get("expect_status"), list) and task.get("expect_status"):
            expected_status_count += 1

    if invalid_rows:
        blockers.append("INVALID_TASK_DEFINITIONS")
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        blockers.append("TASK_IDS_NOT_UNIQUE")
    if len(definition_digests) != len(set(definition_digests)):
        blockers.append("DUPLICATE_TASK_DEFINITIONS")
    if missing_scope_refs:
        blockers.append("UNKNOWN_TASK_SCOPE_REFERENCES")
    if scoped_count != n_tasks:
        blockers.append("TASK_SCOPE_REQUIRED")
    if len(scoped_doc_ids) < MIN_DOCUMENTS:
        blockers.append("TOO_FEW_SCOPED_DOCUMENTS")
    if mode_counts.get("recall", 0) < 1:
        blockers.append("REPEAT_RECALL_REQUIRED")
    agent_applied_pilot = study_class == AGENT_APPLIED_SCOPED_PILOT
    if manual_baseline_count != n_tasks:
        (blockers if canonical_task_pack and not agent_applied_pilot else warnings).append(
            "MANUAL_BASELINE_INCOMPLETE"
        )
    if expected_status_count != n_tasks:
        (blockers if canonical_task_pack else warnings).append(
            "EXPECTED_STATUS_INCOMPLETE"
        )

    corpus_digest = corpus_content_digest(corpus) if supported else None
    task_pack_digest = _file_digest(tasks)
    identity = {
        "corpus_digest": corpus_digest,
        "task_pack_digest": task_pack_digest,
        "solver_fingerprint": solver_fingerprint,
        "instrument_fingerprint": instrument_fingerprint,
        "pdf_extractor": pdf_extractor,
    }
    study_id = (
        canonical_result_fingerprint(identity)
        if all(identity.values())
        else None
    )
    blockers = sorted(set(blockers))
    study_ready = smoke_ready and task_pack_ok and not blockers
    representative_ready = study_ready and study_class == REPRESENTATIVE_USE
    required_reviewer_kind = "agent_applied" if agent_applied_pilot else None
    manual_time_comparison_enabled = bool(
        not agent_applied_pilot and manual_baseline_count == n_tasks and n_tasks
    )
    claim_boundary = (
        "This agent-applied scoped pilot selects only the next Wedge component-development "
        "workflow step. It is not representative-use evidence; manual time "
        "comparison is disabled and no time-saved claim is supported."
        if agent_applied_pilot
        else (
            "Readiness permits a local usefulness study only; it is not evidence of "
            "scientific validity or Nano AI capability."
        )
    )
    return {
        "schema": CHECK_SCHEMA,
        "study_class": study_class,
        "smoke_ready": smoke_ready,
        "study_ready": study_ready,
        "representative_ready": representative_ready,
        "required_reviewer_kind": required_reviewer_kind,
        "manual_time_comparison_enabled": manual_time_comparison_enabled,
        "blockers": blockers,
        "warnings": sorted(set(warnings)),
        "corpus": {
            "n_documents": n_docs,
            "extracted_text_bytes": extracted_bytes,
            "format_counts": dict(sorted(suffix_counts.items())),
            "n_unsupported_files": len(unsupported),
            "n_unreadable_or_empty_files": len(unreadable_or_empty),
            "pypdf_available": pypdf_available,
            "pdf_extractor": pdf_extractor,
        },
        "tasks": {
            "n_tasks": n_tasks,
            "n_exactly_scoped": scoped_count,
            "n_unknown_scope_references": missing_scope_refs,
            "n_unique_scoped_documents": len(scoped_doc_ids),
            "scope_coverage_fraction": (
                round(len(scoped_doc_ids) / n_docs, 4) if n_docs else 0.0
            ),
            "mode_counts": dict(sorted(mode_counts.items())),
            "manual_baseline_count": manual_baseline_count,
            "expected_status_count": expected_status_count,
            "authenticity": "DECLARED_BY_OPERATOR_NOT_INFERRED",
        },
        "identity": identity,
        "study_id": study_id,
        "limits": {
            "documents": [MIN_DOCUMENTS, MAX_DOCUMENTS],
            "tasks": [MIN_TASKS, MAX_TASKS],
            "max_extracted_text_bytes": MAX_EXTRACTED_BYTES,
        },
        "claim_boundary": claim_boundary,
    }


def check_study(corpus: Path, tasks: Path, study_dir: Path, *, demo: bool = False) -> dict:
    report = assess_inputs(corpus, tasks, demo=demo)
    if not valid_study_directory(study_dir):
        report = dict(report)
        report["blockers"] = sorted(set(report["blockers"] + ["STUDY_DIR_NOT_PRIVATE_LOCATION"]))
        report["study_ready"] = False
        report["representative_ready"] = False
        return report
    _write_json(Path(study_dir) / "check.json", report)
    return report


class _RepeatRecallInvariantError(RuntimeError):
    """Internal fail-closed signal; its details must never enter safe summaries."""


def _recall_phase_metadata(
    payload: object,
    *,
    expected_state: str,
    expected_scope: list[str],
    expected_cache_hits: int,
    expected_forced_refreshes: int,
    expected_avoided_solver_runs: int,
    expected_solver_runs: int,
) -> tuple[dict, dict]:
    if not isinstance(payload, dict) or payload.get("schema") != RECALL_SCHEMA:
        raise _RepeatRecallInvariantError("invalid recall payload")
    aggregate = payload.get("aggregate")
    answer = payload.get("answer")
    if not isinstance(aggregate, dict) or not isinstance(answer, dict):
        raise _RepeatRecallInvariantError("missing recall aggregate or answer")
    audit = answer.get("coe_audit")
    audit_ok = isinstance(audit, dict) and audit.get("ok") is True
    checks = (
        payload.get("recall_state") == expected_state,
        payload.get("failure_codes") == [],
        payload.get("selected_doc_ids") == expected_scope,
        payload.get("missing_doc_ids") == [],
        answer.get("selected_doc_ids") == expected_scope,
        answer.get("missing_doc_ids") == [],
        audit_ok,
        aggregate.get("cache_hits") == expected_cache_hits,
        aggregate.get("forced_refreshes") == expected_forced_refreshes,
        aggregate.get("avoided_solver_runs") == expected_avoided_solver_runs,
        aggregate.get("solver_runs") == expected_solver_runs,
    )
    if not all(checks):
        raise _RepeatRecallInvariantError("repeat recall phase invariant failed")
    return answer, {
        "recall_state": expected_state,
        "cache_hits": expected_cache_hits,
        "forced_refreshes": expected_forced_refreshes,
        "avoided_solver_runs": expected_avoided_solver_runs,
        "solver_runs": expected_solver_runs,
        "audit_ok": True,
    }


def _repeat_recall_card(
    task: dict,
    *,
    corpus: Path,
    saved_path: Path,
) -> tuple[dict, dict]:
    query = str(task["query"]).strip()
    task_id = str(task["id"]).strip()
    scope = normalize_doc_ids(task.get("doc_ids")) or []
    if not scope:
        raise _RepeatRecallInvariantError("repeat recall scope missing")
    save_question(
        query,
        mode="ask",
        task_id=task_id,
        corpus=corpus,
        doc_ids=scope,
        path=saved_path,
    )
    first = recall_saved(
        task_id,
        corpus,
        doc_ids=scope,
        path=saved_path,
        persist_coe=False,
    )
    first_answer, first_metadata = _recall_phase_metadata(
        first,
        expected_state="REFRESHED",
        expected_scope=scope,
        expected_cache_hits=0,
        expected_forced_refreshes=1,
        expected_avoided_solver_runs=0,
        expected_solver_runs=1,
    )
    second = recall_saved(
        task_id,
        corpus,
        doc_ids=scope,
        path=saved_path,
        persist_coe=False,
    )
    second_answer, second_metadata = _recall_phase_metadata(
        second,
        expected_state="CACHE_HIT",
        expected_scope=scope,
        expected_cache_hits=1,
        expected_forced_refreshes=0,
        expected_avoided_solver_runs=1,
        expected_solver_runs=0,
    )
    answers_match = canonical_result_fingerprint(
        first_answer
    ) == canonical_result_fingerprint(second_answer)
    if not answers_match:
        raise _RepeatRecallInvariantError("repeat recall answer changed")
    scoped_digest = corpus_content_digest(corpus, doc_ids=scope)
    card = card_from_result(
        query,
        second_answer,
        corpus=corpus,
        mode="recall",
        task_id=task_id,
        expect_status=task.get("expect_status"),
        corpus_digest=scoped_digest,
        doc_ids=scope,
        manual_baseline_seconds=task.get("manual_baseline_seconds"),
    )
    card["repeat_recall"] = {
        "first": first_metadata,
        "second": second_metadata,
        "answer_fingerprint_match": True,
    }
    card["provenance"] = provenance_record(
        query,
        corpus=corpus,
        corpus_digest=scoped_digest,
        mode="recall",
        task_id=task_id,
        expect_status=task.get("expect_status"),
        result_fingerprint=result_output_fingerprint(second_answer, card),
        doc_ids=scope,
    )
    return card, deepcopy(second_answer)


def _execute_study_result(
    query: str,
    *,
    mode: str,
    corpus: Path,
    scope: list[str],
) -> dict:
    if mode == "compare":
        return compare(query, corpus_dir=corpus, doc_ids=scope, persist_coe=False)
    if mode == "find":
        return find_spans(query, corpus_dir=corpus, doc_ids=scope, persist_coe=False)
    return ask(query, corpus_dir=corpus, doc_ids=scope, persist_coe=False)


def _study_cards(
    tasks_path: Path,
    corpus: Path,
    study_dir: Path,
) -> tuple[list[dict], list[dict]]:
    pack = json.loads(tasks_path.read_text(encoding="utf-8"))
    cards = []
    audited_results = []
    saved_path = study_dir / "saved_questions.json"
    for task in pack.get("tasks") or []:
        mode = str(task.get("mode") or "ask").strip().lower()
        query = str(task.get("query") or "").strip()
        task_id = str(task.get("id") or "").strip()
        scope = normalize_doc_ids(task.get("doc_ids")) or []
        if mode == "recall":
            card, result = _repeat_recall_card(
                task, corpus=corpus, saved_path=saved_path
            )
        else:
            result = _execute_study_result(
                query, mode=mode, corpus=corpus, scope=scope
            )
            card = card_from_result(
                query,
                result,
                corpus=corpus,
                mode=mode,
                task_id=task_id,
                expect_status=task.get("expect_status"),
                corpus_digest=corpus_content_digest(corpus, doc_ids=scope),
                doc_ids=scope,
                manual_baseline_seconds=task.get("manual_baseline_seconds"),
            )
        cards.append(card)
        audited_results.append(
            {
                "task_id": task_id,
                "task_class": mode,
                "result": deepcopy(result),
            }
        )
    return cards, audited_results


def run_study(corpus: Path, tasks: Path, study_dir: Path, *, demo: bool = False) -> dict:
    study_dir = Path(study_dir)
    if not valid_study_directory(study_dir):
        return {
            "schema": SUMMARY_SCHEMA,
            "status": "BLOCKED",
            "decision": "INCOMPLETE",
            "study_id": None,
            "blockers": ["STUDY_DIR_NOT_PRIVATE_LOCATION"],
        }

    manifest_path = study_dir / "manifest.json"
    cards_path = study_dir / "cards.json"
    audited_results_path = study_dir / "audited_results.json"
    saved_path = study_dir / "saved_questions.json"
    if (
        manifest_path.exists()
        or cards_path.exists()
        or audited_results_path.exists()
        or saved_path.exists()
    ):
        return {
            "schema": SUMMARY_SCHEMA,
            "status": "BLOCKED",
            "decision": "INCOMPLETE",
            "study_id": None,
            "blockers": ["STUDY_ALREADY_FROZEN_USE_NEW_DIRECTORY"],
        }

    report = check_study(corpus, tasks, study_dir, demo=demo)
    if not report["study_ready"]:
        return {
            "schema": SUMMARY_SCHEMA,
            "status": "BLOCKED",
            "decision": "INCOMPLETE",
            "study_id": report.get("study_id"),
            "study_class": report.get("study_class"),
            "study_ready": False,
            "representative_ready": False,
            "blockers": report["blockers"],
        }

    try:
        cards, audited_results = _study_cards(Path(tasks), Path(corpus), study_dir)
    except _RepeatRecallInvariantError:
        return {
            "schema": SUMMARY_SCHEMA,
            "status": "BLOCKED",
            "decision": "INCOMPLETE",
            "study_id": report.get("study_id"),
            "blockers": ["REPEAT_RECALL_INVARIANT_FAILED"],
        }
    recall_cards = [card for card in cards if card.get("task_class") == "recall"]
    recall_invariants_ok = bool(recall_cards) and all(
        _valid_repeat_recall_card(card) for card in recall_cards
    )
    if not recall_invariants_ok:
        return {
            "schema": SUMMARY_SCHEMA,
            "status": "BLOCKED",
            "decision": "INCOMPLETE",
            "study_id": report.get("study_id"),
            "blockers": ["REPEAT_RECALL_INVARIANT_FAILED"],
        }
    saved_questions_digest = _file_digest(saved_path)
    cards_payload = {
        "schema": CARDS_SCHEMA,
        "study_id": report["study_id"],
        "cards": cards,
    }
    audited_results_payload = {
        "schema": AUDITED_RESULTS_SCHEMA,
        "study_id": report["study_id"],
        "results": audited_results,
    }
    all_audits_ok = all(card.get("coe_audit_ok") is True for card in cards)
    _write_json(cards_path, cards_payload)
    _write_json(audited_results_path, audited_results_payload)
    result_digest = _file_digest(cards_path)
    audited_results_digest = _file_digest(audited_results_path)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "study_id": report["study_id"],
        "study_class": report["study_class"],
        "representative_ready": report["representative_ready"],
        "required_reviewer_kind": report["required_reviewer_kind"],
        "manual_time_comparison_enabled": report[
            "manual_time_comparison_enabled"
        ],
        "created_at": _now(),
        "local_inputs": {
            "corpus": str(Path(corpus).expanduser().resolve()),
            "tasks": str(Path(tasks).expanduser().resolve()),
        },
        "identity": report["identity"],
        "input_summary": {
            "corpus": report["corpus"],
            "tasks": report["tasks"],
        },
        "result": {
            "digest": result_digest,
            "n_cards": len(cards),
            "all_coe_audits_ok": all_audits_ok,
            "n_repeat_recall_cards": len(recall_cards),
            "all_repeat_recall_invariants_ok": recall_invariants_ok,
            "saved_questions_digest": saved_questions_digest,
            "audited_results_digest": audited_results_digest,
        },
        "artifacts": {
            "cards": "cards.json",
            "audited_results": "audited_results.json",
            "review": "review.json",
            "saved_questions": "saved_questions.json",
            "safe_summary": "summary.json",
        },
        "claim_boundary": report["claim_boundary"],
    }
    _write_json(manifest_path, manifest)
    return {
        "schema": SUMMARY_SCHEMA,
        "status": "COMPLETE" if all_audits_ok else "BLOCKED",
        "decision": "REVIEW_REQUIRED" if all_audits_ok else "INCOMPLETE",
        "study_id": report["study_id"],
        "study_class": report["study_class"],
        "study_ready": True,
        "representative_ready": report["representative_ready"],
        "required_reviewer_kind": report["required_reviewer_kind"],
        **(
            {"time_saved_claim_supported": False}
            if report["study_class"] == AGENT_APPLIED_SCOPED_PILOT
            else {}
        ),
        "claim_boundary": report["claim_boundary"],
        "n_tasks": len(cards),
        "n_repeat_recall_tasks": len(recall_cards),
        "all_coe_audits_ok": all_audits_ok,
        "blockers": [] if all_audits_ok else ["COE_AUDIT_FAILURE"],
    }


def _valid_repeat_recall_card(card: object) -> bool:
    if not isinstance(card, dict) or card.get("task_class") != "recall":
        return False
    metadata = card.get("repeat_recall")
    provenance = card.get("provenance")
    if not isinstance(metadata, dict) or not isinstance(provenance, dict):
        return False
    return (
        metadata == _expected_repeat_recall_metadata()
        and card.get("coe_audit_ok") is True
        and card.get("selected_doc_ids") == provenance.get("doc_ids")
        and card.get("missing_doc_ids") == []
    )


def _task_summary(manifest: dict) -> dict:
    input_summary = manifest.get("input_summary")
    if not isinstance(input_summary, dict):
        return {}
    tasks = input_summary.get("tasks")
    return tasks if isinstance(tasks, dict) else {}


def _valid_saved_recall_artifact(
    payload: object,
    *,
    recall_cards: list[dict],
    manifest: dict,
) -> bool:
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != SAVED_QUESTIONS_SCHEMA
        or not recall_cards
    ):
        return False
    questions = payload.get("questions")
    if (
        not isinstance(questions, list)
        or len(questions) != len(recall_cards)
        or not all(isinstance(question, dict) for question in questions)
    ):
        return False
    identity = manifest.get("identity")
    if not isinstance(identity, dict) or not isinstance(
        identity.get("solver_fingerprint"), str
    ):
        return False

    cards_by_task: dict[str, dict] = {}
    for card in recall_cards:
        task_id = card.get("task_id")
        provenance = card.get("provenance")
        scope = card.get("selected_doc_ids")
        if (
            not isinstance(task_id, str)
            or not task_id
            or task_id in cards_by_task
            or not isinstance(provenance, dict)
            or provenance.get("schema") != PROVENANCE_SCHEMA
            or not isinstance(scope, list)
            or not scope
            or any(not isinstance(doc_id, str) or not doc_id for doc_id in scope)
            or provenance.get("doc_ids") != scope
            or card.get("missing_doc_ids") != []
            or not isinstance(card.get("query"), str)
        ):
            return False
        cards_by_task[task_id] = card

    question_ids = [question.get("task_id") for question in questions]
    if (
        any(not isinstance(task_id, str) or not task_id for task_id in question_ids)
        or len(set(question_ids)) != len(question_ids)
        or set(question_ids) != set(cards_by_task)
    ):
        return False

    for question in questions:
        task_id = question["task_id"]
        card = cards_by_task[task_id]
        provenance = card["provenance"]
        query = card["query"]
        scope = card["selected_doc_ids"]
        corpus_digest = provenance.get("corpus_digest")
        answer = question.get("verified_answer")
        saved_provenance = question.get("provenance")
        verified_provenance = question.get("last_verified_provenance")
        answer_audit = answer.get("coe_audit") if isinstance(answer, dict) else None
        expected_task_fingerprint = task_fingerprint(
            query,
            mode="ask",
            task_id=task_id,
            doc_ids=scope,
        )
        provenance_checks = all(
            isinstance(value, dict)
            and value.get("schema") == PROVENANCE_SCHEMA
            and value.get("task_fingerprint") == expected_task_fingerprint
            and value.get("corpus_digest") == corpus_digest
            and value.get("doc_ids") == scope
            for value in (saved_provenance, verified_provenance)
        )
        checks = (
            question.get("mode") == "ask",
            question.get("query") == query,
            question.get("doc_ids") == scope,
            question.get("selected_doc_ids") == scope,
            question.get("missing_doc_ids") == [],
            question.get("saved_corpus_digest") == corpus_digest,
            question.get("saved_scope_digest") == corpus_digest,
            question.get("last_corpus_digest") == corpus_digest,
            question.get("solver_version_fingerprint")
            == identity.get("solver_fingerprint"),
            provenance_checks,
            isinstance(answer, dict),
            isinstance(answer_audit, dict) and answer_audit.get("ok") is True,
            isinstance(answer, dict) and answer.get("query") == query,
            isinstance(answer, dict) and answer.get("selected_doc_ids") == scope,
            isinstance(answer, dict) and answer.get("missing_doc_ids") == [],
            isinstance(answer, dict)
            and question.get("last_result_digest")
            == canonical_result_fingerprint(answer),
            isinstance(answer, dict)
            and provenance.get("result_fingerprint")
            == result_output_fingerprint(answer, card),
        )
        if not all(checks):
            return False
    return True


def _cards_match_task_pack(
    cards: object,
    *,
    tasks_path: Path,
    corpus: Path,
) -> bool:
    if not isinstance(cards, list) or not all(isinstance(card, dict) for card in cards):
        return False
    try:
        payload = json.loads(tasks_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    rows = payload.get("tasks") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        return False
    expected_by_id: dict[str, dict] = {}
    for row in rows:
        raw_task_id = row.get("id")
        task_id = raw_task_id.strip() if isinstance(raw_task_id, str) else None
        if not isinstance(task_id, str) or not task_id or task_id in expected_by_id:
            return False
        expected_by_id[task_id] = row
    card_ids = [card.get("task_id") for card in cards]
    if (
        len(cards) != len(rows)
        or any(not isinstance(task_id, str) or not task_id for task_id in card_ids)
        or len(set(card_ids)) != len(card_ids)
        or set(card_ids) != set(expected_by_id)
    ):
        return False
    resolved_corpus = corpus.expanduser().resolve()
    for card in cards:
        task_id = card["task_id"]
        row = expected_by_id[task_id]
        raw_query = row.get("query")
        query = raw_query.strip() if isinstance(raw_query, str) else None
        mode = str(row.get("mode") or "ask").strip().lower()
        scope = normalize_doc_ids(row.get("doc_ids")) or []
        provenance = card.get("provenance")
        expected_fingerprint = task_fingerprint(
            str(query),
            mode=mode,
            task_id=task_id,
            expect_status=row.get("expect_status"),
            doc_ids=scope,
        )
        checks = (
            isinstance(query, str),
            bool(scope),
            card.get("query") == query,
            card.get("task_class") == mode,
            card.get("corpus") == str(resolved_corpus),
            card.get("selected_doc_ids") == scope,
            card.get("missing_doc_ids") == [],
            card.get("expect_status") == row.get("expect_status"),
            card.get("manual_baseline_seconds")
            == row.get("manual_baseline_seconds"),
            isinstance(provenance, dict),
            isinstance(provenance, dict)
            and provenance.get("schema") == PROVENANCE_SCHEMA,
            isinstance(provenance, dict) and provenance.get("doc_ids") == scope,
            isinstance(provenance, dict)
            and provenance.get("corpus_digest")
            == corpus_content_digest(resolved_corpus, doc_ids=scope),
            isinstance(provenance, dict)
            and provenance.get("task_fingerprint") == expected_fingerprint,
        )
        if not all(checks):
            return False
    return True


def _expected_repeat_recall_metadata() -> dict:
    return {
        "first": {
            "recall_state": "REFRESHED",
            "cache_hits": 0,
            "forced_refreshes": 1,
            "avoided_solver_runs": 0,
            "solver_runs": 1,
            "audit_ok": True,
        },
        "second": {
            "recall_state": "CACHE_HIT",
            "cache_hits": 1,
            "forced_refreshes": 0,
            "avoided_solver_runs": 1,
            "solver_runs": 0,
            "audit_ok": True,
        },
        "answer_fingerprint_match": True,
    }


def _frozen_results_reaudit_ok(
    study_dir: Path,
    manifest: dict,
    cards_payload: dict,
    *,
    tasks_path: Path,
    corpus: Path,
) -> bool:
    """Rederive and re-audit frozen results against exact live task scopes.

    Frozen private results are untrusted. Each task is deterministically rerun
    with persistence disabled (a recall task reruns its underlying ``ask``), and
    its stable canonical result must match before the live-source CoE audit and
    public-card projection are checked. This proves reproducibility against the
    current bound inputs and solver, not historical authenticity; the latter
    would require a separately protected external anchor.
    """
    try:
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict):
            return False
        if artifacts.get("audited_results") != "audited_results.json":
            return False
        if artifacts.get("saved_questions") != "saved_questions.json":
            return False

        audited_payload = json.loads(
            (Path(study_dir) / "audited_results.json").read_text(encoding="utf-8")
        )
        saved_payload = json.loads(
            (Path(study_dir) / "saved_questions.json").read_text(encoding="utf-8")
        )
        tasks_payload = json.loads(Path(tasks_path).read_text(encoding="utf-8"))
        cards = cards_payload.get("cards")
        tasks = tasks_payload.get("tasks")
        raw_rows = audited_payload.get("results")
        saved_questions = saved_payload.get("questions")
        if (
            not isinstance(audited_payload, dict)
            or audited_payload.get("schema") != AUDITED_RESULTS_SCHEMA
            or audited_payload.get("study_id") != manifest.get("study_id")
            or not isinstance(tasks_payload, dict)
            or not isinstance(cards, list)
            or not isinstance(tasks, list)
            or not isinstance(raw_rows, list)
            or not isinstance(saved_questions, list)
            or not all(isinstance(row, dict) for row in cards)
            or not all(isinstance(row, dict) for row in tasks)
            or not all(isinstance(row, dict) for row in raw_rows)
            or not all(isinstance(row, dict) for row in saved_questions)
        ):
            return False

        def _unique_rows_by_task(rows: list[dict]) -> dict[str, dict] | None:
            indexed: dict[str, dict] = {}
            for row in rows:
                task_id = row.get("task_id")
                if (
                    not isinstance(task_id, str)
                    or not task_id
                    or task_id in indexed
                ):
                    return None
                indexed[task_id] = row
            return indexed

        tasks_by_id: dict[str, dict] = {}
        for task in tasks:
            raw_task_id = task.get("id")
            task_id = (
                raw_task_id.strip() if isinstance(raw_task_id, str) else None
            )
            if (
                not isinstance(task_id, str)
                or not task_id
                or task_id in tasks_by_id
            ):
                return False
            tasks_by_id[task_id] = task
        cards_by_id = _unique_rows_by_task(cards)
        raw_by_id = _unique_rows_by_task(raw_rows)
        saved_by_id = _unique_rows_by_task(saved_questions)
        task_ids = set(tasks_by_id)
        if (
            cards_by_id is None
            or raw_by_id is None
            or saved_by_id is None
            or len(cards) != len(tasks)
            or len(raw_rows) != len(tasks)
            or set(cards_by_id) != task_ids
            or set(raw_by_id) != task_ids
        ):
            return False

        recall_ids = {
            task_id
            for task_id, task in tasks_by_id.items()
            if str(task.get("mode") or "ask").strip().lower() == "recall"
        }
        if set(saved_by_id) != recall_ids:
            return False

        all_docs = load_corpus(Path(corpus))
        resolved_corpus = Path(corpus).expanduser().resolve()
        for task_id, task in tasks_by_id.items():
            raw_query = task.get("query")
            query = raw_query.strip() if isinstance(raw_query, str) else None
            mode = str(task.get("mode") or "ask").strip().lower()
            scope = normalize_doc_ids(task.get("doc_ids")) or []
            card = cards_by_id[task_id]
            raw_row = raw_by_id[task_id]
            result = raw_row.get("result")
            if (
                not isinstance(query, str)
                or mode not in ALLOWED_MODES
                or not scope
                or any(doc_id not in all_docs for doc_id in scope)
                or raw_row.get("task_class") != mode
                or not isinstance(result, dict)
                or result.get("query") != query
                or result.get("selected_doc_ids") != scope
                or result.get("missing_doc_ids") != []
            ):
                return False

            rederived_result = _execute_study_result(
                query,
                mode="ask" if mode == "recall" else mode,
                corpus=resolved_corpus,
                scope=scope,
            )
            if (
                canonical_result_fingerprint(rederived_result)
                != canonical_result_fingerprint(result)
            ):
                return False

            scoped_docs = {doc_id: all_docs[doc_id] for doc_id in scope}
            audit_input = deepcopy(result)
            audit_input.pop("coe_audit", None)
            fresh_audit = audit_payload(audit_input, scoped_docs)
            if not isinstance(fresh_audit, dict) or fresh_audit.get("ok") is not True:
                return False

            validated_result = deepcopy(result)
            validated_result["coe_audit"] = fresh_audit
            scoped_digest = corpus_content_digest(resolved_corpus, doc_ids=scope)
            expected_card = card_from_result(
                query,
                validated_result,
                corpus=resolved_corpus,
                mode=mode,
                task_id=task_id,
                expect_status=task.get("expect_status"),
                corpus_digest=scoped_digest,
                doc_ids=scope,
                manual_baseline_seconds=task.get("manual_baseline_seconds"),
            )
            if mode == "recall":
                expected_card["repeat_recall"] = _expected_repeat_recall_metadata()
                expected_card["provenance"] = provenance_record(
                    query,
                    corpus=resolved_corpus,
                    corpus_digest=scoped_digest,
                    mode="recall",
                    task_id=task_id,
                    expect_status=task.get("expect_status"),
                    result_fingerprint=result_output_fingerprint(
                        validated_result, expected_card
                    ),
                    doc_ids=scope,
                )
                saved_answer = saved_by_id[task_id].get("verified_answer")
                if (
                    not isinstance(saved_answer, dict)
                    or canonical_result_fingerprint(saved_answer)
                    != canonical_result_fingerprint(result)
                ):
                    return False

            stable_expected_card = deepcopy(expected_card)
            stable_presented_card = deepcopy(card)
            stable_expected_card.pop("built_at", None)
            stable_presented_card.pop("built_at", None)
            if stable_expected_card != stable_presented_card:
                return False
        return True
    except Exception:
        return False


def _load_frozen(study_dir: Path) -> tuple[dict, dict, list[str]]:
    study_dir = Path(study_dir)
    manifest_path = study_dir / "manifest.json"
    cards_path = study_dir / "cards.json"
    errors: list[str] = []
    if not manifest_path.is_file():
        return {}, {}, ["MANIFEST_MISSING"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}, {}, ["MANIFEST_INVALID_JSON"]
    if not cards_path.is_file():
        return manifest if isinstance(manifest, dict) else {}, {}, ["CARDS_MISSING"]
    try:
        cards_payload = json.loads(cards_path.read_text(encoding="utf-8"))
    except Exception:
        return manifest if isinstance(manifest, dict) else {}, {}, ["CARDS_INVALID_JSON"]
    if not isinstance(manifest, dict) or not isinstance(cards_payload, dict):
        return {}, {}, ["STUDY_ARTIFACT_INVALID_SHAPE"]
    if manifest.get("schema") != MANIFEST_SCHEMA:
        errors.append("MANIFEST_SCHEMA_MISMATCH")
    if cards_payload.get("schema") != CARDS_SCHEMA:
        errors.append("CARDS_SCHEMA_MISMATCH")
    if cards_payload.get("study_id") != manifest.get("study_id"):
        errors.append("STUDY_ID_LINK_MISMATCH")
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        errors.append("MANIFEST_IDENTITY_MISSING")
    elif canonical_result_fingerprint(identity) != manifest.get("study_id"):
        errors.append("MANIFEST_IDENTITY_MISMATCH")
    cards_value = cards_payload.get("cards")
    if not isinstance(cards_value, list):
        cards = []
        errors.append("CARDS_INVALID_SHAPE")
    else:
        cards = cards_value
        if not all(isinstance(card, dict) for card in cards):
            errors.append("CARDS_INVALID_SHAPE")
    task_summary = _task_summary(manifest)
    expected_n_cards = task_summary.get("n_tasks")
    if (
        not isinstance(expected_n_cards, int)
        or isinstance(expected_n_cards, bool)
        or expected_n_cards < 1
        or len(cards) != expected_n_cards
    ):
        errors.append("RESULT_CARD_COUNT_MISMATCH")
    expected_mode_counts = task_summary.get("mode_counts")
    valid_expected_modes = isinstance(expected_mode_counts, dict) and all(
        isinstance(mode, str)
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count >= 0
        for mode, count in (
            expected_mode_counts.items() if isinstance(expected_mode_counts, dict) else []
        )
    )
    current_mode_counts = Counter(
        str(card.get("task_class")) for card in cards if isinstance(card, dict)
    )
    if not valid_expected_modes or dict(current_mode_counts) != expected_mode_counts:
        errors.append("CARD_MODE_COUNT_MISMATCH")
    result = manifest.get("result")
    if not isinstance(result, dict):
        result = {}
        errors.append("RESULT_METADATA_MISSING")
    current_result_digest = _file_digest(cards_path)
    if current_result_digest != result.get("digest"):
        errors.append("RESULT_ARTIFACT_CHANGED")
    if result.get("n_cards") != len(cards):
        errors.append("RESULT_METADATA_MISMATCH")
    current_audit_status = bool(cards) and all(
        isinstance(card, dict) and card.get("coe_audit_ok") is True for card in cards
    )
    if not current_audit_status:
        errors.append("RESULT_AUDIT_INVARIANT_FAILED")
    if result.get("all_coe_audits_ok") is not current_audit_status:
        errors.append("RESULT_METADATA_MISMATCH")
    recall_cards = [
        card
        for card in cards
        if isinstance(card, dict) and card.get("task_class") == "recall"
    ]
    recall_invariants_ok = bool(recall_cards) and all(
        _valid_repeat_recall_card(card) for card in recall_cards
    )
    expected_recall_count = (
        expected_mode_counts.get("recall") if valid_expected_modes else None
    )
    if (
        not isinstance(expected_recall_count, int)
        or isinstance(expected_recall_count, bool)
        or expected_recall_count < 1
        or len(recall_cards) != expected_recall_count
    ):
        errors.append("RECALL_CARD_COUNT_MISMATCH")
    if not recall_invariants_ok:
        errors.append("RECALL_INVARIANT_FAILED")
    if (
        result.get("n_repeat_recall_cards") != len(recall_cards)
        or result.get("all_repeat_recall_invariants_ok") is not recall_invariants_ok
    ):
        errors.append("RECALL_RESULT_METADATA_MISMATCH")
    artifacts = manifest.get("artifacts")
    saved_name = artifacts.get("saved_questions") if isinstance(artifacts, dict) else None
    if saved_name != "saved_questions.json":
        errors.append("RECALL_ARTIFACT_METADATA_MISSING")
    else:
        saved_path = study_dir / saved_name
        if not saved_path.is_file():
            errors.append("RECALL_ARTIFACT_MISSING")
        else:
            if _file_digest(saved_path) != result.get("saved_questions_digest"):
                errors.append("RECALL_ARTIFACT_CHANGED")
            try:
                saved_payload = json.loads(saved_path.read_text(encoding="utf-8"))
            except Exception:
                errors.append("RECALL_ARTIFACT_INVALID_JSON")
            else:
                if not _valid_saved_recall_artifact(
                    saved_payload,
                    recall_cards=recall_cards,
                    manifest=manifest,
                ):
                    errors.append("RECALL_ARTIFACT_CONTENT_MISMATCH")
    audited_name = (
        artifacts.get("audited_results") if isinstance(artifacts, dict) else None
    )
    if audited_name != "audited_results.json":
        errors.append("AUDITED_RESULTS_ARTIFACT_METADATA_MISSING")
    else:
        audited_path = study_dir / audited_name
        if not audited_path.is_file():
            errors.append("AUDITED_RESULTS_ARTIFACT_MISSING")
        else:
            if _file_digest(audited_path) != result.get("audited_results_digest"):
                errors.append("AUDITED_RESULTS_ARTIFACT_CHANGED")
            try:
                audited_payload = json.loads(audited_path.read_text(encoding="utf-8"))
            except Exception:
                errors.append("AUDITED_RESULTS_ARTIFACT_INVALID_JSON")
            else:
                audited_rows = (
                    audited_payload.get("results")
                    if isinstance(audited_payload, dict)
                    else None
                )
                if (
                    not isinstance(audited_payload, dict)
                    or audited_payload.get("schema") != AUDITED_RESULTS_SCHEMA
                    or audited_payload.get("study_id") != manifest.get("study_id")
                    or not isinstance(audited_rows, list)
                    or not all(isinstance(row, dict) for row in audited_rows)
                ):
                    errors.append("AUDITED_RESULTS_ARTIFACT_INVALID_SHAPE")
    return manifest, cards_payload, errors


def verify_study(study_dir: Path) -> tuple[dict, dict, list[str]]:
    if not valid_study_directory(Path(study_dir)):
        return {}, {}, ["STUDY_DIR_NOT_PRIVATE_LOCATION"]
    manifest, cards_payload, errors = _load_frozen(study_dir)
    local_inputs = manifest.get("local_inputs") or {}
    if (
        not isinstance(local_inputs, dict)
        or not isinstance(local_inputs.get("corpus"), str)
        or not isinstance(local_inputs.get("tasks"), str)
        or not local_inputs.get("corpus")
        or not local_inputs.get("tasks")
    ):
        errors.append("INPUT_POINTERS_MISSING")
        return manifest, cards_payload, sorted(set(errors))
    try:
        corpus_path = Path(local_inputs["corpus"])
        tasks_path = Path(local_inputs["tasks"])
        current = assess_inputs(corpus_path, tasks_path)
        cards_match_task_pack = _cards_match_task_pack(
            cards_payload.get("cards"),
            tasks_path=tasks_path,
            corpus=corpus_path,
        )
        frozen_results_reaudit_ok = _frozen_results_reaudit_ok(
            Path(study_dir),
            manifest,
            cards_payload,
            tasks_path=tasks_path,
            corpus=corpus_path,
        )
    except Exception:
        errors.append("INPUT_REASSESSMENT_FAILED")
        return manifest, cards_payload, sorted(set(errors))
    if not cards_match_task_pack:
        errors.append("CARD_TASK_BINDING_FAILED")
    if not frozen_results_reaudit_ok:
        errors.append("FROZEN_RESULT_REAUDIT_FAILED")
    if (
        current.get("study_id") != manifest.get("study_id")
        or current.get("identity") != manifest.get("identity")
    ):
        errors.append("INPUT_OR_SOLVER_IDENTITY_CHANGED")
    expected_summary = {
        "corpus": current.get("corpus"),
        "tasks": current.get("tasks"),
    }
    if manifest.get("input_summary") != expected_summary:
        errors.append("INPUT_SUMMARY_CHANGED")
    expected_contract = {
        "study_class": current.get("study_class"),
        "representative_ready": current.get("representative_ready"),
        "required_reviewer_kind": current.get("required_reviewer_kind"),
        "manual_time_comparison_enabled": current.get(
            "manual_time_comparison_enabled"
        ),
        "claim_boundary": current.get("claim_boundary"),
    }
    frozen_contract = {
        "study_class": manifest.get("study_class"),
        "representative_ready": manifest.get("representative_ready"),
        "required_reviewer_kind": manifest.get("required_reviewer_kind"),
        "manual_time_comparison_enabled": manifest.get(
            "manual_time_comparison_enabled"
        ),
        "claim_boundary": manifest.get("claim_boundary"),
    }
    if frozen_contract != expected_contract:
        errors.append("STUDY_CONTRACT_CHANGED")
    if not current.get("study_ready"):
        errors.append(
            "INPUTS_NO_LONGER_STUDY_READY"
            if current.get("study_class") == AGENT_APPLIED_SCOPED_PILOT
            else "INPUTS_NO_LONGER_REPRESENTATIVE_READY"
        )
    return manifest, cards_payload, sorted(set(errors))


def review_study(
    study_dir: Path,
    *,
    reviewer_kind: str,
    stdin=None,
    stdout=None,
) -> dict:
    reviewer_kind = reviewer_kind.strip().lower()
    if reviewer_kind not in REVIEWER_KINDS or reviewer_kind == "unspecified":
        raise ValueError("study review requires an explicit non-unspecified reviewer kind")
    manifest, cards_payload, errors = verify_study(study_dir)
    if errors:
        return _blocked_summary(errors, study_id=manifest.get("study_id"))
    required_reviewer_kind = manifest.get("required_reviewer_kind")
    if (
        isinstance(required_reviewer_kind, str)
        and reviewer_kind != required_reviewer_kind
    ):
        return _blocked_summary(
            ["PILOT_REVIEWER_KIND_REQUIRED"], study_id=manifest.get("study_id")
        )
    cards = cards_payload.get("cards") or []
    review_path = Path(study_dir) / "review.json"
    state = load_state(review_path)
    if state.get("load_errors"):
        return _blocked_summary(
            ["REVIEW_STATE_INVALID_SHAPE"], study_id=manifest.get("study_id")
        )
    cards = merge_prior_labels(cards, state)
    interactive_review(
        cards,
        state,
        stdin=stdin,
        stdout=stdout,
        state_path=review_path,
        reviewer_kind=reviewer_kind,
    )
    return summarize_study(study_dir)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value >= 0 else None


def _round(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def summarize_study(study_dir: Path) -> dict:
    study_dir = Path(study_dir)
    manifest, cards_payload, errors = verify_study(study_dir)
    if errors:
        summary = _blocked_summary(errors, study_id=manifest.get("study_id"))
        if valid_study_directory(study_dir):
            _write_json(study_dir / "summary.json", summary)
        return summary

    cards = cards_payload.get("cards") or []
    review_path = study_dir / "review.json"
    try:
        state = load_state(review_path)
    except Exception:
        summary = _blocked_summary(
            ["REVIEW_STATE_INVALID_JSON"], study_id=manifest.get("study_id")
        )
        _write_json(study_dir / "summary.json", summary)
        return summary
    if state.get("load_errors"):
        summary = _blocked_summary(
            ["REVIEW_STATE_INVALID_SHAPE"], study_id=manifest.get("study_id")
        )
        _write_json(study_dir / "summary.json", summary)
        return summary
    merged = merge_prior_labels(cards, state)
    labeled = [card for card in merged if card.get("usefulness_label")]

    label_counts = Counter(str(card["usefulness_label"]) for card in labeled)
    failure_counts: Counter[str] = Counter()
    reviewer_counts = Counter(str(card.get("reviewer_kind") or "unspecified") for card in labeled)
    review_seconds = []
    for card in labeled:
        raw_elapsed = card.get("review_elapsed_s")
        if raw_elapsed is None:
            raw_elapsed = card.get("review_seconds")
        value = _number(raw_elapsed)
        if value is not None:
            review_seconds.append(value)
    solver_seconds = []
    for card in cards:
        value = _number(card.get("latency_s"))
        if value is None:
            latency_ms = _number(card.get("latency_ms"))
            value = latency_ms / 1000 if latency_ms is not None else None
        if value is not None:
            solver_seconds.append(value)
    manual_seconds = [
        value
        for card in cards
        if (value := _number(card.get("manual_baseline_seconds"))) is not None and value > 0
    ]

    first_repeated: str | None = None
    chronological = sorted(
        labeled,
        key=lambda card: str(card.get("labeled_at") or "~"),
    )
    running: Counter[str] = Counter()
    incomplete_failure_details = 0
    for card in chronological:
        if str(card.get("usefulness_label") or "") not in FAILURE_LABELS:
            continue
        failure_class = str(card.get("failure_class") or "").upper()
        failure_reason = card.get("correction_reason")
        if not isinstance(failure_reason, str) or not failure_reason.strip():
            failure_reason = card.get("failure_reason")
        suggested_correction = card.get("suggested_correction")
        if (
            failure_class not in FAILURE_CLASSES
            or not isinstance(failure_reason, str)
            or not failure_reason.strip()
            or not isinstance(suggested_correction, str)
            or not suggested_correction.strip()
        ):
            incomplete_failure_details += 1
            continue
        failure_counts[failure_class] += 1
        running[failure_class] += 1
        if first_repeated is None and running[failure_class] == 2:
            first_repeated = failure_class

    fully_reviewed = len(cards) > 0 and len(labeled) == len(cards)
    measured_review = len(review_seconds) == len(cards)
    explicit_reviewers = bool(labeled) and "unspecified" not in reviewer_counts
    audit_failures = sum(card.get("coe_audit_ok") is not True for card in cards)
    recall_cards = [card for card in cards if card.get("task_class") == "recall"]
    recall_coverage = {
        "n_tasks": len(recall_cards),
        "n_first_refreshed": sum(
            (card.get("repeat_recall") or {}).get("first", {}).get("recall_state")
            == "REFRESHED"
            for card in recall_cards
        ),
        "n_second_cache_hits": sum(
            (card.get("repeat_recall") or {}).get("second", {}).get("recall_state")
            == "CACHE_HIT"
            for card in recall_cards
        ),
        "n_solver_runs": sum(
            int((card.get("repeat_recall") or {}).get("first", {}).get("solver_runs") or 0)
            + int((card.get("repeat_recall") or {}).get("second", {}).get("solver_runs") or 0)
            for card in recall_cards
        ),
        "n_avoided_solver_runs": sum(
            int(
                (card.get("repeat_recall") or {})
                .get("first", {})
                .get("avoided_solver_runs")
                or 0
            )
            + int(
                (card.get("repeat_recall") or {})
                .get("second", {})
                .get("avoided_solver_runs")
                or 0
            )
            for card in recall_cards
        ),
    }
    invalid_review_state = sum(
        card.get("prior_label_status") == "IGNORED_INVALID_REVIEW_STATE"
        for card in merged
    )
    study_class = manifest.get("study_class")
    pilot_reviewer_mismatch = bool(
        study_class == AGENT_APPLIED_SCOPED_PILOT
        and labeled
        and set(reviewer_counts) != {"agent_applied"}
    )
    completion_blockers = list(errors)
    if not fully_reviewed:
        completion_blockers.append("REVIEW_INCOMPLETE")
    if not measured_review:
        completion_blockers.append("REVIEW_TIMING_INCOMPLETE")
    if not explicit_reviewers:
        completion_blockers.append("REVIEWER_KIND_MISSING")
    if audit_failures:
        completion_blockers.append("COE_AUDIT_FAILURE")
    if incomplete_failure_details:
        completion_blockers.append("FAILURE_DETAIL_INCOMPLETE")
    if invalid_review_state:
        completion_blockers.append("REVIEW_STATE_INVALID")
    if pilot_reviewer_mismatch:
        completion_blockers.append("PILOT_REVIEWER_KIND_REQUIRED")
    completion_blockers = sorted(set(completion_blockers))
    if (
        not fully_reviewed
        or not measured_review
        or not explicit_reviewers
        or audit_failures
        or incomplete_failure_details
        or invalid_review_state
        or pilot_reviewer_mismatch
    ):
        decision = "INCOMPLETE"
    elif first_repeated:
        decision = "FIX_REPEATED_FAILURE"
    else:
        decision = "NO_REPEATED_FAILURE"

    time_comparison = None
    if (
        manifest.get("manual_time_comparison_enabled") is True
        and len(manual_seconds) == len(cards)
        and measured_review
        and len(solver_seconds) == len(cards)
    ):
        manual_total = sum(manual_seconds)
        tool_total = sum(review_seconds) + sum(solver_seconds)
        time_comparison = {
            "coverage": len(cards),
            "manual_total_seconds": _round(manual_total),
            "tool_plus_review_total_seconds": _round(tool_total),
            "manual_minus_tool_seconds": _round(manual_total - tool_total),
            "scope": "paired operator estimates; descriptive only",
        }

    input_summary = manifest.get("input_summary") or {}
    identity = manifest.get("identity") or {}
    summary = {
        "schema": SUMMARY_SCHEMA,
        "study_id": manifest.get("study_id"),
        "study_class": study_class,
        "study_ready": True,
        "representative_ready": manifest.get("representative_ready") is True,
        "status": "COMPLETE" if decision != "INCOMPLETE" else "INCOMPLETE",
        "decision": decision,
        "blockers": completion_blockers,
        "identity": {
            "corpus_digest": identity.get("corpus_digest"),
            "task_pack_digest": identity.get("task_pack_digest"),
            "solver_fingerprint": identity.get("solver_fingerprint"),
            "instrument_fingerprint": identity.get("instrument_fingerprint"),
            "pdf_extractor": identity.get("pdf_extractor"),
            "result_digest": (manifest.get("result") or {}).get("digest"),
            "review_digest": _file_digest(review_path),
        },
        "coverage": {
            "n_corpus_documents": (input_summary.get("corpus") or {}).get("n_documents"),
            "n_scoped_documents": (input_summary.get("tasks") or {}).get(
                "n_unique_scoped_documents"
            ),
            "scope_coverage_fraction": (input_summary.get("tasks") or {}).get(
                "scope_coverage_fraction"
            ),
            "format_counts": (input_summary.get("corpus") or {}).get("format_counts") or {},
            "n_tasks": len(cards),
            "mode_counts": (input_summary.get("tasks") or {}).get("mode_counts") or {},
            "n_reviewed": len(labeled),
            "n_audit_failures": audit_failures,
            "repeat_recall": recall_coverage,
        },
        "outcomes": {
            "by_label": dict(sorted(label_counts.items())),
            "by_failure_class": dict(sorted(failure_counts.items())),
            "first_repeated_failure_class": first_repeated,
            "n_incomplete_failure_details": incomplete_failure_details,
        },
        "review": {
            "by_reviewer_kind": dict(sorted(reviewer_counts.items())),
            "n_timed": len(review_seconds),
            "total_seconds": _round(sum(review_seconds)) if review_seconds else None,
            "median_seconds": _round(statistics.median(review_seconds)) if review_seconds else None,
        },
        "solver": {
            "n_timed": len(solver_seconds),
            "median_seconds": _round(statistics.median(solver_seconds)) if solver_seconds else None,
        },
        "manual_baseline": {
            "n_tasks_with_baseline": len(manual_seconds),
            "comparison": time_comparison,
            **(
                {"time_saved_claim_supported": False}
                if study_class == AGENT_APPLIED_SCOPED_PILOT
                else {}
            ),
        },
        "review_evidence_kind": (
            "AGENT_APPLIED_RUBRIC"
            if reviewer_counts and set(reviewer_counts) == {"agent_applied"}
            else "MIXED_OR_HUMAN_REVIEW"
            if reviewer_counts
            else "NONE"
        ),
        "next_action": {
            "INCOMPLETE": "finish explicit timed review without changing frozen inputs",
            "FIX_REPEATED_FAILURE": "fix only the first failure class repeated at least twice",
            "NO_REPEATED_FAILURE": (
                "record that none repeated; collect more genuine use before architecture expansion"
            ),
        }[decision],
        "claim_boundary": (
            "This agent-applied scoped pilot aggregate selects only the next Wedge "
            "component-development workflow step. It is not representative-use evidence, and no "
            "time-saved claim is supported."
            if study_class == AGENT_APPLIED_SCOPED_PILOT
            else (
                "This aggregate selects the next Wedge component-development workflow step only. "
                "It does not establish Nano AI capability or model, clinical, or scientific superiority."
            )
        ),
    }
    _write_json(study_dir / "summary.json", summary)
    return summary
