"""Local habit / session workflow (gitignored). Not Layer-1 evidence."""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

from wedge_v1 import __version__ as WEDGE_VERSION
from wedge_v1.coe.audit import audit_payload
from wedge_v1.coe.canonical import canonical_result, canonical_result_fingerprint
from wedge_v1.coe.schema import SOLVER_VERSION
from wedge_v1.ingest import corpus_stats, document_id
from wedge_v1.review import (
    PROVENANCE_SCHEMA,
    REVIEW_PATH,
    build_card,
    corpus_content_digest,
    merge_prior_labels,
    provenance_record,
    task_fingerprint,
    unlabeled,
)
from wedge_v1.review import load_state as load_review
from wedge_v1.runtime import (
    DEFAULT_CORPUS,
    ask,
    load_corpus,
    normalize_doc_ids,
    scan,
    select_documents,
)

ROOT = Path(__file__).resolve().parent
HABIT_PATH = ROOT / "results_owner_habit.json"
SAVED_QUESTIONS = ROOT / "results_saved_questions.json"
SESSION_PATH = ROOT / "results_habit_session.json"
DEFAULT_REVIEW_TASKS = ROOT / "data" / "owner_dogfood_tasks.example.json"
RECALL_SCHEMA = "nano-lm.wedge_v1.saved_answer_recall.v1"
_RECALL_IMPLEMENTATION_VERSION = "wedge_v1.habit.recall.v1"
_VALID_RECALL_STATUSES = {"SUPPORTED", "CONTRADICTED", "ABSTAIN"}
_USE_SAVED_SCOPE = object()


def _solver_source_bytes() -> dict[str, bytes]:
    paths = {ROOT / "habit.py", ROOT / "ingest.py", ROOT / "runtime.py"}
    for folder in (ROOT / "classical", ROOT / "coe", ROOT / "plugins"):
        paths.update(folder.glob("*.py"))
    # Plugin lexicons are executable solver inputs: changing a synonym, OCR
    # substitution, or coreference rule can change an answer without changing
    # Python source. Bind saved answers to those exact bytes as well.
    paths.update((ROOT / "plugins" / "data").glob("*.json"))
    return {
        path.relative_to(ROOT).as_posix(): path.read_bytes()
        for path in sorted(paths)
        if path.is_file()
    }


def solver_implementation_fingerprint(
    sources: dict[str, bytes] | None = None,
) -> str:
    """Bind reusable answers to exact solver, verifier, and static-rule bytes."""
    source_map = _solver_source_bytes() if sources is None else sources
    digest = hashlib.sha256()
    for version in (WEDGE_VERSION, SOLVER_VERSION, _RECALL_IMPLEMENTATION_VERSION):
        digest.update(version.encode("utf-8"))
        digest.update(b"\0")
    for name in sorted(source_map):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source_map[name])
        digest.update(b"\0")
    return digest.hexdigest()


CURRENT_SOLVER_FINGERPRINT = solver_implementation_fingerprint()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def load(path: Path = HABIT_PATH) -> dict:
    if not path.is_file():
        return {"schema": "nano-lm.wedge_v1.habit.v1", "events": []}
    return json.loads(path.read_text(encoding="utf-8"))


def record(kind: str, *, note: str = "", path: Path = HABIT_PATH) -> dict:
    data = load(path)
    data.setdefault("schema", "nano-lm.wedge_v1.habit.v1")
    data.setdefault("events", [])
    data["events"].append(
        {
            "ts": _now().isoformat(),
            "kind": kind,
            "note": note[:200],
        }
    )
    data["events"] = data["events"][-500:]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return data


def weekly_summary(path: Path = HABIT_PATH, days: int = 7) -> dict:
    data = load(path)
    cutoff = _now() - timedelta(days=days)
    recent = []
    for e in data.get("events") or []:
        try:
            ts = datetime.fromisoformat(e["ts"])
        except Exception:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts >= cutoff:
            recent.append(e)
    counts = Counter(e.get("kind") or "unknown" for e in recent)
    return {
        "schema": "nano-lm.wedge_v1.habit_summary.v1",
        "days": days,
        "n_events": len(recent),
        "by_kind": dict(sorted(counts.items())),
        "path": str(path),
        "note": "Local habit counters; not Layer-1 evidence.",
    }


def resolve_session_corpus(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser()
    env = (os.environ.get("OWNER_CORPUS") or os.environ.get("WEDGE_OWNER_CORPUS") or "").strip()
    if env:
        return Path(env).expanduser()
    owner = ROOT / "data" / "owner_corpus"
    if owner.is_dir() and any(owner.rglob("*.md")):
        return owner
    fixture = ROOT / "fixtures" / "owner_corpus"
    if fixture.is_dir():
        return fixture
    return DEFAULT_CORPUS


def recent_documents(corpus: Path, *, limit: int = 12) -> list[dict]:
    if not corpus.is_dir():
        return []
    files = []
    for p in corpus.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".md", ".txt", ".pdf", ".markdown"}:
            continue
        if any(part.startswith(".") for part in p.relative_to(corpus).parts):
            continue
        st = p.stat()
        files.append(
            {
                "doc_id": document_id(p, corpus),
                "path": str(p.relative_to(corpus)),
                "mtime": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
                "bytes": st.st_size,
                "suffix": p.suffix.lower(),
            }
        )
    files.sort(key=lambda x: x["mtime"], reverse=True)
    return files[:limit]


def load_saved_questions(path: Path = SAVED_QUESTIONS) -> list[dict]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("questions") or [])


def corpus_digest(
    corpus: Path,
    doc_ids: list[str] | None = None,
) -> str | None:
    """Return a stable content digest for supported local corpus files."""
    return corpus_content_digest(corpus, doc_ids=doc_ids)


def _stable_result(value):
    return canonical_result(value)


def _result_digest(result: dict) -> str:
    """Integrity digest preserving CoE identity links across run-local IDs."""
    return canonical_result_fingerprint(result)


def _effective_task_id(question: dict) -> str:
    mode = question.get("mode") or "ask"
    scope = normalize_doc_ids(question.get("doc_ids")) if "doc_ids" in question else None
    material = f"{mode}|{question.get('query') or ''}"
    if scope is not None:
        material += "|" + json.dumps(scope, separators=(",", ":"))
    return question.get("task_id") or hashlib.sha256(
        material.encode("utf-8")
    ).hexdigest()[:16]


def _valid_provenance(value: object) -> bool:
    return (
        isinstance(value, dict)
        and value.get("schema") == PROVENANCE_SCHEMA
        and bool(value.get("task_fingerprint"))
    )


def _audit_answer(
    result: object,
    docs: dict[str, str],
    *,
    query: str,
    doc_ids: list[str] | None = None,
) -> tuple[bool, dict]:
    if not isinstance(result, dict):
        return False, {"ok": False, "failure_codes": ["INVALID_ANSWER_PAYLOAD"]}
    if result.get("answer_status") not in _VALID_RECALL_STATUSES:
        return False, {"ok": False, "failure_codes": ["INVALID_ANSWER_STATUS"]}
    if result.get("query") != query:
        return False, {"ok": False, "failure_codes": ["ANSWER_TASK_MISMATCH"]}
    scope = normalize_doc_ids(doc_ids)
    if scope is not None and (
        result.get("selected_doc_ids") != scope
        or bool(result.get("missing_doc_ids"))
    ):
        return False, {"ok": False, "failure_codes": ["ANSWER_SCOPE_MISMATCH"]}
    try:
        audit = audit_payload(result, docs)
    except Exception as exc:
        return False, {
            "ok": False,
            "failure_codes": ["COE_AUDIT_ERROR"],
            "error_type": type(exc).__name__,
        }
    return bool(audit.get("ok")), audit


def _persist_verified_answer(
    question: dict,
    result: dict,
    *,
    corpus_digest_value: str | None,
    task_id: str,
    doc_ids: list[str] | None = None,
) -> None:
    mode = question.get("mode") or "ask"
    scope = normalize_doc_ids(doc_ids)
    current_provenance = provenance_record(
        question.get("query") or "",
        corpus_digest=corpus_digest_value,
        mode=mode,
        task_id=task_id,
        doc_ids=scope,
    )
    question["task_id"] = task_id
    if scope is None:
        question.pop("doc_ids", None)
        question.pop("selected_doc_ids", None)
        question.pop("missing_doc_ids", None)
        question.pop("saved_scope_digest", None)
    else:
        question["doc_ids"] = scope
        question["selected_doc_ids"] = list(result.get("selected_doc_ids") or scope)
        question["missing_doc_ids"] = list(result.get("missing_doc_ids") or [])
        question["saved_scope_digest"] = corpus_digest_value
    question["provenance"] = current_provenance
    question["last_corpus_digest"] = corpus_digest_value
    question["last_result_digest"] = _result_digest(result)
    question["last_verified_provenance"] = current_provenance
    question["solver_version_fingerprint"] = CURRENT_SOLVER_FINGERPRINT
    question["verified_answer"] = deepcopy(result)
    question["verified_at"] = _now().isoformat()


def _safe_recall_failure(code: str) -> dict:
    return {
        "answer_status": "ABSTAIN",
        "claims": [],
        "failure_codes": [code],
        "note": "saved answer was not served because a fresh audited result was unavailable",
    }


def _recall_output(
    task_id: str,
    *,
    state: str,
    answer: dict,
    started: float,
    cache_hits: int = 0,
    forced_refreshes: int = 0,
    avoided_solver_runs: int = 0,
    solver_runs: int = 0,
    refresh_reason: str | None = None,
    failure_codes: list[str] | None = None,
    scope: dict | None = None,
) -> dict:
    payload = {
        "schema": RECALL_SCHEMA,
        "task_id": task_id,
        "recall_state": state,
        "refresh_reason": refresh_reason,
        "failure_codes": list(failure_codes or []),
        "answer": answer,
        "aggregate": {
            "cache_hits": cache_hits,
            "forced_refreshes": forced_refreshes,
            "avoided_solver_runs": avoided_solver_runs,
            "solver_runs": solver_runs,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        },
    }
    if scope is not None:
        payload.update(deepcopy(scope))
    return payload


def save_question(
    query: str,
    *,
    mode: str = "ask",
    task_id: str | None = None,
    corpus: Path | None = None,
    doc_ids: list[str] | None = None,
    path: Path = SAVED_QUESTIONS,
) -> dict:
    scope = normalize_doc_ids(doc_ids)
    id_material = f"{mode}|{query}"
    if scope is not None:
        id_material += "|" + json.dumps(scope, separators=(",", ":"))
    effective_task_id = task_id or hashlib.sha256(
        id_material.encode("utf-8")
    ).hexdigest()[:16]
    saved_digest = corpus_digest(corpus, doc_ids=scope) if corpus else None
    selected_doc_ids = list(scope or [])
    missing_doc_ids: list[str] = []
    if corpus is not None and scope is not None:
        try:
            _, scope_fields, _ = select_documents(load_corpus(corpus), scope)
            selected_doc_ids = list(scope_fields.get("selected_doc_ids") or [])
            missing_doc_ids = list(scope_fields.get("missing_doc_ids") or [])
        except Exception:
            selected_doc_ids = []
            missing_doc_ids = list(scope)
    data = {"schema": "nano-lm.wedge_v1.saved_questions.v1", "questions": load_saved_questions(path)}
    data["questions"] = [
        q
        for q in data["questions"]
        if not (
            q.get("query") == query
            and q.get("mode") == mode
            and (
                normalize_doc_ids(q.get("doc_ids")) if "doc_ids" in q else None
            )
            == scope
        )
    ]
    saved_question = {
        "task_id": effective_task_id,
        "query": query,
        "mode": mode,
        "saved_at": _now().isoformat(),
        "saved_corpus_digest": saved_digest,
        "provenance": provenance_record(
            query,
            corpus_digest=saved_digest,
            mode=mode,
            task_id=effective_task_id,
            doc_ids=scope,
        ),
    }
    if scope is not None:
        saved_question.update(
            {
                "doc_ids": scope,
                "selected_doc_ids": selected_doc_ids,
                "missing_doc_ids": missing_doc_ids,
                "saved_scope_digest": saved_digest,
            }
        )
    data["questions"].insert(0, saved_question)
    data["questions"] = data["questions"][:100]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return data


def saved_question_status(corpus: Path, *, path: Path = SAVED_QUESTIONS) -> list[dict]:
    """Report whether saved answers are current, stale, or not yet verified."""
    rows = []
    for q in load_saved_questions(path):
        mode = q.get("mode") or "ask"
        task_id = _effective_task_id(q)
        scope = normalize_doc_ids(q.get("doc_ids")) if "doc_ids" in q else None
        current_digest = corpus_digest(corpus, doc_ids=scope)
        current_task_fingerprint = task_fingerprint(
            q.get("query") or "",
            mode=mode,
            task_id=task_id,
            doc_ids=scope,
        )
        saved_provenance = q.get("provenance")
        verified_provenance = q.get("last_verified_provenance")
        if (
            not isinstance(saved_provenance, dict)
            or saved_provenance.get("schema") != PROVENANCE_SCHEMA
            or not saved_provenance.get("task_fingerprint")
        ):
            state = "LEGACY"
            reason = "MISSING_PROVENANCE"
        elif saved_provenance.get("task_fingerprint") != current_task_fingerprint:
            state = "STALE"
            reason = "TASK_CHANGED"
        elif not q.get("last_result_digest"):
            state = "UNVERIFIED"
            reason = "NO_VERIFIED_RESULT"
        elif (
            not isinstance(verified_provenance, dict)
            or verified_provenance.get("schema") != PROVENANCE_SCHEMA
            or not verified_provenance.get("task_fingerprint")
            or not verified_provenance.get("corpus_digest")
        ):
            state = "LEGACY"
            reason = "MISSING_VERIFICATION_PROVENANCE"
        elif verified_provenance.get("task_fingerprint") != current_task_fingerprint:
            state = "STALE"
            reason = "TASK_CHANGED"
        elif verified_provenance.get("corpus_digest") != current_digest:
            state = "STALE"
            reason = "CORPUS_CHANGED"
        elif not isinstance(q.get("verified_answer"), dict):
            state = "LEGACY"
            reason = "MISSING_VERIFIED_ANSWER"
        elif q.get("solver_version_fingerprint") != CURRENT_SOLVER_FINGERPRINT:
            state = "STALE"
            reason = "SOLVER_CHANGED"
        elif q.get("last_result_digest") != _result_digest(q["verified_answer"]):
            state = "STALE"
            reason = "RESULT_DIGEST_MISMATCH"
        else:
            state = "CURRENT"
            reason = "PROVENANCE_MATCH"
        rows.append(
            {
                "query": q.get("query"),
                "task_id": task_id,
                "mode": mode,
                "doc_ids": scope,
                "state": state,
                "reason": reason,
                "verified_at": q.get("verified_at"),
            }
        )
    return rows


def recall_saved(
    task_id: str,
    corpus: Path,
    *,
    doc_ids: list[str] | None | object = _USE_SAVED_SCOPE,
    path: Path = SAVED_QUESTIONS,
    persist_coe: bool = True,
) -> dict:
    """Recall a saved answer, refreshing exactly once unless its cache is auditable."""
    started = time.perf_counter()
    data = {
        "schema": "nano-lm.wedge_v1.saved_questions.v1",
        "questions": load_saved_questions(path),
    }
    matches = [q for q in data["questions"] if _effective_task_id(q) == task_id]
    if not matches:
        return _recall_output(
            task_id,
            state="NOT_FOUND",
            answer=_safe_recall_failure("SAVED_TASK_NOT_FOUND"),
            started=started,
            failure_codes=["SAVED_TASK_NOT_FOUND"],
        )
    if len(matches) != 1:
        return _recall_output(
            task_id,
            state="AMBIGUOUS_TASK_ID",
            answer=_safe_recall_failure("AMBIGUOUS_SAVED_TASK_ID"),
            started=started,
            failure_codes=["AMBIGUOUS_SAVED_TASK_ID"],
        )

    question = matches[0]
    query = question.get("query") or ""
    mode = question.get("mode") or "ask"
    saved_scope = (
        normalize_doc_ids(question.get("doc_ids"))
        if "doc_ids" in question
        else None
    )
    scope_overridden = doc_ids is not _USE_SAVED_SCOPE
    effective_scope = (
        normalize_doc_ids(doc_ids)  # type: ignore[arg-type]
        if scope_overridden
        else saved_scope
    )
    scope_changed = scope_overridden and effective_scope != saved_scope
    if mode != "ask":
        return _recall_output(
            task_id,
            state="UNSUPPORTED_MODE",
            answer=_safe_recall_failure("SAVED_TASK_MODE_UNSUPPORTED"),
            started=started,
            failure_codes=["SAVED_TASK_MODE_UNSUPPORTED"],
        )

    current_corpus_digest = corpus_digest(corpus, doc_ids=effective_scope)
    scope_fields: dict = {}
    scope_valid = True
    try:
        docs, scope_fields, scope_valid = select_documents(
            load_corpus(corpus),
            effective_scope,
        )
        docs_error = None
    except Exception as exc:
        docs = {}
        docs_error = type(exc).__name__
        if effective_scope is not None:
            scope_fields = {
                "selected_doc_ids": [],
                "missing_doc_ids": list(effective_scope),
            }
    current_task_fingerprint = task_fingerprint(
        query,
        mode=mode,
        task_id=task_id,
        doc_ids=effective_scope,
    )
    saved_provenance = question.get("provenance")
    verified_provenance = question.get("last_verified_provenance")
    snapshot = question.get("verified_answer")

    refresh_reason = None
    if docs_error:
        refresh_reason = "CORPUS_LOAD_FAILED"
    elif scope_changed:
        refresh_reason = "SCOPE_CHANGED"
    elif not scope_valid:
        refresh_reason = "INVALID_DOCUMENT_SCOPE"
    elif not _valid_provenance(saved_provenance):
        refresh_reason = "MISSING_PROVENANCE"
    elif saved_provenance.get("task_fingerprint") != current_task_fingerprint:
        refresh_reason = "TASK_CHANGED"
    elif not _valid_provenance(verified_provenance):
        refresh_reason = "MISSING_VERIFICATION_PROVENANCE"
    elif verified_provenance.get("task_fingerprint") != current_task_fingerprint:
        refresh_reason = "TASK_CHANGED"
    elif verified_provenance.get("corpus_digest") != current_corpus_digest:
        refresh_reason = "CORPUS_CHANGED"
    elif question.get("solver_version_fingerprint") != CURRENT_SOLVER_FINGERPRINT:
        refresh_reason = "SOLVER_CHANGED"
    elif not isinstance(snapshot, dict):
        refresh_reason = "MISSING_VERIFIED_ANSWER"
    elif question.get("last_result_digest") != _result_digest(snapshot):
        refresh_reason = "RESULT_DIGEST_MISMATCH"
    else:
        cached_ok, _ = _audit_answer(
            snapshot,
            docs,
            query=query,
            doc_ids=effective_scope,
        )
        if not cached_ok:
            refresh_reason = "AUDIT_FAILED"

    if refresh_reason is None:
        return _recall_output(
            task_id,
            state="CACHE_HIT",
            answer=deepcopy(snapshot),
            started=started,
            cache_hits=1,
            avoided_solver_runs=1,
            scope=scope_fields if effective_scope is not None else None,
        )

    try:
        if effective_scope is None:
            fresh = (
                ask(query, corpus_dir=corpus)
                if persist_coe
                else ask(query, corpus_dir=corpus, persist_coe=False)
            )
        else:
            fresh = (
                ask(query, corpus_dir=corpus, doc_ids=effective_scope)
                if persist_coe
                else ask(
                    query,
                    corpus_dir=corpus,
                    doc_ids=effective_scope,
                    persist_coe=False,
                )
            )
    except Exception as exc:
        code = f"SAVED_ANSWER_REFRESH_{type(exc).__name__.upper()}"
        return _recall_output(
            task_id,
            state="REFRESH_FAILED",
            answer=_safe_recall_failure(code),
            started=started,
            forced_refreshes=1,
            solver_runs=1,
            refresh_reason=refresh_reason,
            failure_codes=[code],
            scope=scope_fields if effective_scope is not None else None,
        )

    fresh_ok, fresh_audit = _audit_answer(
        fresh,
        docs,
        query=query,
        doc_ids=effective_scope,
    )
    if not fresh_ok:
        codes = list(fresh_audit.get("failure_codes") or ["SAVED_ANSWER_REFRESH_AUDIT_FAILED"])
        return _recall_output(
            task_id,
            state="REFRESH_FAILED",
            answer=_safe_recall_failure("SAVED_ANSWER_REFRESH_AUDIT_FAILED"),
            started=started,
            forced_refreshes=1,
            solver_runs=1,
            refresh_reason=refresh_reason,
            failure_codes=codes,
            scope=scope_fields if effective_scope is not None else None,
        )

    _persist_verified_answer(
        question,
        fresh,
        corpus_digest_value=current_corpus_digest,
        task_id=task_id,
        doc_ids=effective_scope,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return _recall_output(
        task_id,
        state="REFRESHED",
        answer=deepcopy(fresh),
        started=started,
        forced_refreshes=1,
        solver_runs=1,
        refresh_reason=refresh_reason,
        scope=scope_fields if effective_scope is not None else None,
    )


def disputed_from_scan(corpus: Path) -> list[dict]:
    out = scan(corpus_dir=corpus)
    claims = out.get("claims") or []
    disputed = []
    for c in claims:
        st = str(c.get("status") or "").upper()
        notes = str(c.get("notes") or "").lower()
        if st in {"DISPUTED", "CONFLICT", "COLLISION"} or any(
            k in notes for k in ("conflict", "contradict", "collision", "dose_change")
        ):
            disputed.append(
                {
                    "doc_id": c.get("doc_id"),
                    "task_id": c.get("task_id"),
                    "value": c.get("value"),
                    "status": c.get("status"),
                    "notes": c.get("notes"),
                }
            )
    return disputed[:20]


def review_queue_summary(corpus: Path, *, tasks: list[dict] | None = None) -> dict:
    """Summarize only labels valid for the current corpus and task definitions."""
    state = load_review()
    cached_cards = state.get("cards") or {}
    unscoped_digest = corpus_digest(corpus)
    if tasks is None:
        task_pack = json.loads(DEFAULT_REVIEW_TASKS.read_text(encoding="utf-8"))
        tasks = list(task_pack.get("tasks") or [])

    current_cards = []
    for task in tasks:
        query = task.get("query") or ""
        mode = task.get("mode") or "ask"
        scope = task.get("doc_ids") if "doc_ids" in task else None
        current_cards.append(
            build_card(
                query,
                corpus=corpus,
                mode=mode,
                task_id=task.get("id"),
                expect_status=task.get("expect_status"),
                corpus_digest=(
                    unscoped_digest
                    if scope is None
                    else corpus_digest(corpus, doc_ids=scope)
                ),
                doc_ids=scope,
            )
        )

    validated = merge_prior_labels(current_cards, state)
    pending = unlabeled(validated)
    current_labels = [card for card in validated if card.get("usefulness_label")]
    invalidated = [
        card
        for card in validated
        if str(card.get("prior_label_status") or "").startswith("IGNORED_")
    ]
    invalidated_by_reason = Counter(
        str(card["prior_label_status"]).removeprefix("IGNORED_")
        for card in invalidated
    )
    needs_review = not validated or bool(pending)
    review_command = f"python -m wedge_v1 review --corpus {corpus} --interactive"
    return {
        "n_cards_cached": len(cached_cards),
        "n_cards_current": len(validated),
        "n_labeled": len(current_labels),
        "n_unlabeled_cached": len(pending),
        "n_invalidated_labels": len(invalidated),
        "invalidated_by_reason": dict(sorted(invalidated_by_reason.items())),
        "needs_review": needs_review,
        "review_path": str(REVIEW_PATH),
        "next": (
            review_command
            if needs_review
            else f"python -m wedge_v1 review --corpus {corpus} --summary"
        ),
    }


def rerun_saved(
    corpus: Path,
    *,
    limit: int = 5,
    path: Path = SAVED_QUESTIONS,
) -> list[dict]:
    """Re-verify saved questions and persist corpus/result digests."""
    data = {
        "schema": "nano-lm.wedge_v1.saved_questions.v1",
        "questions": load_saved_questions(path),
    }
    try:
        all_docs = load_corpus(corpus)
    except Exception:
        all_docs = {}
    rows = []
    for q in data["questions"][:limit]:
        mode = q.get("mode") or "ask"
        if mode != "ask":
            continue
        task_id = _effective_task_id(q)
        scope = normalize_doc_ids(q.get("doc_ids")) if "doc_ids" in q else None
        current_corpus_digest = corpus_digest(corpus, doc_ids=scope)
        docs, scope_fields, scope_valid = select_documents(all_docs, scope)
        current_task_fingerprint = task_fingerprint(
            q["query"],
            mode=mode,
            task_id=task_id,
            doc_ids=scope,
        )
        try:
            if scope is None:
                r = ask(q["query"], corpus_dir=corpus)
            else:
                r = ask(q["query"], corpus_dir=corpus, doc_ids=scope)
        except Exception as exc:
            rows.append(
                {
                    "query": q["query"],
                    "task_id": task_id,
                    "answer_status": "ABSTAIN",
                    "n_claims": 0,
                    "latency_s": None,
                    "contradiction_banner": None,
                    "verification_state": "REFRESH_FAILED",
                    "failure_codes": [f"SAVED_ANSWER_REFRESH_{type(exc).__name__.upper()}"],
                    **scope_fields,
                }
            )
            continue
        valid, audit = _audit_answer(
            r,
            docs,
            query=q["query"],
            doc_ids=scope,
        )
        valid = valid and scope_valid
        if not valid:
            rows.append(
                {
                    "query": q["query"],
                    "task_id": task_id,
                    "answer_status": "ABSTAIN",
                    "n_claims": 0,
                    "latency_s": r.get("latency_s"),
                    "contradiction_banner": None,
                    "verification_state": "REFRESH_FAILED",
                    "failure_codes": list(audit.get("failure_codes") or []),
                    **scope_fields,
                }
            )
            continue
        current_result_digest = _result_digest(r)
        prior_result_digest = q.get("last_result_digest")
        saved_provenance = q.get("provenance")
        verified_provenance = q.get("last_verified_provenance")
        if prior_result_digest is None:
            verification_state = "VERIFIED"
        elif (
            not isinstance(saved_provenance, dict)
            or saved_provenance.get("schema") != PROVENANCE_SCHEMA
            or not saved_provenance.get("task_fingerprint")
            or not isinstance(verified_provenance, dict)
            or verified_provenance.get("schema") != PROVENANCE_SCHEMA
            or not verified_provenance.get("task_fingerprint")
            or not verified_provenance.get("corpus_digest")
        ):
            verification_state = "LEGACY_REVERIFIED"
        elif saved_provenance.get("task_fingerprint") != current_task_fingerprint:
            verification_state = "TASK_REVERIFIED"
        elif verified_provenance.get("task_fingerprint") != current_task_fingerprint:
            verification_state = "TASK_REVERIFIED"
        elif verified_provenance.get("corpus_digest") != current_corpus_digest:
            verification_state = "REVERIFIED"
        elif prior_result_digest != current_result_digest:
            verification_state = "CHANGED"
        else:
            verification_state = "CURRENT"
        _persist_verified_answer(
            q,
            r,
            corpus_digest_value=current_corpus_digest,
            task_id=task_id,
            doc_ids=scope,
        )
        rows.append(
            {
                "query": q["query"],
                "task_id": task_id,
                "answer_status": r.get("answer_status"),
                "n_claims": len(r.get("claims") or []),
                "latency_s": r.get("latency_s"),
                "contradiction_banner": r.get("contradiction_banner"),
                "verification_state": verification_state,
                **scope_fields,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return rows


def session(
    corpus: Path | None = None,
    *,
    rerun: bool = False,
    habit_path: Path = HABIT_PATH,
    saved_path: Path = SAVED_QUESTIONS,
    session_path: Path = SESSION_PATH,
) -> dict:
    """Compact daily/session habit surface with a single next action."""
    corpus_path = resolve_session_corpus(corpus)
    stats = corpus_stats(corpus_path) if corpus_path.exists() else {"n_docs": 0, "error": "missing"}
    docs = recent_documents(corpus_path)
    disputed = disputed_from_scan(corpus_path) if stats.get("n_docs") else []
    week = weekly_summary(path=habit_path)
    rq = review_queue_summary(corpus_path)
    saved = load_saved_questions(saved_path)
    memory = saved_question_status(corpus_path, path=saved_path) if saved else []
    rerun_rows = rerun_saved(corpus_path, limit=3, path=saved_path) if saved and rerun else []
    if rerun_rows:
        memory = saved_question_status(corpus_path, path=saved_path)

    next_action = "python -m wedge_v1 review --demo --interactive"
    if not stats.get("n_docs"):
        next_action = (
            "Set OWNER_CORPUS or run: python -m wedge_v1 owner-ready --demo"
        )
    elif rq["needs_review"]:
        next_action = f"python -m wedge_v1 review --corpus {corpus_path} --interactive"
    elif any(item["state"] != "CURRENT" for item in memory):
        next_action = f"python -m wedge_v1 habit --rerun --corpus {corpus_path}"
    elif disputed:
        next_action = f'python -m wedge_v1 compare "<disputed-term>" --corpus {corpus_path}'
    elif saved:
        next_action = f"python -m wedge_v1 habit --rerun --corpus {corpus_path}"
    else:
        next_action = f'python -m wedge_v1 ask "…" --corpus {corpus_path}'

    out = {
        "schema": "nano-lm.wedge_v1.habit_session.v1",
        "corpus": str(corpus_path.resolve()) if corpus_path.exists() else str(corpus_path),
        "ingest": stats,
        "recent_documents": docs,
        "disputed_or_contradictions": disputed,
        "saved_questions_n": len(saved),
        "saved_question_states": memory,
        "rerun_sample": rerun_rows,
        "weekly": week,
        "review_queue": rq,
        "next_action": next_action,
        "private_corpus_configured": bool(
            (os.environ.get("OWNER_CORPUS") or "").strip()
            or ((ROOT / "data" / "owner_corpus").is_dir()
                and any((ROOT / "data" / "owner_corpus").rglob("*.md")))
        ),
        "note": "Local workflow state; fixture results do not establish private-corpus usefulness.",
    }
    session_path.parent.mkdir(parents=True, exist_ok=True)
    session_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    record("habit_session", note=f"docs={stats.get('n_docs')}", path=habit_path)
    return out


def format_session_md(sess: dict) -> str:
    lines = [
        "# wedge_v1 habit session",
        "",
        f"**Corpus:** `{sess.get('corpus')}`",
        f"**Docs:** {(sess.get('ingest') or {}).get('n_docs')}",
        f"**Next:** `{sess.get('next_action')}`",
        "",
        "## Recent documents",
    ]
    for d in sess.get("recent_documents") or []:
        lines.append(f"- `{d['doc_id']}` ({d['suffix']}) mtime={d['mtime'][:19]}")
    if not sess.get("recent_documents"):
        lines.append("_none_")
    lines += ["", "## Contradictions / disputes"]
    for d in sess.get("disputed_or_contradictions") or []:
        lines.append(f"- {d.get('doc_id')}: {d.get('value')} ({d.get('status')})")
    if not sess.get("disputed_or_contradictions"):
        lines.append("_none detected in scan_")
    lines += ["", "## Review queue", json.dumps(sess.get("review_queue"), indent=2), ""]
    lines += ["## Weekly", json.dumps(sess.get("weekly"), indent=2), ""]
    if not sess.get("private_corpus_configured"):
        lines.append(
            "_Private corpus not configured — using a fixture or public path. "
            "Private usefulness validation remains pending._"
        )
    lines.append("")
    return "\n".join(lines)
