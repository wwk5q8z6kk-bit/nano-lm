"""Usefulness review loop — label ask/dogfood outcomes without hand-editing JSON.

Local/gitignored store. Not Layer-1 evidence. No LM.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wedge_v1.coe.canonical import canonical_result, canonical_result_fingerprint
from wedge_v1.ingest import document_id
from wedge_v1.private_output import require_private_output
from wedge_v1.runtime import ask, compare, find_spans, normalize_doc_ids

ROOT = Path(__file__).resolve().parent
REVIEW_PATH = ROOT / "results_review_state.json"
DEFAULT_DOGFOOD = ROOT / "results_wedge_v1_dogfood.json"
OWNER_DOGFOOD = ROOT / "results_owner_dogfood.json"

LABELS = (
    "USEFUL",
    "PARTIALLY_USEFUL",
    "NOT_USEFUL",
    "CORRECT_ABSTENTION",
    "OVER_ABSTENTION",
    "WRONG_EVIDENCE",
    "RETRIEVAL_MISS",
    "INGESTION_FAILURE",
    "CONTRADICTION_HANDLED",
)

LABEL_SHORT = {
    "u": "USEFUL",
    "p": "PARTIALLY_USEFUL",
    "n": "NOT_USEFUL",
    "c": "CORRECT_ABSTENTION",
    "o": "OVER_ABSTENTION",
    "w": "WRONG_EVIDENCE",
    "r": "RETRIEVAL_MISS",
    "i": "INGESTION_FAILURE",
    "x": "CONTRADICTION_HANDLED",
    "s": "SKIP",
    "q": "QUIT",
}

REVIEWER_KINDS = (
    "agent_applied",
    "owner",
    "independent_human",
    "unspecified",
)

FAILURE_CLASSES = (
    "PARTIAL_UTILITY",
    "NOT_USEFUL",
    "OVER_ABSTENTION",
    "WRONG_EVIDENCE",
    "RETRIEVAL_MISS",
    "INGESTION_FAILURE",
    "OTHER_PRODUCT_FIT",
)

_DEFAULT_FAILURE_CLASS = {
    "PARTIALLY_USEFUL": "PARTIAL_UTILITY",
    "NOT_USEFUL": "NOT_USEFUL",
    "OVER_ABSTENTION": "OVER_ABSTENTION",
    "WRONG_EVIDENCE": "WRONG_EVIDENCE",
    "RETRIEVAL_MISS": "RETRIEVAL_MISS",
    "INGESTION_FAILURE": "INGESTION_FAILURE",
}
FAILURE_LABELS = frozenset(_DEFAULT_FAILURE_CLASS)

_REVIEW_FIELDS = (
    "usefulness_label",
    "failure_reason",
    "suggested_correction",
    "correction_reason",
    "failure_class",
    "reviewer_kind",
    "review_elapsed_s",
    "review_seconds",
    "labeled_at",
)

PROVENANCE_SCHEMA = "nano-lm.wedge_v1.state_provenance.v1"
_CORPUS_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}
_REVIEW_OUTPUT_KEYS = (
    "answer_status",
    "answer_or_abstention",
    "evidence_span",
    "evidence",
    "document",
    "comparison_values",
    "claims",
    "hits",
    "solver_used",
    "verifier_outcome",
    "contradiction_banner",
    "abstain_reason",
    "n_claims",
    "repeat_recall",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def card_id(
    query: str,
    corpus: str,
    mode: str = "ask",
    doc_ids: list[str] | None = None,
) -> str:
    scope = normalize_doc_ids(doc_ids)
    base = f"{mode}|{corpus}|{query}"
    if scope is not None:
        base += "|" + json.dumps(scope, separators=(",", ":"))
    raw = base.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def corpus_content_digest(
    corpus: Path,
    doc_ids: list[str] | None = None,
) -> str | None:
    """Hash supported corpus file paths and bytes without persisting contents."""
    corpus = Path(corpus)
    if not corpus.is_dir():
        return None
    scope = normalize_doc_ids(doc_ids)
    selected = set(scope) if scope is not None else None
    files = [
        path
        for path in corpus.rglob("*")
        if path.is_file()
        and path.suffix.lower() in _CORPUS_SUFFIXES
        and not any(part.startswith(".") for part in path.relative_to(corpus).parts)
        and (selected is None or document_id(path, corpus) in selected)
    ]
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(corpus).as_posix()):
        digest.update(path.relative_to(corpus).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def task_fingerprint(
    query: str,
    *,
    mode: str = "ask",
    task_id: str | None = None,
    expect_status: list | None = None,
    doc_ids: list[str] | None = None,
) -> str:
    """Hash the stable task definition; no query text is stored in provenance."""
    expected = sorted(str(value) for value in (expect_status or []))
    definition = {
        "expect_status": expected,
        "mode": str(mode),
        "query": str(query),
        "task_id": str(task_id) if task_id is not None else None,
    }
    scope = normalize_doc_ids(doc_ids)
    if scope is not None:
        definition["doc_ids"] = scope
    raw = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _stable_result(value: Any) -> Any:
    return canonical_result(value)


def result_output_fingerprint(result: dict, review_output: dict) -> str:
    """Hash stable result and review output fields; persist no source text."""
    projection = {
        "result": result,
        "review_output": {
            key: review_output.get(key) for key in _REVIEW_OUTPUT_KEYS
        },
    }
    return canonical_result_fingerprint(projection)


def provenance_record(
    query: str,
    *,
    corpus: Path | None = None,
    corpus_digest: str | None = None,
    mode: str = "ask",
    task_id: str | None = None,
    expect_status: list | None = None,
    result_fingerprint: str | None = None,
    doc_ids: list[str] | None = None,
) -> dict:
    if corpus_digest is None and corpus is not None:
        corpus_digest = corpus_content_digest(corpus, doc_ids=doc_ids)
    record = {
        "schema": PROVENANCE_SCHEMA,
        "corpus_digest": corpus_digest,
        "task_fingerprint": task_fingerprint(
            query,
            mode=mode,
            task_id=task_id,
            expect_status=expect_status,
            doc_ids=doc_ids,
        ),
        "result_fingerprint": result_fingerprint,
    }
    scope = normalize_doc_ids(doc_ids)
    if scope is not None:
        record["doc_ids"] = scope
    return record


def load_state(path: Path | None = None) -> dict:
    path = path if path is not None else REVIEW_PATH
    if not path.is_file():
        return {"schema": "nano-lm.wedge_v1.review.v1", "cards": {}, "labels": {}}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {
            "schema": "nano-lm.wedge_v1.review.v1",
            "cards": {},
            "labels": {},
            "load_errors": ["REVIEW_STATE_INVALID_JSON"],
        }
    if not isinstance(state, dict):
        return {
            "schema": "nano-lm.wedge_v1.review.v1",
            "cards": {},
            "labels": {},
            "load_errors": ["REVIEW_STATE_INVALID_SHAPE"],
        }
    # This key is generated by the loader, never trusted from persisted input.
    # Accepting it would echo attacker-controlled paths or non-string objects
    # through summaries and parser errors.
    if "load_errors" in state:
        return {
            "schema": "nano-lm.wedge_v1.review.v1",
            "cards": {},
            "labels": {},
            "load_errors": ["REVIEW_STATE_INVALID_SHAPE"],
        }
    cards = state.get("cards")
    labels = state.get("labels")
    active_ids = state.get("active_card_ids")
    valid_cards = isinstance(cards, dict) and all(
        isinstance(card_id, str)
        and isinstance(card, dict)
        and isinstance(card.get("id"), str)
        for card_id, card in (cards.items() if isinstance(cards, dict) else [])
    )
    valid_labels = isinstance(labels, dict) and all(
        isinstance(card_id, str)
        and isinstance(label, str)
        and label in LABELS
        and isinstance(cards, dict)
        and card_id in cards
        for card_id, label in (labels.items() if isinstance(labels, dict) else [])
    )
    valid_active_ids = active_ids is None or (
        isinstance(active_ids, list)
        and all(
            isinstance(card_id, str)
            and isinstance(cards, dict)
            and card_id in cards
            for card_id in active_ids
        )
    )
    if not valid_cards or not valid_labels or not valid_active_ids:
        return {
            "schema": "nano-lm.wedge_v1.review.v1",
            "cards": {},
            "labels": {},
            "load_errors": ["REVIEW_STATE_INVALID_SHAPE"],
        }
    return state


def save_state(state: dict, path: Path | None = None) -> None:
    path = path if path is not None else REVIEW_PATH
    require_private_output(path, purpose="review state")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def snapshot_review_cards(state: dict, cards: list[dict]) -> dict:
    """Persist the active review queue so read-only resume/next never reruns solvers."""
    stored = state.setdefault("cards", {})
    active_ids = []
    for card in cards:
        cid = card["id"]
        stored[cid] = dict(card)
        active_ids.append(cid)
    state["active_card_ids"] = active_ids
    state["sensitivity"] = "OWNER_PRIVATE"
    state["snapshot_at"] = _now()
    return state


def cards_from_state(state: dict) -> list[dict]:
    """Return the active persisted queue, or legacy stored cards when necessary."""
    if not isinstance(state, dict) or not isinstance(state.get("cards"), dict):
        return []
    stored = state["cards"]
    active_ids = state.get("active_card_ids")
    if isinstance(active_ids, list):
        return [stored[cid] for cid in active_ids if cid in stored and isinstance(stored[cid], dict)]
    return [card for card in stored.values() if isinstance(card, dict)]


def _primary_span(result: dict) -> dict[str, Any]:
    for c in result.get("claims") or []:
        for e in c.get("evidence") or []:
            if isinstance(e, dict):
                return {
                    "text": e.get("text") or e.get("line") or "",
                    "start": e.get("start"),
                    "end": e.get("end"),
                    "doc_id": c.get("doc_id"),
                    "task_id": c.get("task_id"),
                    "value": c.get("value"),
                }
    for h in result.get("hits") or []:
        if isinstance(h, dict):
            return {
                "text": h.get("text") or h.get("context") or "",
                "start": h.get("start"),
                "end": h.get("end"),
                "doc_id": h.get("doc_id"),
                "task_id": "FIND",
                "value": h.get("text"),
            }
    return {}


def _comparison_values(result: dict) -> list[dict[str, Any]]:
    rows = []
    values_by_doc = result.get("values_by_doc") or {}
    if not isinstance(values_by_doc, dict):
        return rows
    for doc_id, raw_values in sorted(values_by_doc.items(), key=lambda item: str(item[0])):
        values = raw_values if isinstance(raw_values, (list, tuple, set)) else [raw_values]
        clean_values = []
        for value in values:
            text = str(value).strip() if value is not None else ""
            if text and text not in clean_values:
                clean_values.append(text)
        if clean_values:
            rows.append({"doc_id": str(doc_id), "values": clean_values})
    return rows


def _comparison_summary(rows: list[dict[str, Any]]) -> str:
    return "; ".join(
        f"{row['doc_id']}={','.join(str(value) for value in row['values'])}"
        for row in rows
    )


def _review_hits(result: dict, *, mode: str) -> list[dict[str, Any]]:
    raw_hits = result.get("hits") or []
    if raw_hits:
        return _stable_result(raw_hits)
    if mode != "find":
        return []
    rows = []
    for claim in result.get("claims") or []:
        if not isinstance(claim, dict):
            continue
        for evidence in claim.get("evidence") or []:
            if not isinstance(evidence, dict):
                continue
            rows.append(
                {
                    **evidence,
                    "doc_id": evidence.get("doc_id") or claim.get("doc_id"),
                    "status": claim.get("status"),
                    "value": claim.get("value"),
                    "claim_id": claim.get("claim_id"),
                }
            )
    return _stable_result(rows)


def card_from_result(
    query: str,
    result: dict,
    *,
    corpus: Path,
    mode: str = "ask",
    task_id: str | None = None,
    expect_status: list | None = None,
    corpus_digest: str | None = None,
    doc_ids: list[str] | None = None,
    manual_baseline_seconds: float | None = None,
) -> dict:
    """Build a review card from an already-executed audited solver result."""
    scope = normalize_doc_ids(doc_ids)
    span = _primary_span(result)
    comparison_values = _comparison_values(result) if mode == "compare" else []
    comparison_summary = _comparison_summary(comparison_values)
    claims = _stable_result(result.get("claims") or [])
    hits = _review_hits(result, mode=mode)
    status = result.get("answer_status")
    resolved_corpus = Path(corpus).resolve()
    cid = card_id(query, str(resolved_corpus), mode, doc_ids=scope)
    effective_task_id = task_id or cid
    card = {
        "id": cid,
        "task_id": effective_task_id,
        "query": query,
        "task_class": mode,
        "corpus": str(resolved_corpus),
        "answer_status": status,
        "answer_or_abstention": status,
        "evidence_span": comparison_summary or span.get("text") or None,
        "evidence": span,
        "document": (
            ", ".join(row["doc_id"] for row in comparison_values)
            if comparison_values
            else span.get("doc_id")
        ),
        "comparison_values": comparison_values,
        # The review instrument must preserve the complete presented result.  The
        # legacy primary-span fields above remain for older state readers only.
        "claims": claims,
        "hits": hits,
        "solver_used": result.get("solver_path") or [],
        "latency_s": result.get("latency_s"),
        "latency_ms": result.get("latency_ms"),
        "verifier_outcome": status,
        "contradiction_banner": result.get("contradiction_banner"),
        "abstain_reason": result.get("note") or result.get("abstain_reason"),
        "n_claims": len(claims),
        "n_hits": len(hits),
        "expect_status": expect_status,
        "usefulness_label": None,
        "failure_reason": None,
        "suggested_correction": None,
        "correction_reason": None,
        "failure_class": None,
        "reviewer_kind": None,
        "review_elapsed_s": None,
        # Deprecated compatibility alias for pre-Priority-5 study state.
        "review_seconds": None,
        "manual_baseline_seconds": manual_baseline_seconds,
        "coe_audit_ok": (result.get("coe_audit") or {}).get("ok"),
        "built_at": _now(),
    }
    if scope is not None:
        card["selected_doc_ids"] = list(result.get("selected_doc_ids") or [])
        card["missing_doc_ids"] = list(result.get("missing_doc_ids") or [])
    card["provenance"] = provenance_record(
        query,
        corpus=resolved_corpus,
        corpus_digest=corpus_digest,
        mode=mode,
        task_id=effective_task_id,
        expect_status=expect_status,
        result_fingerprint=result_output_fingerprint(result, card),
        doc_ids=scope,
    )
    return card


def build_card(
    query: str,
    *,
    corpus: Path,
    mode: str = "ask",
    task_id: str | None = None,
    expect_status: list | None = None,
    corpus_digest: str | None = None,
    doc_ids: list[str] | None = None,
    persist_coe: bool = True,
    manual_baseline_seconds: float | None = None,
) -> dict:
    """Execute one solver and project its result into the review instrument."""
    scope = normalize_doc_ids(doc_ids)
    if mode == "compare":
        result = compare(
            query, corpus_dir=corpus, doc_ids=scope, persist_coe=persist_coe
        )
    elif mode == "find":
        result = find_spans(
            query, corpus_dir=corpus, doc_ids=scope, persist_coe=persist_coe
        )
    else:
        result = ask(query, corpus_dir=corpus, doc_ids=scope, persist_coe=persist_coe)
    return card_from_result(
        query,
        result,
        corpus=corpus,
        mode=mode,
        task_id=task_id,
        expect_status=expect_status,
        corpus_digest=corpus_digest,
        doc_ids=scope,
        manual_baseline_seconds=manual_baseline_seconds,
    )


def cards_from_dogfood(
    dogfood_path: Path,
    corpus: Path,
    *,
    limit: int | None = None,
    persist_coe: bool = True,
) -> list[dict]:
    pack = json.loads(dogfood_path.read_text(encoding="utf-8"))
    # Prefer task queries from companion tasks file if present
    rows = pack.get("rows") or []
    unscoped_digest = corpus_content_digest(corpus)
    cards = []
    for row in rows:
        q = row.get("query")
        if not q:
            continue
        mode = row.get("mode") or "ask"
        scope = row.get("doc_ids") if "doc_ids" in row else None
        card = build_card(
            q,
            corpus=corpus,
            mode=mode,
            task_id=row.get("id"),
            expect_status=row.get("expect_status"),
            corpus_digest=(
                unscoped_digest
                if scope is None
                else corpus_content_digest(corpus, doc_ids=scope)
            ),
            doc_ids=scope,
            persist_coe=persist_coe,
        )
        cards.append(card)
        if limit is not None and len(cards) >= limit:
            break
    return cards


def cards_from_task_pack(
    tasks_path: Path,
    corpus: Path,
    *,
    limit: int | None = None,
    persist_coe: bool = True,
) -> list[dict]:
    pack = json.loads(tasks_path.read_text(encoding="utf-8"))
    cards = []
    unscoped_digest = corpus_content_digest(corpus)
    for t in pack.get("tasks") or []:
        scope = t.get("doc_ids") if "doc_ids" in t else None
        card = build_card(
            t["query"],
            corpus=corpus,
            mode=t.get("mode") or "ask",
            task_id=t.get("id"),
            expect_status=t.get("expect_status"),
            corpus_digest=(
                unscoped_digest
                if scope is None
                else corpus_content_digest(corpus, doc_ids=scope)
            ),
            doc_ids=scope,
            persist_coe=persist_coe,
            manual_baseline_seconds=t.get("manual_baseline_seconds"),
        )
        cards.append(card)
        if limit is not None and len(cards) >= limit:
            break
    return cards


def _prior_label_candidate(card: dict, state: dict) -> tuple[str | None, dict, str | None]:
    if not isinstance(state, dict):
        return None, {}, None
    labels = state.get("labels")
    prior_cards = state.get("cards")
    if not isinstance(labels, dict) or not isinstance(prior_cards, dict):
        return None, {}, None
    cid = card["id"]
    direct = prior_cards.get(cid) or {}
    direct_label = labels.get(cid) or direct.get("usefulness_label")
    if direct_label:
        return cid, direct, direct_label

    task_id = card.get("task_id")
    if task_id is None:
        return None, {}, None
    candidates = []
    for prior_id, prior in prior_cards.items():
        if not isinstance(prior, dict):
            continue
        if str(prior.get("task_id")) != str(task_id):
            continue
        label = labels.get(prior_id) or prior.get("usefulness_label")
        if label:
            candidates.append((prior_id, prior, label))
    if not candidates:
        return None, {}, None
    candidates.sort(key=lambda item: str(item[1].get("labeled_at") or ""), reverse=True)
    return candidates[0]


def _provenance_relation(current: dict | None, prior: dict | None) -> str:
    if not isinstance(current, dict) or not isinstance(prior, dict):
        return "LEGACY_MISSING_PROVENANCE"
    required = {"corpus_digest", "task_fingerprint"}
    if (
        current.get("schema") != PROVENANCE_SCHEMA
        or prior.get("schema") != PROVENANCE_SCHEMA
        or not required.issubset(current)
        or not required.issubset(prior)
        or any(not current.get(key) for key in required)
        or any(not prior.get(key) for key in required)
    ):
        return "LEGACY_MISSING_PROVENANCE"
    if not current.get("result_fingerprint") or not prior.get("result_fingerprint"):
        return "LEGACY_MISSING_RESULT_FINGERPRINT"
    if prior.get("task_fingerprint") != current.get("task_fingerprint"):
        return "TASK_CHANGED"
    if prior.get("corpus_digest") != current.get("corpus_digest"):
        return "CORPUS_CHANGED"
    if prior.get("result_fingerprint") != current.get("result_fingerprint"):
        return "RESULT_CHANGED"
    return "MATCH"


def _correction_reason(card: dict) -> str | None:
    """Read the canonical reason, with legacy state fallbacks."""
    for key in ("correction_reason", "failure_reason"):
        value = card.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _review_elapsed_s(card: dict) -> float | None:
    """Read a JSON-safe non-negative duration from new or legacy state."""
    value = card.get("review_elapsed_s")
    if value is None:
        value = card.get("review_seconds")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value) or value < 0:
        return None
    return value


def _valid_prior_review_state(prior: dict, label: Any) -> bool:
    """Accept only the canonical state shape produced by ``apply_label``."""
    if not isinstance(prior, dict) or not isinstance(label, str) or label not in LABELS:
        return False
    if prior.get("usefulness_label") != label:
        return False

    reviewer_kind = prior.get("reviewer_kind")
    if not isinstance(reviewer_kind, str) or reviewer_kind not in REVIEWER_KINDS:
        return False

    failure_class = prior.get("failure_class")
    if failure_class in (None, ""):
        failure_class = None
    elif not isinstance(failure_class, str) or failure_class not in FAILURE_CLASSES:
        return False
    if label in FAILURE_LABELS:
        if failure_class is None:
            return False
    elif failure_class is not None:
        return False

    elapsed_values = [
        prior.get(key)
        for key in ("review_elapsed_s", "review_seconds")
        if prior.get(key) is not None
    ]
    normalized_elapsed = []
    for value in elapsed_values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        value = float(value)
        if not math.isfinite(value) or value < 0:
            return False
        normalized_elapsed.append(value)
    if len(normalized_elapsed) == 2 and normalized_elapsed[0] != normalized_elapsed[1]:
        return False
    return True


def _clear_review_metadata(card: dict) -> None:
    """Fail closed by removing every restored or incoming review field."""
    for key in (
        "usefulness_label",
        "failure_reason",
        "suggested_correction",
        "correction_reason",
        "failure_class",
        "reviewer_kind",
        "review_elapsed_s",
        "review_seconds",
        "labeled_at",
    ):
        card[key] = None


def merge_prior_labels(cards: list[dict], state: dict) -> list[dict]:
    """Restore labels only when corpus and task provenance both match."""
    out = []
    for c in cards:
        prior_id, prior, lab = _prior_label_candidate(c, state)
        c = dict(c)
        if lab:
            relation = _provenance_relation(c.get("provenance"), prior.get("provenance"))
            if relation != "MATCH":
                c["prior_label_status"] = f"IGNORED_{relation}"
                c["prior_label_id"] = prior_id
                _clear_review_metadata(c)
                out.append(c)
                continue
            if not _valid_prior_review_state(prior, lab):
                c["prior_label_status"] = "IGNORED_INVALID_REVIEW_STATE"
                c["prior_label_id"] = prior_id
                _clear_review_metadata(c)
                out.append(c)
                continue
            c["usefulness_label"] = lab
            c["failure_reason"] = prior.get("failure_reason")
            c["suggested_correction"] = prior.get("suggested_correction")
            c["correction_reason"] = _correction_reason(prior)
            c["failure_class"] = prior.get("failure_class")
            c["reviewer_kind"] = prior.get("reviewer_kind")
            review_elapsed_s = _review_elapsed_s(prior)
            c["review_elapsed_s"] = review_elapsed_s
            c["review_seconds"] = review_elapsed_s
            c["labeled_at"] = prior.get("labeled_at")
            c["prior_label_status"] = "RESTORED"
        out.append(c)
    return out


def apply_label(
    state: dict,
    card: dict,
    label: str,
    *,
    failure_reason: str = "",
    suggested_correction: str = "",
    correction_reason: str = "",
    failure_class: str | None = None,
    reviewer_kind: str = "unspecified",
    review_elapsed_s: float | None = None,
    review_seconds: float | None = None,
) -> dict:
    if (
        not isinstance(state, dict)
        or not isinstance(state.get("cards", {}), dict)
        or not isinstance(state.get("labels", {}), dict)
    ):
        raise ValueError("review state must contain object-valued cards and labels")
    audit = state.get("audit_log")
    if audit is None:
        audit = []
        state["audit_log"] = audit
    if not isinstance(audit, list):
        raise ValueError("review state audit_log must be a list")
    label = label.upper().strip()
    if label not in LABELS:
        raise ValueError(f"unknown label {label}; choose from {LABELS}")
    reviewer_kind = reviewer_kind.strip().lower()
    if reviewer_kind not in REVIEWER_KINDS:
        raise ValueError(
            f"unknown reviewer kind {reviewer_kind}; choose from {REVIEWER_KINDS}"
        )
    normalized_failure = (
        failure_class or _DEFAULT_FAILURE_CLASS.get(label) or ""
    ).upper().strip()
    if normalized_failure and normalized_failure not in FAILURE_CLASSES:
        raise ValueError(
            f"unknown failure class {normalized_failure}; choose from {FAILURE_CLASSES}"
        )
    if normalized_failure and label not in FAILURE_LABELS:
        raise ValueError(f"failure class is not valid for successful label {label}")
    elapsed_field = "review_elapsed_s"
    if review_elapsed_s is None:
        review_elapsed_s = review_seconds
        elapsed_field = "review_seconds"
    if review_elapsed_s is not None:
        if isinstance(review_elapsed_s, bool):
            raise ValueError(f"{elapsed_field} must be finite and non-negative")
        review_elapsed_s = float(review_elapsed_s)
        if not math.isfinite(review_elapsed_s) or review_elapsed_s < 0:
            raise ValueError(f"{elapsed_field} must be finite and non-negative")
    cid = card["id"]
    prior = (state.get("cards") or {}).get(cid)
    previous_review = (
        {key: prior.get(key) for key in _REVIEW_FIELDS}
        if isinstance(prior, dict) and prior.get("usefulness_label")
        else None
    )
    card = dict(card)
    card["usefulness_label"] = label
    card["failure_reason"] = failure_reason or None
    card["suggested_correction"] = suggested_correction or None
    card["correction_reason"] = _correction_reason(
        {
            "correction_reason": correction_reason,
            "failure_reason": failure_reason,
        }
    )
    card["failure_class"] = normalized_failure or None
    card["reviewer_kind"] = reviewer_kind
    card["review_elapsed_s"] = review_elapsed_s
    # Keep the old field readable while downstream study state migrates.
    card["review_seconds"] = review_elapsed_s
    labeled_at = _now()
    card["labeled_at"] = labeled_at
    state.setdefault("cards", {})[cid] = card
    state.setdefault("labels", {})[cid] = label
    audit.append(
        {
            "schema": "nano-lm.wedge_v1.review_event.v1",
            "event_id": f"review-{len(audit) + 1:06d}",
            "action": "RELABEL" if previous_review else "LABEL",
            "at": labeled_at,
            "card_id": cid,
            "task_id": card.get("task_id"),
            "from_label": (
                previous_review.get("usefulness_label") if previous_review else None
            ),
            "to_label": label,
            "reviewer_kind": reviewer_kind,
            "previous_review": previous_review,
        }
    )
    return state


def undo_label(
    state: dict,
    cards: list[dict],
    task_or_card_id: str,
    *,
    reviewer_kind: str = "unspecified",
) -> dict:
    """Clear one judgment so the next review resumes it; retain an append-only audit."""
    reviewer_kind = reviewer_kind.strip().lower()
    if reviewer_kind not in REVIEWER_KINDS:
        raise ValueError(
            f"unknown reviewer kind {reviewer_kind}; choose from {REVIEWER_KINDS}"
        )
    if not isinstance(state, dict):
        raise ValueError("review state must be an object")
    stored = state.get("cards")
    labels = state.get("labels")
    audit = state.setdefault("audit_log", [])
    if not isinstance(stored, dict) or not isinstance(labels, dict) or not isinstance(audit, list):
        raise ValueError("review state must contain object cards/labels and list audit_log")
    by_id = {str(card.get("id")): card for card in cards}
    by_task = {str(card.get("task_id")): card for card in cards}
    base = by_id.get(str(task_or_card_id)) or by_task.get(str(task_or_card_id))
    if base is None:
        raise KeyError(f"no card for {task_or_card_id}")
    cid = str(base["id"])
    current = stored.get(cid) or base
    previous_label = labels.get(cid) or current.get("usefulness_label")
    if not previous_label:
        raise ValueError(f"card {task_or_card_id} is already unlabeled")
    previous_review = {key: current.get(key) for key in _REVIEW_FIELDS}
    cleared = dict(base)
    for key in _REVIEW_FIELDS:
        cleared[key] = None
    stored[cid] = cleared
    labels.pop(cid, None)
    audit.append(
        {
            "schema": "nano-lm.wedge_v1.review_event.v1",
            "event_id": f"review-{len(audit) + 1:06d}",
            "action": "UNDO",
            "at": _now(),
            "card_id": cid,
            "task_id": cleared.get("task_id"),
            "from_label": previous_label,
            "to_label": None,
            "reviewer_kind": reviewer_kind,
            "previous_review": previous_review,
        }
    )
    return state


def unlabeled(cards: list[dict]) -> list[dict]:
    return [c for c in cards if not c.get("usefulness_label")]


def _display_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _span_label(row: dict, *, fallback_doc: Any = None) -> str:
    doc_id = row.get("doc_id") or fallback_doc or "—"
    start = row.get("start")
    end = row.get("end")
    offsets = f"{start}:{end}" if start is not None or end is not None else "—"
    relation = row.get("relation") or "—"
    return f"doc={doc_id} span={offsets} relation={relation}"


def _format_claims(claims: list[dict]) -> list[str]:
    lines = [f"claims ({len(claims)}):"]
    for claim_index, claim in enumerate(claims, 1):
        if not isinstance(claim, dict):
            lines.append(f"  claim {claim_index}: {_display_value(claim)}")
            continue
        lines.append(
            "  claim "
            f"{claim_index}: task={claim.get('task_id') or '—'} "
            f"doc={claim.get('doc_id') or '—'} status={claim.get('status') or '—'} "
            f"value={_display_value(claim.get('value'))}"
        )
        evidence_rows = claim.get("evidence") or []
        if not evidence_rows:
            lines.append("    evidence: —")
            continue
        for evidence_index, evidence in enumerate(evidence_rows, 1):
            if not isinstance(evidence, dict):
                lines.append(
                    f"    evidence {evidence_index}: {_display_value(evidence)}"
                )
                continue
            lines.append(
                f"    evidence {evidence_index}: "
                f"{_span_label(evidence, fallback_doc=claim.get('doc_id'))} "
                f"text={_display_value(evidence.get('text') or evidence.get('line'))}"
            )
            if evidence.get("context") is not None:
                lines.append(
                    f"      context={_display_value(evidence.get('context'))}"
                )
    return lines


def _format_hits(hits: list[dict]) -> list[str]:
    lines = [f"hits ({len(hits)}):"]
    for hit_index, hit in enumerate(hits, 1):
        if not isinstance(hit, dict):
            lines.append(f"  hit {hit_index}: {_display_value(hit)}")
            continue
        lines.append(
            f"  hit {hit_index}: {_span_label(hit)} "
            f"status={hit.get('status') or '—'} "
            f"value={_display_value(hit.get('value'))} "
            f"text={_display_value(hit.get('text') or hit.get('line'))}"
        )
        if hit.get("context") is not None:
            lines.append(f"    context={_display_value(hit.get('context'))}")
    return lines


def format_card(card: dict, *, index: int | None = None, total: int | None = None) -> str:
    head = f"[{index}/{total}] " if index and total else ""
    lines = [
        f"{head}id={card.get('task_id')}  mode={card.get('task_class')}  status={card.get('answer_status')}",
        f"Q: {card.get('query')}",
    ]
    comparison_values = card.get("comparison_values") or []
    claims = card.get("claims") or []
    hits = card.get("hits") or []
    if comparison_values:
        lines.append(f"values: {_comparison_summary(comparison_values)}")
    if claims:
        lines.extend(_format_claims(claims))
    elif not comparison_values:
        lines.extend(
            [
                f"doc: {card.get('document') or '—'}",
                f"span: {(card.get('evidence_span') or '—')[:160]}",
            ]
        )
    if hits:
        lines.extend(_format_hits(hits))
    lines.extend(
        [
            f"solver: {' → '.join(str(s) for s in (card.get('solver_used') or []))}",
            f"latency: {card.get('latency_s')}s",
            f"verifier: {card.get('verifier_outcome')}",
        ]
    )
    if card.get("contradiction_banner"):
        lines.append(f"contradiction: {card['contradiction_banner']}")
    if card.get("abstain_reason"):
        lines.append(f"abstain: {card['abstain_reason']}")
    if card.get("usefulness_label"):
        lines.append(f"label: {card['usefulness_label']}")
    if str(card.get("prior_label_status") or "").startswith("IGNORED_"):
        lines.append(f"prior label: {card['prior_label_status']}")
    lines.append(
        "labels: [u]seful [p]artial [n]ot [c]orrect-abstain [o]ver-abstain "
        "[w]rong-evidence [r]etrieval-miss [i]ngest [x]contradiction [s]kip [q]uit"
    )
    lines.append(
        "failure syntax: label correction-reason | FAILURE_CLASS | suggested correction"
    )
    return "\n".join(lines)


def label_summary(
    state: dict,
    *,
    cards: list[dict] | None = None,
    path: Path | None = None,
) -> dict:
    if not isinstance(state, dict):
        state = {"cards": {}, "labels": {}, "load_errors": ["REVIEW_STATE_INVALID_SHAPE"]}
    labels = state.get("labels") or {}
    if not isinstance(labels, dict):
        labels = {}
    stored_cards = state.get("cards") or {}
    if not isinstance(stored_cards, dict):
        stored_cards = {}
    reviewed_cards = [
        stored_cards.get(card_id) or {}
        for card_id in labels
    ]
    if cards is not None:
        current_labels = {}
        reviewed_cards = []
        for card in cards:
            _prior_id, prior, label = _prior_label_candidate(card, state)
            if (
                label
                and _provenance_relation(
                    card.get("provenance"), prior.get("provenance")
                )
                == "MATCH"
                and _valid_prior_review_state(prior, label)
            ):
                current_labels[card["id"]] = label
                reviewed_cards.append(prior)
        labels = current_labels
    counts: dict[str, int] = {}
    for lab in labels.values():
        counts[lab] = counts.get(lab, 0) + 1
    review_times = [
        elapsed
        for card in reviewed_cards
        if (elapsed := _review_elapsed_s(card)) is not None
    ]
    return {
        "n_labeled": len(labels),
        "by_label": dict(sorted(counts.items())),
        "n_timed": len(review_times),
        "total_review_elapsed_s": sum(review_times) if review_times else None,
        "median_review_elapsed_s": (
            statistics.median(review_times) if review_times else None
        ),
        "n_with_correction_reason": sum(
            _correction_reason(card) is not None for card in reviewed_cards
        ),
        "load_errors": list(state.get("load_errors") or []),
        "path": str(path if path is not None else REVIEW_PATH),
    }


def interactive_review(
    cards: list[dict],
    state: dict,
    *,
    stdin=None,
    stdout=None,
    state_path: Path | None = None,
    reviewer_kind: str = "unspecified",
    clock=None,
) -> dict:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    clock = clock or time.perf_counter
    snapshot_review_cards(state, cards)
    save_state(state, path=state_path)
    queue = unlabeled(cards)
    total = len(queue)
    stop = False
    for i, card in enumerate(queue, 1):
        started = clock()
        stdout.write(format_card(card, index=i, total=total) + "\n> ")
        stdout.flush()
        while True:
            line = stdin.readline()
            if not line:
                stop = True
                break
            segments = [segment.strip() for segment in line.strip().split("|", 2)]
            first = segments[0].split(maxsplit=1)
            if not first:
                stdout.write("> ")
                stdout.flush()
                continue
            key = first[0].lower()
            if key in ("q", "quit"):
                stop = True
                break
            if key in ("s", "skip"):
                break
            label = LABEL_SHORT.get(key, key.upper())
            if label == "SKIP":
                break
            if label == "QUIT":
                stop = True
                break
            reason = first[1].strip() if len(first) > 1 else ""
            failure_class = (
                segments[1].upper() if len(segments) > 1 and segments[1] else None
            )
            correction = segments[2] if len(segments) > 2 else ""
            try:
                if label in FAILURE_LABELS and (not reason or not correction):
                    raise ValueError(
                        "failure labels require a concrete reason and suggested correction"
                    )
                apply_label(
                    state,
                    card,
                    label,
                    failure_reason=reason,
                    suggested_correction=correction,
                    correction_reason=reason,
                    failure_class=failure_class,
                    reviewer_kind=reviewer_kind,
                    review_elapsed_s=max(0.0, clock() - started),
                )
            except ValueError as e:
                stdout.write(f"{e}\n> ")
                stdout.flush()
                continue
            # A completed judgment is durable immediately.  If the terminal is
            # interrupted, the next invocation resumes at the next card.
            save_state(state, path=state_path)
            stdout.write(f"labeled {card.get('task_id')} → {label}\n")
            break
        if stop:
            break
    save_state(state, path=state_path)
    return state


def batch_label(
    state: dict,
    cards: list[dict],
    specs: list[str],
    *,
    state_path: Path | None = None,
    reviewer_kind: str = "unspecified",
) -> dict:
    """specs like TASKID:LABEL or id:LABEL."""
    by_task = {str(c.get("task_id")): c for c in cards}
    by_id = {c["id"]: c for c in cards}
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"bad label spec {spec}; use ID:LABEL")
        tid, lab = spec.split(":", 1)
        card = by_task.get(tid) or by_id.get(tid)
        if not card:
            raise KeyError(f"no card for {tid}")
        apply_label(state, card, lab, reviewer_kind=reviewer_kind)
    save_state(state, path=state_path)
    return state
