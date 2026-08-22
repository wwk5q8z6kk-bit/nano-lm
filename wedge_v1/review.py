"""Usefulness review loop — label ask/dogfood outcomes without hand-editing JSON.

Local/gitignored store. Not Layer-1 evidence. No LM.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wedge_v1.private_output import require_private_output
from wedge_v1.runtime import ask, compare, find_spans

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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def card_id(query: str, corpus: str, mode: str = "ask") -> str:
    raw = f"{mode}|{corpus}|{query}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def load_state(path: Path | None = None) -> dict:
    path = path if path is not None else REVIEW_PATH
    if not path.is_file():
        return {"schema": "nano-lm.wedge_v1.review.v1", "cards": {}, "labels": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(state: dict, path: Path | None = None) -> None:
    path = path if path is not None else REVIEW_PATH
    path = require_private_output(path, purpose="review state")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


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


def build_card(
    query: str,
    *,
    corpus: Path,
    mode: str = "ask",
    task_id: str | None = None,
    expect_status: list | None = None,
) -> dict:
    if mode == "compare":
        result = compare(query, corpus_dir=corpus)
    elif mode == "find":
        result = find_spans(query, corpus_dir=corpus)
    else:
        result = ask(query, corpus_dir=corpus)
    span = _primary_span(result)
    status = result.get("answer_status")
    cid = card_id(query, str(corpus.resolve()), mode)
    return {
        "id": cid,
        "task_id": task_id or cid,
        "query": query,
        "task_class": mode,
        "corpus": str(Path(corpus).resolve()),
        "answer_status": status,
        "answer_or_abstention": status,
        "evidence_span": span.get("text") or None,
        "evidence": span,
        "document": span.get("doc_id"),
        "solver_used": result.get("solver_path") or [],
        "latency_s": result.get("latency_s"),
        "latency_ms": result.get("latency_ms"),
        "verifier_outcome": status,
        "contradiction_banner": result.get("contradiction_banner"),
        "abstain_reason": result.get("note") or result.get("abstain_reason"),
        "n_claims": len(result.get("claims") or result.get("hits") or []),
        "expect_status": expect_status,
        "usefulness_label": None,
        "failure_reason": None,
        "suggested_correction": None,
        "built_at": _now(),
    }


def cards_from_dogfood(
    dogfood_path: Path,
    corpus: Path,
    *,
    limit: int | None = None,
    rerun: bool = False,
) -> list[dict]:
    """Build review cards from dogfood JSON.

    Default reuses stored statuses (no solver re-run) for fast labeling.
    Pass ``rerun=True`` to rebuild via ask/compare/find.
    """
    pack = json.loads(dogfood_path.read_text(encoding="utf-8"))
    rows = pack.get("rows") or []
    cards: list[dict] = []
    corpus_s = str(Path(corpus).resolve())
    for row in rows:
        q = row.get("query")
        if not q:
            continue
        mode = row.get("mode") or "ask"
        tid = row.get("id") or row.get("task_id")
        if rerun or row.get("got_status") is None:
            card = build_card(
                q,
                corpus=corpus,
                mode=mode,
                task_id=tid,
                expect_status=row.get("expect_status"),
            )
        else:
            status = row.get("got_status")
            cid = card_id(q, corpus_s, mode)
            card = {
                "id": cid,
                "task_id": tid or cid,
                "query": q,
                "task_class": mode,
                "corpus": corpus_s,
                "answer_status": status,
                "answer_or_abstention": status,
                "evidence_span": None,
                "evidence": {},
                "document": None,
                "solver_used": row.get("solver_path") or [],
                "latency_s": row.get("latency_s"),
                "latency_ms": None,
                "verifier_outcome": status,
                "contradiction_banner": None,
                "abstain_reason": row.get("note"),
                "n_claims": row.get("n_claims") or 0,
                "expect_status": row.get("expect_status"),
                "usefulness_label": None,
                "failure_reason": None,
                "suggested_correction": None,
                "built_at": _now(),
                "from_dogfood": True,
                "dogfood_ok": row.get("ok"),
                "fail_kind": row.get("fail_kind"),
            }
        cards.append(card)
        if limit is not None and len(cards) >= limit:
            break
    return cards


def cards_from_task_pack(tasks_path: Path, corpus: Path, *, limit: int | None = None) -> list[dict]:
    pack = json.loads(tasks_path.read_text(encoding="utf-8"))
    cards = []
    for t in pack.get("tasks") or []:
        card = build_card(
            t["query"],
            corpus=corpus,
            mode=t.get("mode") or "ask",
            task_id=t.get("id"),
            expect_status=t.get("expect_status"),
        )
        cards.append(card)
        if limit is not None and len(cards) >= limit:
            break
    return cards


def merge_prior_labels(cards: list[dict], state: dict) -> list[dict]:
    labels = state.get("labels") or {}
    prior_cards = state.get("cards") or {}
    out = []
    for c in cards:
        cid = c["id"]
        prior = prior_cards.get(cid) or {}
        lab = labels.get(cid) or prior.get("usefulness_label")
        if lab:
            c = dict(c)
            c["usefulness_label"] = lab
            c["failure_reason"] = prior.get("failure_reason")
            c["suggested_correction"] = prior.get("suggested_correction")
            c["labeled_at"] = prior.get("labeled_at")
        out.append(c)
    return out


def apply_label(
    state: dict,
    card: dict,
    label: str,
    *,
    failure_reason: str = "",
    suggested_correction: str = "",
) -> dict:
    label = label.upper().strip()
    if label not in LABELS:
        raise ValueError(f"unknown label {label}; choose from {LABELS}")
    cid = card["id"]
    card = dict(card)
    card["usefulness_label"] = label
    card["failure_reason"] = failure_reason or None
    card["suggested_correction"] = suggested_correction or None
    card["labeled_at"] = _now()
    state.setdefault("cards", {})[cid] = card
    state.setdefault("labels", {})[cid] = label
    return state


def unlabeled(cards: list[dict]) -> list[dict]:
    return [c for c in cards if not c.get("usefulness_label")]


def format_card(card: dict, *, index: int | None = None, total: int | None = None) -> str:
    head = f"[{index}/{total}] " if index and total else ""
    lines = [
        f"{head}id={card.get('task_id')}  mode={card.get('task_class')}  status={card.get('answer_status')}",
        f"Q: {card.get('query')}",
        f"doc: {card.get('document') or '—'}",
        f"span: {(card.get('evidence_span') or '—')[:160]}",
        f"solver: {' → '.join(str(s) for s in (card.get('solver_used') or []))}",
        f"latency: {card.get('latency_s')}s",
        f"verifier: {card.get('verifier_outcome')}",
    ]
    if card.get("contradiction_banner"):
        lines.append(f"contradiction: {card['contradiction_banner']}")
    if card.get("abstain_reason"):
        lines.append(f"abstain: {card['abstain_reason']}")
    if card.get("usefulness_label"):
        lines.append(f"label: {card['usefulness_label']}")
    lines.append(
        "labels: [u]seful [p]artial [n]ot [c]orrect-abstain [o]ver-abstain "
        "[w]rong-evidence [r]etrieval-miss [i]ngest [x]contradiction [s]kip [q]uit"
    )
    return "\n".join(lines)


def label_summary(state: dict) -> dict:
    labels = state.get("labels") or {}
    counts: dict[str, int] = {}
    for lab in labels.values():
        counts[lab] = counts.get(lab, 0) + 1
    return {
        "n_labeled": len(labels),
        "by_label": dict(sorted(counts.items())),
        "path": str(REVIEW_PATH),
    }


def interactive_review(cards: list[dict], state: dict, *, stdin=None, stdout=None) -> dict:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    queue = unlabeled(cards)
    total = len(queue)
    for i, card in enumerate(queue, 1):
        stdout.write(format_card(card, index=i, total=total) + "\n> ")
        stdout.flush()
        line = stdin.readline()
        if not line:
            break
        token = line.strip().lower().split()
        if not token:
            continue
        key = token[0]
        if key in ("q", "quit"):
            break
        if key in ("s", "skip"):
            continue
        label = LABEL_SHORT.get(key, key.upper())
        if label == "SKIP":
            continue
        if label == "QUIT":
            break
        reason = ""
        correction = ""
        if len(token) > 1:
            reason = " ".join(token[1:])
        try:
            apply_label(state, card, label, failure_reason=reason, suggested_correction=correction)
        except ValueError as e:
            stdout.write(f"{e}\n")
            continue
        stdout.write(f"labeled {card.get('task_id')} → {label}\n")
    save_state(state)
    return state


def batch_label(state: dict, cards: list[dict], specs: list[str]) -> dict:
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
        apply_label(state, card, lab)
    save_state(state)
    return state
