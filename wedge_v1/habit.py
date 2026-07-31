"""Local habit / session workflow (gitignored). Not Layer-1 evidence."""
from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from wedge_v1.ingest import corpus_stats, load_corpus
from wedge_v1.review import REVIEW_PATH, load_state as load_review, unlabeled
from wedge_v1.runtime import DEFAULT_CORPUS, ask, scan

ROOT = Path(__file__).resolve().parent
HABIT_PATH = ROOT / "results_owner_habit.json"
SAVED_QUESTIONS = ROOT / "results_saved_questions.json"
SESSION_PATH = ROOT / "results_habit_session.json"


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
                "doc_id": p.stem,
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


def save_question(query: str, *, mode: str = "ask", path: Path = SAVED_QUESTIONS) -> dict:
    data = {"schema": "nano-lm.wedge_v1.saved_questions.v1", "questions": load_saved_questions(path)}
    data["questions"] = [
        q for q in data["questions"] if not (q.get("query") == query and q.get("mode") == mode)
    ]
    data["questions"].insert(0, {"query": query, "mode": mode, "saved_at": _now().isoformat()})
    data["questions"] = data["questions"][:100]
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return data


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


def review_queue_summary() -> dict:
    state = load_review()
    cards = list((state.get("cards") or {}).values())
    pending = unlabeled(cards) if cards else []
    labels = state.get("labels") or {}
    return {
        "n_cards_cached": len(cards),
        "n_labeled": len(labels),
        "n_unlabeled_cached": len(pending),
        "review_path": str(REVIEW_PATH),
        "next": (
            "python -m wedge_v1 review --demo --next"
            if pending or not labels
            else "python -m wedge_v1 review --summary"
        ),
    }


def rerun_saved(corpus: Path, *, limit: int = 5) -> list[dict]:
    rows = []
    for q in load_saved_questions()[:limit]:
        mode = q.get("mode") or "ask"
        if mode != "ask":
            continue
        r = ask(q["query"], corpus_dir=corpus)
        rows.append(
            {
                "query": q["query"],
                "answer_status": r.get("answer_status"),
                "n_claims": len(r.get("claims") or []),
                "latency_s": r.get("latency_s"),
                "contradiction_banner": r.get("contradiction_banner"),
            }
        )
    return rows


def session(corpus: Path | None = None) -> dict:
    """Compact daily/session habit surface with a single next action."""
    corpus_path = resolve_session_corpus(corpus)
    stats = corpus_stats(corpus_path) if corpus_path.exists() else {"n_docs": 0, "error": "missing"}
    docs = recent_documents(corpus_path)
    disputed = disputed_from_scan(corpus_path) if stats.get("n_docs") else []
    week = weekly_summary()
    rq = review_queue_summary()
    saved = load_saved_questions()
    rerun = rerun_saved(corpus_path, limit=3) if saved else []

    next_action = "python -m wedge_v1 review --demo --interactive"
    if not stats.get("n_docs"):
        next_action = (
            "Set OWNER_CORPUS or run: python -m wedge_v1 owner-ready --demo"
        )
    elif rq["n_labeled"] == 0:
        next_action = f"python -m wedge_v1 review --corpus {corpus_path} --interactive"
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
        "rerun_sample": rerun,
        "weekly": week,
        "review_queue": rq,
        "next_action": next_action,
        "pending_owner_corpus": not bool(
            (os.environ.get("OWNER_CORPUS") or "").strip()
            or ((ROOT / "data" / "owner_corpus").is_dir()
                and any((ROOT / "data" / "owner_corpus").rglob("*.md")))
        ),
        "note": "Local habit session; fixture≠owner usefulness. Not Layer-1.",
    }
    SESSION_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    record("habit_session", note=f"docs={stats.get('n_docs')}")
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
    if sess.get("pending_owner_corpus"):
        lines.append(
            "_OWNER_CORPUS not set — using fixture/public path. "
            "Private usefulness validation remains pending._"
        )
    lines.append("")
    return "\n".join(lines)
