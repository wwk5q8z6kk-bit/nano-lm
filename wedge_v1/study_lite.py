"""Lite usefulness-study check/run for Active Frontier (no Evidence Core).

Dependency-light readiness + run path. Full study lifecycle may land later;
this unblocks owner contact without broken imports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from wedge_v1.ingest import corpus_stats, load_corpus
from wedge_v1.run_owner_dogfood import score_task
from wedge_v1.runtime import DEFAULT_CORPUS

ROOT = Path(__file__).resolve().parent
PRIVATE_STUDY_ROOT = ROOT / ".studies"
DEFAULT_TASKS = ROOT / "data" / "owner_tasks" / "questions-v1.json"
FIXTURE = ROOT / "fixtures" / "owner_corpus"
OWNER_DIR = ROOT / "data" / "owner_corpus"

MIN_DOCS = 5
MIN_TASKS_SMOKE = 5
MIN_TASKS_REP = 10
MAX_TASKS = 20
ALLOWED_MODES = {"ask", "find", "compare", "recall"}
CHECK_SCHEMA = "nano-lm.wedge_v1.study_check.v1"
SUMMARY_SCHEMA = "nano-lm.wedge_v1.study_summary.v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + chr(10), encoding="utf-8")


def resolve_corpus(corpus: Path | None) -> Path:
    if corpus is not None:
        return Path(corpus).expanduser()
    env = (os.environ.get("OWNER_CORPUS") or os.environ.get("WEDGE_OWNER_CORPUS") or "").strip()
    if env:
        return Path(env).expanduser()
    if OWNER_DIR.is_dir() and any(OWNER_DIR.glob("*.md")):
        return OWNER_DIR
    if FIXTURE.is_dir():
        return FIXTURE
    return DEFAULT_CORPUS


def inventory(corpus: Path) -> dict:
    docs = load_corpus(corpus) if corpus.is_dir() else {}
    stats = corpus_stats(corpus) if corpus.is_dir() else {}
    format_counts: Counter[str] = Counter()
    unsupported = 0
    empty = 0
    if corpus.is_dir():
        for p in corpus.rglob("*"):
            if not p.is_file() or any(part.startswith(".") for part in p.relative_to(corpus).parts):
                continue
            suf = p.suffix.lower()
            if suf in {".md", ".markdown"}:
                format_counts["markdown"] += 1
            elif suf == ".txt":
                format_counts["txt"] += 1
            elif suf == ".pdf":
                format_counts["pdf"] += 1
            else:
                format_counts["other"] += 1
                unsupported += 1
            if suf in {".md", ".markdown", ".txt", ".pdf"} and p.stem not in docs:
                empty += 1
    return {
        "docs": docs,
        "stats": stats,
        "format_counts": dict(format_counts),
        "n_unsupported_files": unsupported,
        "n_unreadable_or_empty_files": empty,
        "doc_ids": sorted(docs),
    }


def load_tasks(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check(*, corpus: Path | None, tasks_path: Path, study_dir: Path) -> dict:
    path = resolve_corpus(corpus)
    inv = inventory(path)
    docs = inv["docs"]
    pack = load_tasks(tasks_path) if tasks_path.is_file() else {"tasks": []}
    tasks = pack.get("tasks") or []
    mode_counts = Counter((t.get("mode") or "ask") for t in tasks)
    scoped = [t for t in tasks if t.get("doc_ids")]
    unknown = 0
    for t in scoped:
        for did in t.get("doc_ids") or []:
            if docs and did not in docs:
                if not any(did.lower() in d.lower() or d.lower() in did.lower() for d in docs):
                    unknown += 1
    blockers: list[str] = []
    warnings: list[str] = []
    fixture_paths = {FIXTURE.resolve(), DEFAULT_CORPUS.resolve()}
    ex = ROOT / "data" / "owner_corpus.example"
    if ex.exists():
        fixture_paths.add(ex.resolve())
    private_loc = path.resolve() not in fixture_paths and "papers" not in str(path.resolve())
    if not path.is_dir():
        blockers.append("CORPUS_PATH_MISSING")
    if len(docs) < MIN_DOCS:
        blockers.append("TOO_FEW_DOCUMENTS")
    if len(tasks) < MIN_TASKS_SMOKE:
        blockers.append("TOO_FEW_TASKS")
    if inv["n_unsupported_files"]:
        warnings.append("UNSUPPORTED_CORPUS_FILES")
    if inv["n_unreadable_or_empty_files"]:
        warnings.append("UNREADABLE_OR_EMPTY_DOCUMENTS")
    if not private_loc:
        warnings.append("CORPUS_NOT_PRIVATE_LOCATION")
    if not tasks_path.is_file():
        blockers.append("TASK_PACK_MISSING")
    if any((t.get("mode") or "ask") not in ALLOWED_MODES for t in tasks):
        blockers.append("INVALID_TASK_MODE")

    smoke_ready = (
        "CORPUS_PATH_MISSING" not in blockers
        and "TOO_FEW_DOCUMENTS" not in blockers
        and "TASK_PACK_MISSING" not in blockers
        and "TOO_FEW_TASKS" not in blockers
    )
    representative_ready = (
        smoke_ready and private_loc and len(docs) >= 10 and len(tasks) >= MIN_TASKS_REP and unknown == 0
    )
    if smoke_ready and not private_loc:
        warnings.append("REPRESENTATIVE_REQUIRES_PRIVATE_CORPUS")

    task_blob = json.dumps(tasks, sort_keys=True)
    corpus_digest = _sha(chr(10).join(f"{k}:{len(v)}" for k, v in sorted(docs.items())))
    out = {
        "schema": CHECK_SCHEMA,
        "created_at": _now(),
        "smoke_ready": smoke_ready,
        "representative_ready": representative_ready,
        "study_id": _sha(str(path.resolve()) + task_blob)[:64],
        "corpus": {
            "path": str(path.resolve()),
            "n_documents": len(docs),
            "extracted_text_bytes": sum(len(v) for v in docs.values()),
            "format_counts": inv["format_counts"],
            "n_unsupported_files": inv["n_unsupported_files"],
            "n_unreadable_or_empty_files": inv["n_unreadable_or_empty_files"],
            "pypdf_available": inv["stats"].get("pypdf_available"),
            "pdf_extractor": "AVAILABLE" if inv["stats"].get("pypdf_available") else "UNAVAILABLE",
        },
        "tasks": {
            "path": str(tasks_path),
            "n_tasks": len(tasks),
            "n_exactly_scoped": len(scoped),
            "n_unique_scoped_documents": len({d for t in scoped for d in (t.get("doc_ids") or [])}),
            "n_unknown_scope_references": unknown,
            "mode_counts": dict(mode_counts),
            "authenticity": "DECLARED_BY_OPERATOR_NOT_INFERRED",
        },
        "identity": {
            "corpus_digest": corpus_digest,
            "task_pack_digest": _sha(task_blob),
            "instrument_fingerprint": _sha("study_lite.v1"),
        },
        "limits": {"documents": [MIN_DOCS, 50], "tasks": [MIN_TASKS_SMOKE, MAX_TASKS]},
        "blockers": blockers,
        "warnings": warnings,
        "claim_boundary": (
            "Readiness permits a local usefulness study only; it is not evidence of "
            "scientific validity or product superiority."
        ),
        "canonical_command": (
            f'python -m wedge_v1 study run --corpus "$OWNER_CORPUS" --tasks {tasks_path} --dir {study_dir}'
        ),
        "demo_command": (
            f"python -m wedge_v1 study check --corpus {FIXTURE} --tasks {DEFAULT_TASKS} "
            f"--dir {PRIVATE_STUDY_ROOT / 'demo'} && "
            f"python -m wedge_v1 study run --corpus {FIXTURE} --tasks {DEFAULT_TASKS} "
            f"--dir {PRIVATE_STUDY_ROOT / 'demo'}"
        ),
    }
    _write_json(study_dir / "check.json", out)
    return out


def run(*, corpus: Path | None, tasks_path: Path, study_dir: Path) -> dict:
    chk = check(corpus=corpus, tasks_path=tasks_path, study_dir=study_dir)
    path = Path(chk["corpus"]["path"])
    if not chk["smoke_ready"]:
        return {
            "schema": SUMMARY_SCHEMA,
            "ok": False,
            "error": "STUDY_NOT_SMOKE_READY",
            "check": chk,
            "written": str(study_dir / "check.json"),
        }
    pack = load_tasks(tasks_path)
    rows = []
    for t in pack["tasks"]:
        task = dict(t)
        if task.get("mode") == "recall":
            task = {**task, "mode": "ask"}
        rows.append(score_task(task, path))
    n_ok = sum(1 for r in rows if r["ok"])
    summary = {
        "schema": SUMMARY_SCHEMA,
        "created_at": _now(),
        "ok": n_ok == len(rows),
        "accuracy": n_ok / max(1, len(rows)),
        "n_ok": n_ok,
        "n_tasks": len(rows),
        "corpus": str(path),
        "corpus_class": (
            "OWNER_PRIVATE"
            if "CORPUS_NOT_PRIVATE_LOCATION" not in chk["warnings"]
            else "FIXTURE_OR_PUBLIC"
        ),
        "rows": rows,
        "check": {
            "smoke_ready": chk["smoke_ready"],
            "representative_ready": chk["representative_ready"],
            "blockers": chk["blockers"],
            "warnings": chk["warnings"],
            "study_id": chk["study_id"],
        },
        "claim_boundary": chk["claim_boundary"],
        "note": "Lite study run; gitignored study dir; not Layer-1 evidence.",
    }
    _write_json(study_dir / "summary.json", summary)
    _write_json(
        study_dir / "results.json",
        {"schema": "nano-lm.wedge_v1.study_results.v1", "rows": rows},
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="wedge_v1 study", description="Lite usefulness study check/run")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--corpus", type=Path, default=None)
        sp.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
        sp.add_argument("--dir", type=Path, default=PRIVATE_STUDY_ROOT / "first-use")

    c = sub.add_parser("check", help="Readiness check (writes check.json)")
    add_common(c)
    r = sub.add_parser("run", help="Run task pack when smoke-ready")
    add_common(r)

    args = ap.parse_args(argv)
    if args.cmd == "check":
        out = check(corpus=args.corpus, tasks_path=args.tasks, study_dir=args.dir)
        print(json.dumps(out, indent=2))
        print("WEDGE_V1_STUDY_CHECK")
        return 0 if out["smoke_ready"] else 1
    if args.cmd == "run":
        out = run(corpus=args.corpus, tasks_path=args.tasks, study_dir=args.dir)
        slim = {k: out[k] for k in out if k != "rows"}
        print(json.dumps(slim, indent=2))
        print(
            json.dumps(
                {
                    "rows": [
                        {"id": x["id"], "ok": x["ok"], "got": x["got_status"]}
                        for x in out.get("rows", [])
                    ]
                },
                indent=2,
            )
        )
        print("WEDGE_V1_STUDY_RUN")
        return 0 if out.get("ok") else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
