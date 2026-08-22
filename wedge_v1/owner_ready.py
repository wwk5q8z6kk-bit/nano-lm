"""Owner-corpus readiness check — minimal private-folder onboarding.

Does not read or print private document bodies into tracked files.
Not Layer-1 evidence.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from wedge_v1.ingest import corpus_stats, load_corpus
from wedge_v1.review import REVIEW_PATH
from wedge_v1.run_owner_dogfood import DEFAULT_OUT, DEFAULT_TASKS, FIXTURE_CORPUS, OWNER_DIR

ROOT = Path(__file__).resolve().parent
READY_OUT = ROOT / "results_owner_ready.json"


def check(corpus: Path | None = None, *, demo: bool = False) -> dict:
    env = (os.environ.get("OWNER_CORPUS") or os.environ.get("WEDGE_OWNER_CORPUS") or "").strip()
    if demo:
        path = FIXTURE_CORPUS
        source = "demo_fixture"
    elif corpus is not None:
        path = Path(corpus).expanduser()
        source = "cli"
    elif env:
        path = Path(env).expanduser()
        source = "env_OWNER_CORPUS"
    elif OWNER_DIR.is_dir():
        path = OWNER_DIR
        source = "gitignored_owner_dir"
    else:
        path = FIXTURE_CORPUS
        source = "fallback_fixture"

    exists = path.is_dir()
    docs = load_corpus(path) if exists else {}
    stats = corpus_stats(path) if exists else {}
    suffixes = {"md": 0, "txt": 0, "pdf": 0, "other": 0}
    if exists:
        for p in path.rglob("*"):
            if not p.is_file() or any(x.startswith(".") for x in p.relative_to(path).parts):
                continue
            suf = p.suffix.lower()
            if suf in {".md", ".markdown"}:
                suffixes["md"] += 1
            elif suf == ".txt":
                suffixes["txt"] += 1
            elif suf == ".pdf":
                suffixes["pdf"] += 1
            else:
                suffixes["other"] += 1

    doc_ids = sorted(docs.keys())
    stable_ids = len(doc_ids) == len(set(doc_ids))
    review_exists = REVIEW_PATH.is_file()
    out_gitignored = True  # by policy; results_*owner* ignored

    blockers = []
    if not exists:
        blockers.append("corpus_path_missing")
    elif not docs:
        blockers.append("corpus_empty_or_unreadable")
    # Real private usefulness remains pending until OWNER_CORPUS env points
    # at a non-fixture folder. Demo/fixture green ≠ owner usefulness.
    fixture_paths = {FIXTURE_CORPUS.resolve()}
    if (ROOT / "fixtures" / "owner_corpus").exists():
        fixture_paths.add((ROOT / "fixtures" / "owner_corpus").resolve())
    if (ROOT / "data" / "corpus").exists():
        fixture_paths.add((ROOT / "data" / "corpus").resolve())
    real_owner = bool(env) and path.resolve() not in fixture_paths
    if demo or not real_owner:
        blockers.append("OWNER_CORPUS_PENDING")

    canonical = (
        'python -m wedge_v1 study check --corpus "$OWNER_CORPUS" '
        '--tasks wedge_v1/data/owner_tasks/questions-v1.json '
        '--dir wedge_v1/.studies/first-use && '
        'python -m wedge_v1 study run --corpus "$OWNER_CORPUS" '
        '--tasks wedge_v1/data/owner_tasks/questions-v1.json '
        '--dir wedge_v1/.studies/first-use'
    )
    demo_cmd = "python -m wedge_v1 study check --corpus wedge_v1/fixtures/owner_corpus --tasks wedge_v1/data/owner_tasks/questions-v1.json --dir wedge_v1/.studies/demo && python -m wedge_v1 study run --corpus wedge_v1/fixtures/owner_corpus --tasks wedge_v1/data/owner_tasks/questions-v1.json --dir wedge_v1/.studies/demo"

    report = {
        "schema": "nano-lm.wedge_v1.owner_ready.v1",
        "corpus": str(path.resolve()) if exists else str(path),
        "source": source,
        "exists": exists,
        "n_docs": len(docs),
        "n_chars": stats.get("n_chars"),
        "suffix_counts": suffixes,
        "pypdf_available": stats.get("pypdf_available"),
        "stable_doc_ids": stable_ids,
        "doc_ids_sample": doc_ids[:12],
        "local_only_outputs": {
            "owner_dogfood": str(DEFAULT_OUT),
            "review_state": str(REVIEW_PATH),
            "gitignored_policy": out_gitignored,
        },
        "review_state_present": review_exists,
        "tasks_pack": str(DEFAULT_TASKS),
        "blockers": blockers,
        "canonical_command": canonical,
        "demo_command": demo_cmd,
        "ready_for_private_run": bool(real_owner and exists and docs and "corpus_empty_or_unreadable" not in blockers and "corpus_path_missing" not in blockers),
        "note": (
            "Private usefulness validation PENDING until OWNER_CORPUS is set "
            "and owner labels reviews. Fixture green ≠ owner usefulness."
        ),
    }
    READY_OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Owner-corpus readiness (no PHI dump)")
    ap.add_argument("--corpus", type=Path, default=None)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args(argv)
    rep = check(args.corpus, demo=args.demo)
    print(json.dumps(rep, indent=2))
    print("WEDGE_V1_OWNER_READY")
    if "corpus_path_missing" in rep["blockers"] or "corpus_empty_or_unreadable" in rep["blockers"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
