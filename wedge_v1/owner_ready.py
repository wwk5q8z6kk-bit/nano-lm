"""Owner-corpus readiness check — minimal private-folder onboarding.

Does not read or print private document bodies into tracked files.
Not Layer-1 evidence.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from wedge_v1.habit import load_saved_questions, saved_question_status
from wedge_v1.ingest import corpus_stats, load_corpus
from wedge_v1.review import REVIEW_PATH
from wedge_v1.run_owner_dogfood import DEFAULT_OUT, DEFAULT_TASKS, FIXTURE_CORPUS, OWNER_DIR

ROOT = Path(__file__).resolve().parent
READY_OUT = ROOT / "results_owner_ready.json"



def _corpus_cli_ref(path: Path, *, exists: bool, env_set: bool) -> str:
    if env_set:
        return '--corpus "$OWNER_CORPUS"'
    if exists:
        return f'--corpus "{path.resolve()}"'
    return '--corpus "$OWNER_CORPUS"'


def _habit_commands(path: Path, *, exists: bool, env_set: bool) -> dict[str, str]:
    corp = _corpus_cli_ref(path, exists=exists, env_set=env_set)
    return {
        "list": f"python -m wedge_v1 habit --list {corp}",
        "save": f'python -m wedge_v1 habit --save "YOUR QUESTION" {corp}',
        "save_scoped": f'python -m wedge_v1 habit --save "YOUR QUESTION" --doc DOC_ID {corp}',
        "rerun": f"python -m wedge_v1 habit --rerun {corp}",
        "session": f"python -m wedge_v1 habit {corp}",
    }


def _habit_memory(path: Path) -> dict:
    if not path.is_dir():
        return {"n_saved": 0, "states": {}, "questions": []}
    rows = saved_question_status(path)
    states: dict[str, int] = {}
    for row in rows:
        state = str(row.get("state") or "UNKNOWN")
        states[state] = states.get(state, 0) + 1
    return {
        "n_saved": len(rows),
        "states": states,
        "questions": [
            {
                "task_id": row.get("task_id"),
                "state": row.get("state"),
                "reason": row.get("reason"),
                "query": (row.get("query") or "")[:120],
            }
            for row in rows[:8]
        ],
    }


def _weekly_k1_command(*, env_set: bool) -> str:
    if env_set:
        prefix = ""
    else:
        prefix = 'export OWNER_CORPUS="$PWD/.local-data/owner_corpus" && '
    return (
        f"{prefix}"
        "python -m wedge_v1 owner-ready && "
        'python -m wedge_v1 habit --list --corpus "$OWNER_CORPUS" && '
        'python -m wedge_v1 habit --rerun --corpus "$OWNER_CORPUS" && '
        'python -m wedge_v1 review --corpus "$OWNER_CORPUS" --interactive'
    )


def format_owner_ready_hints(rep: dict) -> str:
    lines = [
        "",
        "--- owner-ready: weekly K1 habit ---",
        f"Corpus: {rep.get('corpus')}",
        f"Ready (private): {rep.get('ready_for_private_run')}",
        f"Note: {rep.get('note')}",
        "",
        "Habit memory commands:",
    ]
    for key in ("list", "save", "save_scoped", "rerun", "session"):
        cmd = (rep.get("habit_commands") or {}).get(key)
        if cmd:
            lines.append(f"  {cmd}")
    mem = rep.get("habit_memory") or {}
    if mem.get("n_saved"):
        lines.append("")
        lines.append(
            f"Saved questions: {mem['n_saved']} states={mem.get('states') or {}}"
        )
    elif load_saved_questions():
        lines.append("")
        lines.append("Saved questions exist but none match this corpus scope.")
    lines += [
        "",
        "One-liner:",
        f"  {rep.get('weekly_k1_command')}",
        "",
        "Demo:",
        f"  {rep.get('demo_command')}",
    ]
    return "\n".join(lines) + "\n"


def _ready_note(*, real_owner: bool, review_exists: bool) -> str:
    if not real_owner:
        return (
            "Private usefulness validation PENDING until OWNER_CORPUS is set "
            "and owner labels reviews. Fixture green ≠ owner usefulness."
        )
    from wedge_v1.review import load_state

    if review_exists and load_state().get("labels"):
        return "Owner contact active — continue usefulness loop (habit + label triage)."
    return "Private corpus ready — run owner-dogfood + review labels + habit."


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
    fixture_paths = {FIXTURE_CORPUS.resolve(), OWNER_DIR.resolve()}
    if (ROOT / "fixtures" / "owner_corpus").exists():
        fixture_paths.add((ROOT / "fixtures" / "owner_corpus").resolve())
    if (ROOT / "data" / "corpus").exists():
        fixture_paths.add((ROOT / "data" / "corpus").resolve())
    if (ROOT / "data" / "owner_corpus").exists():
        fixture_paths.add((ROOT / "data" / "owner_corpus").resolve())
    path_resolved = path.resolve() if exists else path
    real_owner = (bool(env) or source == "cli") and path_resolved not in fixture_paths
    if demo or not real_owner:
        blockers.append("OWNER_CORPUS_PENDING")

    canonical = (
        'export OWNER_CORPUS="$PWD/.local-data/owner_corpus" && '
        "python -m wedge_v1 owner-ready && "
        'python -m wedge_v1 habit --list --corpus "$OWNER_CORPUS" && '
        'python -m wedge_v1 habit --save "YOUR QUESTION" --corpus "$OWNER_CORPUS" && '
        'python -m wedge_v1 habit --rerun --corpus "$OWNER_CORPUS" && '
        'python -m wedge_v1 review --corpus "$OWNER_CORPUS" --interactive'
    )
    demo_cmd = (
        "python -m wedge_v1 owner-ready --demo && "
        "python -m wedge_v1 habit --list && "
        "python -m wedge_v1 owner-dogfood --demo && "
        "python -m wedge_v1 review --demo --next"
    )
    habit_commands = _habit_commands(path, exists=exists, env_set=bool(env))
    habit_memory = _habit_memory(path)
    weekly_k1 = _weekly_k1_command(env_set=bool(env))

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
        "weekly_k1_command": weekly_k1,
        "habit_commands": habit_commands,
        "habit_memory": habit_memory,
        "ready_for_private_run": bool(real_owner and exists and docs and "corpus_empty_or_unreadable" not in blockers and "corpus_path_missing" not in blockers),
        "note": _ready_note(real_owner=bool(real_owner and exists and docs), review_exists=review_exists),
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
    print(format_owner_ready_hints(rep), end="")
    print("WEDGE_V1_OWNER_READY")
    if "corpus_path_missing" in rep["blockers"] or "corpus_empty_or_unreadable" in rep["blockers"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
