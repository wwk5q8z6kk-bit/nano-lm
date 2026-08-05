"""Content-free readiness check for local Wedge v1 pipeline studies.

The report separates a basic corpus smoke check from the stricter inputs needed
for a representative usefulness study. It never prints document IDs or bodies.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from wedge_v1.run_owner_dogfood import DEFAULT_TASKS, FIXTURE_CORPUS, OWNER_DIR
from wedge_v1.study import PRIVATE_STUDY_ROOT, PRIVATE_TASK_ROOT, assess_inputs

ROOT = Path(__file__).resolve().parent
READY_OUT = ROOT / "results_owner_ready.json"
DEFAULT_PRIVATE_TASKS = PRIVATE_TASK_ROOT / "questions-v1.json"


def _resolve_corpus(corpus: Path | None, *, demo: bool) -> tuple[Path, str]:
    env = (os.environ.get("OWNER_CORPUS") or os.environ.get("WEDGE_OWNER_CORPUS") or "").strip()
    if demo:
        return FIXTURE_CORPUS, "demo_fixture"
    if corpus is not None:
        return Path(corpus).expanduser(), "cli"
    if env:
        return Path(env).expanduser(), "env_OWNER_CORPUS"
    if OWNER_DIR.is_dir():
        return OWNER_DIR, "gitignored_owner_dir"
    return FIXTURE_CORPUS, "fallback_fixture"


def check(
    corpus: Path | None = None,
    *,
    tasks: Path | None = None,
    demo: bool = False,
) -> dict:
    path, source = _resolve_corpus(corpus, demo=demo)
    task_path = DEFAULT_TASKS if demo else Path(tasks).expanduser() if tasks else DEFAULT_PRIVATE_TASKS
    study = assess_inputs(path, task_path, demo=demo)
    real_owner = source in {"cli", "env_OWNER_CORPUS", "gitignored_owner_dir"} and not demo
    blockers = list(study["blockers"])
    if not real_owner:
        blockers.append("OWNER_CORPUS_PENDING")
    blockers = sorted(set(blockers))

    smoke_ready = bool(study["smoke_ready"])
    representative_ready = bool(real_owner and study["representative_ready"])
    return {
        "schema": "nano-lm.wedge_v1.owner_ready.v2",
        "source": source,
        "corpus_exists": path.is_dir(),
        "smoke_ready": smoke_ready,
        "real_private_smoke_ready": bool(real_owner and smoke_ready),
        "representative_ready": representative_ready,
        "ready_for_private_run": representative_ready,
        "corpus": study["corpus"],
        "tasks": study["tasks"],
        "identity": study["identity"],
        "study_id": study["study_id"],
        "blockers": blockers,
        "warnings": study["warnings"],
        "canonical_command": (
            "python -m wedge_v1 study check --corpus \"$OWNER_CORPUS\" "
            "--tasks wedge_v1/data/owner_tasks/questions-v1.json "
            "--dir wedge_v1/.studies/first-use"
        ),
        "demo_command": "python -m wedge_v1 owner-dogfood --demo",
        "local_output_policy": {
            "task_root": str(PRIVATE_TASK_ROOT),
            "study_root": str(PRIVATE_STUDY_ROOT),
            "tracked_output": False,
        },
        "note": (
            "Smoke readiness proves readable local input only. Representative readiness "
            "permits a bounded usefulness study; neither establishes Nano AI "
            "capability nor superiority."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Owner-corpus readiness (content-free output)")
    ap.add_argument("--corpus", type=Path, default=None)
    ap.add_argument("--tasks", type=Path, default=None)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    report = check(args.corpus, tasks=args.tasks, demo=args.demo)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("WEDGE_V1_OWNER_READY")
    return 0 if (report["smoke_ready"] if args.demo else report["representative_ready"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
