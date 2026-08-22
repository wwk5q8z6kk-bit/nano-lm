"""Contain local/private Wedge v1 evaluation artifacts outside commit-visible paths."""
from __future__ import annotations

import os
from pathlib import Path

WEDGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = WEDGE_ROOT.parent

PRIVATE_EXPORT_ROOT = WEDGE_ROOT / ".private"
PRIVATE_TASK_ROOT = WEDGE_ROOT / "data" / "owner_tasks"
PRIVATE_CORPUS_ROOT = WEDGE_ROOT / "data" / "owner_corpus"

PRIVATE_ROOTS = (
    PRIVATE_EXPORT_ROOT,
    WEDGE_ROOT / ".studies",
)

PRIVATE_ROOT_FILES = frozenset(
    {
        "results_saved_questions.json",
        "results_review_state.json",
        "results_habit_session.json",
    }
)


def _resolved(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _outside_repo(path: Path) -> bool:
    return not _within(path, REPO_ROOT) and path.resolve() != REPO_ROOT.resolve()


def private_output_allowed(path: Path | str) -> bool:
    p = _resolved(path)
    if _outside_repo(p):
        return True
    if p.name in PRIVATE_ROOT_FILES and p.parent.resolve() == WEDGE_ROOT.resolve():
        return True
    for root in PRIVATE_ROOTS:
        if _within(p, root) or p == root.resolve():
            return True
    return False


def require_private_output(path: Path | str, *, purpose: str = "private output") -> Path:
    p = Path(path)
    if not private_output_allowed(p):
        raise ValueError(
            f"{purpose} must be outside the repository or inside an ignored private "
            f"namespace (for example {PRIVATE_EXPORT_ROOT})"
        )
    return _resolved(p)


def private_task_pack_allowed(path: Path | str) -> bool:
    p = _resolved(path)
    return _within(p, PRIVATE_TASK_ROOT) or p.parent.resolve() == PRIVATE_TASK_ROOT.resolve()


def require_private_task_pack(path: Path | str) -> Path:
    p = Path(path)
    if not private_task_pack_allowed(p):
        raise ValueError(f"Task pack must live under {PRIVATE_TASK_ROOT}: {p}")
    return _resolved(p)


def private_corpus_allowed(path: Path | str) -> bool:
    p = _resolved(path)
    if _within(p, PRIVATE_CORPUS_ROOT) or p == PRIVATE_CORPUS_ROOT.resolve():
        return True
    env = (os.environ.get("OWNER_CORPUS") or os.environ.get("WEDGE_OWNER_CORPUS") or "").strip()
    if env:
        root = Path(env).expanduser().resolve()
        if _within(p, root) or p == root:
            return True
    if _outside_repo(p):
        return True
    return False


def require_private_corpus(path: Path | str) -> Path:
    p = Path(path)
    if not private_corpus_allowed(p):
        raise ValueError(f"Corpus path not allowed as private corpus: {p}")
    return _resolved(p)


def private_study_inventory_allowed(path: Path | str) -> bool:
    p = _resolved(path)
    return _within(p, PRIVATE_EXPORT_ROOT) or p.parent.resolve() == PRIVATE_EXPORT_ROOT.resolve()


def require_private_study_inventory(path: Path | str) -> Path:
    p = Path(path)
    if not private_study_inventory_allowed(p):
        raise ValueError(f"Study inventory must be under {PRIVATE_EXPORT_ROOT}: {p}")
    return _resolved(p)


def private_query_input_allowed(path: Path | str) -> bool:
    return private_task_pack_allowed(path) or private_output_allowed(path)


def require_private_query_input(path: Path | str) -> Path:
    p = Path(path)
    if not private_query_input_allowed(p):
        raise ValueError(f"Query input path not allowed: {p}")
    return _resolved(p)


def private_paths() -> dict[str, str]:
    return {
        "export": str(PRIVATE_EXPORT_ROOT),
        "tasks": str(PRIVATE_TASK_ROOT),
        "corpus": str(PRIVATE_CORPUS_ROOT),
        "studies": str(WEDGE_ROOT / ".studies"),
    }
