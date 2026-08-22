"""Contain local/private Wedge v1 evaluation artifacts outside commit-visible paths."""
from __future__ import annotations

import os
from pathlib import Path

WEDGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = WEDGE_ROOT.parent

PRIVATE_EXPORT_ROOT = WEDGE_ROOT / ".private"
PRIVATE_TASK_ROOT = WEDGE_ROOT / "data" / "owner_tasks"
PRIVATE_CORPUS_ROOT = WEDGE_ROOT / "data" / "owner_corpus"
PRIVATE_STUDY_ROOT = WEDGE_ROOT / ".studies"

PRIVATE_ROOTS = (
    PRIVATE_EXPORT_ROOT,
    PRIVATE_STUDY_ROOT,
    PRIVATE_TASK_ROOT,
    PRIVATE_CORPUS_ROOT,
)

# Gitignored owner result files historically lived under wedge_v1/ root.
PRIVATE_ROOT_FILES = frozenset(
    {
        "results_owner_dogfood.json",
        "results_owner_smoke.json",
        "results_owner_failure_gallery.json",
        "results_owner_failure_gallery.md",
        "results_owner_dogfood.md",
        "results_owner_dogfood_gallery.md",
        "results_owner_ready.json",
        "results_review_state.json",
        "results_habit_session.json",
        "results_saved_questions.json",
        "results_owner_habit.json",
        "results_corpus_contact_owner.json",
        "results_u_owner_private.json",
    }
)


def _resolved(path: Path) -> Path:
    return Path(path).expanduser().resolve()


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _lexical_absolute(path: Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (Path.cwd() / p)


def _has_symlink_from(path: Path, root: Path) -> bool:
    """True if any component under root is a symlink (fail closed for private writes)."""
    try:
        cur = _resolved(path)
        root_r = root.resolve()
        if not _within(cur, root_r) and cur != root_r:
            return False
        # walk from root toward path
        rel = cur.relative_to(root_r) if cur != root_r else Path()
        probe = root_r
        if probe.is_symlink():
            return True
        for part in rel.parts:
            probe = probe / part
            if probe.is_symlink():
                return True
    except Exception:
        return True
    return False


def private_output_allowed(path: Path | str) -> bool:
    p = _resolved(Path(path))
    if p.name in PRIVATE_ROOT_FILES and p.parent == WEDGE_ROOT.resolve():
        return True
    for root in PRIVATE_ROOTS:
        if _within(p, root) or p == root.resolve():
            return True
    # env override for external private dirs
    extra = (os.environ.get("WEDGE_PRIVATE_OUTPUT_ROOT") or "").strip()
    if extra and (_within(p, Path(extra)) or p == Path(extra).resolve()):
        return True
    return False


def require_private_output(path: Path | str, *, purpose: str = "private output") -> Path:
    p = Path(path)
    if not private_output_allowed(p):
        raise ValueError(
            f"Refusing to write {purpose} outside private roots: {p}. "
            f"Use wedge_v1/.private, wedge_v1/.studies, or gitignored owner paths."
        )
    return _resolved(p)


def private_task_pack_allowed(path: Path | str) -> bool:
    p = _resolved(Path(path))
    return _within(p, PRIVATE_TASK_ROOT) or p.parent == PRIVATE_TASK_ROOT.resolve()


def require_private_task_pack(path: Path | str) -> Path:
    p = Path(path)
    if not private_task_pack_allowed(p):
        raise ValueError(f"Task pack must live under {PRIVATE_TASK_ROOT}: {p}")
    return _resolved(p)


def private_corpus_allowed(path: Path | str) -> bool:
    p = _resolved(Path(path))
    if _within(p, PRIVATE_CORPUS_ROOT) or p == PRIVATE_CORPUS_ROOT.resolve():
        return True
    env = (os.environ.get("OWNER_CORPUS") or os.environ.get("WEDGE_OWNER_CORPUS") or "").strip()
    if env:
        root = Path(env).expanduser().resolve()
        if _within(p, root) or p == root:
            return True
    return False


def require_private_corpus(path: Path | str) -> Path:
    p = Path(path)
    if not private_corpus_allowed(p):
        raise ValueError(f"Corpus path not allowed as private corpus: {p}")
    return _resolved(p)


def private_study_inventory_allowed(path: Path | str) -> bool:
    p = _resolved(Path(path))
    return _within(p, PRIVATE_STUDY_ROOT) or _within(p, PRIVATE_EXPORT_ROOT)


def require_private_study_inventory(path: Path | str) -> Path:
    p = Path(path)
    if not private_study_inventory_allowed(p):
        raise ValueError(f"Study inventory must be under .studies/.private: {p}")
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
        "studies": str(PRIVATE_STUDY_ROOT),
    }
