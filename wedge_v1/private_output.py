"""Contain local/private Wedge v1 evaluation artifacts outside commit-visible paths."""
from __future__ import annotations

import os
from pathlib import Path


WEDGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = WEDGE_ROOT.parent
PRIVATE_EXPORT_ROOT = WEDGE_ROOT / ".private"
PRIVATE_ROOTS = (
    PRIVATE_EXPORT_ROOT,
    WEDGE_ROOT / ".studies",
)
PRIVATE_TASK_ROOT = WEDGE_ROOT / "data" / "owner_tasks"
PRIVATE_CORPUS_ROOT = WEDGE_ROOT / "data" / "owner_corpus"
PRIVATE_ROOT_FILES = {
    "results_review_state.json",
    "results_saved_questions.json",
    "results_habit_session.json",
}


def _resolved(path: Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(_resolved(root))
    except ValueError:
        return False
    return True


def _lexical_absolute(path: Path) -> Path:
    """Make a path absolute without following its final or parent symlinks."""
    return Path(os.path.abspath(Path(path).expanduser()))


def _has_symlink_from(root: Path, path: Path) -> bool:
    """Return whether an existing component from ``root`` through ``path`` is a link."""
    root_abs = _lexical_absolute(root)
    path_abs = _lexical_absolute(path)
    try:
        relative = path_abs.relative_to(root_abs)
    except ValueError:
        return False
    current = root_abs
    if current.is_symlink():
        return True
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def private_output_allowed(path: Path) -> bool:
    """Allow outside-repo paths and the repository's ignored private namespaces."""
    resolved = _resolved(path)
    if not _within(resolved, REPO_ROOT):
        return True
    if any(_within(resolved, root) for root in PRIVATE_ROOTS):
        return True
    if resolved.parent == _resolved(WEDGE_ROOT):
        return (
            (
                resolved.name.startswith("results_owner_")
                and resolved.suffix in {".json", ".md"}
            )
            or resolved.name in PRIVATE_ROOT_FILES
        )
    return False


def require_private_output(path: Path, *, purpose: str = "private output") -> Path:
    """Return ``path`` or fail before private material reaches a tracked location."""
    target = Path(path)
    if not private_output_allowed(target):
        raise ValueError(
            f"{purpose} must be outside the repository or inside an ignored private "
            f"namespace (for example {WEDGE_ROOT / '.private'})"
        )
    return target


def private_task_pack_allowed(path: Path) -> bool:
    """Allow JSON task packs only outside the repo or in the owner-task root.

    A path written lexically under the in-repository owner-task root may not use
    symlinked components. This keeps a redirected ignored directory from becoming
    a route back into a commit-visible location.
    """
    raw = _lexical_absolute(Path(path))
    if raw.suffix.lower() != ".json":
        return False
    owner_root = _lexical_absolute(PRIVATE_TASK_ROOT)
    try:
        raw.relative_to(owner_root)
        raw_in_owner_root = True
    except ValueError:
        raw_in_owner_root = False
    if raw_in_owner_root and _has_symlink_from(owner_root, raw.parent):
        return False
    if raw.is_symlink():
        return False

    resolved = raw.parent.resolve(strict=False) / raw.name
    if not _within(resolved, _resolved(REPO_ROOT)):
        return True
    return _within(resolved, _resolved(PRIVATE_TASK_ROOT))


def require_private_task_pack(path: Path) -> Path:
    """Return a canonical task-pack path or fail without echoing the input path."""
    raw = _lexical_absolute(Path(path))
    if not private_task_pack_allowed(raw):
        raise ValueError(
            "task pack must be a .json file outside the repository or inside "
            "the ignored owner-task namespace"
        )
    return raw.parent.resolve(strict=False) / raw.name


def private_corpus_allowed(path: Path) -> bool:
    """Allow corpus directories only outside the repo or in the owner corpus root."""
    raw = _lexical_absolute(Path(path))
    if raw.is_symlink():
        return False
    owner_root = _lexical_absolute(PRIVATE_CORPUS_ROOT)
    try:
        raw.relative_to(owner_root)
        raw_in_owner_root = True
    except ValueError:
        raw_in_owner_root = False
    if raw_in_owner_root and _has_symlink_from(owner_root, raw):
        return False

    resolved = raw.resolve(strict=False)
    if not _within(resolved, _resolved(REPO_ROOT)):
        return True
    return _within(resolved, _resolved(PRIVATE_CORPUS_ROOT))


def require_private_corpus(path: Path) -> Path:
    """Return a canonical private corpus path without echoing it on failure."""
    raw = _lexical_absolute(Path(path))
    if not private_corpus_allowed(raw):
        raise ValueError(
            "corpus must be outside the repository or inside the ignored owner-corpus "
            "namespace"
        )
    return raw.resolve(strict=False)


def private_study_inventory_allowed(path: Path) -> bool:
    """Allow JSON inventory reports only outside the repo or in ``.private``.

    The repository-local namespace is checked lexically as well as after parent
    resolution so a symlink cannot redirect a private report into tracked state.
    """
    raw = _lexical_absolute(Path(path))
    if raw.suffix.lower() != ".json" or raw.is_symlink():
        return False

    # Inventory reports contain private identifiers.  Reject every existing
    # symlink component, including for an otherwise valid outside-repo path,
    # rather than silently canonicalizing through an alias.
    anchor = Path(raw.anchor)
    if _has_symlink_from(anchor, raw.parent):
        return False

    export_root = _lexical_absolute(PRIVATE_EXPORT_ROOT)
    try:
        raw.relative_to(export_root)
        raw_in_export_root = True
    except ValueError:
        raw_in_export_root = False
    if raw_in_export_root and _has_symlink_from(export_root, raw.parent):
        return False

    resolved = raw.parent.resolve(strict=False) / raw.name
    if not _within(resolved, _resolved(REPO_ROOT)):
        return True
    return _within(resolved, _resolved(PRIVATE_EXPORT_ROOT))


def require_private_study_inventory(path: Path) -> Path:
    """Return a canonical inventory-report path or fail without exposing it."""
    raw = _lexical_absolute(Path(path))
    if not private_study_inventory_allowed(raw):
        raise ValueError(
            "study inventory must be a .json file outside the repository or inside "
            "the ignored private-report namespace"
        )
    return raw.parent.resolve(strict=False) / raw.name


def private_query_input_allowed(path: Path) -> bool:
    """Allow query files only from explicitly private locations."""
    raw = _lexical_absolute(Path(path))
    if raw.is_symlink():
        resolved = raw.resolve(strict=False)
    else:
        resolved = raw.parent.resolve(strict=False) / raw.name
    if not _within(resolved, _resolved(REPO_ROOT)):
        return True
    roots = (*PRIVATE_ROOTS, PRIVATE_TASK_ROOT, PRIVATE_CORPUS_ROOT)
    return any(_within(resolved, _resolved(root)) for root in roots)


def require_private_query_input(path: Path) -> Path:
    """Return a canonical private query-input path without exposing it on error."""
    raw = _lexical_absolute(Path(path))
    if not private_query_input_allowed(raw):
        raise ValueError("query input must be outside the repository or in a private namespace")
    return raw.resolve(strict=False)
