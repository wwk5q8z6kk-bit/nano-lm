"""Re-exec into the project virtualenv before importing heavy deps.

The run command is fixed per experiment node and identical across all eight
cells, so the interpreter choice has to live in committed code rather than in
the command string — otherwise changing it means recreating every node.

The venv is deliberately resolved OUTSIDE the repository worktree. Session
worktrees get destroyed and reassigned, which already cost one venv; a venv
inside the tree is not durable state.

Resolution order, first hit wins:
  1. $NANO_VENV                    — explicit override
  2. ~/.cache/openresearch/venvs/nano-lm-<worktree-name>
  3. ~/.cache/openresearch/venvs/nano-lm
  4. <repo>/.venv                  — legacy in-tree location
If none resolves, the current interpreter is used unchanged and the report
records that, rather than failing: a system interpreter that already satisfies
requirements.txt is a legitimate venue, it just has to be declared.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REEXEC_GUARD = "NANO_VENV_REEXEC"


def _candidates(repo_root: Path) -> list[Path]:
    cache = Path.home() / ".cache" / "openresearch" / "venvs"
    found: list[Path] = []
    override = os.environ.get("NANO_VENV")
    if override:
        found.append(Path(override))
    found.append(cache / f"nano-lm-{repo_root.name}")
    found.extend(sorted(cache.glob("nano-lm-*")))
    found.append(cache / "nano-lm")
    found.append(repo_root / ".venv")
    return found


def resolve_venv(repo_root: Path) -> Path | None:
    for candidate in _candidates(repo_root):
        if (candidate / "bin" / "python").is_file():
            return candidate
    return None


def ensure_venv(repo_root: Path) -> None:
    """Re-exec this process under the project venv, once.

    Compares sys.prefix, NOT the resolved executable path. A venv's
    bin/python is a symlink to the base interpreter, so
    Path(sys.executable).resolve() equals the base interpreter even when
    running inside the venv — an earlier version of this check compared
    resolved paths, found them equal, and silently never re-execed while
    reporting that it had.
    """
    if os.environ.get(_REEXEC_GUARD):
        return
    venv = resolve_venv(repo_root)
    if venv is None:
        return
    if Path(sys.prefix) == venv:
        return
    python = venv / "bin" / "python"
    os.environ[_REEXEC_GUARD] = "1"
    os.execv(str(python), [str(python), *sys.argv])


def interpreter_provenance(repo_root: Path) -> dict[str, object]:
    """What actually ran, recorded in every report."""
    venv = resolve_venv(repo_root)
    in_venv = venv is not None and Path(sys.prefix) == venv
    return {
        "executable": sys.executable,
        "prefix": sys.prefix,
        "resolved_venv": str(venv) if venv else None,
        "running_in_venv": in_venv,
        "reexeced": bool(os.environ.get(_REEXEC_GUARD)),
    }
