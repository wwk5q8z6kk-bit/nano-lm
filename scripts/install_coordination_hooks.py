#!/usr/bin/env python3
"""Install universal coordination hooks & git settings into the shared git-common-dir.

Installs dispatcher shims into `<git-common-dir>/hooks` and fallback runtime copies
into `<git-common-dir>/agent-coord/`, then sets repo-local `core.hooksPath` to the
absolute `<git-common-dir>/hooks` path so ALL linked worktrees (even older branches
without `.githooks/` in their tree) enforce the guardrails without mutating any peer
worktree.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import stat
import subprocess

HOOK_NAMES = ("pre-commit", "pre-push", "post-checkout", "post-merge")


def _git(repo: Path, *args: str) -> str:
    res = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return res.stdout.strip()


def _make_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _dispatcher_script(hook_name: str) -> str:
    return (
        "#!/usr/bin/env bash\n"
        "top=$(git rev-parse --show-toplevel 2>/dev/null || pwd)\n"
        "common=$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null || echo \"$top/.git\")\n"
        f"if [ -x \"$top/.githooks/{hook_name}\" ]; then\n"
        f"  exec \"$top/.githooks/{hook_name}\" \"$@\"\n"
        f"elif [ -x \"$common/agent-coord/hooks/{hook_name}\" ]; then\n"
        f"  exec \"$common/agent-coord/hooks/{hook_name}\" \"$@\"\n"
        "fi\n"
        "exit 0\n"
    )


def install(repo: Path, dry_run: bool = False) -> dict[str, str]:
    top = Path(os.path.realpath(_git(repo, "rev-parse", "--show-toplevel")))
    common_raw = _git(top, "rev-parse", "--path-format=absolute", "--git-common-dir")
    common = Path(os.path.realpath(common_raw))

    coord_src = top / "scripts" / "agent_coord.py"
    hooks_src = top / ".githooks"
    coord_dst = common / "agent-coord"
    fallback_hooks_dst = coord_dst / "hooks"
    live_hooks_dir = common / "hooks"

    if not dry_run:
        coord_dst.mkdir(parents=True, exist_ok=True)
        fallback_hooks_dst.mkdir(parents=True, exist_ok=True)
        live_hooks_dir.mkdir(parents=True, exist_ok=True)

        if coord_src.is_file():
            dst_py = coord_dst / "agent_coord.py"
            shutil.copy2(coord_src, dst_py)
            _make_executable(dst_py)

        for name in HOOK_NAMES:
            src_hook = hooks_src / name
            if src_hook.is_file():
                fb = fallback_hooks_dst / name
                shutil.copy2(src_hook, fb)
                _make_executable(fb)
            disp = live_hooks_dir / name
            disp.write_text(_dispatcher_script(name), encoding="utf-8")
            _make_executable(disp)

        _git(top, "config", "--local", "core.hooksPath", str(live_hooks_dir))
        _git(top, "config", "--local", "rerere.enabled", "true")
        _git(top, "config", "--local", "merge.conflictStyle", "zdiff3")
        _git(top, "config", "--local", "fetch.prune", "true")

    return {
        "top": str(top),
        "common_dir": str(common),
        "hooks_path": str(live_hooks_dir),
        "coord_runtime": str(coord_dst / "agent_coord.py"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Install universal coordination hooks")
    ap.add_argument("--repo", default=".", help="Path inside git repository")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    info = install(Path(args.repo).resolve(), dry_run=args.dry_run)
    prefix = "[dry-run] " if args.dry_run else ""
    print(f"{prefix}installed hooksPath={info['hooks_path']} runtime={info['coord_runtime']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
