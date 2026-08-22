#!/usr/bin/env python3
"""Docs integrity checks for canonical documentation reset PRs."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ACTIVE_JSON = DOCS / "ACTIVE_NOW.json"

PROTECTED_EVIDENCE_PREFIXES = (
    "papers/EVIDENCE_LEDGER",
    "papers/EMPIRICAL_FOUNDATION.md",
    "EVIDENCE_CURRENT.md",
    "POST_ALPHA_EVIDENCE_FREEZE",
)

STUB_TO_ARCHIVE = {
    "papers/STRATEGIC_RESET.md": "docs/archive/legacy/STRATEGIC_RESET_20260731.md",
    "papers/AMBITION.md": "docs/archive/legacy/AMBITION_20260731.md",
    "papers/WEDGE_V1.md": "docs/archive/legacy/WEDGE_V1_20260731.md",
    "papers/EXECUTION_QUEUE.md": "docs/archive/legacy/EXECUTION_QUEUE_20260731.md",
    "papers/AZ_EXECUTION_PLAN.md": "docs/archive/legacy/AZ_EXECUTION_PLAN_POST_E1_20260731.md",
    "papers/PROGRAM_AUTHORITY.md": "docs/archive/legacy/PROGRAM_AUTHORITY_WEDGE_20260731.md",
}

# Paths that may be referenced only as cross-branch lineage on integration base.
CROSS_BRANCH_PATHS = frozenset(
    {
        "artifacts/nano_h6/",
        "nano_ai/",
    }
)

LINK_RE = re.compile(r"\]\(([^)]+)\)")


def _is_external(href: str) -> bool:
    return href.startswith(("http://", "https://", "mailto:", "#"))


def _resolve(from_file: Path, href: str) -> Path | None:
    href = href.split("#", 1)[0].strip()
    if not href or _is_external(href):
        return None
    if href.startswith("/"):
        target = ROOT / href.lstrip("/")
    else:
        target = (from_file.parent / href).resolve()
    return target


def check_links(errors: list[str]) -> None:
    for md in DOCS.rglob("*.md"):
        text = md.read_text(encoding="utf-8")
        for href in LINK_RE.findall(text):
            target = _resolve(md, href)
            if target is None:
                continue
            if not target.exists():
                errors.append(f"broken link in {md.relative_to(ROOT)}: {href}")


def check_stub_archives(errors: list[str]) -> None:
    for stub, archive in STUB_TO_ARCHIVE.items():
        ap = ROOT / archive
        if not ap.is_file():
            errors.append(f"missing archive for stub {stub}: {archive}")
            continue
        if ap.stat().st_size < 500:
            errors.append(f"archive too small (stub target): {archive}")
        stub_path = ROOT / stub
        if not stub_path.is_file():
            errors.append(f"missing stub: {stub}")
        elif "Superseded" not in stub_path.read_text(encoding="utf-8")[:400]:
            errors.append(f"stub missing Superseded banner: {stub}")


def check_cross_branch_markers(errors: list[str]) -> None:
    """Canonical docs must mark or avoid absent cross-branch paths."""
    for md in list(DOCS.rglob("*.md")) + [ROOT / "README.md"]:
        if not md.is_file():
            continue
        text = md.read_text(encoding="utf-8")
        for prefix in CROSS_BRANCH_PATHS:
            if prefix not in text:
                continue
            # Allow if explicitly marked cross-branch / not integrated
            if "cross-branch" in text.lower() or "not yet integrated" in text.lower():
                continue
            if "not yet integrated" in text or "CROSS_BRANCH" in text:
                continue
            # MODEL_RESEARCH and RUNPOD must contain integration table
            if md.name in {"MODEL_RESEARCH_PROGRAM.md", "RUNPOD.md"}:
                if "not yet integrated" in text:
                    continue
            errors.append(
                f"{md.relative_to(ROOT)} references {prefix} without cross-branch marker"
            )


def check_protected_diff(errors: list[str]) -> None:
    """In a docs-reset PR, protected evidence paths must not change vs merge base."""
    try:
        merge_base = subprocess.check_output(
            ["git", "merge-base", "HEAD", "origin/master"],
            cwd=ROOT,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return
    try:
        diff_names = subprocess.check_output(
            ["git", "diff", "--name-only", merge_base, "HEAD"],
            cwd=ROOT,
            text=True,
        ).splitlines()
    except subprocess.CalledProcessError:
        return
    for name in diff_names:
        for prefix in PROTECTED_EVIDENCE_PREFIXES:
            if name == prefix or name.startswith(prefix):
                errors.append(f"protected evidence path modified in PR: {name}")
                break


def check_single_authority_per_concern(errors: list[str]) -> None:
    """ACTIVE_NOW.json is sole canonical source for synced status fields."""
    if not ACTIVE_JSON.is_file():
        errors.append("missing docs/ACTIVE_NOW.json")
        return
    data = json.loads(ACTIVE_JSON.read_text(encoding="utf-8"))
    if data.get("schema") != "nano-lm.docs.active_now.v1":
        errors.append("ACTIVE_NOW.json schema mismatch")


def main() -> int:
    errors: list[str] = []
    check_links(errors)
    check_stub_archives(errors)
    check_cross_branch_markers(errors)
    check_protected_diff(errors)
    check_single_authority_per_concern(errors)
    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        return 1
    print("DOCS_INTEGRITY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
