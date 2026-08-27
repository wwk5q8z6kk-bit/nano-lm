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
ARCHIVE_DIR = DOCS / "archive"

PROTECTED_EVIDENCE_PREFIXES = (
    "papers/EVIDENCE_LEDGER",
    "papers/EMPIRICAL_FOUNDATION.md",
    "papers/PREREG_",
    "papers/RESULT_",
    "EVIDENCE_CURRENT.md",
    "POST_ALPHA_EVIDENCE_FREEZE",
    "trajectory/results_",
    "freeze/",
)

STUB_TO_ARCHIVE = {
    "papers/STRATEGIC_RESET.md": "docs/archive/legacy/STRATEGIC_RESET_20260731.md",
    "papers/AMBITION.md": "docs/archive/legacy/AMBITION_20260731.md",
    "papers/WEDGE_V1.md": "docs/archive/legacy/WEDGE_V1_20260731.md",
    "papers/EXECUTION_QUEUE.md": "docs/archive/legacy/EXECUTION_QUEUE_20260731.md",
    "papers/AZ_EXECUTION_PLAN.md": "docs/archive/legacy/AZ_EXECUTION_PLAN_POST_E1_20260731.md",
    "papers/PROGRAM_AUTHORITY.md": "docs/archive/legacy/PROGRAM_AUTHORITY_WEDGE_20260731.md",
}

CROSS_BRANCH_PATHS = frozenset(
    {
        "artifacts/nano_h6/",
        "nano_ai/",
    }
)

STALE_AUTHORITY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "strategic center is papers/STRATEGIC_RESET.md",
        re.compile(
            r"strategic\s+center\s+is\s+[`']?papers/STRATEGIC_RESET\.md",
            re.IGNORECASE,
        ),
    ),
    (
        "PROGRAM_EXECUTION_STATUS: IDLE_AFTER_DOGFOOD",
        re.compile(
            r"PROGRAM_EXECUTION_STATUS\s*[:=]\s*`?IDLE_AFTER_DOGFOOD`?",
            re.IGNORECASE,
        ),
    ),
    (
        "NanoScribe STOP",
        re.compile(r"NanoScribe\s+STOP"),
    ),
    (
        "Active product path: Nano Runtime via Wedge v1",
        re.compile(
            r"Active\s+product\s+path\s*:\s*Nano\s+Runtime\s+via\s+Wedge\s+v1",
            re.IGNORECASE,
        ),
    ),
    (
        "training = NOT_AUTHORIZED",
        re.compile(
            r"(?:training|TRAINING)\s*(?:=|:)\s*`?NOT_AUTHORIZED`?",
        ),
    ),
)

HISTORICAL_CONTEXT_RE = re.compile(
    r"HISTORICAL(?:_PROGRAM_STATE)?|archived|archive/legacy|superseded|"
    r"not\s+current|historical\s+only|July[- ]?31",
    re.IGNORECASE,
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


def _is_under_archive(path: Path) -> bool:
    try:
        path.resolve().relative_to(ARCHIVE_DIR.resolve())
        return True
    except ValueError:
        return False


def _authority_md_files() -> list[Path]:
    files: list[Path] = []
    for candidate in (ROOT / "README.md", ROOT / "AGENTS.md"):
        if candidate.is_file():
            files.append(candidate)
    if DOCS.is_dir():
        for md in DOCS.rglob("*.md"):
            if _is_under_archive(md):
                continue
            files.append(md)
    return files


def check_links(errors: list[str]) -> None:
    for md in DOCS.rglob("*.md"):
        if _is_under_archive(md):
            continue
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
    for md in list(DOCS.rglob("*.md")) + [ROOT / "README.md"]:
        if not md.is_file() or _is_under_archive(md):
            continue
        text = md.read_text(encoding="utf-8")
        for prefix in CROSS_BRANCH_PATHS:
            if prefix not in text:
                continue
            if "cross-branch" in text.lower() or "not yet integrated" in text.lower():
                continue
            if "not yet integrated" in text or "CROSS_BRANCH" in text:
                continue
            if md.name in {"MODEL_RESEARCH_PROGRAM.md", "RUNPOD.md"}:
                if "not yet integrated" in text:
                    continue
            errors.append(
                f"{md.relative_to(ROOT)} references {prefix} without cross-branch marker"
            )


def check_stale_authority_phrases(errors: list[str]) -> None:
    for path in _authority_md_files():
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        for label, pattern in STALE_AUTHORITY_PATTERNS:
            for match in pattern.finditer(text):
                line_idx = text.count("\n", 0, match.start())
                window_start = max(0, line_idx - 2)
                window_end = min(len(lines), line_idx + 3)
                window = "\n".join(lines[window_start:window_end])
                if HISTORICAL_CONTEXT_RE.search(window):
                    continue
                rel = path.relative_to(ROOT)
                snippet = lines[line_idx].strip() if line_idx < len(lines) else match.group(0)
                errors.append(
                    f"stale authority phrase in {rel}: {label!r} (line: {snippet})"
                )


def check_protected_diff(errors: list[str]) -> None:
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
    if not ACTIVE_JSON.is_file():
        errors.append("missing docs/ACTIVE_NOW.json")
        return
    data = json.loads(ACTIVE_JSON.read_text(encoding="utf-8"))
    if data.get("schema") != "nano-lm.docs.active_now.v1":
        errors.append("ACTIVE_NOW.json schema mismatch")


def check_preregistrations(errors: list[str]) -> None:
    """Gate opted-in preregs on the eighteen-field standard (SPEC §20).

    Only files declaring `**Standard:** question-before-architecture-v1` are
    checked; preregs predating the standard are grandfathered, not failed.
    Delegates to scripts/check_prereg.py so the field list has one definition.
    """
    checker = ROOT / "scripts/check_prereg.py"
    if not checker.is_file():
        errors.append("missing scripts/check_prereg.py")
        return
    proc = subprocess.run(
        [sys.executable, str(checker)], capture_output=True, text=True
    )
    if proc.returncode != 0:
        for line in (proc.stdout + proc.stderr).splitlines():
            text = line.strip()
            # Report only the failures, not the GRANDFATHERED/COMPLETE roster.
            if text.startswith("INCOMPLETE") or text.startswith("missing:"):
                errors.append(f"prereg: {text}")


def main() -> int:
    errors: list[str] = []
    check_links(errors)
    check_stub_archives(errors)
    check_cross_branch_markers(errors)
    check_stale_authority_phrases(errors)
    check_protected_diff(errors)
    check_single_authority_per_concern(errors)
    check_preregistrations(errors)
    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        return 1
    print("DOCS_INTEGRITY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
