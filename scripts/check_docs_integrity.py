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

REQUIRED_CANONICAL_DOCS = (
    "docs/PROJECT_AUTHORITY.md",
    "docs/PROJECT_CHARTER.md",
    "docs/ACTIVE_NOW.md",
    "docs/ACTIVE_NOW.json",
    "docs/EXECUTION_PLAN.md",
    "docs/subsystems/NANOSCRIBE.md",
    "docs/research/ACCELERATED_CAMPAIGN.md",
    "docs/knowledge/AGENT_PROGRAM_KNOWLEDGE.md",
    "docs/knowledge/PROGRAM_CHECKPOINTS.json",
    "artifacts/campaign/CAMPAIGN_AUTONOMOUS_EXECUTION.md",
)

DOC_CODE_ANCHORS: dict[str, tuple[str, ...]] = {
    "docs/subsystems/NANOSCRIBE.md": (
        "nanoscribe/encounter.py",
        "nanoscribe/adapt.py",
        "nanoscribe/harness.py",
        "nanoscribe/tracks.py",
    ),
    "docs/infrastructure/TOOL_CALLING.md": (
        "nanoscribe/tool_calling.py",
        "nanoscribe/structured_inference.py",
        "nanoscribe/tool_inference.py",
    ),
    "docs/research/ACCELERATED_CAMPAIGN.md": (
        "scripts/campaign_control_plane.py",
        "artifacts/campaign/experiment_manifest.v1.schema.json",
    ),
}

CHECKPOINT_STATUSES = frozenset({"done", "in_progress", "pending", "cancelled"})

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


def check_required_canonical_docs(errors: list[str]) -> None:
    for rel in REQUIRED_CANONICAL_DOCS:
        path = ROOT / rel
        if not path.is_file():
            errors.append(f"missing required canonical doc: {rel}")


def check_doc_code_anchors(errors: list[str]) -> None:
    for doc_rel, anchors in DOC_CODE_ANCHORS.items():
        doc_path = ROOT / doc_rel
        if not doc_path.is_file():
            errors.append(f"doc anchor source missing: {doc_rel}")
            continue
        for anchor in anchors:
            if not (ROOT / anchor).exists():
                errors.append(f"{doc_rel} references missing path: {anchor}")


def check_program_checkpoints(errors: list[str]) -> None:
    checkpoints = ROOT / "docs/knowledge/PROGRAM_CHECKPOINTS.json"
    if not checkpoints.is_file():
        errors.append("missing docs/knowledge/PROGRAM_CHECKPOINTS.json")
        return
    try:
        data = json.loads(checkpoints.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"PROGRAM_CHECKPOINTS.json invalid JSON: {exc}")
        return
    if data.get("schema") != "nano-lm.docs.program_checkpoints.v1":
        errors.append("PROGRAM_CHECKPOINTS.json schema mismatch")
    for phase_key in ("phase_c_v2", "phase_d_p1_exit"):
        phase = data.get(phase_key)
        if not isinstance(phase, dict):
            errors.append(f"PROGRAM_CHECKPOINTS missing {phase_key}")
            continue
        gates = phase.get("gates")
        if not isinstance(gates, list) or not gates:
            errors.append(f"PROGRAM_CHECKPOINTS {phase_key} gates empty")
            continue
        for gate in gates:
            if not isinstance(gate, dict):
                errors.append(f"PROGRAM_CHECKPOINTS invalid gate in {phase_key}")
                continue
            gid = gate.get("id", "?")
            status = gate.get("status")
            if status not in CHECKPOINT_STATUSES:
                errors.append(f"PROGRAM_CHECKPOINTS gate {gid} bad status: {status!r}")
            evidence = gate.get("evidence", [])
            if status == "done":
                if not evidence:
                    errors.append(f"PROGRAM_CHECKPOINTS gate {gid} done but no evidence")
                for rel in evidence:
                    if not (ROOT / rel).exists():
                        errors.append(f"PROGRAM_CHECKPOINTS gate {gid} missing evidence: {rel}")
            elif status == "in_progress" and evidence:
                if not any((ROOT / rel).exists() for rel in evidence):
                    errors.append(
                        f"PROGRAM_CHECKPOINTS gate {gid} (in_progress) has no existing evidence paths"
                    )


def check_agents_knowledge_link(errors: list[str]) -> None:
    agents = ROOT / "AGENTS.md"
    if not agents.is_file():
        return
    text = agents.read_text(encoding="utf-8")
    if "docs/knowledge/AGENT_PROGRAM_KNOWLEDGE" not in text:
        errors.append("AGENTS.md must link docs/knowledge/AGENT_PROGRAM_KNOWLEDGE.md")


def main() -> int:
    errors: list[str] = []
    check_links(errors)
    check_stub_archives(errors)
    check_cross_branch_markers(errors)
    check_stale_authority_phrases(errors)
    check_protected_diff(errors)
    check_single_authority_per_concern(errors)
    check_required_canonical_docs(errors)
    check_doc_code_anchors(errors)
    check_program_checkpoints(errors)
    check_agents_knowledge_link(errors)
    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        return 1
    print("DOCS_INTEGRITY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
