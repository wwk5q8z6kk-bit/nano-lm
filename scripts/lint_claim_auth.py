#!/usr/bin/env python3
"""Fail if AUTHORIZE_* tokens appear outside the auth allowlist.

Design from papers/FIRST_PRINCIPLES_RISK_MITIGATION.md blocker B1.
Does not authorize anything; hygiene only.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAT = re.compile(r"AUTHORIZE_[A-Z0-9_]+")

ALLOW_PATH_PREFIXES = (
    "papers/EXECUTION_QUEUE.md",
    "trajectory/",  # AUTH_RECORD receipts may live here; still scanned for inventiveness elsewhere
)
ALLOW_EXACT = {
    "papers/EXECUTION_QUEUE.md",
    "papers/DECISION_GATES.md",
    "papers/FIRST_PRINCIPLES_RISK_MITIGATION.md",
    "papers/AMBITION.md",
    "AGENTS.md",
    "audit/discussion-to-implementation/SWARM_QUEEN_SYNTHESIS_2026-07-31.md",
    "audit/discussion-to-implementation/DIFF_E_REMEDIATION_ACCEPTANCE.md",
}

# Paths that may *mention* AUTHORIZE_ as documentation of the rule
DOC_MENTION_ALLOW = ALLOW_EXACT | {
    "papers/LABORATORY_CONSTITUTION.md",
    "papers/CLAIM_GLOSSARY.md",
    "scripts/lint_claim_auth.py",
}

FORBIDDEN_CREATE_TAG = re.compile(
    r"create(?:\s+annotated)?\s+tag\s+`?post-alpha-evidence-freeze-2026-07-31`?",
    re.I,
)


def iter_text_files():
    skip_dirs = {".git", ".venv", "node_modules", "__pycache__", ".quarantine", "artifacts/local_raw_archive"}
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.suffix.lower() not in {".md", ".json", ".py", ".txt", ".yml", ".yaml"}:
            continue
        yield path


def main() -> int:
    errors = []
    for path in iter_text_files():
        rel = str(path.relative_to(ROOT))
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            errors.append(f"{rel}: read error {e}")
            continue

        if FORBIDDEN_CREATE_TAG.search(text) and "DO NOT" not in text and "do not" not in text.lower():
            # allow lines that explicitly forbid
            for i, line in enumerate(text.splitlines(), 1):
                if FORBIDDEN_CREATE_TAG.search(line) and "do not" not in line.lower() and "preserve" not in line.lower():
                    errors.append(f"{rel}:{i}: instructs creating premature freeze tag")

        if rel in DOC_MENTION_ALLOW or rel.startswith("papers/") and rel.endswith("DECISION_GATES.md"):
            continue

        # Ambition / roadmap / portfolio must not contain AUTHORIZE_ execute tokens
        if rel in {
            "papers/RESEARCH_PORTFOLIO.md",
            "papers/TECHNOLOGY_ROADMAP.md",
            "papers/MASTER_PLAN.md",
            "papers/NANOSCRIBE_VNEXT.md",
        }:
            for i, line in enumerate(text.splitlines(), 1):
                if PAT.search(line):
                    errors.append(f"{rel}:{i}: AUTHORIZE_* in non-auth ambition doc: {line.strip()[:120]}")

        # AUTH_RECORD special-case: must declare valid_only_if_queued if it authorizes
        if rel.endswith("AUTH_RECORD.md"):
            if "AUTHORIZE_" in text and "valid_only_if_queued" not in text and "VALID_ONLY_IF_QUEUED" not in text:
                errors.append(f"{rel}: AUTH_RECORD contains AUTHORIZE_* without valid_only_if_queued marker")

    if errors:
        print("lint_claim_auth: FAIL")
        for e in errors[:50]:
            print(" ", e)
        if len(errors) > 50:
            print(f"  ... and {len(errors)-50} more")
        return 1
    print("lint_claim_auth: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
