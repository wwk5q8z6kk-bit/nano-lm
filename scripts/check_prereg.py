#!/usr/bin/env python3
"""Check preregistrations against the eighteen-field standard (SPEC §20).

Standard ratified 2026-08-26 —
`research/decision_records/2026-08-26-question-before-architecture.md`.

Presence, not quality. This tool cannot tell whether an invariance requirement
is *right*; it can tell whether one was written down before the run. An
unanswered field is a stop, not a caveat (SPEC §25).

Preregistrations written before the standard are GRANDFATHERED, not failed:
history is recorded, not retrofitted (SPEC §3 principle 8). A prereg opts in by
declaring `**Standard:** question-before-architecture-v1`.

Exit 0 = all opted-in preregs complete. Exit 1 = at least one is incomplete.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREREG_DIR = ROOT / "research/preregistrations"
STANDARD = "question-before-architecture-v1"

# (canonical name, regexes that may head the section)
FIELDS: tuple[tuple[str, str], ...] = (
    ("product question", r"product\s+question"),
    ("scientific question", r"scientific\s+question"),
    ("instrument", r"instrument"),
    ("measured bottleneck", r"measured\s+bottleneck"),
    ("hypothesis", r"hypothesis"),
    ("baseline / control", r"baseline(\s*/\s*|\s+and\s+|\s+)control|baseline"),
    ("manipulation", r"manipulation"),
    ("invariance requirements", r"invariance\s+(requirements?|conditions?)"),
    ("confound analysis", r"confound\s+analysis|confounds?"),
    ("outcome measures", r"outcome\s+measures?"),
    ("decision rule", r"decision\s+rule"),
    ("kill condition", r"kill\s+condition"),
    ("falsifier", r"falsifier"),
    ("authorization", r"authoriz"),
    ("provenance", r"provenance"),
    ("resource accounting", r"resource\s+accounting"),
    ("reproducibility", r"reproducibility"),
    ("interpretation boundary", r"interpretation\s+boundary"),
)

PLACEHOLDER = re.compile(r"\b(todo|tbd|tk|xxx)\b", re.I)
# Template guidance is written as parenthetical blocks and blockquotes, both of
# which may span lines. Strip the whole span, not the first line of it —
# otherwise a wrapped prompt's continuation lines read as an answer.
PARENTHETICAL = re.compile(r"\([^()]*\)", re.S)
ANGLE = re.compile(r"<[^<>]*>")
MIN_CHARS = 20


def sections(text: str) -> dict[str, str]:
    """Map heading text -> body, for ## / ### headings above the RESULT split."""
    body = re.split(r"^#\s+RESULT\b", text, maxsplit=1, flags=re.M)[0]
    out: dict[str, str] = {}
    parts = re.split(r"^(#{2,3})\s+(.+?)\s*$", body, flags=re.M)
    for i in range(2, len(parts), 3):
        heading = re.sub(r"^\d+[.)]\s*", "", parts[i]).strip()
        out[heading.lower()] = parts[i + 1]
    return out


def substantive(body: str) -> bool:
    """A field is answered if real prose survives stripping template guidance."""
    kept = "\n".join(ln for ln in body.splitlines() if not ln.lstrip().startswith(">"))
    # Repeat: stripping innermost pairs first collapses one nesting level a pass.
    while True:
        stripped = PARENTHETICAL.sub(" ", kept)
        if stripped == kept:
            break
        kept = stripped
    kept = ANGLE.sub(" ", kept)
    if PLACEHOLDER.search(kept):
        return False
    return len("".join(kept.split())) >= MIN_CHARS


def check(path: Path) -> tuple[str, list[str]]:
    """Return (status, missing_fields)."""
    text = path.read_text(encoding="utf-8")
    if STANDARD not in text:
        return "GRANDFATHERED", []
    found = sections(text)
    missing: list[str] = []
    for canonical, pattern in FIELDS:
        rx = re.compile(pattern, re.I)
        body = next((b for h, b in found.items() if rx.search(h)), None)
        if body is None:
            missing.append(f"{canonical} (no section)")
        elif not substantive(body):
            missing.append(f"{canonical} (empty/placeholder)")
    return ("INCOMPLETE" if missing else "COMPLETE"), missing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", type=Path, help="prereg files (default: all)")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="also fail on GRANDFATHERED preregs (do not use in CI yet)",
    )
    args = ap.parse_args()

    targets = args.paths or sorted(PREREG_DIR.glob("PREREG_*.md"))
    if not targets:
        print("no preregistrations found", file=sys.stderr)
        return 1

    failed = False
    for path in targets:
        status, missing = check(path)
        try:
            rel = path.resolve().relative_to(ROOT)
        except ValueError:  # a path outside the repo, e.g. an ad-hoc draft
            rel = path
        print(f"{status:14} {rel}")
        for field in missing:
            print(f"               missing: {field}")
        if status == "INCOMPLETE" or (args.strict and status == "GRANDFATHERED"):
            failed = True

    if failed:
        print("PREREG_INCOMPLETE", file=sys.stderr)
        return 1
    print("PREREG_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
