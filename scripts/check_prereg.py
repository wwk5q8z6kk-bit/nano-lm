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
# Opt-in must be a declaration line, not any prose mention of the standard's
# name — otherwise a document that merely discusses the standard opts itself in.
STANDARD = "question-before-architecture-v1"
OPT_IN = re.compile(
    rf"^\s*\*\*Standard:\*\*\s*{re.escape(STANDARD)}\s*$", re.M
)

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


# Non-interventional studies (measurement, construct validity, calibration) have
# no manipulation. SPEC §20 lets such a field be "N/A — <reason>"; a bare N/A is
# an unanswered field. Fields 8 and 12 are never N/A — a measurement study can
# still fail its own preconditions.
NEVER_NA = frozenset({"invariance requirements", "kill condition"})
NA = re.compile(r"^\s*(?:\*\*)?n/?a(?:\*\*)?\b[\s:—–-]*(.*)$", re.I | re.M)
NA_MIN_REASON = 12


def na_state(body: str, field: str) -> str:
    """'none' | 'justified' | 'bare' | 'illegal' — how this field uses N/A."""
    m = NA.search(body)
    if not m:
        return "none"
    if field in NEVER_NA:
        return "illegal"
    return (
        "justified"
        if len("".join(m.group(1).split())) >= NA_MIN_REASON
        else "bare"
    )


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
    # Placeholder tokens do not *disqualify* a field — they simply do not count
    # toward it. "TBD" alone is unfilled; "base commit is TBD pending a port",
    # inside a full provenance section, is an answer. Treating any occurrence as
    # a placeholder false-flagged exactly that case on the first real prereg.
    kept = PLACEHOLDER.sub(" ", kept)
    return len("".join(kept.split())) >= MIN_CHARS


def check(path: Path) -> tuple[str, list[str]]:
    """Return (status, missing_fields)."""
    text = path.read_text(encoding="utf-8")
    if not OPT_IN.search(text):
        return "GRANDFATHERED", []
    found = sections(text)
    missing: list[str] = []
    for canonical, pattern in FIELDS:
        rx = re.compile(pattern, re.I)
        body = next((b for h, b in found.items() if rx.search(h)), None)
        if body is None:
            missing.append(f"{canonical} (no section)")
            continue
        match na_state(body, canonical):
            case "justified":
                continue  # non-interventional, reason given
            case "bare":
                missing.append(f"{canonical} (bare N/A — state a reason)")
                continue
            case "illegal":
                missing.append(f"{canonical} (never N/A — SPEC §20)")
                continue
        if not substantive(body):
            missing.append(f"{canonical} (empty/placeholder)")
    return ("INCOMPLETE" if missing else "COMPLETE"), missing


# Prose copies of the eighteen-field list. `FIELDS` above is NORMATIVE — same
# principle as SPEC §0, where `nano/architecture.py` outranks prose about the
# taxonomy. Two divergent field-lists is the exact failure the 2026-08-26
# consolidation ended (§20 carried six fields, §25 carried nine); the list now
# exists in four places, so it is checked rather than trusted.
SPEC = "docs/NANO_VNEXT_MASTER_SPEC.md"
RECORD = "research/decision_records/2026-08-26-question-before-architecture.md"
TEMPLATE = "research/preregistrations/TEMPLATE.md"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def _copies() -> dict[str, list[str]]:
    """Extract the field list as written in each prose location."""
    out: dict[str, list[str]] = {}
    spec = (ROOT / SPEC).read_text(encoding="utf-8")
    m = re.search(r"```\n(product question →.*?)\n```", spec, re.S)
    # Always register the key. Skipping it when the regex misses would drop the
    # spec from the comparison entirely — a reformat there would then pass
    # silently, which is the failure this whole check exists to prevent.
    out[f"{SPEC} §20"] = (
        [x.strip() for x in m.group(1).replace("\n", " ").split("→") if x.strip()]
        if m
        else []
    )
    rec = (ROOT / RECORD).read_text(encoding="utf-8")
    out[RECORD] = re.findall(r"^\|\s*\d+\s*\|\s*\*\*(.+?)\*\*\s*\|", rec, re.M)
    tpl = (ROOT / TEMPLATE).read_text(encoding="utf-8")
    out[TEMPLATE] = re.findall(r"^##\s*\d+\.\s*(.+?)\s*$", tpl, re.M)
    return out


def check_field_list_consistency() -> list[str]:
    """Every prose copy must match FIELDS in name and order."""
    errs: list[str] = []
    ref = [_norm(c) for c, _ in FIELDS]
    for where, listed in _copies().items():
        if not listed:
            errs.append(f"{where}: field list not found (format changed?)")
            continue
        got = [_norm(x) for x in listed]
        if got == ref:
            continue
        if len(got) != len(ref):
            errs.append(f"{where}: has {len(got)} fields, FIELDS has {len(ref)}")
        for i, (a, b) in enumerate(zip(ref, got), start=1):
            if a != b:
                errs.append(f"{where}: field {i} is {b!r}, FIELDS says {a!r}")
                break
    return errs


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

    if missing_files := [p for p in targets if not p.is_file()]:
        for p in missing_files:
            print(f"no such file: {p}", file=sys.stderr)
        return 1

    failed = False

    # Runs even when no prereg opts in — the copies can drift with zero preregs
    # in the tree, which is exactly the current state.
    for err in check_field_list_consistency():
        print(f"FIELD-LIST     {err}", file=sys.stderr)
        failed = True

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
