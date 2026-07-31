#!/usr/bin/env python3
"""Classify owner chat into typed speech acts (B23 / OWNER_SPEECH_ACTS.md).

Does not mint AUTHORIZE_* or OWNER_* markers. Fail-closed: unknown → UNTYPED.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

# Exact / regex → (force, scope_bits)
RULES: list[tuple[re.Pattern[str], str, list[str]]] = [
    (re.compile(r"^\s*idle\b", re.I), "IDLE", []),
    (re.compile(r"^\s*(park|stop)\b", re.I), "IDLE", []),
    (re.compile(r"authorize\s+commit", re.I), "AUTHORIZE_COMMIT", ["commit"]),
    (re.compile(r"^\s*proceed\b", re.I), "AUTHORIZE_COMMIT", ["commit"]),
    (re.compile(r"authorize\s+push", re.I), "AUTHORIZE_PUSH", ["push"]),
    (re.compile(r"authorize\s+tag", re.I), "AUTHORIZE_TAG", ["tag"]),
    (re.compile(r"\bRATIFY_E4_EXECUTE\b"), "DISPOSE_E4", []),
    (re.compile(r"\bVOID_E4_AUTH\b"), "DISPOSE_E4", []),
    (re.compile(r"\bPARK_AS_EXPLORATORY\b"), "DISPOSE_E4", []),
    (re.compile(r"AUTHORIZE_[A-Z0-9_]+"), "AUTHORIZE_EXECUTE_CANDIDATE", ["execute"]),
    (re.compile(r"^\s*continue\b", re.I), "CONTINUE_SESSION", []),
    (re.compile(r"keep\s+going", re.I), "CONTINUE_SESSION", []),
    (re.compile(r"/autonomous-skill", re.I), "CONTINUE_SESSION", []),
]


def classify(text: str) -> dict:
    t = text.strip()
    tip_policy = None
    for pat, force, bits in RULES:
        if pat.search(t):
            if force == "AUTHORIZE_TAG":
                for pol in ("clean-lineage", "non-freeze-snapshot", "verdict-annotation", "defer"):
                    if re.search(rf"\b{re.escape(pol)}\b", t, re.I):
                        tip_policy = pol
                        break
                if tip_policy is None:
                    tip_policy = "REQUIRED_UNSPECIFIED"
            return {
                "force": force,
                "scope_bits": bits,
                "tip_policy": tip_policy,
                "may_mint_owner_marker": force
                in {"AUTHORIZE_COMMIT", "AUTHORIZE_PUSH", "AUTHORIZE_TAG"},
                "note": (
                    "CONTINUE_SESSION: ungated M0 only; never commit/tag/push/execute"
                    if force == "CONTINUE_SESSION"
                    else None
                ),
            }
    return {
        "force": "UNTYPED",
        "scope_bits": [],
        "tip_policy": None,
        "may_mint_owner_marker": False,
        "note": "Ask owner to pick a force from papers/OWNER_SPEECH_ACTS.md",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("text", nargs="?", help="Owner message (or stdin)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    text = args.text if args.text is not None else sys.stdin.read()
    out = classify(text)
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"force={out['force']}")
        print(f"scope_bits={','.join(out['scope_bits']) or '(none)'}")
        if out.get("tip_policy"):
            print(f"tip_policy={out['tip_policy']}")
        if out.get("note"):
            print(f"note={out['note']}")
        print(f"may_mint_owner_marker={out['may_mint_owner_marker']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
