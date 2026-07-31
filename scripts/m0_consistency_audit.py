#!/usr/bin/env python3
"""Ungated M0 consistency audit. Does not authorize execute/commit."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    issues: list[str] = []
    ok: list[str] = []

    for rel in [
        "papers/PUBLIC_ONE_PAGER.md",
        "papers/MITIGATION_STATUS_SCORECARD.md",
        "papers/M0_CONTINUE_RESIDUALS.md",
        "papers/FIRST_PRINCIPLES_RISK_MITIGATION.md",
        "papers/OWNER_SPEECH_ACTS.md",
        "papers/PAPER_ALPHA_CORRECTION_NOTE.md",
    ]:
        if (ROOT / rel).is_file():
            ok.append(f"present:{rel}")
        else:
            issues.append(f"MISSING:{rel}")

    for rel in [
        "wedge_v1/results_wedge_v1_classical.json",
        "wedge_v1/results_wedge_v1_phase3_eclass.json",
        "wedge_v1/results_wedge_v1_phase3.json",
    ]:
        p = ROOT / rel
        if not p.is_file():
            issues.append(f"MISSING:{rel}")
            continue
        d = json.loads(p.read_text())
        if d.get("corpus_class") == "SYNTHETIC_MINI":
            ok.append(f"corpus_class:{rel}")
        else:
            issues.append(f"NO_CORPUS_CLASS:{rel}")

    ledger = json.loads((ROOT / "papers/EVIDENCE_LEDGER.json").read_text())
    claims = ledger.get("claims") or []
    need = {"C_E1_GATE", "C_E1_PRODUCT_THESIS", "C_E1_MEASUREMENT"}
    have = {c["claim_id"] for c in claims if c.get("context_of_use")}
    missing = sorted(need - have)
    if missing:
        issues.append(f"LEDGER_MISSING_COU:{missing}")
    else:
        ok.append("ledger:E1_context_of_use")

    for p in ROOT.rglob("AUTH_RECORD.md"):
        if ".git" in p.parts:
            continue
        txt = p.read_text(encoding="utf-8", errors="replace")
        rel = str(p.relative_to(ROOT))
        if "AUTHORIZE_" in txt and "scope_bits" not in txt:
            issues.append(f"AUTH_NO_SCOPE:{rel}")
        else:
            ok.append(f"scope_bits:{rel}")


    # E4 kill packaging (canonical status)
    e4u = ROOT / "trajectory/results_e4_utility.json"
    if e4u.is_file():
        ok.append("present:trajectory/results_e4_utility.json")
    else:
        # soft: only issue if canonical claims EXECUTED
        canon = ROOT / "audit/discussion-to-implementation/CANONICAL_STATUS_TABLE.md"
        if canon.is_file() and "E4_RESULT_STATUS: KILL" in canon.read_text():
            issues.append("MISSING:trajectory/results_e4_utility.json (canonical says E4 KILL)")

    utc = datetime.now(timezone.utc).isoformat()
    out = ROOT / "papers/M0_CONSISTENCY_AUDIT.md"
    issue_lines = "\n".join(f"- {x}" for x in issues) if issues else "- (none)"
    ok_lines = "\n".join(f"- {x}" for x in ok)
    out.write_text(
        f"# M0 consistency audit\n\n**UTC:** {utc}\n\n"
        f"## OK ({len(ok)})\n{ok_lines}\n\n"
        f"## Issues ({len(issues)})\n{issue_lines}\n"
    )
    print(f"ok={len(ok)} issues={len(issues)}")
    for i in issues:
        print(" ", i)
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
