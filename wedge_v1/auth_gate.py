"""Fail-closed auth gate for wedge runners (LAB.B1 / LAB.B14).

Does not mint AUTHORIZE_*. Reads AUTH_RECORD + EXECUTION_QUEUE only.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
QUEUE = REPO / "papers" / "EXECUTION_QUEUE.md"


def _parse_scope_bits(auth_path: Path) -> set[str]:
    text = auth_path.read_text(encoding="utf-8")
    m = re.search(r"scope_bits:\s*\[([^\]]*)\]", text)
    if not m:
        raise SystemExit(f"AUTH_GATE_FAIL: missing scope_bits in {auth_path}")
    bits = {b.strip() for b in m.group(1).split(",") if b.strip()}
    if not bits:
        raise SystemExit(f"AUTH_GATE_FAIL: empty scope_bits in {auth_path}")
    return bits


def _queue_mentions(auth_id: str) -> dict:
    q = QUEUE.read_text(encoding="utf-8") if QUEUE.exists() else ""
    return {
        "queued_active": bool(
            re.search(rf"AUTHORIZED_WORK:\s*{re.escape(auth_id)}", q)
            or re.search(rf"\|\s*2\s*\|.*{re.escape(auth_id)}.*\*\*ACTIVE\*\*", q)
        ),
        "queued_done": bool(
            re.search(rf"LAST_COMPLETED:\s*{re.escape(auth_id)}", q)
            or re.search(rf"{re.escape(auth_id)}[^\n]*\*\*DONE\*\*", q)
            or (
                auth_id == "AUTHORIZE_WEDGE_V1_CLASSICAL_BASELINE"
                and ("PHASE2_RESULT:" in q or "Wedge Phase 1–2 classical" in q or "Classical baseline" in q)
                and "DONE" in q
            )
            or (
                auth_id == "AUTHORIZE_WEDGE_V1_PHASE3_ECLASS_PROBE"
                and ("PHASE3_ECLASS_RESULT:" in q or "Phase 3 E-class" in q)
                and ("DONE" in q or "ECLASS_CLOSED" in q)
            )
        ),
        "idle": "IDLE_AFTER_WEDGE_V1" in q or "NONE_PENDING" in q or "AUTHORIZED_NONEXECUTION_WORK: NONE" in q,
    }


def require_auth(
    *,
    auth_id: str,
    auth_record: Path,
    need_bits: set[str],
    mode: str = "execute_eval",
) -> dict:
    """mode: execute_eval | integrity_remediation

    - execute_eval: auth must be actively queued
    - integrity_remediation: allow if historically DONE (idle queue OK)
    """
    bits = _parse_scope_bits(auth_record)
    missing = need_bits - bits
    if missing:
        raise SystemExit(f"AUTH_GATE_FAIL: missing scope_bits {sorted(missing)} in {auth_record}")

    text = auth_record.read_text(encoding="utf-8")
    if auth_id not in text:
        raise SystemExit(f"AUTH_GATE_FAIL: auth_id {auth_id} not in {auth_record}")

    qstate = _queue_mentions(auth_id)
    if mode == "execute_eval":
        if not qstate["queued_active"]:
            raise SystemExit(
                f"AUTH_GATE_FAIL: {auth_id} not actively queued "
                f"(CONTINUE_SESSION cannot unlock). Queue={qstate}"
            )
    elif mode == "integrity_remediation":
        # Completed arms may re-score for firewall/honesty fixes while idle.
        if not (qstate["queued_done"] or qstate["queued_active"] or qstate["idle"]):
            raise SystemExit(f"AUTH_GATE_FAIL: remediation refused; queue state={qstate}")
    else:
        raise SystemExit(f"AUTH_GATE_FAIL: unknown mode {mode}")

    # Refuse if a sibling OWNER_* receipt exists and is CONSUMED-only for this force.
    # (Wedge AUTH_RECORD path remains primary; OWNER_* receipts are speech-act scoped.)
    return {"auth_id": auth_id, "scope_bits": sorted(bits), "mode": mode, "queue": qstate}
