#!/usr/bin/env python3
"""Fail-closed check for OWNER_* authorization receipts.

Invariant: only STATUS=ACTIVE + unexpired + unused receipts may grant force.
Consumed receipts must never satisfy a later authorization check.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


def _field(text: str, key: str) -> str | None:
    m = re.search(rf"^{re.escape(key)}:\s*(.+)$", text, re.M)
    return m.group(1).strip() if m else None


def _list_field(text: str, key: str) -> list[str]:
    m = re.search(rf"^{re.escape(key)}:\s*$\n((?:\s+-\s+.+$\n?)+)", text, re.M)
    if not m:
        # inline yaml-ish path_allowlist older form
        m2 = re.search(rf"^{re.escape(key)}:\s*$\n((?:\s+-\s+.+$\n?)+)", text, re.M)
        if not m2:
            return []
        m = m2
    return [ln.split("-", 1)[1].strip() for ln in m.group(1).splitlines() if "-" in ln]


def assert_receipt_active(
    receipt_path: Path,
    *,
    requested_force: str,
    requested_paths: set[str] | None = None,
    now: datetime | None = None,
) -> dict:
    text = receipt_path.read_text(encoding="utf-8")
    now = now or datetime.now(timezone.utc)
    status = (_field(text, "STATUS") or "").upper()
    force = _field(text, "FORCE_ID") or _field(text, "owner_force")
    reusable = (_field(text, "REUSABLE") or "true").lower()
    consumed_at = _field(text, "CONSUMED_AT")
    expires_at = _field(text, "EXPIRES_AT")

    if status != "ACTIVE":
        raise SystemExit(
            f"AUTH_RECEIPT_FAIL: {receipt_path} STATUS={status or 'MISSING'} "
            f"(need ACTIVE; CONSUMED receipts are non-reusable)"
        )
    if reusable in {"false", "no", "0"}:
        raise SystemExit(f"AUTH_RECEIPT_FAIL: {receipt_path} REUSABLE=false")
    if consumed_at and consumed_at.lower() not in {"null", "none", "~"}:
        raise SystemExit(f"AUTH_RECEIPT_FAIL: {receipt_path} already CONSUMED_AT={consumed_at}")
    if force and force != requested_force:
        raise SystemExit(
            f"AUTH_RECEIPT_FAIL: force mismatch have={force} requested={requested_force}"
        )
    if expires_at:
        # accept Z timestamps
        exp = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        if now > exp:
            raise SystemExit(f"AUTH_RECEIPT_FAIL: {receipt_path} expired at {expires_at}")

    auth_paths = set(_list_field(text, "AUTHORIZED_PATHS")) or set(_list_field(text, "path_allowlist"))
    if requested_paths is not None and auth_paths:
        extra = set(requested_paths) - auth_paths
        if extra:
            raise SystemExit(f"AUTH_RECEIPT_FAIL: paths outside allowlist: {sorted(extra)}")

    # legacy boolean must not claim grant when STATUS missing/ACTIVE unclear
    if re.search(r"^authorize_commit:\s*true\s*$", text, re.M) and status != "ACTIVE":
        raise SystemExit(f"AUTH_RECEIPT_FAIL: authorize_commit true but STATUS!={status}")

    return {
        "receipt": str(receipt_path),
        "force_id": force,
        "status": status,
        "authorized_paths": sorted(auth_paths),
    }


def mark_consumed(receipt_path: Path, *, result_commit: str | None = None) -> None:
    text = receipt_path.read_text(encoding="utf-8")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    text = re.sub(r"^STATUS:\s*.*$", "STATUS: CONSUMED", text, count=1, flags=re.M)
    if re.search(r"^REUSABLE:", text, re.M):
        text = re.sub(r"^REUSABLE:\s*.*$", "REUSABLE: false", text, count=1, flags=re.M)
    else:
        text = "REUSABLE: false\n" + text
    if re.search(r"^CONSUMED_AT:", text, re.M):
        text = re.sub(r"^CONSUMED_AT:\s*.*$", f"CONSUMED_AT: {now}", text, count=1, flags=re.M)
    else:
        text += f"\nCONSUMED_AT: {now}\n"
    if result_commit:
        if re.search(r"^AUTHORIZED_RESULT_COMMIT:", text, re.M):
            text = re.sub(
                r"^AUTHORIZED_RESULT_COMMIT:\s*.*$",
                f"AUTHORIZED_RESULT_COMMIT: {result_commit}",
                text,
                count=1,
                flags=re.M,
            )
        else:
            text += f"AUTHORIZED_RESULT_COMMIT: {result_commit}\n"
    # clear grant booleans
    for k in ("authorize_commit", "authorize_push", "authorize_tag", "tag_push"):
        text = re.sub(rf"^{k}:\s*.*$", f"{k}: false", text, count=1, flags=re.M)
    text = re.sub(r"^scope_bits:\s*.*$", "scope_bits: []", text, count=1, flags=re.M)
    receipt_path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("receipt", type=Path)
    ap.add_argument("--force", required=True)
    ap.add_argument("--path", action="append", default=[])
    ap.add_argument("--expect-consumed", action="store_true",
                    help="Pass only if receipt is CONSUMED/non-reusable")
    args = ap.parse_args(argv)
    text = args.receipt.read_text(encoding="utf-8")
    status = (_field(text, "STATUS") or "").upper()
    if args.expect_consumed:
        if status != "CONSUMED":
            print(f"FAIL: expected CONSUMED, got {status or 'MISSING'}", file=sys.stderr)
            return 1
        if (_field(text, "REUSABLE") or "").lower() not in {"false", "no", "0"}:
            print("FAIL: CONSUMED receipt must set REUSABLE: false", file=sys.stderr)
            return 1
        if (_field(text, "authorize_commit") or "").lower() == "true":
            print("FAIL: CONSUMED receipt still has authorize_commit: true", file=sys.stderr)
            return 1
        print(f"OK: {args.receipt} CONSUMED non-reusable")
        return 0
    assert_receipt_active(
        args.receipt,
        requested_force=args.force,
        requested_paths=set(args.path) or None,
    )
    print(f"OK: {args.receipt} ACTIVE for {args.force}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
