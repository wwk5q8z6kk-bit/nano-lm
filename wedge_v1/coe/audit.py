"""Local CoE Audit suite for Nano Runtime outputs (not paper-only)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wedge_v1.coe.record import load_record
from wedge_v1.coe.schema import digest_docs


def _check(name: str, ok: bool, reason: str = "", *, abstain: bool = False) -> dict:
    return {
        "check": name,
        "result": "abstain" if abstain else ("pass" if ok else "fail"),
        "reason": reason,
    }


def audit_payload(payload: dict, docs: dict[str, str] | None = None) -> dict:
    """Independent checks on an ask/report payload (+ optional live docs)."""
    checks: list[dict] = []
    coe_claims = payload.get("coe_claims") or []
    claims = payload.get("claims") or []
    trace = payload.get("trace") or {}

    # Evidence existence / offsets
    missing_src = 0
    bad_offset = 0
    unsupported = 0
    if docs is None:
        checks.append(_check("evidence_existence", True, "docs not provided", abstain=True))
        checks.append(_check("offset_validity", True, "docs not provided", abstain=True))
    else:
        for tc in coe_claims:
            for atom in tc.get("evidence_atoms") or []:
                did = atom.get("doc_id")
                if did and did not in docs:
                    missing_src += 1
                    continue
                body = docs.get(did or "", "")
                start, end, text = atom.get("start"), atom.get("end"), atom.get("text") or ""
                if start is not None and end is not None and body:
                    slice_ = body[int(start) : int(end)]
                    if text and slice_ != text and text not in body:
                        bad_offset += 1
                elif not text:
                    unsupported += 1
        checks.append(
            _check(
                "evidence_existence",
                missing_src == 0,
                f"missing_docs={missing_src}",
            )
        )
        checks.append(
            _check(
                "offset_validity",
                bad_offset == 0,
                f"invalid_offsets={bad_offset}",
            )
        )

    # Claim support: every PRESENT claim needs ≥1 atom
    present_without = 0
    for tc in coe_claims:
        if tc.get("status") in {"PRESENT", "CONFIRMED"} and not (tc.get("evidence_atoms") or []):
            present_without += 1
    for c in claims:
        if c.get("status") in {"PRESENT", "CONFIRMED"} and not (c.get("evidence") or []):
            present_without += 1
    checks.append(
        _check(
            "claim_support",
            present_without == 0,
            f"present_without_evidence={present_without}",
        )
    )

    # Trace completeness
    has_trace = bool(trace.get("events") is not None or payload.get("coe", {}).get("run_id"))
    checks.append(_check("trace_completeness", has_trace, "missing ask_trace/coe run" if not has_trace else "ok"))

    # Method–execution alignment: solver_path vs trace.solvers
    path = payload.get("solver_path") or payload.get("solver") or []
    tsolvers = trace.get("solvers") or []
    aligned = (not path) or (not tsolvers) or any(s in path for s in tsolvers) or any(s in tsolvers for s in path)
    checks.append(
        _check(
            "method_execution_alignment",
            bool(aligned),
            f"path={path!r} solvers={tsolvers!r}",
        )
    )

    # Spec compliance: no LM when classical-only
    lm = payload.get("lm_invoked")
    checks.append(_check("spec_classical_only", lm is not True, f"lm_invoked={lm}"))

    # Citation faithfulness proxy: coe invariant present + claims have claim_id
    bound = all(c.get("claim_id") for c in claims) if claims else True
    inv = (payload.get("coe") or {}).get("invariant") == "EVIDENCE_CREATED_WITH_CLAIM"
    checks.append(
        _check(
            "citation_faithfulness_binding",
            bound and (inv or not claims),
            "missing claim_id or coe invariant" if not (bound and inv) else "ok",
        )
    )

    # Contradiction ignored
    banner = payload.get("contradiction_banner")
    status = payload.get("answer_status")
    contrad_ok = not (banner and status == "SUPPORTED")
    checks.append(
        _check(
            "contradiction_not_ignored",
            contrad_ok,
            "banner present but status SUPPORTED" if not contrad_ok else "ok",
        )
    )

    # Incomplete conjunction signal from ask()
    if (payload.get("abstain_class") == "coe_incomplete_conjunction"
            or "incomplete conjunction" in str(payload.get("note") or "").lower()):
        checks.append(
            _check(
                "complete_conjunction",
                False,
                str(payload.get("note") or "incomplete"),
            )
        )
    elif payload.get("predicate_support") and len(payload.get("predicate_support") or []) >= 2:
        incomplete = any(not s.get("supported") for s in payload["predicate_support"])
        checks.append(
            _check(
                "complete_conjunction",
                not incomplete,
                "all predicates supported" if not incomplete else "partial support",
            )
        )

    fails = [c for c in checks if c["result"] == "fail"]
    return {
        "schema": "nano-lm.wedge_v1.coe_audit.v1",
        "n_checks": len(checks),
        "n_fail": len(fails),
        "n_pass": sum(1 for c in checks if c["result"] == "pass"),
        "n_abstain": sum(1 for c in checks if c["result"] == "abstain"),
        "ok": len(fails) == 0,
        "checks": checks,
        "failure_codes": _codes_from_checks(fails),
        "run_id": (payload.get("coe") or {}).get("run_id"),
    }


def _codes_from_checks(fails: list[dict]) -> list[str]:
    m = {
        "evidence_existence": "COE_MISSING_SOURCE",
        "offset_validity": "COE_INVALID_OFFSET",
        "claim_support": "COE_UNSUPPORTED_PREDICATE",
        "trace_completeness": "COE_CONFIG_MISSING",
        "method_execution_alignment": "COE_METHOD_TRACE_MISMATCH",
        "spec_classical_only": "COE_UNREGISTERED_SOLVER",
        "citation_faithfulness_binding": "COE_POSTHOC_CITATION",
        "contradiction_not_ignored": "COE_CONTRADICTION_IGNORED",
        "complete_conjunction": "COE_INCOMPLETE_CONJUNCTION",
    }
    return [m.get(c["check"], "COE_DERIVATION_UNKNOWN") for c in fails]


def audit_record(path: Path) -> dict:
    events = load_record(path)
    types = {e.get("event_type") for e in events}
    required = {"RUN_STARTED", "RUN_FINALIZED"}
    missing = sorted(required - types)
    return {
        "schema": "nano-lm.wedge_v1.coe_record_audit.v1",
        "path": str(path),
        "n_events": len(events),
        "event_types": sorted(types),
        "ok": not missing,
        "missing_required": missing,
    }
