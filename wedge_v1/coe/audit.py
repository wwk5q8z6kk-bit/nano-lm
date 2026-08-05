"""Local CoE audit checks for internal Wedge v1 pipeline outputs (not paper-only)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from wedge_v1.coe.record import load_record
from wedge_v1.coe.schema import DerivationKind, digest_docs, digest_text

_PRESENTABLE_STATUSES = {"PRESENT", "CONFIRMED", "PROBABLE", "DISPUTED"}
_KNOWN_DERIVATIONS = {kind.value for kind in DerivationKind if kind is not DerivationKind.UNKNOWN}
_REQUIRED_SPAN_KEYS = (
    "atom_id",
    "doc_id",
    "doc_digest",
    "relation",
    "start",
    "end",
    "text",
)
_NON_SEMANTIC_KEYS = {
    "doc_id",
    "end",
    "field",
    "method",
    "query",
    "relation",
    "start",
}


def _check(name: str, ok: bool, reason: str = "", *, abstain: bool = False) -> dict:
    return {
        "check": name,
        "result": "abstain" if abstain else ("pass" if ok else "fail"),
        "reason": reason,
    }


def _semantic_fragments(value: Any) -> list[tuple[str, bool]]:
    if value is None or isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [(str(value), True)]
    if isinstance(value, str):
        normalized = " ".join(value.casefold().split())
        return [(normalized, False)] if normalized else []
    if isinstance(value, dict):
        out: list[tuple[str, bool]] = []
        for key, child in value.items():
            if str(key).casefold() not in _NON_SEMANTIC_KEYS:
                out.extend(_semantic_fragments(child))
        return out
    if isinstance(value, (list, tuple, set)):
        out = []
        for child in value:
            out.extend(_semantic_fragments(child))
        return out
    return []


def _semantic_value_supported(value: Any, atoms: list[dict]) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return bool(atoms)
    fragments = _semantic_fragments(value)
    if not fragments:
        return False
    texts = [" ".join(str(atom.get("text") or "").casefold().split()) for atom in atoms]
    texts = [text for text in texts if text]
    if not texts:
        return False
    for fragment, numeric in fragments:
        if numeric:
            pattern = re.compile(rf"(?<![\w.]){re.escape(fragment)}(?![\w.])")
            if not any(pattern.search(text) for text in texts):
                return False
        elif not any(fragment in text for text in texts):
            return False
    return True


def _passing_verification(claim: dict) -> bool:
    records = claim.get("verification") or []
    return bool(records) and all(record.get("outcome") == "pass" for record in records)


def _known_derivation(claim: dict) -> bool:
    return claim.get("derivation") in _KNOWN_DERIVATIONS


def _source_binding_valid(claim: dict, docs: dict[str, str], corpus_digest: str) -> bool:
    if claim.get("corpus_digest") != corpus_digest:
        return False
    atoms = claim.get("evidence_atoms") or []
    if not atoms:
        return False
    source_doc_ids = claim.get("source_doc_ids") or []
    atom_doc_ids = {
        atom.get("doc_id")
        for atom in atoms
        if isinstance(atom.get("doc_id"), str) and atom.get("doc_id")
    }
    valid_source_list = isinstance(source_doc_ids, list) and all(
        isinstance(doc_id, str) and doc_id for doc_id in source_doc_ids
    )
    if (
        not valid_source_list
        or set(source_doc_ids) != atom_doc_ids
        or any(doc_id not in docs for doc_id in source_doc_ids)
    ):
        return False
    for atom in atoms:
        doc_id = atom.get("doc_id")
        if not isinstance(doc_id, str) or doc_id not in docs:
            return False
        if atom.get("doc_digest") != digest_text(docs[doc_id]):
            return False
    return True


def _presentable_claim_ready(
    claim: dict,
    docs: dict[str, str] | None,
    corpus_digest: str | None,
) -> bool:
    atoms = claim.get("evidence_atoms") or []
    structurally_bound = bool(atoms) and all(
        atom.get("relation") != "UNSUPPORTED"
        and isinstance(atom.get("doc_id"), str)
        and atom.get("start") is not None
        and atom.get("end") is not None
        for atom in atoms
    )
    if not structurally_bound:
        return False
    if not _passing_verification(claim) or not _known_derivation(claim):
        return False
    if not _semantic_value_supported(claim.get("raw_value"), atoms):
        return False
    if docs is not None:
        assert corpus_digest is not None
        if not _source_binding_valid(claim, docs, corpus_digest):
            return False
    return True


def _public_evidence_rows(claim: dict) -> list[dict] | None:
    evidence = claim.get("evidence") or []
    if not isinstance(evidence, list) or any(not isinstance(row, dict) for row in evidence):
        return None
    return evidence


def _primary_claim_matches_typed(claim: dict, typed: dict) -> bool:
    if (
        str(claim.get("task_id") or "UNKNOWN") != str(typed.get("claim_type") or "UNKNOWN")
        or str(claim.get("status") or "PRESENT") != str(typed.get("status") or "PRESENT")
        or claim.get("value") != typed.get("raw_value")
    ):
        return False

    public_doc_id = claim.get("doc_id")
    if public_doc_id is not None and (
        not isinstance(public_doc_id, str)
        or not public_doc_id
        or typed.get("source_doc_ids") != [public_doc_id]
    ):
        return False

    atoms = typed.get("evidence_atoms") or []
    if not isinstance(atoms, list) or any(not isinstance(atom, dict) for atom in atoms):
        return False
    evidence = _public_evidence_rows(claim)
    if evidence is None or len(evidence) != len(atoms):
        return False
    for public, atom in zip(evidence, atoms, strict=True):
        if public.get("context"):
            return False
        if public.get("doc_id") != atom.get("doc_id"):
            return False
        if public.get("atom_id") != atom.get("atom_id"):
            return False
        if public.get("doc_digest") != atom.get("doc_digest"):
            return False
        if public.get("relation") != atom.get("relation"):
            return False
        public_text = public.get("text") or public.get("line")
        if public_text != atom.get("text"):
            return False
        for key in ("start", "end"):
            if isinstance(public.get(key), bool) or public.get(key) != atom.get(key):
                return False

    public_coe = claim.get("coe")
    if not isinstance(public_coe, dict):
        return False
    if public_coe.get("derivation") != typed.get("derivation"):
        return False
    if public_coe.get("corpus_digest") != typed.get("corpus_digest"):
        return False
    return public_coe.get("evidence_atom_ids") == [atom.get("atom_id") for atom in atoms]


def _nested_claim_rows(payload: dict) -> list[tuple[str, dict]]:
    rows: list[tuple[str, dict]] = []

    def visit(surface: str, value: Any) -> None:
        if isinstance(value, dict):
            if "claim_id" in value:
                rows.append((surface, value))
            for child in value.values():
                visit(surface, child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(surface, child)

    for key, value in payload.items():
        if key in {"claims", "coe_claims", "coe", "coe_audit"}:
            continue
        visit(key, value)
    return rows


def _span_matches_typed(span: dict, typed: dict) -> bool:
    if any(key not in span for key in _REQUIRED_SPAN_KEYS):
        return False
    atoms = typed.get("evidence_atoms") or []
    for atom in atoms:
        if not isinstance(atom, dict):
            continue
        if all(span.get(key) == atom.get(key) for key in _REQUIRED_SPAN_KEYS):
            return True
    return False


def _span_list_matches_typed(spans: Any, typed: dict) -> bool:
    atoms = typed.get("evidence_atoms") or []
    return (
        isinstance(spans, list)
        and len(spans) == len(atoms)
        and all(
            isinstance(span, dict)
            and isinstance(atom, dict)
            and all(span.get(key) == atom.get(key) for key in _REQUIRED_SPAN_KEYS)
            and all(key in span for key in _REQUIRED_SPAN_KEYS)
            for span, atom in zip(spans, atoms, strict=True)
        )
    )


def _nested_fact_matches_typed(surface: str, row: dict, typed: dict) -> bool:
    raw_value = typed.get("raw_value")
    status = typed.get("status")

    if surface in {"contradictions_nearby", "contradictions_corpus"}:
        if row.get("status") != status:
            return False
        if isinstance(raw_value, dict):
            if "field" in raw_value and row.get("field") != raw_value.get("field"):
                return False
            if "values" in raw_value:
                return row.get("values") == raw_value.get("values")
            if "docs" in raw_value:
                return row.get("values") == raw_value.get("docs")
        return "value" in row and row.get("value") == raw_value

    if surface == "epistemic_merge":
        if not isinstance(raw_value, dict) or not isinstance(raw_value.get("values"), dict):
            return False
        values = raw_value["values"]
        return (
            row.get("field_id") == raw_value.get("field")
            and row.get("status") == status
            and row.get("values_by_doc") == values
            and row.get("unique_values")
            == sorted(set(values.values()), key=lambda item: str(item))
            and row.get("disputed") is (status == "DISPUTED")
        )

    if surface == "hits":
        return row.get("status") == status and row.get("value") == raw_value

    for row_key, typed_key in (
        ("status", "status"),
        ("field", "field"),
        ("field_id", "field"),
    ):
        if row_key in row:
            expected = typed.get(typed_key) if typed_key == "status" else (
                raw_value.get(typed_key) if isinstance(raw_value, dict) else None
            )
            if row.get(row_key) != expected:
                return False
    return True


def _claim_binding_problems(payload: dict, claims: list, coe_claims: list) -> dict[str, int]:
    malformed = sum(not isinstance(claim, dict) for claim in [*claims, *coe_claims])
    public = [claim for claim in claims if isinstance(claim, dict)]
    typed = [claim for claim in coe_claims if isinstance(claim, dict)]
    public_ids = [claim.get("claim_id") for claim in public]
    typed_ids = [claim.get("claim_id") for claim in typed]
    missing_ids = sum(not isinstance(claim_id, str) or not claim_id for claim_id in [*public_ids, *typed_ids])
    valid_public_ids = [
        claim_id for claim_id in public_ids if isinstance(claim_id, str) and claim_id
    ]
    valid_typed_ids = [
        claim_id for claim_id in typed_ids if isinstance(claim_id, str) and claim_id
    ]
    duplicate_ids = (
        len(valid_public_ids) - len(set(valid_public_ids))
        + len(valid_typed_ids) - len(set(valid_typed_ids))
    )
    typed_by_id = {
        claim["claim_id"]: claim
        for claim in typed
        if isinstance(claim.get("claim_id"), str) and claim.get("claim_id")
    }
    public_id_set = set(valid_public_ids)
    typed_id_set = set(typed_by_id)
    coverage = len(public_id_set.symmetric_difference(typed_id_set))
    identity_mismatch = 0
    for claim in public:
        claim_id = claim.get("claim_id")
        if (
            not isinstance(claim_id, str)
            or not claim_id
            or claim_id not in typed_by_id
            or not _primary_claim_matches_typed(claim, typed_by_id[claim_id])
        ):
            identity_mismatch += 1

    dangling_nested = 0
    mismatched_nested_evidence = 0
    mismatched_nested_fact = 0
    for surface, row in _nested_claim_rows(payload):
        claim_id = row.get("claim_id")
        if (
            not isinstance(claim_id, str)
            or not claim_id
            or claim_id not in typed_by_id
            or claim_id not in public_id_set
        ):
            dangling_nested += 1
            continue
        typed_claim = typed_by_id[claim_id]
        if not _nested_fact_matches_typed(surface, row, typed_claim):
            mismatched_nested_fact += 1
        nested_spans = row.get("evidence_spans")
        if "evidence_spans" in row and not _span_list_matches_typed(
            nested_spans, typed_claim
        ):
            mismatched_nested_evidence += 1
        if (
            surface in {"hits", "evidence_spans"}
            or any(key in row for key in _REQUIRED_SPAN_KEYS)
        ) and not _span_matches_typed(row, typed_claim):
            mismatched_nested_evidence += 1

    return {
        "malformed": malformed,
        "missing_ids": missing_ids,
        "duplicate_ids": duplicate_ids,
        "coverage": coverage,
        "identity_mismatch": identity_mismatch,
        "dangling_nested": dangling_nested,
        "mismatched_nested_fact": mismatched_nested_fact,
        "mismatched_nested_evidence": mismatched_nested_evidence,
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
    live_corpus_digest = digest_docs(docs) if docs is not None else None
    if docs is None:
        checks.append(_check("evidence_existence", True, "docs not provided", abstain=True))
        checks.append(_check("offset_validity", True, "docs not provided", abstain=True))
        checks.append(_check("source_version_binding", True, "docs not provided", abstain=True))
    else:
        for tc in coe_claims:
            for atom in tc.get("evidence_atoms") or []:
                did = atom.get("doc_id")
                if not isinstance(did, str) or not did or did not in docs:
                    missing_src += 1
                    continue
                body = docs[did]
                if atom.get("relation") == "UNSUPPORTED":
                    unsupported += 1
                    continue
                start, end, text = atom.get("start"), atom.get("end"), atom.get("text") or ""
                if isinstance(start, bool) or isinstance(end, bool):
                    bad_offset += 1
                    continue
                try:
                    start_i, end_i = int(start), int(end)
                except (TypeError, ValueError):
                    bad_offset += 1
                    continue
                if (
                    start_i < 0
                    or end_i <= start_i
                    or end_i > len(body)
                    or not text
                    or body[start_i:end_i] != text
                ):
                    bad_offset += 1
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
                bad_offset == 0 and unsupported == 0,
                f"invalid_offsets={bad_offset} unsupported_atoms={unsupported}",
            )
        )

        stale_corpus = int((payload.get("coe") or {}).get("corpus_digest") != live_corpus_digest)
        stale_claim_corpus = 0
        stale_docs = 0
        missing_doc_digests = 0
        invalid_source_sets = 0
        for tc in coe_claims:
            if tc.get("corpus_digest") != live_corpus_digest:
                stale_claim_corpus += 1
            source_doc_ids = tc.get("source_doc_ids") or []
            atom_doc_ids = {
                atom.get("doc_id")
                for atom in tc.get("evidence_atoms") or []
                if isinstance(atom.get("doc_id"), str) and atom.get("doc_id")
            }
            valid_source_list = isinstance(source_doc_ids, list) and all(
                isinstance(doc_id, str) and doc_id for doc_id in source_doc_ids
            )
            if (
                not valid_source_list
                or set(source_doc_ids) != atom_doc_ids
                or any(doc_id not in docs for doc_id in source_doc_ids)
            ):
                invalid_source_sets += 1
            for atom in tc.get("evidence_atoms") or []:
                doc_id = atom.get("doc_id")
                if not isinstance(doc_id, str) or doc_id not in docs:
                    continue
                atom_digest = atom.get("doc_digest")
                if not atom_digest:
                    missing_doc_digests += 1
                elif atom_digest != digest_text(docs[doc_id]):
                    stale_docs += 1
        checks.append(
            _check(
                "source_version_binding",
                stale_corpus == 0
                and stale_claim_corpus == 0
                and stale_docs == 0
                and missing_doc_digests == 0
                and invalid_source_sets == 0,
                (
                    f"payload_corpus_mismatch={stale_corpus} "
                    f"claim_corpus_mismatch={stale_claim_corpus} "
                    f"document_digest_mismatch={stale_docs} "
                    f"missing_document_digest={missing_doc_digests} "
                    f"invalid_source_sets={invalid_source_sets}"
                ),
            )
        )

    # Every claim eligible for presentation needs bound atoms, a known
    # derivation, a passing verifier outcome, and value/evidence agreement.
    present_without = 0
    bad_verifier = 0
    unknown_derivation = 0
    semantic_mismatch = 0
    if coe_claims:
        for tc in coe_claims:
            if tc.get("status") not in _PRESENTABLE_STATUSES:
                continue
            atoms = tc.get("evidence_atoms") or []
            if not _passing_verification(tc):
                bad_verifier += 1
            if not _known_derivation(tc):
                unknown_derivation += 1
            if not _semantic_value_supported(tc.get("raw_value"), atoms):
                semantic_mismatch += 1
            if not _presentable_claim_ready(tc, docs, live_corpus_digest):
                present_without += 1
    else:
        for c in claims:
            if c.get("status") in _PRESENTABLE_STATUSES:
                present_without += 1
                bad_verifier += 1
                unknown_derivation += 1
                semantic_mismatch += 1
    checks.append(
        _check(
            "claim_support",
            present_without == 0,
            f"present_unverified={present_without}",
        )
    )
    checks.append(
        _check(
            "verifier_outcome",
            bad_verifier == 0,
            f"present_without_passing_verifier={bad_verifier}",
        )
    )
    checks.append(
        _check(
            "derivation_known",
            unknown_derivation == 0,
            f"present_with_unknown_derivation={unknown_derivation}",
        )
    )
    checks.append(
        _check(
            "semantic_value_alignment",
            semantic_mismatch == 0,
            f"present_with_value_evidence_mismatch={semantic_mismatch}",
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

    # Citation faithfulness: every public fact must retain its exact typed claim,
    # atom identity, and nested reference linkage.
    binding_problems = _claim_binding_problems(payload, claims, coe_claims)
    bound = not any(binding_problems.values())
    inv = (payload.get("coe") or {}).get("invariant") == "EVIDENCE_CREATED_WITH_CLAIM"
    binding_ok = bound and (inv or not claims)
    binding_reason = " ".join(
        f"{name}={count}" for name, count in binding_problems.items()
    )
    checks.append(
        _check(
            "citation_faithfulness_binding",
            binding_ok,
            binding_reason if not binding_ok else "ok",
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
        "source_version_binding": "COE_STALE_SOURCE_VERSION",
        "claim_support": "COE_UNSUPPORTED_PREDICATE",
        "verifier_outcome": "COE_UNSUPPORTED_PREDICATE",
        "derivation_known": "COE_DERIVATION_UNKNOWN",
        "semantic_value_alignment": "COE_UNSUPPORTED_PREDICATE",
        "trace_completeness": "COE_CONFIG_MISSING",
        "method_execution_alignment": "COE_METHOD_TRACE_MISMATCH",
        "spec_classical_only": "COE_UNREGISTERED_SOLVER",
        "citation_faithfulness_binding": "COE_POSTHOC_CITATION",
        "contradiction_not_ignored": "COE_CONTRADICTION_IGNORED",
        "complete_conjunction": "COE_INCOMPLETE_CONJUNCTION",
    }
    return list(dict.fromkeys(m.get(c["check"], "COE_DERIVATION_UNKNOWN") for c in fails))


def audit_record(path: Path) -> dict:
    try:
        events = load_record(path)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "schema": "nano-lm.wedge_v1.coe_record_audit.v1",
            "path": str(path),
            "n_events": 0,
            "event_types": [],
            "ok": False,
            "missing_required": ["RUN_FINALIZED", "RUN_STARTED"],
            "problems": [f"record_unreadable:{type(exc).__name__}"],
        }

    well_formed_events = all(isinstance(event, dict) for event in events)
    types = {
        event.get("event_type")
        for event in events
        if isinstance(event, dict) and isinstance(event.get("event_type"), str)
    }
    required = {"RUN_STARTED", "RUN_FINALIZED"}
    missing = sorted(required - types)
    run_ids = [event.get("run_id") for event in events if isinstance(event, dict)]
    unique_run_ids = {
        run_id for run_id in run_ids if isinstance(run_id, str) and run_id
    }
    one_run_id = (
        well_formed_events
        and len(unique_run_ids) == 1
        and len(run_ids) == len(events)
        and all(run_id in unique_run_ids for run_id in run_ids)
    )

    start_indices = [
        index for index, event in enumerate(events)
        if isinstance(event, dict) and event.get("event_type") == "RUN_STARTED"
    ]
    final_indices = [
        index for index, event in enumerate(events)
        if isinstance(event, dict) and event.get("event_type") == "RUN_FINALIZED"
    ]
    exact_boundaries = (
        len(start_indices) == 1
        and len(final_indices) == 1
        and start_indices[0] == 0
        and final_indices[0] == len(events) - 1
    )

    seen_event_ids: set[str] = set()
    valid_parent_references = well_formed_events
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            valid_parent_references = False
            continue
        event_id = event.get("event_id")
        parents = event.get("parent_ids")
        if (
            not isinstance(event_id, str)
            or not event_id
            or event_id in seen_event_ids
            or not isinstance(parents, list)
            or (index == 0 and parents != [])
            or (index > 0 and not parents)
            or any(not isinstance(parent, str) for parent in parents)
            or (isinstance(parents, list) and len(parents) != len(set(parents)))
            or any(parent not in seen_event_ids for parent in parents)
        ):
            valid_parent_references = False
        if isinstance(event_id, str) and event_id:
            seen_event_ids.add(event_id)

    final_count_matches = False
    if len(final_indices) == 1:
        final_payload = events[final_indices[0]].get("payload") or {}
        declared_count = final_payload.get("n_events") if isinstance(final_payload, dict) else None
        final_count_matches = (
            isinstance(declared_count, int)
            and not isinstance(declared_count, bool)
            and declared_count == len(events) - 1
        )

    start_run_matches = False
    if len(start_indices) == 1 and len(unique_run_ids) == 1:
        start_payload = events[start_indices[0]].get("payload") or {}
        start_run_matches = (
            isinstance(start_payload, dict)
            and start_payload.get("run_id") == next(iter(unique_run_ids))
        )

    problems = []
    if not well_formed_events:
        problems.append("malformed_event")
    if not one_run_id:
        problems.append("mixed_or_missing_run_id")
    if not exact_boundaries:
        problems.append("invalid_start_final_order_or_count")
    if not valid_parent_references:
        problems.append("invalid_parent_reference")
    if not final_count_matches:
        problems.append("final_event_count_mismatch")
    if not start_run_matches:
        problems.append("start_run_id_mismatch")
    return {
        "schema": "nano-lm.wedge_v1.coe_record_audit.v1",
        "path": str(path),
        "n_events": len(events),
        "event_types": sorted(types),
        "run_id": next(iter(unique_run_ids)) if len(unique_run_ids) == 1 else None,
        "ok": not missing and not problems,
        "missing_required": missing,
        "problems": problems,
    }
