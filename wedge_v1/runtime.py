"""Internal Wedge v1 evidence and validation pipeline.

The supporting runtime uses classical and E-class solvers, verifies evidence
spans, and abstains when its supported methods cannot answer safely. It is not
the Nano AI core and does not implement Nano's scribe inference path.
"""
from __future__ import annotations

from copy import deepcopy
import json
import re
from dataclasses import asdict
from pathlib import Path

import time

from wedge_v1.classical import solvers as S
from wedge_v1.classical.bm25 import top_paragraphs
from wedge_v1.arch.failure_codes import FailureCode
from wedge_v1.arch.trace import AskTrace, classify_abstain_failures
from wedge_v1.classical.verifier import verify_claim
from wedge_v1.classical.merge import (
    epistemic_entry,
    merge_all,
    merge_for_term,
    predicate_claims_for_domains,
)
from wedge_v1.plugins.cascade import run_cascade
from wedge_v1.plugins.lexicon import synonyms as _synonym_map
from wedge_v1.coe.predicates import (
    decompose,
    evaluate_predicates,
    incomplete_conjunction,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "data" / "corpus"
GOLD_PATH = ROOT / "data" / "gold" / "gold.json"



def load_corpus(corpus_dir: Path | None = None, *, normalize: bool | str = "auto") -> dict[str, str]:
    from wedge_v1.ingest import load_corpus as _ingest

    path = Path(corpus_dir) if corpus_dir else DEFAULT_CORPUS
    return _ingest(path, normalize=normalize)


def normalize_doc_ids(doc_ids: list[str] | None) -> list[str] | None:
    """Return the canonical exact-document scope used by every public surface."""
    if doc_ids is None:
        return None
    return sorted({str(doc_id).strip() for doc_id in doc_ids if str(doc_id).strip()})


def select_documents(
    docs: dict[str, str],
    doc_ids: list[str] | None,
) -> tuple[dict[str, str], dict, bool]:
    """Apply an exact scope without ever falling back to the full corpus."""
    normalized = normalize_doc_ids(doc_ids)
    if normalized is None:
        return docs, {}, True

    selected_ids = [doc_id for doc_id in normalized if doc_id in docs]
    missing_ids = [doc_id for doc_id in normalized if doc_id not in docs]
    scope = {
        "selected_doc_ids": selected_ids,
        "missing_doc_ids": missing_ids,
    }
    valid = bool(normalized) and not missing_ids
    if not valid:
        return {}, scope, False
    return {doc_id: docs[doc_id] for doc_id in selected_ids}, scope, True


def _scope_abstention(
    *,
    op: str,
    query: str,
    corpus_path: Path,
    scope: dict,
    persist: bool,
    extra: dict | None = None,
) -> dict:
    """Fail closed before retrieval when an explicit document scope is invalid."""
    missing = list(scope.get("missing_doc_ids") or [])
    code = "UNKNOWN_DOCUMENT_ID" if missing else "EMPTY_DOCUMENT_SCOPE"
    note = (
        "one or more exact document IDs were not found"
        if missing
        else "explicit document scope is empty"
    )
    trace = AskTrace(query=query, corpus_dir=str(corpus_path), op=op)
    trace.add_solver("document_scope")
    trace.n_docs = 0
    trace.event("document_scope", "selection rejected", **scope)
    trace_payload = trace.finalize("ABSTAIN")
    trace_payload["failure_codes"] = list(
        dict.fromkeys([*(trace_payload.get("failure_codes") or []), code])
    )
    payload = {
        "query": query,
        "corpus_dir": str(corpus_path),
        "answer_status": "ABSTAIN",
        "claims": [],
        "coe_claims": [],
        "unsupported": [note],
        "note": note,
        "solver_path": ["document_scope"],
        "failure_codes": [code],
        "trace": trace_payload,
        "n_docs": 0,
        "n_claims_presented": 0,
        "lm_invoked": False,
        **scope,
    }
    if extra:
        payload.update(extra)
    return _finalize_public_payload(
        payload,
        {},
        op=op,
        query=query,
        persist=persist,
        scope=scope,
    )


def _load_gold() -> dict | None:
    if GOLD_PATH.exists():
        return json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    return None


def claim_to_dict(c: S.Claim) -> dict:
    return asdict(c)


STOP = {
    "how", "long", "before", "what", "when", "where", "which", "does", "the", "and",
    "for", "is", "of", "in", "a", "an", "to", "on", "at", "by", "or", "as", "be",
    "are", "was", "were", "with", "from", "this", "that", "these", "those", "into",
    "will", "did", "under", "about", "into", "over", "than", "then", "also", "only",
}


def _content_tokens(q: str) -> list[str]:
    words = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", q) if t.lower() not in STOP]
    raw_nums = re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?(?![A-Za-z])", q)
    # Drop digits that only appear inside hyphenated tokens (e.g. GPT-4)
    hyphen_chunks = set(re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+", q))
    nums = []
    for n in raw_nums:
        if any(n in chunk and chunk != n for chunk in hyphen_chunks):
            continue
        nums.append(n)
    # preserve order, unique
    out, seen = [], set()
    for t in words + nums:
        k = t.lower()
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out


def _token_equivalence_groups(tokens: list[str]) -> list[set[str]]:
    """Lexical synonym groups for relevance (W4 expand map; not LM)."""
    syn = _synonym_map()
    groups: list[set[str]] = []
    for tok in tokens:
        low = tok.lower()
        group = {low}
        for src, dsts in syn.items():
            src_l = src.lower()
            dst_l = {d.lower() for d in dsts}
            if low == src_l or low in dst_l or src_l in low or low in src_l:
                group.add(src_l)
                group.update(dst_l)
                group.update(re.findall(r"[a-z0-9]+", src_l))
        groups.append(group)
    return groups


def _relevant_claim(
    c: S.Claim, tokens: list[str], docs: dict[str, str] | None = None
) -> bool:
    """Reject weak lexical coincidences (one stop-ish content token in a huge corpus).

    Two-scope relevance (W-ABSTAIN-1, papers/PREREG_ABSTENTION_W1.md). A claim is
    relevant when the source DOCUMENT carries every query token -- which preserves
    the anti-false-positive property this filter exists for -- and the presented
    SPAN carries a majority of them, which keeps the answer anchored in the text
    actually shown. Requiring every token inside one ~240-char span was the
    measured cause of over-abstention: natural queries put the subject in a
    heading and the predicate in the answering sentence.

    `docs` is optional so existing callers and tests keep working; without it the
    document scope is skipped and the original span-only rule applies.
    """
    if not tokens:
        return False
    blob = json.dumps(c.value, default=str).lower() + " " + " ".join(
        str(e.get("text", "")) + " " + str(e.get("line", "")) for e in (c.evidence or [])
    ).lower()
    # E-class structural probes: accepted when triggered
    if c.task_id in {"T35", "T36", "T39", "T26", "T29", "T30", "MERGE"} and c.status in {
        "PRESENT", "CONFIRMED", "DISPUTED"
    }:
        return True
    # Exact FIND of a query number/literal: value itself must be one of the tokens
    if c.task_id == "FIND":
        # A single token hit is not an answer to a multi-token question.
        # Require the same coverage rule as passage claims (falls through).
        pass
    groups = _token_equivalence_groups(tokens)
    hits = sum(1 for group in groups if any(t in blob for t in group))
    if len(tokens) < 3:
        # Short queries keep the original rule unchanged.
        return hits >= (2 if len(tokens) == 2 else 1)

    # Span scope: a majority of query tokens must appear in what is presented.
    span_need = max(2, (len(tokens) + 1) // 2)
    if hits < span_need:
        return False

    # Document scope: the source document must carry every query token.
    if docs is None:
        return hits >= len(tokens)
    body = (docs.get(c.doc_id) or "").lower()
    if not body:
        return hits >= len(tokens)
    return all(any(t.lower() in body for t in group) for group in groups)



def find_spans(
    needle: str,
    corpus_dir: Path | None = None,
    max_hits: int = 20,
    *,
    doc_ids: list[str] | None = None,
    persist_coe: bool = True,
) -> dict:
    """Exact substring locate with evidence spans (classical)."""
    corpus_path = Path(corpus_dir) if corpus_dir else DEFAULT_CORPUS
    docs, scope, scope_valid = select_documents(load_corpus(corpus_dir), doc_ids)
    if not scope_valid:
        return _scope_abstention(
            op="find",
            query=needle.strip(),
            corpus_path=corpus_path,
            scope=scope,
            persist=persist_coe,
        )
    if not docs:
        return {"answer_status": "NO_CORPUS", "claims": [], "solver_path": ["load_corpus"]}
    needle = needle.strip()
    if not needle:
        return _finalize_public_payload(
            {
                "query": needle,
                "corpus_dir": str(corpus_path),
                "answer_status": "ABSTAIN",
                "claims": [],
                "unsupported": ["empty needle"],
                "solver_path": ["find_spans"],
                "n_docs": len(docs),
            },
            docs,
            candidates=[],
            op="find",
            query=needle,
            persist=persist_coe,
            scope=scope,
        )
    claims = []
    for did, text in docs.items():
        start = 0
        while True:
            i = text.find(needle, start)
            if i < 0:
                break
            # window for context
            lo = max(0, i - 40)
            hi = min(len(text), i + len(needle) + 40)
            ctx = text[lo:hi].replace("\n", " ")
            claims.append(
                S.Claim(
                    "FIND",
                    did,
                    needle,
                    evidence=[{"start": i, "end": i + len(needle), "text": text[i : i + len(needle)], "context": ctx}],
                    status="PRESENT",
                    notes="exact_span",
                )
            )
            start = i + max(1, len(needle))
            if len(claims) >= max_hits:
                break
        if len(claims) >= max_hits:
            break
    if not claims:
        status = "ABSTAIN"
        unsupported = [needle]
    else:
        status = "SUPPORTED"
        unsupported = []
    return _finalize_public_payload(
        {
            "query": needle,
            "corpus_dir": str(corpus_path),
            "answer_status": status,
            "claims": [],
            "unsupported": unsupported,
            "solver_path": ["find_spans"],
            "n_docs": len(docs),
            "n_hits": len(claims),
        },
        docs,
        candidates=claims,
        op="find",
        query=needle,
        persist=persist_coe,
        scope=scope,
    )



def nearby_contradictions(docs: dict[str, str]) -> list[dict]:
    """Light contradiction surface for ask() banners (W3 merge; classical only)."""
    out: list[dict] = []
    kind_map = {
        "ttl_seconds": "numeric_ttl",
        "metformin_dose_mg": "numeric_dose",
        "sample_n": "numeric_sample_n",
    }
    for claim in merge_all(docs):
        if claim.status != "DISPUTED" or not isinstance(claim.value, dict):
            continue
        field = claim.value.get("field") or "unknown"
        out.append(
            {
                "kind": kind_map.get(field, f"numeric_{field}"),
                "field": field,
                "values": claim.value.get("values", {}),
                "status": "DISPUTED",
                "evidence_spans": epistemic_entry(claim).get("evidence_spans") or [],
            }
        )
    coll = S.flag_entity_collision(docs)
    if coll.status == "DISPUTED":
        out.append({"kind": "entity_collision", "value": coll.value, "status": "DISPUTED"})
    return out


def _candidate_contradictions(claims: list[S.Claim]) -> list[dict]:
    """Build temporary display rows from claims that will enter the CoE boundary."""
    kind_map = {
        "ttl_seconds": "numeric_ttl",
        "metformin_dose_mg": "numeric_dose",
        "sample_n": "numeric_sample_n",
    }
    rows = []
    for claim in claims:
        if claim.status != "DISPUTED" or not isinstance(claim.value, dict):
            continue
        field = claim.value.get("field")
        rows.append(
            {
                "kind": kind_map.get(str(field), f"numeric_{field}") if field else "disputed_claim",
                "field": field,
                "values": claim.value.get("values") or claim.value.get("docs"),
                "status": "DISPUTED",
                "evidence_spans": _bound_evidence_spans(claim_to_dict(claim)),
            }
        )
    return rows



def _has_evidence_atom(c: S.Claim) -> bool:
    """W2: presentable claims need at least one evidence atom with text or offsets."""
    for e in c.evidence or []:
        if not isinstance(e, dict):
            continue
        if e.get("text") or e.get("line") or (e.get("start") is not None and e.get("end") is not None):
            return True
    return False


def _bm25_claims(docs: dict[str, str], q: str, k: int = 5) -> tuple[list[S.Claim], list[dict]]:
    """W1: promote only margin-gated BM25 hits; return (present_claims, review_hits)."""
    present: list[S.Claim] = []
    review: list[dict] = []
    for hit in top_paragraphs(docs, q, k=k):
        evidence_text = hit["text"][:240]
        evidence_start = int(hit["start"])
        body = docs.get(hit["doc_id"], "")
        if body[evidence_start : evidence_start + len(evidence_text)] != evidence_text:
            evidence_start = body.find(evidence_text, evidence_start, int(hit["end"]))
        evidence_end = evidence_start + len(evidence_text) if evidence_start >= 0 else int(hit["end"])
        ev = [{
            "start": evidence_start,
            "end": evidence_end,
            "text": evidence_text,
            "bm25": hit["bm25"],
            "margin": hit.get("margin"),
            "rank": hit.get("rank"),
        }]
        if hit.get("promote"):
            present.append(
                S.Claim(
                    "T19",
                    hit["doc_id"],
                    hit["text"][:500],
                    evidence=ev,
                    status="PRESENT",
                    notes="bm25_paragraph",
                )
            )
        else:
            # REVIEW rows are retrieval diagnostics, not evidence-bearing facts.
            # Keep only non-content scoring metadata; raw source text, source IDs,
            # and offsets may surface only after promotion into a bound claim.
            review.append(
                {
                    "bm25": hit["bm25"],
                    "top2_bm25": hit.get("top2_bm25"),
                    "margin": hit.get("margin"),
                    "rank": hit.get("rank"),
                    "promote": False,
                    "status": "REVIEW",
                    "reason": "low_bm25_margin",
                }
            )
    return present, review



def _multi_fact_domains(q: str) -> list[str]:
    """Detect distinct fact domains in a conjunction-style query."""
    ql = q.lower()
    found = []
    if any(k in ql for k in ("ttl", "cache", "expire", "cached", "timeout")):
        found.append("ttl_cache")
    if any(k in ql for k in ("dose", "metformin", "mg")):
        found.append("dose")
    if any(k in ql for k in ("author", "title", "year", "doi")):
        found.append("biblio")
    return found


def _is_unsupported_composition_query(q: str) -> bool:
    ql = q.lower()
    if " and " not in ql and ";" not in ql:
        return False
    return len(_multi_fact_domains(q)) >= 2


def _corpus_has_lexical(docs: dict[str, str], tokens: list[str]) -> bool:
    if not tokens:
        return False
    blob = "\n".join(docs.values()).lower()
    return any(tok.lower() in blob for tok in tokens if len(tok) >= 3)


def _claim_covers_domain(claims: list, domain: str) -> bool:
    blob = " ".join(
        f"{getattr(c, 'value', '')} {getattr(c, 'notes', '')} {getattr(c, 'task_id', '')}".lower()
        for c in claims
    )
    if domain == "ttl_cache":
        return any(k in blob for k in ("ttl", "seconds", "cache", "300", "timeout"))
    if domain == "dose":
        return any(k in blob for k in ("metformin", "mg", "dose", "500", "850"))
    if domain == "biblio":
        return any(k in blob for k in ("author", "title", "year", "doi"))
    return False



_COE_EVIDENCE_CHECKS = {
    "evidence_existence",
    "offset_validity",
    "source_version_binding",
    "claim_support",
    "verifier_outcome",
    "derivation_known",
    "semantic_value_alignment",
    "citation_faithfulness_binding",
}
_COE_PRESENTATION_CHECKS = _COE_EVIDENCE_CHECKS | {
    "contradiction_not_ignored",
    "complete_conjunction",
}
_PRESENTABLE_STATUSES = {"PRESENT", "CONFIRMED", "PROBABLE", "DISPUTED"}
_FACTUAL_LIST_SURFACES = {
    "contradictions_nearby",
    "contradictions_corpus",
    "epistemic_merge",
    "hits",
}
_FACTUAL_MAP_SURFACES = {"values_by_doc", "field_values"}
_COE_CHECK_FAILURE_CODES = {
    "evidence_existence": "COE_MISSING_SOURCE",
    "offset_validity": "COE_INVALID_OFFSET",
    "source_version_binding": "COE_STALE_SOURCE_VERSION",
    "claim_support": "COE_UNSUPPORTED_PREDICATE",
    "verifier_outcome": "COE_UNSUPPORTED_PREDICATE",
    "derivation_known": "COE_DERIVATION_UNKNOWN",
    "semantic_value_alignment": "COE_UNSUPPORTED_PREDICATE",
    "citation_faithfulness_binding": "COE_POSTHOC_CITATION",
    "contradiction_not_ignored": "COE_CONTRADICTION_IGNORED",
    "complete_conjunction": "COE_INCOMPLETE_CONJUNCTION",
}
_COE_AUDIT_SCHEMA = "nano-lm.wedge_v1.coe_audit.v1"


def _validated_coe_audit(audit: object) -> dict:
    """Require the complete v1 audit shape before it becomes authoritative."""
    if not isinstance(audit, dict):
        raise TypeError("final CoE audit did not return a mapping")
    if audit.get("schema") != _COE_AUDIT_SCHEMA:
        raise ValueError("final CoE audit schema is missing or invalid")
    if type(audit.get("ok")) is not bool:
        raise ValueError("final CoE audit ok flag is missing or invalid")

    count_keys = ("n_checks", "n_fail", "n_pass", "n_abstain")
    if any(type(audit.get(key)) is not int or audit[key] < 0 for key in count_keys):
        raise ValueError("final CoE audit counts are missing or invalid")
    checks = audit.get("checks")
    if not isinstance(checks, list) or any(
        not isinstance(check, dict)
        or not isinstance(check.get("check"), str)
        or check.get("result") not in {"pass", "fail", "abstain"}
        or not isinstance(check.get("reason"), str)
        for check in checks
    ):
        raise ValueError("final CoE audit checks are missing or invalid")
    failure_codes = audit.get("failure_codes")
    if not isinstance(failure_codes, list) or any(
        not isinstance(code, str) for code in failure_codes
    ):
        raise ValueError("final CoE audit failure_codes are missing or invalid")
    observed = {
        result: sum(check["result"] == result for check in checks)
        for result in ("pass", "fail", "abstain")
    }
    if (
        audit["n_checks"] != len(checks)
        or audit["n_pass"] != observed["pass"]
        or audit["n_fail"] != observed["fail"]
        or audit["n_abstain"] != observed["abstain"]
        or audit["ok"] is not (audit["n_fail"] == 0)
    ):
        raise ValueError("final CoE audit counts or outcome are inconsistent")
    return audit


def _fail_closed_coe(
    payload: dict,
    *,
    docs: dict[str, str],
    reason: str,
    failure_codes: list[str],
    audit: dict | None = None,
) -> dict:
    """Remove unverified facts, then independently audit the final abstention."""
    rejected_claim_count = len(payload.get("claims") or [])
    codes = list(dict.fromkeys([*(payload.get("failure_codes") or []), *failure_codes]))
    payload["answer_status"] = "ABSTAIN"
    payload["claims"] = []
    payload["coe_claims"] = []
    for key in _FACTUAL_LIST_SURFACES:
        if key in payload:
            payload[key] = []
    for key in _FACTUAL_MAP_SURFACES:
        if key in payload:
            payload[key] = {}
    if "contradiction_banner" in payload:
        payload["contradiction_banner"] = None
    payload["abstain_class"] = "coe_verification_failure"
    payload["failure_codes"] = codes
    payload["note"] = "claim presentation blocked: source evidence could not be verified"
    unsupported = list(payload.get("unsupported") or [])
    if reason not in unsupported:
        unsupported.append(reason)
    payload["unsupported"] = unsupported

    coe = payload.setdefault("coe", {})
    coe.update(
        {
            "invariant": "EVIDENCE_CREATED_WITH_CLAIM",
            "completeness": False,
            "rejected_claim_count": rejected_claim_count,
            "verification_error": reason,
            "failure_codes": failure_codes,
        }
    )
    if audit is not None:
        payload["coe_rejection_audit"] = deepcopy(audit)
    payload.pop("coe_audit", None)

    trace = payload.get("trace")
    if isinstance(trace, dict):
        trace["answer_status"] = "ABSTAIN"
        trace["n_claims_presented"] = 0
        trace["failure_codes"] = list(
            dict.fromkeys([*(trace.get("failure_codes") or []), *failure_codes])
        )
        events = trace.setdefault("events", [])
        if isinstance(events, list):
            events.append(
                {
                    "stage": "coe_verification",
                    "detail": "claim presentation blocked",
                    "meta": {"failure_codes": failure_codes},
                }
            )

    try:
        from wedge_v1.coe.audit import audit_payload

        final_audit = _validated_coe_audit(audit_payload(payload, docs))
    except Exception as exc:
        final_audit_error = f"{type(exc).__name__}: {exc}"
        final_audit = {
            "schema": _COE_AUDIT_SCHEMA,
            "n_checks": 1,
            "n_fail": 1,
            "n_pass": 0,
            "n_abstain": 0,
            "ok": False,
            "checks": [
                {
                    "check": "final_payload_audit",
                    "result": "fail",
                    "reason": final_audit_error,
                }
            ],
            "failure_codes": ["COE_CONFIG_MISSING"],
            "run_id": coe.get("run_id"),
        }
        coe["final_audit_error"] = final_audit_error

    payload["coe_audit"] = final_audit
    if final_audit.get("ok") is not True:
        final_codes = list(final_audit.get("failure_codes") or [])
        payload["failure_codes"] = list(
            dict.fromkeys([*(payload.get("failure_codes") or []), *final_codes])
        )
        coe["failure_codes"] = list(
            dict.fromkeys([*(coe.get("failure_codes") or []), *final_codes])
        )
        if isinstance(trace, dict):
            trace["failure_codes"] = list(
                dict.fromkeys([*(trace.get("failure_codes") or []), *final_codes])
            )
    return payload


def _finalize_with_coe(payload: dict, docs: dict[str, str], *, persist: bool = True) -> dict:
    """Bind and audit evidence before any claim is eligible for presentation."""
    pristine = deepcopy(payload)
    try:
        from wedge_v1.coe.audit import audit_payload
        from wedge_v1.coe.bind import bind_ask_payload
    except Exception as exc:  # pragma: no cover - import failures are environment-dependent
        return _fail_closed_coe(
            pristine,
            docs=docs,
            reason=f"CoE verifier unavailable: {type(exc).__name__}: {exc}",
            failure_codes=["COE_CONFIG_MISSING"],
        )

    persistence_error = None
    try:
        bound = bind_ask_payload(deepcopy(pristine), docs, persist=persist)
    except Exception as exc:
        if not persist:
            return _fail_closed_coe(
                pristine,
                docs=docs,
                reason=f"CoE binding failed: {type(exc).__name__}: {exc}",
                failure_codes=["COE_CONFIG_MISSING"],
            )
        persistence_error = f"{type(exc).__name__}: {exc}"
        try:
            bound = bind_ask_payload(deepcopy(pristine), docs, persist=False)
        except Exception as bind_exc:
            return _fail_closed_coe(
                pristine,
                docs=docs,
                reason=f"CoE binding failed: {type(bind_exc).__name__}: {bind_exc}",
                failure_codes=["COE_CONFIG_MISSING"],
            )

    if persistence_error is not None:
        bound.setdefault("coe", {})["persistence"] = "UNAVAILABLE"
        bound["coe"]["persistence_error"] = persistence_error

    try:
        audit = audit_payload(bound, docs)
    except Exception as exc:
        return _fail_closed_coe(
            bound,
            docs=docs,
            reason=f"CoE audit failed: {type(exc).__name__}: {exc}",
            failure_codes=["COE_CONFIG_MISSING"],
        )
    bound["coe_audit"] = audit

    checks_to_enforce = set(_COE_EVIDENCE_CHECKS)
    if bound.get("answer_status") in {"SUPPORTED", "CONTRADICTED"}:
        checks_to_enforce.update(_COE_PRESENTATION_CHECKS)
    failed_checks = [
        c["check"]
        for c in audit.get("checks") or []
        if c.get("result") == "fail" and c.get("check") in checks_to_enforce
    ]
    if not (bound.get("coe") or {}).get("completeness", False) or failed_checks:
        failure_codes = [
            _COE_CHECK_FAILURE_CODES[name]
            for name in failed_checks
            if name in _COE_CHECK_FAILURE_CODES
        ]
        if not failure_codes:
            failure_codes = ["COE_UNSUPPORTED_PREDICATE"]
        return _fail_closed_coe(
            bound,
            docs=docs,
            reason="CoE source validation failed",
            failure_codes=list(dict.fromkeys(failure_codes)),
            audit=audit,
        )
    return bound


def _preflight_candidate_claims(
    candidates: list[S.Claim],
    docs: dict[str, str],
    *,
    query: str,
    solver_path: list[str],
) -> tuple[list[S.Claim], int]:
    """Return verifier- and semantic-preflight survivors in original order.

    The final payload is still bound and audited as a whole. This preflight keeps
    one invalid candidate from suppressing unrelated, valid extractions.
    """
    raw_presentable = sum(1 for c in candidates if c.status in _PRESENTABLE_STATUSES)
    verified = [verify_claim(deepcopy(c)) for c in candidates]
    eligible = [
        c
        for c in verified
        if c.status in _PRESENTABLE_STATUSES and c.meta.get("verify") == "pass"
    ]
    rejected = raw_presentable - len(eligible)
    if not eligible:
        return [], rejected

    try:
        from wedge_v1.coe.bind import bind_ask_payload

        probe = bind_ask_payload(
            {
                "query": query,
                "answer_status": "SUPPORTED",
                "claims": [claim_to_dict(c) for c in eligible],
                "solver_path": list(solver_path),
                "trace": {
                    "events": [{"stage": "claim_preflight", "detail": "candidate binding"}],
                    "solvers": list(solver_path),
                    "failure_codes": [],
                },
                "lm_invoked": False,
            },
            docs,
            persist=False,
        )
    except Exception:
        # The authoritative finalizer below will fail closed if the binder is not
        # available. Keeping candidates here avoids creating a second policy path.
        return eligible, rejected

    invalid_ids = set((probe.get("coe") or {}).get("invalid_claim_ids") or [])
    enriched = probe.get("claims") or []
    if len(enriched) != len(eligible):
        return [], raw_presentable
    kept = [
        claim
        for claim, bound in zip(eligible, enriched, strict=True)
        if bound.get("claim_id") not in invalid_ids
    ]
    rejected += len(eligible) - len(kept)
    return kept, rejected


def _preflight_candidates(
    candidates: list[S.Claim],
    docs: dict[str, str],
    *,
    query: str,
    solver_path: list[str],
) -> tuple[list[dict], int]:
    """Project shared semantic-preflight survivors for the public finalizer."""
    kept, rejected = _preflight_candidate_claims(
        candidates,
        docs,
        query=query,
        solver_path=solver_path,
    )
    return [claim_to_dict(c) for c in kept], rejected


def _bound_evidence_spans(claim: dict) -> list[dict]:
    spans = []
    for evidence in claim.get("evidence") or []:
        if not isinstance(evidence, dict):
            continue
        spans.append(
            {
                key: evidence.get(key)
                for key in (
                    "atom_id",
                    "doc_id",
                    "doc_digest",
                    "relation",
                    "start",
                    "end",
                    "text",
                )
            }
        )
    return spans


def _bound_contradiction_rows(claims: list[dict]) -> list[dict]:
    kind_map = {
        "ttl_seconds": "numeric_ttl",
        "metformin_dose_mg": "numeric_dose",
        "sample_n": "numeric_sample_n",
    }
    rows = []
    for claim in claims:
        if claim.get("status") != "DISPUTED" or not claim.get("claim_id"):
            continue
        value = claim.get("value")
        field = value.get("field") if isinstance(value, dict) else None
        values = None
        if isinstance(value, dict):
            values = value.get("values") or value.get("docs")
        row = {
            "kind": kind_map.get(str(field), f"numeric_{field}") if field else "disputed_claim",
            "status": "DISPUTED",
            "claim_id": claim["claim_id"],
            "evidence_spans": _bound_evidence_spans(claim),
        }
        if field:
            row["field"] = field
        if values is not None:
            row["values"] = values
        else:
            row["value"] = value
        rows.append(row)
    return rows


def _refresh_bound_surfaces(payload: dict) -> dict:
    """Rebuild duplicate UX facts only from claims that survived CoE audit."""
    claims = [c for c in (payload.get("claims") or []) if isinstance(c, dict)]
    if not claims:
        for key in _FACTUAL_LIST_SURFACES:
            if key in payload:
                payload[key] = []
        for key in _FACTUAL_MAP_SURFACES:
            if key in payload:
                payload[key] = {}
        if "contradiction_banner" in payload:
            payload["contradiction_banner"] = None
        return payload

    if "contradictions_nearby" in payload or "contradictions_corpus" in payload:
        rows = _bound_contradiction_rows(claims)
        if "contradictions_nearby" in payload:
            payload["contradictions_nearby"] = rows
        if "contradictions_corpus" in payload:
            payload["contradictions_corpus"] = list(rows)
        if "contradiction_banner" in payload:
            kinds = sorted({str(row["kind"]) for row in rows})
            payload["contradiction_banner"] = (
                f"query-relevant contradictions: {', '.join(kinds)}" if kinds else None
            )

    numeric_claims = []
    for claim in claims:
        value = claim.get("value")
        if not isinstance(value, dict) or not isinstance(value.get("values"), dict):
            continue
        if claim.get("task_id") == "MERGE" or "numeric_compare" in str(claim.get("notes") or ""):
            numeric_claims.append(claim)

    if "epistemic_merge" in payload:
        rows = []
        for claim in numeric_claims:
            value = claim["value"]
            values = dict(value["values"])
            unique = sorted(set(values.values()), key=lambda item: str(item))
            rows.append(
                {
                    "field_id": value.get("field"),
                    "status": claim.get("status"),
                    "values_by_doc": values,
                    "unique_values": unique,
                    "disputed": claim.get("status") == "DISPUTED",
                    "evidence_spans": _bound_evidence_spans(claim),
                    "claim_id": claim.get("claim_id"),
                }
            )
        payload["epistemic_merge"] = rows

    exact_hits = []
    for claim in claims:
        if claim.get("task_id") != "COMPARE" or "exact_compare_hit" not in str(claim.get("notes") or ""):
            continue
        evidence = _bound_evidence_spans(claim)
        if not evidence:
            continue
        span = evidence[0]
        exact_hits.append(
            {
                **span,
                "status": claim.get("status"),
                "value": claim.get("value"),
                "closest_number": None,
                "numeric": None,
                "claim_id": claim.get("claim_id"),
            }
        )
    if "hits" in payload:
        payload["hits"] = exact_hits

    if "values_by_doc" in payload:
        if numeric_claims:
            values = numeric_claims[0]["value"]["values"]
            payload["values_by_doc"] = {
                doc_id: value if isinstance(value, list) else [str(value)]
                for doc_id, value in values.items()
            }
        else:
            grouped: dict[str, list[str]] = {}
            for hit in exact_hits:
                doc_id = str(hit.get("doc_id") or "")
                text = str(hit.get("text") or "")
                if doc_id and text:
                    grouped.setdefault(doc_id, []).append(text)
            payload["values_by_doc"] = grouped
    if "field_values" in payload:
        values = numeric_claims[0]["value"]["values"] if numeric_claims else {}
        payload["field_values"] = {
            doc_id: str(value)
            for doc_id, value in values.items()
            if not isinstance(value, (dict, list, tuple, set))
        }
    return payload


def _finalize_public_payload(
    payload: dict,
    docs: dict[str, str],
    *,
    candidates: list[S.Claim] | None = None,
    op: str,
    query: str,
    persist: bool = True,
    scope: dict | None = None,
) -> dict:
    """Shared verifier, binding, audit, and presentation boundary for public ops."""
    if scope:
        payload.update(deepcopy(scope))
    rejected = 0
    if candidates is not None:
        claims, rejected = _preflight_candidates(
            candidates,
            docs,
            query=query,
            solver_path=list(payload.get("solver_path") or []),
        )
        payload["claims"] = claims
        payload["n_claims_presented"] = len(claims)
        payload["n_claims_rejected"] = rejected
        if payload.get("answer_status") in {"SUPPORTED", "CONTRADICTED"} and not claims:
            payload["answer_status"] = "ABSTAIN"
            payload["note"] = "no verifier-approved evidence-bound claims"
            unsupported = list(payload.get("unsupported") or [])
            if query and query not in unsupported:
                unsupported.append(query)
            payload["unsupported"] = unsupported

    if not isinstance(payload.get("trace"), dict):
        trace = AskTrace(
            query=query,
            corpus_dir=str(payload.get("corpus_dir") or ""),
            op=op,
        )
        trace.n_docs = len(docs)
        trace.n_claims_raw = len(candidates or [])
        trace.n_claims_presented = len(payload.get("claims") or [])
        for solver in payload.get("solver_path") or []:
            trace.add_solver(str(solver))
        for code in payload.get("failure_codes") or []:
            try:
                trace.add_failure(FailureCode(str(code)))
            except ValueError:
                continue
        trace.event(
            "claim_verification",
            "public presentation boundary",
            rejected=rejected,
            presented=len(payload.get("claims") or []),
        )
        if rejected:
            trace.add_failure(FailureCode.VERIFIER_REJECTION)
        payload["trace"] = trace.finalize(str(payload.get("answer_status") or "ABSTAIN"))
        payload.setdefault("latency_ms", payload["trace"]["latency_ms"])

    if rejected:
        codes = list(payload.get("failure_codes") or [])
        if FailureCode.VERIFIER_REJECTION.value not in codes:
            codes.append(FailureCode.VERIFIER_REJECTION.value)
        payload["failure_codes"] = codes
    payload.setdefault("lm_invoked", False)
    finalized = _refresh_bound_surfaces(_finalize_with_coe(payload, docs, persist=persist))
    if scope:
        finalized.setdefault("coe", {})["selected_doc_ids"] = list(
            scope.get("selected_doc_ids") or []
        )
    return finalized



def ask(
    query: str,
    corpus_dir: Path | None = None,
    *,
    doc_ids: list[str] | None = None,
    persist_coe: bool = True,
) -> dict:
    """Span-first Q&A over a local folder. Never invents unsupported claims."""
    corpus_path = Path(corpus_dir) if corpus_dir else DEFAULT_CORPUS
    trace = AskTrace(query=query, corpus_dir=str(corpus_path), op="ask")
    docs, scope, scope_valid = select_documents(load_corpus(corpus_dir), doc_ids)
    if not scope_valid:
        return _scope_abstention(
            op="ask",
            query=query.strip(),
            corpus_path=corpus_path,
            scope=scope,
            persist=persist_coe,
        )
    if not docs:
        trace.add_solver("load_corpus")
        trace.add_failure(FailureCode.NO_CORPUS)
        tr = trace.finalize("NO_CORPUS")
        return {
            "query": query,
            "corpus_dir": str(corpus_path),
            "answer_status": "NO_CORPUS",
            "claims": [],
            "unsupported": ["corpus empty or missing"],
            "solver_path": ["load_corpus"],
            "failure_codes": tr["failure_codes"],
            "trace": tr,
            "latency_s": round(tr["latency_ms"] / 1000, 4),
            "latency_ms": tr["latency_ms"],
        }

    q = query.strip()
    claims: list[S.Claim] = []
    solver_path = ["load_corpus"]
    trace.add_solver("load_corpus")
    trace.n_docs = len(docs)
    trace.event("ingest", "loaded", n_docs=len(docs))
    tokens = _content_tokens(q)
    predicates = decompose(q)
    domains = [p.domain for p in predicates]
    composition = len(predicates) >= 2
    if composition:
        trace.event(
            "predicate_decompose",
            "atomic_conjunction",
            n=len(predicates),
            domains=domains,
        )
        # Seed cascade with corpus-agnostic merge claims for known domains
        merge_seed = predicate_claims_for_domains(docs, domains)
        claims.extend(merge_seed)
        if merge_seed:
            solver_path.append("merge_atomic_predicates")
            trace.add_solver("merge_atomic_predicates")

    # Any contradiction shown by ask must also exist as a primary candidate so it
    # can be verified, bound, and audited before the nested UX row is rebuilt.
    query_merges = merge_for_term(docs, q)
    existing_merge_notes = {c.notes for c in claims if c.task_id == "MERGE"}
    added_query_merges = [c for c in query_merges if c.notes not in existing_merge_notes]
    claims.extend(added_query_merges)
    if query_merges:
        solver_path.append("query_epistemic_merge")
        trace.add_solver("query_epistemic_merge")

    q_content = " ".join(tokens) if tokens else q
    for did, text in docs.items():
        paragraph_claim = S.keyword_paragraph(did, text, q_content)
        for evidence in paragraph_claim.evidence or []:
            evidence_text = evidence.get("text") if isinstance(evidence, dict) else None
            if evidence_text:
                exact_start = text.find(evidence_text)
                if exact_start >= 0:
                    evidence["start"] = exact_start
                    evidence["end"] = exact_start + len(evidence_text)
        claims.append(paragraph_claim)
        if tokens:
            claims.append(S.quote_sentence(did, text, tokens[0]))
    solver_path.append("keyword_paragraph+quote")
    trace.add_solver("keyword_paragraph+quote")
    for tok in tokens[:5]:
        claims.append(S.mention_docs(docs, tok))
        for did, body in docs.items():
            c = S.yes_no_mention(did, body, tok)
            if c.value is True:
                claims.append(c)
    solver_path.append("mention+yes_no")
    trace.add_solver("mention+yes_no")

    hyphen_chunks = set(re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+", q))
    for num in re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?(?![A-Za-z])", q):
        if any(num in chunk and chunk != num for chunk in hyphen_chunks):
            continue
        for did, body in docs.items():
            i = body.find(num)
            if i >= 0:
                line = body[max(0, body.rfind("\n", 0, i) + 1) : body.find("\n", i)]
                if not line:
                    line = body[i : i + len(num)]
                claims.append(
                    S.Claim(
                        "FIND",
                        did,
                        num,
                        evidence=[{
                            "start": i,
                            "end": i + len(num),
                            "text": body[i : i + len(num)],
                            "line": line.strip()[:240],
                        }],
                        status="PRESENT",
                        notes="numeric_span",
                    )
                )
    for lit in re.findall(r'"([^"]{3,80})"', q):
        for did, body in docs.items():
            i = body.find(lit)
            if i >= 0:
                claims.append(
                    S.Claim(
                        "FIND",
                        did,
                        lit,
                        evidence=[{
                            "start": i,
                            "end": i + len(lit),
                            "text": body[i : i + len(lit)],
                        }],
                        status="PRESENT",
                        notes="literal_span",
                    )
                )
    for tok in tokens:
        if "-" in tok or any(ch.isdigit() for ch in tok) or (tok.isupper() and len(tok) >= 2):
            for did, body in docs.items():
                i = body.find(tok)
                if i >= 0:
                    claims.append(
                        S.Claim(
                            "FIND",
                            did,
                            tok,
                            evidence=[{
                                "start": i,
                                "end": i + len(tok),
                                "text": body[i : i + len(tok)],
                            }],
                            status="PRESENT",
                            notes="token_span",
                        )
                    )
    solver_path.append("numeric+literal_spans")
    trace.add_solver("numeric+literal_spans")

    bm25_present, bm25_review = _bm25_claims(docs, q, k=5)
    claims.extend(bm25_present)
    if bm25_present:
        solver_path.append("bm25_paragraphs")
        trace.add_solver("bm25_paragraphs")
    if bm25_review:
        solver_path.append("bm25_low_margin_review")
        trace.add_solver("bm25_low_margin_review")
        trace.n_bm25_review = len(bm25_review)
        trace.add_failure(FailureCode.LOW_MARGIN_RETRIEVAL)
        trace.event("bm25_margin_gate", "review_hits", n=len(bm25_review))

    ql = q.lower()
    # W4: pluggable cascade (synonym/ocr/coref) — no fixture doc-id gates
    cascade = run_cascade(docs, q)
    claims.extend(cascade.claims)
    for mod in cascade.modules_run:
        name = f"plugin.{mod}"
        solver_path.append(name)
        trace.add_solver(name)
    if cascade.modules_run:
        trace.event("plugin_cascade", "ran", modules=cascade.modules_run, n=len(cascade.claims))
    if "dose" in ql or "metformin" in ql:
        claims.append(S.symbolic_dose_change(docs))
        claims.append(S.union_dosages(docs))
        solver_path.append("eclass_symbolic_dose+union")
        trace.add_solver("eclass_symbolic_dose+union")

    # Optional decidable verifier pass (records rejects; presentation still uses evidence gate)
    verified = [verify_claim(c) for c in claims]
    empty_rejected = sum(1 for c in verified if c.meta.get("verify") == "fail_no_evidence")
    trace.n_empty_evidence_rejected = empty_rejected
    if empty_rejected:
        trace.add_failure(FailureCode.EMPTY_EVIDENCE_REJECTED)

    presented = [
        c
        for c in verified
        if c.status in {"PRESENT", "CONFIRMED", "DISPUTED", "PROBABLE"}
        and _has_evidence_atom(c)
        and _relevant_claim(c, tokens, docs)
    ]
    presented_sorted = sorted(
        presented,
        key=lambda c: (0 if c.task_id == "FIND" else 1, 0 if c.evidence else 1, c.task_id),
    )
    trace.n_claims_raw = len(claims)
    presented_sorted, preflight_rejected = _preflight_candidate_claims(
        presented_sorted,
        docs,
        query=q,
        solver_path=solver_path,
    )
    trace.event(
        "claim_preflight",
        "rejected_candidates" if preflight_rejected else "complete",
        rejected=preflight_rejected,
        survivors=len(presented_sorted),
    )
    if preflight_rejected:
        trace.add_failure(FailureCode.VERIFIER_REJECTION)
    candidate_contradictions = _candidate_contradictions(presented_sorted)
    has_lex = _corpus_has_lexical(docs, tokens)

    # W3/CoE: atomic predicates — compound answers require full conjunction support
    pred_support = evaluate_predicates(predicates, docs, presented_sorted)
    pred_payload = {
        "predicates": [p.to_dict() for p in predicates],
        "predicate_support": [s.to_dict() for s in pred_support],
    }
    if incomplete_conjunction(pred_support):
        missing = [s.domain for s in pred_support if not s.supported]
        trace.add_failure(FailureCode.UNSUPPORTED_COMPOSITION)
        trace.add_failure(FailureCode.COE_INCOMPLETE_CONJUNCTION)
        trace.event("incomplete_conjunction", "missing_predicates", missing=missing)
        for code in classify_abstain_failures(
            has_lexical_hit=has_lex,
            bm25_review_n=len(bm25_review),
            empty_rejected=empty_rejected,
            oos_expected=False,
            composition_blocked=True,
        ):
            trace.add_failure(code)
        # Present only claims that support *supported* predicates (optional transparency)
        supported_domains = {s.domain for s in pred_support if s.supported}
        partial = [
            c for c in presented_sorted
            if _claim_covers_domain([c], next(iter(supported_domains), ""))
            or any(d in str(c.value).lower() + c.notes.lower() for d in supported_domains)
        ][:6]
        trace.n_claims_presented = len(partial)
        tr = trace.finalize("ABSTAIN")
        payload = {
            "query": q,
            "corpus_dir": str(corpus_path),
            "answer_status": "ABSTAIN",
            "claims": [claim_to_dict(c) for c in partial],
            "unsupported": [q],
            "solver_path": solver_path + ["predicate_conjunction_gate"],
            "note": f"incomplete conjunction; unsupported predicates: {missing}",
            "n_docs": len(docs),
            "n_claims_rejected": preflight_rejected,
            "contradictions_nearby": candidate_contradictions,
            "contradictions_corpus": candidate_contradictions,
            "bm25_review": bm25_review,
            "abstain_class": "coe_incomplete_conjunction",
            "failure_codes": tr["failure_codes"],
            "trace": tr,
            "latency_s": round(tr["latency_ms"] / 1000, 4),
            "latency_ms": tr["latency_ms"],
            "lm_invoked": False,
            **pred_payload,
        }
        return _finalize_public_payload(
            payload,
            docs,
            op="ask",
            query=q,
            persist=persist_coe,
            scope=scope,
        )

    if not presented_sorted:
        oos = not has_lex and len(bm25_review) == 0
        for code in classify_abstain_failures(
            has_lexical_hit=has_lex,
            bm25_review_n=len(bm25_review),
            empty_rejected=empty_rejected,
            oos_expected=oos,
            composition_blocked=composition,
        ):
            trace.add_failure(code)
        primary = (trace.failure_codes[0].value if trace.failure_codes else "retrieval_miss_or_unsupported")
        trace.n_claims_presented = 0
        tr = trace.finalize("ABSTAIN")
        payload = {
            "query": q,
            "corpus_dir": str(corpus_path),
            "answer_status": "ABSTAIN",
            "claims": [],
            "unsupported": [q],
            "solver_path": solver_path,
            "note": "no span-supported claim under classical+eclass cascade",
            "n_docs": len(docs),
            "n_claims_rejected": preflight_rejected,
            "contradictions_nearby": candidate_contradictions,
            "contradictions_corpus": candidate_contradictions,
            "bm25_review": bm25_review,
            "abstain_class": primary.lower(),
            "failure_codes": tr["failure_codes"],
            "trace": tr,
            "latency_s": round(tr["latency_ms"] / 1000, 4),
            "latency_ms": tr["latency_ms"],
            "lm_invoked": False,
            "predicates": [p.to_dict() for p in predicates],
            "predicate_support": [s.to_dict() for s in evaluate_predicates(predicates, docs, [])],
        }
        return _finalize_public_payload(
            payload,
            docs,
            op="ask",
            query=q,
            persist=persist_coe,
            scope=scope,
        )

    disputed = [c for c in presented_sorted if c.status == "DISPUTED"]
    selected_presented = list(disputed[:12])
    for claim in presented_sorted:
        if len(selected_presented) >= 12:
            break
        if claim not in selected_presented:
            selected_presented.append(claim)
    trace.n_claims_presented = len(selected_presented)
    nearby = candidate_contradictions
    ql_local = q.lower()
    relevant_nearby = []
    for n in nearby:
        kind = n.get("kind", "")
        if kind == "numeric_dose" and ("dose" in ql_local or "metformin" in ql_local):
            relevant_nearby.append(n)
        elif kind == "numeric_ttl" and any(k in ql_local for k in ("ttl", "cache", "expire", "cached")):
            relevant_nearby.append(n)
        elif kind == "entity_collision":
            val = n.get("value")
            strings: list[str] = []
            if isinstance(val, dict):
                strings = [str(v) for v in val.values()]
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        strings.append(str(item.get("string") or ""))
                    else:
                        strings.append(str(item))
            elif val is not None:
                strings = [str(val)]
            if any(s.lower() in ql_local for s in strings if s):
                relevant_nearby.append(n)
    status = "CONTRADICTED" if (disputed or relevant_nearby) else "SUPPORTED"
    if status == "CONTRADICTED":
        trace.add_failure(FailureCode.MULTI_DOC_CONTRADICTION)
        if any(n.get("kind") == "numeric_dose" for n in relevant_nearby):
            trace.add_failure(FailureCode.NUMERIC_CONTRADICTION)
    banner = None
    if relevant_nearby:
        kinds = sorted({n["kind"] for n in relevant_nearby})
        banner = f"query-relevant contradictions: {', '.join(kinds)}"
        trace.event("contradiction", banner, n=len(relevant_nearby))
    tr = trace.finalize(status)
    payload = {
        "query": q,
        "corpus_dir": str(corpus_path),
        "answer_status": status,
        "claims": [claim_to_dict(c) for c in selected_presented],
        "unsupported": [],
        "solver_path": solver_path,
        "n_docs": len(docs),
        "n_claims_rejected": preflight_rejected,
        "contradictions_nearby": relevant_nearby,
        "contradictions_corpus": nearby,
        "contradiction_banner": banner,
        "bm25_review": bm25_review,
        "failure_codes": tr["failure_codes"],
        "trace": tr,
        "latency_s": round(tr["latency_ms"] / 1000, 4),
        "latency_ms": tr["latency_ms"],
        "lm_invoked": False,
        "predicates": [p.to_dict() for p in predicates],
        "predicate_support": [s.to_dict() for s in evaluate_predicates(predicates, docs, presented_sorted)],
    }
    return _finalize_public_payload(
        payload,
        docs,
        op="ask",
        query=q,
        persist=persist_coe,
        scope=scope,
    )



def scan(
    corpus_dir: Path | None = None,
    *,
    doc_ids: list[str] | None = None,
    persist_coe: bool = True,
) -> dict:
    """Run inventory extractors across corpus (metadata, dosages, contradictions)."""
    corpus_path = Path(corpus_dir) if corpus_dir else DEFAULT_CORPUS
    docs, scope, scope_valid = select_documents(load_corpus(corpus_dir), doc_ids)
    if not scope_valid:
        return _scope_abstention(
            op="scan",
            query="scan",
            corpus_path=corpus_path,
            scope=scope,
            persist=persist_coe,
        )
    if not docs:
        return {"answer_status": "NO_CORPUS", "claims": [], "solver_path": ["load_corpus"]}

    claims: list[S.Claim] = []
    for did, text in docs.items():
        claims.extend([
            S.extract_title(did, text),
            S.extract_authors(did, text),
            S.extract_year(did, text),
            S.detect_doc_type(did, text),
            S.extract_dosages(did, text),
            S.extract_compounds(did, text),
            S.extract_sample_n(did, text),
            S.extract_doi(did, text),
        ])
    claims.append(S.union_dosages(docs))
    claims.append(S.symbolic_dose_change(docs))
    claims.append(S.flag_entity_collision(docs))
    casc = run_cascade(docs, want={"synonym", "ocr", "coref"})
    claims.extend(casc.claims)

    return _finalize_public_payload(
        {
            "query": "scan",
            "corpus_dir": str(corpus_path),
            "answer_status": "SUPPORTED",
            "claims": [],
            "unsupported": [],
            "solver_path": ["scan_classical+eclass", "plugin_cascade"],
            "n_docs": len(docs),
            "n_claims": len(claims),
        },
        docs,
        candidates=claims,
        op="scan",
        query="scan",
        persist=persist_coe,
        scope=scope,
    )



def compare(
    term: str,
    corpus_dir: Path | None = None,
    window: int = 100,
    *,
    doc_ids: list[str] | None = None,
    persist_coe: bool = True,
) -> dict:
    """Compare exact term spans and evidence-bound numeric fields across docs."""
    t0 = time.perf_counter()
    corpus_path = Path(corpus_dir) if corpus_dir else DEFAULT_CORPUS
    docs, scope, scope_valid = select_documents(load_corpus(corpus_dir), doc_ids)
    if not scope_valid:
        return _scope_abstention(
            op="compare",
            query=term.strip(),
            corpus_path=corpus_path,
            scope=scope,
            persist=persist_coe,
            extra={"term": term, "latency_s": round(time.perf_counter() - t0, 4)},
        )
    if not docs:
        return {
            "term": term,
            "corpus_dir": str(corpus_path),
            "answer_status": "NO_CORPUS",
            "claims": [],
            "solver_path": ["load_corpus"],
            "latency_s": round(time.perf_counter() - t0, 4),
        }

    needle = term.strip()
    if not needle:
        return _finalize_public_payload(
            {
                "query": needle,
                "term": term,
                "corpus_dir": str(corpus_path),
                "answer_status": "ABSTAIN",
                "claims": [],
                "unsupported": ["empty term"],
                "solver_path": ["compare"],
                "n_docs": len(docs),
                "n_hits": 0,
                "latency_s": round(time.perf_counter() - t0, 4),
            },
            docs,
            candidates=[],
            op="compare",
            query=needle,
            persist=persist_coe,
            scope=scope,
        )

    pattern = re.compile(re.escape(needle), re.I)
    hits: list[dict] = []
    # Field-like: "TTL as 300 seconds", "metformin dose: 500 mg", "QPS remains 12000"
    field_re = re.compile(
        rf"{re.escape(needle)}\s+"
        rf"(?:(?:dose|value)\s*[:=]\s*|(?:as|is|remains)\s+|[=:]\s*)?"
        rf"(\d+(?:\.\d+)?)\s*(?:seconds|mg|sec|qps)?",
        re.I,
    )
    field_vals: dict[str, str] = {}
    field_evidence: list[dict] = []

    for did, text in docs.items():
        for m in pattern.finditer(text):
            i, j = m.start(), m.end()
            lo = max(0, i - window)
            hi = min(len(text), j + window)
            ctx = text[lo:hi].replace("\n", " ")
            hits.append(
                {
                    "doc_id": did,
                    "start": i,
                    "end": j,
                    "text": text[i:j],
                    "context": ctx[:240],
                }
            )
        fm = field_re.search(text)
        if fm:
            group = 1
            value = fm.group(1)
            field_vals[did] = value
            field_evidence.append(
                {
                    "doc_id": did,
                    "start": fm.start(group),
                    "end": fm.end(group),
                    "text": value,
                    "field": needle,
                }
            )

    if not hits:
        return _finalize_public_payload(
            {
                "query": needle,
                "term": needle,
                "corpus_dir": str(corpus_path),
                "answer_status": "ABSTAIN",
                "claims": [],
                "unsupported": [needle],
                "solver_path": ["compare"],
                "n_docs": len(docs),
                "n_hits": 0,
                "latency_s": round(time.perf_counter() - t0, 4),
            },
            docs,
            candidates=[],
            op="compare",
            query=needle,
            persist=persist_coe,
            scope=scope,
        )

    merge_claims = [c for c in merge_for_term(docs, needle) if c.status != "ABSTAIN"]
    numeric_claims = list(merge_claims)
    if not numeric_claims and field_vals:
        numeric_claims.append(
            S.Claim(
                "COMPARE",
                None,
                {"field": needle, "values": field_vals},
                evidence=field_evidence,
                status="DISPUTED" if len(set(field_vals.values())) >= 2 else "PRESENT",
                notes="numeric_compare",
            )
        )

    exact_claims = [
        S.Claim(
            "COMPARE",
            hit["doc_id"],
            hit["text"],
            evidence=[dict(hit)],
            status="PRESENT",
            notes="exact_compare_hit",
        )
        for hit in hits[:24]
    ]
    candidates = [*numeric_claims, *exact_claims]
    status = (
        "CONTRADICTED"
        if any(claim.status == "DISPUTED" for claim in numeric_claims)
        else "SUPPORTED"
    )
    solver_path = ["compare"]
    if numeric_claims:
        solver_path.append("epistemic_merge")

    return _finalize_public_payload(
        {
            "query": needle,
            "term": needle,
            "corpus_dir": str(corpus_path),
            "answer_status": status,
            "claims": [],
            "epistemic_merge": [],
            "hits": [],
            "values_by_doc": {},
            "field_values": {},
            "unsupported": [],
            "solver_path": solver_path,
            "n_docs": len(docs),
            "n_docs_hit": len({h["doc_id"] for h in hits}),
            "n_hits": len(hits),
            "failure_codes": (
                [
                    FailureCode.MULTI_DOC_CONTRADICTION.value,
                    FailureCode.NUMERIC_CONTRADICTION.value,
                ]
                if status == "CONTRADICTED"
                else []
            ),
            "latency_s": round(time.perf_counter() - t0, 4),
        },
        docs,
        candidates=candidates,
        op="compare",
        query=needle,
        persist=persist_coe,
        scope=scope,
    )


def format_report_md(payload: dict, *, title: str | None = None) -> str:
    """Human-readable markdown for ask/find/scan/compare payloads (internal tooling UX)."""
    lines: list[str] = []
    status = payload.get("answer_status", "?")
    lines.append(f"# {title or 'wedge_v1 report'}")
    lines.append("")
    if payload.get("query"):
        lines.append(f"**Query:** {payload['query']}")
    if payload.get("term"):
        lines.append(f"**Term:** `{payload['term']}`")
    lines.append(f"**Status:** `{status}`")
    if "selected_doc_ids" in payload:
        selected = ", ".join(payload.get("selected_doc_ids") or []) or "—"
        missing = ", ".join(payload.get("missing_doc_ids") or []) or "—"
        lines.append(f"**Selected docs:** {selected}")
        lines.append(f"**Missing docs:** {missing}")
    banner = payload.get("contradiction_banner")
    if banner:
        lines.append("")
        lines.append(f"> **Contradiction banner:** {banner}")
    if "n_docs" in payload:
        lines.append(f"**Docs:** {payload['n_docs']}")
    if "n_hits" in payload:
        lines.append(f"**Hits:** {payload['n_hits']}")
    if "latency_s" in payload:
        lines.append(f"**Latency:** {payload['latency_s']}s")
    if "latency_ms" in payload:
        lines.append(f"**Latency:** {payload['latency_ms']}ms")
    path = payload.get("solver_path") or payload.get("solver") or []
    if path:
        lines.append("**Solver path:** " + " → ".join(str(p) for p in path))
    lines.append("")
    claims = payload.get("claims") or []
    if not claims:
        lines.append("_No claims._")
        lines.append("")
    else:
        lines.append(f"## Claims ({len(claims)})")
        lines.append("")
        for i, c in enumerate(claims, 1):
            if not isinstance(c, dict):
                continue
            val = c.get("value")
            st = c.get("status", "")
            tid = c.get("task_id", "")
            doc = c.get("doc_id", "")
            lines.append(f"### {i}. `{tid}` — {st}")
            lines.append(f"- **Doc:** `{doc}`")
            lines.append(f"- **Value:** `{val}`")
            for e in (c.get("evidence") or [])[:5]:
                if not isinstance(e, dict):
                    continue
                span = e.get("text") or e.get("line") or ""
                ctx = e.get("context")
                start_i, end_i = e.get("start"), e.get("end")
                loc = f" [{start_i}:{end_i}]" if start_i is not None else ""
                lines.append(f"- **Evidence{loc}:** {span}")
                if ctx:
                    lines.append(f"  - context: {ctx}")
            notes = c.get("notes")
            if notes:
                lines.append(f"- _notes:_ {notes}")
            lines.append("")
    unsupported = payload.get("unsupported") or []
    if unsupported:
        lines.append("## Unsupported / abstain reasons")
        for u in unsupported:
            lines.append(f"- {u}")
        lines.append("")
    if payload.get("abstain_reason"):
        lines.append(f"**Abstain reason:** {payload['abstain_reason']}")
        lines.append("")
    epistemic = payload.get("epistemic_merge") or []
    if epistemic:
        lines.append("## Epistemic merge (typed fields, both spans)")
        for row in epistemic[:12]:
            if not isinstance(row, dict):
                continue
            fid = row.get("field_id", "?")
            st = row.get("status", "?")
            vals = row.get("values_by_doc") or {}
            lines.append(f"### `{fid}` — {st}")
            for doc_id, val in sorted(vals.items()):
                lines.append(f"- **{doc_id}:** `{val}`")
            for sp in (row.get("evidence_spans") or [])[:8]:
                if not isinstance(sp, dict):
                    continue
                loc = ""
                if sp.get("start") is not None:
                    loc = f" [{sp.get('start')}:{sp.get('end')}]"
                lines.append(
                    f"  - span `{sp.get('doc_id')}`{loc}: {sp.get('text', '')}"
                )
            lines.append("")

    contradictions = (
        payload.get("contradictions_nearby")
        or payload.get("contradictions")
        or payload.get("disputes")
        or []
    )
    if contradictions:
        lines.append("## Contradictions")
        for item in contradictions[:20]:
            if isinstance(item, dict):
                kind = item.get("kind") or item.get("task_id") or "flag"
                doc = item.get("doc_id", "")
                val = item.get("values", item.get("value"))
                lines.append(f"- `{kind}` doc=`{doc}` value=`{val}`")
            else:
                lines.append(f"- `{item}`")
        lines.append("")
    spans = payload.get("evidence_spans") or []
    if spans:
        lines.append(f"## Evidence spans ({len(spans)})")
        for e in spans[:12]:
            if not isinstance(e, dict):
                continue
            lines.append(
                f"- `{e.get('doc_id')}` [{e.get('start')}:{e.get('end')}] {str(e.get('text', ''))[:160]}"
            )
        lines.append("")
    lines.append("---")
    lines.append("_Verification-first local slice. No generative fill-in._")
    lines.append("")
    return chr(10).join(lines)
