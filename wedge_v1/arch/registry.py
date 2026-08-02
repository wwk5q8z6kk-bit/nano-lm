"""Failure-driven architecture registry — components, layers, invariants.

Implementation-facing. Not a governance charter. Not Layer-1 evidence.
"""
from __future__ import annotations

from typing import Any

from wedge_v1.arch.failure_codes import FailureCode, HISTORICAL_LESSONS

LAYERS: dict[str, str] = {
    "L0": "Local data boundary and trust model",
    "L1": "Corpus discovery and file lifecycle",
    "L2": "Parsing, OCR, layout, tables, normalized structure",
    "L3": "Stable document identity, versioning, incremental indexing",
    "L4": "Sparse/exact/structural/(optional dense) retrieval",
    "L5": "Candidate atom extraction",
    "L6": "Typed claims and minimal-condition decomposition",
    "L7": "Evidence binding and provenance",
    "L8": "Verification and contradiction analysis",
    "L9": "Confidence, calibration, abstention, review routing",
    "L10": "Solver registry and cost-aware routing",
    "L11": "User corrections and local learning",
    "L12": "Habit, saved questions, longitudinal memory",
    "L13": "Optional compact-model subsystem",
    "L14": "Resource/latency/memory/energy governance",
    "L15": "Security, privacy, poisoning, prompt-injection defenses",
    "L16": "Observability, replay, debugging, experiment tracing",
    "L17": "Evaluation harness and regression corpus",
    "L18": "Product surfaces and workflow integration",
}

# component_id → registry row
COMPONENTS: dict[str, dict[str, Any]] = {
    "ingest.load_corpus": {
        "layers": ["L1", "L2", "L3"],
        "path": "wedge_v1/ingest.py",
        "interface": "load_corpus(dir) -> {doc_id: text}",
        "invariants": ["doc_id=stem", "md/txt before pdf", "no PHI to tracked paths"],
        "failures": [FailureCode.NO_CORPUS, FailureCode.INGESTION_LAYOUT_FAILURE],
        "status": "implemented",
    },
    "ingest.sla": {
        "layers": ["L2", "L17"],
        "path": "wedge_v1/ingest_sla.py",
        "interface": "measure_ingest_sla() -> recover_gap + FIELD_SLA",
        "invariants": ["normalize before intelligence", "MIN_FIELD_RECOVERY=0.90"],
        "failures": [FailureCode.INGESTION_LAYOUT_FAILURE],
        "status": "implemented",
        "workstream": "W5",
    },
    "retrieve.bm25": {
        "layers": ["L4"],
        "path": "wedge_v1/classical/bm25.py",
        "interface": "top_paragraphs(docs,q,k) -> hits with margin/promote",
        "invariants": ["margin gate BM25_MARGIN_TAU", "low margin → REVIEW not PRESENT"],
        "failures": [
            FailureCode.RETRIEVAL_MISS,
            FailureCode.WRONG_SPAN_RETRIEVAL,
            FailureCode.LOW_MARGIN_RETRIEVAL,
        ],
        "status": "implemented",
        "workstream": "W1",
    },
    "retrieve.exact_find": {
        "layers": ["L4", "L5"],
        "path": "wedge_v1/runtime.py:find_spans",
        "interface": "find_spans(needle) -> hits with offsets",
        "invariants": ["offsets resolve in doc text"],
        "failures": [FailureCode.RETRIEVAL_MISS, FailureCode.EVIDENCE_ABSENT],
        "status": "implemented",
    },
    "extract.classical_cascade": {
        "layers": ["L5", "L6"],
        "path": "wedge_v1/classical/solvers.py + runtime.ask",
        "interface": "Claim(task_id,doc_id,value,evidence,status)",
        "invariants": ["presented claims require evidence atoms"],
        "failures": [
            FailureCode.EMPTY_EVIDENCE_REJECTED,
            FailureCode.FIXTURE_TIED_SOLVER,
            FailureCode.RULE_BRITTLENESS,
        ],
        "status": "implemented",
        "workstream": "W2",
    },
    "verify.decidable_r": {
        "layers": ["L7", "L8"],
        "path": "wedge_v1/classical/verifier.py",
        "interface": "verify_claim(Claim) -> Claim",
        "invariants": ["no PRESENT without evidence", "malformed evidence → REJECTED"],
        "failures": [FailureCode.VERIFIER_REJECTION, FailureCode.EMPTY_EVIDENCE_REJECTED],
        "status": "partial",  # not fully wired into ask() presentation path
        "workstream": "W2",
    },
    "merge.contradiction": {
        "layers": ["L8"],
        "path": "wedge_v1/runtime.py:compare/nearby_contradictions",
        "interface": "compare(term) / nearby_contradictions(docs)",
        "invariants": ["contradiction states preserved; not auto-resolved"],
        "failures": [
            FailureCode.MULTI_DOC_CONTRADICTION,
            FailureCode.NUMERIC_CONTRADICTION,
            FailureCode.ENTITY_TYPE_COLLISION,
        ],
        "status": "implemented",
        "workstream": "W3",
    },
    "plugins.synonym": {
        "layers": ["L5", "L10"],
        "path": "wedge_v1/plugins/synonym.py",
        "interface": "probe_paraphrase(docs,q) -> Claim T35",
        "invariants": ["lexicon expand from synonyms.json", "no fixture doc-id gate"],
        "failures": [FailureCode.RETRIEVAL_MISS, FailureCode.OVER_ABSTENTION],
        "status": "implemented",
        "workstream": "W4",
    },
    "plugins.ocr": {
        "layers": ["L2", "L5"],
        "path": "wedge_v1/plugins/ocr.py",
        "interface": "probe_docs(docs) -> Claim T37[]",
        "invariants": ["ocr_substitutions.json table", "evidence on normalized spans"],
        "failures": [FailureCode.INGESTION_LAYOUT_FAILURE],
        "status": "implemented",
        "workstream": "W4",
    },
    "plugins.coref": {
        "layers": ["L5", "L6"],
        "path": "wedge_v1/plugins/coref.py",
        "interface": "probe_docs(docs) -> Claim T39[]",
        "invariants": ["coref_entities.json lexicon", "any doc with pronoun"],
        "failures": [FailureCode.UNSUPPORTED_COMPOSITION],
        "status": "implemented",
        "workstream": "W4",
    },
    "route.solver_path": {
        "layers": ["L10"],
        "path": "wedge_v1/runtime.py:ask solver_path list",
        "interface": "logged solver_path[]; deterministic order",
        "invariants": ["inspectable; no opaque learned router"],
        "failures": [FailureCode.FIXTURE_TIED_SOLVER],
        "status": "implemented",
    },
    "abstain.policy": {
        "layers": ["L9"],
        "path": "wedge_v1/runtime.py:ask + arch.trace",
        "interface": "answer_status + failure_codes[]",
        "invariants": ["prefer abstain over unsupported composition"],
        "failures": [
            FailureCode.CORRECT_ABSTENTION,
            FailureCode.OVER_ABSTENTION,
            FailureCode.UNSUPPORTED_COMPOSITION,
        ],
        "status": "in_progress",
        "workstream": "W1",
    },
    "obs.ask_trace": {
        "layers": ["L16"],
        "path": "wedge_v1/arch/trace.py",
        "interface": "payload['trace'] AskTrace.v1",
        "invariants": ["every ask/find/compare attaches trace when corpus loads"],
        "failures": [],
        "status": "implemented",
    },
    "eval.adversarial": {
        "layers": ["L17"],
        "path": "wedge_v1/eval/adversarial.py",
        "interface": "run_adversarial_suite()",
        "invariants": ["synthetic packs probe mechanisms; ≠ owner usefulness"],
        "failures": list(FailureCode),
        "status": "implemented",
    },
    "product.review_habit": {
        "layers": ["L11", "L12", "L18"],
        "path": "wedge_v1/review.py + habit.py",
        "interface": "review labels; habit session next_action",
        "invariants": ["labels gitignored; fixture≠owner"],
        "failures": [FailureCode.HIGH_REVIEW_BURDEN],
        "status": "implemented",
    },
    "model.compact": {
        "layers": ["L13"],
        "path": "wedge_v1/lm/",
        "interface": "evaluate_admission() / run_marginal_probe() stub default",
        "invariants": ["typed candidate claims only; information parity"],
        "failures": [FailureCode.OVER_ABSTENTION],
        "status": "partial",
        "workstream": "W6",
    },
}


def registry_snapshot() -> dict[str, Any]:
    return {
        "schema": "nano-lm.frontier.architecture_registry.v1",
        "layers": LAYERS,
        "components": {
            k: {
                **v,
                "failures": [f.value if isinstance(f, FailureCode) else f for f in v.get("failures") or []],
            }
            for k, v in COMPONENTS.items()
        },
        "historical_lessons": {
            k: [c.value for c in v] for k, v in HISTORICAL_LESSONS.items()
        },
        "note": "Active Frontier registry — product/architecture; not Layer-1.",
    }
