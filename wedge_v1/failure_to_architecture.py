"""Map failure-gallery tallies → Active Frontier architecture workstreams (W1–W6).

Product/architecture tooling — not Evidence Core.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
ARCH_DOC = ROOT.parent / "frontier" / "ARCHITECTURE_EVOLUTION.md"

# bucket → workstream IDs
BUCKET_TO_WS: dict[str, list[str]] = {
    "UNSUPPORTED_COMPOSITION": ["W1", "W3"],
    "unsupported_composition": ["W1", "W3"],
    "RETRIEVAL_MISS": ["W1"],
    "WRONG_SPAN_RETRIEVAL": ["W1", "W2"],
    "LOW_MARGIN_RETRIEVAL": ["W1"],
    "OVER_ABSTENTION": ["W1", "W4", "W6"],
    "over_abstention": ["W1", "W4", "W6"],
    "CORRECT_ABSTENTION": [],
    "MULTI_DOC_CONTRADICTION": [],
    "NUMERIC_CONTRADICTION": [],
    "EVIDENCE_ABSENT": ["W1", "W5"],
    "EMPTY_EVIDENCE_REJECTED": ["W2"],

    "low_margin_review": ["W1"],
    "ok_with_low_margin_review": ["W1"],
    "over_abstain": ["W1", "W4", "W6"],
    "over_abstain_or_unsupported": ["W1", "W4", "W6"],
    "wrong_or_miss_needle": ["W1", "W2"],
    "wrong_or_empty_span": ["W2"],
    "silent_miss": ["W1", "W2"],
    "under_abstain": ["W2", "W3"],
    "status_mismatch": ["W3", "W4"],
    "contradicted": [],
    "ok_contradicted": [],
    "ok_supported": [],
    "ok_abstain": [],
    "no_corpus": ["W5"],
    "ingestion_layout_failure": ["W5"],
}

WS_BLURB = {
    "W1": "Retrieval-margin + miss taxonomy (BM25 gate, locate vs absent)",
    "W2": "Evidence-atom hard gate (no empty PRESENT)",
    "W3": "Corpus-agnostic multi-doc epistemic merge",
    "W4": "Pluggable E-class cascade (no fixture doc-id control flow)",
    "W5": "Ingest SLA before intelligence (OCR/PDF normalize)",
    "W6": "Marginal model value (only after irreducible abstain)",
}



def _owner_contact_done() -> bool:
    """True when env points at a usable private corpus and reviews have labels."""
    import os
    from wedge_v1.owner_ready import check as owner_check
    from wedge_v1.review import REVIEW_PATH, load_state

    env = (os.environ.get("OWNER_CORPUS") or os.environ.get("WEDGE_OWNER_CORPUS") or "").strip()
    if not env:
        return False
    try:
        ready = owner_check(Path(env))
    except Exception:
        return False
    if not ready.get("ready_for_private_run"):
        return False
    state = load_state(REVIEW_PATH)
    return bool(state.get("labels"))

def recommend(gallery: dict[str, Any]) -> dict[str, Any]:
    tallies = dict(gallery.get("tallies") or {})
    # dogfood gallery uses buckets:{name:[ids]}
    if "buckets" in gallery and not tallies:
        for k, ids in (gallery.get("buckets") or {}).items():
            tallies[k] = len(ids) if isinstance(ids, list) else int(ids or 0)
    # fine_counts + failure_code tallies
    for k, n in (gallery.get("fine_counts") or {}).items():
        tallies[k] = tallies.get(k, 0) + int(n or 0)
    for k, n in (gallery.get("failure_code_tallies") or {}).items():
        tallies[k] = tallies.get(k, 0) + int(n or 0)
    votes: dict[str, int] = {w: 0 for w in WS_BLURB}
    for bucket, n in tallies.items():
        for ws in BUCKET_TO_WS.get(bucket, []):
            votes[ws] += int(n)
    # Always keep W6 last unless abstain dominates
    ranked = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
    next_ws = [w for w, v in ranked if v > 0]
    from wedge_v1.classical.eclass_probes import lm_still_needed, probe_t35, probe_t36, probe_t39
    from wedge_v1.lm.admission import evaluate_admission
    from wedge_v1.runtime import DEFAULT_CORPUS, load_corpus

    docs = load_corpus(DEFAULT_CORPUS)
    still = lm_still_needed([probe_t35(docs), probe_t36(docs), probe_t39(docs)])
    adm = evaluate_admission(gallery, eclass_lm_still_needed=still)
    if adm.get("lm_probe_indicated"):
        next_ws = (["W6"] + [w for w in next_ws if w != "W6"])[:3]
    elif not next_ws:
        # All-green synthetic gallery → owner-corpus contact, not endless W3.
        next_ws = ["W6"] if still else []
    return {
        "schema": "nano-lm.frontier.failure_to_architecture.v1",
        "tallies": tallies,
        "workstream_votes": votes,
        "recommended_next": (
            next_ws[:3]
            if next_ws
            else (
                ["OWNER_USEFULNESS_LOOP"]
                if _owner_contact_done()
                else ["OWNER_CORPUS_CONTACT"]
            )
        ),
        "blurbs": (
            {w: WS_BLURB[w] for w in next_ws[:3]}
            if next_ws
            else (
                {"OWNER_USEFULNESS_LOOP": "Habit + label triage on private corpus; fix observed fails"}
                if _owner_contact_done()
                else {"OWNER_CORPUS_CONTACT": "Real private folder (≥10 docs) + usefulness labels"}
            )
        ),
        "lm_admission": {
            "verdict": adm.get("verdict"),
            "lm_probe_indicated": adm.get("lm_probe_indicated"),
            "reasons": adm.get("reasons"),
            "execute_auth": adm.get("execute_auth"),
        },
        "architecture_doc": str(ARCH_DOC),
        "note": "Proving-ground recommendation — not a Layer-1 claim.",
    }


def load_gallery(path: Path) -> dict:
    from wedge_v1.failure_gallery import load_gallery_file

    return load_gallery_file(path)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    p = argparse.ArgumentParser(description="Failure → architecture workstream recommender")
    p.add_argument(
        "--gallery",
        type=Path,
        default=None,
        help="Dogfood or live gallery JSON",
    )
    p.add_argument("--live", nargs="*", help="Optional live questions to classify via ask()")
    p.add_argument("--corpus", type=Path, default=None)
    args = p.parse_args(argv)
    from wedge_v1.failure_gallery import resolve_default_gallery

    if args.live is not None and len(args.live) > 0:
        from wedge_v1.failure_gallery import run_gallery

        g = run_gallery(args.live, corpus_dir=args.corpus)
    elif (args.gallery or resolve_default_gallery()).exists():
        g = load_gallery(args.gallery or resolve_default_gallery())
    else:
        # default live probes on synthetic corpus
        from wedge_v1.failure_gallery import run_gallery

        g = run_gallery(
            [
                "How long before cache entries expire?",
                "What metformin dose is stated?",
                "capital of Mars colonies in 3100",
            ],
            corpus_dir=args.corpus,
        )

    out = recommend(g)
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write(chr(10))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
