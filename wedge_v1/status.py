"""Internal Wedge v1 pipeline status snapshot (not Layer-1 evidence)."""
from __future__ import annotations

import json
import os
from pathlib import Path

from wedge_v1.failure_gallery import resolve_default_gallery, load_gallery_file
from wedge_v1.failure_to_architecture import recommend
from wedge_v1.ingest_sla import measure_ingest_sla
from wedge_v1.lm.admission import evaluate_admission
from wedge_v1.classical.eclass_probes import lm_still_needed, probe_t35, probe_t36, probe_t39
from wedge_v1.runtime import DEFAULT_CORPUS, load_corpus

ROOT = Path(__file__).resolve().parent


def build_status(*, corpus: Path | None = None, demo: bool = False) -> dict:
    corpus_path = corpus
    if demo:
        corpus_path = ROOT / "fixtures" / "owner_corpus"
    elif corpus_path is None:
        env = (os.environ.get("WEDGE_OWNER_CORPUS") or os.environ.get("OWNER_CORPUS") or "").strip()
        if env:
            corpus_path = Path(env).expanduser()

    owner_ready = None
    if corpus_path is not None:
        from wedge_v1.owner_ready import check

        owner_ready = check(corpus_path, demo=demo)

    gal_path = resolve_default_gallery()
    gallery = load_gallery_file(gal_path) if gal_path.is_file() else {}
    evolve = recommend(gallery) if gallery else {}

    docs = load_corpus(corpus_path or DEFAULT_CORPUS)
    still = lm_still_needed([probe_t35(docs), probe_t36(docs), probe_t39(docs)])
    admission = evaluate_admission(
        gallery,
        eclass_lm_still_needed=still,
        owner_corpus_contact=bool(
            corpus_path and owner_ready and owner_ready.get("representative_ready")
        ),
    )

    sla = measure_ingest_sla()
    return {
        "schema": "nano-lm.wedge_v1.runtime_status.v1",
        "corpus": str(corpus_path) if corpus_path else None,
        "owner_ready": owner_ready,
        "gallery_source": str(gal_path) if gal_path.is_file() else None,
        "dogfood_accuracy": gallery.get("accuracy"),
        "evolve": {
            "recommended_next": evolve.get("recommended_next"),
            "lm_admission": evolve.get("lm_admission") or {
                "verdict": admission.get("verdict"),
                "reasons": admission.get("reasons"),
            },
        },
        "lm_admission": admission,
        "ingest_sla": {
            "verdict": sla.get("verdict"),
            "field_recovery": (sla.get("fields_normalized") or {}).get("recovery_rate"),
            "recover_gap_u": sla.get("recover_gap_vs_clean_U"),
        },
        "workstreams": {"W1": "DONE", "W2": "DONE", "W3": "DONE", "W4": "DONE", "W5": "DONE", "W6": admission.get("verdict")},
        "next_action": (
            "Create 10–20 genuine exactly scoped tasks, then run `python -m wedge_v1 study check ...`"
            if not (owner_ready and owner_ready.get("representative_ready"))
            else "Run the frozen study, review every result, then export its aggregate summary"
        ),
        "note": (
            "Internal Wedge v1 pipeline/architecture snapshot; not a Nano "
            "AI-capability or Evidence Ledger claim."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    p = argparse.ArgumentParser(description="Internal Wedge v1 pipeline status")
    p.add_argument("--corpus", type=Path, default=None)
    p.add_argument("--demo", action="store_true")
    p.add_argument("-o", "--output", type=Path, default=None)
    args = p.parse_args(argv)
    out = build_status(corpus=args.corpus, demo=args.demo)
    text = json.dumps(out, indent=2) + chr(10)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0
