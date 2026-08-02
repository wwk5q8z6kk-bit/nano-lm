"""W6 runner — admission + marginal stub probe (no external LM by default)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from wedge_v1.failure_gallery import load_gallery_file, resolve_default_gallery
from wedge_v1.lm.admission import evaluate_admission
from wedge_v1.lm.marginal import run_marginal_probe

ROOT = Path(__file__).resolve().parent
DEFAULT_GALLERY = resolve_default_gallery()


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="W6 marginal LM probe (gated; stub default)")
    p.add_argument("--gallery", type=Path, default=DEFAULT_GALLERY)
    p.add_argument("--corpus", type=Path, default=None)
    p.add_argument("--admit-only", action="store_true", help="Admission check only")
    p.add_argument("--min-irreducible", type=int, default=2)
    p.add_argument("--owner-corpus", action="store_true", help="Owner-corpus contact flag")
    p.add_argument("--no-persist", action="store_true")
    args = p.parse_args(argv)

    gal_path = args.gallery or DEFAULT_GALLERY
    gallery = load_gallery_file(gal_path) if Path(gal_path).exists() else {}

    if args.admit_only:
        from wedge_v1.classical.eclass_probes import lm_still_needed, probe_t35, probe_t36, probe_t39
        from wedge_v1.runtime import load_corpus, DEFAULT_CORPUS

        docs = load_corpus(args.corpus or DEFAULT_CORPUS)
        still = lm_still_needed([probe_t35(docs), probe_t36(docs), probe_t39(docs)])
        out = evaluate_admission(
            gallery,
            eclass_lm_still_needed=still,
            min_irreducible=args.min_irreducible,
            owner_corpus_contact=args.owner_corpus,
        )
        json.dump(out, sys.stdout, indent=2)
        sys.stdout.write("\n")
        print("W6_ADMIT_DONE")
        return 0

    out = run_marginal_probe(
        gallery=gallery,
        corpus_dir=args.corpus,
        dry_run=True,
        min_irreducible=args.min_irreducible,
        owner_corpus_contact=args.owner_corpus,
        persist=not args.no_persist,
    )
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    print("W6_MARGINAL_PROBE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
