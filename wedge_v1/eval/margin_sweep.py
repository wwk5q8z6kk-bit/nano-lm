"""B1 diagnostic: sweep the BM25 promotion margin against the frozen r4 pack.

Recorded evidence (`.studies/p5-repo-papers-20260802-r4/review.json`): all three
OVER_ABSTENTION cards terminated at `bm25_low_margin_review`, and two name the
cause verbatim -- "low-margin retrieval suppressed all literal evidence". This
sweep measures whether `BM25_MARGIN_TAU` is in fact the top cause, and at what
cost.

Two metrics, reported together so a fix cannot be claimed by trading one for
the other:

  recovered  -- of the 3 known-answerable queries that wrongly ABSTAINed at
                tau=0.5, how many produce a span-supported claim at this tau.
  held       -- of the 4 CORRECT_ABSTENTION queries, how many still ABSTAIN.
                This is the guard: a tau that recovers all 3 by answering
                everything has not fixed anything.

Read-only with respect to study artifacts; writes nothing outside stdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1] / ".studies/p5-repo-papers-20260802-r4"
DEFAULT_TAUS = (0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0)


def load_cards() -> tuple[list[dict], list[dict]]:
    """Return (over_abstention_cards, correct_abstention_cards) from the frozen review."""
    review = json.loads((STUDY / "review.json").read_text(encoding="utf-8"))
    over, correct = [], []
    for card in review["cards"].values():
        if card.get("failure_class") == "OVER_ABSTENTION":
            over.append(card)
        elif card.get("usefulness_label") == "CORRECT_ABSTENTION":
            correct.append(card)
    return over, correct


def run_query(corpus: Path, query: str, tau: float) -> dict:
    """Run one ask at a given promotion margin.

    `runtime.ask` exposes no tau parameter and `bm25.top_paragraphs` reads the
    module constant at call time, so the diagnostic sets the constant rather
    than altering runtime signatures.
    """
    from wedge_v1 import runtime
    from wedge_v1.classical import bm25

    original = bm25.BM25_MARGIN_TAU
    try:
        bm25.BM25_MARGIN_TAU = tau
        return runtime.ask(query, corpus, persist_coe=False)
    finally:
        bm25.BM25_MARGIN_TAU = original


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taus", type=float, nargs="*", default=list(DEFAULT_TAUS))
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    over, correct = load_cards()
    if not over:
        print("no OVER_ABSTENTION cards found; nothing to diagnose")
        return 1

    corpus = Path(over[0]["corpus"])
    rows = []
    for tau in args.taus:
        recovered, held, errors = 0, 0, []
        for card in over:
            try:
                out = run_query(corpus, card["query"], tau)
                if out.get("answer_status") != "ABSTAIN" and out.get("claims"):
                    recovered += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{card['id']}: {type(exc).__name__}: {exc}")
        for card in correct:
            try:
                out = run_query(corpus, card["query"], tau)
                if out.get("answer_status") == "ABSTAIN":
                    held += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{card['id']}: {type(exc).__name__}: {exc}")
        rows.append({
            "tau": tau,
            "recovered": recovered,
            "recoverable": len(over),
            "held": held,
            "correct_abstentions": len(correct),
            "errors": errors,
        })

    if args.json:
        print(json.dumps({"study": STUDY.name, "rows": rows}, indent=2))
        return 0

    print(f"study: {STUDY.name}   corpus: {corpus.name}")
    print(f"{'tau':>6}  {'recovered':>12}  {'held':>16}  errors")
    for r in rows:
        print(
            f"{r['tau']:>6.2f}  {r['recovered']:>4}/{r['recoverable']:<7}"
            f"  {r['held']:>6}/{r['correct_abstentions']:<9}  {len(r['errors'])}"
        )
    print("\nrecovered = wrongly-abstained queries that now answer (higher better)")
    print("held      = correct abstentions still held (must stay at max)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
