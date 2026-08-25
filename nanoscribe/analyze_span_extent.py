#!/usr/bin/env python3
"""Split the non-grounded span-port slots into LOCATED vs NOT-LOCATED.

The landed result reports `asserted_grounded` 16/192 against a perfect-reader
ceiling of 120 and reads it as "the model reads, but poorly". That reading
conflates two different bottlenecks:

  - **retrieval failure** — the model quoted the wrong part of the transcript,
    or quoted nothing that binds. It did not find the evidence.
  - **delimitation failure** — the model quoted a span that *contains* the gold
    span, e.g. `STATED: "My calf has been aching for three days."` when the gold
    span is `calf`. It found the right evidence and returned the wrong extent.

These point at different next experiments (retrieval capacity vs span-boundary
supervision), and the current artifact cannot tell them apart, because
`exact_gold_span` and `asserted_grounded` are both exact-extent predicates that
score an over-extended quote and a wrong-sentence quote identically: zero.

This script joins each run's per-slot `raw_line` against the gold evidence text
reconstructed from the run's own commit, and reports containment as a distinct
category.

Usage:
    python3 nanoscribe/analyze_span_extent.py \
        --log /tmp/lk_analysis/L000.json --gold-tree /tmp/lk_gold
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any

_PUNCT = " \t\r\n.,;:!?\"'`‘’“”"


def _norm(text: str) -> str:
    """Casefold + NFKC + strip edge punctuation + collapse internal whitespace.

    Deliberately looser than the scorer's exact-offset comparison: the question
    here is 'is the gold string present inside what the model returned', which
    must not be defeated by a trailing period or a curly quote.
    """
    t = unicodedata.normalize("NFKC", text).casefold()
    t = " ".join(t.split())
    return t.strip(_PUNCT)


def _gold_by_slot(gold_tree: str) -> dict[str, dict[str, str]]:
    """(instance/atom_id) -> {gold_span, raw_value} from the run's own commit."""
    sys.path.insert(0, gold_tree)
    from nanoscribe.campaign_datasets import campaign_cases
    from nanoscribe.campaign_instances import split_encounter_id

    out: dict[str, dict[str, str]] = {}
    for case in campaign_cases("campaign_v2"):
        _base, inst = split_encounter_id(case.encounter_id)
        ev_by_id = {ev.evidence_id: ev.text for ev in case.gold.evidence}
        for atom in case.gold.atoms:
            spans = [ev_by_id[e] for e in atom.evidence_ids if e in ev_by_id]
            if not spans:
                continue
            out[f"{inst}/{atom.atom_id}"] = {
                "gold_span": spans[0],
                "raw_value": atom.raw_value or "",
            }
    return out


def _model_quote(raw_line: str, gold_tree: str) -> str | None:
    sys.path.insert(0, gold_tree)
    from nanoscribe.adapt import parse_label_and_quotes

    _label, quotes = parse_label_and_quotes(raw_line)
    return quotes[0] if quotes else None


def classify(per_atom: dict[str, Any], gold: dict[str, dict[str, str]], gold_tree: str):
    rows = []
    for slot, rec in per_atom.items():
        g = gold.get(slot)
        quote = _model_quote(rec["raw_line"], gold_tree)
        nq = _norm(quote) if quote else ""
        ng = _norm(g["gold_span"]) if g else ""

        if not rec["gold_present"] or not g:
            category = "no_gold_span"            # enc-4 absent slots etc.
        elif rec["cell"] == "asserted_grounded":
            category = "grounded_exact"
        elif not quote:
            category = "no_quote"                # abstained or bare label
        elif ng and ng in nq:
            # The model returned a span that CONTAINS the gold span.
            category = "located_over_extended"
        elif nq and nq in ng:
            category = "located_under_extended"
        else:
            category = "not_located"
        rows.append(
            {
                "slot": slot,
                "category": category,
                "cell": rec["cell"],
                "gold_span": g["gold_span"] if g else None,
                "model_quote": quote,
                "raw_line": rec["raw_line"],
                "span_character_f1": rec["span_character_f1"],
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True, help="extracted run JSON")
    ap.add_argument("--gold-tree", required=True, help="worktree at the run's commit")
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default="")
    args = ap.parse_args(argv)

    payload = json.loads(Path(args.log).read_text())
    gold = _gold_by_slot(args.gold_tree)
    rows = classify(payload["per_atom"], gold, args.gold_tree)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1

    n_total = len(rows)
    n_gold = sum(1 for r in rows if r["category"] != "no_gold_span")
    n_grounded = counts.get("grounded_exact", 0)
    n_over = counts.get("located_over_extended", 0)
    n_under = counts.get("located_under_extended", 0)
    n_located = n_grounded + n_over + n_under
    n_nongrounded_gold = n_gold - n_grounded

    print("=" * 76)
    print(f"SPAN EXTENT ANALYSIS {args.label}")
    print("=" * 76)
    print(f"  slots total                    {n_total}")
    print(f"  slots with a gold span         {n_gold}   (the ceiling)")
    print("-" * 76)
    for key in (
        "grounded_exact",
        "located_over_extended",
        "located_under_extended",
        "not_located",
        "no_quote",
        "no_gold_span",
    ):
        print(f"  {key:<26} {counts.get(key, 0)}")
    print("-" * 76)
    print(f"  LOCATED (exact + over + under) {n_located}/{n_gold} "
          f"= {n_located / n_gold:.1%} of ceiling")
    print(f"  of which exact extent          {n_grounded}")
    print(f"  of which wrong extent          {n_over + n_under}")
    if n_nongrounded_gold:
        print(f"  share of the {n_nongrounded_gold} non-grounded gold slots that "
              f"CONTAIN the gold span: {n_over}/{n_nongrounded_gold} "
              f"= {n_over / n_nongrounded_gold:.1%}")
    print("=" * 76)

    ex = [r for r in rows if r["category"] == "located_over_extended"][:8]
    if ex:
        print("\n  examples of located-but-over-extended:")
        for r in ex:
            print(f"    gold {r['gold_span']!r}")
            print(f"      -> {r['raw_line'][:78]}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(
                {
                    "label": args.label,
                    "source_log": args.log,
                    "counts": counts,
                    "n_total": n_total,
                    "n_gold_present": n_gold,
                    "n_located": n_located,
                    "located_share_of_ceiling": round(n_located / n_gold, 4),
                    "n_nongrounded_gold": n_nongrounded_gold,
                    "over_extended_share_of_nongrounded": (
                        round(n_over / n_nongrounded_gold, 4) if n_nongrounded_gold else None
                    ),
                    "rows": rows,
                },
                indent=2,
                sort_keys=True,
            )
        )
        print(f"\n  written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
