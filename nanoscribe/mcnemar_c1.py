#!/usr/bin/env python3
"""Exact McNemar for the C1 contrast, on the shared slot set.

The landed artifact reports C1 as a paired t-test at df=11 over 12 per-instance
counts. That treats the instance mean as the unit and assumes normality of a
count difference with sd~1 — but the per-slot outcome is BINARY over a slot set
that is identical in both cells under greedy decoding. That is a within-item
paired-binary design, whose exact test is McNemar: conditional on d discordant
slots the null is Binomial(d, 1/2), which needs no normality assumption and uses
the slot, not the instance mean, as the unit.

Clustering: slots are nested in encounters (m~3.2 slots/encounter). The exact
test below ignores that, so its p-values are anti-conservative. A Kish DEFF of
1.44 is applied to the discordant count as a sensitivity check; where an effect
survives both, clustering is not the deciding factor.
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value. b, c are the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = max(b, c)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / (2**n)
    return min(1.0, 2 * tail)


def contrast(pa: dict, pb: dict, predicate) -> dict:
    """pa = reference cell, pb = treatment cell."""
    b = c = 0
    for slot in pa:
        x, y = predicate(pa[slot]), predicate(pb[slot])
        if y and not x:
            b += 1          # treatment gains
        elif x and not y:
            c += 1          # treatment loses
    p = mcnemar_exact(b, c)
    # Clustering sensitivity: shrink the discordant counts by DEFF.
    deff = 1.44
    p_cl = mcnemar_exact(int(b / deff), int(c / deff))
    return {
        "n_ref": sum(1 for s in pa if predicate(pa[s])),
        "n_trt": sum(1 for s in pb if predicate(pb[s])),
        "gains_b": b,
        "losses_c": c,
        "p_exact": p,
        "p_clustered_deff1.44": p_cl,
        "direction": "treatment higher" if b > c else ("treatment lower" if c > b else "tie"),
    }


CELLS = {
    "asserted_grounded": lambda v: v["cell"] == "asserted_grounded",
    "asserted_unbound": lambda v: v["cell"] == "asserted_unbound",
    "abstained_correct": lambda v: v["cell"] == "abstained_correct",
    "assertion_state_correct": lambda v: v["assertion_state_correct"],
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True, help="reference cell JSON (C1 off)")
    ap.add_argument("--trt", required=True, help="treatment cell JSON (C1 on)")
    ap.add_argument("--ref-extent", help="span-extent JSON for ref (adds LOCATED)")
    ap.add_argument("--trt-extent", help="span-extent JSON for trt")
    ap.add_argument("--label", default="")
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    A = json.loads(Path(args.ref).read_text())
    B = json.loads(Path(args.trt).read_text())
    pa, pb = A["per_atom"], B["per_atom"]
    assert set(pa) == set(pb), "slot sets differ — contrast is not paired"

    results = {name: contrast(pa, pb, fn) for name, fn in CELLS.items()}

    if args.ref_extent and args.trt_extent:
        la = {r["slot"]: r for r in json.loads(Path(args.ref_extent).read_text())["rows"]}
        lb = {r["slot"]: r for r in json.loads(Path(args.trt_extent).read_text())["rows"]}
        located = {"grounded_exact", "located_over_extended", "located_under_extended"}
        results["LOCATED (any extent)"] = contrast(
            la, lb, lambda r: r["category"] in located
        )

    print("=" * 88)
    print(f"EXACT McNEMAR — C1 contrast {args.label}")
    print(f"  ref = {A['condition']}   trt = {B['condition']}   n slots = {len(pa)}")
    print("=" * 88)
    print(f"  {'outcome':<26} {'ref':>5} {'trt':>5} {'b':>5} {'c':>5} "
          f"{'p_exact':>10} {'p_clust':>9}  direction")
    print("-" * 88)
    for name, r in results.items():
        print(
            f"  {name:<26} {r['n_ref']:>5} {r['n_trt']:>5} {r['gains_b']:>5} "
            f"{r['losses_c']:>5} {r['p_exact']:>10.2e} "
            f"{r['p_clustered_deff1.44']:>9.2e}  {r['direction']}"
        )
    print("=" * 88)

    g, u = results["asserted_grounded"], results["asserted_unbound"]
    co = (g["direction"] == u["direction"]) and g["direction"] != "tie"
    print(f"\n  CO-MOVEMENT RULE (PREREG §5): grounded {g['direction']}, "
          f"unbound {u['direction']}")
    print(f"  -> {'FIRES — UNRESOLVED (coverage shift)' if co else 'does not fire'}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"label": args.label, "ref": A["condition"], "trt": B["condition"],
             "n_slots": len(pa), "results": results,
             "co_movement_fires": co}, indent=2, sort_keys=True))
        print(f"\n  written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
