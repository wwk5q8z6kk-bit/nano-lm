#!/usr/bin/env python3
"""Three-axis P1 diagnostic: selection x delimitation x calibration.

A single span score cannot say which competence failed, and this program has
now twice been misled by treating it as if it could — `dc3b310`'s 83% (mostly
leakage) and the 16/192 headline first read as "the model reads poorly" when it
locates the right evidence 79% of the time. The axes are separable and are
separated here:

  SELECTION      did the model reach the right evidence at all?
  DELIMITATION   given it reached it, did it return the right extent?
  CALIBRATION    given absent or unreachable evidence, did it decline?

The third axis is not decoration. The leakage study showed a manipulation that
raised coverage AND grounded assertions AND overall precision while raising
ungrounded assertions faster — an unambiguous improvement on every aggregate
that was a regression in the property the program exists to establish. Any arm
scored without it can buy its gains by asserting more.

Reads the committed per-slot exports in `artifacts/slots/`, so every number
re-derives from the versioned trail rather than from a live log.

Usage:
  python3 nanoscribe/triaxial.py                     # all exported cells
  python3 nanoscribe/triaxial.py --cells ed-A-freeform ed-B-menu
  python3 nanoscribe/triaxial.py --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from math import sqrt
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
sys.path[:] = [p for p in sys.path if p != str(Path(__file__).resolve().parent)]

SLOTS_DIR = Path(_repo_root, "artifacts", "slots")


def _norm(text: str | None) -> str:
    cleaned = "".join(
        c if c.isalnum() or c.isspace() else " " for c in (text or "")
    )
    return " ".join(cleaned.casefold().split())


_OFFSET_RE = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]")


def source_text(encounter_id: str) -> str | None:
    from nanoscribe.campaign_datasets import case_for
    from nanoscribe.campaign_instances import split_encounter_id

    base, instance_id = split_encounter_id(encounter_id)
    try:
        return case_for(base, instance_id).model_input.source.text
    except Exception:
        return None


def native_span(slot: dict, arm: str | None) -> str | None:
    """The span the model actually designated, in whatever format the arm uses.

    An arm that answers with a character range or a menu index never emits a
    quote, so a quote-based selection test returns False for every slot in it —
    0/192 with a confidence interval, indistinguishable from a real result and
    caused entirely by the scorer. That is the same
    cell-invariant-by-construction defect the adversarial-baseline invariant
    exists to ban, so it is resolved here rather than reported.
    """
    quotes = slot.get("parsed_quotes") or []
    if quotes:
        return " ".join(quotes)
    if arm == "offsets":
        match = _OFFSET_RE.search(slot.get("raw_line") or "")
        if not match:
            return None
        start, end = int(match.group(1)), int(match.group(2))
        text = source_text(slot["encounter_id"])
        if text is None or not (0 <= start < end <= len(text)):
            return None
        return text[start:end]
    return None


SCORABLE_ARMS = {None, "free_form", "offsets"}


AXES = ("selection", "delimitation", "calibration")


def axis_eligibility(doc: dict) -> dict[str, tuple[bool, str]]:
    """THE CREDIT GUARD, enforced rather than remembered.

        No intervention receives credit for an axis if it changes the answer
        space, permits or disallows abstention, or supplies information that
        makes that axis mechanically easier.

    Three arms in this program have already earned an apparent gain by one of
    those routes. The C1 answer template handed over the exact string and
    doubled exact_gold_span. The menu restricted the answer space to a
    candidate list, turning selection into discrimination among supplied spans
    — and removed the option to decline, so its perfect coverage is a property
    of the format. Reading any of those as capability is the failure this
    program keeps re-encountering, so the check is mechanical: each clause is
    computed from the run's own committed slots, and an ineligible axis reports
    no rate at all rather than a number with a caveat attached.
    """
    arm = doc.get("arm")
    slots = doc["slots"]
    n = len(slots)
    abstained = sum(1 for s in slots if s["joint_cell"].startswith("abstained"))
    in_template = sum(1 for s in slots if s.get("gold_value_in_answer_template"))
    in_question = sum(1 for s in slots if s.get("gold_value_in_question"))

    out: dict[str, tuple[bool, str]] = {}

    why: list[str] = []
    if arm == "menu":
        why.append(
            "answer space restricted to a supplied candidate list — selection "
            "becomes discrimination among given spans, not search of the source"
        )
    if in_question:
        why.append(f"gold value supplied in the question on {in_question}/{n} slots")
    out["selection"] = (not why, "; ".join(why))

    why = []
    if in_template:
        why.append(
            f"gold value supplied in the answer template on {in_template}/{n} "
            "slots — the boundaries are handed over, not produced"
        )
    if arm in ("menu", "offsets"):
        why.append(
            f"arm {arm!r} changes the response type, so boundary generation is "
            "not exercised as it is in the control"
        )
    out["delimitation"] = (not why, "; ".join(why))

    why = []
    if abstained == 0 and n:
        why.append(
            "no slot abstained in the entire run — the format appears to "
            "disallow declining, so abstention behaviour cannot be attributed "
            "to the model"
        )
    out["calibration"] = (not why, "; ".join(why))
    return out


def selection_scorable(doc: dict) -> tuple[bool, str]:
    """Can this arm's outputs be resolved to spans at all?"""
    arm = doc.get("arm")
    if arm == "menu":
        return (
            False,
            "menu indices are not resolvable without the per-slot menu, which "
            "the run payload does not carry; see the concurrent session's "
            "hand-resolution (LOCATED 30/120 = 25%)",
        )
    resolved = sum(1 for s in doc["slots"] if native_span(s, arm) is not None)
    if resolved == 0:
        return (False, f"arm {arm!r}: no slot resolved to a span")
    return (True, "")


def located(slot: dict, arm: str | None = None) -> bool:
    """Model's quote CONTAINS the gold span — right evidence, any extent.

    Deliberately generous: the question this axis answers is whether the model
    reached the evidence, so a long quote that contains it counts. That makes
    this an UPPER bound on selection; the disjoint rate is the matching lower
    bound on outright retrieval failure, and both are reported.
    """
    if not slot["gold_present"]:
        return False
    span = native_span(slot, arm)
    if span is None:
        return False
    gold = _norm(slot.get("gold_value"))
    return bool(gold) and gold in _norm(span)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def score(slots: list[dict], arm: str | None = None, scorable: bool = True) -> dict:
    present = [s for s in slots if s["gold_present"]]
    absent = [s for s in slots if not s["gold_present"]]

    loc = [s for s in present if located(s, arm)]
    grounded = [s for s in present if s["joint_cell"] == "asserted_grounded"]
    disjoint = [
        s
        for s in present
        if native_span(s, arm) is not None and not located(s, arm)
    ]

    # Calibration axis. Assertions on absent slots are commissions by
    # construction — there is nothing there to be right about.
    abstained_absent = [s for s in absent if s["joint_cell"] == "abstained_correct"]
    asserted_absent = [s for s in absent if s["joint_cell"].startswith("asserted")]
    asserted_all = [s for s in slots if s["joint_cell"].startswith("asserted")]
    unbound_all = [s for s in slots if s["joint_cell"] == "asserted_unbound"]

    def rate(k: int, n: int) -> dict:
        lo, hi = wilson(k, n)
        return {"k": k, "n": n, "rate": (k / n) if n else 0.0, "ci95": [lo, hi]}

    return {
        "selection_scorable": scorable,
        # AXIS 1 — selection
        "selection_recall": rate(len(loc), len(present)) if scorable else None,
        "selection_disjoint": rate(len(disjoint), len(present)) if scorable else None,
        # AXIS 2 — delimitation, conditional on having reached the evidence
        "delimitation_given_location": (
            rate(len(grounded), len(loc)) if scorable else None
        ),
        # end-to-end, for reference only
        "grounded_of_present": rate(len(grounded), len(present)),
        # AXIS 3 — calibration
        "abstention_on_absent": rate(len(abstained_absent), len(absent)),
        "commission_on_absent": rate(len(asserted_absent), len(absent)),
        "unbound_of_asserted": rate(len(unbound_all), len(asserted_all)),
        "coverage": rate(len(asserted_all), len(slots)),
        # secondary
        "exact_gold_span": rate(
            sum(1 for s in slots if s["exact_gold_span"]), len(slots)
        ),
    }


def load(cell: str) -> dict:
    return json.loads((SLOTS_DIR / f"slots-{cell}.json").read_text())


def available_cells() -> list[str]:
    return sorted(p.stem.removeprefix("slots-") for p in SLOTS_DIR.glob("slots-*.json"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    cells = args.cells or available_cells()
    out = {}
    for cell in cells:
        doc = load(cell)
        ok, why = selection_scorable(doc)
        elig = axis_eligibility(doc)
        axes = score(doc["slots"], doc.get("arm"), ok)
        # Credit guard: an ineligible axis reports nothing, not a caveated rate.
        if not elig["selection"][0]:
            axes["selection_recall"] = None
            axes["selection_disjoint"] = None
        if not elig["delimitation"][0]:
            axes["delimitation_given_location"] = None
        if not elig["calibration"][0]:
            axes["abstention_on_absent"] = None
            axes["commission_on_absent"] = None
        out[cell] = {
            "run_id": doc["run_id"],
            "arm": doc.get("arm"),
            "selection_unscorable_reason": None if ok else why,
            "eligibility": {k: {"eligible": v[0], "reason": v[1]} for k, v in elig.items()},
            "axes": axes,
        }

    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0

    print("THREE-AXIS P1 DIAGNOSTIC — selection x delimitation x calibration")
    print("all rates with 95% Wilson intervals; source: artifacts/slots/\n")
    head = (
        f"{'cell':<16}{'SELECT':>18}{'DELIM|loc':>18}"
        f"{'ABSTAIN(abs)':>18}{'UNBOUND|asrt':>18}{'exact':>9}"
    )
    print(head)
    print("-" * len(head))
    for cell in cells:
        a = out[cell]["axes"]

        def f(key: str) -> str:
            r = a[key]
            if r is None:
                return f"{'n/a':>17}"
            return f"{r['rate']:>6.1%} [{r['ci95'][0]:.0%},{r['ci95'][1]:.0%}]"

        print(
            f"{cell:<16}{f('selection_recall'):>18}"
            f"{f('delimitation_given_location'):>18}"
            f"{f('abstention_on_absent'):>18}"
            f"{f('unbound_of_asserted'):>18}"
            f"{a['exact_gold_span']['rate']:>8.1%}"
        )
    notes = [(c, out[c]["selection_unscorable_reason"]) for c in cells
             if out[c]["selection_unscorable_reason"]]
    if notes:
        print("\nSELECTION AXIS NOT SCORABLE (format cannot be resolved to spans):")
        for cell, why in notes:
            print(f"  {cell}: {why}")

    denied = [
        (c, axis, out[c]["eligibility"][axis]["reason"])
        for c in cells
        for axis in AXES
        if not out[c]["eligibility"][axis]["eligible"]
    ]
    if denied:
        print("\nCREDIT GUARD — axis denied, no rate reported:")
        for cell, axis, why in denied:
            print(f"  {cell:<16} {axis:<13} {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
