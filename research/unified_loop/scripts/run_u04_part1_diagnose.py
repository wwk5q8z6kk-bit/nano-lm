#!/usr/bin/env python3
"""EXP-U04 Part 1: Exact slot-level taxonomy of the 76/276 Gold x Real Verifier drops and 42/276 menu misses."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

WORKTREE_ROOT = Path(__file__).resolve().parents[3]
MAIN_REPO = Path("/Users/mac/Projects/nano-lm")
for p in (str(WORKTREE_ROOT), str(MAIN_REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluations.v2 import scribe_custody  # noqa: E402
from nano_ai.contract import FieldState  # noqa: E402
from nanoscribe.check_note import transcript_source  # noqa: E402
from nanoscribe.grounded import (  # noqa: E402
    grounded_candidate_choices,
    parse_grounded_summary,
    verify_grounded_span,
)
from nanoscribe.scribe import compact_solver, scribe  # noqa: E402

OUT_DIR = WORKTREE_ROOT / "research" / "unified_loop" / "experiments" / "U04_nanoscribe_verifier_qwen"
COHORTS = [
    ("C12_independent_dev_20261023", MAIN_REPO / "artifacts" / "p1_independent_dev_20261023"),
    ("C13_scope_validation_20261023", MAIN_REPO / "artifacts" / "p1_scope_validation_20261023"),
    ("C14_retention_validation_20261026", MAIN_REPO / "artifacts" / "p1_retention_validation_20261026"),
]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    drop_reasons = Counter()
    drop_by_cohort_field = Counter()
    menu_miss_by_cohort_field = Counter()
    drop_examples: list[dict[str, Any]] = []
    menu_miss_examples: list[dict[str, Any]] = []
    total_asserted = 0
    retained_asserted = 0
    menu_covered = 0

    for cname, cdir in COHORTS:
        sources = json.loads((cdir / "sources.json").read_text(encoding="utf-8"))
        gold_by_id = {g["case_id"]: g["fields"] for g in json.loads((cdir / "gold.json").read_text(encoding="utf-8"))}

        for case in sources:
            cid = case["case_id"]
            transcript = case["transcript"]
            source = transcript_source(transcript, encounter_id=cid)
            gfields = gold_by_id[cid]

            gold_raw = scribe_custody.serialize_gold_summary(transcript, gfields, policy="grounded")
            parsed_gold = parse_grounded_summary(transcript, gold_raw, source=source)
            draft = scribe(
                transcript,
                encounter_id=cid,
                solver=compact_solver(lambda _, r=gold_raw: r, binder="grounded"),
                context_policy="grounded",
            )
            claims_by_field = {c["field"]: c for c in draft["note"]["claims"]} if draft.get("note") else {}

            used_menu: set[str] = set()
            for label, field in scribe_custody.LABELS.items():
                gspec = gfields[label]
                gspans = scribe_custody.locate_gold_spans(transcript, gspec.get("spans", ()))
                # Check menu coverage in exact LABEL order
                choices = grounded_candidate_choices(transcript, field, used_menu)
                cov = scribe_custody.menu_covers_gold(transcript, field, gspec, choices)
                if gspec["state"] in ("S", "A"):
                    total_asserted += 1
                    if cov:
                        menu_covered += 1
                    else:
                        menu_miss_by_cohort_field[f"{cname}:{label}:{gspec['state']}"] += 1
                        menu_miss_examples.append({
                            "cohort": cname,
                            "case_id": cid,
                            "field": label,
                            "gold_state": gspec["state"],
                            "gold_spans": [sp.text for sp in gspans],
                            "choices": choices,
                        })

                cl = claims_by_field.get(field.value)
                matched = cl is not None and scribe_custody._match_claim_to_gold(cl, gspec, gspans)
                if gspec["state"] in ("S", "A"):
                    if matched:
                        retained_asserted += 1
                    else:
                        # Trace exact per-span verifier reasons on the parsed_gold slot!
                        gslot = parsed_gold.fields[field]
                        span_reasons = [
                            {"span": sp.text, "start": sp.start, "end": sp.end, "verify": verify_grounded_span(source, field, gslot.state, sp)}
                            for sp in gslot.spans
                        ]
                        reason_tag = "|".join(f"{r[1]}" for r in (sr["verify"] for sr in span_reasons)) or f"claim:{cl.get('state') if cl else 'none'}:{cl.get('abstain_reason') if cl else 'none'}"
                        drop_reasons[reason_tag] += 1
                        drop_by_cohort_field[f"{cname}:{label}:{gspec['state']}"] += 1
                        # Find turn text enclosing first span
                        turn_snippets = []
                        for sp in gslot.spans:
                            for t in source.turns:
                                if t.start <= sp.start and sp.end <= t.end:
                                    turn_snippets.append(f"[{t.speaker.value}] {t.text}")
                        drop_examples.append({
                            "cohort": cname,
                            "case_id": cid,
                            "field": label,
                            "gold_state": gspec["state"],
                            "gold_spans": [sp.text for sp in gspans],
                            "serialized_gold_raw": gold_raw,
                            "claim_state": cl.get("state") if cl else None,
                            "claim_abstain": cl.get("abstain_reason") if cl else None,
                            "span_reasons": span_reasons,
                            "turns": turn_snippets,
                        })

                # Replicate oracle choice consumption from evaluate_cohort_2x2 if helpful
                if choices:
                    used_menu.add(choices[0])

    summary = {
        "total_asserted": total_asserted,
        "retained_asserted": retained_asserted,
        "dropped_asserted": total_asserted - retained_asserted,
        "menu_covered_asserted": menu_covered,
        "menu_missed_asserted": total_asserted - menu_covered,
        "drop_reasons_counts": dict(drop_reasons.most_common()),
        "drop_by_cohort_field": dict(drop_by_cohort_field.most_common()),
        "menu_miss_by_cohort_field": dict(menu_miss_by_cohort_field.most_common()),
        "drop_examples": drop_examples,
        "menu_miss_examples": menu_miss_examples,
    }
    out_path = OUT_DIR / "part1_diagnostics.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[U04-P1] asserted={total_asserted}, retained={retained_asserted}, dropped={total_asserted - retained_asserted}")
    print("[U04-P1] drop_reasons:", dict(drop_reasons.most_common()))
    print("[U04-P1] drop_by_cohort_field:", dict(drop_by_cohort_field.most_common()))
    print("[U04-P1] menu_covered:", menu_covered, "missed:", total_asserted - menu_covered, dict(menu_miss_by_cohort_field.most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
