"""DP-1 confirmation: denial-polarity correction on the calibration partition.

Tests the rule frozen in `papers/PREREG_DENIAL_POLARITY.md` against criteria
fixed before this was run. Reuses the loader and shared inference authority the
trainer uses (`run_threshold_sweep._load`), and scores with the authority's own
comparator (`evidence_query_inference._proposal_exact`) under the authority's
own positional alignment, so the numbers here are commensurable with H6's.

Scope limit, recorded before measuring: `contract._DENIAL_PATTERNS` covers only
MEDICATION and ALLERGY. The rule cannot fire on chief_complaint, duration, or
severity, so absent errors in those fields are structurally unrecoverable by it.
Per-field firing is reported so this is visible rather than absorbed.

    python3 -m nano_ai.training.run_denial_probe \
        --checkpoint artifacts/nano_h6/kaggle/results-20260805/results/seed-20260805/epoch-2.pt \
        --calibration artifacts/nano_h5/data \
        --output artifacts/nano_h6/analysis/denial_probe_calibration.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import Counter
from pathlib import Path

import torch

from nano_ai.contract import FieldState, _is_field_denial

# Frozen in papers/PREREG_DENIAL_POLARITY.md §4 before measurement.
C1_RECOVERY_FRACTION = 0.60
C4_SPECIFICITY = 0.90

_OTHER_STATES = (FieldState.CONFLICTING, FieldState.UNCERTAIN, FieldState.MISSING)


def _fires(proposal) -> bool:
    """The rule under test: a SUPPORTED proposal whose evidence denies."""
    return proposal.state is FieldState.SUPPORTED and any(
        _is_field_denial(proposal.field, span.text) for span in proposal.spans
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("sft/tokenizer.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    from nano_ai.training.evidence_query_inference import (
        _proposal_exact,
        batched_evidence_query_inference,
    )
    from nano_ai.training.run_threshold_sweep import _load

    model, inputs, gold = _load(
        args.checkpoint, args.calibration, args.tokenizer, args.device
    )
    with torch.inference_mode():
        inference = batched_evidence_query_inference(
            model, inputs, device=args.device, batch_size=args.batch_size
        )

    before = Counter()
    after = Counter()
    totals = Counter()
    recoverable = 0  # gold absent, predicted supported
    fired_by_gold_state = Counter()
    fired_by_field = Counter()
    absent_gold_by_field = Counter()
    overall_before = overall_after = overall_n = 0

    for prediction, gold_row in zip(inference.predictions, gold, strict=True):
        # Matches the authority: an errored row scores as not-exact on every
        # field rather than dropping out of the denominator.
        proposed = prediction.proposals if prediction.error is None else ()
        for index, gold_proposal in enumerate(gold_row.proposals):
            gold_state = gold_proposal.state
            totals[gold_state] += 1
            overall_n += 1
            if gold_state is FieldState.ABSENT:
                absent_gold_by_field[gold_proposal.field.value] += 1

            was_exact = bool(proposed) and _proposal_exact(
                proposed[index], gold_proposal
            )
            before[gold_state] += was_exact
            overall_before += was_exact

            now_exact = was_exact
            if proposed and _fires(proposed[index]):
                corrected = dataclasses.replace(
                    proposed[index], state=FieldState.ABSENT, state_code="A"
                )
                now_exact = _proposal_exact(corrected, gold_proposal)
                fired_by_gold_state[gold_state] += 1
                fired_by_field[proposed[index].field.value] += 1
            after[gold_state] += now_exact
            overall_after += now_exact

            if (
                gold_state is FieldState.ABSENT
                and proposed
                and proposed[index].state is FieldState.SUPPORTED
            ):
                recoverable += 1

    absent, supported = FieldState.ABSENT, FieldState.SUPPORTED
    fired_total = sum(fired_by_gold_state.values())
    specificity = fired_by_gold_state[absent] / fired_total if fired_total else 1.0
    c1_required = before[absent] + C1_RECOVERY_FRACTION * recoverable

    criteria = {
        "C1_absent_recovery": {
            "before": before[absent],
            "after": after[absent],
            "of_total": totals[absent],
            "recoverable_population": recoverable,
            "required_at_least": round(c1_required, 1),
            "passed": after[absent] >= c1_required,
        },
        "C2_supported_no_regression": {
            "before": before[supported],
            "after": after[supported],
            "of_total": totals[supported],
            "false_flips": max(0, before[supported] - after[supported]),
            "passed": after[supported] >= before[supported],
        },
        "C3_other_states_unchanged": {
            "states": {
                state.value: {"before": before[state], "after": after[state]}
                for state in _OTHER_STATES
            },
            "passed": all(before[s] == after[s] for s in _OTHER_STATES),
        },
        "C4_rule_specificity": {
            "fired_total": fired_total,
            "fired_with_gold_absent": fired_by_gold_state[absent],
            "specificity": round(specificity, 4),
            "required_at_least": C4_SPECIFICITY,
            "passed": specificity >= C4_SPECIFICITY,
        },
    }

    payload = {
        "schema": "nano.denial-polarity-probe.v1",
        "preregistration": "papers/PREREG_DENIAL_POLARITY.md",
        "partition": "calibration (development not opened)",
        "checkpoint": str(args.checkpoint),
        "criteria": criteria,
        "verdict": "ACCEPT" if all(c["passed"] for c in criteria.values()) else "REJECT",
        "descriptive": {
            "note": "reported for completeness; not part of the frozen criteria",
            "overall_joint_before": overall_before,
            "overall_joint_after": overall_after,
            "overall_fields": overall_n,
            "totals_by_gold_state": {s.value: n for s, n in totals.items()},
            "rule_fired_by_field": dict(fired_by_field),
            "absent_gold_by_field": dict(absent_gold_by_field),
            "fields_with_denial_patterns": ["medication", "allergy"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    d = payload["descriptive"]
    print(f"fields: {overall_n}   gold states: {d['totals_by_gold_state']}")
    print(f"overall joint: {overall_before} -> {overall_after}")
    print(f"absent gold by field: {d['absent_gold_by_field']}")
    print(f"rule fired by field: {d['rule_fired_by_field']}\n")
    for name, data in criteria.items():
        print(f"{name:32s} passed={data['passed']}")
    print(f"\nVERDICT: {payload['verdict']}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
