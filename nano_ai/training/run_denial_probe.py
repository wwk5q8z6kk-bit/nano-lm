"""DP-1 confirmation: denial-polarity correction on the calibration partition.

Tests the rule frozen in `papers/PREREG_DENIAL_POLARITY.md` against criteria
fixed before this was run. Reuses the loader, encoder, and shared inference
authority the trainer uses (via `run_threshold_sweep._load`), so the proposals
scored here are the ones the model actually produces.

The rule is applied post-hoc to stored proposals. Nothing in the training or
evaluation authority is modified, no checkpoint is written, and the development
partition is never opened.

    python3 -m nano_ai.training.run_denial_probe \
        --checkpoint artifacts/nano_h6/kaggle/results-20260805/results/seed-20260805/epoch-2.pt \
        --calibration artifacts/nano_h5/data \
        --output artifacts/nano_h6/analysis/denial_probe_calibration.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from nano_ai.contract import FieldState, _is_field_denial

# Frozen in papers/PREREG_DENIAL_POLARITY.md §4 before measurement.
C1_RECOVERY_FRACTION = 0.60
C4_SPECIFICITY = 0.90


def _proposal_matches(proposal, gold) -> bool:
    """Joint-exact: same state, and same normalized spans."""
    if proposal.state != gold.state:
        return False
    got = tuple((s.start, s.end) for s in proposal.spans)
    want = tuple((s.start, s.end) for s in gold.spans)
    return got == want


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
        batched_evidence_query_inference,
    )
    from nano_ai.training.run_threshold_sweep import _load

    model, inputs, gold_rows = _load(
        args.checkpoint, args.calibration, args.tokenizer, args.device
    )
    with torch.inference_mode():
        inference = batched_evidence_query_inference(
            model, inputs, device=args.device, batch_size=args.batch_size
        )

    before = Counter()
    after = Counter()
    totals = Counter()
    mislabelled_as_supported = 0
    rewritten_by_gold_state = Counter()

    for prediction, gold_row in zip(inference.predictions, gold_rows, strict=True):
        if prediction.error is not None or prediction.proposals is None:
            continue
        for proposal, gold in zip(prediction.proposals, gold_row.proposals, strict=True):
            gold_state = gold.state
            totals[gold_state] += 1
            was_exact = _proposal_matches(proposal, gold)
            before[gold_state] += was_exact

            # --- the rule under test -------------------------------------
            rewrite = proposal.state == FieldState.SUPPORTED and any(
                _is_field_denial(proposal.field, span.text) for span in proposal.spans
            )
            # -------------------------------------------------------------

            if rewrite:
                rewritten_by_gold_state[gold_state] += 1
                corrected_state = FieldState.ABSENT
            else:
                corrected_state = proposal.state

            now_exact = was_exact
            if rewrite:
                got = tuple((s.start, s.end) for s in proposal.spans)
                want = tuple((s.start, s.end) for s in gold.spans)
                now_exact = corrected_state == gold_state and got == want
            after[gold_state] += now_exact

            if (
                gold_state == FieldState.ABSENT
                and proposal.state == FieldState.SUPPORTED
            ):
                mislabelled_as_supported += 1

    absent, supported = FieldState.ABSENT, FieldState.SUPPORTED
    rewritten_total = sum(rewritten_by_gold_state.values())
    specificity = (
        rewritten_by_gold_state[absent] / rewritten_total if rewritten_total else 1.0
    )
    c1_required = before[absent] + C1_RECOVERY_FRACTION * mislabelled_as_supported

    criteria = {
        "C1_absent_recovery": {
            "before": before[absent],
            "after": after[absent],
            "recoverable_population": mislabelled_as_supported,
            "required_at_least": round(c1_required, 1),
            "passed": after[absent] >= c1_required,
        },
        "C2_supported_no_regression": {
            "before": before[supported],
            "after": after[supported],
            "passed": after[supported] >= before[supported],
        },
        "C3_other_states_unchanged": {
            state.value: {"before": before[state], "after": after[state]}
            for state in (
                FieldState.CONFLICTING,
                FieldState.UNCERTAIN,
                FieldState.MISSING,
            )
        },
        "C4_rule_specificity": {
            "rewritten_total": rewritten_total,
            "rewritten_with_gold_absent": rewritten_by_gold_state[absent],
            "specificity": round(specificity, 4),
            "required_at_least": C4_SPECIFICITY,
            "passed": specificity >= C4_SPECIFICITY,
        },
    }
    criteria["C3_other_states_unchanged"]["passed"] = all(
        before[state] == after[state]
        for state in (FieldState.CONFLICTING, FieldState.UNCERTAIN, FieldState.MISSING)
    )

    payload = {
        "schema": "nano.denial-polarity-probe.v1",
        "preregistration": "papers/PREREG_DENIAL_POLARITY.md",
        "partition": "calibration (development not opened)",
        "checkpoint": str(args.checkpoint),
        "totals_by_gold_state": {s.value: n for s, n in totals.items()},
        "criteria": criteria,
        "verdict": (
            "ACCEPT"
            if all(criteria[k].get("passed") for k in criteria)
            else "REJECT"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"partition totals: {payload['totals_by_gold_state']}")
    for name, data in criteria.items():
        print(f"{name:32s} passed={data.get('passed')}")
    print(f"\nVERDICT: {payload['verdict']}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
