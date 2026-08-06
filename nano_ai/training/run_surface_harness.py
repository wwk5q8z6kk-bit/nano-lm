"""Run the surface-robustness harness over checkpoints and axes.

EXPLORATORY. Selects nothing, gates nothing, trains nothing, writes no
checkpoint. Reads the sealed development documents, substitutes one concept's
wording at a time, and reports the metrics defined in `nano_ai/surface.py`.

    python3 -m nano_ai.training.run_surface_harness \
        --checkpoint seed05=artifacts/.../seed-20260805/epoch-2.pt \
        --checkpoint seed06=artifacts/.../seed-20260806/epoch-2.pt \
        --axis denial --axis hedge \
        --output artifacts/nano_h6/analysis/surface_harness.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from nano_ai.adapters.state_span import StateSpanFormatError, parse_state_span_summary
from nano_ai.surface import ArmObservation, aggregate, apply_arm, report_lines
from nano_ai.surface_arms import ALL_AXES, VALUE_TEMPLATE_FIELDS


def _apply(example, arm):
    """Rewrite one example under an arm, or None when it cannot be applied."""
    rewritten = apply_arm(example.transcript, example.target, arm)
    if rewritten is None:
        return None
    transcript, target = rewritten
    payload = example.to_dict()
    payload["transcript"] = transcript
    payload["target"] = target
    rebuilt = type(example).from_dict(payload)
    try:  # the rewritten span must still be uniquely locatable
        parse_state_span_summary(rebuilt.target, rebuilt.transcript)
    except (StateSpanFormatError, ValueError):
        return None
    return rebuilt


def _score(model, examples, tokenizer, device, batch_size, *, fields=None):
    """Score joint-exact accuracy per gold state.

    `fields` restricts which fields contribute, by `FieldName.value` (e.g.
    "medication"). Needed for the `value`/`template` axes: they only ever
    rewrite medication/allergy, so counting chief_complaint/duration/severity
    -- untouched and identical across every arm -- would dilute sensitivity
    toward zero rather than measure it. `denial`/`hedge` pass `fields=None`
    (unrestricted), unchanged from prior behaviour.
    """
    from nano_ai.training.evaluate_pointer import build_pointer_inference_inputs
    from nano_ai.training.evidence_query_inference import (
        _proposal_exact,
        batched_evidence_query_inference,
    )
    from nano_ai.training.train_pointer import encode_pointer_partition

    records = encode_pointer_partition(tokenizer, examples, expected_split="dev")
    inputs = build_pointer_inference_inputs(examples, records)
    with torch.inference_mode():
        inference = batched_evidence_query_inference(
            model, inputs, device=device, batch_size=batch_size
        )
    correct: dict[str, int] = {}
    total: dict[str, int] = {}
    for prediction, example in zip(inference.predictions, examples, strict=True):
        proposed = prediction.proposals if prediction.error is None else ()
        for index, gold in enumerate(
            parse_state_span_summary(example.target, example.transcript)
        ):
            if fields is not None and gold.field.value not in fields:
                continue
            key = gold.state.value
            total[key] = total.get(key, 0) + 1
            if bool(proposed) and _proposal_exact(proposed[index], gold):
                correct[key] = correct.get(key, 0) + 1
    return correct, total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", action="append", required=True, metavar="NAME=PATH",
        help="repeatable; NAME labels the seed",
    )
    parser.add_argument("--axis", action="append", choices=sorted(ALL_AXES), default=None)
    parser.add_argument(
        "--development", type=Path, default=Path("artifacts/nano_h6/kaggle/dataset-dev")
    )
    parser.add_argument("--tokenizer", type=Path, default=Path("sft/tokenizer.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-examples", type=int, default=None,
                        help="smoke-test only; truncates the document set")
    args = parser.parse_args()

    from nano_ai.training.evaluate_state_span import load_development_bundle
    from nano_ai.training.run_generalization_probe import _load_model
    from nano_ai.training.train_pointer import load_pointer_tokenizer

    axes = args.axis or sorted(ALL_AXES)
    seeds = []
    for entry in args.checkpoint:
        if "=" not in entry:
            parser.error(f"--checkpoint must be NAME=PATH, got {entry!r}")
        name, _, path = entry.partition("=")
        seeds.append((name, Path(path)))

    tokenizer = load_pointer_tokenizer(args.tokenizer)
    manifest_sha = hashlib.sha256(
        (args.development / "manifest.json").read_bytes()
    ).hexdigest()
    base = load_development_bundle(
        args.development, expected_manifest_sha256=manifest_sha
    ).examples
    if args.max_examples:
        base = base[: args.max_examples]

    observations: list[ArmObservation] = []
    skipped: list[dict[str, object]] = []
    for seed_name, checkpoint in seeds:
        model = _load_model(checkpoint, args.device)
        for axis in axes:
            for arm in ALL_AXES[axis]:
                rewritten, dropped = [], 0
                for example in base:
                    new = _apply(example, arm)
                    if new is None:
                        dropped += 1
                        continue
                    rewritten.append(new)
                if not rewritten:
                    skipped.append({"seed": seed_name, "axis": axis, "arm": arm.label})
                    continue
                correct, total = _score(
                    model, tuple(rewritten), tokenizer, args.device, args.batch_size,
                    fields=VALUE_TEMPLATE_FIELDS if axis in _FIELD_RESTRICTED_AXES else None,
                )
                for state, n in total.items():
                    observations.append(
                        ArmObservation(
                            arm=arm.label,
                            axis=axis,
                            seed=seed_name,
                            state=state,
                            correct=correct.get(state, 0),
                            total=n,
                            in_distribution=arm.in_distribution,
                        )
                    )
                acc = correct.get(_PRIMARY[axis], 0) / max(1, total.get(_PRIMARY[axis], 0))
                print(
                    f"  {seed_name:8s} {axis:7s} {arm.label:16s} "
                    f"{_PRIMARY[axis]:11s} {acc:6.1%}  dropped={dropped}"
                )

    summaries = []
    for axis in axes:
        state = _PRIMARY[axis]
        for scope, flag in (("in_distribution", True), ("held_out", False), ("all", None)):
            summary = aggregate(observations, axis=axis, state=state, in_distribution=flag)
            if summary is not None:
                summary["scope"] = scope
                summaries.append(summary)
        # cross-effects: does rewriting one concept disturb the others?
        for other in sorted({o.state for o in observations if o.axis == axis}):
            if other == state:
                continue
            summary = aggregate(observations, axis=axis, state=other)
            if summary is not None:
                summary["scope"] = "cross_effect"
                summaries.append(summary)

    payload = {
        "schema": "nano.surface-harness.v1",
        "status": "EXPLORATORY -- selects nothing, gates nothing",
        "checkpoints": {name: str(path) for name, path in seeds},
        "axes": axes,
        "arm_provenance": {
            axis: {arm.label: arm.provenance for arm in ALL_AXES[axis]} for axis in axes
        },
        "summaries": summaries,
        "observations": [
            {
                "arm": o.arm, "axis": o.axis, "seed": o.seed, "state": o.state,
                "correct": o.correct, "total": o.total,
                "in_distribution": o.in_distribution,
            }
            for o in observations
        ],
        "skipped": skipped,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print()
    for summary in summaries:
        if summary["scope"] in {"in_distribution", "held_out"}:
            print(f"[{summary['scope']}]")
            for line in report_lines(summary):
                print("  " + line)
    print(f"\nwrote {args.output}")
    return 0


# The state each axis is designed to move.
_PRIMARY = {
    "denial": "absent",
    "hedge": "uncertain",
    "value": "supported",
    "template": "supported",
    "conflicting_value": "conflicting",
    "conflicting_structure": "conflicting",
}

# Axes whose arms only ever rewrite medication/allergy -- see `_score`'s
# `fields` docstring for why chief_complaint/duration/severity must be
# excluded rather than counted as unchanging noise. `conflicting_value`'s
# TRAIN arms only apply to medication/allergy documents (dropped=150/250
# elsewhere), but its DEV baseline is a no-op that applies to all 250 -- an
# unrestricted DEV would be scored against a different, easier-on-average
# field mix than TRAIN, confounding the value-swap effect with a field-type
# effect. `conflicting_structure` needs no entry: all of its arms (DEV,
# ORDER, DISTANCE[n]) apply to the same 250 documents already.
_FIELD_RESTRICTED_AXES = {"value", "template", "conflicting_value"}


if __name__ == "__main__":
    raise SystemExit(main())
