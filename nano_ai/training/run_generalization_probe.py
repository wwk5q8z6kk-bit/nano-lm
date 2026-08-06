"""Compare per-state joint accuracy in-distribution vs out-of-distribution.

EXPLORATORY diagnosis of an already-decided experiment (H6: REJECTED). This
selects nothing, changes no frozen artifact, and moves no threshold.

Motivation: DP-1 (`papers/PREREG_DENIAL_POLARITY.md`) passed vacuously on the
calibration partition -- only 4 fields were recoverable there, because `absent`
already scores 95.8% in-distribution. That is irreconcilable with development's
48.2% for the same checkpoint, so the difference is a property of the
partitions, not of the model's competence at the state.

The development manifest declares the cause outright:

    "isolation": {"denial_phrases_disjoint": true,
                  "uncertainty_phrases_disjoint": true,
                  "open_value_lexicons_disjoint": true, ...}

Development is a deliberate lexical-generalization stress test. This probe
measures, per epistemic state, how much accuracy each state loses when the
lexical cue it depends on is swapped for an unseen synonym -- and pairs each
state with the isolation flag that governs it.

    python3 -m nano_ai.training.run_generalization_probe \
        --checkpoint artifacts/nano_h6/kaggle/results-20260805/results/seed-20260805/epoch-2.pt \
        --output artifacts/nano_h6/analysis/generalization_gap.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from nano_ai.adapters.state_span import parse_state_span_summary
from nano_ai.contract import FieldName, FieldState, _is_field_denial

# Which lexical pool each state's decision depends on, and whether the
# development generator made that pool disjoint from training.
_CUE = {
    FieldState.ABSENT: ("denial phrase", "denial_phrases_disjoint"),
    FieldState.UNCERTAIN: ("hedge phrase", "uncertainty_phrases_disjoint"),
    FieldState.SUPPORTED: ("open value", "open_value_lexicons_disjoint"),
    FieldState.CONFLICTING: ("two-span structure", None),
    FieldState.MISSING: ("no mention", None),
}


def _score(model, examples, tokenizer, device, batch_size, split):
    from nano_ai.training.evaluate_pointer import build_pointer_inference_inputs
    from nano_ai.training.evidence_query_inference import (
        _proposal_exact,
        batched_evidence_query_inference,
    )
    from nano_ai.training.train_pointer import encode_pointer_partition

    records = encode_pointer_partition(tokenizer, examples, expected_split="train")
    inputs = build_pointer_inference_inputs(examples, records)
    with torch.inference_mode():
        inference = batched_evidence_query_inference(
            model, inputs, device=device, batch_size=batch_size
        )

    correct, total = Counter(), Counter()
    denial_spans = defaultdict(Counter)
    for prediction, example in zip(inference.predictions, examples, strict=True):
        proposed = prediction.proposals if prediction.error is None else ()
        for index, gold in enumerate(parse_state_span_summary(example.target, example.transcript)):
            total[gold.state] += 1
            correct[gold.state] += bool(proposed) and _proposal_exact(
                proposed[index], gold
            )
            if gold.state is FieldState.ABSENT:
                for span in gold.spans:
                    denial_spans[gold.field.value][span.text] += 1
    return correct, total, denial_spans


def _load_model(checkpoint: Path, device: str):
    from nano_ai.training.evidence_query_model import NanoEvidenceQueryPointerModel

    weights = torch.load(checkpoint, map_location=device, weights_only=True)
    weights = weights.get("model", weights)
    if "state_boundary_query_offsets" in weights:
        from nano_ai.training.state_conditioned_evidence_query_model import (
            NanoStateConditionedEvidenceQueryPointerModel,
        )

        model = NanoStateConditionedEvidenceQueryPointerModel()
    else:
        model = NanoEvidenceQueryPointerModel()
    model.load_state_dict(weights)
    return model.to(device).eval()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=Path("artifacts/nano_h5/data"))
    parser.add_argument(
        "--development", type=Path, default=Path("artifacts/nano_h6/kaggle/dataset-dev")
    )
    parser.add_argument("--tokenizer", type=Path, default=Path("sft/tokenizer.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    from nano_ai.training import replay_mixture_data
    from nano_ai.training.evaluate_state_span import load_development_bundle
    from nano_ai.training.train_pointer import load_pointer_tokenizer

    tokenizer = load_pointer_tokenizer(args.tokenizer)
    model = _load_model(args.checkpoint, args.device)

    in_dist = replay_mixture_data.load_replay_mixture_dataset(args.calibration).calibration
    manifest_sha = hashlib.sha256(
        (args.development / "manifest.json").read_bytes()
    ).hexdigest()
    dev_bundle = load_development_bundle(
        args.development, expected_manifest_sha256=manifest_sha
    )
    manifest_isolation = dict(dev_bundle.manifest.get("isolation", {}))

    id_correct, id_total, id_spans = _score(
        model, in_dist, tokenizer, args.device, args.batch_size
    )
    ood_correct, ood_total, ood_spans = _score(
        model, dev_bundle.examples, tokenizer, args.device, args.batch_size
    )

    rows = []
    for state in FieldState:
        if not id_total[state] and not ood_total[state]:
            continue
        cue, flag = _CUE[state]
        id_acc = id_correct[state] / id_total[state] if id_total[state] else None
        ood_acc = ood_correct[state] / ood_total[state] if ood_total[state] else None
        rows.append(
            {
                "state": state.value,
                "cue": cue,
                "cue_pool_disjoint_in_development": (
                    bool(manifest_isolation.get(flag)) if flag else False
                ),
                "in_distribution": {
                    "correct": id_correct[state],
                    "total": id_total[state],
                    "accuracy": id_acc,
                },
                "out_of_distribution": {
                    "correct": ood_correct[state],
                    "total": ood_total[state],
                    "accuracy": ood_acc,
                },
                "generalization_gap": (
                    round(id_acc - ood_acc, 4)
                    if id_acc is not None and ood_acc is not None
                    else None
                ),
            }
        )

    # The comparison that matters: the model learned phrases; does the
    # hand-written regex cover both pools it never saw together?
    regex = {}
    for label, spans in (("in_distribution", id_spans), ("out_of_distribution", ood_spans)):
        pool = {}
        for field, counter in spans.items():
            covered = sum(
                count
                for text, count in counter.items()
                if _is_field_denial(FieldName(field), text)
            )
            pool[field] = {
                "distinct_phrases": sorted(counter),
                "occurrences": sum(counter.values()),
                "regex_covers": covered,
            }
        regex[label] = pool

    payload = {
        "schema": "nano.generalization-gap.v1",
        "status": "EXPLORATORY -- diagnoses a decided experiment; selects nothing",
        "checkpoint": str(args.checkpoint),
        "development_isolation": manifest_isolation,
        "per_state": rows,
        "denial_phrase_pools": regex,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"{'state':<13}{'cue disjoint':<14}{'in-dist':>12}{'held-out':>12}{'gap':>9}")
    for row in rows:
        i, o = row["in_distribution"], row["out_of_distribution"]
        gap = row["generalization_gap"]
        print(
            f"{row['state']:<13}"
            f"{('YES' if row['cue_pool_disjoint_in_development'] else '-'):<14}"
            f"{i['correct']:>5}/{i['total']:<6}"
            f"{o['correct']:>5}/{o['total']:<6}"
            f"{(f'{gap:+.1%}' if gap is not None else 'n/a'):>9}"
        )
    print("\ndenial phrase pools (gold spans) and hand-written regex coverage:")
    for label, pool in regex.items():
        for field, data in sorted(pool.items()):
            print(
                f"  {label:<20} {field:<11} "
                f"{len(data['distinct_phrases'])} distinct  "
                f"regex {data['regex_covers']}/{data['occurrences']}  "
                f"{data['distinct_phrases']}"
            )
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
