"""Isolate the lexical variable: same transcripts, different denial phrasings.

EXPLORATORY. Selects nothing, gates nothing, trains nothing, writes no checkpoint.

`run_generalization_probe` showed `absent` at 95.8% in-distribution and 48.2%
held out -- but the held-out set contains only TWO novel denial phrases, and the
two behave very differently (allergy 33.5%, medication 62.0%). n=2 is too thin to
support "the model memorised the phrase list", and everything downstream depends
on that claim, so it needs a real denominator.

Design. Hold the development transcripts fixed and substitute only the denial
phrase, in both the transcript and the target. Three arms over identical
documents:

  TRAIN     the four phrasings per field the model was trained on
  DEV       the phrasing actually used in the sealed development set
  EXTERNAL  phrasings built from the negspacy en_clinical termset (MIT,
            vendored, evaluation-only) -- authored with no knowledge of this
            project

Because the transcripts, worlds, fields, and gold structure are identical across
arms, any accuracy difference is attributable to the denial surface form alone.

    python3 -m nano_ai.training.run_lexical_substitution_probe \
        --checkpoint artifacts/nano_h6/kaggle/results-20260805/results/seed-20260805/epoch-2.pt \
        --output artifacts/nano_h6/analysis/lexical_substitution.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

from nano_ai.adapters.state_span import StateSpanFormatError, parse_state_span_summary
from nano_ai.contract import FieldName, FieldState, _is_field_denial

# The phrasings the sealed development set actually uses, per field.
_DEV_PHRASE = {
    FieldName.MEDICATION: "Nothing at all.",
    FieldName.ALLERGY: "None whatsoever.",
}

# Arm TRAIN -- verbatim from the calibration partition (the training distribution).
_TRAIN_PHRASES = {
    FieldName.MEDICATION: (
        "No, nothing.", "No nothing yet!",
        "I deny taking medications.", "I denied taking medicine.",
    ),
    FieldName.ALLERGY: (
        "No allergies!", "Not that I know of!",
        "I deny allergies.", "I denied any allergy.",
    ),
}

# Arm EXTERNAL -- patient-voice denials whose negation trigger comes from the
# vendored negspacy en_clinical termset. Each is annotated with its trigger so
# the provenance of every item is checkable.
_EXTERNAL_PHRASES = {
    FieldName.MEDICATION: (
        ("I don't take medications.", "don't"),
        ("I'm not on any medications.", "not"),
        ("I never took medications.", "never"),
        ("I have no medications.", "no"),
        ("Negative for medications.", "negative for"),
        ("I didn't take medications.", "didn't"),
        ("I cannot take medications.", "cannot"),
        ("Absence of medications.", "absence of"),
    ),
    FieldName.ALLERGY: (
        ("I don't have allergies.", "don't"),
        ("I'm not allergic.", "not"),
        ("I never had allergies.", "never"),
        ("I have no allergies.", "no"),
        ("Negative for allergies.", "negative for"),
        ("I didn't have allergies.", "didn't"),
        ("No signs of allergies.", "no signs of"),
        ("Absence of allergies.", "absence of"),
    ),
}


def _substitute(example, replacement: dict[FieldName, str]):
    """Swap the denial phrase in transcript and target; keep all else identical."""
    transcript, target = example.transcript, example.target
    for field, new in replacement.items():
        old = _DEV_PHRASE[field]
        if old not in transcript:
            continue
        # The phrase occurs once on a patient line and once inside the target.
        if transcript.count(old) != 1:
            return None
        transcript = transcript.replace(old, new)
        target = target.replace(f"[{old}]", f"[{new}]")
    payload = example.to_dict()
    payload["transcript"] = transcript
    payload["target"] = target
    return type(example).from_dict(payload)


def _score_absent(model, examples, tokenizer, device, batch_size):
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

    correct, total = Counter(), Counter()
    predicted = Counter()
    for prediction, example in zip(inference.predictions, examples, strict=True):
        proposed = prediction.proposals if prediction.error is None else ()
        for index, gold in enumerate(parse_state_span_summary(example.target, example.transcript)):
            if gold.state is not FieldState.ABSENT:
                continue
            total[gold.field] += 1
            ok = bool(proposed) and _proposal_exact(proposed[index], gold)
            correct[gold.field] += ok
            if proposed:
                predicted[proposed[index].state.value] += 1
    return correct, total, predicted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--development", type=Path, default=Path("artifacts/nano_h6/kaggle/dataset-dev")
    )
    parser.add_argument("--tokenizer", type=Path, default=Path("sft/tokenizer.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    from nano_ai.training.evaluate_state_span import load_development_bundle
    from nano_ai.training.run_generalization_probe import _load_model
    from nano_ai.training.train_pointer import load_pointer_tokenizer

    tokenizer = load_pointer_tokenizer(args.tokenizer)
    model = _load_model(args.checkpoint, args.device)
    manifest_sha = hashlib.sha256(
        (args.development / "manifest.json").read_bytes()
    ).hexdigest()
    base = load_development_bundle(
        args.development, expected_manifest_sha256=manifest_sha
    ).examples

    arms: list[tuple[str, dict[FieldName, str], str | None]] = [
        ("DEV", dict(_DEV_PHRASE), None),
    ]
    for i in range(4):
        arms.append(
            (
                f"TRAIN[{i}]",
                {f: _TRAIN_PHRASES[f][i] for f in _TRAIN_PHRASES},
                None,
            )
        )
    for i in range(len(_EXTERNAL_PHRASES[FieldName.MEDICATION])):
        arms.append(
            (
                f"EXTERNAL[{i}]",
                {f: _EXTERNAL_PHRASES[f][i][0] for f in _EXTERNAL_PHRASES},
                _EXTERNAL_PHRASES[FieldName.MEDICATION][i][1],
            )
        )

    results = []
    for name, replacement, trigger in arms:
        swapped, dropped = [], 0
        for example in base:
            new = _substitute(example, replacement)
            if new is None:
                dropped += 1
                continue
            try:  # the substituted span must still be uniquely locatable
                parse_state_span_summary(new.target, new.transcript)
            except (StateSpanFormatError, ValueError):
                dropped += 1
                continue
            swapped.append(new)
        if not swapped:
            results.append({"arm": name, "error": "no usable examples", "dropped": dropped})
            continue

        correct, total, predicted = _score_absent(
            model, tuple(swapped), tokenizer, args.device, args.batch_size
        )
        n = sum(total.values())
        c = sum(correct.values())
        rule = {
            f.value: all(_is_field_denial(f, replacement[f]) for f in (f,))
            for f in replacement
        }
        results.append(
            {
                "arm": name,
                "phrases": {f.value: p for f, p in replacement.items()},
                "external_trigger": trigger,
                "examples_used": len(swapped),
                "examples_dropped": dropped,
                "absent_correct": c,
                "absent_total": n,
                "absent_accuracy": round(c / n, 4) if n else None,
                "per_field": {
                    f.value: {
                        "correct": correct[f],
                        "total": total[f],
                        "accuracy": round(correct[f] / total[f], 4) if total[f] else None,
                    }
                    for f in total
                },
                "predicted_state_mix": dict(predicted),
                "rule_recognises_phrase": rule,
            }
        )
        acc = results[-1]["absent_accuracy"]
        print(
            f"{name:14s} absent {c:5d}/{n:<6d} = {acc:6.1%}"
            f"   rule={ {k: ('Y' if v else 'n') for k, v in rule.items()} }"
            f"   {list(results[-1]['phrases'].values())}"
        )

    payload = {
        "schema": "nano.lexical-substitution.v1",
        "status": "EXPLORATORY -- selects nothing, gates nothing",
        "design": "identical development transcripts; only the denial phrase varies",
        "checkpoint": str(args.checkpoint),
        "external_lexicon": "data/external/negspacy/en_clinical_termset.json (MIT)",
        "arms": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
