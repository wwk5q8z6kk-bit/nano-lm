"""Is Nano's surface sensitivity a small-model artifact? (cross-model control)

EXPLORATORY. Selects nothing, gates nothing, trains nothing. Local, $0.

Why this matters to Nano specifically. `RESULT_PER_STATE_DIAGNOSIS.md` showed
every span-carrying state recovers to 95-100% on familiar wording and collapses
on unfamiliar wording, and `DECISION_MEMO_20260806.md` proposes H7-V (widen the
training vocabulary) on that basis. H7-V is the right call **only if the failure
is a property of Nano's narrow training vocabulary**. If a general
instruction-tuned model, which has seen every one of these phrasings thousands of
times, shows the same collapse on the same documents, then the difficulty is in
the task rather than in Nano's diet, and widening the vocabulary would not fix it.

So this runs the *identical* denial arms over the *identical* development
documents through a local general LM, and compares surface sensitivity against
Nano's. It is a control for the H7-V hypothesis, not a benchmark of the LM.

Backend: `mlx-community/Llama-3.2-3B-Instruct-4bit` via mlx_lm, already cached
locally. No network, no API, no cost.

    python3 -m nano_ai.training.run_crossmodel_surface_probe \
        --limit 40 --output artifacts/nano_h6/analysis/crossmodel_surface.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from nano_ai.adapters.state_span import parse_state_span_summary
from nano_ai.contract import FieldState
from nano_ai.surface import apply_arm
from nano_ai.surface_arms import DENIAL_ARMS

_FIELD_QUESTION = {
    "medication": "any medication the patient is taking",
    "allergy": "any allergy the patient has",
}

_PROMPT = """You are reading a clinic transcript.

{transcript}

Question: regarding {topic}, which of these does the patient's own words do?

STATED       - the patient names a specific one
DENIED       - the patient says there is none / denies having any
NOT_MENTIONED - the topic never comes up

Answer with exactly one word: STATED, DENIED, or NOT_MENTIONED."""

_ANSWER = re.compile(r"\b(STATED|DENIED|NOT_MENTIONED)\b")

# Gold state -> the answer a correct reader gives.
_EXPECTED = {
    FieldState.SUPPORTED: "STATED",
    FieldState.ABSENT: "DENIED",
    FieldState.MISSING: "NOT_MENTIONED",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct-4bit")
    parser.add_argument(
        "--development", type=Path, default=Path("artifacts/nano_h6/kaggle/dataset-dev")
    )
    parser.add_argument("--limit", type=int, default=40, help="documents per arm")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from mlx_lm import generate, load
    from nano_ai.training.evaluate_state_span import load_development_bundle

    manifest_sha = hashlib.sha256(
        (args.development / "manifest.json").read_bytes()
    ).hexdigest()
    examples = load_development_bundle(
        args.development, expected_manifest_sha256=manifest_sha
    ).examples

    # Only documents whose medication AND allergy are gold-absent, so every arm
    # rewrites the same population and the denominator cannot drift between arms
    # -- the defect the advisor caught in `conflicting_value`.
    pool = []
    for example in examples:
        golds = parse_state_span_summary(example.target, example.transcript)
        absent = {
            g.field.value: g
            for g in golds
            if g.state is FieldState.ABSENT and g.field.value in _FIELD_QUESTION
        }
        if len(absent) == len(_FIELD_QUESTION):
            pool.append(example)
    pool = pool[: args.limit]
    print(f"pool: {len(pool)} documents, both fields gold-absent")

    model, tokenizer = load(args.model)
    results = []
    for arm in DENIAL_ARMS:
        correct = Counter()
        total = Counter()
        unparsed = 0
        for example in pool:
            applied = apply_arm(example.transcript, example.target, arm)
            if applied is None:
                continue
            transcript, _ = applied
            for field, topic in _FIELD_QUESTION.items():
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": _PROMPT.format(
                        transcript=transcript, topic=topic)}],
                    add_generation_prompt=True,
                )
                out = generate(model, tokenizer, prompt=prompt, max_tokens=6, verbose=False)
                match = _ANSWER.search(out.upper())
                total[field] += 1
                if match is None:
                    unparsed += 1
                elif match.group(1) == _EXPECTED[FieldState.ABSENT]:
                    correct[field] += 1
        n = sum(total.values())
        c = sum(correct.values())
        row = {
            "arm": arm.label,
            "in_distribution_for_nano": arm.in_distribution,
            "provenance": arm.provenance,
            "n": n,
            "correct": c,
            "accuracy": round(c / n, 4) if n else None,
            "unparsed": unparsed,
        }
        results.append(row)
        print(f"  {arm.label:16s} {c:4d}/{n:<5d} = {row['accuracy']:6.1%}  unparsed={unparsed}")

    def group(pred):
        vals = [r["accuracy"] for r in results if pred(r) and r["accuracy"] is not None]
        if not vals:
            return None
        return {
            "arms": len(vals),
            "mean": round(sum(vals) / len(vals), 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
            "sensitivity": round(max(vals) - min(vals), 4),
        }

    summary = {
        # "in-distribution" here means in-distribution *for Nano*. The LM saw
        # none of these documents; the split is kept so the two models are
        # compared on identical arm groupings.
        "nano_in_distribution_arms": group(lambda r: r["in_distribution_for_nano"]),
        "nano_held_out_arms": group(
            lambda r: not r["in_distribution_for_nano"] and r["arm"] != "DEV"
        ),
        "dev_arm": next((r["accuracy"] for r in results if r["arm"] == "DEV"), None),
    }
    payload = {
        "schema": "nano.crossmodel-surface.v1",
        "status": "EXPLORATORY -- control for the H7-V hypothesis; gates nothing",
        "model": args.model,
        "documents_per_arm": len(pool),
        "arms": results,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print("\n=== surface sensitivity, general 3B LM ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
