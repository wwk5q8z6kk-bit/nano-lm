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

# Second prompt mode. Variant D of the 2026-08-06 probe found the model quoting
# the span correctly and explaining its meaning correctly ("has tried no
# medication") while emitting the wrong label in the same sentence -- Nano's
# exact failure. If asking for the meaning FIRST and the label SECOND recovers
# accuracy, the deficit is in label assignment, not comprehension, and Nano's
# remedy is an output-format change rather than (only) more vocabulary.
_PROMPT_TWOSTAGE = """You are reading a clinic transcript.

{transcript}

Regarding {topic}:
Step 1 - quote the patient's exact words that bear on it.
Step 2 - paraphrase what those words mean.
Step 3 - on the last line write exactly one word: STATED if the patient names a
specific one, DENIED if the patient says there is none, NOT_MENTIONED if the
topic never comes up."""

# Gold state -> the answer a correct reader gives.
_EXPECTED = {
    FieldState.SUPPORTED: "STATED",
    FieldState.ABSENT: "DENIED",
    FieldState.MISSING: "NOT_MENTIONED",
}

# A model that answers DENIED unconditionally scores 100% on a probe that only
# examines gold-absent fields. The v1 run of this probe had exactly that defect
# -- the third instance of a gate denominated in something the system controls
# (after `fabric/slice.py:247` and DP-1's C1). The arms can only rewrite denial
# phrases, so the arm-varying measurement necessarily lives on absent fields;
# the guard is a fixed CONTROL block of gold-supported and gold-missing fields,
# never rewritten, scored every run. If the model cannot say STATED and
# NOT_MENTIONED there, its DENIED accuracy is meaningless.
_CONTROL_STATES = (FieldState.SUPPORTED, FieldState.MISSING)

# Collapse is per-class RECALL on a BALANCED control block, not raw answer
# share. The v2 guard used raw share and produced a false positive: the control
# was 93% gold-supported, so a model answering STATED 97% of the time scored
# 96.7% correct and was still flagged "collapsed". Denominating a guard in a
# quantity the *gold distribution* controls is the same defect it was written to
# catch. A model is collapsed when it cannot produce some label at all.
_MIN_CONTROL_CLASS_RECALL = 0.20


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct-4bit")
    parser.add_argument(
        "--development", type=Path, default=Path("artifacts/nano_h6/kaggle/dataset-dev")
    )
    parser.add_argument("--limit", type=int, default=40, help="documents per arm")
    parser.add_argument("--mode", choices=("direct", "twostage"), default="direct")
    parser.add_argument(
        "--adapter-path", type=Path, default=None,
        help="LoRA adapter to load; omit for the untuned base model",
    )
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

    # Control block: fields the arms never touch, covering the other answers.
    by_state: dict[FieldState, list] = {s: [] for s in _CONTROL_STATES}
    for example in examples:
        for gold in parse_state_span_summary(example.target, example.transcript):
            if gold.field.value in _FIELD_QUESTION and gold.state in _CONTROL_STATES:
                by_state[gold.state].append(
                    (example.transcript, gold.field.value, gold.state)
                )
    # Balanced: equal count per control class, so the guard cannot be tripped by
    # a skewed gold distribution.
    per_class = min([len(v) for v in by_state.values()] + [args.limit])
    control = [row for rows in by_state.values() for row in rows[:per_class]]
    print(f"control: {len(control)} never-rewritten fields, balanced "
          f"{per_class}/class ({[s.value for s in _CONTROL_STATES]})")

    model, tokenizer = load(
        args.model,
        **({"adapter_path": str(args.adapter_path)} if args.adapter_path else {}),
    )
    print(f"model: {args.model}   adapter: {args.adapter_path or '(none -- base)'}")

    def ask(transcript: str, topic: str) -> str | None:
        template = _PROMPT if args.mode == "direct" else _PROMPT_TWOSTAGE
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": template.format(
                transcript=transcript, topic=topic)}],
            add_generation_prompt=True,
        )
        out = generate(
            model, tokenizer, prompt=prompt,
            max_tokens=6 if args.mode == "direct" else 120, verbose=False,
        )
        found = _ANSWER.findall(out.upper())
        return found[-1] if found else None

    # --- run the control FIRST; a collapsed model invalidates everything after
    control_correct = 0
    control_answers: Counter[str] = Counter()
    class_hit: Counter[str] = Counter()
    class_n: Counter[str] = Counter()
    for transcript, field, state in control:
        answer = ask(transcript, _FIELD_QUESTION[field])
        control_answers[answer or "UNPARSED"] += 1
        class_n[state.value] += 1
        if answer == _EXPECTED[state]:
            control_correct += 1
            class_hit[state.value] += 1
    control_n = len(control)
    class_recall = {
        k: round(class_hit[k] / class_n[k], 4) for k in class_n if class_n[k]
    }
    worst = min(class_recall.values()) if class_recall else 0.0
    collapsed = worst < _MIN_CONTROL_CLASS_RECALL
    print(f"control accuracy: {control_correct}/{control_n} = "
          f"{control_correct / control_n:.1%}   per-class recall={class_recall}")
    print(f"  answers={dict(control_answers)}")
    if collapsed:
        print(f"  *** COLLAPSED: worst control-class recall {worst:.0%} < "
              f"{_MIN_CONTROL_CLASS_RECALL:.0%}. Arm accuracies are NOT interpretable. ***")

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
                match = ask(transcript, topic)
                total[field] += 1
                if match is None:
                    unparsed += 1
                elif match == _EXPECTED[FieldState.ABSENT]:
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
        "adapter_path": str(args.adapter_path) if args.adapter_path else None,
        "task_trained": args.adapter_path is not None,
        "mode": args.mode,
        "control": {
            "n": control_n,
            "correct": control_correct,
            "accuracy": round(control_correct / control_n, 4) if control_n else None,
            "answer_distribution": dict(control_answers),
            "per_class_recall": class_recall,
            "worst_class_recall": round(worst, 4),
            "balanced_per_class": per_class,
            "collapsed": collapsed,
            "note": "gold-supported and gold-missing fields, never rewritten by any "
                    "arm. Guards against a model that answers DENIED unconditionally "
                    "scoring 100% on an absent-only probe.",
        },
        "arm_accuracies_interpretable": not collapsed,
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
