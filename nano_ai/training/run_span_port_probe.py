"""P2 falsifier — route (b) span relocation rates on a span-bearing LoRA.

Preregistration: `papers/PREREG_SPAN_PORT_B.md` (bars frozen before this
probe scores any Qwen2.5-1.5B span-bearing adapter).

Reports `no_match_rate` and `ambiguous_rate` over emitted span strings, plus
state held-out mean and the balanced control gate. Does not train.

    python3 -m nano_ai.training.run_span_port_probe \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --adapter-path artifacts/nano_h6/span_port/adapters_Qwen2.5-1.5B \
        --output artifacts/nano_h6/span_port/probe_Qwen2.5-1.5B.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from nano_ai.adapters.state_span import parse_state_span_summary
from nano_ai.contract import FieldState
from nano_ai.surface import apply_arm
from nano_ai.surface_arms import DENIAL_ARMS
from nano_ai.training.span_port_format import (
    EXPECTED_LABEL,
    FIELD_QUESTION,
    PROMPT_WITH_SPANS,
    classify_span_relocation,
    parse_label_and_spans,
)

# Frozen in papers/PREREG_SPAN_PORT_B.md
NO_MATCH_ACCEPT = 0.10
NO_MATCH_FALSIFY = 0.25
_MIN_CONTROL_CLASS_RECALL = 0.20
_CONTROL_STATES = (FieldState.SUPPORTED, FieldState.MISSING)


def _decision(no_match_rate: float | None, interpretable: bool) -> str:
    if not interpretable:
        return "uninterpretable"
    if no_match_rate is None:
        return "no_spans_emitted"
    if no_match_rate <= NO_MATCH_ACCEPT:
        return "route_b_sufficient"
    if no_match_rate > NO_MATCH_FALSIFY:
        return "route_b_falsified"
    return "gray_zone"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--development", type=Path, default=Path("artifacts/nano_h6/kaggle/dataset-dev")
    )
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    from mlx_lm import generate, load
    from nano_ai.training.evaluate_state_span import load_development_bundle

    manifest_sha = hashlib.sha256(
        (args.development / "manifest.json").read_bytes()
    ).hexdigest()
    examples = load_development_bundle(
        args.development, expected_manifest_sha256=manifest_sha
    ).examples

    pool = []
    for example in examples:
        golds = parse_state_span_summary(example.target, example.transcript)
        absent = {
            g.field.value: g
            for g in golds
            if g.state is FieldState.ABSENT and g.field.value in FIELD_QUESTION
        }
        if len(absent) == len(FIELD_QUESTION):
            pool.append(example)
    pool = pool[: args.limit]
    print(f"pool: {len(pool)} documents, both fields gold-absent")

    by_state: dict[FieldState, list] = {s: [] for s in _CONTROL_STATES}
    for example in examples:
        for gold in parse_state_span_summary(example.target, example.transcript):
            if gold.field.value in FIELD_QUESTION and gold.state in _CONTROL_STATES:
                by_state[gold.state].append(
                    (example.transcript, gold.field.value, gold.state, gold.spans)
                )
    per_class = min([len(v) for v in by_state.values()] + [args.limit])
    control = [row for rows in by_state.values() for row in rows[:per_class]]
    print(
        f"control: {len(control)} never-rewritten fields, balanced "
        f"{per_class}/class"
    )

    model, tokenizer = load(
        args.model, adapter_path=str(args.adapter_path)
    )
    print(f"model: {args.model}   adapter: {args.adapter_path}")

    def ask(transcript: str, topic: str) -> tuple[str | None, tuple[str, ...], str]:
        prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": PROMPT_WITH_SPANS.format(
                        transcript=transcript, topic=topic
                    ),
                }
            ],
            add_generation_prompt=True,
        )
        out = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            verbose=False,
        )
        label, spans = parse_label_and_spans(out)
        return label, spans, out

    control_correct = 0
    control_answers: Counter[str] = Counter()
    class_hit: Counter[str] = Counter()
    class_n: Counter[str] = Counter()
    for transcript, field, state, _spans in control:
        answer, _emitted, _raw = ask(transcript, FIELD_QUESTION[field])
        control_answers[answer or "UNPARSED"] += 1
        class_n[state.value] += 1
        if answer == EXPECTED_LABEL[state]:
            control_correct += 1
            class_hit[state.value] += 1
    control_n = len(control)
    class_recall = {
        k: round(class_hit[k] / class_n[k], 4) for k in class_n if class_n[k]
    }
    worst = min(class_recall.values()) if class_recall else 0.0
    interpretable = worst >= _MIN_CONTROL_CLASS_RECALL
    print(
        f"control accuracy: {control_correct}/{control_n} = "
        f"{control_correct / control_n:.1%}   per-class recall={class_recall}"
    )
    if not interpretable:
        print(
            f"  *** COLLAPSED: worst control-class recall {worst:.0%} < "
            f"{_MIN_CONTROL_CLASS_RECALL:.0%}. Rates are NOT interpretable. ***"
        )

    relocation = Counter()
    emitted_spans = 0
    missing_span_num = 0
    missing_span_den = 0
    unparsed_span_fields = 0
    arm_rows = []

    for arm in DENIAL_ARMS:
        correct = 0
        total = 0
        unparsed = 0
        for example in pool:
            applied = apply_arm(example.transcript, example.target, arm)
            if applied is None:
                continue
            transcript, target = applied
            golds = {
                g.field.value: g
                for g in parse_state_span_summary(target, transcript)
                if g.field.value in FIELD_QUESTION
            }
            for field, topic in FIELD_QUESTION.items():
                gold = golds[field]
                label, spans, _raw = ask(transcript, topic)
                total += 1
                if label is None:
                    unparsed += 1
                elif label == EXPECTED_LABEL[FieldState.ABSENT]:
                    correct += 1

                if gold.spans:
                    missing_span_den += 1
                    if not spans:
                        missing_span_num += 1
                if label is None:
                    unparsed_span_fields += 1
                    continue
                for text in spans:
                    emitted_spans += 1
                    relocation[classify_span_relocation(transcript, text)] += 1

        row = {
            "arm": arm.label,
            "in_distribution_for_nano": arm.in_distribution,
            "n": total,
            "correct": correct,
            "accuracy": round(correct / total, 4) if total else None,
            "unparsed": unparsed,
        }
        arm_rows.append(row)
        print(
            f"  {arm.label:16s} {correct:4d}/{total:<5d} = "
            f"{(row['accuracy'] or 0):6.1%}  unparsed={unparsed}"
        )

    def group(pred):
        vals = [
            r["accuracy"]
            for r in arm_rows
            if pred(r) and r["accuracy"] is not None
        ]
        if not vals:
            return None
        return {
            "arms": len(vals),
            "mean": round(sum(vals) / len(vals), 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
        }

    no_match_rate = (
        round(relocation["no_match"] / emitted_spans, 4) if emitted_spans else None
    )
    ambiguous_rate = (
        round(relocation["ambiguous"] / emitted_spans, 4) if emitted_spans else None
    )
    relocated_rate = (
        round(relocation["relocated"] / emitted_spans, 4) if emitted_spans else None
    )
    missing_span_rate = (
        round(missing_span_num / missing_span_den, 4) if missing_span_den else None
    )
    decision = _decision(no_match_rate, interpretable)

    payload = {
        "schema": "nano.span-port-probe.v1",
        "preregistration": "papers/PREREG_SPAN_PORT_B.md",
        "model": args.model,
        "adapter_path": str(args.adapter_path),
        "bars": {
            "no_match_accept": NO_MATCH_ACCEPT,
            "no_match_falsify": NO_MATCH_FALSIFY,
            "min_control_class_recall": _MIN_CONTROL_CLASS_RECALL,
        },
        "control": {
            "n": control_n,
            "accuracy": round(control_correct / control_n, 4) if control_n else None,
            "per_class_recall": class_recall,
            "worst_class_recall": round(worst, 4),
            "answers": dict(control_answers),
        },
        "interpretable": interpretable,
        "state_summary": {
            "nano_in_distribution_arms": group(
                lambda r: r["in_distribution_for_nano"]
            ),
            "nano_held_out_arms": group(
                lambda r: not r["in_distribution_for_nano"]
            ),
        },
        "span_relocation": {
            "emitted_spans": emitted_spans,
            "counts": dict(relocation),
            "no_match_rate": no_match_rate,
            "ambiguous_rate": ambiguous_rate,
            "relocated_rate": relocated_rate,
            "missing_span_rate": missing_span_rate,
            "missing_span_numerator": missing_span_num,
            "missing_span_denominator": missing_span_den,
            "unparsed_span_fields": unparsed_span_fields,
        },
        "decision": decision,
        "arms": arm_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"spans emitted={emitted_spans}  no_match={no_match_rate}  "
        f"ambiguous={ambiguous_rate}  relocated={relocated_rate}  "
        f"missing_span={missing_span_rate}"
    )
    print(f"decision: {decision}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
