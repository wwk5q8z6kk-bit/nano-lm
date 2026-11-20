"""Span-port gray-zone mitigation 2 — normalization snap at the proposal layer.

Preregistration: `papers/PREREG_SPAN_PORT_B_SNAP.md` (frozen before this
probe scores any snapped run). Identical to `run_span_port_probe.py` except:
spans the unmodified locator classifies `no_match` are passed through
`snap_to_patient_span`; on success the snapped text is re-classified by the
same locator. Labels are never touched, so all state and control metrics must
equal the baseline probe artifact — any deviation voids the run as a wiring
bug.

    python3 -m nano_ai.training.run_span_port_probe_snap \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --adapter-path artifacts/nano_h6/span_port/adapters_Qwen2.5-1.5B \
        --output artifacts/nano_h6/span_port/probe_Qwen2.5-1.5B_snap.json
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
from nano_ai.training.run_span_port_probe import (
    NO_MATCH_ACCEPT,
    NO_MATCH_FALSIFY,
    _MIN_CONTROL_CLASS_RECALL,
    _CONTROL_STATES,
    _decision,
)
from nano_ai.training.span_port_format import (
    EXPECTED_LABEL,
    FIELD_QUESTION,
    PROMPT_WITH_SPANS,
    classify_span_relocation,
    parse_label_and_spans,
)
from nano_ai.training.span_field_filter import rejects_cross_field_quote
from nano_ai.training.span_snap import snap_to_patient_span


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
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("artifacts/nano_h6/span_port/probe_Qwen2.5-1.5B.json"),
        help="Baseline probe artifact whose state metrics this run must equal.",
    )
    parser.add_argument(
        "--failures-output",
        type=Path,
        default=Path(
            "artifacts/nano_h6/span_port/failures_Qwen2.5-1.5B_snap.json"
        ),
    )
    parser.add_argument(
        "--skip-baseline-check",
        action="store_true",
        help=(
            "P3 multi-seed use: a different training seed legitimately "
            "changes raw counts, so the seed-A baseline-identity check "
            "does not apply (papers/PREREG_P3_MULTISEED.md)."
        ),
    )
    parser.add_argument(
        "--constrained",
        action="store_true",
        help=(
            "Substring-constrained decoding: quoted spans are restricted "
            "to verbatim patient-turn text (papers/DECISION_REGISTER_RECIPE.md). "
            "Implies that no_match is ~0 by construction; gold agreement is "
            "the honest span metric for such runs."
        ),
    )
    parser.add_argument(
        "--field-filter",
        action="store_true",
        help=(
            "Cross-field consistency filter: drop proposed quotes that name "
            "the other field's topic before relocation "
            "(papers/PREREG_FIELD_FILTER.md). Dropped quotes degrade to "
            "abstention and leave the emitted-span denominator."
        ),
    )
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

    model, tokenizer = load(args.model, adapter_path=str(args.adapter_path))
    print(f"model: {args.model}   adapter: {args.adapter_path}")

    string_index = None
    if args.constrained:
        from nano_ai.training.constrained_decode import (
            SubstringConstraint,
            build_string_index,
        )

        print("building vocab string index for constrained decode…", flush=True)
        string_index = build_string_index(tokenizer)
        print(f"  {len(string_index)} distinct token strings")

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
        kwargs = {}
        if string_index is not None:
            kwargs["logits_processors"] = [
                SubstringConstraint(transcript, tokenizer, string_index)
            ]
        out = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            verbose=False,
            **kwargs,
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
    control_accuracy = f"{control_correct / control_n:.1%}" if control_n else "n/a"
    print(
        f"control accuracy: {control_correct}/{control_n} = "
        f"{control_accuracy}   per-class recall={class_recall}"
    )

    raw_relocation: Counter[str] = Counter()
    relocation: Counter[str] = Counter()
    snap_rescued = 0
    snap_rejected_by_binder = 0
    field_filtered = 0
    emitted_spans = 0
    missing_span_num = 0
    missing_span_den = 0
    unparsed_span_fields = 0
    arm_rows = []
    failures = []

    for arm in DENIAL_ARMS:
        correct = 0
        total = 0
        unparsed = 0
        for doc_index, example in enumerate(pool):
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
                label, spans, raw = ask(transcript, topic)
                total += 1
                if label is None:
                    unparsed += 1
                elif label == EXPECTED_LABEL[FieldState.ABSENT]:
                    correct += 1

                if args.field_filter:
                    kept = tuple(
                        t
                        for t in spans
                        if not rejects_cross_field_quote(field, t)
                    )
                    field_filtered += len(spans) - len(kept)
                    spans = kept
                if gold.spans:
                    missing_span_den += 1
                    if not spans:
                        missing_span_num += 1
                if label is None:
                    unparsed_span_fields += 1
                    continue
                for text in spans:
                    emitted_spans += 1
                    verdict = classify_span_relocation(transcript, text)
                    raw_relocation[verdict] += 1
                    final_text = text
                    if verdict == "no_match":
                        snapped = snap_to_patient_span(transcript, text)
                        if snapped is not None:
                            verdict = classify_span_relocation(
                                transcript, snapped
                            )
                            final_text = snapped
                            if verdict == "relocated":
                                snap_rescued += 1
                            else:
                                snap_rejected_by_binder += 1
                    relocation[verdict] += 1
                    if verdict != "relocated":
                        failures.append(
                            {
                                "arm": arm.label,
                                "doc_index": doc_index,
                                "field": field,
                                "label": label,
                                "text": text,
                                "final_text": final_text,
                                "verdict": verdict,
                                "raw_generation": raw,
                            }
                        )

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
            f"{(row['accuracy'] or 0):6.1%}  unparsed={unparsed}  "
            f"snap_rescued={snap_rescued}"
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

    def rates(counts: Counter[str]) -> dict:
        if not emitted_spans:
            return {
                "no_match_rate": None,
                "ambiguous_rate": None,
                "relocated_rate": None,
            }
        return {
            "no_match_rate": round(counts["no_match"] / emitted_spans, 4),
            "ambiguous_rate": round(counts["ambiguous"] / emitted_spans, 4),
            "relocated_rate": round(counts["relocated"] / emitted_spans, 4),
        }

    post = rates(relocation)
    missing_span_rate = (
        round(missing_span_num / missing_span_den, 4) if missing_span_den else None
    )
    decision = _decision(post["no_match_rate"], interpretable)

    state_summary = {
        "nano_in_distribution_arms": group(lambda r: r["in_distribution_for_nano"]),
        "nano_held_out_arms": group(lambda r: not r["in_distribution_for_nano"]),
    }
    if args.skip_baseline_check:
        matches_baseline = None
    else:
        baseline = json.loads(args.baseline.read_text())
        matches_baseline = (
            state_summary == baseline["state_summary"]
            and round(control_correct / control_n, 4)
            == baseline["control"]["accuracy"]
            and emitted_spans == baseline["span_relocation"]["emitted_spans"]
            and dict(raw_relocation) == baseline["span_relocation"]["counts"]
        )

    payload = {
        "schema": "nano.span-port-probe-snap.v1",
        "preregistration": "papers/PREREG_SPAN_PORT_B_SNAP.md",
        "model": args.model,
        "adapter_path": str(args.adapter_path),
        "prompt": "PROMPT_WITH_SPANS (baseline; quote-forcing rejected)",
        "decode": "substring_constrained" if args.constrained else "greedy",
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
        "state_summary": state_summary,
        "span_relocation_raw": {
            "counts": dict(raw_relocation),
            **rates(raw_relocation),
        },
        "span_relocation": {
            "emitted_spans": emitted_spans,
            "counts": dict(relocation),
            **post,
            "missing_span_rate": missing_span_rate,
            "missing_span_numerator": missing_span_num,
            "missing_span_denominator": missing_span_den,
            "unparsed_span_fields": unparsed_span_fields,
        },
        "snap": {
            "rescued": snap_rescued,
            "rejected_by_binder": snap_rejected_by_binder,
        },
        "field_filter": {
            "enabled": bool(args.field_filter),
            "filtered_spans": field_filtered,
        },
        "baseline_artifact": None if args.skip_baseline_check else str(args.baseline),
        "matches_baseline": matches_baseline,
        "decision": (
            decision if matches_baseline is not False else "void_wiring_mismatch"
        ),
        "arms": arm_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.failures_output.write_text(
        json.dumps(failures, indent=2, sort_keys=True) + "\n"
    )

    print(
        f"spans emitted={emitted_spans}  raw no_match={raw_relocation['no_match']}"
        f"  snap_rescued={snap_rescued}"
        f"  snap_rejected_by_binder={snap_rejected_by_binder}"
    )
    print(
        f"post-snap: no_match={post['no_match_rate']}  "
        f"relocated={post['relocated_rate']}  missing_span={missing_span_rate}"
    )
    print(f"matches_baseline={matches_baseline}")
    print(f"decision: {payload['decision']}")
    print(f"wrote {args.output} and {args.failures_output} ({len(failures)} failures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
