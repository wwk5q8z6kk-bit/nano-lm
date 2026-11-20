"""Gold-agreement measurement for route-(b) relocated spans. Report-only.

Closes the validity gap left open across the span-port series: `relocated`
means the emitted text is unique patient-turn text — it does NOT mean the
text is the gold evidence. A model quoting any unique patient sentence
would count as grounded. This script regenerates the probe deterministically
(accepted configuration: baseline prompt + normalization snap fallback) and
compares every relocated span's position against the gold spans of its
field.

Per relocated span: `exact` (same text and start), `overlapping` (≥1 shared
character with a gold span), or `disjoint` (grounded to non-evidence text —
the dangerous class). Covers both the denial arms (DENIED gold) and the
control block's supported fields (STATED gold).

NO BARS. This produces the baseline data from which P3's preregistered
gold-denominator floors are to be chosen (`papers/FORWARD_PLAN_20260806.md`
§5 P3); choosing a threshold here, after seeing results, would invert the
freeze discipline.

    python3 -m nano_ai.training.evaluate_span_gold_agreement \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --adapter-path artifacts/nano_h6/span_port/adapters_Qwen2.5-1.5B \
        --output artifacts/nano_h6/span_port/gold_agreement.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from nano_ai.adapters.state_span import (
    StateSpanFormatError,
    _locate_unique_patient_span,
    parse_state_span_summary,
)
from nano_ai.contract import FieldState
from nano_ai.surface import apply_arm
from nano_ai.surface_arms import DENIAL_ARMS
from nano_ai.training.span_port_format import (
    FIELD_QUESTION,
    PROMPT_WITH_SPANS,
    parse_label_and_spans,
)
from nano_ai.training.span_snap import snap_to_patient_span


def _agreement(located, gold_spans) -> tuple[str, int]:
    """Classify a located span against gold spans; return (class, overlap)."""
    best_overlap = 0
    exact = False
    for gold in gold_spans:
        overlap = max(
            0, min(located.end, gold.end) - max(located.start, gold.start)
        )
        best_overlap = max(best_overlap, overlap)
        if gold.start == located.start and gold.text == located.text:
            exact = True
    if exact:
        return "exact", best_overlap
    if best_overlap > 0:
        return "overlapping", best_overlap
    return "disjoint", 0


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
        "--constrained",
        action="store_true",
        help="substring-constrained decoding (papers/DECISION_REGISTER_RECIPE.md)",
    )
    parser.add_argument(
        "--field-filter",
        action="store_true",
        help="drop quotes naming the other field's topic before scoring "
        "(papers/PREREG_FIELD_FILTER.md)",
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

    supported = []
    for example in examples:
        for gold in parse_state_span_summary(example.target, example.transcript):
            if (
                gold.field.value in FIELD_QUESTION
                and gold.state is FieldState.SUPPORTED
                and gold.spans
            ):
                supported.append((example.transcript, gold.field.value, gold.spans))
    supported = supported[: args.limit]
    print(f"supported control fields with gold spans: {len(supported)}")

    model, tokenizer = load(args.model, adapter_path=str(args.adapter_path))
    print(f"model: {args.model}   adapter: {args.adapter_path}")

    string_index = None
    if args.constrained:
        from nano_ai.training.constrained_decode import (
            SubstringConstraint,
            build_string_index,
        )

        string_index = build_string_index(tokenizer)

    def ask(transcript: str, topic: str) -> tuple[str | None, tuple[str, ...]]:
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
        return parse_label_and_spans(out)

    def score(transcript: str, text: str, gold_spans, sink: Counter, rows, meta):
        """Snap-fallback locate, then gold agreement. Mirrors the accepted probe."""
        if args.field_filter and meta.get("field") in ("medication", "allergy"):
            from nano_ai.training.span_field_filter import (
                rejects_cross_field_quote,
            )

            if rejects_cross_field_quote(meta["field"], text):
                sink["field_filtered"] += 1
                return
        candidate = text
        try:
            located = _locate_unique_patient_span(transcript, candidate)
        except StateSpanFormatError:
            snapped = snap_to_patient_span(transcript, candidate)
            if snapped is None:
                sink["not_relocated"] += 1
                return
            try:
                located = _locate_unique_patient_span(transcript, snapped)
            except StateSpanFormatError:
                sink["not_relocated"] += 1
                return
        cls, overlap = _agreement(located, gold_spans)
        sink[cls] += 1
        if cls == "disjoint":
            rows.append(
                {
                    **meta,
                    "text": located.text,
                    "start": located.start,
                    "gold": [g.text for g in gold_spans],
                }
            )

    arm_counts: dict[str, Counter] = {}
    disjoint_rows: list[dict] = []
    for arm in DENIAL_ARMS:
        sink = arm_counts.setdefault(arm.label, Counter())
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
                label, spans = ask(transcript, topic)
                if label is None or not gold.spans:
                    continue
                for text in spans:
                    score(
                        transcript,
                        text,
                        gold.spans,
                        sink,
                        disjoint_rows,
                        {"section": "arms", "arm": arm.label, "doc_index": doc_index, "field": field},
                    )
        print(f"  {arm.label:16s} {dict(sink)}")

    control_counts: Counter = Counter()
    for transcript, field, gold_spans in supported:
        label, spans = ask(transcript, FIELD_QUESTION[field])
        if label is None:
            continue
        for text in spans:
            score(
                transcript,
                text,
                gold_spans,
                control_counts,
                disjoint_rows,
                {"section": "control_supported", "arm": None, "doc_index": None, "field": field},
            )
    print(f"  control_supported {dict(control_counts)}")

    def summarize(counts: Counter) -> dict:
        located = counts["exact"] + counts["overlapping"] + counts["disjoint"]
        return {
            "relocated": located,
            "not_relocated": counts["not_relocated"],
            "field_filtered": counts["field_filtered"],
            "exact": counts["exact"],
            "overlapping": counts["overlapping"],
            "disjoint": counts["disjoint"],
            "agreement_rate": round(
                (counts["exact"] + counts["overlapping"]) / located, 4
            )
            if located
            else None,
            "exact_rate": round(counts["exact"] / located, 4) if located else None,
            "disjoint_rate": round(counts["disjoint"] / located, 4)
            if located
            else None,
        }

    arms_total: Counter = Counter()
    for sink in arm_counts.values():
        arms_total.update(sink)

    payload = {
        "schema": "nano.span-gold-agreement.v1",
        "purpose": (
            "report-only baseline for P3 floor selection; no bars, no decision"
        ),
        "configuration": "baseline prompt + snap fallback (accepted per "
        "papers/RESULT_SPAN_PORT_B_SNAP.md)",
        "model": args.model,
        "adapter_path": str(args.adapter_path),
        "arms_summary": summarize(arms_total),
        "per_arm": {label: summarize(c) for label, c in arm_counts.items()},
        "control_supported_summary": summarize(control_counts),
        "disjoint_examples": disjoint_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"arms:    {payload['arms_summary']}")
    print(f"control: {payload['control_supported_summary']}")
    print(f"disjoint examples: {len(disjoint_rows)}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
