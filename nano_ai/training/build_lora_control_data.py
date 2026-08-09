"""Build the LoRA training payload for the cross-model control.

EXPLORATORY control experiment. Produces no Nano checkpoint and touches no
H-cycle gate. Authority record: `papers/DATASET_AUTHORITY_LORA_CONTROL.md`.

The question: does task training on a vocabulary-rich base model eliminate the
surface sensitivity Nano shows? To answer it, the base model must be trained on
*exactly* Nano's task in *exactly* the probe's output format, from *only* the
fit partition.

Three leaks would invalidate the result, and each is refused here rather than
merely avoided:
  1. training on the development partition   -> only `fit.jsonl` is read
  2. training on the vendored external lexicons that the held-out arms are
     built from                              -> asserted absent from every prompt
  3. training on the two development denial phrasings
                                             -> asserted absent from every prompt

Output is JSONL in mlx-lm / HF chat format, split train/valid.

    python3 -m nano_ai.training.build_lora_control_data \
        --out artifacts/nano_h6/lora_control/data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from nano_ai.adapters.state_span import parse_state_span_summary
from nano_ai.contract import FieldState
from nano_ai.surface_arms import DEV_DENIAL_ALLERGY, DEV_DENIAL_MEDICATION
from nano_ai.training.run_crossmodel_surface_probe import (
    _EXPECTED,
    _FIELD_QUESTION,
    _PROMPT,
)
from nano_ai.training.span_port_format import (
    PROMPT_WITH_SPANS,
    format_span_answer,
)

FIT_SHA256 = "79f7581efbd989f4"  # prefix; full value in the authority record


def _leak_guard(text: str, external_phrases: tuple[str, ...]) -> str | None:
    """Return the offending phrase if a training prompt leaks an eval phrasing."""
    for phrase in (DEV_DENIAL_MEDICATION, DEV_DENIAL_ALLERGY, *external_phrases):
        if phrase and phrase in text:
            return phrase
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/nano_h5/data"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--valid-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument(
        "--with-spans", action="store_true",
        help="v2 target: emit the grounding span text beside the state, so the "
             "tuned model produces something relocatable. See "
             "papers/DECISION_SPAN_PORT.md -- the v1 build parsed gold.spans and "
             "discarded them, which is why the tuned model emitted no span.",
    )
    parser.add_argument(
        "--balance", action="store_true",
        help="downsample every label to the minority-class count. The v1 build "
             "took the natural mix (supported 71.5%%, absent 22.9%%, missing 5.6%%) "
             "and four of five bases in the transfer curve collapsed on exactly "
             "the 5.6%% class -- see papers/RESULT_TRANSFER_CURVE.md.",
    )
    args = parser.parse_args()

    from nano_ai.surface_arms import DENIAL_ARMS
    from nano_ai.training import replay_mixture_data

    # Every held-out arm phrasing, which must never appear in training text.
    external = tuple(
        replacement
        for arm in DENIAL_ARMS
        if not arm.in_distribution
        for _, replacement in arm.mapping
    )

    bundle = replay_mixture_data.load_replay_mixture_dataset(args.data_dir)
    fit = bundle.fit
    print(f"fit examples: {len(fit)}   (calibration and development NOT read)")

    rows: list[dict] = []
    row_labels: list[str] = []
    states = Counter()
    for example in fit:
        for gold in parse_state_span_summary(example.target, example.transcript):
            if gold.field.value not in _FIELD_QUESTION:
                continue
            if gold.state not in _EXPECTED:
                continue  # uncertain / conflicting are not part of this 3-way task
            leak = _leak_guard(example.transcript, external)
            if leak is not None:
                raise SystemExit(
                    f"LEAK: fit transcript contains held-out phrasing {leak!r}. "
                    "Refusing to build training data."
                )
            template = PROMPT_WITH_SPANS if args.with_spans else _PROMPT
            prompt = template.format(
                transcript=example.transcript,
                topic=_FIELD_QUESTION[gold.field.value],
            )
            answer = _EXPECTED[gold.state]
            if args.with_spans:
                # Route (b): emit span TEXT; relocate later. Prompt and target
                # must agree — the first v2 build kept the state-only "one word"
                # prompt while writing quoted spans (measurement defect).
                answer = format_span_answer(
                    answer, tuple(span.text for span in gold.spans)
                )
            rows.append(
                {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": answer},
                    ]
                }
            )
            row_labels.append(gold.state.value)
            states[gold.state.value] += 1

    print(f"examples built: {len(rows)}   label mix: {dict(states)}")
    if len(states) < 2:
        raise SystemExit("refusing to train on a single-label dataset")

    if args.balance:
        by_label: dict[str, list] = {}
        for row, label in zip(rows, row_labels):
            by_label.setdefault(label, []).append(row)
        keep = min(len(v) for v in by_label.values())
        balanced_rng = random.Random(args.seed)
        rows = []
        for label in sorted(by_label):
            pool = by_label[label]
            balanced_rng.shuffle(pool)
            rows.extend(pool[:keep])
        states = Counter({k: keep for k in by_label})
        print(f"balanced: {keep} per label x {len(by_label)} labels = {len(rows)} rows")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    cut = max(1, int(len(rows) * args.valid_fraction))
    valid, train = rows[:cut], rows[cut:]

    args.out.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, part in (("train", train), ("valid", valid)):
        path = args.out / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in part))
        written[name] = {
            "records": len(part),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        print(f"  {name}: {len(part):,} records  {path.stat().st_size:,} B")

    (args.out / "PROVENANCE.json").write_text(
        json.dumps(
            {
                "schema": "nano.lora-control-data.v1",
                "purpose": "cross-model control -- holds task training constant, "
                           "varies vocabulary exposure and scale",
                "source_partition": "artifacts/nano_h5/data/fit.jsonl (fit only)",
                "authority_record": "papers/DATASET_AUTHORITY_LORA_CONTROL.md",
                "development_partition_read": False,
                "calibration_partition_read": False,
                "external_lexicons_in_training_text": False,
                "leak_guard": "every fit transcript checked against the two "
                              "development denial phrasings and all held-out arm "
                              "phrasings; build aborts on any match",
                "label_mix": dict(states),
                "balanced": bool(args.balance),
                "target_format": "state_and_span_text" if args.with_spans else "state_label_only",
                "prompt_format": (
                    "state_and_span_line" if args.with_spans else "state_label_one_word"
                ),
                "seed": args.seed,
                "files": written,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"wrote {args.out}/PROVENANCE.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
