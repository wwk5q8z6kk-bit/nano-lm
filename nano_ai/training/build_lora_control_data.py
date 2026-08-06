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
from nano_ai.training.run_crossmodel_surface_probe import (
    _EXPECTED,
    _FIELD_QUESTION,
    _PROMPT,
)
from nano_ai.surface_arms import DEV_DENIAL_ALLERGY, DEV_DENIAL_MEDICATION

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
            prompt = _PROMPT.format(
                transcript=example.transcript,
                topic=_FIELD_QUESTION[gold.field.value],
            )
            rows.append(
                {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": _EXPECTED[gold.state]},
                    ]
                }
            )
            states[gold.state.value] += 1

    print(f"examples built: {len(rows)}   label mix: {dict(states)}")
    if len(states) < 2:
        raise SystemExit("refusing to train on a single-label dataset")

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
