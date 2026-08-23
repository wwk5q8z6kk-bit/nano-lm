"""Native training data loading — disjoint distill train partition only."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from nanoscribe.distill_train_suite import distill_train_cases, distill_train_manifest
from nanoscribe.harness import HarnessCase
from nanoscribe.prompt import build_span_port_prompt


@dataclass(frozen=True, slots=True)
class NativeTrainExample:
    encounter_id: str
    prompt: str
    target: str
    atom_id: str


def _target_from_case(case: HarnessCase) -> str:
    if not case.gold.atoms:
        return "NOT_MENTIONED"
    atom = case.gold.atoms[0]
    if atom.assertion_state.value == "denied":
        return f"DENIED: {atom.raw_value}"
    if atom.assertion_state.value == "uncertain":
        return f"UNCERTAIN: {atom.raw_value}"
    quote = atom.raw_value
    for span in case.gold.evidence:
        if span.text:
            quote = span.text
            break
    return f"ASSERTED: {quote}"


def examples_from_cases(cases: list[HarnessCase]) -> list[NativeTrainExample]:
    rows: list[NativeTrainExample] = []
    for case in cases:
        for spec in case.atom_specs:
            prompt = build_span_port_prompt(case.model_input.source, spec)
            rows.append(
                NativeTrainExample(
                    encounter_id=case.encounter_id,
                    prompt=prompt,
                    target=_target_from_case(case),
                    atom_id=spec.atom_id,
                )
            )
    return rows


def load_train_examples(path: str | Path | None = None) -> list[NativeTrainExample]:
    if path is None:
        return examples_from_cases(distill_train_cases())
    data = json.loads(Path(path).read_text())
    if data.get("schema") == "nano.distill.train.v1":
        return examples_from_cases(distill_train_cases())
    entries = data.get("entries", [])
    rows: list[NativeTrainExample] = []
    for entry in entries:
        rows.append(
            NativeTrainExample(
                encounter_id=str(entry["encounter_id"]),
                prompt=str(entry.get("prompt", "")),
                target=str(entry.get("target", "NOT_MENTIONED")),
                atom_id=str(entry.get("atom_id", entry["encounter_id"])),
            )
        )
    return rows


def export_distill_train_json(path: str | Path = "artifacts/campaign/p1_distill_train_v1.json") -> dict[str, Any]:
    cases = distill_train_cases()
    examples = examples_from_cases(cases)
    payload = {
        **distill_train_manifest(),
        "entries": [
            {
                "encounter_id": ex.encounter_id,
                "atom_id": ex.atom_id,
                "prompt": ex.prompt,
                "target": ex.target,
            }
            for ex in examples
        ],
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


class NativeBatchIterator:
    """Simple batch iterator for CPU smoke training."""

    def __init__(self, examples: list[NativeTrainExample], batch_size: int, seed: int = 0) -> None:
        self.examples = list(examples)
        self.batch_size = max(1, batch_size)
        self.seed = seed
        self._index = 0

    def __iter__(self) -> Iterator[list[NativeTrainExample]]:
        import random

        rng = random.Random(self.seed)
        order = list(range(len(self.examples)))
        rng.shuffle(order)
        for start in range(0, len(order), self.batch_size):
            idxs = order[start : start + self.batch_size]
            yield [self.examples[i] for i in idxs]

    def __len__(self) -> int:
        if not self.examples:
            return 0
        return (len(self.examples) + self.batch_size - 1) // self.batch_size
