"""Measure the denial rule against an independently-authored negation lexicon.

EXPLORATORY. Selects nothing, gates nothing, trains nothing.

Why this exists. `run_generalization_probe` showed the model scores 95.8% on the
four denial phrasings per field it saw in training and 48.2% on the two it did
not. The natural reading -- "use the deterministic rule instead" -- needs its own
test, because `contract._DENIAL_PATTERNS` was hand-authored by someone with
access to the generator's vocabulary. Its 100% coverage of both partitions may be
enumeration rather than generalization.

This probe settles that. It builds denial utterances from the negspacy
`en_clinical` termset (MIT, vendored at `data/external/negspacy/` with its
upstream hash), an inventory written years earlier for a different purpose by
people who never saw this project. Recall against it is the rule's real
generalization, as distinct from its fit to a closed synthetic vocabulary.

    python3 -m nano_ai.training.run_external_denial_probe \
        --output artifacts/nano_h6/analysis/external_denial_coverage.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nano_ai.contract import FieldName, _is_field_denial

# How a patient actually voices each field in reply to "any X?"
_FIELD_NOUNS = {
    FieldName.MEDICATION: (
        "medications", "medication", "meds", "medicine", "any medications",
        "any medicine", "drugs", "pills",
    ),
    FieldName.ALLERGY: (
        "allergies", "allergy", "any allergies", "known allergies",
        "drug allergies", "allergic reactions",
    ),
}

# Frames a spoken reply takes. `{t}` is the external trigger, `{n}` the noun.
_PRECEDING_FRAMES = ("{t} {n}.", "I {t} {n}.", "{t} any {n}.")
_FOLLOWING_FRAMES = ("{n} {t}.", "The {n} {t}.")


def _utterances(termset: dict[str, list[str]]) -> dict[FieldName, list[tuple[str, str]]]:
    """Cross external triggers with field nouns. Returns (utterance, trigger)."""
    built: dict[FieldName, list[tuple[str, str]]] = {}
    for field, nouns in _FIELD_NOUNS.items():
        rows: list[tuple[str, str]] = []
        seen: set[str] = set()
        for group, frames in (
            ("preceding_negations", _PRECEDING_FRAMES),
            ("following_negations", _FOLLOWING_FRAMES),
        ):
            for trigger in termset.get(group, ()):
                for noun in nouns:
                    for frame in frames:
                        text = frame.format(t=trigger, n=noun)
                        text = text[0].upper() + text[1:]
                        if text not in seen:
                            seen.add(text)
                            rows.append((text, trigger))
        built[field] = rows
    return built


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--termset",
        type=Path,
        default=Path("data/external/negspacy/en_clinical_termset.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.termset.read_text())
    termset = source["termset"]
    built = _utterances(termset)

    per_field = {}
    misses_by_trigger: dict[str, int] = {}
    total_hit = total_n = 0
    for field, rows in built.items():
        hit = 0
        examples_missed = []
        for text, trigger in rows:
            if _is_field_denial(field, text):
                hit += 1
            else:
                misses_by_trigger[trigger] = misses_by_trigger.get(trigger, 0) + 1
                if len(examples_missed) < 12:
                    examples_missed.append(text)
        per_field[field.value] = {
            "utterances": len(rows),
            "recognized": hit,
            "recall": round(hit / len(rows), 4) if rows else None,
            "missed_examples": examples_missed,
        }
        total_hit += hit
        total_n += len(rows)

    payload = {
        "schema": "nano.external-denial-coverage.v1",
        "status": "EXPLORATORY -- selects nothing, gates nothing",
        "rule": "nano_ai/contract.py::_is_field_denial (v0 _DENIAL_PATTERNS)",
        "lexicon": {
            k: source[k]
            for k in ("name", "version", "source_uri", "license",
                      "upstream_file_sha256", "role")
        },
        "trigger_counts": {k: len(v) for k, v in termset.items()},
        "per_field": per_field,
        "overall": {
            "utterances": total_n,
            "recognized": total_hit,
            "recall": round(total_hit / total_n, 4) if total_n else None,
        },
        "triggers_never_recognized": sorted(misses_by_trigger),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"lexicon: {source['name']}  ({source['license']})")
    print(f"triggers: {payload['trigger_counts']}\n")
    for field, data in sorted(per_field.items()):
        print(f"  {field:12s} recall {data['recognized']:5d}/{data['utterances']:<6d}"
              f" = {data['recall']:.1%}")
        for text in data["missed_examples"][:6]:
            print(f"                MISS  {text!r}")
    o = payload["overall"]
    print(f"\n  OVERALL      recall {o['recognized']}/{o['utterances']} = {o['recall']:.1%}")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
