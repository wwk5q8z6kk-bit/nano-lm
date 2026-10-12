"""Frozen, leakage-audited screen for Nano H7 boundary precision.

The H7 screen is evaluation-only.  It uses newly authored surfaces and open
values, creates exact evidence spans from annotated patient turns, and proves
that its records and normalized line components do not occur in the data
families used by H5/H6.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from nano_ai.adapters.state_span import parse_state_span_summary
from nano_ai.contract import (
    FIELD_ORDER,
    EvidenceSpan,
    FieldName,
    FieldOutput,
    FieldState,
)
from nano_ai.training.state_span_data import (
    StateSpanExample,
    encode_state_span,
    generate_split,
    supported_value_sets,
)
from nano_ai.training.surface_transfer_data import generate_partition

GENERATOR_VERSION = "nano.h7-boundary-screen.v1"
MANIFEST_SCHEMA_VERSION = "nano.h7-boundary-screen-manifest.v1"
SCREEN_SEED = 20260807
SCREEN_WORLDS = 250
SCREEN_RECORDS = SCREEN_WORLDS * 4
VARIANTS = ("normal", "missing", "uncertain", "conflicting")
SOURCE_FAMILIES = ("legacy-train", "legacy-dev", "h4-fit", "h4-calibration")

_QUESTIONS: Mapping[FieldName, tuple[str, ...]] = {
    FieldName.CHIEF_COMPLAINT: (
        "Focus on one symptom: what should the entry say?",
        "Which physical complaint is the exact one to capture?",
    ),
    FieldName.DURATION: (
        "Give the precise elapsed interval, please.",
        "What exact time period belongs in this entry?",
    ),
    FieldName.SEVERITY: (
        "Choose the single intensity word that fits best.",
        "Which exact severity label applies at this moment?",
    ),
    FieldName.MEDICATION: (
        "State the treatment name alone if one was used.",
        "Which exact remedy, if any, belongs in the entry?",
    ),
    FieldName.ALLERGY: (
        "State the triggering substance exactly, if there is one.",
        "Which exact allergen, if any, should be entered?",
    ),
}

_SUPPORTED_TEMPLATES: Mapping[str, tuple[str, ...]] = {
    "value_only": ("{value}!",),
    "value_first": ("{value}; that is the complete entry.",),
    "value_medial": ("The exact wording is {value}, nothing broader.",),
    "value_final": ("The narrow answer I want captured is {value}",),
}
_CORRECTION_TEMPLATES: Mapping[str, tuple[str, ...]] = {
    "value_only": ("Correction — {value}!",),
    "value_first": ("{value}; use that corrected entry.",),
    "value_medial": ("I need to replace it with {value}, specifically.",),
    "value_final": ("Please correct the earlier answer to {value}",),
}
_DENIALS = {
    FieldName.MEDICATION: ("No, nothing yet!",),
    FieldName.ALLERGY: ("No known allergies!",),
}
_UNCERTAIN = (
    "I cannot pin that answer down.",
    "My recollection is too imprecise to answer.",
)

_LEXICONS: Mapping[FieldName, tuple[str, ...]] = {
    FieldName.CHIEF_COMPLAINT: (
        "jaw clicking",
        "wrist burning",
        "heel prickling",
        "rib aching",
        "scalp tingling",
        "elbow buzzing",
        "ankle stinging",
        "hip pulling",
        "finger cramping",
        "heel numbness",
        "jaw fluttering",
        "wrist stiffness",
        "rib soreness",
        "scalp burning",
        "elbow pressure",
        "ankle aching",
        "hip tingling",
        "finger burning",
        "heel tightness",
        "jaw prickling",
    ),
    FieldName.DURATION: tuple(f"{hours} hours" for hours in range(21, 41)),
    FieldName.SEVERITY: ("mild", "moderate", "severe"),
    FieldName.MEDICATION: (
        "calamine lotion",
        "saline mist",
        "zinc lozenges",
        "magnesium gel",
        "menthol balm",
        "cucumber cooling gel",
        "arnica cream",
        "ginger capsules",
        "chamomile tablets",
        "capsaicin patch",
        "witch hazel gel",
        "peppermint capsules",
        "bismuth tablets",
        "lidocaine roll-on",
        "aloe spray",
        "charcoal tablets",
        "lavender balm",
        "turmeric capsules",
        "eucalyptus rub",
        "zinc oxide cream",
    ),
    FieldName.ALLERGY: (
        "buckwheat flour",
        "marigold pollen",
        "sisal fiber",
        "hemp pollen",
        "mango peel",
        "okra pollen",
        "chickpea flour",
        "teff flour",
        "juniper pollen",
        "dahlia pollen",
        "flaxseed",
        "persimmon peel",
        "sorghum flour",
        "agave sap",
        "coconut husk",
        "papaya latex",
        "chestnut flour",
        "sesame oil",
        "millet flour",
        "sunflower pollen",
    ),
}


@dataclass(frozen=True, slots=True)
class _Turn:
    speaker: Literal["Doctor", "Patient"]
    text: str
    field: FieldName
    evidence_text: str | None = None


@dataclass(frozen=True, slots=True)
class _World:
    index: int
    target_field: FieldName
    values: Mapping[FieldName, str | None]
    questions: Mapping[FieldName, str]
    boundaries: Mapping[FieldName, str | None]


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def records_bytes(records: Sequence[StateSpanExample]) -> bytes:
    return b"".join(canonical_json_bytes(record.to_dict()) for record in records)


def _sample_world(index: int, rng: random.Random) -> _World:
    values: dict[FieldName, str | None] = {}
    boundaries: dict[FieldName, str | None] = {}
    for field_index, field in enumerate(FIELD_ORDER):
        choices = _LEXICONS[field]
        if (field is FieldName.MEDICATION and index % 4 == 0) or (
            field is FieldName.ALLERGY and index % 5 == 0
        ):
            values[field] = None
            boundaries[field] = None
        else:
            values[field] = choices[(index * 7 + field_index * 3) % len(choices)]
            boundaries[field] = tuple(_SUPPORTED_TEMPLATES)[
                (index + field_index) % len(_SUPPORTED_TEMPLATES)
            ]
    return _World(
        index=index,
        target_field=FIELD_ORDER[index % len(FIELD_ORDER)],
        values=values,
        questions={field: rng.choice(_QUESTIONS[field]) for field in FIELD_ORDER},
        boundaries=boundaries,
    )


def _supported_turn(
    field: FieldName,
    value: str | None,
    boundary: str | None,
    *,
    correction: bool = False,
) -> _Turn:
    if value is None:
        return _Turn("Patient", _DENIALS[field][0], field, _DENIALS[field][0])
    if boundary is None:
        raise AssertionError("supported value is missing its boundary class")
    templates = _CORRECTION_TEMPLATES if correction else _SUPPORTED_TEMPLATES
    text = templates[boundary][0].format(value=value)
    return _Turn("Patient", text, field, value)


def _alternate_value(field: FieldName, current: str | None) -> str:
    return next(value for value in _LEXICONS[field] if value != current)


def _turns(world: _World, variant: str, rng: random.Random) -> tuple[_Turn, ...]:
    turns: list[_Turn] = []
    for field in FIELD_ORDER:
        turns.append(_Turn("Doctor", world.questions[field], field))
        if field is world.target_field and variant == "missing":
            continue
        if field is world.target_field and variant == "uncertain":
            phrase = rng.choice(_UNCERTAIN)
            turns.append(_Turn("Patient", phrase, field, phrase))
            continue
        turns.append(
            _supported_turn(field, world.values[field], world.boundaries[field])
        )
    if variant == "conflicting":
        target = world.target_field
        alternative = _alternate_value(target, world.values[target])
        boundary = tuple(_CORRECTION_TEMPLATES)[
            (world.index + FIELD_ORDER.index(target) + 1) % len(_CORRECTION_TEMPLATES)
        ]
        turns.extend(
            (
                _Turn("Doctor", "One correction only: what replaces that answer?", target),
                _supported_turn(target, alternative, boundary, correction=True),
            )
        )
    return tuple(turns)


def _render(turns: Sequence[_Turn]) -> tuple[str, Mapping[FieldName, tuple[EvidenceSpan, ...]]]:
    lines: list[str] = []
    spans: dict[FieldName, list[EvidenceSpan]] = {field: [] for field in FIELD_ORDER}
    offset = 0
    for turn in turns:
        line = f"{turn.speaker}: {turn.text}"
        if turn.evidence_text is not None:
            relative = line.index(turn.evidence_text)
            start = offset + relative
            spans[turn.field].append(
                EvidenceSpan(start=start, end=start + len(turn.evidence_text), text=turn.evidence_text)
            )
        lines.append(line)
        offset += len(line) + 1
    transcript = "\n".join(lines)
    return transcript, {field: tuple(field_spans) for field, field_spans in spans.items()}


def _field_outputs(
    world: _World,
    variant: str,
    spans: Mapping[FieldName, tuple[EvidenceSpan, ...]],
) -> tuple[FieldOutput, ...]:
    outputs: list[FieldOutput] = []
    for field in FIELD_ORDER:
        evidence = spans[field]
        if field is world.target_field and variant == "missing":
            outputs.append(FieldOutput(field=field, state=FieldState.MISSING))
        elif field is world.target_field and variant == "uncertain":
            outputs.append(FieldOutput(field=field, state=FieldState.UNCERTAIN, evidence=evidence))
        elif field is world.target_field and variant == "conflicting":
            outputs.append(FieldOutput(field=field, state=FieldState.CONFLICTING, evidence=evidence))
        elif world.values[field] is None:
            outputs.append(FieldOutput(field=field, state=FieldState.ABSENT, evidence=evidence))
        else:
            outputs.append(
                FieldOutput(
                    field=field,
                    state=FieldState.SUPPORTED,
                    value=world.values[field],
                    evidence=evidence,
                )
            )
    return tuple(outputs)


def generate_screen(*, worlds: int = SCREEN_WORLDS, seed: int = SCREEN_SEED) -> tuple[StateSpanExample, ...]:
    """Generate the fixed H7 screen in canonical world/variant order."""

    if isinstance(worlds, bool) or not isinstance(worlds, int) or worlds <= 0:
        raise ValueError("worlds must be a positive integer")
    if worlds % len(FIELD_ORDER):
        raise ValueError("worlds must be divisible by five")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    rng = random.Random(seed)
    records: list[StateSpanExample] = []
    for index in range(worlds):
        world = _sample_world(index, rng)
        for variant in VARIANTS:
            transcript, spans = _render(_turns(world, variant, rng))
            fields = _field_outputs(world, variant, spans)
            for field_index, output in enumerate(fields):
                output.validate_against(transcript, path=f"$.fields[{field_index}]")
            expected_state = (
                None
                if variant == "normal"
                else {
                    "missing": FieldState.MISSING,
                    "uncertain": FieldState.UNCERTAIN,
                    "conflicting": FieldState.CONFLICTING,
                }[variant]
            )
            records.append(
                StateSpanExample(
                    split="dev",
                    example_id=f"dev-h7-screen-{index:04d}-{variant}",
                    world_id=f"dev-world-h7-screen-{index:04d}",
                    variant=variant,
                    target_field=world.target_field,
                    target_state=expected_state,
                    transcript=transcript,
                    target=encode_state_span(fields),
                )
            )
    return tuple(records)


def _source_records() -> Mapping[str, tuple[StateSpanExample, ...]]:
    return {
        "legacy-train": generate_split("train"),
        "legacy-dev": generate_split("dev"),
        "h4-fit": generate_partition("fit"),
        "h4-calibration": generate_partition("calibration"),
    }


def _all_normalization_values() -> tuple[str, ...]:
    values = {
        value
        for split in ("train", "dev")
        for field_values in supported_value_sets(split).values()
        for value in field_values
    }
    values.update(value for field_values in _LEXICONS.values() for value in field_values)
    return tuple(sorted(values, key=lambda item: (-len(item), item.casefold(), item)))


def _line_skeletons(records: Iterable[StateSpanExample]) -> set[str]:
    values = _all_normalization_values()
    skeletons: set[str] = set()
    for record in records:
        for line in record.transcript.splitlines():
            skeleton = line
            for value in values:
                skeleton = skeleton.replace(value, "{value}")
            skeletons.add(skeleton)
    return skeletons


def _state_field_quota(records: Sequence[StateSpanExample]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        for proposal in parse_state_span_summary(record.target, record.transcript):
            counts[f"{proposal.field.value}:{proposal.state.value}"] += 1
    return dict(sorted(counts.items()))


def _require_digest(value: str, role: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{role} must be a lowercase SHA-256 digest")


def build_manifest(
    records: Sequence[StateSpanExample],
    *,
    generator_sha256: str,
    control_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Validate eligibility and return the frozen H7 screen identity."""

    _require_digest(generator_sha256, "generator")
    _require_digest(control_checkpoint_sha256, "control checkpoint")
    if len(records) != SCREEN_RECORDS:
        raise ValueError("H7 screen record count changed")
    if len({record.world_id for record in records}) != SCREEN_WORLDS:
        raise ValueError("H7 screen world count changed")
    if len({record.example_id for record in records}) != len(records):
        raise ValueError("H7 screen contains duplicate record IDs")
    if any(record.split != "dev" for record in records):
        raise ValueError("H7 screen must remain evaluation-only")

    sources = _source_records()
    screen_transcripts = {record.transcript for record in records}
    source_transcripts = {
        record.transcript for family in sources.values() for record in family
    }
    exact_overlap = screen_transcripts & source_transcripts
    screen_skeletons = _line_skeletons(records)
    source_skeletons = _line_skeletons(
        record for family in sources.values() for record in family
    )
    skeleton_overlap = screen_skeletons & source_skeletons
    source_values = {
        value.casefold()
        for split in ("train", "dev")
        for field, field_values in supported_value_sets(split).items()
        if field is not FieldName.SEVERITY
        for value in field_values
    }
    screen_values = {
        value.casefold()
        for field, field_values in _LEXICONS.items()
        if field is not FieldName.SEVERITY
        for value in field_values
    }
    value_overlap = screen_values & source_values
    if exact_overlap:
        raise ValueError("H7 exact transcript overlap with H5/H6 sources")
    if skeleton_overlap:
        raise ValueError("H7 normalized line-component overlap with H5/H6 sources")
    if value_overlap:
        raise ValueError(f"H7 open-value overlap with H5/H6 sources: {sorted(value_overlap)}")

    payload = records_bytes(records)
    transcript_digests = sorted(
        _sha256(record.transcript.encode("utf-8")) for record in records
    )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generator": GENERATOR_VERSION,
        "generator_sha256": generator_sha256,
        "control_checkpoint_sha256": control_checkpoint_sha256,
        "screen": {
            "seed": SCREEN_SEED,
            "records": len(records),
            "worlds": len({record.world_id for record in records}),
            "sha256": _sha256(payload),
            "transcript_multiset_sha256": _sha256(canonical_json_bytes(transcript_digests)),
            "state_field_quota": _state_field_quota(records),
        },
        "isolation": {
            "audited_source_families": list(SOURCE_FAMILIES),
            "exact_transcript_overlap": len(exact_overlap),
            "normalized_line_component_overlap": len(skeleton_overlap),
            "open_value_overlap_excluding_fixed_severity": len(value_overlap),
            "new_world_namespace": True,
            "evaluation_only": True,
            "gold_built_from_annotated_patient_turns": True,
            "private_or_phi_records": 0,
        },
    }


def _write_no_clobber(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def write_screen(
    output_dir: Path,
    *,
    control_checkpoint_sha256: str,
) -> Mapping[str, Any]:
    """Freeze the H7 screen and manifest without overwriting any artifact."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"H7 output directory already exists: {output}")
    generator_sha256 = _sha256(Path(__file__).read_bytes())
    records = generate_screen()
    manifest = build_manifest(
        records,
        generator_sha256=generator_sha256,
        control_checkpoint_sha256=control_checkpoint_sha256,
    )
    if _sha256(Path(__file__).read_bytes()) != generator_sha256:
        raise RuntimeError("H7 generator changed during generation")
    output.mkdir(parents=True, exist_ok=False)
    _write_no_clobber(output / "screen.jsonl", records_bytes(records))
    _write_no_clobber(output / "manifest.json", canonical_json_bytes(manifest))
    return manifest


__all__ = [
    "GENERATOR_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "SCREEN_RECORDS",
    "SCREEN_SEED",
    "SCREEN_WORLDS",
    "build_manifest",
    "generate_screen",
    "records_bytes",
    "write_screen",
]
