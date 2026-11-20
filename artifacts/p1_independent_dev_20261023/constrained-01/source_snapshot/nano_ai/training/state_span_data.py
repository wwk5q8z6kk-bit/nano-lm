"""Deterministic, leakage-isolated data for Nano's native state/span target.

This generator deliberately has no benchmark dependency.  It creates paired
normal/missing/uncertain/conflicting variants from split-specific worlds so the
model must learn field state and exact patient-span emission together.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nano_ai.adapters.deterministic_v0 import _extract_fields
from nano_ai.contract import FIELD_ORDER, FieldName, FieldOutput, FieldState, NanoInput

DATASET_SCHEMA_VERSION = "nano.state-span-dataset.v0"
MANIFEST_SCHEMA_VERSION = "nano.state-span-manifest.v0"
TARGET_GRAMMAR_VERSION = "nano-state-span-grammar-v0"
TRAIN_SEED = 20260803
DEV_SEED = 20260804
TRAIN_WORLDS = 3_000
DEV_WORLDS = 250
VARIANTS = ("normal", "missing", "uncertain", "conflicting")
STATE_VARIANTS = {
    "missing": FieldState.MISSING,
    "uncertain": FieldState.UNCERTAIN,
    "conflicting": FieldState.CONFLICTING,
}

_LABELS = {
    FieldName.CHIEF_COMPLAINT: "CC",
    FieldName.DURATION: "DUR",
    FieldName.SEVERITY: "SEV",
    FieldName.MEDICATION: "MED",
    FieldName.ALLERGY: "ALG",
}

# These values are known historical test sentinels.  The generator does not
# inspect any benchmark artifact; the fixed blacklist prevents accidental reuse.
FORBIDDEN_HISTORICAL_SENTINELS = frozenset(
    {
        "toothache",
        "neck pain",
        "heartburn",
        "melatonin",
        "throat lozenges",
        "sulfa drugs",
    }
)

_QUESTIONS: Mapping[str, Mapping[FieldName, tuple[str, ...]]] = {
    "train": {
        FieldName.CHIEF_COMPLAINT: (
            "Good morning, what brings you in today?",
            "Hello, what can I do for you?",
            "Hi there, what seems to be the trouble?",
            "So, tell me what's going on.",
        ),
        FieldName.DURATION: (
            "How long has this been going on?",
            "When did it start?",
        ),
        FieldName.SEVERITY: (
            "How bad would you say it is?",
            "Is it mild, moderate, or severe?",
        ),
        FieldName.MEDICATION: (
            "Have you taken anything for it?",
            "Are you on any medication for this?",
        ),
        FieldName.ALLERGY: (
            "Any allergies I should know about?",
            "Are you allergic to anything?",
        ),
    },
    "dev": {
        FieldName.CHIEF_COMPLAINT: (
            "What brings you to the clinic today?",
            "Morning — what's been bothering you?",
            "Come in, have a seat. What's the issue today?",
        ),
        FieldName.DURATION: (
            "How many days has it been?",
            "Since when have you had it?",
        ),
        FieldName.SEVERITY: ("On a scale from mild to severe, where is it?",),
        FieldName.MEDICATION: ("Did you try any medicine?",),
        FieldName.ALLERGY: ("Do you have any known allergies?",),
    },
}

_ANSWERS: Mapping[str, Mapping[FieldName, tuple[str, ...]]] = {
    "train": {
        FieldName.CHIEF_COMPLAINT: (
            "I've been having {value}.",
            "I came in because of {value}.",
            "It's {value}, doctor.",
            "Well, I've got {value} that won't go away.",
            "I'm dealing with {value}.",
        ),
        FieldName.DURATION: (
            "For about {value} now.",
            "For about {value}.",
            "It started {value} ago.",
            "Around {value}.",
            "I'd say it's been {value}.",
        ),
        FieldName.SEVERITY: (
            "I'd call it {value}.",
            "It's {value}, I would say.",
        ),
        FieldName.MEDICATION: (
            "I've been taking {value}.",
            "Just {value}.",
            "Some {value}, but it barely helps.",
        ),
        FieldName.ALLERGY: (
            "I'm allergic to {value}.",
            "Yes, {value}.",
            "Just {value}.",
        ),
    },
    "dev": {
        FieldName.CHIEF_COMPLAINT: (
            "Honestly, {value} has been troubling me.",
            "It started as {value} and hasn't stopped.",
        ),
        FieldName.DURATION: (
            "Since about {value} back.",
            "Started roughly {value} prior.",
            "On and off for maybe {value}.",
            "Coming up on {value} now.",
        ),
        FieldName.SEVERITY: (
            "Pretty {value}.",
            "Definitely {value}.",
        ),
        FieldName.MEDICATION: ("Only {value} so far.",),
        FieldName.ALLERGY: ("I do — {value}.",),
    },
}

_DENIALS: Mapping[str, Mapping[FieldName, tuple[str, ...]]] = {
    "train": {
        FieldName.MEDICATION: (
            "No, nothing yet.",
            "I haven't taken anything.",
        ),
        FieldName.ALLERGY: (
            "No allergies.",
            "Not that I know of.",
        ),
    },
    "dev": {
        FieldName.MEDICATION: ("Nothing at all.",),
        FieldName.ALLERGY: ("None whatsoever.",),
    },
}

_UNCERTAIN = {
    "train": ("I'm not certain.", "I cannot recall.", "Hard to say."),
    "dev": ("I'm unsure.", "I don't remember.", "I cannot say for certain."),
}

_TRAIN_CC_PARTS = (
    "temple",
    "cheek",
    "gum",
    "eyelid",
    "forearm",
    "thumb",
    "palm",
    "knuckle",
    "shin",
    "calf",
    "thigh",
    "groin",
    "tailbone",
    "collarbone",
    "sternum",
    "abdomen",
    "flank",
    "navel",
    "nostril",
    "sinus",
    "tongue",
    "lip",
    "foot arch",
    "big toe",
)
_TRAIN_CC_SIGNS = (
    "pressure",
    "throbbing",
    "tenderness",
    "tightness",
    "irritation",
    "twitching",
    "weakness",
    "heaviness",
    "sensitivity",
    "warmth",
)
_DEV_CC_PARTS = (
    "brow",
    "earlobe",
    "shoulder blade",
    "kneecap",
    "hamstring",
    "Achilles tendon",
    "sole",
    "diaphragm",
    "pelvis",
    "neck muscle",
)
_DEV_CC_SIGNS = (
    "aching",
    "burning",
    "soreness",
    "numbness",
    "tingling",
    "cramping",
    "swelling",
    "itching",
)

_TRAIN_MEDICATIONS = (
    "gabapentin capsule",
    "omeprazole tablet",
    "amlodipine tablet",
    "metformin tablet",
    "diclofenac gel",
    "lidocaine patch",
    "fluticasone spray",
    "budesonide inhaler",
    "albuterol inhaler",
    "sumatriptan tablet",
    "propranolol tablet",
    "losartan tablet",
    "atorvastatin tablet",
    "levothyroxine tablet",
    "sertraline tablet",
    "fluoxetine capsule",
    "duloxetine capsule",
    "baclofen tablet",
    "cyclobenzaprine tablet",
    "celecoxib capsule",
    "ketoconazole cream",
    "mupirocin ointment",
    "azelaic acid gel",
    "benzoyl peroxide wash",
    "calcium carbonate chew",
    "electrolyte powder",
    "ginger capsule",
    "peppermint oil capsule",
    "probiotic capsule",
    "fiber supplement",
    "iron tablet",
    "folate tablet",
    "vitamin d drops",
    "vitamin b12 spray",
    "coenzyme q10 capsule",
    "turmeric capsule",
    "glucosamine tablet",
    "chondroitin capsule",
    "meloxicam tablet",
    "indomethacin capsule",
    "acetazolamide tablet",
    "meclizine tablet",
    "ondansetron tablet",
    "loperamide capsule",
    "polyethylene glycol powder",
    "docusate capsule",
    "bisacodyl tablet",
    "simethicone chew",
    "guaifenesin tablet",
    "dextromethorphan syrup",
    "benzonatate capsule",
    "ipratropium spray",
    "mometasone cream",
    "tacrolimus ointment",
    "clotrimazole cream",
    "terbinafine cream",
    "acyclovir tablet",
    "valacyclovir tablet",
    "nitrofurantoin capsule",
    "fosfomycin packet",
    "cefdinir capsule",
    "levofloxacin tablet",
    "rifaximin tablet",
    "mesalamine tablet",
)
_DEV_MEDICATIONS = (
    "topiramate tablet",
    "pantoprazole tablet",
    "diltiazem capsule",
    "empagliflozin tablet",
    "capsaicin cream",
    "scopolamine patch",
    "azelastine spray",
    "tiotropium inhaler",
    "rizatriptan wafer",
    "carvedilol tablet",
    "pravastatin tablet",
    "escitalopram tablet",
    "tizanidine tablet",
    "etodolac tablet",
    "adapalene gel",
    "magnesium glycinate capsule",
    "riboflavin tablet",
    "psyllium powder",
    "famciclovir tablet",
    "cefpodoxime tablet",
    "hyoscyamine tablet",
    "nystatin cream",
    "zafirlukast tablet",
    "sucralfate suspension",
)

_TRAIN_ALLERGIES = (
    "almonds",
    "cashews",
    "walnuts",
    "pistachios",
    "hazelnuts",
    "pecans",
    "sesame seeds",
    "mustard seed",
    "celery root",
    "lupin flour",
    "chickpeas",
    "lentils",
    "soybeans",
    "wheat",
    "rye",
    "barley",
    "oats",
    "kiwi fruit",
    "mango",
    "papaya",
    "avocado",
    "banana",
    "peaches",
    "plums",
    "cherries",
    "strawberries",
    "tomatoes",
    "egg whites",
    "cow's milk",
    "cod fish",
    "nickel",
    "cobalt",
    "chromium",
    "adhesive tape",
    "wool",
    "dust mites",
    "cat dander",
    "dog dander",
    "ragweed pollen",
    "birch pollen",
    "grass pollen",
    "oak pollen",
    "cephalexin",
    "amoxicillin",
    "azithromycin",
    "clindamycin",
    "doxycycline",
    "tramadol",
    "codeine",
    "morphine",
    "chlorhexidine",
    "iodine contrast",
    "lanolin",
    "fragrance mix",
    "tea tree oil",
    "bee venom",
    "wasp venom",
    "fire ant venom",
    "horse dander",
    "rabbit dander",
    "sunflower seeds",
    "pumpkin seeds",
    "green peas",
    "buckwheat",
)
_DEV_ALLERGIES = (
    "macadamia nuts",
    "pine nuts",
    "poppy seeds",
    "fennel seed",
    "dragon fruit",
    "passion fruit",
    "apricots",
    "cranberries",
    "goat's milk",
    "quail eggs",
    "tilapia",
    "anchovies",
    "acrylic adhesive",
    "rubber accelerator",
    "guinea pig dander",
    "cedar pollen",
    "elm pollen",
    "cefuroxime",
    "linezolid",
    "vancomycin",
    "hydromorphone",
    "povidone iodine",
    "balsam of peru",
    "hornet venom",
)


@dataclass(frozen=True, slots=True)
class StateSpanExample:
    """One paired training or development record."""

    split: str
    example_id: str
    world_id: str
    variant: str
    target_field: FieldName
    target_state: FieldState | None
    transcript: str
    target: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DATASET_SCHEMA_VERSION,
            "split": self.split,
            "example_id": self.example_id,
            "world_id": self.world_id,
            "variant": self.variant,
            "target_field": self.target_field.value,
            "target_state": (
                None if self.target_state is None else self.target_state.value
            ),
            "transcript": self.transcript,
            "target": self.target,
        }

    @classmethod
    def from_dict(cls, value: object) -> StateSpanExample:
        if not isinstance(value, dict):
            raise TypeError("state-span record must be an object")
        expected = {
            "schema_version",
            "split",
            "example_id",
            "world_id",
            "variant",
            "target_field",
            "target_state",
            "transcript",
            "target",
        }
        if set(value) != expected or value["schema_version"] != DATASET_SCHEMA_VERSION:
            raise ValueError("invalid state-span record schema")
        raw_state = value["target_state"]
        state = None if raw_state is None else FieldState(raw_state)
        example = cls(
            split=_text(value["split"], "split"),
            example_id=_text(value["example_id"], "example_id"),
            world_id=_text(value["world_id"], "world_id"),
            variant=_text(value["variant"], "variant"),
            target_field=FieldName(value["target_field"]),
            target_state=state,
            transcript=_text(value["transcript"], "transcript"),
            target=_text(value["target"], "target"),
        )
        if example.split not in {"train", "dev"} or example.variant not in VARIANTS:
            raise ValueError("invalid state-span split or variant")
        if (example.variant == "normal") != (example.target_state is None):
            raise ValueError("normal/state variant mismatch")
        if example.variant != "normal" and STATE_VARIANTS[example.variant] is not state:
            raise ValueError("variant and target_state disagree")
        return example


@dataclass(frozen=True, slots=True)
class _World:
    values: Mapping[FieldName, str | None]
    target_field: FieldName
    questions: Mapping[FieldName, str]
    answers: Mapping[FieldName, str]


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be non-empty edge-trimmed text")
    return value


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


def _complaints(split: str) -> tuple[str, ...]:
    if split == "train":
        return tuple(
            f"{part} {sign}" for part in _TRAIN_CC_PARTS for sign in _TRAIN_CC_SIGNS
        )
    return tuple(f"{part} {sign}" for part in _DEV_CC_PARTS for sign in _DEV_CC_SIGNS)


def _durations(split: str) -> tuple[str, ...]:
    if split == "train":
        return (
            *(f"{n} days" for n in range(2, 13)),
            *(f"{n} weeks" for n in range(1, 5)),
        )
    return (
        *(f"{n} days" for n in range(13, 20)),
        *(f"{n} weeks" for n in range(5, 9)),
    )


def supported_value_sets(split: str) -> Mapping[FieldName, frozenset[str]]:
    """Return the fixed split lexicons without generating any records."""

    if split not in {"train", "dev"}:
        raise ValueError("split must be train or dev")
    return {
        FieldName.CHIEF_COMPLAINT: frozenset(_complaints(split)),
        FieldName.DURATION: frozenset(_durations(split)),
        FieldName.SEVERITY: frozenset({"mild", "moderate", "severe"}),
        FieldName.MEDICATION: frozenset(
            _TRAIN_MEDICATIONS if split == "train" else _DEV_MEDICATIONS
        ),
        FieldName.ALLERGY: frozenset(
            _TRAIN_ALLERGIES if split == "train" else _DEV_ALLERGIES
        ),
    }


def assert_static_split_isolation() -> None:
    """Fail if an edit introduces train/dev or historical-value contamination."""

    train = supported_value_sets("train")
    dev = supported_value_sets("dev")
    for field in FIELD_ORDER:
        if field is not FieldName.SEVERITY and train[field] & dev[field]:
            raise ValueError(f"train/dev value overlap for {field.value}")
    all_open_values = set().union(
        *(values for field, values in train.items() if field is not FieldName.SEVERITY),
        *(values for field, values in dev.items() if field is not FieldName.SEVERITY),
    )
    if {value.casefold() for value in all_open_values} & FORBIDDEN_HISTORICAL_SENTINELS:
        raise ValueError("historical sentinel leaked into native state/span data")
    for field in FIELD_ORDER:
        if set(_QUESTIONS["train"][field]) & set(_QUESTIONS["dev"][field]):
            raise ValueError(f"train/dev question overlap for {field.value}")
        if set(_ANSWERS["train"][field]) & set(_ANSWERS["dev"][field]):
            raise ValueError(f"train/dev answer overlap for {field.value}")
    for field in (FieldName.MEDICATION, FieldName.ALLERGY):
        if set(_DENIALS["train"][field]) & set(_DENIALS["dev"][field]):
            raise ValueError(f"train/dev denial overlap for {field.value}")
    if set(_UNCERTAIN["train"]) & set(_UNCERTAIN["dev"]):
        raise ValueError("train/dev uncertainty phrase overlap")


def _render_answer(
    split: str, field: FieldName, value: str | None, rng: random.Random
) -> str:
    if value is None:
        return rng.choice(_DENIALS[split][field])
    return rng.choice(_ANSWERS[split][field]).format(value=value)


def _sample_world(split: str, index: int, rng: random.Random) -> _World:
    lexicons = supported_value_sets(split)
    values: dict[FieldName, str | None] = {
        FieldName.CHIEF_COMPLAINT: rng.choice(
            tuple(sorted(lexicons[FieldName.CHIEF_COMPLAINT]))
        ),
        FieldName.DURATION: rng.choice(tuple(sorted(lexicons[FieldName.DURATION]))),
        FieldName.SEVERITY: rng.choice(tuple(sorted(lexicons[FieldName.SEVERITY]))),
        FieldName.MEDICATION: (
            None
            if index % 4 == 0
            else rng.choice(tuple(sorted(lexicons[FieldName.MEDICATION])))
        ),
        FieldName.ALLERGY: (
            None
            if index % 5 == 0
            else rng.choice(tuple(sorted(lexicons[FieldName.ALLERGY])))
        ),
    }
    questions = {field: rng.choice(_QUESTIONS[split][field]) for field in FIELD_ORDER}
    answers = {
        field: _render_answer(split, field, values[field], rng) for field in FIELD_ORDER
    }
    return _World(
        values=values,
        target_field=FIELD_ORDER[index % len(FIELD_ORDER)],
        questions=questions,
        answers=answers,
    )


def _normal_lines(world: _World) -> list[tuple[str, str, FieldName | None]]:
    lines: list[tuple[str, str, FieldName | None]] = []
    for field in FIELD_ORDER:
        lines.append(("Doctor", world.questions[field], field))
        lines.append(("Patient", world.answers[field], field))
    return lines


def _alternate_value(split: str, field: FieldName, current: str | None) -> str:
    candidates = supported_value_sets(split)[field]
    return next(value for value in sorted(candidates) if value != current)


def _variant_lines(
    split: str,
    world: _World,
    variant: str,
    rng: random.Random,
) -> list[tuple[str, str, FieldName | None]]:
    lines = _normal_lines(world)
    target = world.target_field
    answer_index = FIELD_ORDER.index(target) * 2 + 1
    if variant == "normal":
        return lines
    if variant == "missing":
        del lines[answer_index]
        return lines
    if variant == "uncertain":
        lines[answer_index] = ("Patient", rng.choice(_UNCERTAIN[split]), target)
        return lines
    if variant != "conflicting":
        raise ValueError(f"unsupported variant: {variant}")
    alternative = _alternate_value(split, target, world.values[target])
    lines.extend(
        (
            ("Doctor", world.questions[target], target),
            (
                "Patient",
                rng.choice(_ANSWERS[split][target]).format(value=alternative),
                target,
            ),
        )
    )
    return lines


def _transcript(lines: Sequence[tuple[str, str, FieldName | None]]) -> str:
    return "\n".join(f"{speaker}: {text}" for speaker, text, _ in lines)


def encode_state_span(fields: Iterable[FieldOutput]) -> str:
    """Encode contract fields as the exact five-slot native candidate grammar."""

    encoded: list[str] = []
    for expected, output in zip(FIELD_ORDER, fields, strict=True):
        if output.field is not expected:
            raise ValueError("fields are not in canonical order")
        if output.state is FieldState.SUPPORTED:
            payload = output.evidence[0].text
            code = f"S[{payload}]"
        elif output.state is FieldState.ABSENT:
            payload = output.evidence[0].text
            code = f"A[{payload}]"
        elif output.state is FieldState.MISSING:
            code = "M"
        elif output.state is FieldState.UNCERTAIN:
            if not output.evidence:
                code = "U[]"
            else:
                code = f"U[{output.evidence[0].text}]"
        else:
            code = "C[" + ";".join(span.text for span in output.evidence) + "]"
        if any(delimiter in code[2:-1] for delimiter in ("|", "]")):
            raise ValueError("evidence text collides with the native grammar")
        encoded.append(f"{_LABELS[expected]}:{code}")
    return "|".join(encoded)


def _make_example(
    split: str,
    world_index: int,
    variant: str,
    world: _World,
    rng: random.Random,
) -> StateSpanExample:
    transcript = _transcript(_variant_lines(split, world, variant, rng))
    request = NanoInput(
        item_id=f"{split}-{world_index:04d}-{variant}", transcript=transcript
    )
    fields = _extract_fields(request)
    expected_state = None if variant == "normal" else STATE_VARIANTS[variant]
    if expected_state is not None:
        observed = fields[FIELD_ORDER.index(world.target_field)].state
        if observed is not expected_state:
            raise AssertionError(
                f"generated {variant} state mismatch for {world.target_field.value}: {observed.value}"
            )
    return StateSpanExample(
        split=split,
        example_id=request.item_id,
        world_id=f"{split}-world-{world_index:04d}",
        variant=variant,
        target_field=world.target_field,
        target_state=expected_state,
        transcript=transcript,
        target=encode_state_span(fields),
    )


def generate_split(
    split: str,
    *,
    worlds: int | None = None,
    seed: int | None = None,
) -> tuple[StateSpanExample, ...]:
    """Generate a complete split; world counts must preserve five-field balance."""

    if split not in {"train", "dev"}:
        raise ValueError("split must be train or dev")
    default_worlds = TRAIN_WORLDS if split == "train" else DEV_WORLDS
    count = default_worlds if worlds is None else worlds
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("worlds must be a positive integer")
    if count % len(FIELD_ORDER):
        raise ValueError("worlds must be divisible by five")
    resolved_seed = (
        (TRAIN_SEED if split == "train" else DEV_SEED) if seed is None else seed
    )
    if isinstance(resolved_seed, bool) or not isinstance(resolved_seed, int):
        raise TypeError("seed must be an integer")
    assert_static_split_isolation()
    rng = random.Random(resolved_seed)
    examples: list[StateSpanExample] = []
    for index in range(count):
        world = _sample_world(split, index, rng)
        examples.extend(
            _make_example(split, index, variant, world, rng) for variant in VARIANTS
        )
    return tuple(examples)


def _records_bytes(examples: Sequence[StateSpanExample]) -> bytes:
    return b"".join(canonical_json_bytes(example.to_dict()) for example in examples)


def _quota(examples: Sequence[StateSpanExample]) -> dict[str, int]:
    counts = Counter(
        f"{example.target_state.value}:{example.target_field.value}"
        for example in examples
        if example.target_state is not None
    )
    return dict(sorted(counts.items()))


def build_manifest(
    train: Sequence[StateSpanExample],
    dev: Sequence[StateSpanExample],
    *,
    generator_sha256: str,
    tokenizer_sha256: str,
    base_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Create the frozen pre-training identity for a generated dataset pair."""

    for label, digest in (
        ("generator", generator_sha256),
        ("tokenizer", tokenizer_sha256),
        ("base checkpoint", base_checkpoint_sha256),
    ):
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"{label} digest must be lowercase SHA-256")
    train_bytes = _records_bytes(train)
    dev_bytes = _records_bytes(dev)
    train_worlds = {example.world_id for example in train}
    dev_worlds = {example.world_id for example in dev}
    if train_worlds & dev_worlds:
        raise ValueError("train/dev world overlap")
    train_hashes = {_sha256(example.transcript.encode("utf-8")) for example in train}
    dev_hashes = {_sha256(example.transcript.encode("utf-8")) for example in dev}
    if train_hashes & dev_hashes:
        raise ValueError("train/dev transcript overlap")
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "target_grammar": TARGET_GRAMMAR_VERSION,
        "generator_sha256": generator_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "train": {
            "seed": TRAIN_SEED,
            "records": len(train),
            "worlds": len(train_worlds),
            "sha256": _sha256(train_bytes),
            "transcript_multiset_sha256": _sha256(
                "\n".join(sorted(train_hashes)).encode("utf-8")
            ),
            "state_field_quota": _quota(train),
        },
        "dev": {
            "seed": DEV_SEED,
            "records": len(dev),
            "worlds": len(dev_worlds),
            "sha256": _sha256(dev_bytes),
            "transcript_multiset_sha256": _sha256(
                "\n".join(sorted(dev_hashes)).encode("utf-8")
            ),
            "state_field_quota": _quota(dev),
        },
        "isolation": {
            "worlds_disjoint": True,
            "transcripts_disjoint": True,
            "open_value_lexicons_disjoint": True,
            "question_templates_disjoint": True,
            "answer_templates_disjoint": True,
            "denial_phrases_disjoint": True,
            "uncertainty_phrases_disjoint": True,
            "fresh_v0_read_by_generator": False,
        },
    }


def write_dataset(
    output_dir: Path,
    *,
    tokenizer_sha256: str,
    base_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Generate and create a no-clobber train/dev data family plus manifest."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    train = generate_split("train")
    dev = generate_split("dev")
    generator_sha256 = _sha256(Path(__file__).read_bytes())
    manifest = build_manifest(
        train,
        dev,
        generator_sha256=generator_sha256,
        tokenizer_sha256=tokenizer_sha256,
        base_checkpoint_sha256=base_checkpoint_sha256,
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "train.jsonl").write_bytes(_records_bytes(train))
    (output / "dev.jsonl").write_bytes(_records_bytes(dev))
    (output / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    return manifest


def load_records(
    path: Path, *, expected_sha256: str | None = None
) -> tuple[StateSpanExample, ...]:
    snapshot = Path(path).read_bytes()
    if expected_sha256 is not None and _sha256(snapshot) != expected_sha256:
        raise ValueError(f"dataset digest mismatch: {path.name}")
    records: list[StateSpanExample] = []
    for line_number, line in enumerate(snapshot.splitlines(), 1):
        try:
            value = json.loads(line)
            records.append(StateSpanExample.from_dict(value))
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid record at {path.name}:{line_number}") from exc
    if not records:
        raise ValueError(f"dataset is empty: {path.name}")
    return tuple(records)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Nano native state/span SFT data"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--base-checkpoint-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = write_dataset(
        args.output_dir,
        tokenizer_sha256=args.tokenizer_sha256,
        base_checkpoint_sha256=args.base_checkpoint_sha256,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DATASET_SCHEMA_VERSION",
    "DEV_SEED",
    "DEV_WORLDS",
    "FORBIDDEN_HISTORICAL_SENTINELS",
    "MANIFEST_SCHEMA_VERSION",
    "TARGET_GRAMMAR_VERSION",
    "TRAIN_SEED",
    "TRAIN_WORLDS",
    "StateSpanExample",
    "assert_static_split_isolation",
    "build_manifest",
    "canonical_json_bytes",
    "encode_state_span",
    "generate_split",
    "load_records",
    "supported_value_sets",
    "write_dataset",
]
