"""Deterministic H5 replay-mixture data for Nano's scribe training.

H5 is one fixed, step-matched intervention: 1,400 regenerated legacy worlds
and 1,400 H4 surface-transfer worlds, with all four paired variants retained.
The H4 200-world calibration partition is reused byte-for-byte.  Legacy records
are regenerated from the pinned source and seed; this module never reads a
legacy record file, development data, a benchmark, or a sealed partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from nano_ai.adapters.state_span import parse_state_span_summary
from nano_ai.contract import FIELD_ORDER, FieldName, FieldState
from nano_ai.training import state_span_data, surface_transfer_data
from nano_ai.training.state_span_data import (
    STATE_VARIANTS,
    VARIANTS,
    StateSpanExample,
    canonical_json_bytes,
)
from nano_ai.training.train_evidence_query_h4 import (
    H4TrainingBundle,
    load_h4_training_bundle,
)
from nano_ai.training.train_state_span import TrainingInputError

MANIFEST_SCHEMA_VERSION = "nano.replay-mixture-manifest.v1"
GENERATOR_VERSION = "nano.replay-mixture-dataset.v1"
TRAINING_RECIPE_VERSION = "nano-evidence-query-replay-mixture-v1"
SELECTION_POLICY_VERSION = "nano.replay-mixture-selection.v1"
NORMALIZATION_VERSION = "nano.replay-mixture-component-skeleton.v1"

LEGACY_SOURCE = "legacy"
SURFACE_SOURCE = "surface"
SOURCES = (LEGACY_SOURCE, SURFACE_SOURCE)
SOURCE_WORLDS = 1_400
SOURCE_POOL_WORLDS = 2_800
LEGACY_AUTHORITY_WORLDS = 3_000
TOTAL_FIT_WORLDS = 2_800
VARIANTS_PER_WORLD = 4
SOURCE_RECORDS = SOURCE_WORLDS * VARIANTS_PER_WORLD
FIT_RECORDS = TOTAL_FIT_WORLDS * VARIANTS_PER_WORLD
CALIBRATION_WORLDS = 200
CALIBRATION_RECORDS = CALIBRATION_WORLDS * VARIANTS_PER_WORLD
TRAINING_SEEDS = (20260805, 20260806)
BATCH_SIZE = 32
EPOCHS = 3
STEPS_PER_EPOCH = 350
STEPS_PER_SEED = 1_050
CALIBRATION_PARTITION_NAME = "calibration"

# Calibration open-value occurrence in legacy replay is an expected limitation,
# not an eligibility rule.  These frozen observations make the limitation
# visible without changing the founder-selected causal 50:50 intervention.
EXPECTED_LEGACY_LITERAL_SUBSTRING_DISJOINT_WORLDS = 1_053
EXPECTED_LEGACY_LITERAL_SUBSTRING_DISJOINT_BY_FIELD: Mapping[str, int] = {
    FieldName.CHIEF_COMPLAINT.value: 263,
    FieldName.DURATION.value: 178,
    FieldName.SEVERITY.value: 199,
    FieldName.MEDICATION.value: 206,
    FieldName.ALLERGY.value: 207,
}
EXPECTED_LEGACY_BALANCED_LITERAL_SUBSTRING_DISJOINT_LIMIT = 890
EXPECTED_SELECTED_LEGACY_LITERAL_SUBSTRING_DISJOINT_WORLDS = 539
EXPECTED_SELECTED_LEGACY_LITERAL_SUBSTRING_DISJOINT_BY_FIELD: Mapping[str, int] = {
    FieldName.CHIEF_COMPLAINT.value: 139,
    FieldName.DURATION.value: 94,
    FieldName.SEVERITY.value: 110,
    FieldName.MEDICATION.value: 90,
    FieldName.ALLERGY.value: 106,
}

LEGACY_GENERATOR_SHA256 = (
    "2d3fd33d694893aea5aa80b514e34256e9f6046f71b13f6410dbabec1278d707"
)
LEGACY_MANIFEST_SHA256 = (
    "47ee157ac037c0771100b8546c90da91dbd2006198700bb642f1561d2124c1a3"
)
LEGACY_TRAIN_RECORDS_SHA256 = (
    "19331b7e1d8d37c2b19f4bd5288b9229aa6a3be3a0288111da3204cb5e68182b"
)
LEGACY_TRAIN_TRANSCRIPT_IDENTITY_SHA256 = (
    "54d130a0193c36b7113d369f62d5275cd3ee13dc827fb31abdc4f90794e0131f"
)
LEGACY_TOKENIZER_SHA256 = (
    "bae49648bfcc4904c50e2f006ee184bd26e74454ee170663e30a8e71640ce3c9"
)
LEGACY_BASE_CHECKPOINT_SHA256 = (
    "0e4f348eea00c660236cfd9e5bc2d9a71274adfc4d738db6f664664c9a06725b"
)
H4_CALIBRATION_RECORDS_SHA256 = (
    "ce0562ccb44ee83963eace0d873773addaee8e49f29499a6b720b16335930e70"
)
H4_GENERATOR_SHA256 = "29ef84d44ecbf1af5d3f3e08960abe21173e4f5a756dc2c6a7082edaae408508"
H4_MANIFEST_SHA256 = "444c5d6dcb613959a28ee3ce2e1dd7e25e2fabc2603ba4192d445ed3bedb406f"
H4_FIT_RECORDS_SHA256 = (
    "8e019b5ee827facd179bcda521838849e2053b02fed0e392e322c5691fda3ae1"
)
EXPECTED_SELECTED_SOURCE_WORLD_ID_SHA256: Mapping[str, str] = {
    LEGACY_SOURCE: ("29d5171c7d00bbd4b456094f4cd8a8d3585996276f8cfcc7417c6b26525dab1b"),
    SURFACE_SOURCE: (
        "93f702e2cc4cfb223fb4a2a439b209a387b5543baba149dabec587778c1805fc"
    ),
}

EXPECTED_STATE_CLASS_COUNTS: Mapping[str, int] = {
    FieldState.SUPPORTED.value: 42_980,
    FieldState.ABSENT.value: 4_620,
    FieldState.MISSING.value: 2_800,
    FieldState.UNCERTAIN.value: 2_800,
    FieldState.CONFLICTING.value: 2_800,
}
EXPECTED_STATE_FIELD_QUOTA: Mapping[str, int] = {
    **{
        f"{field.value}:{FieldState.SUPPORTED.value}": 9_520
        for field in (
            FieldName.CHIEF_COMPLAINT,
            FieldName.DURATION,
            FieldName.SEVERITY,
        )
    },
    f"{FieldName.MEDICATION.value}:{FieldState.SUPPORTED.value}": 7_140,
    f"{FieldName.MEDICATION.value}:{FieldState.ABSENT.value}": 2_380,
    f"{FieldName.ALLERGY.value}:{FieldState.SUPPORTED.value}": 7_280,
    f"{FieldName.ALLERGY.value}:{FieldState.ABSENT.value}": 2_240,
    **{
        f"{field.value}:{state.value}": 560
        for field in FIELD_ORDER
        for state in (
            FieldState.MISSING,
            FieldState.UNCERTAIN,
            FieldState.CONFLICTING,
        )
    },
}
EXPECTED_SOURCE_STATE_CLASS_COUNTS: Mapping[str, int] = {
    state: count // 2 for state, count in EXPECTED_STATE_CLASS_COUNTS.items()
}
EXPECTED_SOURCE_STATE_FIELD_QUOTA: Mapping[str, int] = {
    field_state: count // 2 for field_state, count in EXPECTED_STATE_FIELD_QUOTA.items()
}

# Each source contributes the same fixed, proportional base-state strata.
# This retains exact field balance and the original supported/absent class
# exposure while selection within every stratum is solely SHA-256 ordered.
_STRATUM_LOW_QUOTA = 70
_STRATUM_HIGH_QUOTA = 210
_LEGACY_WORLD_RE = re.compile(r"train-world-(\d{4})\Z")
_SURFACE_WORLD_RE = re.compile(r"train-world-fit-(\d{4})\Z")
_CALIBRATION_WORLD_RE = re.compile(r"train-world-calibration-(\d{4})\Z")
_REPLAY_WORLD_RE = re.compile(r"train-world-replay-(legacy|surface)-(\d{4})\Z")
_REPLAY_RECORD_RE = re.compile(
    r"train-replay-(legacy|surface)-(\d{4})-"
    r"(normal|missing|uncertain|conflicting)\Z"
)


@dataclass(frozen=True, slots=True)
class _WorldFamily:
    source: str
    source_world_id: str
    target_field: FieldName
    stratum: str
    records: tuple[StateSpanExample, ...]
    exact_transcripts: frozenset[str]
    exact_transcript_templates: frozenset[str]
    exact_component_line_templates: frozenset[str]
    normalized_component_skeletons: frozenset[str]
    selection_score: str


@dataclass(frozen=True, slots=True)
class ReplayMixture:
    fit: tuple[StateSpanExample, ...]
    calibration: tuple[StateSpanExample, ...]
    manifest: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ReplayMixtureBundle:
    manifest: Mapping[str, Any]
    manifest_sha256: str
    fit: tuple[StateSpanExample, ...]
    calibration: tuple[StateSpanExample, ...]
    input_sha256: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class ReplayValueOverlapAudit:
    """Non-blocking calibration open-value occurrence audit for H3 replay."""

    candidate_worlds: int
    calibration_open_values: int
    literal_substring_disjoint_worlds: int
    worlds_with_literal_substring_occurrence: int
    literal_substring_disjoint_by_target_field: Mapping[str, int]
    balanced_literal_substring_disjoint_limit: int
    literal_substring_occurrence_world_counts: Mapping[str, int]
    selected_worlds: int
    selected_literal_substring_disjoint_worlds: int
    selected_worlds_with_literal_substring_occurrence: int
    selected_literal_substring_disjoint_by_target_field: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": "expected_recorded_nonblocking",
            "method": "nfkc_casefold_literal_substring_in_any_family_transcript",
            "normalization": "unicode_nfkc_then_casefold",
            "match_semantics": "literal_substring",
            "substring_metric_is_conservative": True,
            "substring_can_match_within_longer_values": True,
            "exact_value_identity_not_claimed": True,
            "candidate_worlds": self.candidate_worlds,
            "calibration_open_values": self.calibration_open_values,
            "literal_substring_disjoint_worlds": (
                self.literal_substring_disjoint_worlds
            ),
            "worlds_with_literal_substring_occurrence": (
                self.worlds_with_literal_substring_occurrence
            ),
            "literal_substring_disjoint_by_target_field": dict(
                self.literal_substring_disjoint_by_target_field
            ),
            "balanced_literal_substring_disjoint_limit": (
                self.balanced_literal_substring_disjoint_limit
            ),
            "selected_worlds": self.selected_worlds,
            "selected_literal_substring_disjoint_worlds": (
                self.selected_literal_substring_disjoint_worlds
            ),
            "selected_worlds_with_literal_substring_occurrence": (
                self.selected_worlds_with_literal_substring_occurrence
            ),
            "selected_literal_substring_disjoint_by_target_field": dict(
                self.selected_literal_substring_disjoint_by_target_field
            ),
            "literal_substring_occurrence_world_counts": dict(
                self.literal_substring_occurrence_world_counts
            ),
        }


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_digest(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TrainingInputError(f"H5 {label} is not lowercase SHA-256")
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _parse_json(payload: bytes, *, label: str) -> Any:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise TrainingInputError(f"invalid H5 JSON in {label}") from exc


def _read_file(path: Path, *, label: str) -> bytes:
    try:
        return Path(path).read_bytes()
    except OSError as exc:
        raise TrainingInputError(f"H5 {label} is unavailable") from exc


def _records_bytes(examples: Sequence[StateSpanExample]) -> bytes:
    return b"".join(canonical_json_bytes(example.to_dict()) for example in examples)


def _true_transcript_multiset_sha256(
    examples: Sequence[StateSpanExample],
) -> str:
    digests = sorted(
        _sha256(example.transcript.encode("utf-8")) for example in examples
    )
    return _sha256(canonical_json_bytes(digests))


def _string_multiset_sha256(values: Iterable[str]) -> str:
    return _sha256(canonical_json_bytes(sorted(values)))


def _legacy_manifest_transcript_identity(
    examples: Sequence[StateSpanExample],
) -> str:
    """Reproduce the historical manifest's set-based transcript identity."""

    unique = {_sha256(example.transcript.encode("utf-8")) for example in examples}
    return _sha256("\n".join(sorted(unique)).encode("utf-8"))


def _legacy_mutation_quota(examples: Sequence[StateSpanExample]) -> dict[str, int]:
    counts = Counter(
        f"{example.target_state.value}:{example.target_field.value}"
        for example in examples
        if example.target_state is not None
    )
    return dict(sorted(counts.items()))


def _require_complete_worlds(
    examples: Sequence[StateSpanExample],
    *,
    source: str,
    expected_worlds: int,
) -> Mapping[str, tuple[StateSpanExample, ...]]:
    grouped: dict[str, list[StateSpanExample]] = defaultdict(list)
    seen_ids: set[str] = set()
    for example in examples:
        if example.split != "train" or example.example_id in seen_ids:
            raise TrainingInputError(f"H5 {source} record identity drifted")
        seen_ids.add(example.example_id)
        grouped[example.world_id].append(example)
    if len(grouped) != expected_worlds:
        raise TrainingInputError(f"H5 {source} world count drifted")
    ordered: dict[str, tuple[StateSpanExample, ...]] = {}
    for world_id, family in grouped.items():
        by_variant = {record.variant: record for record in family}
        if (
            len(family) != VARIANTS_PER_WORLD
            or set(by_variant) != set(VARIANTS)
            or len({record.target_field for record in family}) != 1
        ):
            raise TrainingInputError(f"H5 {source} world {world_id} is incomplete")
        for variant, record in by_variant.items():
            expected_state = None if variant == "normal" else STATE_VARIANTS[variant]
            if record.target_state != expected_state:
                raise TrainingInputError(
                    f"H5 {source} world {world_id} mutation state drifted"
                )
        ordered[world_id] = tuple(by_variant[variant] for variant in VARIANTS)
    return ordered


def _validate_legacy_manifest(value: object) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "target_grammar",
        "generator_sha256",
        "tokenizer_sha256",
        "base_checkpoint_sha256",
        "train",
        "dev",
        "isolation",
    }:
        raise TrainingInputError("H5 legacy manifest shape drifted")
    if (
        value["schema_version"] != state_span_data.MANIFEST_SCHEMA_VERSION
        or value["target_grammar"] != state_span_data.TARGET_GRAMMAR_VERSION
        or value["generator_sha256"] != LEGACY_GENERATOR_SHA256
        or value["tokenizer_sha256"] != LEGACY_TOKENIZER_SHA256
        or value["base_checkpoint_sha256"] != LEGACY_BASE_CHECKPOINT_SHA256
    ):
        raise TrainingInputError("H5 legacy manifest identity drifted")
    train = value["train"]
    if not isinstance(train, dict) or train != {
        "seed": state_span_data.TRAIN_SEED,
        "records": 12_000,
        "worlds": 3_000,
        "sha256": LEGACY_TRAIN_RECORDS_SHA256,
        "transcript_multiset_sha256": LEGACY_TRAIN_TRANSCRIPT_IDENTITY_SHA256,
        "state_field_quota": {
            f"{state.value}:{field.value}": 600
            for state in (
                FieldState.CONFLICTING,
                FieldState.MISSING,
                FieldState.UNCERTAIN,
            )
            for field in FIELD_ORDER
        },
    }:
        raise TrainingInputError("H5 legacy training metadata drifted")
    isolation = value["isolation"]
    if (
        not isinstance(isolation, dict)
        or isolation.get("fresh_v0_read_by_generator") is not False
        or any(
            isolation.get(key) is not True
            for key in (
                "worlds_disjoint",
                "transcripts_disjoint",
                "open_value_lexicons_disjoint",
                "question_templates_disjoint",
                "answer_templates_disjoint",
                "denial_phrases_disjoint",
                "uncertainty_phrases_disjoint",
            )
        )
    ):
        raise TrainingInputError("H5 legacy isolation metadata drifted")
    return value


def load_legacy_reproduction(
    manifest_path: Path,
) -> tuple[tuple[StateSpanExample, ...], Mapping[str, Any], str]:
    """Authenticate metadata and reproduce legacy training worlds in memory."""

    path = Path(manifest_path)
    if path.name != "manifest.json":
        raise TrainingInputError("H5 legacy authority must be manifest.json")
    manifest_snapshot = _read_file(path, label="legacy manifest")
    manifest_sha256 = _sha256(manifest_snapshot)
    if manifest_sha256 != LEGACY_MANIFEST_SHA256:
        raise TrainingInputError("H5 legacy manifest digest drifted")
    manifest = _validate_legacy_manifest(
        _parse_json(manifest_snapshot, label="legacy manifest")
    )
    generator_path = Path(state_span_data.__file__).resolve()
    generator_before = _sha256(_read_file(generator_path, label="legacy generator"))
    if generator_before != LEGACY_GENERATOR_SHA256:
        raise TrainingInputError("H5 frozen legacy generator source drifted")

    records = state_span_data.generate_split(
        "train",
        worlds=state_span_data.TRAIN_WORLDS,
        seed=state_span_data.TRAIN_SEED,
    )
    _require_complete_worlds(records, source=LEGACY_SOURCE, expected_worlds=3_000)
    if (
        _sha256(_records_bytes(records)) != LEGACY_TRAIN_RECORDS_SHA256
        or _legacy_manifest_transcript_identity(records)
        != LEGACY_TRAIN_TRANSCRIPT_IDENTITY_SHA256
        or _legacy_mutation_quota(records) != manifest["train"]["state_field_quota"]
    ):
        raise TrainingInputError("H5 regenerated legacy records failed authentication")
    if (
        _sha256(_read_file(generator_path, label="legacy generator"))
        != generator_before
        or _sha256(_read_file(path, label="legacy manifest")) != manifest_sha256
    ):
        raise TrainingInputError("H5 legacy authority changed during reproduction")
    return records, manifest, manifest_sha256


def _normalization_values() -> tuple[str, ...]:
    values: set[str] = set()
    for field_values in state_span_data.supported_value_sets("train").values():
        values.update(field_values)
    lexicons = surface_transfer_data.partition_lexicons()
    for partition in (
        surface_transfer_data.FIT_PARTITION,
        surface_transfer_data.CALIBRATION_PARTITION,
    ):
        for field_values in lexicons[partition].values():
            values.update(field_values)
    return tuple(sorted(values, key=lambda item: (-len(item), item.casefold())))


def normalize_transcript(transcript: str, *, values: Sequence[str]) -> str:
    """Normalize one transcript component into its value-free skeleton."""

    normalized = unicodedata.normalize("NFKC", transcript).casefold()
    for value in values:
        normalized_value = unicodedata.normalize("NFKC", value).casefold()
        normalized = normalized.replace(normalized_value, "{value}")
    return re.sub(r"\s+", " ", normalized).strip()


def _exact_template(component: str, *, values: Sequence[str]) -> str:
    template = component
    for value in values:
        template = template.replace(value, "{value}")
    return template


def _source_index_from_world_id(*, source: str, world_id: str) -> int:
    patterns = {
        LEGACY_SOURCE: _LEGACY_WORLD_RE,
        SURFACE_SOURCE: _SURFACE_WORLD_RE,
        CALIBRATION_PARTITION_NAME: _CALIBRATION_WORLD_RE,
    }
    pattern = patterns.get(source)
    match = None if pattern is None else pattern.fullmatch(world_id)
    if match is None:
        raise TrainingInputError(f"H5 {source} source world namespace drifted")
    return int(match.group(1))


def _expected_record_id(*, source: str, index: int, variant: str) -> str:
    if source == LEGACY_SOURCE:
        return f"train-{index:04d}-{variant}"
    if source == SURFACE_SOURCE:
        return f"train-fit-{index:04d}-{variant}"
    if source == CALIBRATION_PARTITION_NAME:
        return f"train-calibration-{index:04d}-{variant}"
    raise TrainingInputError(f"H5 unknown source namespace: {source}")


def _selection_stratum(
    normal: StateSpanExample, *, source: str, source_index: int
) -> str:
    proposals = parse_state_span_summary(normal.target, normal.transcript)
    states = {proposal.field: proposal.state for proposal in proposals}
    expected_field = FIELD_ORDER[source_index % len(FIELD_ORDER)]
    medication_absent = source_index % 4 == 0
    allergy_absent = source_index % 5 == 0
    expected_medication = (
        FieldState.ABSENT if medication_absent else FieldState.SUPPORTED
    )
    expected_allergy = FieldState.ABSENT if allergy_absent else FieldState.SUPPORTED
    if (
        normal.target_field is not expected_field
        or states.get(FieldName.MEDICATION) is not expected_medication
        or states.get(FieldName.ALLERGY) is not expected_allergy
    ):
        raise TrainingInputError(f"H5 {source} frozen index stratum drifted")
    return (
        f"{normal.target_field.value}"
        f"|medication_absent={str(medication_absent).lower()}"
        f"|allergy_absent={str(allergy_absent).lower()}"
    )


def _stratum_quota(stratum: str) -> int:
    _field, medication, _allergy = stratum.split("|")
    return (
        _STRATUM_LOW_QUOTA
        if medication == "medication_absent=true"
        else _STRATUM_HIGH_QUOTA
    )


def _selection_score(
    *, source: str, source_world_id: str, records: Sequence[StateSpanExample]
) -> str:
    identity = {
        "policy": SELECTION_POLICY_VERSION,
        "source": source,
        "source_world_id": source_world_id,
        "records_sha256": _sha256(_records_bytes(records)),
    }
    return _sha256(canonical_json_bytes(identity))


def _world_families(
    examples: Sequence[StateSpanExample],
    *,
    source: str,
    expected_worlds: int,
    values: Sequence[str],
) -> tuple[_WorldFamily, ...]:
    grouped = _require_complete_worlds(
        examples, source=source, expected_worlds=expected_worlds
    )
    families: list[_WorldFamily] = []
    for world_id, records in grouped.items():
        source_index = _source_index_from_world_id(source=source, world_id=world_id)
        if any(
            record.example_id
            != _expected_record_id(
                source=source, index=source_index, variant=record.variant
            )
            for record in records
        ):
            raise TrainingInputError(f"H5 {source} record namespace drifted")
        normal = records[VARIANTS.index("normal")]
        components = tuple(
            line for record in records for line in record.transcript.splitlines()
        )
        families.append(
            _WorldFamily(
                source=source,
                source_world_id=world_id,
                target_field=normal.target_field,
                stratum=_selection_stratum(
                    normal, source=source, source_index=source_index
                ),
                records=records,
                exact_transcripts=frozenset(record.transcript for record in records),
                exact_transcript_templates=frozenset(
                    _exact_template(record.transcript, values=values)
                    for record in records
                ),
                exact_component_line_templates=frozenset(
                    _exact_template(component, values=values)
                    for component in components
                ),
                normalized_component_skeletons=frozenset(
                    normalize_transcript(component, values=values)
                    for component in components
                ),
                selection_score=_selection_score(
                    source=source, source_world_id=world_id, records=records
                ),
            )
        )
    observed_indices = {
        _source_index_from_world_id(source=source, world_id=family.source_world_id)
        for family in families
    }
    if observed_indices != set(range(expected_worlds)):
        raise TrainingInputError(f"H5 {source} source index range drifted")
    return tuple(families)


def _union_exact_transcripts(
    families: Sequence[_WorldFamily],
) -> frozenset[str]:
    return frozenset(
        transcript for family in families for transcript in family.exact_transcripts
    )


def _union_exact_transcript_templates(
    families: Sequence[_WorldFamily],
) -> frozenset[str]:
    return frozenset(
        template
        for family in families
        for template in family.exact_transcript_templates
    )


def _union_exact_component_line_templates(
    families: Sequence[_WorldFamily],
) -> frozenset[str]:
    return frozenset(
        template
        for family in families
        for template in family.exact_component_line_templates
    )


def _union_component_skeletons(
    families: Sequence[_WorldFamily],
) -> frozenset[str]:
    return frozenset(
        skeleton
        for family in families
        for skeleton in family.normalized_component_skeletons
    )


def _calibration_open_values() -> tuple[str, ...]:
    lexicons = surface_transfer_data.partition_lexicons()[
        surface_transfer_data.CALIBRATION_PARTITION
    ]
    values = {
        value
        for field in FIELD_ORDER
        if field is not FieldName.SEVERITY
        for value in lexicons[field]
    }
    return tuple(sorted(values, key=lambda value: (value.casefold(), value)))


def _family_value_hits(
    family: _WorldFamily, *, open_values: Sequence[str]
) -> tuple[str, ...]:
    transcript = unicodedata.normalize(
        "NFKC", "\n".join(record.transcript for record in family.records)
    ).casefold()
    return tuple(
        value
        for value in open_values
        if unicodedata.normalize("NFKC", value).casefold() in transcript
    )


def _value_overlap_audit(
    candidates: Sequence[_WorldFamily],
    *,
    selected: Sequence[_WorldFamily] = (),
) -> ReplayValueOverlapAudit:
    open_values = _calibration_open_values()
    hits_by_world = {
        family.source_world_id: _family_value_hits(family, open_values=open_values)
        for family in candidates
    }
    disjoint = tuple(
        family for family in candidates if not hits_by_world[family.source_world_id]
    )
    disjoint_ids = {family.source_world_id for family in disjoint}
    by_field = Counter(family.target_field.value for family in disjoint)
    value_counts: Counter[str] = Counter()
    for hits in hits_by_world.values():
        value_counts.update(hits)
    selected_disjoint = tuple(
        family for family in selected if family.source_world_id in disjoint_ids
    )
    selected_by_field = Counter(
        family.target_field.value for family in selected_disjoint
    )
    audit = ReplayValueOverlapAudit(
        candidate_worlds=len(candidates),
        calibration_open_values=len(open_values),
        literal_substring_disjoint_worlds=len(disjoint),
        worlds_with_literal_substring_occurrence=len(candidates) - len(disjoint),
        literal_substring_disjoint_by_target_field={
            field.value: by_field[field.value] for field in FIELD_ORDER
        },
        balanced_literal_substring_disjoint_limit=(
            min(by_field[field.value] for field in FIELD_ORDER) * len(FIELD_ORDER)
        ),
        literal_substring_occurrence_world_counts=dict(sorted(value_counts.items())),
        selected_worlds=len(selected),
        selected_literal_substring_disjoint_worlds=len(selected_disjoint),
        selected_worlds_with_literal_substring_occurrence=(
            len(selected) - len(selected_disjoint)
        ),
        selected_literal_substring_disjoint_by_target_field={
            field.value: selected_by_field[field.value] for field in FIELD_ORDER
        },
    )
    if len(candidates) == SOURCE_POOL_WORLDS and (
        audit.literal_substring_disjoint_worlds
        != EXPECTED_LEGACY_LITERAL_SUBSTRING_DISJOINT_WORLDS
        or dict(audit.literal_substring_disjoint_by_target_field)
        != dict(EXPECTED_LEGACY_LITERAL_SUBSTRING_DISJOINT_BY_FIELD)
        or audit.balanced_literal_substring_disjoint_limit
        != EXPECTED_LEGACY_BALANCED_LITERAL_SUBSTRING_DISJOINT_LIMIT
    ):
        raise TrainingInputError("H5 legacy calibration-value audit drifted")
    if selected and (
        audit.selected_literal_substring_disjoint_worlds
        != EXPECTED_SELECTED_LEGACY_LITERAL_SUBSTRING_DISJOINT_WORLDS
        or dict(audit.selected_literal_substring_disjoint_by_target_field)
        != dict(EXPECTED_SELECTED_LEGACY_LITERAL_SUBSTRING_DISJOINT_BY_FIELD)
    ):
        raise TrainingInputError("H5 selected legacy calibration-value audit drifted")
    return audit


def audit_calibration_value_overlap(
    legacy: Sequence[StateSpanExample],
) -> ReplayValueOverlapAudit:
    """Measure, but do not exclude, legacy worlds containing calibration values."""

    values = _normalization_values()
    if len(legacy) == LEGACY_AUTHORITY_WORLDS * VARIANTS_PER_WORLD:
        families = _world_families(
            legacy,
            source=LEGACY_SOURCE,
            expected_worlds=LEGACY_AUTHORITY_WORLDS,
            values=values,
        )
        candidates = tuple(
            family for family in families if _source_index(family) < SOURCE_POOL_WORLDS
        )
    elif len(legacy) == SOURCE_POOL_WORLDS * VARIANTS_PER_WORLD:
        candidates = _world_families(
            legacy,
            source=LEGACY_SOURCE,
            expected_worlds=SOURCE_POOL_WORLDS,
            values=values,
        )
    else:
        raise TrainingInputError("H5 legacy audit record count drifted")
    return _value_overlap_audit(candidates)


def _select_source(
    candidates: Sequence[_WorldFamily], *, source: str
) -> tuple[tuple[_WorldFamily, ...], Mapping[str, Any]]:
    by_stratum: dict[str, list[_WorldFamily]] = defaultdict(list)
    for family in candidates:
        if family.source != source:
            raise TrainingInputError("H5 source selection received a foreign world")
        by_stratum[family.stratum].append(family)

    selected: list[_WorldFamily] = []
    strata: dict[str, Any] = {}
    for field in FIELD_ORDER:
        field_strata = sorted(
            stratum for stratum in by_stratum if stratum.startswith(f"{field.value}|")
        )
        if len(field_strata) != 2:
            raise TrainingInputError(
                f"H5 {source} eligibility lacks frozen strata for {field.value}"
            )
        field_selected = 0
        for stratum in field_strata:
            quota = _stratum_quota(stratum)
            ranked = sorted(
                by_stratum[stratum],
                key=lambda family: (family.selection_score, family.source_world_id),
            )
            expected_pool = quota * 2
            if len(ranked) != expected_pool:
                raise TrainingInputError(
                    f"H5 {source} {stratum} has {len(ranked)} worlds; "
                    f"requires frozen pool {expected_pool}"
                )
            chosen = ranked[:quota]
            selected.extend(chosen)
            field_selected += len(chosen)
            strata[stratum] = {
                "candidate_worlds": len(ranked),
                "selected_worlds": len(chosen),
                "selection_fraction": "1/2",
                "first_score": chosen[0].selection_score,
                "final_score": chosen[-1].selection_score,
                "selected_source_world_id_multiset_sha256": _string_multiset_sha256(
                    family.source_world_id for family in chosen
                ),
            }
        if field_selected != SOURCE_WORLDS // len(FIELD_ORDER):
            raise RuntimeError(f"H5 {source} field quota drifted")
    if (
        len(selected) != SOURCE_WORLDS
        or len({family.source_world_id for family in selected}) != SOURCE_WORLDS
    ):
        raise RuntimeError(f"H5 {source} selection cardinality drifted")
    return tuple(selected), dict(sorted(strata.items()))


def _source_index(family: _WorldFamily) -> int:
    return _source_index_from_world_id(
        source=family.source, world_id=family.source_world_id
    )


def _namespace_family(family: _WorldFamily) -> tuple[StateSpanExample, ...]:
    source_index = _source_index(family)
    world_id = f"train-world-replay-{family.source}-{source_index:04d}"
    return tuple(
        replace(
            record,
            example_id=(
                f"train-replay-{family.source}-{source_index:04d}-{record.variant}"
            ),
            world_id=world_id,
        )
        for record in family.records
    )


def _state_counts(
    examples: Sequence[StateSpanExample],
) -> tuple[dict[str, int], dict[str, int]]:
    state: Counter[str] = Counter()
    field_state: Counter[str] = Counter()
    for example in examples:
        proposals = parse_state_span_summary(example.target, example.transcript)
        for proposal in proposals:
            state[proposal.state.value] += 1
            field_state[f"{proposal.field.value}:{proposal.state.value}"] += 1
    return dict(sorted(state.items())), dict(sorted(field_state.items()))


def _load_replay_partition(
    payload: bytes, *, filename: str
) -> tuple[StateSpanExample, ...]:
    records: list[StateSpanExample] = []
    for line_number, line in enumerate(payload.splitlines(), 1):
        if not line:
            raise TrainingInputError(f"blank H5 record at {filename}:{line_number}")
        value = _parse_json(line, label=f"{filename}:{line_number}")
        try:
            records.append(StateSpanExample.from_dict(value))
        except (TypeError, ValueError) as exc:
            raise TrainingInputError(
                f"invalid H5 record at {filename}:{line_number}"
            ) from exc
    if not records or _records_bytes(records) != payload:
        raise TrainingInputError(f"H5 {filename} is not canonical JSONL")
    return tuple(records)


def _output_source_world_id(*, source: str, source_index: int) -> str:
    if source == LEGACY_SOURCE:
        return f"train-world-{source_index:04d}"
    if source == SURFACE_SOURCE:
        return f"train-world-fit-{source_index:04d}"
    raise TrainingInputError(f"H5 unknown replay source: {source}")


def _replay_output_order_key(source: str, source_index: int) -> tuple[str, str]:
    source_world_id = _output_source_world_id(source=source, source_index=source_index)
    score = _sha256(
        canonical_json_bytes(
            {
                "policy": SELECTION_POLICY_VERSION,
                "role": "fit-output-order",
                "source": source,
                "source_world_id": source_world_id,
            }
        )
    )
    return score, source_world_id


def _validate_replay_fit(
    records: Sequence[StateSpanExample],
) -> Mapping[str, tuple[StateSpanExample, ...]]:
    if len(records) != FIT_RECORDS:
        raise TrainingInputError("H5 fit record count drifted")
    grouped: dict[str, list[StateSpanExample]] = defaultdict(list)
    identities: dict[str, tuple[str, int]] = {}
    record_ids: set[str] = set()
    for record in records:
        world_match = _REPLAY_WORLD_RE.fullmatch(record.world_id)
        record_match = _REPLAY_RECORD_RE.fullmatch(record.example_id)
        if (
            record.split != "train"
            or world_match is None
            or record_match is None
            or record.example_id in record_ids
        ):
            raise TrainingInputError("H5 fit replay namespace drifted")
        source = world_match.group(1)
        source_index = int(world_match.group(2))
        if (
            record_match.group(1) != source
            or int(record_match.group(2)) != source_index
            or record_match.group(3) != record.variant
            or source_index >= SOURCE_POOL_WORLDS
        ):
            raise TrainingInputError("H5 fit replay index boundary drifted")
        record_ids.add(record.example_id)
        grouped[record.world_id].append(record)
        identities[record.world_id] = source, source_index
    if len(grouped) != TOTAL_FIT_WORLDS:
        raise TrainingInputError("H5 fit replay world count drifted")

    selected_indices: dict[str, set[int]] = {source: set() for source in SOURCES}
    selected_strata: dict[str, Counter[str]] = {source: Counter() for source in SOURCES}
    source_records: dict[str, list[StateSpanExample]] = {
        source: [] for source in SOURCES
    }
    validated: dict[str, tuple[StateSpanExample, ...]] = {}
    for world_id, family in grouped.items():
        source, source_index = identities[world_id]
        by_variant = {record.variant: record for record in family}
        if (
            len(family) != VARIANTS_PER_WORLD
            or set(by_variant) != set(VARIANTS)
            or len({record.target_field for record in family}) != 1
        ):
            raise TrainingInputError(f"H5 fit world {world_id} is incomplete")
        ordered = tuple(by_variant[variant] for variant in VARIANTS)
        for variant, record in by_variant.items():
            expected_state = None if variant == "normal" else STATE_VARIANTS[variant]
            if record.target_state != expected_state:
                raise TrainingInputError(
                    f"H5 fit world {world_id} mutation state drifted"
                )
        normal = by_variant["normal"]
        stratum = _selection_stratum(normal, source=source, source_index=source_index)
        selected_indices[source].add(source_index)
        selected_strata[source][stratum] += 1
        source_records[source].extend(ordered)
        validated[world_id] = ordered

    for source in SOURCES:
        if len(selected_indices[source]) != SOURCE_WORLDS:
            raise TrainingInputError(f"H5 {source} replay world count drifted")
        expected_strata = {
            (
                f"{field.value}|medication_absent={str(medication_absent).lower()}"
                f"|allergy_absent={str(field is FIELD_ORDER[0]).lower()}"
            ): (_STRATUM_LOW_QUOTA if medication_absent else _STRATUM_HIGH_QUOTA)
            for field in FIELD_ORDER
            for medication_absent in (False, True)
        }
        if dict(selected_strata[source]) != expected_strata:
            raise TrainingInputError(f"H5 {source} replay strata drifted")
        state_counts, state_field_quota = _state_counts(source_records[source])
        if state_counts != dict(
            EXPECTED_SOURCE_STATE_CLASS_COUNTS
        ) or state_field_quota != dict(EXPECTED_SOURCE_STATE_FIELD_QUOTA):
            raise TrainingInputError(f"H5 {source} replay distribution drifted")
        selected_source_ids = (
            _output_source_world_id(source=source, source_index=index)
            for index in selected_indices[source]
        )
        if (
            _string_multiset_sha256(selected_source_ids)
            != (EXPECTED_SELECTED_SOURCE_WORLD_ID_SHA256[source])
        ):
            raise TrainingInputError(f"H5 {source} replay selection drifted")

    observed_chunks: list[tuple[str, int]] = []
    for offset in range(0, len(records), VARIANTS_PER_WORLD):
        chunk = records[offset : offset + VARIANTS_PER_WORLD]
        if (
            len({record.world_id for record in chunk}) != 1
            or tuple(record.variant for record in chunk) != VARIANTS
        ):
            raise TrainingInputError("H5 fit output family order drifted")
        observed_chunks.append(identities[chunk[0].world_id])
    expected_chunks = sorted(
        (
            (source, source_index)
            for source in SOURCES
            for source_index in selected_indices[source]
        ),
        key=lambda identity: _replay_output_order_key(*identity),
    )
    if observed_chunks != expected_chunks:
        raise TrainingInputError("H5 fit deterministic output order drifted")

    state_counts, state_field_quota = _state_counts(records)
    if state_counts != dict(EXPECTED_STATE_CLASS_COUNTS) or state_field_quota != dict(
        EXPECTED_STATE_FIELD_QUOTA
    ):
        raise TrainingInputError("H5 fit combined distribution drifted")
    return validated


def _pair_overlap(
    left: Sequence[_WorldFamily], right: Sequence[_WorldFamily]
) -> dict[str, Any]:
    intersections = {
        "world_ids": (
            {family.source_world_id for family in left}
            & {family.source_world_id for family in right}
        ),
        "record_ids": (
            {record.example_id for family in left for record in family.records}
            & {record.example_id for family in right for record in family.records}
        ),
        "exact_transcripts": (
            _union_exact_transcripts(left) & _union_exact_transcripts(right)
        ),
        "exact_transcript_templates": (
            _union_exact_transcript_templates(left)
            & _union_exact_transcript_templates(right)
        ),
        "exact_component_line_templates": (
            _union_exact_component_line_templates(left)
            & _union_exact_component_line_templates(right)
        ),
        "normalized_component_line_skeletons": (
            _union_component_skeletons(left) & _union_component_skeletons(right)
        ),
    }
    return {
        key: {
            "count": len(values),
            "intersection_sha256": _string_multiset_sha256(values),
        }
        for key, values in sorted(intersections.items())
    }


def _require_no_pair_overlap(pairwise: Mapping[str, Mapping[str, Any]]) -> None:
    for pair, dimensions in pairwise.items():
        for dimension, identity in dimensions.items():
            if not isinstance(identity, dict) or identity.get("count") != 0:
                raise TrainingInputError(f"H5 {pair} {dimension} overlap is forbidden")


def _record_partition_overlap(
    left: Sequence[StateSpanExample],
    right: Sequence[StateSpanExample],
    *,
    values: Sequence[str],
) -> dict[str, Any]:
    def identity_sets(
        records: Sequence[StateSpanExample],
    ) -> Mapping[str, frozenset[str]]:
        components = tuple(
            line for record in records for line in record.transcript.splitlines()
        )
        return {
            "world_ids": frozenset(record.world_id for record in records),
            "record_ids": frozenset(record.example_id for record in records),
            "exact_transcripts": frozenset(record.transcript for record in records),
            "exact_transcript_templates": frozenset(
                _exact_template(record.transcript, values=values) for record in records
            ),
            "exact_component_line_templates": frozenset(
                _exact_template(component, values=values) for component in components
            ),
            "normalized_component_line_skeletons": frozenset(
                normalize_transcript(component, values=values)
                for component in components
            ),
        }

    left_sets = identity_sets(left)
    right_sets = identity_sets(right)
    return {
        dimension: {
            "count": len(left_sets[dimension] & right_sets[dimension]),
            "intersection_sha256": _string_multiset_sha256(
                left_sets[dimension] & right_sets[dimension]
            ),
        }
        for dimension in sorted(left_sets)
    }


def _output_order(families: Sequence[_WorldFamily]) -> tuple[_WorldFamily, ...]:
    return tuple(
        sorted(
            families,
            key=lambda family: (
                _sha256(
                    canonical_json_bytes(
                        {
                            "policy": SELECTION_POLICY_VERSION,
                            "role": "fit-output-order",
                            "source": family.source,
                            "source_world_id": family.source_world_id,
                        }
                    )
                ),
                family.source_world_id,
            ),
        )
    )


def _surface_source_partitions(
    surface_manifest: Mapping[str, Any],
    *,
    surface_fit: Sequence[StateSpanExample],
    calibration: Sequence[StateSpanExample],
) -> Mapping[str, Any]:
    generator_sha256 = surface_manifest.get("generator_sha256")
    if generator_sha256 != H4_GENERATOR_SHA256:
        raise TrainingInputError("H5 H4 generator authority drifted")
    partitions = surface_manifest.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != {
        surface_transfer_data.FIT_PARTITION,
        surface_transfer_data.CALIBRATION_PARTITION,
    }:
        raise TrainingInputError("H5 H4 source manifest partitions are malformed")
    expected = {
        surface_transfer_data.FIT_PARTITION: _sha256(_records_bytes(surface_fit)),
        surface_transfer_data.CALIBRATION_PARTITION: _sha256(
            _records_bytes(calibration)
        ),
    }
    if expected[surface_transfer_data.FIT_PARTITION] != H4_FIT_RECORDS_SHA256:
        raise TrainingInputError("H5 H4 fit authority drifted")
    if (
        expected[surface_transfer_data.CALIBRATION_PARTITION]
        != H4_CALIBRATION_RECORDS_SHA256
    ):
        raise TrainingInputError("H5 H4 calibration authority drifted")
    for partition, digest in expected.items():
        identity = partitions.get(partition)
        if (
            not isinstance(identity, dict)
            or identity.get("ordered_records_sha256") != digest
        ):
            raise TrainingInputError(
                f"H5 {partition} is not unchanged from the H4 authority"
            )
    return partitions


def build_replay_mixture_from_records(
    legacy: Sequence[StateSpanExample],
    surface_fit: Sequence[StateSpanExample],
    calibration: Sequence[StateSpanExample],
    *,
    legacy_manifest_sha256: str,
    surface_manifest: Mapping[str, Any],
    surface_manifest_sha256: str,
    generator_sha256: str,
) -> ReplayMixture:
    """Build and audit H5 from already-authenticated in-memory sources."""

    for label, digest in (
        ("legacy manifest", legacy_manifest_sha256),
        ("surface manifest", surface_manifest_sha256),
        ("H5 generator", generator_sha256),
    ):
        _require_digest(digest, label=label)
    if legacy_manifest_sha256 != LEGACY_MANIFEST_SHA256:
        raise TrainingInputError("H5 legacy manifest authority drifted")
    if surface_manifest_sha256 != H4_MANIFEST_SHA256:
        raise TrainingInputError("H5 H4 manifest authority drifted")
    if (
        len(legacy) != LEGACY_AUTHORITY_WORLDS * VARIANTS_PER_WORLD
        or _sha256(_records_bytes(legacy)) != LEGACY_TRAIN_RECORDS_SHA256
    ):
        raise TrainingInputError("H5 legacy in-memory authority drifted")
    source_identity = _surface_source_partitions(
        surface_manifest,
        surface_fit=surface_fit,
        calibration=calibration,
    )
    values = _normalization_values()
    legacy_authority_families = _world_families(
        legacy,
        source=LEGACY_SOURCE,
        expected_worlds=LEGACY_AUTHORITY_WORLDS,
        values=values,
    )
    legacy_families = tuple(
        family
        for family in legacy_authority_families
        if _source_index(family) < SOURCE_POOL_WORLDS
    )
    if len(legacy_families) != SOURCE_POOL_WORLDS:
        raise TrainingInputError("H5 legacy fit-world boundary drifted")
    surface_families = _world_families(
        surface_fit,
        source=SURFACE_SOURCE,
        expected_worlds=SOURCE_POOL_WORLDS,
        values=values,
    )
    calibration_families = _world_families(
        calibration,
        source=CALIBRATION_PARTITION_NAME,
        expected_worlds=CALIBRATION_WORLDS,
        values=values,
    )

    source_pairwise = {
        "legacy_vs_surface": _pair_overlap(legacy_families, surface_families),
        "legacy_vs_calibration": _pair_overlap(legacy_families, calibration_families),
        "surface_vs_calibration": _pair_overlap(surface_families, calibration_families),
    }
    _require_no_pair_overlap(source_pairwise)

    selected_legacy, legacy_strata = _select_source(
        legacy_families, source=LEGACY_SOURCE
    )
    selected_surface, surface_strata = _select_source(
        surface_families, source=SURFACE_SOURCE
    )
    selected_source_world_id_sha256 = {
        LEGACY_SOURCE: _string_multiset_sha256(
            family.source_world_id for family in selected_legacy
        ),
        SURFACE_SOURCE: _string_multiset_sha256(
            family.source_world_id for family in selected_surface
        ),
    }
    if selected_source_world_id_sha256 != dict(
        EXPECTED_SELECTED_SOURCE_WORLD_ID_SHA256
    ):
        raise TrainingInputError("H5 source selection identity drifted")
    value_overlap_audit = _value_overlap_audit(
        legacy_families, selected=selected_legacy
    )

    selected_pairwise = {
        "legacy_vs_surface": _pair_overlap(selected_legacy, selected_surface),
        "legacy_vs_calibration": _pair_overlap(selected_legacy, calibration_families),
        "surface_vs_calibration": _pair_overlap(selected_surface, calibration_families),
    }
    _require_no_pair_overlap(selected_pairwise)

    source_distribution: dict[str, tuple[dict[str, int], dict[str, int]]] = {}
    for source, families in (
        (LEGACY_SOURCE, selected_legacy),
        (SURFACE_SOURCE, selected_surface),
    ):
        source_records = tuple(
            record for family in families for record in family.records
        )
        source_distribution[source] = _state_counts(source_records)
        if source_distribution[source][0] != dict(
            EXPECTED_SOURCE_STATE_CLASS_COUNTS
        ) or source_distribution[source][1] != dict(EXPECTED_SOURCE_STATE_FIELD_QUOTA):
            raise TrainingInputError(
                f"H5 {source} half-stratum class distribution drifted"
            )

    selected = _output_order((*selected_legacy, *selected_surface))
    fit = tuple(record for family in selected for record in _namespace_family(family))
    calibration_output = tuple(calibration)
    if (
        len(fit) != FIT_RECORDS
        or len({record.world_id for record in fit}) != TOTAL_FIT_WORLDS
        or len({record.example_id for record in fit}) != FIT_RECORDS
        or {record.world_id for record in fit}
        & {record.world_id for record in calibration_output}
        or {record.example_id for record in fit}
        & {record.example_id for record in calibration_output}
        or {record.transcript for record in fit}
        & {record.transcript for record in calibration_output}
    ):
        raise RuntimeError("H5 output partition identity drifted")
    state_counts, state_field_quota = _state_counts(fit)
    if state_counts != dict(EXPECTED_STATE_CLASS_COUNTS) or state_field_quota != dict(
        EXPECTED_STATE_FIELD_QUOTA
    ):
        raise TrainingInputError("H5 step-matched class distribution drifted")

    calibration_sha256 = _sha256(_records_bytes(calibration_output))
    if calibration_sha256 != H4_CALIBRATION_RECORDS_SHA256:
        raise TrainingInputError("H5 H4 calibration identity drifted")

    selected_by_source = {
        LEGACY_SOURCE: selected_legacy,
        SURFACE_SOURCE: selected_surface,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generator": GENERATOR_VERSION,
        "recipe": TRAINING_RECIPE_VERSION,
        "selection_policy": SELECTION_POLICY_VERSION,
        "normalization": {
            "version": NORMALIZATION_VERSION,
            "value_count": len(values),
            "value_set_sha256": _sha256(canonical_json_bytes(sorted(values))),
            "unit": "transcript_line",
        },
        "generator_sha256": generator_sha256,
        "training_identity": {
            "training_seeds": list(TRAINING_SEEDS),
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "steps_per_epoch": STEPS_PER_EPOCH,
            "steps_per_seed": STEPS_PER_SEED,
            "fit_worlds": TOTAL_FIT_WORLDS,
            "fit_records": FIT_RECORDS,
            "legacy_ratio": "1/2",
            "surface_ratio": "1/2",
            "ratio_sweep": False,
            "ratio_schedule": False,
            "source_reweighting": False,
        },
        "sources": {
            "legacy": {
                "role": "regenerated_in_memory_from_frozen_seed",
                "generator_sha256": LEGACY_GENERATOR_SHA256,
                "manifest_sha256": legacy_manifest_sha256,
                "complete_authority_worlds": len(legacy_authority_families),
                "candidate_fit_worlds": len(legacy_families),
                "candidate_index_range": [0, SOURCE_POOL_WORLDS - 1],
                "selected_worlds": len(selected_legacy),
                "selected_records": len(selected_legacy) * VARIANTS_PER_WORLD,
                "state_class_counts": source_distribution[LEGACY_SOURCE][0],
                "state_field_quota": source_distribution[LEGACY_SOURCE][1],
                "selected_source_world_id_multiset_sha256": (
                    selected_source_world_id_sha256[LEGACY_SOURCE]
                ),
                "strata": legacy_strata,
            },
            "surface": {
                "role": "h4_fit_replay",
                "generator_sha256": surface_manifest["generator_sha256"],
                "manifest_sha256": surface_manifest_sha256,
                "fit_records_sha256": H4_FIT_RECORDS_SHA256,
                "calibration_records_sha256": H4_CALIBRATION_RECORDS_SHA256,
                "candidate_fit_worlds": len(surface_families),
                "candidate_index_range": [0, SOURCE_POOL_WORLDS - 1],
                "selected_worlds": len(selected_surface),
                "selected_records": len(selected_surface) * VARIANTS_PER_WORLD,
                "state_class_counts": source_distribution[SURFACE_SOURCE][0],
                "state_field_quota": source_distribution[SURFACE_SOURCE][1],
                "selected_source_world_id_multiset_sha256": (
                    selected_source_world_id_sha256[SURFACE_SOURCE]
                ),
                "strata": surface_strata,
            },
        },
        "partitions": {
            "fit": {
                "namespace": "train-world-replay-{legacy|surface}-NNNN",
                "records": len(fit),
                "worlds": len({record.world_id for record in fit}),
                "legacy_worlds": len(selected_by_source[LEGACY_SOURCE]),
                "surface_worlds": len(selected_by_source[SURFACE_SOURCE]),
                "ordered_records_sha256": _sha256(_records_bytes(fit)),
                "transcript_multiset_sha256": _true_transcript_multiset_sha256(fit),
                "world_id_multiset_sha256": _string_multiset_sha256(
                    record.world_id for record in fit
                ),
                "record_id_multiset_sha256": _string_multiset_sha256(
                    record.example_id for record in fit
                ),
                "state_class_counts": state_counts,
                "state_field_quota": state_field_quota,
                "gradient_bearing": True,
            },
            "calibration": {
                **dict(source_identity[surface_transfer_data.CALIBRATION_PARTITION]),
                "ordered_records_sha256": calibration_sha256,
                "reused_unchanged_from_h4": True,
                "gradient_bearing": False,
            },
        },
        "overlap_audit": {
            "hard_failure_dimensions": [
                "world_ids",
                "record_ids",
                "exact_transcripts",
                "exact_transcript_templates",
                "exact_component_line_templates",
                "normalized_component_line_skeletons",
            ],
            "candidate_partition_intersections": source_pairwise,
            "selected_partition_intersections": selected_pairwise,
            "calibration_open_value_literal_substring_occurrence": (
                value_overlap_audit.to_dict()
            ),
            "calibration_open_value_literal_substring_occurrence_is_eligibility_rule": (
                False
            ),
            "all_hard_intersections_zero": True,
            "calibration_records_modified": False,
        },
        "restrictions": {
            "legacy_record_artifact_read": False,
            "legacy_checkpoint_read": False,
            "development_read": False,
            "benchmark_read": False,
            "sealed_confirmation_read": False,
        },
    }
    return ReplayMixture(fit=fit, calibration=calibration_output, manifest=manifest)


def _require_h4_authority_files(h4_data_dir: Path) -> Mapping[str, str]:
    """Authenticate the accepted H4 source before any H4 regeneration occurs."""

    root = Path(h4_data_dir)
    expected = {
        "manifest": H4_MANIFEST_SHA256,
        surface_transfer_data.FIT_PARTITION: H4_FIT_RECORDS_SHA256,
        surface_transfer_data.CALIBRATION_PARTITION: H4_CALIBRATION_RECORDS_SHA256,
    }
    paths = {
        "manifest": root / "manifest.json",
        surface_transfer_data.FIT_PARTITION: root / "fit.jsonl",
        surface_transfer_data.CALIBRATION_PARTITION: root / "calibration.jsonl",
    }
    observed = {
        name: _sha256(_read_file(paths[name], label=f"H4 {name}")) for name in expected
    }
    if observed != expected:
        raise TrainingInputError("H5 accepted H4 data authority drifted")
    generator_sha256 = _sha256(
        _read_file(Path(surface_transfer_data.__file__), label="H4 generator")
    )
    if generator_sha256 != H4_GENERATOR_SHA256:
        raise TrainingInputError("H5 accepted H4 generator authority drifted")
    return observed


def build_replay_mixture(
    *, legacy_manifest_path: Path, h4_data_dir: Path
) -> ReplayMixture:
    """Authenticate both authorities, then build the fixed H5 mixture."""

    generator_path = Path(__file__).resolve()
    generator_before = _sha256(_read_file(generator_path, label="H5 generator"))
    legacy, _legacy_manifest, legacy_manifest_sha256 = load_legacy_reproduction(
        Path(legacy_manifest_path)
    )
    h4_authority = _require_h4_authority_files(Path(h4_data_dir))
    h4_bundle: H4TrainingBundle = load_h4_training_bundle(Path(h4_data_dir))
    if h4_bundle.input_sha256 != h4_authority:
        raise TrainingInputError("H5 H4 authority changed during authentication")
    if _require_h4_authority_files(Path(h4_data_dir)) != h4_authority:
        raise TrainingInputError("H5 H4 authority changed during construction")
    mixture = build_replay_mixture_from_records(
        legacy,
        h4_bundle.fit,
        h4_bundle.calibration,
        legacy_manifest_sha256=legacy_manifest_sha256,
        surface_manifest=h4_bundle.manifest,
        surface_manifest_sha256=h4_bundle.manifest_sha256,
        generator_sha256=generator_before,
    )
    if _sha256(_read_file(generator_path, label="H5 generator")) != generator_before:
        raise TrainingInputError("H5 generator changed during construction")
    return mixture


def _require_manifest_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TrainingInputError(f"H5 manifest {label} is malformed")
    return value


def _validate_source_manifest(
    sources: object,
    *,
    fit: Sequence[StateSpanExample],
) -> None:
    source_mapping = _require_manifest_mapping(sources, label="sources")
    if set(source_mapping) != set(SOURCES):
        raise TrainingInputError("H5 manifest source set drifted")
    expected_keys = {
        LEGACY_SOURCE: {
            "role",
            "generator_sha256",
            "manifest_sha256",
            "complete_authority_worlds",
            "candidate_fit_worlds",
            "candidate_index_range",
            "selected_worlds",
            "selected_records",
            "state_class_counts",
            "state_field_quota",
            "selected_source_world_id_multiset_sha256",
            "strata",
        },
        SURFACE_SOURCE: {
            "role",
            "generator_sha256",
            "manifest_sha256",
            "fit_records_sha256",
            "calibration_records_sha256",
            "candidate_fit_worlds",
            "candidate_index_range",
            "selected_worlds",
            "selected_records",
            "state_class_counts",
            "state_field_quota",
            "selected_source_world_id_multiset_sha256",
            "strata",
        },
    }
    for source in SOURCES:
        metadata = _require_manifest_mapping(
            source_mapping.get(source), label=f"source {source}"
        )
        if set(metadata) != expected_keys[source]:
            raise TrainingInputError(f"H5 {source} source metadata shape drifted")
        if (
            metadata["role"]
            != (
                "regenerated_in_memory_from_frozen_seed"
                if source == LEGACY_SOURCE
                else "h4_fit_replay"
            )
            or metadata["candidate_fit_worlds"] != SOURCE_POOL_WORLDS
            or metadata["candidate_index_range"] != [0, SOURCE_POOL_WORLDS - 1]
            or metadata["selected_worlds"] != SOURCE_WORLDS
            or metadata["selected_records"] != SOURCE_RECORDS
            or metadata["state_class_counts"]
            != dict(EXPECTED_SOURCE_STATE_CLASS_COUNTS)
            or metadata["state_field_quota"] != dict(EXPECTED_SOURCE_STATE_FIELD_QUOTA)
            or metadata["selected_source_world_id_multiset_sha256"]
            != EXPECTED_SELECTED_SOURCE_WORLD_ID_SHA256[source]
        ):
            raise TrainingInputError(f"H5 {source} source metadata drifted")
        if source == LEGACY_SOURCE:
            if (
                metadata["generator_sha256"] != LEGACY_GENERATOR_SHA256
                or metadata["manifest_sha256"] != LEGACY_MANIFEST_SHA256
                or metadata["complete_authority_worlds"] != LEGACY_AUTHORITY_WORLDS
            ):
                raise TrainingInputError("H5 legacy source authority drifted")
        else:
            if (
                metadata["generator_sha256"] != H4_GENERATOR_SHA256
                or metadata["manifest_sha256"] != H4_MANIFEST_SHA256
                or metadata["fit_records_sha256"] != H4_FIT_RECORDS_SHA256
                or metadata["calibration_records_sha256"]
                != H4_CALIBRATION_RECORDS_SHA256
            ):
                raise TrainingInputError("H5 surface source authority drifted")

        source_families: dict[int, tuple[StateSpanExample, ...]] = {}
        for record in fit:
            match = _REPLAY_WORLD_RE.fullmatch(record.world_id)
            if match is not None and match.group(1) == source:
                source_index = int(match.group(2))
                source_families.setdefault(source_index, ())
                source_families[source_index] += (record,)
        strata_observed: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for source_index, family in source_families.items():
            by_variant = {record.variant: record for record in family}
            original_world_id = _output_source_world_id(
                source=source, source_index=source_index
            )
            original_records = tuple(
                replace(
                    by_variant[variant],
                    example_id=_expected_record_id(
                        source=source, index=source_index, variant=variant
                    ),
                    world_id=original_world_id,
                )
                for variant in VARIANTS
            )
            stratum = _selection_stratum(
                by_variant["normal"],
                source=source,
                source_index=source_index,
            )
            strata_observed[stratum].append(
                (
                    source_index,
                    _selection_score(
                        source=source,
                        source_world_id=original_world_id,
                        records=original_records,
                    ),
                )
            )

        strata_metadata = _require_manifest_mapping(
            metadata["strata"], label=f"{source} strata"
        )
        if set(strata_metadata) != set(strata_observed) or len(strata_metadata) != 10:
            raise TrainingInputError(f"H5 {source} stratum set drifted")
        for stratum, members in strata_observed.items():
            identity = _require_manifest_mapping(
                strata_metadata[stratum], label=f"{source} {stratum}"
            )
            quota = _stratum_quota(stratum)
            ranked = sorted(members, key=lambda item: (item[1], item[0]))
            source_ids = (
                _output_source_world_id(source=source, source_index=index)
                for index, _score in members
            )
            if identity != {
                "candidate_worlds": quota * 2,
                "selected_worlds": quota,
                "selection_fraction": "1/2",
                "first_score": ranked[0][1],
                "final_score": ranked[-1][1],
                "selected_source_world_id_multiset_sha256": (
                    _string_multiset_sha256(source_ids)
                ),
            }:
                raise TrainingInputError(f"H5 {source} stratum metadata drifted")


def _validate_overlap_manifest(
    overlap: object,
    *,
    fit: Sequence[StateSpanExample],
    calibration: Sequence[StateSpanExample],
) -> None:
    value = _require_manifest_mapping(overlap, label="overlap audit")
    expected_keys = {
        "hard_failure_dimensions",
        "candidate_partition_intersections",
        "selected_partition_intersections",
        "calibration_open_value_literal_substring_occurrence",
        "calibration_open_value_literal_substring_occurrence_is_eligibility_rule",
        "all_hard_intersections_zero",
        "calibration_records_modified",
    }
    hard_dimensions = [
        "world_ids",
        "record_ids",
        "exact_transcripts",
        "exact_transcript_templates",
        "exact_component_line_templates",
        "normalized_component_line_skeletons",
    ]
    if (
        set(value) != expected_keys
        or value["hard_failure_dimensions"] != hard_dimensions
        or value[
            "calibration_open_value_literal_substring_occurrence_is_eligibility_rule"
        ]
        is not False
        or value["all_hard_intersections_zero"] is not True
        or value["calibration_records_modified"] is not False
    ):
        raise TrainingInputError("H5 overlap audit contract drifted")

    expected_pairs = {
        "legacy_vs_surface",
        "legacy_vs_calibration",
        "surface_vs_calibration",
    }
    candidate = _require_manifest_mapping(
        value["candidate_partition_intersections"],
        label="candidate intersections",
    )
    selected = _require_manifest_mapping(
        value["selected_partition_intersections"],
        label="selected intersections",
    )
    if set(candidate) != expected_pairs or set(selected) != expected_pairs:
        raise TrainingInputError("H5 overlap audit pair set drifted")
    _require_no_pair_overlap(candidate)

    fit_by_source = {
        source: tuple(
            record
            for record in fit
            if record.world_id.startswith(f"train-world-replay-{source}-")
        )
        for source in SOURCES
    }
    values = _normalization_values()
    expected_selected = {
        "legacy_vs_surface": _record_partition_overlap(
            fit_by_source[LEGACY_SOURCE],
            fit_by_source[SURFACE_SOURCE],
            values=values,
        ),
        "legacy_vs_calibration": _record_partition_overlap(
            fit_by_source[LEGACY_SOURCE], calibration, values=values
        ),
        "surface_vs_calibration": _record_partition_overlap(
            fit_by_source[SURFACE_SOURCE], calibration, values=values
        ),
    }
    if selected != expected_selected:
        raise TrainingInputError("H5 selected overlap audit drifted")
    _require_no_pair_overlap(selected)

    literal_audit = _require_manifest_mapping(
        value["calibration_open_value_literal_substring_occurrence"],
        label="calibration literal-substring audit",
    )
    occurrence_counts = literal_audit.get("literal_substring_occurrence_world_counts")
    expected_literal_audit = {
        "policy": "expected_recorded_nonblocking",
        "method": "nfkc_casefold_literal_substring_in_any_family_transcript",
        "normalization": "unicode_nfkc_then_casefold",
        "match_semantics": "literal_substring",
        "substring_metric_is_conservative": True,
        "substring_can_match_within_longer_values": True,
        "exact_value_identity_not_claimed": True,
        "candidate_worlds": SOURCE_POOL_WORLDS,
        "calibration_open_values": len(_calibration_open_values()),
        "literal_substring_disjoint_worlds": (
            EXPECTED_LEGACY_LITERAL_SUBSTRING_DISJOINT_WORLDS
        ),
        "worlds_with_literal_substring_occurrence": (
            SOURCE_POOL_WORLDS - EXPECTED_LEGACY_LITERAL_SUBSTRING_DISJOINT_WORLDS
        ),
        "literal_substring_disjoint_by_target_field": dict(
            EXPECTED_LEGACY_LITERAL_SUBSTRING_DISJOINT_BY_FIELD
        ),
        "balanced_literal_substring_disjoint_limit": (
            EXPECTED_LEGACY_BALANCED_LITERAL_SUBSTRING_DISJOINT_LIMIT
        ),
        "selected_worlds": SOURCE_WORLDS,
        "selected_literal_substring_disjoint_worlds": (
            EXPECTED_SELECTED_LEGACY_LITERAL_SUBSTRING_DISJOINT_WORLDS
        ),
        "selected_worlds_with_literal_substring_occurrence": (
            SOURCE_WORLDS - EXPECTED_SELECTED_LEGACY_LITERAL_SUBSTRING_DISJOINT_WORLDS
        ),
        "selected_literal_substring_disjoint_by_target_field": dict(
            EXPECTED_SELECTED_LEGACY_LITERAL_SUBSTRING_DISJOINT_BY_FIELD
        ),
        "literal_substring_occurrence_world_counts": occurrence_counts,
    }
    if literal_audit != expected_literal_audit or not isinstance(
        occurrence_counts, dict
    ):
        raise TrainingInputError("H5 calibration literal-substring audit drifted")
    allowed_values = set(_calibration_open_values())
    if any(
        key not in allowed_values
        or not isinstance(count, int)
        or isinstance(count, bool)
        or count < 0
        or count > SOURCE_POOL_WORLDS
        for key, count in occurrence_counts.items()
    ):
        raise TrainingInputError("H5 calibration occurrence counts drifted")


def _validate_replay_manifest(
    value: object,
    *,
    fit: Sequence[StateSpanExample],
    calibration: Sequence[StateSpanExample],
    digests: Mapping[str, str],
    generator_sha256: str,
) -> Mapping[str, Any]:
    manifest = _require_manifest_mapping(value, label="root")
    if set(manifest) != {
        "schema_version",
        "generator",
        "recipe",
        "selection_policy",
        "normalization",
        "generator_sha256",
        "training_identity",
        "sources",
        "partitions",
        "overlap_audit",
        "restrictions",
    }:
        raise TrainingInputError("H5 manifest shape drifted")
    values = _normalization_values()
    if (
        manifest["schema_version"] != MANIFEST_SCHEMA_VERSION
        or manifest["generator"] != GENERATOR_VERSION
        or manifest["recipe"] != TRAINING_RECIPE_VERSION
        or manifest["selection_policy"] != SELECTION_POLICY_VERSION
        or manifest["generator_sha256"] != generator_sha256
        or manifest["normalization"]
        != {
            "version": NORMALIZATION_VERSION,
            "value_count": len(values),
            "value_set_sha256": _sha256(canonical_json_bytes(sorted(values))),
            "unit": "transcript_line",
        }
        or manifest["training_identity"]
        != {
            "training_seeds": list(TRAINING_SEEDS),
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "steps_per_epoch": STEPS_PER_EPOCH,
            "steps_per_seed": STEPS_PER_SEED,
            "fit_worlds": TOTAL_FIT_WORLDS,
            "fit_records": FIT_RECORDS,
            "legacy_ratio": "1/2",
            "surface_ratio": "1/2",
            "ratio_sweep": False,
            "ratio_schedule": False,
            "source_reweighting": False,
        }
    ):
        raise TrainingInputError("H5 manifest recipe identity drifted")

    _validate_source_manifest(manifest["sources"], fit=fit)
    partitions = _require_manifest_mapping(manifest["partitions"], label="partitions")
    if set(partitions) != {"fit", CALIBRATION_PARTITION_NAME}:
        raise TrainingInputError("H5 manifest partition set drifted")
    expected_fit = {
        "namespace": "train-world-replay-{legacy|surface}-NNNN",
        "records": FIT_RECORDS,
        "worlds": TOTAL_FIT_WORLDS,
        "legacy_worlds": SOURCE_WORLDS,
        "surface_worlds": SOURCE_WORLDS,
        "ordered_records_sha256": digests["fit"],
        "transcript_multiset_sha256": _true_transcript_multiset_sha256(fit),
        "world_id_multiset_sha256": _string_multiset_sha256(
            record.world_id for record in fit
        ),
        "record_id_multiset_sha256": _string_multiset_sha256(
            record.example_id for record in fit
        ),
        "state_class_counts": dict(EXPECTED_STATE_CLASS_COUNTS),
        "state_field_quota": dict(EXPECTED_STATE_FIELD_QUOTA),
        "gradient_bearing": True,
    }
    if partitions["fit"] != expected_fit:
        raise TrainingInputError("H5 fit manifest identity drifted")
    calibration_identity = _require_manifest_mapping(
        partitions[CALIBRATION_PARTITION_NAME],
        label="calibration partition",
    )
    if (
        calibration_identity.get("namespace") != "train-calibration"
        or calibration_identity.get("records") != CALIBRATION_RECORDS
        or calibration_identity.get("worlds") != CALIBRATION_WORLDS
        or calibration_identity.get("ordered_records_sha256")
        != H4_CALIBRATION_RECORDS_SHA256
        or digests[CALIBRATION_PARTITION_NAME] != H4_CALIBRATION_RECORDS_SHA256
        or calibration_identity.get("reused_unchanged_from_h4") is not True
        or calibration_identity.get("gradient_bearing") is not False
    ):
        raise TrainingInputError("H5 calibration manifest identity drifted")
    _validate_overlap_manifest(
        manifest["overlap_audit"], fit=fit, calibration=calibration
    )
    if manifest["restrictions"] != {
        "legacy_record_artifact_read": False,
        "legacy_checkpoint_read": False,
        "development_read": False,
        "benchmark_read": False,
        "sealed_confirmation_read": False,
    }:
        raise TrainingInputError("H5 restricted-input policy drifted")
    return manifest


def load_replay_mixture_dataset(data_dir: Path) -> ReplayMixtureBundle:
    """Load and fail-closed authenticate the three-file H5 dataset bundle."""

    root = Path(data_dir)
    paths = {
        "manifest": root / "manifest.json",
        "fit": root / "fit.jsonl",
        CALIBRATION_PARTITION_NAME: root / "calibration.jsonl",
    }
    snapshots = {name: _read_file(path, label=name) for name, path in paths.items()}
    digests = {name: _sha256(payload) for name, payload in snapshots.items()}
    fit = _load_replay_partition(snapshots["fit"], filename="fit.jsonl")
    calibration = _load_replay_partition(
        snapshots[CALIBRATION_PARTITION_NAME], filename="calibration.jsonl"
    )
    _validate_replay_fit(fit)
    _require_complete_worlds(
        calibration,
        source=CALIBRATION_PARTITION_NAME,
        expected_worlds=CALIBRATION_WORLDS,
    )
    if (
        len(calibration) != CALIBRATION_RECORDS
        or {record.world_id for record in fit}
        & {record.world_id for record in calibration}
        or {record.example_id for record in fit}
        & {record.example_id for record in calibration}
        or {record.transcript for record in fit}
        & {record.transcript for record in calibration}
    ):
        raise TrainingInputError("H5 fit/calibration isolation drifted")

    manifest_value = _parse_json(snapshots["manifest"], label="manifest.json")
    if canonical_json_bytes(manifest_value) != snapshots["manifest"]:
        raise TrainingInputError("H5 manifest is not canonical JSON")
    generator_path = Path(__file__).resolve()
    generator_sha256 = _sha256(_read_file(generator_path, label="generator source"))
    manifest = _validate_replay_manifest(
        manifest_value,
        fit=fit,
        calibration=calibration,
        digests=digests,
        generator_sha256=generator_sha256,
    )
    if {
        name: _read_file(path, label=name) for name, path in paths.items()
    } != snapshots or _sha256(
        _read_file(generator_path, label="generator source")
    ) != generator_sha256:
        raise TrainingInputError("H5 dataset changed during authentication")
    return ReplayMixtureBundle(
        manifest=manifest,
        manifest_sha256=digests["manifest"],
        fit=fit,
        calibration=calibration,
        input_sha256=digests,
    )


def _write_no_clobber(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise OSError(f"could not create H5 dataset file: {path.name}") from exc


def write_replay_mixture_dataset(
    output_dir: Path, *, legacy_manifest_path: Path, h4_data_dir: Path
) -> Mapping[str, Any]:
    """Build all H5 evidence before creating a no-clobber output directory."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"H5 output directory already exists: {output}")
    generator_path = Path(__file__).resolve()
    generator_before = _sha256(_read_file(generator_path, label="H5 generator"))
    legacy_manifest_before = _sha256(
        _read_file(Path(legacy_manifest_path), label="legacy manifest")
    )
    h4_input_paths = {
        "manifest": Path(h4_data_dir) / "manifest.json",
        "fit": Path(h4_data_dir) / "fit.jsonl",
        "calibration": Path(h4_data_dir) / "calibration.jsonl",
    }
    h4_before = {
        name: _sha256(_read_file(path, label=f"H4 {name}"))
        for name, path in h4_input_paths.items()
    }
    mixture = build_replay_mixture(
        legacy_manifest_path=Path(legacy_manifest_path), h4_data_dir=Path(h4_data_dir)
    )
    if (
        _sha256(_read_file(generator_path, label="H5 generator")) != generator_before
        or _sha256(_read_file(Path(legacy_manifest_path), label="legacy manifest"))
        != legacy_manifest_before
        or {
            name: _sha256(_read_file(path, label=f"H4 {name}"))
            for name, path in h4_input_paths.items()
        }
        != h4_before
    ):
        raise TrainingInputError("H5 authority changed before output creation")
    output.mkdir(parents=True, exist_ok=False)
    _write_no_clobber(output / "fit.jsonl", _records_bytes(mixture.fit))
    _write_no_clobber(output / "calibration.jsonl", _records_bytes(mixture.calibration))
    _write_no_clobber(output / "manifest.json", canonical_json_bytes(mixture.manifest))
    return mixture.manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Nano's deterministic H5 replay mixture"
    )
    parser.add_argument("--legacy-manifest", type=Path, required=True)
    parser.add_argument("--h4-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = write_replay_mixture_dataset(
        args.output_dir,
        legacy_manifest_path=args.legacy_manifest,
        h4_data_dir=args.h4_data_dir,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BATCH_SIZE",
    "CALIBRATION_RECORDS",
    "CALIBRATION_WORLDS",
    "EPOCHS",
    "EXPECTED_SOURCE_STATE_CLASS_COUNTS",
    "EXPECTED_STATE_CLASS_COUNTS",
    "EXPECTED_STATE_FIELD_QUOTA",
    "FIT_RECORDS",
    "GENERATOR_VERSION",
    "LEGACY_MANIFEST_SHA256",
    "MANIFEST_SCHEMA_VERSION",
    "NORMALIZATION_VERSION",
    "SELECTION_POLICY_VERSION",
    "SOURCE_RECORDS",
    "SOURCE_WORLDS",
    "STEPS_PER_EPOCH",
    "STEPS_PER_SEED",
    "TOTAL_FIT_WORLDS",
    "TRAINING_RECIPE_VERSION",
    "TRAINING_SEEDS",
    "VARIANTS_PER_WORLD",
    "ReplayMixtureBundle",
    "build_replay_mixture",
    "build_replay_mixture_from_records",
    "load_legacy_reproduction",
    "load_replay_mixture_dataset",
    "normalize_transcript",
    "write_replay_mixture_dataset",
]
