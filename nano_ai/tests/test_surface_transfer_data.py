from __future__ import annotations

import ast
import hashlib
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import pytest

from nano_ai.adapters.state_span import parse_state_span_summary
from nano_ai.contract import FIELD_ORDER, FieldName, FieldState
from nano_ai.training import surface_transfer_data
from nano_ai.training.pointer_data import (
    encode_pointer_partition,
    load_pointer_tokenizer,
)
from nano_ai.training.surface_transfer_data import (
    CALIBRATION_PARTITION,
    CALIBRATION_RECORDS,
    CALIBRATION_WORLDS,
    FIT_PARTITION,
    FIT_RECORDS,
    FIT_WORLDS,
    generate_partition,
    partition_lexicons,
    records_bytes,
    transcript_multiset_sha256,
    write_dataset,
)


@lru_cache(maxsize=2)
def _records(partition: str):  # type: ignore[no-untyped-def]
    return generate_partition(partition)


_EXPECTED_QUOTAS = {
    FIT_PARTITION: {
        FieldName.CHIEF_COMPLAINT: (9_520, 0, 560, 560, 560),
        FieldName.DURATION: (9_520, 0, 560, 560, 560),
        FieldName.SEVERITY: (9_520, 0, 560, 560, 560),
        FieldName.MEDICATION: (7_140, 2_380, 560, 560, 560),
        FieldName.ALLERGY: (7_280, 2_240, 560, 560, 560),
    },
    CALIBRATION_PARTITION: {
        FieldName.CHIEF_COMPLAINT: (680, 0, 40, 40, 40),
        FieldName.DURATION: (680, 0, 40, 40, 40),
        FieldName.SEVERITY: (680, 0, 40, 40, 40),
        FieldName.MEDICATION: (510, 170, 40, 40, 40),
        FieldName.ALLERGY: (520, 160, 40, 40, 40),
    },
}
_STATE_ORDER = (
    FieldState.SUPPORTED,
    FieldState.ABSENT,
    FieldState.MISSING,
    FieldState.UNCERTAIN,
    FieldState.CONFLICTING,
)


def test_h4_generation_is_deterministic_and_namespaces_are_independent() -> None:
    for partition, expected_records, expected_worlds in (
        (FIT_PARTITION, FIT_RECORDS, FIT_WORLDS),
        (CALIBRATION_PARTITION, CALIBRATION_RECORDS, CALIBRATION_WORLDS),
    ):
        first = _records(partition)
        second = generate_partition(partition)
        assert len(first) == expected_records
        assert len({row.world_id for row in first}) == expected_worlds
        assert len({row.transcript for row in first}) == expected_records
        assert records_bytes(first) == records_bytes(second)
        assert all(row.split == "train" for row in first)
        assert all(row.example_id.startswith(f"train-{partition}-") for row in first)
        assert all(
            row.world_id.startswith(f"train-world-{partition}-") for row in first
        )
    fit = _records(FIT_PARTITION)
    calibration = _records(CALIBRATION_PARTITION)
    assert {row.example_id for row in fit}.isdisjoint(
        row.example_id for row in calibration
    )
    assert {row.world_id for row in fit}.isdisjoint(row.world_id for row in calibration)
    assert {row.transcript for row in fit}.isdisjoint(
        row.transcript for row in calibration
    )


def test_h4_has_four_complete_paired_variants_and_exact_state_quotas() -> None:
    for partition in (FIT_PARTITION, CALIBRATION_PARTITION):
        records = _records(partition)
        by_world: dict[str, list[str]] = defaultdict(list)
        counts: Counter[tuple[FieldName, FieldState]] = Counter()
        for row in records:
            by_world[row.world_id].append(row.variant)
            for proposal in parse_state_span_summary(row.target, row.transcript):
                counts[(proposal.field, proposal.state)] += 1
        assert all(
            variants == ["normal", "missing", "uncertain", "conflicting"]
            for variants in by_world.values()
        )
        for field in FIELD_ORDER:
            assert tuple(counts[(field, state)] for state in _STATE_ORDER) == (
                _EXPECTED_QUOTAS[partition][field]
            )


def test_h4_lexicons_are_exact_disjoint_and_every_open_value_is_exercised() -> None:
    lexicons = partition_lexicons()
    expected_sizes = {
        FIT_PARTITION: (192, 12, 3, 48, 48),
        CALIBRATION_PARTITION: (48, 3, 3, 16, 16),
    }
    for partition in (FIT_PARTITION, CALIBRATION_PARTITION):
        assert tuple(len(lexicons[partition][field]) for field in FIELD_ORDER) == (
            expected_sizes[partition]
        )
    for field in FIELD_ORDER:
        overlap = set(lexicons[FIT_PARTITION][field]) & set(
            lexicons[CALIBRATION_PARTITION][field]
        )
        assert overlap == (
            set(lexicons[FIT_PARTITION][field])
            if field is FieldName.SEVERITY
            else set()
        )

    for partition in (FIT_PARTITION, CALIBRATION_PARTITION):
        observed: dict[FieldName, set[str]] = defaultdict(set)
        for row in _records(partition):
            for proposal in parse_state_span_summary(row.target, row.transcript):
                if proposal.state in {FieldState.SUPPORTED, FieldState.CONFLICTING}:
                    observed[proposal.field].update(span.text for span in proposal.spans)
        for field in FIELD_ORDER:
            assert set(lexicons[partition][field]) <= observed[field]


def test_h4_value_surfaces_are_complete_and_conflicts_are_balanced() -> None:
    lexicons = partition_lexicons()
    expected_surfaces = {FIT_PARTITION: 12, CALIBRATION_PARTITION: 4}
    expected_conflicts = {FIT_PARTITION: 560, CALIBRATION_PARTITION: 40}
    for partition in (FIT_PARTITION, CALIBRATION_PARTITION):
        _rows, traces = surface_transfer_data._generate_with_trace(partition)
        questions: dict[tuple[str, str], set[str]] = defaultdict(set)
        boundaries: dict[tuple[str, str], set[str]] = defaultdict(set)
        answers: dict[tuple[str, str], set[str]] = defaultdict(set)
        alternatives: dict[str, Counter[str]] = defaultdict(Counter)
        for index, trace in enumerate(traces):
            if index % 4 == 0:
                question_by_field = dict(trace.question_templates)
                answer_by_field = {
                    field: (boundary, template)
                    for field, boundary, template in trace.answer_templates
                }
                for field, value in trace.values:
                    key = (field, value)
                    questions[key].add(question_by_field[field])
                    boundary, template = answer_by_field[field]
                    boundaries[key].add(boundary)
                    answers[key].add(template)
            if trace.correction_value is not None:
                field, alternative = trace.correction_value
                alternatives[field][alternative] += 1
                current = dict(trace.values).get(field)
                assert current is None or current != alternative

        for field in FIELD_ORDER:
            for value in lexicons[partition][field]:
                key = (field.value, value)
                assert len(questions[key]) == expected_surfaces[partition]
                assert len(boundaries[key]) == 4
                assert len(answers[key]) == expected_surfaces[partition]
            counts = [
                alternatives[field.value][value]
                for value in lexicons[partition][field]
            ]
            assert sum(counts) == expected_conflicts[partition]
            assert max(counts) - min(counts) <= 1
            assert sum(count > 0 for count in counts) == min(
                expected_conflicts[partition], len(counts)
            )


def test_direct_gold_spans_are_unique_patient_owned_and_round_trip() -> None:
    for partition in (FIT_PARTITION, CALIBRATION_PARTITION):
        for row in _records(partition):
            proposals = parse_state_span_summary(row.target, row.transcript)
            offsets: list[tuple[int, int]] = []
            for proposal in proposals:
                for span in proposal.spans:
                    span.validate_against(row.transcript)
                    assert row.transcript[span.start : span.end] == span.text
                    offsets.append((span.start, span.end))
            assert len(offsets) == len(set(offsets))


def test_transcript_digest_is_a_true_multiset() -> None:
    row = generate_partition(FIT_PARTITION, worlds=5)[0]
    assert transcript_multiset_sha256((row,)) != transcript_multiset_sha256((row, row))
    expected = hashlib.sha256(
        surface_transfer_data.canonical_json_bytes(
            sorted(
                [
                    hashlib.sha256(row.transcript.encode("utf-8")).hexdigest(),
                    hashlib.sha256(row.transcript.encode("utf-8")).hexdigest(),
                ]
            )
        )
    ).hexdigest()
    assert transcript_multiset_sha256((row, row)) == expected


def test_manifest_freezes_leakage_signatures_and_step_matched_identity(
    tmp_path: Path,
) -> None:
    output = tmp_path / "h4-data"
    manifest = write_dataset(
        output,
        tokenizer_sha256="1" * 64,
        base_checkpoint_sha256="2" * 64,
    )
    assert manifest["schema_version"] == "nano.surface-transfer-manifest.v1"
    assert manifest["generator"] == "nano.surface-transfer-dataset.v1"
    assert manifest["recipe"] == "nano-evidence-query-data-only-v1"
    assert manifest["training_identity"] == {
        "training_seeds": [20260805, 20260806],
        "batch_size": 32,
        "epochs": 3,
        "steps_per_epoch": 350,
        "steps_per_seed": 1_050,
        "gradient_records": 11_200,
        "calibration_records": 800,
        "calibration_gradient_bearing": False,
    }
    assert manifest["isolation"] == {
        "world_ids_disjoint": True,
        "record_ids_disjoint": True,
        "exact_transcripts_disjoint": True,
        "open_values_disjoint_except_severity": True,
        "exact_templates_disjoint": True,
        "normalized_component_skeletons_disjoint": True,
        "known_development_role": "leakage_rejection_only_after_design_freeze",
        "open_values_disjoint": True,
        "fit_component_skeletons_disjoint": True,
        "calibration_component_skeletons_disjoint": True,
        "development_records_in_training": 0,
        "development_used_for_template_or_value_selection": False,
        "gold_built_from_annotated_turn_offsets": True,
        "deterministic_baseline_used_for_labels": False,
        "historical_sentinels_present": False,
        "historical_fresh_v0_read": False,
        "sealed_confirmation_read": False,
    }
    for partition, filename in (
        (FIT_PARTITION, "fit.jsonl"),
        (CALIBRATION_PARTITION, "calibration.jsonl"),
    ):
        identity = manifest["partitions"][partition]
        assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == identity[
            "ordered_records_sha256"
        ]
        assert len(identity["transcript_multiset_sha256"]) == 64
        assert len(identity["component_line_skeleton_multiset_sha256"]) == 64
        assert len(identity["templates"]["sha256"]) == 64
        assert identity["quality"]["unique_transcripts"] == identity["records"]
        assert identity["quality"]["all_transcripts_unique"] is True
        assert identity["quality"]["all_supported_conflicts_deranged"] is True
        assert len(identity["quality"]["supported_value_surface_sha256"]) == 64
        assert len(identity["quality"]["conflict_alternative_sha256"]) == 64
    with pytest.raises(FileExistsError):
        write_dataset(
            output,
            tokenizer_sha256="1" * 64,
            base_checkpoint_sha256="2" * 64,
        )


def test_h4_encodes_under_frozen_tokenizer_within_context_limit() -> None:
    root = Path(__file__).resolve().parents[2]
    tokenizer = load_pointer_tokenizer(root / "sft" / "tokenizer.json")
    for partition in (FIT_PARTITION, CALIBRATION_PARTITION):
        encoded = encode_pointer_partition(
            tokenizer,
            _records(partition),
            expected_split="train",
        )
        assert len(encoded) == len(_records(partition))
        assert max(len(item.token_ids) for item in encoded) <= 512


def test_h4_generator_has_no_benchmark_or_confirmation_imports() -> None:
    source_path = Path(surface_transfer_data.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not any("benchmark" in name or "fresh" in name for name in imports)
    assert "fresh_v1_partition" not in source
