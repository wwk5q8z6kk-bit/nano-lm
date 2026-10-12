from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import pytest

from nano_ai.contract import FIELD_ORDER
from nano_ai.training import replay_mixture_data as replay
from nano_ai.training import state_span_data, surface_transfer_data
from nano_ai.training.train_state_span import TrainingInputError


@pytest.fixture(scope="session")
def replay_sources() -> tuple[
    tuple[state_span_data.StateSpanExample, ...],
    tuple[state_span_data.StateSpanExample, ...],
    tuple[state_span_data.StateSpanExample, ...],
    dict[str, object],
]:
    legacy = state_span_data.generate_split("train")
    surface_fit = surface_transfer_data.generate_partition(
        surface_transfer_data.FIT_PARTITION
    )
    calibration = surface_transfer_data.generate_partition(
        surface_transfer_data.CALIBRATION_PARTITION
    )
    surface_manifest: dict[str, object] = {
        "generator_sha256": replay.H4_GENERATOR_SHA256,
        "partitions": {
            surface_transfer_data.FIT_PARTITION: {
                "ordered_records_sha256": hashlib.sha256(
                    replay._records_bytes(surface_fit)
                ).hexdigest(),
            },
            surface_transfer_data.CALIBRATION_PARTITION: {
                "namespace": "train-calibration",
                "records": replay.CALIBRATION_RECORDS,
                "worlds": replay.CALIBRATION_WORLDS,
                "ordered_records_sha256": hashlib.sha256(
                    replay._records_bytes(calibration)
                ).hexdigest(),
            },
        },
    }
    return legacy, surface_fit, calibration, surface_manifest


@pytest.fixture(scope="session")
def mixture(
    replay_sources: tuple[
        tuple[state_span_data.StateSpanExample, ...],
        tuple[state_span_data.StateSpanExample, ...],
        tuple[state_span_data.StateSpanExample, ...],
        dict[str, object],
    ],
) -> replay.ReplayMixture:
    legacy, surface_fit, calibration, surface_manifest = replay_sources
    generator_sha256 = hashlib.sha256(Path(replay.__file__).read_bytes()).hexdigest()
    return replay.build_replay_mixture_from_records(
        legacy,
        surface_fit,
        calibration,
        legacy_manifest_sha256=replay.LEGACY_MANIFEST_SHA256,
        surface_manifest=surface_manifest,
        surface_manifest_sha256=replay.H4_MANIFEST_SHA256,
        generator_sha256=generator_sha256,
    )


def _write_bundle(root: Path, value: replay.ReplayMixture) -> None:
    (root / "fit.jsonl").write_bytes(replay._records_bytes(value.fit))
    (root / "calibration.jsonl").write_bytes(replay._records_bytes(value.calibration))
    (root / "manifest.json").write_bytes(
        state_span_data.canonical_json_bytes(value.manifest)
    )


def test_fixed_replay_contract_is_exactly_step_matched(
    mixture: replay.ReplayMixture,
) -> None:
    assert len(mixture.fit) == 11_200
    assert len({record.world_id for record in mixture.fit}) == 2_800
    assert len(mixture.calibration) == 800
    identity = mixture.manifest["training_identity"]
    assert identity == {
        "training_seeds": [20260805, 20260806],
        "batch_size": 32,
        "epochs": 3,
        "steps_per_epoch": 350,
        "steps_per_seed": 1_050,
        "fit_worlds": 2_800,
        "fit_records": 11_200,
        "legacy_ratio": "1/2",
        "surface_ratio": "1/2",
        "ratio_sweep": False,
        "ratio_schedule": False,
        "source_reweighting": False,
    }
    assert mixture.manifest["partitions"]["calibration"]["gradient_bearing"] is False
    expected_surface_authority = {
        "generator_sha256": replay.H4_GENERATOR_SHA256,
        "manifest_sha256": replay.H4_MANIFEST_SHA256,
        "fit_records_sha256": replay.H4_FIT_RECORDS_SHA256,
        "calibration_records_sha256": replay.H4_CALIBRATION_RECORDS_SHA256,
    }
    observed_surface = mixture.manifest["sources"][replay.SURFACE_SOURCE]
    assert {
        key: observed_surface[key] for key in expected_surface_authority
    } == expected_surface_authority


def test_h4_replay_authority_rejects_digest_shaped_drift(
    replay_sources: tuple[
        tuple[state_span_data.StateSpanExample, ...],
        tuple[state_span_data.StateSpanExample, ...],
        tuple[state_span_data.StateSpanExample, ...],
        dict[str, object],
    ],
) -> None:
    legacy, surface_fit, calibration, surface_manifest = replay_sources
    generator_sha256 = hashlib.sha256(Path(replay.__file__).read_bytes()).hexdigest()

    with pytest.raises(TrainingInputError, match="H4 manifest authority drifted"):
        replay.build_replay_mixture_from_records(
            legacy,
            surface_fit,
            calibration,
            legacy_manifest_sha256=replay.LEGACY_MANIFEST_SHA256,
            surface_manifest=surface_manifest,
            surface_manifest_sha256="c" * 64,
            generator_sha256=generator_sha256,
        )

    changed_manifest = dict(surface_manifest)
    changed_manifest["generator_sha256"] = "b" * 64
    with pytest.raises(TrainingInputError, match="H4 generator authority drifted"):
        replay.build_replay_mixture_from_records(
            legacy,
            surface_fit,
            calibration,
            legacy_manifest_sha256=replay.LEGACY_MANIFEST_SHA256,
            surface_manifest=changed_manifest,
            surface_manifest_sha256=replay.H4_MANIFEST_SHA256,
            generator_sha256=generator_sha256,
        )


def test_selection_pins_half_of_every_frozen_index_stratum(
    mixture: replay.ReplayMixture,
) -> None:
    field_counts: Counter[tuple[str, str]] = Counter()
    selected_indices: dict[str, set[int]] = {
        replay.LEGACY_SOURCE: set(),
        replay.SURFACE_SOURCE: set(),
    }
    for world_id in {record.world_id for record in mixture.fit}:
        _train, _world, _replay, source, raw_index = world_id.split("-")
        index = int(raw_index)
        selected_indices[source].add(index)
        field_counts[source, FIELD_ORDER[index % len(FIELD_ORDER)].value] += 1

    for source in replay.SOURCES:
        assert len(selected_indices[source]) == 1_400
        assert selected_indices[source].isdisjoint(range(2_800, 3_000))
        assert {field: field_counts[source, field.value] for field in FIELD_ORDER} == {
            field: 280 for field in FIELD_ORDER
        }
        source_manifest = mixture.manifest["sources"][source]
        assert source_manifest["candidate_index_range"] == [0, 2_799]
        assert (
            source_manifest["selected_source_world_id_multiset_sha256"]
            == (replay.EXPECTED_SELECTED_SOURCE_WORLD_ID_SHA256[source])
        )
        assert len(source_manifest["strata"]) == 10
        for stratum in source_manifest["strata"].values():
            assert (stratum["candidate_worlds"], stratum["selected_worlds"]) in {
                (140, 70),
                (420, 210),
            }
            assert stratum["selection_fraction"] == "1/2"


def test_state_distribution_is_equal_per_source_and_exact_combined(
    mixture: replay.ReplayMixture,
) -> None:
    for source in replay.SOURCES:
        source_manifest = mixture.manifest["sources"][source]
        assert source_manifest["state_class_counts"] == dict(
            replay.EXPECTED_SOURCE_STATE_CLASS_COUNTS
        )
        assert source_manifest["state_field_quota"] == dict(
            replay.EXPECTED_SOURCE_STATE_FIELD_QUOTA
        )
    fit_manifest = mixture.manifest["partitions"]["fit"]
    assert fit_manifest["state_class_counts"] == dict(
        replay.EXPECTED_STATE_CLASS_COUNTS
    )
    assert fit_manifest["state_field_quota"] == dict(replay.EXPECTED_STATE_FIELD_QUOTA)


def test_calibration_literal_substring_audit_is_nonblocking_and_explicit(
    mixture: replay.ReplayMixture,
) -> None:
    overlap = mixture.manifest["overlap_audit"]
    audit = overlap["calibration_open_value_literal_substring_occurrence"]
    assert (
        overlap[
            "calibration_open_value_literal_substring_occurrence_is_eligibility_rule"
        ]
        is False
    )
    assert audit["policy"] == "expected_recorded_nonblocking"
    assert audit["normalization"] == "unicode_nfkc_then_casefold"
    assert audit["match_semantics"] == "literal_substring"
    assert audit["substring_metric_is_conservative"] is True
    assert audit["substring_can_match_within_longer_values"] is True
    assert audit["exact_value_identity_not_claimed"] is True
    assert audit["literal_substring_disjoint_worlds"] == 1_053
    assert audit["worlds_with_literal_substring_occurrence"] == 1_747
    assert audit["literal_substring_disjoint_by_target_field"] == {
        "chief_complaint": 263,
        "duration": 178,
        "severity": 199,
        "medication": 206,
        "allergy": 207,
    }
    assert audit["balanced_literal_substring_disjoint_limit"] == 890
    assert audit["selected_literal_substring_disjoint_worlds"] == 539
    assert audit["selected_worlds_with_literal_substring_occurrence"] == 861
    assert audit["selected_literal_substring_disjoint_by_target_field"] == {
        "chief_complaint": 139,
        "duration": 94,
        "severity": 110,
        "medication": 90,
        "allergy": 106,
    }


def test_all_structural_overlap_dimensions_are_hard_zero(
    mixture: replay.ReplayMixture,
) -> None:
    overlap = mixture.manifest["overlap_audit"]
    expected_dimensions = {
        "world_ids",
        "record_ids",
        "exact_transcripts",
        "exact_transcript_templates",
        "exact_component_line_templates",
        "normalized_component_line_skeletons",
    }
    for scope in (
        "candidate_partition_intersections",
        "selected_partition_intersections",
    ):
        assert set(overlap[scope]) == {
            "legacy_vs_surface",
            "legacy_vs_calibration",
            "surface_vs_calibration",
        }
        for dimensions in overlap[scope].values():
            assert set(dimensions) == expected_dimensions
            assert all(identity["count"] == 0 for identity in dimensions.values())
    with pytest.raises(TrainingInputError, match="overlap is forbidden"):
        replay._require_no_pair_overlap(
            {"mutated": {"exact_transcripts": {"count": 1}}}
        )


def test_h4_calibration_is_reused_in_original_order(
    mixture: replay.ReplayMixture,
    replay_sources: tuple[
        tuple[state_span_data.StateSpanExample, ...],
        tuple[state_span_data.StateSpanExample, ...],
        tuple[state_span_data.StateSpanExample, ...],
        dict[str, object],
    ],
) -> None:
    _legacy, _fit, original_calibration, _manifest = replay_sources
    output = replay._records_bytes(mixture.calibration)
    assert output == replay._records_bytes(original_calibration)
    assert hashlib.sha256(output).hexdigest() == (
        "ce0562ccb44ee83963eace0d873773addaee8e49f29499a6b720b16335930e70"
    )
    assert mixture.manifest["partitions"]["calibration"] == {
        "namespace": "train-calibration",
        "records": 800,
        "worlds": 200,
        "ordered_records_sha256": replay.H4_CALIBRATION_RECORDS_SHA256,
        "reused_unchanged_from_h4": True,
        "gradient_bearing": False,
    }


def test_authenticated_loader_round_trip(
    mixture: replay.ReplayMixture, tmp_path: Path
) -> None:
    _write_bundle(tmp_path, mixture)
    bundle = replay.load_replay_mixture_dataset(tmp_path)
    assert bundle.fit == mixture.fit
    assert bundle.calibration == mixture.calibration
    assert bundle.manifest == mixture.manifest
    assert bundle.manifest_sha256 == bundle.input_sha256["manifest"]
    assert set(bundle.input_sha256) == {"manifest", "fit", "calibration"}


def test_loader_rejects_any_replay_index_outside_frozen_h3_fit_range(
    mixture: replay.ReplayMixture, tmp_path: Path
) -> None:
    first_legacy = next(
        record
        for record in mixture.fit
        if record.world_id.startswith("train-world-replay-legacy-")
    )
    mutated = first_legacy.to_dict()
    mutated["world_id"] = "train-world-replay-legacy-2800"
    mutated["example_id"] = f"train-replay-legacy-2800-{first_legacy.variant}"
    fit_payload = replay._records_bytes(mixture.fit)
    original_line = state_span_data.canonical_json_bytes(first_legacy.to_dict())
    mutated_line = state_span_data.canonical_json_bytes(mutated)
    (tmp_path / "fit.jsonl").write_bytes(
        fit_payload.replace(original_line, mutated_line, 1)
    )
    (tmp_path / "calibration.jsonl").write_bytes(
        replay._records_bytes(mixture.calibration)
    )
    (tmp_path / "manifest.json").write_bytes(
        state_span_data.canonical_json_bytes(mixture.manifest)
    )
    with pytest.raises(TrainingInputError, match="index boundary"):
        replay.load_replay_mixture_dataset(tmp_path)
