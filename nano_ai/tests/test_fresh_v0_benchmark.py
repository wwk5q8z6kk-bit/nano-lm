from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from nano_ai.adapters.deterministic_v0 import DeterministicV0Solver
from nano_ai.benchmarking import (
    BENCHMARK_CASE_SCHEMA_VERSION,
    aggregate_benchmark_report,
)
from nano_ai.benchmarks import fresh_v0
from nano_ai.contract import FIELD_ORDER
from nano_ai.evaluation import evaluate_solver

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def collision_index() -> fresh_v0.CollisionIndex:
    return fresh_v0.build_collision_index(REPOSITORY_ROOT)


@pytest.fixture(scope="module")
def synthetic_document(
    collision_index: fresh_v0.CollisionIndex,
) -> dict[str, object]:
    definitions = fresh_v0.load_v1_definitions(REPOSITORY_ROOT)
    # The production seed is deliberately not exercised by tests.  The sealed
    # schema still carries the preregistered seed because this is a temporary
    # structural fixture, not the real fresh partition.
    records = fresh_v0._generate_records(
        definitions,
        collision_index,
        seed=17,
    )
    return fresh_v0._partition_document_from_records(records)


def _write_sealed(
    root: Path,
    document: dict[str, object],
) -> tuple[Path, str]:
    root.mkdir(parents=True, exist_ok=True)
    partition_path = root / "partition.json"
    partition_path.write_bytes(fresh_v0.canonical_json_bytes(document))
    manifest = fresh_v0.build_sealed_manifest(
        document,
        partition_path=partition_path.name,
        repository_root=REPOSITORY_ROOT,
    )
    manifest_path = root / "manifest.json"
    manifest_bytes = fresh_v0.canonical_json_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    return manifest_path, hashlib.sha256(manifest_bytes).hexdigest()


def test_pinned_history_and_literal_only_generator(
    collision_index: fresh_v0.CollisionIndex,
    tmp_path: Path,
) -> None:
    assert collision_index.record_count == 8217
    assert collision_index.unique_transcript_count == 6689
    assert collision_index.exact_multiset_sha256 == (
        "6231146627fa6232e6d4d7e098452f752347123de4ede1eada688d98185bc07f"
    )
    assert collision_index.normalized_multiset_sha256 == (
        "d285aa393cf1388340a644df3ac486fdc17618973efd040b6bb418de2185d566"
    )
    assert collision_index.regenerated_v1_train_count == 8000

    source = (REPOSITORY_ROOT / fresh_v0.GENERATOR_PATH).read_bytes()
    modified = source + b'\nraise RuntimeError("must never execute")\n'
    generator = tmp_path / fresh_v0.GENERATOR_PATH
    generator.parent.mkdir(parents=True)
    generator.write_bytes(modified)
    definitions = fresh_v0.load_v1_definitions(
        tmp_path,
        expected_sha256=hashlib.sha256(modified).hexdigest(),
    )
    assert definitions["CC_HELD"][0] == ("a toothache", "toothache")


def test_sealed_loader_returns_private_inputs_and_frozen_snapshots(
    tmp_path: Path,
    synthetic_document: dict[str, object],
) -> None:
    manifest_path, seal = _write_sealed(tmp_path, synthetic_document)
    benchmark = fresh_v0.load_sealed_benchmark(
        manifest_path,
        expected_manifest_sha256=seal,
        repository_root=REPOSITORY_ROOT,
    )

    assert benchmark.benchmark_id == fresh_v0.BENCHMARK_ID
    assert benchmark.status == "fresh_unmeasured"
    assert benchmark.manifest_sha256 == seal
    assert (
        benchmark.partition_sha256
        == hashlib.sha256((tmp_path / "partition.json").read_bytes()).hexdigest()
    )
    assert len(benchmark.cases) == 220
    assert all(
        set(case.request.to_dict()) == {"schema_version", "item_id", "transcript"}
        for case in benchmark.cases
    )
    assert all(
        case.provenance["benchmark"]["schema_version"] == BENCHMARK_CASE_SCHEMA_VERSION
        for case in benchmark.cases
    )

    families = Counter(
        case.provenance["benchmark"]["family"] for case in benchmark.cases
    )
    assert families == {"normal": 160, "state_challenge": 60}
    challenges = [
        case
        for case in benchmark.cases
        if case.provenance["benchmark"]["family"] == "state_challenge"
    ]
    assert Counter(
        case.provenance["benchmark"]["target_state"] for case in challenges
    ) == {"missing": 20, "uncertain": 20, "conflicting": 20}
    assert Counter(
        (
            case.provenance["benchmark"]["target_state"],
            case.provenance["benchmark"]["target_field"],
            case.provenance["benchmark"]["value_band"],
        )
        for case in challenges
    ) == {
        (state, field.value, band): 2
        for state in ("missing", "uncertain", "conflicting")
        for field in FIELD_ORDER
        for band in ("seen", "held")
    }

    evaluation = evaluate_solver(DeterministicV0Solver(), benchmark.cases)
    report = aggregate_benchmark_report(benchmark.cases, evaluation)
    assert evaluation.quality["grounded_exact_field_count"] == 1100
    assert report.primary["grounded_exact_field_accuracy"] == 1.0


def test_manifest_and_partition_seals_fail_closed(
    tmp_path: Path,
    synthetic_document: dict[str, object],
) -> None:
    manifest_path, seal = _write_sealed(tmp_path, synthetic_document)
    with pytest.raises(fresh_v0.BenchmarkIntegrityError, match="manifest seal"):
        fresh_v0.load_sealed_benchmark(
            manifest_path,
            expected_manifest_sha256="0" * 64,
            repository_root=REPOSITORY_ROOT,
        )

    (tmp_path / "partition.json").write_bytes(
        (tmp_path / "partition.json").read_bytes() + b" "
    )
    with pytest.raises(fresh_v0.BenchmarkIntegrityError, match="partition seal"):
        fresh_v0.load_sealed_benchmark(
            manifest_path,
            expected_manifest_sha256=seal,
            repository_root=REPOSITORY_ROOT,
        )


def test_resealed_target_tamper_reaches_structural_validation(
    tmp_path: Path,
    synthetic_document: dict[str, object],
) -> None:
    document = json.loads(json.dumps(synthetic_document))
    challenge = next(
        row
        for row in document["cases"]
        if row["benchmark"]["family"] == "state_challenge"
    )
    challenge["benchmark"]["target_field"] = next(
        field.value
        for field in FIELD_ORDER
        if field.value != challenge["benchmark"]["target_field"]
    )
    manifest_path, _ = _write_sealed(tmp_path, synthetic_document)
    partition_bytes = fresh_v0.canonical_json_bytes(document)
    (tmp_path / "partition.json").write_bytes(partition_bytes)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["partition"]["sha256"] = hashlib.sha256(partition_bytes).hexdigest()
    manifest_bytes = fresh_v0.canonical_json_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    seal = hashlib.sha256(manifest_bytes).hexdigest()
    with pytest.raises(
        fresh_v0.BenchmarkIntegrityError, match="target state disagrees"
    ):
        fresh_v0.load_sealed_benchmark(
            manifest_path,
            expected_manifest_sha256=seal,
            repository_root=REPOSITORY_ROOT,
        )


def test_duplicate_json_keys_are_rejected_before_schema_validation(
    tmp_path: Path,
) -> None:
    raw = b'{"schema_version":"x","schema_version":"x"}'
    path = tmp_path / "manifest.json"
    path.write_bytes(raw)
    with pytest.raises(fresh_v0.BenchmarkIntegrityError, match="duplicate JSON key"):
        fresh_v0.load_sealed_benchmark(
            path,
            expected_manifest_sha256=hashlib.sha256(raw).hexdigest(),
            repository_root=REPOSITORY_ROOT,
        )


def test_source_hash_change_is_rejected(tmp_path: Path) -> None:
    generator = tmp_path / fresh_v0.GENERATOR_PATH
    generator.parent.mkdir(parents=True)
    generator.write_text("CC_TRAIN = []\n", encoding="utf-8")
    with pytest.raises(fresh_v0.BenchmarkIntegrityError, match="hash mismatch"):
        fresh_v0.load_v1_definitions(tmp_path)
