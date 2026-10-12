from __future__ import annotations

import ast
import copy
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

import pytest

from nano_ai.benchmarks import fresh_v1
from nano_ai.contract import FIELD_ORDER

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# These seals cover canonical bytes generated from the source under test.  An
# intentional generator edit must explicitly refresh both values after review.
EXPECTED_PARTITION_SHA256 = (
    "aa45ecf91f5014876d4ca20097cf4a958b5592e47bfed59fbb5f8e64a9331e55"
)
EXPECTED_MANIFEST_SHA256 = (
    "5fdd5b3aed14dc55b4070fce9223de24bda45cbb857400b306d2e4b7fe9f3759"
)


@pytest.fixture(scope="module")
def partition_document() -> dict[str, object]:
    return fresh_v1.build_fresh_partition_document(REPOSITORY_ROOT)


def _write_sealed(
    root: Path, document: dict[str, object]
) -> tuple[Path, str, dict[str, object]]:
    root.mkdir(parents=True, exist_ok=False)
    partition_bytes = fresh_v1.canonical_json_bytes(document)
    partition_path = root / "partition.json"
    partition_path.write_bytes(partition_bytes)
    manifest = fresh_v1.build_sealed_manifest(
        document,
        partition_path=partition_path.name,
        repository_root=REPOSITORY_ROOT,
    )
    manifest_path = root / "manifest.json"
    manifest_bytes = fresh_v1.canonical_json_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    return manifest_path, hashlib.sha256(manifest_bytes).hexdigest(), manifest


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def test_exact_regeneration_counts_quotas_and_pinned_hashes(
    partition_document: dict[str, object],
) -> None:
    regenerated = fresh_v1.build_fresh_partition_document(REPOSITORY_ROOT)
    first_bytes = fresh_v1.canonical_json_bytes(partition_document)
    assert fresh_v1.canonical_json_bytes(regenerated) == first_bytes
    assert hashlib.sha256(first_bytes).hexdigest() == EXPECTED_PARTITION_SHA256

    cases = partition_document["cases"]
    assert isinstance(cases, list)
    assert len(cases) == 220
    assert sum(len(row["gold"]["fields"]) for row in cases) == 1100
    assert len({row["transcript"] for row in cases}) == 220
    assert len({_normalized(row["transcript"]) for row in cases}) == 220

    families = Counter(row["benchmark"]["family"] for row in cases)
    assert families == {"normal": 160, "state_challenge": 60}
    normal_bands = Counter(
        row["benchmark"]["value_band"]
        for row in cases
        if row["benchmark"]["family"] == "normal"
    )
    assert normal_bands == {"pool_a": 80, "pool_b": 80}
    challenges = [
        row for row in cases if row["benchmark"]["family"] == "state_challenge"
    ]
    assert Counter(row["benchmark"]["target_state"] for row in challenges) == {
        "missing": 20,
        "uncertain": 20,
        "conflicting": 20,
    }
    assert Counter(
        (
            row["benchmark"]["target_state"],
            row["benchmark"]["target_field"],
            row["benchmark"]["value_band"],
        )
        for row in challenges
    ) == {
        (state, field.value, band): 2
        for state in ("missing", "uncertain", "conflicting")
        for field in FIELD_ORDER
        for band in fresh_v1.VALUE_BANDS
    }

    manifest = fresh_v1.build_sealed_manifest(
        partition_document,
        partition_path="partition.json",
        repository_root=REPOSITORY_ROOT,
    )
    assert (
        hashlib.sha256(fresh_v1.canonical_json_bytes(manifest)).hexdigest()
        == EXPECTED_MANIFEST_SHA256
    )


def test_manifest_seals_sources_data_and_candidate_blind_protocol(
    partition_document: dict[str, object],
) -> None:
    manifest = fresh_v1.build_sealed_manifest(
        partition_document,
        partition_path="partition.json",
        repository_root=REPOSITORY_ROOT,
    )
    generator_bytes = (REPOSITORY_ROOT / fresh_v1.GENERATOR_PATH).read_bytes()
    native_bytes = (REPOSITORY_ROOT / fresh_v1.NATIVE_SOURCE_PATH).read_bytes()
    partition_bytes = fresh_v1.canonical_json_bytes(partition_document)

    assert manifest["generator"] == {
        "path": fresh_v1.GENERATOR_PATH,
        "sha256": hashlib.sha256(generator_bytes).hexdigest(),
    }
    assert manifest["native_source"] == {
        "path": fresh_v1.NATIVE_SOURCE_PATH,
        "sha256": hashlib.sha256(native_bytes).hexdigest(),
    }
    assert manifest["partition"] == {
        "path": "partition.json",
        "records": 220,
        "fields": 1100,
        "sha256": hashlib.sha256(partition_bytes).hexdigest(),
    }
    assert manifest["status"] == "sealed_unmeasured"
    assert manifest["protocol"] == {
        "purpose": "final_confirmation_only",
        "candidate_blind_generation": True,
        "candidate_inputs_used": False,
        "result_inputs_used": False,
        "inference_performed": False,
    }
    assert manifest["challenge_quota"] == {
        "states": ["missing", "uncertain", "conflicting"],
        "fields": [field.value for field in FIELD_ORDER],
        "value_bands": ["pool_a", "pool_b"],
        "per_state_field_band": 2,
        "total": 60,
    }


def test_defined_values_and_templates_are_native_disjoint_without_imports() -> None:
    audit = fresh_v1.audit_native_surface_disjointness(REPOSITORY_ROOT)
    assert not any(audit.value_overlaps.values())
    assert not audit.template_overlaps
    assert audit.manifest_dict()["native_records_imported"] is False
    assert audit.manifest_dict()["value_pools_disjoint"] is True
    assert audit.manifest_dict()["surface_templates_disjoint"] is True

    native = fresh_v1.load_native_surface_inventory(REPOSITORY_ROOT)
    fresh_pools = fresh_v1.fresh_value_pools()
    for field in FIELD_ORDER:
        assert fresh_pools["pool_a"][field].isdisjoint(fresh_pools["pool_b"][field])
        assert (fresh_pools["pool_a"][field] | fresh_pools["pool_b"][field]).isdisjoint(
            native.values[field]
        )

    generator_source = (REPOSITORY_ROOT / fresh_v1.GENERATOR_PATH).read_text(
        encoding="utf-8"
    )
    tree = ast.parse(generator_source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert "nano_ai.training.state_span_data" not in imported_modules
    assert imported_modules.isdisjoint(
        {
            "nano_ai.benchmark_runner",
            "nano_ai.benchmarking",
            "nano_ai.evaluation",
        }
    )
    assert not any(
        forbidden in module.casefold()
        for module in imported_modules
        for forbidden in ("candidate", "checkpoint", "comparison", "result")
    )


def test_sealed_loader_keeps_evaluator_truth_out_of_requests_and_checks_offsets(
    tmp_path: Path,
    partition_document: dict[str, object],
) -> None:
    manifest_path, seal, _ = _write_sealed(tmp_path / "sealed", partition_document)
    frozen = fresh_v1.load_sealed_confirmation_partition(
        manifest_path,
        expected_manifest_sha256=seal,
        repository_root=REPOSITORY_ROOT,
    )

    assert frozen.benchmark_id == fresh_v1.BENCHMARK_ID
    assert frozen.status == "sealed_unmeasured"
    assert frozen.purpose == "final_confirmation_only"
    assert len(frozen.cases) == 220
    for case in frozen.cases:
        request = case.request.to_dict()
        assert set(request) == {"schema_version", "item_id", "transcript"}
        assert "gold" not in request
        assert "benchmark" not in request
        for field in case.gold.fields:
            for span in field.evidence:
                assert case.request.transcript[span.start : span.end] == span.text


def test_no_clobber_writer_and_cli_surface(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "fresh-v1"
    assert (
        fresh_v1.main(
            [
                "--repository-root",
                str(REPOSITORY_ROOT),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["purpose"] == "final_confirmation_only"
    assert payload["manifest_path"] == str(output / "manifest.json")
    assert payload["partition_path"] == str(output / "partition.json")
    original_manifest = (output / "manifest.json").read_bytes()
    original_partition = (output / "partition.json").read_bytes()

    with pytest.raises(FileExistsError, match="already exists"):
        fresh_v1.write_fresh_confirmation_partition(
            output, repository_root=REPOSITORY_ROOT
        )
    assert (output / "manifest.json").read_bytes() == original_manifest
    assert (output / "partition.json").read_bytes() == original_partition


def test_seals_and_strict_structure_fail_closed(
    tmp_path: Path,
    partition_document: dict[str, object],
) -> None:
    manifest_path, seal, _ = _write_sealed(tmp_path / "sealed", partition_document)
    with pytest.raises(fresh_v1.ConfirmationIntegrityError, match="manifest seal"):
        fresh_v1.load_sealed_confirmation_partition(
            manifest_path,
            expected_manifest_sha256="0" * 64,
            repository_root=REPOSITORY_ROOT,
        )

    (manifest_path.parent / "partition.json").write_bytes(
        (manifest_path.parent / "partition.json").read_bytes() + b" "
    )
    with pytest.raises(fresh_v1.ConfirmationIntegrityError, match="partition seal"):
        fresh_v1.load_sealed_confirmation_partition(
            manifest_path,
            expected_manifest_sha256=seal,
            repository_root=REPOSITORY_ROOT,
        )

    tampered = copy.deepcopy(partition_document)
    target = next(
        row
        for row in tampered["cases"]
        if row["benchmark"]["family"] == "state_challenge"
    )
    target["benchmark"]["target_field"] = next(
        field.value
        for field in FIELD_ORDER
        if field.value != target["benchmark"]["target_field"]
    )
    with pytest.raises(
        fresh_v1.ConfirmationIntegrityError, match="target state disagrees"
    ):
        fresh_v1.validate_partition_document(tampered, repository_root=REPOSITORY_ROOT)

    bad_offset = copy.deepcopy(partition_document)
    evidence = next(
        field["evidence"][0]
        for field in bad_offset["cases"][0]["gold"]["fields"]
        if field["evidence"]
    )
    evidence["start"] += 1
    with pytest.raises(fresh_v1.ConfirmationIntegrityError, match="invalid case"):
        fresh_v1.validate_partition_document(
            bad_offset, repository_root=REPOSITORY_ROOT
        )


def test_duplicate_json_keys_are_rejected_before_schema_validation(
    tmp_path: Path,
) -> None:
    raw = b'{"schema_version":"x","schema_version":"x"}'
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(raw)
    with pytest.raises(fresh_v1.ConfirmationIntegrityError, match="duplicate JSON key"):
        fresh_v1.load_sealed_confirmation_partition(
            manifest_path,
            expected_manifest_sha256=hashlib.sha256(raw).hexdigest(),
            repository_root=REPOSITORY_ROOT,
        )
