from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import nano_ai.fixtures as fixtures_module
from nano_ai.contract import FieldName, FieldState, NanoInput
from nano_ai.fixtures import (
    CONTRACT_SMOKE_SHA256,
    FixtureCase,
    FixtureIntegrityError,
    load_contract_smoke,
    load_manifest,
    verify_repository_partitions,
)

_SUPPORTED_TRANSCRIPT = (
    "Doctor: Good morning, what brings you in today?\n"
    "Patient: I've been having shortness of breath.\n"
    "Doctor: How long has this been going on?\n"
    "Patient: Coming up on 3 weeks now.\n"
    "Doctor: How bad would you say it is?\n"
    "Patient: I'd call it moderate.\n"
    "Doctor: Have you taken anything for it?\n"
    "Patient: I've been taking zinc tablets.\n"
    "Doctor: Any allergies I should know about?\n"
    "Patient: I'm allergic to penicillin."
)
_ABSENCE_TRANSCRIPT = (
    "Doctor: Good morning, what brings you in today?\n"
    "Patient: I've been having shortness of breath.\n"
    "Doctor: How long has this been going on?\n"
    "Patient: Coming up on 8 days now.\n"
    "Doctor: How bad would you say it is?\n"
    "Patient: I'd call it mild.\n"
    "Doctor: Have you taken anything for it?\n"
    "Patient: No, nothing yet.\n"
    "Doctor: Any allergies I should know about?\n"
    "Patient: No allergies."
)


def _write_manifest(tmp_path: Path, manifest: dict[str, object]) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_manifest_locks_historical_roles_without_claiming_fresh_test() -> None:
    manifest = load_manifest()

    assert manifest["partitions"]["rstar_train"] == {
        "path": "trajectory/e4/data/rstar_train.json",
        "records": 800,
        "sha256": ("0c0065c3f01d93a7850d93ab7cb29de8a100a386191401beaf04652ca84e4d13"),
        "role": "historical_training",
        "exposure": "previously_generated_and_used",
    }
    assert manifest["partitions"]["rstar_dev"]["role"] == "historical_validation"
    assert manifest["partitions"]["rstar_eval"] == {
        "path": "trajectory/e4/data/rstar_eval.json",
        "records": 220,
        "sha256": ("37e96ff94bff41d25e5b06f22b4d941fc7a2c6067540148e88022779e8fc2ff7"),
        "role": "historical_regression",
        "exposure": "fully_measured_before_nano_v0",
    }
    assert manifest["scientific_status"]["fresh_test_partition"] is None
    assert manifest["scientific_status"]["historical_eval_is_fresh"] is False


def test_contract_smoke_is_self_contained_and_preserves_historical_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_repository_access(*args: object, **kwargs: object) -> None:
        pytest.fail("load_contract_smoke accessed the repository checkout")

    monkeypatch.setattr(fixtures_module, "_repo_path", unexpected_repository_access)
    cases = load_contract_smoke()

    assert [case.request.transcript for case in cases[:2]] == [
        _SUPPORTED_TRANSCRIPT,
        _ABSENCE_TRANSCRIPT,
    ]
    assert cases[0].provenance["historical_source"] == {
        "partition": "rstar_dev",
        "index": 0,
        "record_sha256": (
            "1d4baa5ba45d2828d5b3d069e1c984b7722eac2207688f0f61bb21c8e3066ae0"
        ),
    }
    assert cases[1].provenance["historical_source"]["index"] == 4


def test_contract_smoke_is_valid_and_truth_is_not_in_input() -> None:
    cases = load_contract_smoke()

    assert len(cases) == 3
    assert set(cases[0].request.to_dict()) == {
        "schema_version",
        "item_id",
        "transcript",
    }
    for case in cases:
        case.gold.validate_against(case.request)


def test_contract_smoke_covers_absence_and_all_abstention_states() -> None:
    cases = load_contract_smoke()
    by_id = {case.case_id: case for case in cases}

    absence = {
        field.field: field.state for field in by_id["rstar-dev-0004"].gold.fields
    }
    assert absence[FieldName.MEDICATION] is FieldState.ABSENT
    assert absence[FieldName.ALLERGY] is FieldState.ABSENT

    edge = {
        field.field: field.state for field in by_id["contract-edge-0000"].gold.fields
    }
    assert edge[FieldName.DURATION] is FieldState.UNCERTAIN
    assert edge[FieldName.MEDICATION] is FieldState.MISSING
    assert edge[FieldName.ALLERGY] is FieldState.CONFLICTING


def test_partition_relabeling_fails_closed(tmp_path: Path) -> None:
    manifest = load_manifest()
    manifest["partitions"]["rstar_eval"]["role"] = "fresh_test"
    path = _write_manifest(tmp_path, manifest)

    with pytest.raises(FixtureIntegrityError, match="frozen policy: rstar_eval"):
        load_manifest(path)


def test_historical_eval_cannot_be_reclassified_as_fresh(tmp_path: Path) -> None:
    manifest = load_manifest()
    manifest["scientific_status"]["historical_eval_is_fresh"] = True
    path = _write_manifest(tmp_path, manifest)

    with pytest.raises(FixtureIntegrityError, match="scientific status"):
        load_manifest(path)


def test_extra_gold_annotation_key_fails_closed(tmp_path: Path) -> None:
    manifest = load_manifest()
    manifest["contract_smoke"][0]["gold"][0]["rationale"] = "not allowed"
    path = _write_manifest(tmp_path, manifest)

    with pytest.raises(FixtureIntegrityError, match="gold annotation"):
        load_manifest(path)


def test_contract_smoke_digest_is_pinned_outside_manifest(tmp_path: Path) -> None:
    manifest = load_manifest()
    manifest["contract_smoke"][0]["gold"][0]["value"] = "the shortness of breath"
    path = _write_manifest(tmp_path, manifest)

    assert len(CONTRACT_SMOKE_SHA256) == 64
    with pytest.raises(FixtureIntegrityError, match="contract smoke digest mismatch"):
        load_manifest(path)


def test_historical_snapshot_index_must_fit_declared_partition(tmp_path: Path) -> None:
    manifest = load_manifest()
    manifest["contract_smoke"][0]["source"]["historical_source"]["index"] = 100
    path = _write_manifest(tmp_path, manifest)

    with pytest.raises(FixtureIntegrityError, match="historical record index"):
        load_manifest(path)


def test_fixture_case_rejects_gold_that_does_not_match_request() -> None:
    case = load_contract_smoke()[0]
    unrelated = NanoInput(item_id=case.case_id, transcript="Patient: unrelated text.")

    with pytest.raises(FixtureIntegrityError, match="gold violates"):
        FixtureCase(
            case_id=case.case_id,
            partition=case.partition,
            request=unrelated,
            gold=case.gold,
            provenance={},
        )


def test_repository_partition_verification_checks_frozen_sources_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_reader = fixtures_module._read_bytes_snapshot
    partition_reads: list[str] = []

    def tracked_reader(path: Path, *, description: str) -> bytes:
        if description.startswith("repository partition "):
            partition_reads.append(description)
        return real_reader(path, description=description)

    monkeypatch.setattr(fixtures_module, "_read_bytes_snapshot", tracked_reader)
    verified = verify_repository_partitions()

    assert set(verified) == {"rstar_train", "rstar_dev", "rstar_eval"}
    assert verified["rstar_dev"]["records"] == 100
    assert partition_reads == [
        "repository partition rstar_train",
        "repository partition rstar_dev",
        "repository partition rstar_eval",
    ]


def test_repository_partition_snapshot_tampering_fails_before_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_reader = fixtures_module._read_bytes_snapshot

    def tampered_reader(path: Path, *, description: str) -> bytes:
        snapshot = real_reader(path, description=description)
        if description == "repository partition rstar_dev":
            return snapshot + b"\n"
        return snapshot

    monkeypatch.setattr(fixtures_module, "_read_bytes_snapshot", tampered_reader)
    with pytest.raises(
        FixtureIntegrityError, match="partition hash mismatch: rstar_dev"
    ):
        verify_repository_partitions()


def test_contract_smoke_loads_from_simulated_installed_package(tmp_path: Path) -> None:
    site_packages = tmp_path / "site-packages"
    package_copy = site_packages / "nano_ai"
    shutil.copytree(
        Path(fixtures_module.__file__).resolve().parent,
        package_copy,
        ignore=shutil.ignore_patterns("__pycache__", "tests"),
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(site_packages)
    command = (
        "from pathlib import Path; "
        "import nano_ai.fixtures as f; "
        "cases=f.load_contract_smoke(); "
        "assert len(cases)==3; "
        "assert cases[0].provenance['historical_source']['partition']=='rstar_dev'; "
        "assert Path(f.__file__).resolve().is_relative_to(Path.cwd()/'site-packages')"
    )

    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
