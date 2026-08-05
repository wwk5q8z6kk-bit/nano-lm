"""Contracts for confirmed, copy-on-write private task-scope repair.

Every path and private marker in this module is created below ``tmp_path``.
The repair may replace only stale document IDs covered by the inventory's
complete exact-unique-basename proposal set.  It must never mutate the source
pack or treat scope repair as evidence that the study is ready.
"""
from __future__ import annotations

import copy
import json
import os
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import wedge_v1.study_scope_repair as repair_module
from wedge_v1.cli import main as cli_main
from wedge_v1.study_capture import OWNER_PRIVATE, TASK_PACK_SCHEMA
from wedge_v1.study_inventory import create_study_inventory
from wedge_v1.study_scope_repair import (
    SCOPE_REPAIR_RECEIPT_SCHEMA,
    ScopeRepairError,
    create_scope_repaired_pack,
)

PROPOSAL_RULE = "EXACT_UNIQUE_BASENAME"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _task(
    index: int,
    *,
    doc_ids: list[str],
    include_baseline: bool = True,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": f"PRIVATE_TASK_ID_{index}_ZXQ",
        "mode": "compare" if len(set(doc_ids)) >= 2 else "ask",
        "query": f"PRIVATE_QUERY_{index}_ZXQ",
        "doc_ids": doc_ids,
        "expect_status": ["SUPPORTED"],
        "owner_annotation": {"preserve": [index, True]},
    }
    if include_baseline:
        row["manual_baseline_seconds"] = 30 + index
    return row


def _write_pack(path: Path, tasks: list[dict[str, Any]]) -> None:
    _write_json(
        path,
        {
            "schema": TASK_PACK_SCHEMA,
            "storage_class": OWNER_PRIVATE,
            "owner_metadata": {"preserve": {"nested": True}},
            "tasks": tasks,
        },
    )


def _make_case(
    tmp_path: Path,
    *,
    include_baseline: bool = True,
    path_marker: str = "PRIVATE_PATH_MARKER_ZXQ",
) -> dict[str, Path]:
    workspace = tmp_path / path_marker
    corpus = workspace / "corpus"
    nested = corpus / "current-collection"
    nested.mkdir(parents=True)
    (nested / "alpha.md").write_text(
        "PRIVATE_DOCUMENT_ALPHA_BODY_ZXQ", encoding="utf-8"
    )
    (nested / "beta.md").write_text(
        "PRIVATE_DOCUMENT_BETA_BODY_ZXQ", encoding="utf-8"
    )
    other = corpus / "other-collection"
    other.mkdir()
    (other / "gamma.md").write_text(
        "PRIVATE_DOCUMENT_GAMMA_BODY_ZXQ", encoding="utf-8"
    )

    tasks = workspace / "PRIVATE_SOURCE_TASKS_ZXQ.json"
    _write_pack(
        tasks,
        [
            _task(
                0,
                doc_ids=["alpha", "beta"],
                include_baseline=include_baseline,
            )
        ],
    )
    inventory = workspace / "PRIVATE_INVENTORY_ZXQ.json"
    create_study_inventory(corpus, inventory, tasks=tasks)
    return {
        "workspace": workspace,
        "corpus": corpus,
        "tasks": tasks,
        "inventory": inventory,
        "output": workspace / "PRIVATE_REPAIRED_TASKS_ZXQ.json",
    }


def _assert_repair_error(
    call: Callable[..., object], *args: object, **kwargs: object
) -> ScopeRepairError:
    with pytest.raises(ScopeRepairError) as caught:
        call(*args, **kwargs)
    error = caught.value
    assert error.status in {"REJECTED", "INDETERMINATE"}
    assert isinstance(error.code, str)
    assert error.code
    return error


def _repair(case: dict[str, Path], *, confirmed: bool = True) -> dict[str, Any]:
    return create_scope_repaired_pack(
        case["corpus"],
        case["tasks"],
        case["inventory"],
        case["output"],
        confirmed=confirmed,
    )


def _cli_argv(case: dict[str, Path], *, confirmed: bool = True) -> list[str]:
    argv = [
        "study",
        "repair-scopes",
        "--corpus",
        str(case["corpus"]),
        "--tasks",
        str(case["tasks"]),
        "--inventory",
        str(case["inventory"]),
        "--out",
        str(case["output"]),
    ]
    if confirmed:
        argv.append("--confirm-all-exact-basename-proposals")
    return argv


def _assert_content_free(serialized: str) -> None:
    for marker in (
        "PRIVATE_PATH_MARKER_ZXQ",
        "PRIVATE_SOURCE_TASKS_ZXQ",
        "PRIVATE_INVENTORY_ZXQ",
        "PRIVATE_REPAIRED_TASKS_ZXQ",
        "PRIVATE_TASK_ID_0_ZXQ",
        "PRIVATE_QUERY_0_ZXQ",
        "PRIVATE_DOCUMENT_ALPHA_BODY_ZXQ",
        "PRIVATE_DOCUMENT_BETA_BODY_ZXQ",
        "PRIVATE_DOCUMENT_GAMMA_BODY_ZXQ",
        "alpha",
        "beta",
        "gamma",
        "current-collection",
        "other-collection",
    ):
        assert marker not in serialized


def test_repair_is_scope_only_copy_on_write_deterministic_and_owner_only(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    source_before = case["tasks"].read_bytes()
    source_value = _read_json(case["tasks"])
    expected = copy.deepcopy(source_value)
    expected["tasks"][0]["doc_ids"] = [
        "current-collection/alpha",
        "current-collection/beta",
    ]

    first_receipt = _repair(case)

    assert case["tasks"].read_bytes() == source_before
    assert _read_json(case["output"]) == expected
    assert stat.S_IMODE(case["output"].stat().st_mode) == 0o600
    assert first_receipt["schema"] == SCOPE_REPAIR_RECEIPT_SCHEMA
    assert first_receipt["status"] == "CREATED"
    assert first_receipt["code"] == "SCOPE_REPAIR_CREATED"
    assert first_receipt["n_tasks"] == 1
    assert first_receipt["n_tasks_changed"] == 1
    assert first_receipt["n_scope_references_repaired"] == 2
    assert first_receipt["n_unique_scope_mappings"] == 2
    assert first_receipt.get("representative_ready") is not True
    _assert_content_free(json.dumps(first_receipt, sort_keys=True))

    second_output = case["workspace"] / "second-repaired.json"
    second_receipt = create_scope_repaired_pack(
        case["corpus"],
        case["tasks"],
        case["inventory"],
        second_output,
        confirmed=True,
    )

    assert second_output.read_bytes() == case["output"].read_bytes()
    assert second_receipt == first_receipt
    assert case["tasks"].read_bytes() == source_before


def test_cli_success_requires_explicit_flag_and_emits_content_free_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case = _make_case(tmp_path)

    rc = cli_main(_cli_argv(case))

    captured = capsys.readouterr()
    receipt = json.loads(captured.out)
    assert rc == 0
    assert captured.err == ""
    assert receipt["schema"] == SCOPE_REPAIR_RECEIPT_SCHEMA
    assert receipt["status"] == "CREATED"
    assert receipt["code"] == "SCOPE_REPAIR_CREATED"
    _assert_content_free(captured.out)
    assert case["output"].is_file()


def test_confirmation_is_required_before_any_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case = _make_case(tmp_path)
    source_before = case["tasks"].read_bytes()

    error = _assert_repair_error(_repair, case, confirmed=False)

    assert error.code == "CONFIRMATION_REQUIRED"
    assert error.status == "REJECTED"
    assert not case["output"].exists()
    assert case["tasks"].read_bytes() == source_before

    rc = cli_main(_cli_argv(case, confirmed=False))
    captured = capsys.readouterr()
    receipt = json.loads(captured.out)
    assert rc == 2
    assert captured.err == ""
    assert receipt == {
        "schema": SCOPE_REPAIR_RECEIPT_SCHEMA,
        "status": "REJECTED",
        "code": "CONFIRMATION_REQUIRED",
    }
    _assert_content_free(captured.out)
    assert not case["output"].exists()


def test_repair_rejects_task_pack_changed_since_inventory(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    changed = _read_json(case["tasks"])
    changed["owner_metadata"]["post_inventory_change"] = True
    _write_json(case["tasks"], changed)
    changed_bytes = case["tasks"].read_bytes()

    error = _assert_repair_error(_repair, case)

    assert error.code == "INVENTORY_TASK_PACK_DIGEST_MISMATCH"
    assert not case["output"].exists()
    assert case["tasks"].read_bytes() == changed_bytes


def test_repair_rejects_corpus_changed_since_inventory(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    source_before = case["tasks"].read_bytes()
    target = case["corpus"] / "current-collection" / "alpha.md"
    target.write_text("POST_INVENTORY_CORPUS_CHANGE_ZXQ", encoding="utf-8")

    error = _assert_repair_error(_repair, case)

    assert error.code in {
        "INVENTORY_CORPUS_DIGEST_MISMATCH",
        "INVENTORY_CORPUS_IDS_MISMATCH",
    }
    assert not case["output"].exists()
    assert case["tasks"].read_bytes() == source_before


def test_repair_rejects_forged_proposal_even_when_target_exists(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    source_before = case["tasks"].read_bytes()
    artifact = _read_json(case["inventory"])
    diagnostics = artifact["task_diagnostics"]
    forged_target = "other-collection/gamma"
    for row in diagnostics["scope_proposals"]:
        if row["from_doc_id"] == "alpha":
            row["to_doc_id"] = forged_target
    for task in diagnostics["tasks"]:
        for row in task["scope_proposals"]:
            if row["from_doc_id"] == "alpha":
                row["to_doc_id"] = forged_target
    _write_json(case["inventory"], artifact)

    error = _assert_repair_error(_repair, case)

    assert error.code == "INVENTORY_PROPOSALS_INVALID"
    assert not case["output"].exists()
    assert case["tasks"].read_bytes() == source_before


def test_repair_is_all_or_nothing_when_inventory_proposals_are_partial(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    source_before = case["tasks"].read_bytes()
    artifact = _read_json(case["inventory"])
    diagnostics = artifact["task_diagnostics"]
    diagnostics["scope_proposals"] = [
        row
        for row in diagnostics["scope_proposals"]
        if row["from_doc_id"] != "beta"
    ]
    for task in diagnostics["tasks"]:
        task["scope_proposals"] = [
            row
            for row in task["scope_proposals"]
            if row["from_doc_id"] != "beta"
        ]
    diagnostics["aggregate"]["n_scope_proposals"] = 1
    diagnostics["aggregate"]["n_scope_proposal_occurrences"] = 1
    _write_json(case["inventory"], artifact)

    error = _assert_repair_error(_repair, case)

    assert error.code == "INVENTORY_PROPOSALS_STALE_OR_PARTIAL"
    assert not case["output"].exists()
    assert case["tasks"].read_bytes() == source_before


def test_repair_never_clobbers_an_existing_output(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    source_before = case["tasks"].read_bytes()
    original = b"OWNER_CONTROLLED_OUTPUT_BYTES_ZXQ"
    case["output"].write_bytes(original)

    error = _assert_repair_error(_repair, case)

    assert error.code == "REPAIR_OUTPUT_ALREADY_EXISTS"
    assert case["output"].read_bytes() == original
    assert case["tasks"].read_bytes() == source_before


def test_missing_manual_baseline_is_preserved_without_a_readiness_claim(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path, include_baseline=False)
    source_before = _read_json(case["tasks"])

    receipt = _repair(case)

    repaired = _read_json(case["output"])
    assert "manual_baseline_seconds" not in source_before["tasks"][0]
    assert "manual_baseline_seconds" not in repaired["tasks"][0]
    expected = copy.deepcopy(source_before)
    expected["tasks"][0]["doc_ids"] = [
        "current-collection/alpha",
        "current-collection/beta",
    ]
    assert repaired == expected
    assert receipt.get("representative_ready") is not True
    assert receipt.get("decision") != "PROCEED"


@pytest.mark.parametrize(
    "failure_kind",
    ["confirmation", "stale-tasks", "stale-corpus", "forged-inventory"],
)
def test_cli_failures_are_content_free(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    failure_kind: str,
) -> None:
    case = _make_case(tmp_path)
    confirmed = True
    if failure_kind == "confirmation":
        confirmed = False
    elif failure_kind == "stale-tasks":
        changed = _read_json(case["tasks"])
        changed["owner_metadata"]["PRIVATE_CHANGED_MARKER_ZXQ"] = True
        _write_json(case["tasks"], changed)
    elif failure_kind == "stale-corpus":
        (case["corpus"] / "current-collection" / "alpha.md").write_text(
            "PRIVATE_CHANGED_CORPUS_MARKER_ZXQ", encoding="utf-8"
        )
    else:
        artifact = _read_json(case["inventory"])
        for row in artifact["task_diagnostics"]["scope_proposals"]:
            row["rule"] = "PRIVATE_FORGED_RULE_MARKER_ZXQ"
        for task in artifact["task_diagnostics"]["tasks"]:
            for row in task["scope_proposals"]:
                row["rule"] = "PRIVATE_FORGED_RULE_MARKER_ZXQ"
        _write_json(case["inventory"], artifact)

    rc = cli_main(_cli_argv(case, confirmed=confirmed))

    captured = capsys.readouterr()
    receipt = json.loads(captured.out)
    assert rc == 2
    assert captured.err == ""
    assert receipt["schema"] == SCOPE_REPAIR_RECEIPT_SCHEMA
    assert receipt["status"] == "REJECTED"
    assert set(receipt) == {"schema", "status", "code"}
    _assert_content_free(captured.out)
    for marker in (
        "PRIVATE_CHANGED_MARKER_ZXQ",
        "PRIVATE_CHANGED_CORPUS_MARKER_ZXQ",
        "PRIVATE_FORGED_RULE_MARKER_ZXQ",
    ):
        assert marker not in captured.out
        assert marker not in captured.err
    assert not case["output"].exists()


def test_directory_fsync_failure_is_indeterminate_after_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _make_case(tmp_path)
    source_before = case["tasks"].read_bytes()
    real_fsync = repair_module.inventory_module.os.fsync

    def fail_directory_sync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("PRIVATE_FSYNC_FAILURE_MARKER_ZXQ")
        real_fsync(fd)

    monkeypatch.setattr(repair_module.inventory_module.os, "fsync", fail_directory_sync)

    error = _assert_repair_error(_repair, case)

    assert error.status == "INDETERMINATE"
    assert error.code == "REPAIR_OUTPUT_DURABILITY_UNCERTAIN"
    assert case["output"].is_file()
    assert _read_json(case["output"])["schema"] == TASK_PACK_SCHEMA
    assert case["tasks"].read_bytes() == source_before
    assert not list(case["workspace"].glob(".*.tmp"))
    assert "PRIVATE_FSYNC_FAILURE_MARKER_ZXQ" not in str(error)
