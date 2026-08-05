"""Adversarial contracts for private corpus inventory and scope diagnostics.

All inputs and outputs in this module live below ``tmp_path``.  The inventory
artifact is intentionally private: CLI receipts and exceptions must never echo
document IDs, task values, query text, document bodies, or absolute paths.
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest

import wedge_v1.private_output as private_output
import wedge_v1.study_inventory as inventory_module
from wedge_v1.cli import main as cli_main
from wedge_v1.study_capture import MAX_PACK_BYTES, OWNER_PRIVATE, TASK_PACK_SCHEMA
from wedge_v1.study_inventory import StudyInventoryError, create_study_inventory


INVENTORY_SCHEMA = "nano-lm.wedge_v1.study_inventory.v1"
RECEIPT_SCHEMA = "nano-lm.wedge_v1.study_inventory_receipt.v1"
PROPOSAL_RULE = "EXACT_UNIQUE_BASENAME"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _task(
    index: int,
    *,
    doc_ids: list[str],
    query: str | None = None,
    mode: str = "ask",
) -> dict[str, Any]:
    return {
        "id": f"task-{index}",
        "mode": mode,
        "query": query or f"private question {index}",
        "doc_ids": doc_ids,
        "expect_status": ["SUPPORTED"],
        "manual_baseline_seconds": 30 + index,
    }


def _write_pack(path: Path, tasks: list[dict[str, Any]]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": TASK_PACK_SCHEMA,
                "storage_class": OWNER_PRIVATE,
                "tasks": tasks,
            }
        ),
        encoding="utf-8",
    )


def _assert_inventory_error(call, *args, **kwargs) -> StudyInventoryError:
    with pytest.raises(StudyInventoryError) as caught:
        call(*args, **kwargs)
    error = caught.value
    assert isinstance(error.code, str)
    assert error.code
    assert error.status in {"REJECTED", "INDETERMINATE"}
    return error


def _proposal_rows(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    diagnostics = artifact.get("task_diagnostics")
    assert isinstance(diagnostics, dict)
    tasks = diagnostics.get("tasks")
    assert isinstance(tasks, list)
    rows: list[dict[str, Any]] = []
    for task in tasks:
        assert isinstance(task, dict)
        proposals = task.get("scope_proposals")
        assert isinstance(proposals, list)
        assert all(isinstance(row, dict) for row in proposals)
        rows.extend(proposals)
    return rows


def _aggregate_proposal_rows(
    artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    diagnostics = artifact.get("task_diagnostics")
    assert isinstance(diagnostics, dict)
    proposals = diagnostics.get("scope_proposals")
    assert isinstance(proposals, list)
    assert all(isinstance(row, dict) for row in proposals)
    return proposals


def test_inventory_records_supported_documents_without_contents(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "zeta.md").write_text("PRIVATE_BODY_ZETA", encoding="utf-8")
    nested = corpus / "nested"
    nested.mkdir()
    (nested / "alpha.txt").write_text("PRIVATE_BODY_ALPHA", encoding="utf-8")
    output = tmp_path / "inventory.json"

    receipt = create_study_inventory(corpus, output)

    artifact = _read_json(output)
    assert artifact["schema"] == INVENTORY_SCHEMA
    assert artifact["storage_class"] == OWNER_PRIVATE
    assert receipt["schema"] == RECEIPT_SCHEMA
    assert receipt["status"] == "CREATED"
    assert receipt["code"] == "INVENTORY_READY"
    assert receipt["n_valid_documents"] == 2
    assert receipt["tasks_checked"] is False
    documents = artifact["documents"]
    assert [row["doc_id"] for row in documents] == ["nested/alpha", "zeta"]
    assert all(row["readable"] is True for row in documents)
    assert {row["format"] for row in documents} == {"markdown", "text"}
    serialized = output.read_text(encoding="utf-8")
    assert "PRIVATE_BODY_ALPHA" not in serialized
    assert "PRIVATE_BODY_ZETA" not in serialized
    assert str(corpus.resolve()) not in serialized
    assert artifact["task_diagnostics"] is None


def test_optional_task_diagnostics_propose_unique_nested_identity(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    source_dir = corpus / "collection"
    source_dir.mkdir(parents=True)
    (source_dir / "source.md").write_text("PRIVATE_DOCUMENT_BODY", encoding="utf-8")
    tasks = tmp_path / "tasks.json"
    query_marker = "PRIVATE_QUERY_MARKER_ZXQ"
    _write_pack(tasks, [_task(0, doc_ids=["source"], query=query_marker)])
    before = tasks.read_bytes()
    output = tmp_path / "inventory.json"

    receipt = create_study_inventory(corpus, output, tasks=tasks)

    artifact = _read_json(output)
    diagnostics = artifact["task_diagnostics"]
    assert diagnostics["aggregate"]["n_tasks"] == 1
    assert diagnostics["tasks"][0]["unknown_doc_ids"] == ["source"]
    assert diagnostics["aggregate"]["n_scope_proposals"] == 1
    assert _proposal_rows(artifact) == [
        {
            "from_doc_id": "source",
            "rule": PROPOSAL_RULE,
            "to_doc_id": "collection/source",
        }
    ]
    assert _aggregate_proposal_rows(artifact) == _proposal_rows(artifact)
    assert receipt["status"] == "CREATED"
    assert receipt["code"] == "TASK_SCOPE_ISSUES"
    assert receipt["tasks_checked"] is True
    assert receipt["n_scope_proposals"] == 1
    assert query_marker not in output.read_text(encoding="utf-8")
    assert "PRIVATE_DOCUMENT_BODY" not in output.read_text(encoding="utf-8")
    assert tasks.read_bytes() == before


def test_scope_proposals_are_exact_case_sensitive_unique_and_never_applied(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    for directory in ("nested", "left", "right"):
        (corpus / directory).mkdir(parents=True, exist_ok=True)
    (corpus / "nested" / "unique.md").write_text("one", encoding="utf-8")
    (corpus / "nested" / "casesensitive.md").write_text("two", encoding="utf-8")
    (corpus / "left" / "shared.md").write_text("three", encoding="utf-8")
    (corpus / "right" / "shared.txt").write_text("four", encoding="utf-8")
    tasks = tmp_path / "tasks.json"
    _write_pack(
        tasks,
        [
            _task(0, doc_ids=["unique"]),
            _task(1, doc_ids=["shared"]),
            _task(2, doc_ids=["CaseSensitive"]),
        ],
    )
    before = tasks.read_bytes()
    output = tmp_path / "inventory.json"

    create_study_inventory(corpus, output, tasks=tasks)

    artifact = _read_json(output)
    assert _proposal_rows(artifact) == [
        {
            "from_doc_id": "unique",
            "rule": PROPOSAL_RULE,
            "to_doc_id": "nested/unique",
        }
    ]
    assert _aggregate_proposal_rows(artifact) == _proposal_rows(artifact)
    diagnostics = artifact["task_diagnostics"]
    assert sum(len(row["unknown_doc_ids"]) for row in diagnostics["tasks"]) == 3
    assert diagnostics["aggregate"]["n_scope_proposals"] == 1
    assert diagnostics["aggregate"]["n_ambiguous_scope_references"] == 1
    assert diagnostics["aggregate"]["n_unmatched_scope_references"] == 1
    assert tasks.read_bytes() == before


def test_cli_receipt_and_private_artifact_do_not_disclose_sensitive_values(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path_marker = "PRIVATE_ABSOLUTE_PATH_MARKER_ZXQ"
    doc_marker = "PRIVATE_DOC_ID_MARKER_ZXQ"
    query_marker = "PRIVATE_QUERY_MARKER_ZXQ"
    body_marker = "PRIVATE_BODY_MARKER_ZXQ"
    task_marker = "PRIVATE_TASK_ID_MARKER_ZXQ"
    output_marker = "PRIVATE_OUTPUT_PATH_MARKER_ZXQ"
    workspace = tmp_path / path_marker
    corpus = workspace / "corpus"
    corpus.mkdir(parents=True)
    (corpus / f"{doc_marker}.md").write_text(body_marker, encoding="utf-8")
    tasks = workspace / "tasks.json"
    row = _task(0, doc_ids=[doc_marker], query=query_marker)
    row["id"] = task_marker
    _write_pack(tasks, [row])
    output = workspace / f"{output_marker}.json"

    rc = cli_main(
        [
            "study",
            "inventory",
            "--corpus",
            str(corpus),
            "--out",
            str(output),
            "--tasks",
            str(tasks),
        ]
    )

    captured = capsys.readouterr()
    receipt = json.loads(captured.out)
    assert rc == 0
    assert captured.err == ""
    assert receipt["schema"] == RECEIPT_SCHEMA
    assert receipt["status"] == "CREATED"
    assert receipt["code"] == "INVENTORY_READY"
    for marker in (
        path_marker,
        doc_marker,
        query_marker,
        body_marker,
        task_marker,
        output_marker,
    ):
        assert marker not in captured.out
        assert marker not in captured.err
    artifact_text = output.read_text(encoding="utf-8")
    assert doc_marker in artifact_text
    assert task_marker in artifact_text
    assert query_marker not in artifact_text
    assert body_marker not in artifact_text
    assert path_marker not in artifact_text
    assert str(corpus.resolve()) not in artifact_text
    assert str(tasks.resolve()) not in artifact_text
    assert str(output.resolve()) not in artifact_text


def test_inventory_classifies_unsupported_and_unreadable_or_empty_files(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "readable.md").write_text("readable text", encoding="utf-8")
    (corpus / "empty.txt").write_bytes(b"")
    unsupported_marker = "UNSUPPORTED_PRIVATE_BODY_ZXQ"
    (corpus / "artifact.json").write_text(unsupported_marker, encoding="utf-8")
    output = tmp_path / "inventory.json"

    receipt = create_study_inventory(corpus, output)

    artifact = _read_json(output)
    by_id = {row["doc_id"]: row for row in artifact["documents"]}
    assert by_id["readable"]["readable"] is True
    assert by_id["empty"]["readable"] is False
    assert by_id["empty"]["extracted_text_bytes"] == 0
    assert artifact["unsupported_files"] == [
        {
            "codes": ["UNSUPPORTED_FORMAT"],
            "format": "unsupported",
            "relative_path": "artifact.json",
        }
    ]
    assert receipt["n_valid_documents"] == 1
    assert receipt["n_unreadable_documents"] == 1
    assert receipt["n_unsupported_files"] == 1
    assert receipt["status"] == "CREATED"
    assert receipt["code"] == "CORPUS_CLEANUP_REQUIRED"
    assert unsupported_marker not in output.read_text(encoding="utf-8")


def test_visible_fifo_is_rejected_content_free_without_being_opened(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFOs are not supported on this platform")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text("body", encoding="utf-8")
    fifo_marker = "PRIVATE_FIFO_MARKER_ZXQ"
    fifo = corpus / fifo_marker
    os.mkfifo(fifo)
    output = tmp_path / "inventory.json"
    real_open = inventory_module.os.open

    def reject_fifo_open(path: object, *args: object, **kwargs: object) -> int:
        if Path(path) == fifo:
            pytest.fail("inventory attempted to open a visible FIFO")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(inventory_module.os, "open", reject_fifo_open)

    rc = cli_main(
        [
            "study",
            "inventory",
            "--corpus",
            str(corpus),
            "--out",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    receipt = json.loads(captured.out)
    assert rc == 2
    assert captured.err == ""
    assert receipt == {
        "schema": RECEIPT_SCHEMA,
        "status": "REJECTED",
        "code": "CORPUS_ENTRY_NOT_REGULAR",
    }
    assert fifo_marker not in captured.out
    assert not output.exists()


def test_empty_canonical_task_pack_publishes_capture_next_action(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text("body", encoding="utf-8")
    tasks = tmp_path / "tasks.json"
    _write_pack(tasks, [])
    output = tmp_path / "inventory.json"

    receipt = create_study_inventory(corpus, output, tasks=tasks)

    assert receipt["status"] == "CREATED"
    assert receipt["code"] == "TASK_PACK_EMPTY"
    assert receipt["next_action"] == "CAPTURE_GENUINE_TASKS"
    assert receipt["tasks_checked"] is True
    artifact = _read_json(output)
    diagnostics = artifact["task_diagnostics"]
    assert diagnostics["aggregate"]["n_tasks"] == 0
    assert diagnostics["tasks"] == []
    assert diagnostics["scope_proposals"] == []


def test_inventory_never_clobbers_existing_output(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text("body", encoding="utf-8")
    output = tmp_path / "inventory.json"
    original = b"OWNER CONTROLLED BYTES"
    output.write_bytes(original)

    error = _assert_inventory_error(create_study_inventory, corpus, output)

    assert error.code == "INVENTORY_ALREADY_EXISTS"
    assert output.read_bytes() == original


def test_inventory_rejects_commit_visible_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    public = repository / "public"
    public.mkdir(parents=True)
    private_root = repository / "wedge_v1" / ".private_exports"
    monkeypatch.setattr(private_output, "REPO_ROOT", repository)
    monkeypatch.setattr(
        private_output, "PRIVATE_EXPORT_ROOT", private_root, raising=False
    )
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text("body", encoding="utf-8")
    output = public / "inventory.json"

    error = _assert_inventory_error(create_study_inventory, corpus, output)

    assert error.code == "UNSAFE_INVENTORY_PATH"
    assert not output.exists()


def test_inventory_rejects_symlink_leaf_without_touching_target(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text("body", encoding="utf-8")
    target = tmp_path / "target.json"
    target.write_bytes(b"TARGET BYTES")
    output = tmp_path / "inventory.json"
    output.symlink_to(target)

    _assert_inventory_error(create_study_inventory, corpus, output)

    assert output.is_symlink()
    assert target.read_bytes() == b"TARGET BYTES"


def test_inventory_rejects_symlinked_parent(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text("body", encoding="utf-8")
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    alias = tmp_path / "parent-alias"
    alias.symlink_to(real_parent, target_is_directory=True)
    output = alias / "inventory.json"

    _assert_inventory_error(create_study_inventory, corpus, output)

    assert not (real_parent / "inventory.json").exists()


def test_inventory_file_is_owner_only(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text("body", encoding="utf-8")
    output = tmp_path / "inventory.json"

    create_study_inventory(corpus, output)

    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_inventory_auto_creates_only_the_canonical_private_export_root_as_0700(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    (repository / "wedge_v1").mkdir(parents=True)
    private_root = repository / "wedge_v1" / ".private_exports"
    monkeypatch.setattr(private_output, "REPO_ROOT", repository)
    monkeypatch.setattr(
        private_output, "PRIVATE_EXPORT_ROOT", private_root, raising=False
    )
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text("body", encoding="utf-8")
    output = private_root / "inventory.json"

    create_study_inventory(corpus, output)

    assert private_root.is_dir()
    assert stat.S_IMODE(private_root.stat().st_mode) == 0o700
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_inventory_serialization_and_order_are_deterministic(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    for directory in ("z-dir", "a-dir"):
        (corpus / directory).mkdir(parents=True, exist_ok=True)
    (corpus / "z-dir" / "z.md").write_text("z", encoding="utf-8")
    (corpus / "a-dir" / "b.md").write_text("b", encoding="utf-8")
    (corpus / "a-dir" / "a.txt").write_text("a", encoding="utf-8")
    (corpus / "z-unsupported.json").write_text("{}", encoding="utf-8")
    (corpus / "a-unsupported.csv").write_text("a,b", encoding="utf-8")
    tasks = tmp_path / "tasks.json"
    _write_pack(
        tasks,
        [
            _task(1, doc_ids=["z"]),
            _task(0, doc_ids=["a"]),
        ],
    )
    first = tmp_path / "inventory-first.json"
    second = tmp_path / "inventory-second.json"

    first_receipt = create_study_inventory(corpus, first, tasks=tasks)
    second_receipt = create_study_inventory(corpus, second, tasks=tasks)

    assert first.read_bytes() == second.read_bytes()
    assert first_receipt == second_receipt
    artifact = _read_json(first)
    assert [row["doc_id"] for row in artifact["documents"]] == [
        "a-dir/a",
        "a-dir/b",
        "z-dir/z",
    ]
    assert [row["relative_path"] for row in artifact["unsupported_files"]] == [
        "a-unsupported.csv",
        "z-unsupported.json",
    ]
    assert _proposal_rows(artifact) == [
        {"from_doc_id": "z", "rule": PROPOSAL_RULE, "to_doc_id": "z-dir/z"},
        {"from_doc_id": "a", "rule": PROPOSAL_RULE, "to_doc_id": "a-dir/a"},
    ]
    assert _aggregate_proposal_rows(artifact) == [
        {"from_doc_id": "a", "rule": PROPOSAL_RULE, "to_doc_id": "a-dir/a"},
        {"from_doc_id": "z", "rule": PROPOSAL_RULE, "to_doc_id": "z-dir/z"},
    ]


@pytest.mark.parametrize(
    ("payload", "expected_code"),
    [
        (b"{", "TASK_PACK_INVALID_JSON"),
        (b"[]", "TASK_PACK_INVALID_ROOT"),
        (b'{"tasks": []}', "TASK_PACK_SCHEMA_UNSUPPORTED"),
        (
            (
                '{"schema":"%s","schema":"%s",'
                '"storage_class":"%s","tasks":[]}'
                % (TASK_PACK_SCHEMA, TASK_PACK_SCHEMA, OWNER_PRIVATE)
            ).encode("utf-8"),
            "TASK_PACK_DUPLICATE_JSON_KEYS",
        ),
        (
            json.dumps(
                {
                    "schema": TASK_PACK_SCHEMA,
                    "storage_class": "PUBLIC",
                    "tasks": [],
                }
            ).encode("utf-8"),
            "TASK_PACK_STORAGE_CLASS_INVALID",
        ),
    ],
    ids=[
        "invalid-json",
        "wrong-root",
        "unversioned",
        "duplicate-key",
        "wrong-storage-class",
    ],
)
def test_inventory_rejects_malformed_task_packs_without_output(
    tmp_path: Path, payload: bytes, expected_code: str
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text("body", encoding="utf-8")
    tasks = tmp_path / "tasks.json"
    tasks.write_bytes(payload)
    before = tasks.read_bytes()
    output = tmp_path / "inventory.json"

    error = _assert_inventory_error(
        create_study_inventory, corpus, output, tasks=tasks
    )

    assert error.status == "REJECTED"
    assert error.code == expected_code
    assert output.exists() is False
    assert tasks.read_bytes() == before
    assert not list(tmp_path.glob(".study-inventory-*.tmp"))


def test_inventory_publishes_semantic_task_pack_diagnostics(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text("body", encoding="utf-8")
    tasks = tmp_path / "tasks.json"
    tasks.write_text(
        json.dumps(
            {
                "schema": TASK_PACK_SCHEMA,
                "storage_class": OWNER_PRIVATE,
                "tasks": [
                    {
                        "id": "task-without-baseline",
                        "mode": "ask",
                        "query": "private question",
                        "doc_ids": ["note"],
                        "expect_status": ["SUPPORTED"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    before = tasks.read_bytes()
    output = tmp_path / "inventory.json"

    receipt = create_study_inventory(corpus, output, tasks=tasks)

    assert receipt["status"] == "CREATED"
    assert receipt["code"] == "TASK_INPUTS_REQUIRE_REVIEW"
    assert receipt["tasks_checked"] is True
    artifact = _read_json(output)
    diagnostics = artifact["task_diagnostics"]
    assert diagnostics["tasks"][0]["scope_status"] == "VALID"
    assert diagnostics["tasks"][0]["codes"] == [
        "TASK_MANUAL_BASELINE_INVALID"
    ]
    assert tasks.read_bytes() == before
    assert not list(tmp_path.glob(".study-inventory-*.tmp"))


def test_inventory_rejects_oversized_task_pack_without_reading_past_limit(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text("body", encoding="utf-8")
    tasks = tmp_path / "tasks.json"
    tasks.write_bytes(b" " * (MAX_PACK_BYTES + 1))
    output = tmp_path / "inventory.json"

    error = _assert_inventory_error(
        create_study_inventory, corpus, output, tasks=tasks
    )

    assert error.code == "TASK_PACK_TOO_LARGE"
    assert not output.exists()


def test_publication_failure_leaves_no_output_or_temporary_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text("body", encoding="utf-8")
    export = tmp_path / "export"
    export.mkdir()
    output = export / "inventory.json"

    def fail_link(*_args, **_kwargs):
        raise OSError("injected publication failure")

    monkeypatch.setattr(inventory_module.os, "link", fail_link)

    error = _assert_inventory_error(create_study_inventory, corpus, output)

    assert error.code == "INVENTORY_PUBLISH_FAILED"
    assert not output.exists()
    assert list(export.iterdir()) == []


def test_directory_fsync_failure_is_indeterminate_after_atomic_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text("body", encoding="utf-8")
    output = tmp_path / "inventory.json"

    real_fsync = inventory_module.os.fsync

    def fail_directory_sync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("injected directory fsync failure")
        real_fsync(fd)

    monkeypatch.setattr(inventory_module.os, "fsync", fail_directory_sync)

    error = _assert_inventory_error(create_study_inventory, corpus, output)

    assert error.code == "INVENTORY_DURABILITY_UNCERTAIN"
    assert error.status == "INDETERMINATE"
    assert output.is_file()
    assert _read_json(output)["schema"] == INVENTORY_SCHEMA
    assert not list(tmp_path.glob(".study-inventory-*.tmp"))
