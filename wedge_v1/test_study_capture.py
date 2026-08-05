"""Contract tests for private study-task capture.

Every filesystem target in this module lives under ``tmp_path``.  Containment
tests redirect the repository roots into that temporary tree so a broken
implementation cannot touch real owner state.
"""
from __future__ import annotations

import io
import json
import multiprocessing
from pathlib import Path
from queue import Empty
from typing import Any

import pytest

import wedge_v1.private_output as private_output
import wedge_v1.study_capture as capture_module
from wedge_v1.cli import main as cli_main
from wedge_v1.study import assess_inputs
from wedge_v1.study_capture import (
    MAX_PACK_BYTES,
    MAX_QUERY_BYTES,
    MAX_TASKS,
    OWNER_PRIVATE,
    TASK_PACK_SCHEMA,
    TaskCaptureError,
    append_task,
    initialize_task_pack,
    read_private_query,
)


VALID_STATUSES = ["SUPPORTED", "CONTRADICTED", "ABSTAIN"]


def _read_pack(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _task_kwargs(index: int = 0, **overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "task_id": f"task-{index}",
        "mode": "ask",
        "query": f"private question {index}",
        "doc_ids": [f"document-{index}"],
        "expect_status": ["SUPPORTED"],
        "manual_baseline_seconds": 30 + index,
    }
    values.update(overrides)
    return values


def _assert_capture_error(call, *args, **kwargs) -> TaskCaptureError:
    with pytest.raises(TaskCaptureError) as caught:
        call(*args, **kwargs)
    error = caught.value
    assert isinstance(error.code, str)
    assert error.code
    return error


def _append_process(
    path: str,
    index: int,
    ready: Any,
    start: Any,
    results: Any,
) -> None:
    """Start one append at the same time as the other spawned workers."""
    from wedge_v1.study_capture import append_task as append_in_child

    ready.put(index)
    if not start.wait(timeout=10):
        results.put((index, "start-timeout"))
        return
    try:
        append_in_child(Path(path), **_task_kwargs(index))
    except Exception as exc:  # pragma: no cover - reported in the parent
        results.put((index, f"{type(exc).__name__}:{exc}"))
    else:
        results.put((index, "ok"))


def test_initialize_creates_a_canonical_empty_pack(tmp_path: Path) -> None:
    path = tmp_path / "tasks.json"

    initialize_task_pack(path)

    assert _read_pack(path) == {
        "schema": TASK_PACK_SCHEMA,
        "storage_class": OWNER_PRIVATE,
        "tasks": [],
    }


@pytest.mark.parametrize("initial", [b"", b"{", b"already here"])
def test_initialize_never_clobbers_an_existing_entry(
    tmp_path: Path, initial: bytes
) -> None:
    path = tmp_path / "tasks.json"
    path.write_bytes(initial)

    _assert_capture_error(initialize_task_pack, path)

    assert path.read_bytes() == initial


def test_initialize_never_replaces_an_existing_directory(tmp_path: Path) -> None:
    path = tmp_path / "tasks.json"
    path.mkdir()

    _assert_capture_error(initialize_task_pack, path)

    assert path.is_dir()


@pytest.mark.parametrize(
    "raw_pack",
    [
        b"{",
        b"[]",
        b'{"tasks": []}',
        json.dumps(
            {
                "schema": "not-the-capture-schema",
                "storage_class": OWNER_PRIVATE,
                "tasks": [],
            }
        ).encode("utf-8"),
        json.dumps(
            {
                "schema": TASK_PACK_SCHEMA,
                "storage_class": "PUBLIC",
                "tasks": [],
            }
        ).encode("utf-8"),
        json.dumps(
            {
                "schema": TASK_PACK_SCHEMA,
                "storage_class": OWNER_PRIVATE,
                "tasks": {},
            }
        ).encode("utf-8"),
        json.dumps(
            {
                "schema": TASK_PACK_SCHEMA,
                "storage_class": OWNER_PRIVATE,
                "tasks": [{"id": "incomplete-row"}],
            }
        ).encode("utf-8"),
        (
            '{"schema":"%s","schema":"%s",'
            '"storage_class":"%s","tasks":[]}'
            % (TASK_PACK_SCHEMA, TASK_PACK_SCHEMA, OWNER_PRIVATE)
        ).encode("utf-8"),
        (
            '{"schema":"%s","storage_class":"%s",'
            '"tasks":[],"invalid_number":NaN}'
            % (TASK_PACK_SCHEMA, OWNER_PRIVATE)
        ).encode("utf-8"),
    ],
    ids=[
        "invalid-json",
        "wrong-root",
        "legacy-unmarked",
        "wrong-schema",
        "wrong-storage-class",
        "tasks-not-list",
        "invalid-existing-row",
        "duplicate-json-key",
        "non-finite-json-number",
    ],
)
def test_append_rejects_noncanonical_or_malformed_packs_without_mutation(
    tmp_path: Path, raw_pack: bytes
) -> None:
    path = tmp_path / "tasks.json"
    path.write_bytes(raw_pack)

    _assert_capture_error(append_task, path, **_task_kwargs())

    assert path.read_bytes() == raw_pack


def test_append_canonicalizes_the_task_definition(tmp_path: Path) -> None:
    path = tmp_path / "tasks.json"
    initialize_task_pack(path)

    append_task(
        path,
        task_id="  TASK-1  ",
        mode="  ASK  ",
        query="  What does the private record say?  ",
        doc_ids=[" doc-b ", "doc-a", "doc-b"],
        expect_status=VALID_STATUSES,
        manual_baseline_seconds=15.5,
    )

    assert _read_pack(path)["tasks"] == [
        {
            "id": "TASK-1",
            "mode": "ask",
            "query": "What does the private record say?",
            "doc_ids": ["doc-a", "doc-b"],
            "expect_status": VALID_STATUSES,
            "manual_baseline_seconds": 15.5,
        }
    ]


def test_duplicate_id_and_canonical_definition_are_distinct_errors(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tasks.json"
    initialize_task_pack(path)
    append_task(
        path,
        **_task_kwargs(
            task_id="task-a",
            query="same private question",
            doc_ids=["doc-b", "doc-a"],
        ),
    )

    before = path.read_bytes()
    duplicate_id = _assert_capture_error(
        append_task,
        path,
        **_task_kwargs(
            2,
            task_id=" task-a ",
            query="different private question",
        ),
    )
    assert path.read_bytes() == before

    duplicate_definition = _assert_capture_error(
        append_task,
        path,
        **_task_kwargs(
            3,
            task_id="task-b",
            mode="ASK",
            query=" same private question ",
            doc_ids=[" doc-a ", "doc-b", "doc-a"],
        ),
    )
    assert path.read_bytes() == before
    assert duplicate_id.code != duplicate_definition.code


def test_compare_requires_two_unique_documents(tmp_path: Path) -> None:
    path = tmp_path / "tasks.json"
    initialize_task_pack(path)
    before = path.read_bytes()

    _assert_capture_error(
        append_task,
        path,
        **_task_kwargs(
            mode="compare",
            doc_ids=["same-document", " same-document "],
        ),
    )

    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "expect_status",
    [[], ["NOT_A_STATUS"], ["SUPPORTED", ""], "SUPPORTED"],
)
def test_append_rejects_invalid_expected_statuses_without_mutation(
    tmp_path: Path, expect_status: Any
) -> None:
    path = tmp_path / "tasks.json"
    initialize_task_pack(path)
    before = path.read_bytes()

    _assert_capture_error(
        append_task,
        path,
        **_task_kwargs(expect_status=expect_status),
    )

    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "baseline",
    [None, True, False, 0, -1, float("nan"), float("inf"), float("-inf"), "30"],
)
def test_append_rejects_nonpositive_nonfinite_or_non_numeric_baselines(
    tmp_path: Path, baseline: Any
) -> None:
    path = tmp_path / "tasks.json"
    initialize_task_pack(path)
    before = path.read_bytes()

    _assert_capture_error(
        append_task,
        path,
        **_task_kwargs(manual_baseline_seconds=baseline),
    )

    assert path.read_bytes() == before


def test_append_enforces_the_task_limit_without_mutation(tmp_path: Path) -> None:
    path = tmp_path / "tasks.json"
    initialize_task_pack(path)
    for index in range(MAX_TASKS):
        append_task(path, **_task_kwargs(index))

    assert len(_read_pack(path)["tasks"]) == MAX_TASKS
    before = path.read_bytes()

    _assert_capture_error(append_task, path, **_task_kwargs(MAX_TASKS))

    assert path.read_bytes() == before


def test_append_preserves_unknown_fields_and_existing_order(tmp_path: Path) -> None:
    path = tmp_path / "tasks.json"
    initialize_task_pack(path)
    append_task(path, **_task_kwargs(0))
    payload = _read_pack(path)
    payload["owner_metadata"] = {"keep": [1, 2, 3]}
    payload["tasks"][0]["owner_annotation"] = {"keep": True}
    path.write_text(json.dumps(payload), encoding="utf-8")

    append_task(path, **_task_kwargs(1))

    captured = _read_pack(path)
    assert captured["owner_metadata"] == {"keep": [1, 2, 3]}
    assert captured["tasks"][0]["owner_annotation"] == {"keep": True}
    assert [row["id"] for row in captured["tasks"]] == ["task-0", "task-1"]


def test_initialize_rejects_a_dangling_symlink_without_following_it(
    tmp_path: Path,
) -> None:
    target = tmp_path / "missing-target.json"
    link = tmp_path / "tasks.json"
    link.symlink_to(target)

    _assert_capture_error(initialize_task_pack, link)

    assert link.is_symlink()
    assert not target.exists()


def test_append_rejects_a_symlink_leaf_without_mutating_its_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "real-tasks.json"
    initialize_task_pack(target)
    before = target.read_bytes()
    link = tmp_path / "tasks.json"
    link.symlink_to(target)

    _assert_capture_error(append_task, link, **_task_kwargs())

    assert target.read_bytes() == before
    assert link.is_symlink()


def test_task_path_policy_allows_only_the_redirected_private_repo_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repository"
    private_root = repo / "wedge_v1" / "data" / "owner_tasks"
    public_root = repo / "public"
    private_root.mkdir(parents=True)
    public_root.mkdir(parents=True)
    monkeypatch.setattr(private_output, "REPO_ROOT", repo)
    monkeypatch.setattr(private_output, "PRIVATE_TASK_ROOT", private_root)

    unsafe = public_root / "tasks.json"
    _assert_capture_error(initialize_task_pack, unsafe)
    assert not unsafe.exists()

    allowed = private_root / "tasks.json"
    initialize_task_pack(allowed)
    assert allowed.is_file()


def test_task_path_policy_rejects_a_symlinked_parent_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repository"
    private_root = repo / "wedge_v1" / "data" / "owner_tasks"
    public_root = repo / "public"
    private_root.mkdir(parents=True)
    public_root.mkdir(parents=True)
    monkeypatch.setattr(private_output, "REPO_ROOT", repo)
    monkeypatch.setattr(private_output, "PRIVATE_TASK_ROOT", private_root)
    escaped_parent = private_root / "escape"
    escaped_parent.symlink_to(public_root, target_is_directory=True)
    escaped_target = escaped_parent / "tasks.json"

    _assert_capture_error(initialize_task_pack, escaped_target)

    assert not (public_root / "tasks.json").exists()


def test_capture_errors_and_output_do_not_echo_private_values(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path_marker = "PRIVATE_PATH_MARKER_ZXQ"
    id_marker = "PRIVATE_ID_MARKER_ZXQ"
    query_marker = "PRIVATE_QUERY_MARKER_ZXQ"
    doc_marker = "PRIVATE_DOC_MARKER_ZXQ"
    path = tmp_path / f"{path_marker}.json"
    initialize_task_pack(path)
    append_task(
        path,
        **_task_kwargs(
            task_id=id_marker,
            query=query_marker,
            doc_ids=[doc_marker],
        ),
    )

    error = _assert_capture_error(
        append_task,
        path,
        **_task_kwargs(
            2,
            task_id=id_marker,
            query=f"second-{query_marker}",
            doc_ids=[f"second-{doc_marker}"],
        ),
    )

    message = str(error)
    for marker in (path_marker, id_marker, query_marker, doc_marker):
        assert marker not in message
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == ""


def test_cli_reads_query_from_stdin_and_emits_content_free_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path_marker = "PRIVATE_PATH_MARKER_ZXQ"
    id_marker = "PRIVATE_ID_MARKER_ZXQ"
    query_marker = "PRIVATE_QUERY_MARKER_ZXQ"
    doc_marker = "PRIVATE_DOC_MARKER_ZXQ"
    path = tmp_path / f"{path_marker}.json"
    assert cli_main(["study", "init", "--tasks", str(path)]) == 0
    capsys.readouterr()
    argv = [
        "study",
        "add",
        "--tasks",
        str(path),
        "--id",
        id_marker,
        "--mode",
        "ask",
        "--doc",
        doc_marker,
        "--expect-status",
        "SUPPORTED",
        "--manual-baseline-seconds",
        "30",
    ]
    assert query_marker not in argv
    monkeypatch.setattr("sys.stdin", io.StringIO(query_marker))

    assert cli_main(argv) == 0

    output = capsys.readouterr()
    payload = json.loads(output.out)
    assert output.err == ""
    assert payload["status"] == "CAPTURED"
    assert payload["representative_ready"] is False
    for marker in (path_marker, id_marker, query_marker, doc_marker):
        assert marker not in output.out
        assert marker not in output.err
    assert _read_pack(path)["tasks"][0]["query"] == query_marker


def test_private_query_file_is_read_without_echoing_content(tmp_path: Path) -> None:
    marker = "PRIVATE_QUERY_FILE_MARKER_ZXQ"
    source = tmp_path / "query.txt"
    source.write_text(marker, encoding="utf-8")

    assert read_private_query(query_file=source, stdin=io.StringIO("ignored")) == marker


def test_publish_failure_leaves_original_pack_and_no_query_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = "PRIVATE_QUERY_TEMP_MARKER_ZXQ"
    path = tmp_path / "tasks.json"
    initialize_task_pack(path)
    before = path.read_bytes()

    def fail_replace(*_args, **_kwargs):
        raise OSError("injected")

    monkeypatch.setattr(capture_module.os, "replace", fail_replace)
    error = _assert_capture_error(
        append_task,
        path,
        **_task_kwargs(query=marker),
    )

    assert error.code == "TASK_PACK_PUBLISH_FAILED"
    assert marker not in str(error)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".study-task-pack-*.tmp"))


def test_append_rejects_an_oversized_result_before_publication(tmp_path: Path) -> None:
    path = tmp_path / "tasks.json"
    initialize_task_pack(path)
    payload = _read_pack(path)
    payload["padding"] = "x" * (MAX_PACK_BYTES - 2_000)
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = path.read_bytes()
    assert len(before) < MAX_PACK_BYTES

    error = _assert_capture_error(
        append_task,
        path,
        **_task_kwargs(query="q" * 4_000),
    )

    assert error.code == "TASK_PACK_TOO_LARGE"
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".study-task-pack-*.tmp"))


def test_direct_append_enforces_the_query_byte_limit(tmp_path: Path) -> None:
    path = tmp_path / "tasks.json"
    initialize_task_pack(path)
    before = path.read_bytes()

    error = _assert_capture_error(
        append_task,
        path,
        **_task_kwargs(query="q" * (MAX_QUERY_BYTES + 1)),
    )

    assert error.code == "TASK_QUERY_TOO_LARGE"
    assert path.read_bytes() == before


def test_query_file_enforces_the_byte_limit(tmp_path: Path) -> None:
    source = tmp_path / "query.txt"
    source.write_bytes(b"q" * (MAX_QUERY_BYTES + 1))

    error = _assert_capture_error(
        read_private_query,
        query_file=source,
        stdin=io.StringIO("ignored"),
    )

    assert error.code == "TASK_QUERY_TOO_LARGE"


def _raise_durability_uncertain(_parent_fd: int) -> None:
    raise TaskCaptureError(
        "TASK_PACK_DURABILITY_UNCERTAIN", status="INDETERMINATE"
    )


def test_cli_init_reports_post_publication_sync_failure_as_indeterminate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "tasks.json"
    monkeypatch.setattr(capture_module, "_fsync_parent", _raise_durability_uncertain)

    rc = cli_main(["study", "init", "--tasks", str(path)])

    output = capsys.readouterr()
    payload = json.loads(output.out)
    assert rc == 2
    assert output.err == ""
    assert payload == {
        "schema": "nano-lm.wedge_v1.study_capture.v1",
        "status": "INDETERMINATE",
        "code": "TASK_PACK_DURABILITY_UNCERTAIN",
        "next_action": "INSPECT_TASK_PACK_BEFORE_RETRY",
    }
    assert path.is_file()


def test_cli_add_reports_post_publication_sync_failure_as_indeterminate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "tasks.json"
    initialize_task_pack(path)
    monkeypatch.setattr(capture_module, "_fsync_parent", _raise_durability_uncertain)
    monkeypatch.setattr("sys.stdin", io.StringIO("private query"))

    rc = cli_main(
        [
            "study",
            "add",
            "--tasks",
            str(path),
            "--id",
            "task-1",
            "--mode",
            "ask",
            "--doc",
            "document-1",
            "--expect-status",
            "SUPPORTED",
            "--manual-baseline-seconds",
            "30",
        ]
    )

    output = capsys.readouterr()
    payload = json.loads(output.out)
    assert rc == 2
    assert output.err == ""
    assert payload["status"] == "INDETERMINATE"
    assert payload["code"] == "TASK_PACK_DURABILITY_UNCERTAIN"
    assert payload["next_action"] == "INSPECT_TASK_PACK_BEFORE_RETRY"
    assert len(_read_pack(path)["tasks"]) == 1


def test_distinct_concurrent_appends_are_both_retained(tmp_path: Path) -> None:
    path = tmp_path / "tasks.json"
    initialize_task_pack(path)
    context = multiprocessing.get_context("spawn")
    ready = context.Queue()
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_append_process,
            args=(str(path), index, ready, start, results),
        )
        for index in (1, 2)
    ]
    for process in processes:
        process.start()
    try:
        for _ in processes:
            ready.get(timeout=10)
        start.set()
        for process in processes:
            process.join(timeout=15)
        if any(process.is_alive() for process in processes):
            pytest.fail("concurrent capture worker did not terminate")
        outcomes = [results.get(timeout=5) for _ in processes]
    except Empty as exc:  # pragma: no cover - diagnostic for process failures
        pytest.fail(f"concurrent capture worker returned no result: {exc}")
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    assert all(process.exitcode == 0 for process in processes)
    assert sorted(outcomes) == [(1, "ok"), (2, "ok")]
    assert {row["id"] for row in _read_pack(path)["tasks"]} == {
        "task-1",
        "task-2",
    }


def _build_representative_pack(
    path: Path, *, include_unknown_scope: bool = False
) -> None:
    initialize_task_pack(path)
    for index in range(10):
        doc_ids = [f"private-note-{index}"]
        if include_unknown_scope and index == 9:
            doc_ids.append("unknown-private-note")
        append_task(
            path,
            **_task_kwargs(
                index,
                mode="recall" if index == 0 else "ask",
                doc_ids=doc_ids,
            ),
        )


def test_generated_pack_is_accepted_by_study_check_and_unknown_scope_blocks(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for index in range(10):
        (corpus / f"private-note-{index}.md").write_text(
            f"Private note {index} contains a distinct useful record.\n",
            encoding="utf-8",
        )
    accepted = tmp_path / "accepted-tasks.json"
    _build_representative_pack(accepted)

    accepted_report = assess_inputs(corpus, accepted)

    assert accepted_report["representative_ready"] is True
    assert accepted_report["blockers"] == []
    assert accepted_report["tasks"]["n_tasks"] == 10

    unknown = tmp_path / "unknown-scope-tasks.json"
    _build_representative_pack(unknown, include_unknown_scope=True)

    unknown_report = assess_inputs(corpus, unknown)

    assert unknown_report["representative_ready"] is False
    assert unknown_report["blockers"] == ["UNKNOWN_TASK_SCOPE_REFERENCES"]
