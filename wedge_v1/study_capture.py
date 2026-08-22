"""Private, atomic capture for representative-use study task packs.

The capture surface records authentic operator inputs; it never runs the solver
and never claims that a pack is representative-ready. ``study check`` remains
the authority for corpus IDs, coverage, and readiness.
"""
from __future__ import annotations

import errno
import fcntl
import json
import math
import os
import secrets
import stat
from pathlib import Path
from typing import Any, Iterable, TextIO

from wedge_v1 import private_output as private_paths
from wedge_v1.runtime import normalize_doc_ids


TASK_PACK_SCHEMA = "nano-lm.wedge_v1.study_tasks.v1"
CAPTURE_RESULT_SCHEMA = "nano-lm.wedge_v1.study_capture.v1"
OWNER_PRIVATE = "OWNER_PRIVATE"
ALLOWED_MODES = ("ask", "find", "compare", "recall")
ALLOWED_STATUSES = ("SUPPORTED", "CONTRADICTED", "ABSTAIN")
MAX_TASKS = 20
MAX_PACK_BYTES = 1_000_000
MAX_QUERY_BYTES = 64_000

_LOCK_NAME = ".study-task-pack.lock"
_TEMP_PREFIX = ".study-task-pack-"
_TEMP_SUFFIX = ".tmp"
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)


class TaskCaptureError(ValueError):
    """Machine-readable, content-free task-capture failure."""

    def __init__(self, code: str, *, status: str = "REJECTED"):
        self.code = code
        self.status = status
        super().__init__(code)


def _fail(code: str, *, status: str = "REJECTED") -> None:
    raise TaskCaptureError(code, status=status)


def _capture_path(path: Path) -> Path:
    try:
        return private_paths.require_private_task_pack(Path(path))
    except (OSError, ValueError):
        _fail("UNSAFE_TASK_PACK_PATH")


def _prepare_init_parent(target: Path) -> None:
    parent = target.parent
    if parent.exists():
        if not parent.is_dir() or parent.is_symlink():
            _fail("TASK_PACK_PARENT_INVALID")
        return

    owner_root = Path(private_paths.PRIVATE_TASK_ROOT).expanduser().resolve(strict=False)
    if parent != owner_root:
        _fail("TASK_PACK_PARENT_MISSING")
    try:
        parent.mkdir(mode=0o700, parents=False, exist_ok=False)
    except FileExistsError:
        if not parent.is_dir() or parent.is_symlink():
            _fail("TASK_PACK_PARENT_INVALID")
    except OSError:
        _fail("TASK_PACK_PARENT_CREATE_FAILED")


def _open_parent(target: Path) -> int:
    try:
        fd = os.open(target.parent, os.O_RDONLY | _DIRECTORY | _NOFOLLOW)
    except OSError:
        _fail("TASK_PACK_PARENT_INVALID")
    try:
        info = os.fstat(fd)
        if not stat.S_ISDIR(info.st_mode):
            _fail("TASK_PACK_PARENT_INVALID")
        return fd
    except Exception:
        os.close(fd)
        raise


def _parent_is_current(target: Path, parent_fd: int) -> bool:
    try:
        live = os.stat(target.parent, follow_symlinks=False)
        opened = os.fstat(parent_fd)
    except OSError:
        return False
    return stat.S_ISDIR(live.st_mode) and (live.st_dev, live.st_ino) == (
        opened.st_dev,
        opened.st_ino,
    )


def _fsync_parent(parent_fd: int) -> None:
    try:
        os.fsync(parent_fd)
    except OSError:
        _fail("TASK_PACK_DURABILITY_UNCERTAIN", status="INDETERMINATE")


def _write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        try:
            written = os.write(fd, payload[offset:])
        except OSError:
            _fail("TASK_PACK_WRITE_FAILED")
        if written <= 0:
            _fail("TASK_PACK_WRITE_FAILED")
        offset += written


def _new_temp(parent_fd: int) -> tuple[int, str]:
    for _ in range(32):
        name = f"{_TEMP_PREFIX}{secrets.token_hex(8)}{_TEMP_SUFFIX}"
        try:
            fd = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW,
                0o600,
                dir_fd=parent_fd,
            )
            return fd, name
        except FileExistsError:
            continue
        except OSError:
            _fail("TASK_PACK_TEMP_CREATE_FAILED")
    _fail("TASK_PACK_TEMP_CREATE_FAILED")


def _unlink_quietly(name: str | None, parent_fd: int) -> None:
    if not name:
        return
    try:
        os.unlink(name, dir_fd=parent_fd)
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _serialize(payload: dict[str, Any]) -> bytes:
    try:
        value = json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        serialized = (value + "\n").encode("utf-8")
    except (TypeError, ValueError):
        _fail("TASK_PACK_SERIALIZATION_FAILED")
    if len(serialized) > MAX_PACK_BYTES:
        _fail("TASK_PACK_TOO_LARGE")
    return serialized


def _reject_constant(_value: str) -> None:
    _fail("TASK_PACK_INVALID_JSON")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("TASK_PACK_DUPLICATE_JSON_KEYS")
        result[key] = value
    return result


def _parse(payload: bytes) -> dict[str, Any]:
    if len(payload) > MAX_PACK_BYTES:
        _fail("TASK_PACK_TOO_LARGE")
    try:
        text = payload.decode("utf-8")
        loaded = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except TaskCaptureError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError):
        _fail("TASK_PACK_INVALID_JSON")
    if not isinstance(loaded, dict):
        _fail("TASK_PACK_INVALID_ROOT")
    return loaded


def _canonical_statuses(values: object) -> list[str]:
    if not isinstance(values, list) or not values:
        _fail("TASK_EXPECTED_STATUS_INVALID")
    statuses: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            _fail("TASK_EXPECTED_STATUS_INVALID")
        status = value.strip().upper()
        if status not in ALLOWED_STATUSES or status in statuses:
            _fail("TASK_EXPECTED_STATUS_INVALID")
        statuses.append(status)
    return statuses


def _canonical_baseline(value: object) -> int | float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        _fail("TASK_MANUAL_BASELINE_INVALID")
    return value


def _canonical_task(row: object) -> dict[str, Any]:
    if not isinstance(row, dict):
        _fail("TASK_DEFINITION_INVALID")

    raw_id = row.get("id")
    raw_query = row.get("query")
    raw_mode = row.get("mode", "ask")
    if not isinstance(raw_id, str) or not raw_id.strip():
        _fail("TASK_ID_INVALID")
    if not isinstance(raw_query, str) or not raw_query.strip():
        _fail("TASK_QUERY_INVALID")
    if not isinstance(raw_mode, str):
        _fail("TASK_MODE_INVALID")
    mode = raw_mode.strip().lower()
    if mode not in ALLOWED_MODES:
        _fail("TASK_MODE_INVALID")
    query = raw_query.strip()
    try:
        query_size = len(query.encode("utf-8"))
    except UnicodeError:
        _fail("TASK_QUERY_INVALID")
    if query_size > MAX_QUERY_BYTES:
        _fail("TASK_QUERY_TOO_LARGE")

    raw_scope = row.get("doc_ids")
    if (
        not isinstance(raw_scope, list)
        or not raw_scope
        or any(not isinstance(value, str) or not value.strip() for value in raw_scope)
    ):
        _fail("TASK_SCOPE_INVALID")
    scope = normalize_doc_ids(raw_scope) or []
    if not scope:
        _fail("TASK_SCOPE_INVALID")
    if mode == "compare" and len(scope) < 2:
        _fail("COMPARE_SCOPE_TOO_SMALL")

    return {
        "id": raw_id.strip(),
        "mode": mode,
        "query": query,
        "doc_ids": scope,
        "expect_status": _canonical_statuses(row.get("expect_status")),
        "manual_baseline_seconds": _canonical_baseline(
            row.get("manual_baseline_seconds")
        ),
    }


def _validate_pack(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("schema") != TASK_PACK_SCHEMA:
        _fail("TASK_PACK_SCHEMA_UNSUPPORTED")
    if payload.get("storage_class") != OWNER_PRIVATE:
        _fail("TASK_PACK_STORAGE_CLASS_INVALID")
    rows = payload.get("tasks")
    if not isinstance(rows, list):
        _fail("TASK_PACK_TASKS_INVALID")
    if len(rows) > MAX_TASKS:
        _fail("TASK_PACK_CAPACITY_EXCEEDED")

    canonical = [_canonical_task(row) for row in rows]
    ids = [row["id"] for row in canonical]
    if len(ids) != len(set(ids)):
        _fail("DUPLICATE_TASK_ID")
    definitions = [
        (row["mode"], row["query"], tuple(row["doc_ids"])) for row in canonical
    ]
    if len(definitions) != len(set(definitions)):
        _fail("DUPLICATE_TASK_DEFINITION")
    return canonical


def _lock(parent_fd: int) -> int:
    try:
        fd = os.open(
            _LOCK_NAME,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | _NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
    except FileExistsError:
        try:
            fd = os.open(_LOCK_NAME, os.O_RDWR | _NOFOLLOW, dir_fd=parent_fd)
        except OSError:
            _fail("TASK_PACK_LOCK_FAILED")
    except OSError:
        _fail("TASK_PACK_LOCK_FAILED")
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            _fail("TASK_PACK_LOCK_FAILED")
        os.fchmod(fd, stat.S_IMODE(info.st_mode) & 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        return fd
    except Exception:
        os.close(fd)
        raise


def _read_pack(parent_fd: int, name: str) -> tuple[bytes, os.stat_result]:
    try:
        fd = os.open(
            name, os.O_RDONLY | _NOFOLLOW | _NONBLOCK, dir_fd=parent_fd
        )
    except FileNotFoundError:
        _fail("TASK_PACK_MISSING")
    except OSError:
        _fail("TASK_PACK_NOT_REGULAR")
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            _fail("TASK_PACK_NOT_REGULAR")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, min(65_536, MAX_PACK_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_PACK_BYTES:
                _fail("TASK_PACK_TOO_LARGE")
        return b"".join(chunks), info
    finally:
        os.close(fd)


def _same_entry(parent_fd: int, name: str, prior: os.stat_result) -> bool:
    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        return False
    return stat.S_ISREG(current.st_mode) and (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
    ) == (
        prior.st_dev,
        prior.st_ino,
        prior.st_size,
        prior.st_mtime_ns,
    )


def initialize_task_pack(path: Path) -> dict[str, Any]:
    """Create a canonical empty private pack without overwriting any entry."""
    target = _capture_path(Path(path))
    _prepare_init_parent(target)
    parent_fd = _open_parent(target)
    temp_fd: int | None = None
    temp_name: str | None = None
    published = False
    payload = {
        "schema": TASK_PACK_SCHEMA,
        "storage_class": OWNER_PRIVATE,
        "tasks": [],
    }
    serialized = _serialize(payload)
    try:
        if not _parent_is_current(target, parent_fd):
            _fail("TASK_PACK_PARENT_CHANGED")
        temp_fd, temp_name = _new_temp(parent_fd)
        try:
            _write_all(temp_fd, serialized)
            try:
                os.fsync(temp_fd)
            except OSError:
                _fail("TASK_PACK_WRITE_FAILED")
        finally:
            os.close(temp_fd)
            temp_fd = None

        if not _parent_is_current(target, parent_fd):
            _fail("TASK_PACK_PARENT_CHANGED")
        try:
            os.link(
                temp_name,
                target.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            _fail("TASK_PACK_ALREADY_EXISTS")
        except OSError as exc:
            if exc.errno in {errno.EEXIST, errno.EISDIR, errno.ELOOP}:
                _fail("TASK_PACK_ALREADY_EXISTS")
            _fail("TASK_PACK_PUBLISH_FAILED")
        published = True
        _unlink_quietly(temp_name, parent_fd)
        temp_name = None
        _fsync_parent(parent_fd)
    finally:
        if temp_fd is not None:
            os.close(temp_fd)
        _unlink_quietly(temp_name, parent_fd)
        os.close(parent_fd)

    return {
        "schema": CAPTURE_RESULT_SCHEMA,
        "status": "INITIALIZED",
        "n_tasks": 0,
        "remaining_capacity": MAX_TASKS,
        "next_action": "CAPTURE_GENUINE_TASKS",
        "published": published,
    }


def append_task(
    path: Path,
    *,
    task_id: str,
    mode: str,
    query: str,
    doc_ids: Iterable[str],
    expect_status: Iterable[str],
    manual_baseline_seconds: int | float,
) -> dict[str, Any]:
    """Append one canonical task under an inter-process lock."""
    target = _capture_path(Path(path))
    if not target.parent.is_dir() or target.parent.is_symlink():
        _fail("TASK_PACK_PARENT_INVALID")
    candidate = _canonical_task(
        {
            "id": task_id,
            "mode": mode,
            "query": query,
            "doc_ids": list(doc_ids),
            "expect_status": list(expect_status),
            "manual_baseline_seconds": manual_baseline_seconds,
        }
    )

    parent_fd = _open_parent(target)
    lock_fd: int | None = None
    temp_fd: int | None = None
    temp_name: str | None = None
    published = False
    try:
        if not _parent_is_current(target, parent_fd):
            _fail("TASK_PACK_PARENT_CHANGED")
        lock_fd = _lock(parent_fd)
        raw, prior = _read_pack(parent_fd, target.name)
        payload = _parse(raw)
        canonical_rows = _validate_pack(payload)
        if len(canonical_rows) >= MAX_TASKS:
            _fail("TASK_PACK_CAPACITY_EXCEEDED")
        if candidate["id"] in {row["id"] for row in canonical_rows}:
            _fail("DUPLICATE_TASK_ID")
        candidate_definition = (
            candidate["mode"],
            candidate["query"],
            tuple(candidate["doc_ids"]),
        )
        definitions = {
            (row["mode"], row["query"], tuple(row["doc_ids"]))
            for row in canonical_rows
        }
        if candidate_definition in definitions:
            _fail("DUPLICATE_TASK_DEFINITION")

        updated = dict(payload)
        updated["tasks"] = [*payload["tasks"], candidate]
        serialized = _serialize(updated)
        temp_fd, temp_name = _new_temp(parent_fd)
        try:
            _write_all(temp_fd, serialized)
            try:
                os.fsync(temp_fd)
            except OSError:
                _fail("TASK_PACK_WRITE_FAILED")
        finally:
            os.close(temp_fd)
            temp_fd = None

        if not _parent_is_current(target, parent_fd) or not _same_entry(
            parent_fd, target.name, prior
        ):
            _fail("TASK_PACK_CHANGED_DURING_ADD")
        try:
            os.replace(
                temp_name,
                target.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
        except OSError:
            _fail("TASK_PACK_PUBLISH_FAILED")
        published = True
        temp_name = None
        _fsync_parent(parent_fd)
    finally:
        if temp_fd is not None:
            os.close(temp_fd)
        _unlink_quietly(temp_name, parent_fd)
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
        os.close(parent_fd)

    n_tasks = len(canonical_rows) + 1
    return {
        "schema": CAPTURE_RESULT_SCHEMA,
        "status": "CAPTURED",
        "n_tasks": n_tasks,
        "remaining_capacity": MAX_TASKS - n_tasks,
        "representative_ready": False,
        "next_action": "RUN_STUDY_CHECK_IN_NEW_DIRECTORY",
        "identity_effect": "TASK_PACK_DIGEST_CHANGED",
        "published": published,
    }


def read_private_query(*, query_file: Path | None, stdin: TextIO) -> str:
    """Read a private query without accepting it in process arguments."""
    if query_file is None:
        query = stdin.read(MAX_QUERY_BYTES + 1)
    else:
        try:
            source = private_paths.require_private_query_input(Path(query_file))
        except (OSError, ValueError):
            _fail("UNSAFE_QUERY_INPUT_PATH")
        fd: int | None = None
        try:
            fd = os.open(source, os.O_RDONLY | _NOFOLLOW | _NONBLOCK)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                _fail("QUERY_INPUT_INVALID")
            if info.st_size > MAX_QUERY_BYTES:
                _fail("TASK_QUERY_TOO_LARGE")
            chunks: list[bytes] = []
            total = 0
            while total < MAX_QUERY_BYTES + 1:
                chunk = os.read(fd, min(65_536, MAX_QUERY_BYTES + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
            raw = b"".join(chunks)
            query = raw.decode("utf-8")
        except TaskCaptureError:
            raise
        except (OSError, UnicodeError):
            _fail("QUERY_INPUT_INVALID")
        finally:
            if fd is not None:
                os.close(fd)
    if len(query.encode("utf-8")) > MAX_QUERY_BYTES:
        _fail("TASK_QUERY_TOO_LARGE")
    if not query.strip():
        _fail("TASK_QUERY_INVALID")
    return query.strip()
