"""Private corpus-ID inventory and task-scope diagnostics.

The detailed artifact is owner-private.  The public receipt contains counts and
codes only: never paths, document IDs, task IDs, queries, or document content.
This surface discovers exact identities and proposes reversible repairs; it
does not mutate a task pack or determine representative-study readiness.
"""
from __future__ import annotations

import errno
import hashlib
import io
import json
import math
import os
import secrets
import stat
from pathlib import Path, PurePosixPath
from typing import Any

from wedge_v1 import private_output as private_paths
from wedge_v1.ingest import (
    SUPPORTED_SUFFIXES,
    TEXT_SUFFIXES,
    document_id,
    needs_ocr_normalize,
)
from wedge_v1.runtime import normalize_doc_ids
from wedge_v1.study_capture import (
    ALLOWED_MODES,
    ALLOWED_STATUSES,
    MAX_PACK_BYTES,
    MAX_QUERY_BYTES,
    MAX_TASKS,
    OWNER_PRIVATE,
    TASK_PACK_SCHEMA,
)


INVENTORY_SCHEMA = "nano-lm.wedge_v1.study_inventory.v1"
INVENTORY_RECEIPT_SCHEMA = "nano-lm.wedge_v1.study_inventory_receipt.v1"
MAX_REPORT_BYTES = 5_000_000
MAX_VISIBLE_FILES = 10_000
MAX_SOURCE_BYTES = 100_000_000

_TEMP_PREFIX = ".study-inventory-"
_TEMP_SUFFIX = ".tmp"
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)


class StudyInventoryError(ValueError):
    """Machine-readable failure that is safe to print as a public receipt."""

    def __init__(self, code: str, *, status: str = "REJECTED"):
        self.code = code
        self.status = status
        super().__init__(code)


def _fail(code: str, *, status: str = "REJECTED") -> None:
    raise StudyInventoryError(code, status=status)


def _canonical_corpus(path: Path) -> Path:
    try:
        corpus = private_paths.require_private_corpus(Path(path))
    except (OSError, ValueError):
        _fail("UNSAFE_CORPUS_PATH")
    if not corpus.is_dir() or corpus.is_symlink():
        _fail("CORPUS_PATH_MISSING")
    return corpus


def _canonical_output(path: Path) -> Path:
    try:
        return private_paths.require_private_study_inventory(Path(path))
    except (OSError, ValueError):
        _fail("UNSAFE_INVENTORY_PATH")


def _canonical_tasks(path: Path) -> Path:
    try:
        return private_paths.require_private_task_pack(Path(path))
    except (OSError, ValueError):
        _fail("UNSAFE_TASK_PACK_PATH")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def _stat_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
    )


def _visible_entries(
    corpus: Path,
) -> tuple[list[Path], dict[str, tuple[int, int, int, int, int]]]:
    entries: list[Path] = []
    identities: dict[str, tuple[int, int, int, int, int]] = {}
    try:
        for path in corpus.rglob("*"):
            relative = path.relative_to(corpus)
            if any(part.startswith(".") for part in relative.parts):
                continue
            if path.is_symlink():
                _fail("CORPUS_SYMLINK_ENTRY")
            info = path.stat(follow_symlinks=False)
            if stat.S_ISREG(info.st_mode):
                entries.append(path)
                identities[relative.as_posix()] = _stat_identity(info)
                if len(entries) > MAX_VISIBLE_FILES:
                    _fail("CORPUS_FILE_LIMIT_EXCEEDED")
            elif not stat.S_ISDIR(info.st_mode):
                _fail("CORPUS_ENTRY_NOT_REGULAR")
    except StudyInventoryError:
        raise
    except OSError:
        _fail("CORPUS_SCAN_ERROR")
    return (
        sorted(entries, key=lambda item: item.relative_to(corpus).as_posix()),
        identities,
    )


def _read_corpus_bytes(path: Path, *, remaining: int) -> bytes:
    fd: int | None = None
    try:
        fd = os.open(path, os.O_RDONLY | _NOFOLLOW | _NONBLOCK)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            _fail("CORPUS_ENTRY_NOT_REGULAR")
        if info.st_size > remaining:
            _fail("CORPUS_SOURCE_TOO_LARGE")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, min(65_536, remaining + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > remaining:
                _fail("CORPUS_SOURCE_TOO_LARGE")
        raw = b"".join(chunks)
        current = path.stat(follow_symlinks=False)
        if _stat_identity(current) != _stat_identity(info):
            _fail("CORPUS_CHANGED_DURING_INVENTORY")
        return raw
    except StudyInventoryError:
        raise
    except OSError:
        _fail("CORPUS_READ_ERROR")
    finally:
        if fd is not None:
            os.close(fd)


def _extract_pdf(raw: bytes) -> str | None:
    try:
        from pypdf import PdfReader
    except Exception:
        return None
    try:
        reader = PdfReader(io.BytesIO(raw))
        body = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
        return body or None
    except Exception:
        return None


def _load_snapshot(
    corpus: Path, supported: list[Path]
) -> tuple[dict[str, str], str]:
    sources: dict[str, Path] = {}
    for path in supported:
        identifier = document_id(path, corpus)
        if identifier in sources:
            _fail("DUPLICATE_DOCUMENT_IDENTITY")
        sources[identifier] = path

    docs: dict[str, str] = {}
    digest = hashlib.sha256()
    total_source_bytes = 0
    for identifier, source in sources.items():
        raw = _read_corpus_bytes(
            source, remaining=MAX_SOURCE_BYTES - total_source_bytes
        )
        total_source_bytes += len(raw)
        relative = source.relative_to(corpus).as_posix()
        digest.update(relative.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        digest.update(raw)
        digest.update(b"\0")
        if source.suffix.lower() in TEXT_SUFFIXES:
            docs[identifier] = raw.decode("utf-8", errors="replace")
        else:
            body = _extract_pdf(raw)
            if body is not None:
                docs[identifier] = body

    if docs:
        from wedge_v1.plugins.ocr import normalize_text

        docs = {
            identifier: (
                normalize_text(text)[0] if needs_ocr_normalize(text) else text
            )
            for identifier, text in docs.items()
        }
    return docs, digest.hexdigest()


def _format_name(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return "markdown"
    if suffix == ".txt":
        return "text"
    if suffix == ".pdf":
        return "pdf"
    return "unsupported"


def _pdf_extractor_identity(n_pdf: int) -> str:
    if not n_pdf:
        return "NOT_USED"
    try:
        import pypdf
    except Exception:
        return "UNAVAILABLE"
    version = str(getattr(pypdf, "__version__", "UNKNOWN")).strip() or "UNKNOWN"
    return f"pypdf:{version}"


def _read_regular_file(path: Path, *, limit: int) -> bytes:
    fd: int | None = None
    try:
        fd = os.open(path, os.O_RDONLY | _NOFOLLOW | _NONBLOCK)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            _fail("TASK_PACK_NOT_REGULAR")
        if info.st_size > limit:
            _fail("TASK_PACK_TOO_LARGE")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, min(65_536, limit + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > limit:
                _fail("TASK_PACK_TOO_LARGE")
        return b"".join(chunks)
    except StudyInventoryError:
        raise
    except FileNotFoundError:
        _fail("TASK_PACK_MISSING")
    except OSError:
        _fail("TASK_PACK_NOT_REGULAR")
    finally:
        if fd is not None:
            os.close(fd)


def _reject_constant(_value: str) -> None:
    _fail("TASK_PACK_INVALID_JSON")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("TASK_PACK_DUPLICATE_JSON_KEYS")
        result[key] = value
    return result


def _parse_task_pack(raw: bytes) -> dict[str, Any]:
    try:
        loaded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except StudyInventoryError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError):
        _fail("TASK_PACK_INVALID_JSON")
    if not isinstance(loaded, dict) or not isinstance(loaded.get("tasks"), list):
        _fail("TASK_PACK_INVALID_ROOT")
    if loaded.get("schema") != TASK_PACK_SCHEMA:
        _fail("TASK_PACK_SCHEMA_UNSUPPORTED")
    if loaded.get("storage_class") != OWNER_PRIVATE:
        _fail("TASK_PACK_STORAGE_CLASS_INVALID")
    if len(loaded["tasks"]) > MAX_TASKS:
        _fail("TASK_PACK_CAPACITY_EXCEEDED")
    return loaded


def _valid_baseline(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _valid_expected_statuses(value: object) -> bool:
    if not isinstance(value, list) or not value:
        return False
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            return False
        status = item.strip().upper()
        if status not in ALLOWED_STATUSES or status in normalized:
            return False
        normalized.append(status)
    return True


def _scope_proposal(missing_id: str, valid_ids: list[str]) -> dict[str, str] | None:
    basename = PurePosixPath(missing_id).name
    matches = [
        candidate
        for candidate in valid_ids
        if PurePosixPath(candidate).name == basename
    ]
    if len(matches) != 1:
        return None
    return {
        "from_doc_id": missing_id,
        "to_doc_id": matches[0],
        "rule": "EXACT_UNIQUE_BASENAME",
    }


def _diagnose_tasks(
    pack: dict[str, Any], raw: bytes, valid_ids: list[str]
) -> tuple[dict[str, Any], dict[str, int]]:
    rows = pack["tasks"]
    known = set(valid_ids)
    diagnostics: list[dict[str, Any]] = []
    invalid_scope_count = 0
    operator_input_count = 0
    proposal_count = 0
    ambiguous_count = 0
    unmatched_count = 0
    unknown_count = 0
    proposal_rows: dict[tuple[str, str], dict[str, str]] = {}

    basename_counts: dict[str, int] = {}
    for value in valid_ids:
        name = PurePosixPath(value).name
        basename_counts[name] = basename_counts.get(name, 0) + 1

    id_counts: dict[str, int] = {}
    definition_counts: dict[tuple[str, str, tuple[str, ...]], int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_id = row.get("id")
        if isinstance(raw_id, str) and raw_id.strip():
            identifier = raw_id.strip()
            id_counts[identifier] = id_counts.get(identifier, 0) + 1
        raw_mode = row.get("mode", "ask")
        raw_query = row.get("query")
        raw_scope = row.get("doc_ids")
        if (
            isinstance(raw_mode, str)
            and isinstance(raw_query, str)
            and isinstance(raw_scope, list)
            and raw_scope
            and all(isinstance(value, str) and value.strip() for value in raw_scope)
        ):
            definition = (
                raw_mode.strip().lower(),
                raw_query.strip(),
                tuple(normalize_doc_ids(raw_scope) or []),
            )
            definition_counts[definition] = definition_counts.get(definition, 0) + 1

    for index, row in enumerate(rows):
        codes: list[str] = []
        task_id: str | None = None
        mode: str | None = None
        scope: list[str] = []

        if not isinstance(row, dict):
            codes.append("TASK_DEFINITION_INVALID")
            raw_scope: object = None
        else:
            raw_id = row.get("id")
            if isinstance(raw_id, str) and raw_id.strip():
                task_id = raw_id.strip()
                if id_counts.get(task_id, 0) > 1:
                    codes.append("DUPLICATE_TASK_ID")
            else:
                codes.append("TASK_ID_INVALID")

            raw_mode = row.get("mode", "ask")
            if isinstance(raw_mode, str) and raw_mode.strip().lower() in ALLOWED_MODES:
                mode = raw_mode.strip().lower()
            else:
                codes.append("TASK_MODE_INVALID")

            query = row.get("query")
            if not isinstance(query, str) or not query.strip():
                codes.append("TASK_QUERY_INVALID")
            else:
                try:
                    query_size = len(query.strip().encode("utf-8"))
                except UnicodeEncodeError:
                    codes.append("TASK_QUERY_INVALID")
                else:
                    if query_size > MAX_QUERY_BYTES:
                        codes.append("TASK_QUERY_TOO_LARGE")
            if not _valid_expected_statuses(row.get("expect_status")):
                codes.append("TASK_EXPECTED_STATUS_INVALID")
            if not _valid_baseline(row.get("manual_baseline_seconds")):
                codes.append("TASK_MANUAL_BASELINE_INVALID")
            raw_scope = row.get("doc_ids")

        if (
            not isinstance(raw_scope, list)
            or not raw_scope
            or any(not isinstance(value, str) or not value.strip() for value in raw_scope)
        ):
            codes.append("TASK_SCOPE_REQUIRED")
        else:
            scope = normalize_doc_ids(raw_scope) or []
            if not scope:
                codes.append("TASK_SCOPE_REQUIRED")

        if isinstance(row, dict) and isinstance(row.get("query"), str) and mode:
            definition = (mode, row["query"].strip(), tuple(scope))
            if definition_counts.get(definition, 0) > 1:
                codes.append("DUPLICATE_TASK_DEFINITION")

        resolved = [value for value in scope if value in known]
        missing = [value for value in scope if value not in known]
        unknown_count += len(missing)
        proposals: list[dict[str, str]] = []
        for missing_id in missing:
            proposal = _scope_proposal(missing_id, valid_ids)
            if proposal is not None:
                proposals.append(proposal)
                proposal_count += 1
                proposal_rows[(proposal["from_doc_id"], proposal["to_doc_id"])] = proposal
            elif basename_counts.get(PurePosixPath(missing_id).name, 0) > 1:
                ambiguous_count += 1
            else:
                unmatched_count += 1
        if missing:
            codes.append("UNKNOWN_DOCUMENT_ID")
        if mode == "compare" and len(scope) < 2:
            codes.append("COMPARE_SCOPE_TOO_SMALL")

        scope_codes = {
            "TASK_SCOPE_REQUIRED",
            "UNKNOWN_DOCUMENT_ID",
            "COMPARE_SCOPE_TOO_SMALL",
        }
        scope_status = "INVALID" if scope_codes.intersection(codes) else "VALID"
        if scope_status == "INVALID":
            invalid_scope_count += 1
        if codes:
            operator_input_count += 1
        diagnostics.append(
            {
                "index": index,
                "task_id": task_id,
                "mode": mode,
                "requested_doc_ids": scope,
                "resolved_doc_ids": resolved,
                "unknown_doc_ids": missing,
                "scope_proposals": proposals,
                "scope_status": scope_status,
                "codes": sorted(set(codes)),
            }
        )

    aggregate = {
        "n_tasks": len(rows),
        "n_invalid_task_scopes": invalid_scope_count,
        "n_tasks_requiring_operator_input": operator_input_count,
        "n_unknown_scope_references": unknown_count,
        "n_scope_proposal_occurrences": proposal_count,
        "n_scope_proposals": len(proposal_rows),
        "n_ambiguous_scope_references": ambiguous_count,
        "n_unmatched_scope_references": unmatched_count,
    }
    return (
        {
            "task_pack_digest": hashlib.sha256(raw).hexdigest(),
            "pack_codes": [],
            "aggregate": aggregate,
            "tasks": diagnostics,
            "scope_proposals": [
                proposal_rows[key] for key in sorted(proposal_rows)
            ],
            "proposal_boundary": (
                "EXACT_UNIQUE_BASENAME proposals are discovery aids only. Confirm each "
                "mapping against the source documents before editing a new task-pack revision."
            ),
        },
        aggregate,
    )


def _build_inventory(corpus: Path, tasks: Path | None) -> tuple[dict[str, Any], dict[str, int]]:
    entries, before_identity = _visible_entries(corpus)
    supported = [path for path in entries if path.suffix.lower() in SUPPORTED_SUFFIXES]
    unsupported = [path for path in entries if path.suffix.lower() not in SUPPORTED_SUFFIXES]
    loaded, digest = _load_snapshot(corpus, supported)
    after_entries, after_identity = _visible_entries(corpus)
    if (
        before_identity != after_identity
        or [path.relative_to(corpus).as_posix() for path in entries]
        != [path.relative_to(corpus).as_posix() for path in after_entries]
    ):
        _fail("CORPUS_CHANGED_DURING_INVENTORY")

    documents: list[dict[str, Any]] = []
    valid_ids: list[str] = []
    unreadable_count = 0
    for path in supported:
        doc_id = document_id(path, corpus)
        text = loaded.get(doc_id)
        readable = isinstance(text, str) and bool(text.strip())
        if readable:
            valid_ids.append(doc_id)
        else:
            unreadable_count += 1
        documents.append(
            {
                "doc_id": doc_id,
                "relative_path": path.relative_to(corpus).as_posix(),
                "format": _format_name(path),
                "readable": readable,
                "extracted_text_bytes": len(text.encode("utf-8")) if readable else 0,
                "codes": [] if readable else ["UNREADABLE_OR_EMPTY"],
            }
        )
    valid_ids.sort()

    unsupported_rows = [
        {
            "relative_path": path.relative_to(corpus).as_posix(),
            "format": "unsupported",
            "codes": ["UNSUPPORTED_FORMAT"],
        }
        for path in unsupported
    ]
    pdf_count = sum(1 for path in supported if path.suffix.lower() == ".pdf")
    artifact: dict[str, Any] = {
        "schema": INVENTORY_SCHEMA,
        "storage_class": OWNER_PRIVATE,
        "identity": {
            "corpus_digest": digest,
            "pdf_extractor": _pdf_extractor_identity(pdf_count),
            "task_pack_digest": None,
        },
        "corpus": {
            "n_visible_files": len(entries),
            "n_valid_documents": len(valid_ids),
            "n_unreadable_documents": unreadable_count,
            "n_unsupported_files": len(unsupported_rows),
            "valid_document_ids": valid_ids,
        },
        "documents": documents,
        "unsupported_files": unsupported_rows,
        "task_diagnostics": None,
        "claim_boundary": (
            "This private inventory discovers corpus identities and task-scope issues. "
            "It does not establish task authenticity, scope correctness, readiness, or utility."
        ),
    }

    aggregate = {
        "n_valid_documents": len(valid_ids),
        "n_unreadable_documents": unreadable_count,
        "n_unsupported_files": len(unsupported_rows),
        "n_tasks": 0,
        "n_invalid_task_scopes": 0,
        "n_tasks_requiring_operator_input": 0,
        "n_scope_proposals": 0,
        "n_scope_proposal_occurrences": 0,
        "n_unknown_scope_references": 0,
        "n_ambiguous_scope_references": 0,
        "n_unmatched_scope_references": 0,
    }
    if tasks is not None:
        raw = _read_regular_file(tasks, limit=MAX_PACK_BYTES)
        pack = _parse_task_pack(raw)
        task_diagnostics, task_aggregate = _diagnose_tasks(pack, raw, valid_ids)
        artifact["task_diagnostics"] = task_diagnostics
        artifact["identity"]["task_pack_digest"] = task_diagnostics["task_pack_digest"]
        aggregate.update(task_aggregate)
    return artifact, aggregate


def _serialize(payload: dict[str, Any]) -> bytes:
    try:
        value = json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        serialized = (value + "\n").encode("utf-8")
    except (TypeError, ValueError):
        _fail("INVENTORY_SERIALIZATION_FAILED")
    if len(serialized) > MAX_REPORT_BYTES:
        _fail("INVENTORY_TOO_LARGE")
    return serialized


def _prepare_parent(target: Path) -> None:
    parent = target.parent
    if parent.exists():
        if not parent.is_dir() or parent.is_symlink():
            _fail("INVENTORY_PARENT_INVALID")
        return

    private_root = Path(private_paths.PRIVATE_EXPORT_ROOT).expanduser().resolve(strict=False)
    if parent != private_root:
        _fail("INVENTORY_PARENT_MISSING")
    try:
        parent.mkdir(mode=0o700, parents=False, exist_ok=False)
        parent.chmod(0o700)
    except FileExistsError:
        if not parent.is_dir() or parent.is_symlink():
            _fail("INVENTORY_PARENT_INVALID")
    except OSError:
        _fail("INVENTORY_PARENT_CREATE_FAILED")


def _open_parent(target: Path) -> int:
    try:
        fd = os.open(target.parent, os.O_RDONLY | _DIRECTORY | _NOFOLLOW)
    except OSError:
        _fail("INVENTORY_PARENT_INVALID")
    try:
        info = os.fstat(fd)
        if not stat.S_ISDIR(info.st_mode):
            _fail("INVENTORY_PARENT_INVALID")
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
            _fail("INVENTORY_TEMP_CREATE_FAILED")
    _fail("INVENTORY_TEMP_CREATE_FAILED")


def _write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        try:
            written = os.write(fd, payload[offset:])
        except OSError:
            _fail("INVENTORY_WRITE_FAILED")
        if written <= 0:
            _fail("INVENTORY_WRITE_FAILED")
        offset += written


def _unlink_quietly(name: str | None, parent_fd: int) -> None:
    if not name:
        return
    try:
        os.unlink(name, dir_fd=parent_fd)
    except OSError:
        pass


def _fsync_parent(parent_fd: int) -> None:
    try:
        os.fsync(parent_fd)
    except OSError:
        _fail("INVENTORY_DURABILITY_UNCERTAIN", status="INDETERMINATE")


def _publish_no_clobber(target: Path, payload: bytes) -> None:
    _prepare_parent(target)
    parent_fd = _open_parent(target)
    temp_fd: int | None = None
    temp_name: str | None = None
    published = False
    try:
        if not _parent_is_current(target, parent_fd):
            _fail("INVENTORY_PARENT_CHANGED")
        temp_fd, temp_name = _new_temp(parent_fd)
        try:
            _write_all(temp_fd, payload)
            try:
                os.fsync(temp_fd)
            except OSError:
                _fail("INVENTORY_WRITE_FAILED")
        finally:
            os.close(temp_fd)
            temp_fd = None

        if not _parent_is_current(target, parent_fd):
            _fail("INVENTORY_PARENT_CHANGED")
        try:
            os.link(
                temp_name,
                target.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            _fail("INVENTORY_ALREADY_EXISTS")
        except OSError as exc:
            if exc.errno in {errno.EEXIST, errno.EISDIR, errno.ELOOP}:
                _fail("INVENTORY_ALREADY_EXISTS")
            _fail("INVENTORY_PUBLISH_FAILED")
        published = True
        _unlink_quietly(temp_name, parent_fd)
        temp_name = None
        _fsync_parent(parent_fd)
    finally:
        if temp_fd is not None:
            os.close(temp_fd)
        _unlink_quietly(temp_name, parent_fd)
        os.close(parent_fd)
    if not published:  # pragma: no cover - defensive invariant
        _fail("INVENTORY_PUBLISH_FAILED")


def _receipt(
    aggregate: dict[str, int], artifact_digest: str, *, tasks_checked: bool
) -> dict[str, Any]:
    scope_issues = aggregate["n_invalid_task_scopes"] > 0
    task_issues = aggregate["n_tasks_requiring_operator_input"] > 0
    corpus_issues = (
        aggregate["n_valid_documents"] == 0
        or aggregate["n_unreadable_documents"] > 0
        or aggregate["n_unsupported_files"] > 0
    )
    if scope_issues:
        code, next_action = (
            "TASK_SCOPE_ISSUES",
            "CONFIRM_SCOPE_REPAIRS",
        )
    elif task_issues:
        code, next_action = (
            "TASK_INPUTS_REQUIRE_REVIEW",
            "REPAIR_TASK_PACK",
        )
    elif corpus_issues:
        code, next_action = (
            "CORPUS_CLEANUP_REQUIRED",
            "INSPECT_PRIVATE_INVENTORY",
        )
    elif tasks_checked and aggregate["n_tasks"] == 0:
        code, next_action = "TASK_PACK_EMPTY", "CAPTURE_GENUINE_TASKS"
    elif tasks_checked:
        code, next_action = "INVENTORY_READY", "RUN_STUDY_CHECK"
    else:
        code, next_action = (
            "INVENTORY_READY",
            "CAPTURE_GENUINE_TASKS",
        )
    return {
        "schema": INVENTORY_RECEIPT_SCHEMA,
        "status": "CREATED",
        "code": code,
        "tasks_checked": tasks_checked,
        **aggregate,
        "artifact_digest": artifact_digest,
        "next_action": next_action,
        "claim_boundary": (
            "The private inventory is diagnostic only; study check remains the "
            "readiness authority."
        ),
    }


def create_study_inventory(
    corpus: Path, output: Path, tasks: Path | None = None
) -> dict[str, Any]:
    """Publish one deterministic, no-clobber private inventory and return a receipt."""
    corpus_path = _canonical_corpus(Path(corpus))
    output_path = _canonical_output(Path(output))
    if _is_within(output_path, corpus_path):
        _fail("INVENTORY_INSIDE_CORPUS")
    task_path = _canonical_tasks(Path(tasks)) if tasks is not None else None
    artifact, aggregate = _build_inventory(corpus_path, task_path)
    serialized = _serialize(artifact)
    digest = hashlib.sha256(serialized).hexdigest()
    _publish_no_clobber(output_path, serialized)
    return _receipt(aggregate, digest, tasks_checked=task_path is not None)
