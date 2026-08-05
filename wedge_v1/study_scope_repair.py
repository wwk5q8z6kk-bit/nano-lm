"""Copy-on-write repair of stale exact document scopes.

The source task pack and detailed inventory are owner-private.  Public receipts
contain only aggregate counts, digests, and fixed codes.  A repair changes only
``doc_ids`` in a new task-pack revision; it never edits the source, invents a
manual baseline, runs a solver, or claims representative readiness.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from wedge_v1 import private_output as private_paths
from wedge_v1 import study_inventory as inventory_module
from wedge_v1.runtime import normalize_doc_ids
from wedge_v1.study_capture import (
    MAX_PACK_BYTES,
    MAX_TASKS,
    OWNER_PRIVATE,
    TASK_PACK_SCHEMA,
)

SCOPE_REPAIR_RECEIPT_SCHEMA = "nano-lm.wedge_v1.study_scope_repair.v1"
_ALLOWED_SOURCE_CODES = {
    "UNKNOWN_DOCUMENT_ID",
    "TASK_MANUAL_BASELINE_INVALID",
}


class ScopeRepairError(ValueError):
    """Machine-readable scope-repair failure safe for public output."""

    def __init__(self, code: str, *, status: str = "REJECTED"):
        self.code = code
        self.status = status
        super().__init__(code)


def _fail(code: str, *, status: str = "REJECTED") -> None:
    raise ScopeRepairError(code, status=status)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_corpus(path: Path) -> Path:
    try:
        corpus = private_paths.require_private_corpus(Path(path))
    except (OSError, ValueError):
        _fail("UNSAFE_CORPUS_PATH")
    if not corpus.is_dir() or corpus.is_symlink():
        _fail("CORPUS_PATH_MISSING")
    return corpus


def _canonical_task_path(path: Path, *, output: bool) -> Path:
    try:
        target = private_paths.require_private_task_pack(Path(path))
    except (OSError, ValueError):
        _fail(
            "UNSAFE_OUTPUT_TASK_PACK_PATH"
            if output
            else "UNSAFE_SOURCE_TASK_PACK_PATH"
        )
    return target


def _canonical_inventory(path: Path) -> Path:
    try:
        return private_paths.require_private_study_inventory(Path(path))
    except (OSError, ValueError):
        _fail("UNSAFE_INVENTORY_PATH")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def _read_regular(path: Path, *, limit: int, kind: str) -> bytes:
    try:
        return inventory_module._read_regular_file(path, limit=limit)
    except inventory_module.StudyInventoryError as exc:
        if kind == "source":
            mapping = {
                "TASK_PACK_MISSING": "SOURCE_TASK_PACK_MISSING",
                "TASK_PACK_NOT_REGULAR": "SOURCE_TASK_PACK_NOT_REGULAR",
                "TASK_PACK_TOO_LARGE": "SOURCE_TASK_PACK_TOO_LARGE",
            }
            _fail(mapping.get(exc.code, "SOURCE_TASK_PACK_READ_FAILED"))
        mapping = {
            "TASK_PACK_MISSING": "INVENTORY_MISSING",
            "TASK_PACK_NOT_REGULAR": "INVENTORY_NOT_REGULAR",
            "TASK_PACK_TOO_LARGE": "INVENTORY_TOO_LARGE",
        }
        _fail(mapping.get(exc.code, "INVENTORY_READ_FAILED"))


def _parse_json(raw: bytes, *, kind: str) -> Any:
    duplicate_code = (
        "SOURCE_TASK_PACK_DUPLICATE_JSON_KEYS"
        if kind == "source"
        else "INVENTORY_DUPLICATE_JSON_KEYS"
    )
    invalid_code = (
        "SOURCE_TASK_PACK_INVALID_JSON"
        if kind == "source"
        else "INVENTORY_INVALID_JSON"
    )

    def reject_constant(_value: str) -> None:
        _fail(invalid_code)

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(duplicate_code)
            result[key] = value
        return result

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except ScopeRepairError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError):
        _fail(invalid_code)


def _parse_source_pack(raw: bytes) -> dict[str, Any]:
    loaded = _parse_json(raw, kind="source")
    if not isinstance(loaded, dict) or not isinstance(loaded.get("tasks"), list):
        _fail("SOURCE_TASK_PACK_INVALID_ROOT")
    if loaded.get("schema") != TASK_PACK_SCHEMA:
        _fail("SOURCE_TASK_PACK_SCHEMA_UNSUPPORTED")
    if loaded.get("storage_class") != OWNER_PRIVATE:
        _fail("SOURCE_TASK_PACK_STORAGE_CLASS_INVALID")
    if len(loaded["tasks"]) > MAX_TASKS:
        _fail("SOURCE_TASK_PACK_CAPACITY_EXCEEDED")
    return loaded


def _parse_inventory(raw: bytes) -> dict[str, Any]:
    loaded = _parse_json(raw, kind="inventory")
    if not isinstance(loaded, dict):
        _fail("INVENTORY_INVALID_ROOT")
    if loaded.get("schema") != inventory_module.INVENTORY_SCHEMA:
        _fail("INVENTORY_SCHEMA_UNSUPPORTED")
    if loaded.get("storage_class") != OWNER_PRIVATE:
        _fail("INVENTORY_STORAGE_CLASS_INVALID")
    identity = loaded.get("identity")
    corpus = loaded.get("corpus")
    diagnostics = loaded.get("task_diagnostics")
    if (
        not isinstance(identity, dict)
        or not isinstance(identity.get("corpus_digest"), str)
        or not isinstance(identity.get("task_pack_digest"), str)
        or not isinstance(corpus, dict)
        or not isinstance(corpus.get("valid_document_ids"), list)
        or not all(
            isinstance(value, str) and value
            for value in corpus["valid_document_ids"]
        )
        or not isinstance(diagnostics, dict)
        or not isinstance(diagnostics.get("scope_proposals"), list)
    ):
        _fail("INVENTORY_STRUCTURE_INVALID")
    return loaded


def _proposal_map(rows: object, valid_ids: list[str]) -> dict[str, str]:
    if not isinstance(rows, list) or not rows:
        _fail("INVENTORY_PROPOSALS_INVALID")
    proposals: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            _fail("INVENTORY_PROPOSALS_INVALID")
        source = row.get("from_doc_id")
        target = row.get("to_doc_id")
        if (
            not isinstance(source, str)
            or not source
            or not isinstance(target, str)
            or not target
            or row.get("rule") != "EXACT_UNIQUE_BASENAME"
        ):
            _fail("INVENTORY_PROPOSALS_INVALID")
        rederived = inventory_module._scope_proposal(source, valid_ids)
        if rederived != {
            "from_doc_id": source,
            "to_doc_id": target,
            "rule": "EXACT_UNIQUE_BASENAME",
        }:
            _fail("INVENTORY_PROPOSALS_INVALID")
        prior = proposals.get(source)
        if prior is not None and prior != target:
            _fail("INVENTORY_PROPOSALS_INVALID")
        proposals[source] = target
    return proposals


def _validate_source_diagnostics(
    source_pack: dict[str, Any], source_raw: bytes, valid_ids: list[str]
) -> tuple[dict[str, str], dict[str, int]]:
    diagnostics, aggregate = inventory_module._diagnose_tasks(
        source_pack, source_raw, valid_ids
    )
    if aggregate["n_tasks"] == 0 or aggregate["n_unknown_scope_references"] == 0:
        _fail("SCOPE_REPAIR_UNAVAILABLE")
    if (
        aggregate["n_ambiguous_scope_references"] != 0
        or aggregate["n_unmatched_scope_references"] != 0
        or aggregate["n_scope_proposal_occurrences"]
        != aggregate["n_unknown_scope_references"]
    ):
        _fail("SCOPE_REPAIR_INCOMPLETE")
    for task in diagnostics["tasks"]:
        codes = set(task["codes"])
        if not codes.issubset(_ALLOWED_SOURCE_CODES):
            _fail("SOURCE_TASKS_REQUIRE_REVIEW")
        if len(task["unknown_doc_ids"]) != len(task["scope_proposals"]):
            _fail("SCOPE_REPAIR_INCOMPLETE")
    return _proposal_map(diagnostics["scope_proposals"], valid_ids), aggregate


def _apply_proposals(
    source_pack: dict[str, Any], proposals: dict[str, str]
) -> tuple[dict[str, Any], int, int]:
    updated = deepcopy(source_pack)
    tasks_changed = 0
    references_changed = 0
    for row in updated["tasks"]:
        if not isinstance(row, dict):
            _fail("SOURCE_TASKS_REQUIRE_REVIEW")
        scope = normalize_doc_ids(row.get("doc_ids"))
        if not scope:
            _fail("SOURCE_TASKS_REQUIRE_REVIEW")
        repaired: list[str] = []
        changed = False
        for doc_id in scope:
            replacement = proposals.get(doc_id, doc_id)
            repaired.append(replacement)
            if replacement != doc_id:
                references_changed += 1
                changed = True
        normalized = normalize_doc_ids(repaired)
        if not normalized:
            _fail("SCOPE_REPAIR_VALIDATION_FAILED")
        row["doc_ids"] = normalized
        if changed:
            tasks_changed += 1
    return updated, tasks_changed, references_changed


def _serialize_pack(payload: dict[str, Any]) -> bytes:
    try:
        serialized = (
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError):
        _fail("SCOPE_REPAIR_SERIALIZATION_FAILED")
    if len(serialized) > MAX_PACK_BYTES:
        _fail("SCOPE_REPAIR_OUTPUT_TOO_LARGE")
    return serialized


def _publish(target: Path, payload: bytes) -> None:
    try:
        inventory_module._publish_no_clobber(target, payload)
    except inventory_module.StudyInventoryError as exc:
        mapping = {
            "INVENTORY_ALREADY_EXISTS": "REPAIR_OUTPUT_ALREADY_EXISTS",
            "INVENTORY_PARENT_INVALID": "REPAIR_OUTPUT_PARENT_INVALID",
            "INVENTORY_PARENT_MISSING": "REPAIR_OUTPUT_PARENT_MISSING",
            "INVENTORY_PARENT_CREATE_FAILED": "REPAIR_OUTPUT_PARENT_CREATE_FAILED",
            "INVENTORY_PARENT_CHANGED": "REPAIR_OUTPUT_PARENT_CHANGED",
            "INVENTORY_TEMP_CREATE_FAILED": "REPAIR_OUTPUT_TEMP_CREATE_FAILED",
            "INVENTORY_WRITE_FAILED": "REPAIR_OUTPUT_WRITE_FAILED",
            "INVENTORY_PUBLISH_FAILED": "REPAIR_OUTPUT_PUBLISH_FAILED",
            "INVENTORY_DURABILITY_UNCERTAIN": "REPAIR_OUTPUT_DURABILITY_UNCERTAIN",
        }
        _fail(mapping.get(exc.code, "REPAIR_OUTPUT_PUBLISH_FAILED"), status=exc.status)


def create_scope_repaired_pack(
    corpus: Path,
    source_tasks: Path,
    inventory: Path,
    output_tasks: Path,
    *,
    confirmed: bool,
) -> dict[str, Any]:
    """Create a new canonical pack with every fresh exact proposal applied."""
    if not confirmed:
        _fail("CONFIRMATION_REQUIRED")

    corpus_path = _canonical_corpus(Path(corpus))
    source_path = _canonical_task_path(Path(source_tasks), output=False)
    inventory_path = _canonical_inventory(Path(inventory))
    output_path = _canonical_task_path(Path(output_tasks), output=True)
    if output_path == source_path:
        _fail("OUTPUT_EQUALS_SOURCE")
    if output_path == inventory_path:
        _fail("OUTPUT_EQUALS_INVENTORY")
    if _is_within(output_path, corpus_path):
        _fail("OUTPUT_INSIDE_CORPUS")

    source_raw = _read_regular(source_path, limit=MAX_PACK_BYTES, kind="source")
    inventory_raw = _read_regular(
        inventory_path, limit=inventory_module.MAX_REPORT_BYTES, kind="inventory"
    )
    source_pack = _parse_source_pack(source_raw)
    recorded_inventory = _parse_inventory(inventory_raw)
    source_digest = _sha256(source_raw)
    if recorded_inventory["identity"]["task_pack_digest"] != source_digest:
        _fail("INVENTORY_TASK_PACK_DIGEST_MISMATCH")

    fresh_corpus, _ = inventory_module._build_inventory(corpus_path, None)
    if (
        recorded_inventory["identity"]["corpus_digest"]
        != fresh_corpus["identity"]["corpus_digest"]
    ):
        _fail("INVENTORY_CORPUS_DIGEST_MISMATCH")
    valid_ids = fresh_corpus["corpus"]["valid_document_ids"]
    if recorded_inventory["corpus"]["valid_document_ids"] != valid_ids:
        _fail("INVENTORY_CORPUS_IDS_MISMATCH")

    fresh_proposals, source_aggregate = _validate_source_diagnostics(
        source_pack, source_raw, valid_ids
    )
    recorded_proposals = _proposal_map(
        recorded_inventory["task_diagnostics"]["scope_proposals"], valid_ids
    )
    if recorded_proposals != fresh_proposals:
        _fail("INVENTORY_PROPOSALS_STALE_OR_PARTIAL")

    repaired, tasks_changed, references_changed = _apply_proposals(
        source_pack, fresh_proposals
    )
    repaired_raw = _serialize_pack(repaired)
    repaired_diagnostics, repaired_aggregate = inventory_module._diagnose_tasks(
        repaired, repaired_raw, valid_ids
    )
    if (
        repaired_aggregate["n_invalid_task_scopes"] != 0
        or repaired_aggregate["n_unknown_scope_references"] != 0
        or repaired_aggregate["n_ambiguous_scope_references"] != 0
        or repaired_aggregate["n_unmatched_scope_references"] != 0
        or any(
            set(task["codes"]) - {"TASK_MANUAL_BASELINE_INVALID"}
            for task in repaired_diagnostics["tasks"]
        )
    ):
        _fail("SCOPE_REPAIR_VALIDATION_FAILED")
    if (
        tasks_changed == 0
        or references_changed
        != source_aggregate["n_scope_proposal_occurrences"]
    ):
        _fail("SCOPE_REPAIR_VALIDATION_FAILED")

    _publish(output_path, repaired_raw)
    return {
        "schema": SCOPE_REPAIR_RECEIPT_SCHEMA,
        "status": "CREATED",
        "code": "SCOPE_REPAIR_CREATED",
        "n_tasks": len(repaired["tasks"]),
        "n_tasks_changed": tasks_changed,
        "n_scope_references_repaired": references_changed,
        "n_unique_scope_mappings": len(fresh_proposals),
        "source_task_pack_digest": source_digest,
        "inventory_digest": _sha256(inventory_raw),
        "output_task_pack_digest": _sha256(repaired_raw),
        "representative_ready": False,
        "next_action": "RERUN_INVENTORY_AND_STUDY_CHECK",
        "claim_boundary": (
            "This copy-on-write repair changes exact document scopes only; it does "
            "not establish task authenticity, manual baselines, or study readiness."
        ),
    }
