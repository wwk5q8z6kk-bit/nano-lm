"""Shared JSON validation helpers for artifact specs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from nanoscribe.artifacts.errors import artifact_fail


def load_json(raw: str) -> Any:
    if not isinstance(raw, str):
        artifact_fail("type_error", "expected JSON text", "$")
    try:
        return json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        artifact_fail("invalid_json", str(exc), "$")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            artifact_fail("duplicate_key", f"duplicate JSON key: {key}", "$")
        result[key] = value
    return result


def require_mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        artifact_fail("type_error", "expected an object", path)
    if any(not isinstance(key, str) for key in value):
        artifact_fail("key_type", "object keys must be strings", path)
    return value


def require_nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        artifact_fail("invalid_string", "expected a non-empty, edge-trimmed string", path)
    return value


def require_string_list(value: object, path: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        artifact_fail("type_error", "expected an array", path)
    if any(not isinstance(item, str) for item in value):
        artifact_fail("type_error", "expected an array of strings", path)
    return tuple(value)


def require_schema_version(value: object, expected: str, path: str) -> str:
    if value != expected:
        artifact_fail("schema_version", f"schema_version must be {expected}", path)
    return str(value)
