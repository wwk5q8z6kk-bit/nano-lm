"""Canonical projections for comparing CoE-bearing public results."""
from __future__ import annotations

import hashlib
import json
from typing import Any


_VOLATILE_KEYS = {
    "built_at",
    "corpus_dir",
    "execution_event_ids",
    "latency_ms",
    "latency_s",
    "record_path",
    "run_id",
    "timestamp",
    "ts",
}

_SCALAR_ID_KEYS = {
    "atom_id": "atom",
    "claim_id": "claim",
    "event_id": "event",
}

_LIST_ID_KEYS = {
    "contradiction_ids": "claim",
    "evidence_atom_ids": "atom",
    "invalid_claim_ids": "claim",
    "parent_ids": "event",
}


class _Canonicalizer:
    def __init__(self) -> None:
        self._ids: dict[str, dict[str, str]] = {
            "atom": {},
            "claim": {},
            "event": {},
        }

    def _id(self, kind: str, value: Any) -> Any:
        if not isinstance(value, str) or not value:
            return value
        seen = self._ids[kind]
        if value not in seen:
            seen[value] = f"<{kind}:{len(seen)}>"
        return seen[value]

    def visit(self, value: Any) -> Any:
        if isinstance(value, dict):
            out = {}
            for raw_key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
                key = str(raw_key)
                if key in _VOLATILE_KEYS:
                    continue
                if key in _SCALAR_ID_KEYS:
                    out[key] = self._id(_SCALAR_ID_KEYS[key], item)
                    continue
                if key in _LIST_ID_KEYS and isinstance(item, (list, tuple)):
                    kind = _LIST_ID_KEYS[key]
                    out[key] = [self._id(kind, child) for child in item]
                    continue
                out[key] = self.visit(item)
            return out
        if isinstance(value, (list, tuple)):
            return [self.visit(item) for item in value]
        if isinstance(value, set):
            return sorted((self.visit(item) for item in value), key=str)
        return value


def canonical_result(value: Any) -> Any:
    """Remove run-local noise while preserving identifier linkage structure."""
    return _Canonicalizer().visit(value)


def canonical_result_fingerprint(value: Any) -> str:
    raw = json.dumps(
        canonical_result(value),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
