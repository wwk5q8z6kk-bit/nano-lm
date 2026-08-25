"""Corpus manifest read/write helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    path: Path
    payload: dict[str, Any]

    @property
    def corpus_id(self) -> str:
        return str(self.payload.get("corpus_id", ""))

    @property
    def content_hash(self) -> str:
        return str(self.payload.get("content_hash", ""))

    @property
    def gates(self) -> dict[str, Any]:
        return dict(self.payload.get("gates", {}))

    @property
    def all_gates_pass(self) -> bool:
        return bool(self.gates.get("all_pass"))

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


def load_manifest(path: str | Path) -> CorpusManifest:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text())
    return CorpusManifest(path=manifest_path, payload=payload)


def write_manifest(path: str | Path, payload: dict[str, Any]) -> Path:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
    return manifest_path
