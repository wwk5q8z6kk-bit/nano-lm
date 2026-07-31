"""Append-oriented JSONL Chain-of-Evidence execution record.

Local, inspectable, deterministic, append-safe. Not claimed tamper-proof.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wedge_v1.coe.schema import digest_text


EVENT_TYPES = frozenset({
    "CORPUS_OPENED",
    "DOCUMENT_VERSION_SELECTED",
    "QUERY_NORMALIZED",
    "RETRIEVER_EXECUTED",
    "CANDIDATE_RETURNED",
    "SPAN_SELECTED",
    "CLAIM_CONSTRUCTED",
    "CONDITION_DECOMPOSED",
    "VERIFIER_EXECUTED",
    "CONTRADICTION_FOUND",
    "CLAIM_REJECTED",
    "CLAIM_PRESENTED",
    "USER_CORRECTED",
    "RUN_STARTED",
    "RUN_FINALIZED",
    "AUDIT_EXECUTED",
})


def _eid() -> str:
    return f"ev_{uuid.uuid4().hex[:12]}"


@dataclass
class EvidenceRecord:
    run_id: str
    path: Path
    component_version: str = "wedge_v1.coe.record.v1"
    parent_stack: list[str] = field(default_factory=list)
    n_events: int = 0
    _fh: Any = field(default=None, repr=False)

    @classmethod
    def create(cls, directory: Path, *, run_id: str | None = None) -> "EvidenceRecord":
        directory.mkdir(parents=True, exist_ok=True)
        rid = run_id or f"run_{uuid.uuid4().hex[:12]}"
        path = directory / f"{rid}.jsonl"
        rec = cls(run_id=rid, path=path)
        rec._fh = path.open("a", encoding="utf-8")
        rec.emit("RUN_STARTED", payload={"run_id": rid})
        return rec

    def emit(
        self,
        event_type: str,
        *,
        payload: dict | None = None,
        parent_ids: list[str] | None = None,
        input_digest: str | None = None,
        output_digest: str | None = None,
        failure_code: str | None = None,
        latency_ms: int | None = None,
    ) -> str:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"unknown event_type {event_type}")
        eid = _eid()
        parents = parent_ids if parent_ids is not None else list(self.parent_stack[-1:])
        body = {
            "event_id": eid,
            "run_id": self.run_id,
            "event_type": event_type,
            "parent_ids": parents,
            "timestamp": time.time(),
            "component_version": self.component_version,
            "input_digest": input_digest,
            "output_digest": output_digest,
            "payload": payload or {},
            "failure_code": failure_code,
            "latency_ms": latency_ms,
        }
        line = json.dumps(body, sort_keys=True, default=str)
        assert self._fh is not None
        self._fh.write(line + chr(10))
        self._fh.flush()
        self.n_events += 1
        self.parent_stack.append(eid)
        return eid

    def close(self) -> None:
        if self._fh:
            self.emit("RUN_FINALIZED", payload={"n_events": self.n_events})
            self._fh.close()
            self._fh = None

    def read_events(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out


def load_record(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def corpus_digest_from_path(corpus_dir: Path, docs: dict[str, str]) -> str:
    from wedge_v1.coe.schema import digest_docs

    return digest_docs(docs)
