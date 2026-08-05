"""Write-once JSONL Chain-of-Evidence execution record.

Local and inspectable. Existing run files are never appended to or replaced.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    def create(cls, directory: Path, *, run_id: str | None = None) -> EvidenceRecord:
        directory.mkdir(parents=True, exist_ok=True)
        rid = run_id or f"run_{uuid.uuid4().hex[:12]}"
        if not rid or rid in {".", ".."} or Path(rid).name != rid:
            raise ValueError("run_id must be a single path-safe name")
        path = directory / f"{rid}.jsonl"
        rec = cls(run_id=rid, path=path)
        # Exclusive creation makes run IDs write-once and avoids a check/open race.
        # If the path already exists, FileExistsError leaves its contents untouched.
        rec._fh = path.open("x", encoding="utf-8")
        try:
            rec.emit("RUN_STARTED", payload={"run_id": rid})
        except Exception:
            rec._fh.close()
            rec._fh = None
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            raise
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

    def discard(self) -> None:
        """Remove a newly-created run that failed before durable completion."""
        if self._fh:
            self._fh.close()
            self._fh = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def read_events(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out


def load_record(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def corpus_digest_from_path(corpus_dir: Path, docs: dict[str, str]) -> str:
    from wedge_v1.coe.schema import digest_docs

    return digest_docs(docs)
