"""Source-preserving dual-memory interface (MM01 / B06 / X07).

Pure Python standard library (zero third-party deps) so it runs in all CI
environments. Separates lossy fixed-value-capacity fast state (`CompressedFastSketch`)
from an immutable, content-addressed verbatim store (`ExactEvidenceStore`) with
supersession chains, conflict detection, verifiable deletion receipts, and
honest total byte + scan-operation accounting.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


def _sha256_hex(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """Immutable verbatim span record with subject, provenance, and bitemporal bounds."""

    record_id: str
    subject_id: str
    key: str
    verbatim_value: str
    source_id: str
    start: int
    end: int
    valid_time: int
    observed_time: int
    polarity: str = "supported"  # "supported" | "denied" | "uncertain"
    supersedes_id: str | None = None

    def __post_init__(self) -> None:
        if not self.subject_id or not self.key or not self.source_id:
            raise ValueError("subject_id, key, and source_id must be non-empty")
        if self.polarity not in {"supported", "denied", "uncertain"}:
            raise ValueError(f"invalid polarity: {self.polarity}")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("span offsets must satisfy 0 <= start < end")
        if not self.verbatim_value:
            raise ValueError("verbatim_value must be non-empty")

    @property
    def payload_bytes(self) -> int:
        return (
            len(self.verbatim_value.encode("utf-8"))
            + len(self.key.encode("utf-8"))
            + len(self.source_id.encode("utf-8"))
            + len(self.subject_id.encode("utf-8"))
            + 32  # fixed metadata overhead (offsets, timestamps, polarity, ids)
        )


@dataclass(frozen=True, slots=True)
class DeletionReceipt:
    """Cryptographic-digest receipt proving hard erasure of raw payload bytes."""

    receipt_id: str
    subject_id: str
    key: str
    deleted_record_ids: tuple[str, ...]
    bytes_freed: int
    observed_time: int


@dataclass(frozen=True, slots=True)
class MemoryQueryResult:
    """Typed output of querying either fast sketch or dual memory."""

    key: str
    status: str  # "supported" | "denied" | "conflicting" | "missing" | "deleted" | "uncertain"
    values: tuple[str, ...]
    record_ids: tuple[str, ...]
    source_exact: bool
    bytes_read: int
    scan_ops: int


class ExactEvidenceStore:
    """Append-only exact evidence store supporting supersession, conflict & hard deletion."""

    def __init__(self) -> None:
        self._records: dict[str, EvidenceRecord] = {}
        self._by_subject_key: dict[tuple[str, str], list[str]] = {}
        self._superseded: set[str] = set()
        self._deleted_tombstones: dict[tuple[str, str], DeletionReceipt] = {}
        self.scan_ops: int = 0

    def append(
        self,
        *,
        subject_id: str,
        key: str,
        verbatim_value: str,
        source_id: str,
        start: int,
        end: int,
        valid_time: int,
        observed_time: int,
        polarity: str = "supported",
        supersedes_id: str | None = None,
    ) -> EvidenceRecord:
        if supersedes_id is not None:
            if supersedes_id not in self._records:
                raise KeyError(f"cannot supersede unknown record {supersedes_id!r}")
            prior = self._records[supersedes_id]
            if prior.subject_id != subject_id:
                raise ValueError("cross-subject supersession forbidden")

        rec_id = "ev-" + _sha256_hex(
            {
                "s": subject_id,
                "k": key,
                "v": verbatim_value,
                "src": source_id,
                "a": start,
                "b": end,
                "vt": valid_time,
                "ot": observed_time,
                "p": polarity,
                "sup": supersedes_id or "",
            }
        )[:16]
        rec = EvidenceRecord(
            record_id=rec_id,
            subject_id=subject_id,
            key=key,
            verbatim_value=verbatim_value,
            source_id=source_id,
            start=start,
            end=end,
            valid_time=valid_time,
            observed_time=observed_time,
            polarity=polarity,
            supersedes_id=supersedes_id,
        )
        if supersedes_id is not None:
            self._superseded.add(supersedes_id)
        self._records[rec_id] = rec
        bucket = self._by_subject_key.setdefault((subject_id, key), [])
        if rec_id not in bucket:
            bucket.append(rec_id)
        # Active observation clears prior deletion tombstone for future writes on that key
        self._deleted_tombstones.pop((subject_id, key), None)
        return rec

    def delete_key(self, *, subject_id: str, key: str, observed_time: int) -> DeletionReceipt:
        """Hard-delete raw payload bytes for `(subject_id, key)` and issue receipt."""
        ids = tuple(self._by_subject_key.pop((subject_id, key), ()))
        freed = 0
        for rid in ids:
            rec = self._records.pop(rid, None)
            if rec is not None:
                freed += rec.payload_bytes
            self._superseded.discard(rid)
        receipt = DeletionReceipt(
            receipt_id="del-" + _sha256_hex({"s": subject_id, "k": key, "ids": list(ids), "t": observed_time})[:16],
            subject_id=subject_id,
            key=key,
            deleted_record_ids=ids,
            bytes_freed=freed,
            observed_time=observed_time,
        )
        self._deleted_tombstones[(subject_id, key)] = receipt
        return receipt

    def query(self, *, subject_id: str, key: str) -> MemoryQueryResult:
        self.scan_ops += 1
        bucket = self._by_subject_key.get((subject_id, key))
        if not bucket:
            if (subject_id, key) in self._deleted_tombstones:
                return MemoryQueryResult(key=key, status="deleted", values=(), record_ids=(), source_exact=True, bytes_read=32, scan_ops=1)
            return MemoryQueryResult(key=key, status="missing", values=(), record_ids=(), source_exact=True, bytes_read=0, scan_ops=1)

        active = [self._records[rid] for rid in bucket if rid in self._records and rid not in self._superseded]
        if not active:
            return MemoryQueryResult(key=key, status="missing", values=(), record_ids=(), source_exact=True, bytes_read=0, scan_ops=1)

        bytes_read = sum(r.payload_bytes for r in active)
        polarities = {r.polarity for r in active}
        distinct_vals = tuple(dict.fromkeys(r.verbatim_value for r in active))
        rec_ids = tuple(r.record_id for r in active)

        if polarities == {"uncertain"}:
            return MemoryQueryResult(key=key, status="uncertain", values=distinct_vals, record_ids=rec_ids, source_exact=True, bytes_read=bytes_read, scan_ops=1)
        if len(polarities) > 1 or (polarities == {"supported"} and len(distinct_vals) > 1):
            return MemoryQueryResult(key=key, status="conflicting", values=distinct_vals, record_ids=rec_ids, source_exact=True, bytes_read=bytes_read, scan_ops=1)
        only_pol = next(iter(polarities))
        status = "denied" if only_pol == "denied" else "supported"
        return MemoryQueryResult(key=key, status=status, values=distinct_vals, record_ids=rec_ids, source_exact=True, bytes_read=bytes_read, scan_ops=1)

    @property
    def evidence_store_bytes(self) -> int:
        return sum(r.payload_bytes for r in self._records.values())

    @property
    def index_overhead_bytes(self) -> int:
        # 48 bytes per indexed slot entry + tombstone metadata
        return len(self._records) * 48 + len(self._deleted_tombstones) * 64


class CompressedFastSketch:
    """Fixed-capacity lossy associative slot memory simulating capacity/collision interference."""

    def __init__(self, num_slots: int = 8, slot_bytes: int = 32) -> None:
        if num_slots <= 0 or slot_bytes <= 0:
            raise ValueError("num_slots and slot_bytes must be positive")
        self.num_slots = num_slots
        self.slot_bytes = slot_bytes
        self._slots: list[tuple[tuple[str, str], str, str] | None] = [None] * num_slots
        self.overwrites: int = 0
        self.scan_ops: int = 0

    def _slot_idx(self, subject_id: str, key: str) -> int:
        identity = json.dumps([subject_id, key], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        digest = hashlib.blake2b(identity, digest_size=8).digest()
        return int.from_bytes(digest, "little") % self.num_slots

    def write(self, subject_id: str, key: str, value: str, polarity: str = "supported") -> None:
        idx = self._slot_idx(subject_id, key)
        if self._slots[idx] is not None and self._slots[idx][0] != (subject_id, key):
            self.overwrites += 1
        truncated = value.encode("utf-8")[:self.slot_bytes].decode("utf-8", errors="ignore")
        self._slots[idx] = ((subject_id, key), truncated, polarity)

    def delete_key(self, subject_id: str, key: str) -> None:
        idx = self._slot_idx(subject_id, key)
        slot = self._slots[idx]
        if slot is not None and slot[0] == (subject_id, key):
            self._slots[idx] = None

    def query(self, subject_id: str, key: str) -> MemoryQueryResult:
        self.scan_ops += 1
        idx = self._slot_idx(subject_id, key)
        slot = self._slots[idx]
        if slot is None or slot[0] != (subject_id, key):
            return MemoryQueryResult(key=key, status="missing", values=(), record_ids=(), source_exact=False, bytes_read=self.slot_bytes, scan_ops=1)
        status = "denied" if slot[2] == "denied" else "uncertain" if slot[2] == "uncertain" else "supported"
        return MemoryQueryResult(key=key, status=status, values=(slot[1],), record_ids=(), source_exact=False, bytes_read=self.slot_bytes, scan_ops=1)

    @property
    def fast_value_capacity_bytes(self) -> int:
        return self.num_slots * self.slot_bytes

    @property
    def fast_state_bytes(self) -> int:
        """Logical byte capacity plus the actual UTF-8 identity and polarity payload.

        This excludes Python allocation and object overhead.
        """
        metadata = sum(
            len(text.encode("utf-8"))
            for slot in self._slots if slot is not None
            for text in (*slot[0], slot[2])
        )
        return self.fast_value_capacity_bytes + metadata


class DualMemoryController:
    """Dual-channel memory pairing fixed fast sketch with lossless exact evidence store (X07)."""

    def __init__(self, fast_slots: int = 8, slot_bytes: int = 32) -> None:
        self.fast = CompressedFastSketch(num_slots=fast_slots, slot_bytes=slot_bytes)
        self.store = ExactEvidenceStore()
        self.corruption_checks: int = 0
        self.corruption_failures: int = 0

    def commit(
        self,
        *,
        subject_id: str,
        key: str,
        verbatim_value: str,
        source_id: str,
        start: int,
        end: int,
        valid_time: int,
        observed_time: int,
        polarity: str = "supported",
        supersedes_id: str | None = None,
    ) -> EvidenceRecord:
        rec = self.store.append(
            subject_id=subject_id,
            key=key,
            verbatim_value=verbatim_value,
            source_id=source_id,
            start=start,
            end=end,
            valid_time=valid_time,
            observed_time=observed_time,
            polarity=polarity,
            supersedes_id=supersedes_id,
        )
        self.fast.write(subject_id, key, verbatim_value, polarity=polarity)
        return rec

    def delete(self, *, subject_id: str, key: str, observed_time: int) -> DeletionReceipt:
        self.fast.delete_key(subject_id, key)
        return self.store.delete_key(subject_id=subject_id, key=key, observed_time=observed_time)

    def query_dual(self, *, subject_id: str, key: str) -> MemoryQueryResult:
        """Query exact store first for verbatim fidelity; falls back cleanly to typed states."""
        return self.store.query(subject_id=subject_id, key=key)

    def query_fast_only(self, *, subject_id: str, key: str) -> MemoryQueryResult:
        return self.fast.query(subject_id=subject_id, key=key)

    def verify_no_corruption(self, *, subject_id: str, key: str, expected_verbatim: str) -> bool:
        self.corruption_checks += 1
        res = self.query_dual(subject_id=subject_id, key=key)
        ok = res.status in {"supported", "denied"} and expected_verbatim in res.values
        if not ok:
            self.corruption_failures += 1
        return ok

    def resource_accounting(self) -> dict[str, Any]:
        fast_b = self.fast.fast_state_bytes
        store_b = self.store.evidence_store_bytes
        idx_b = self.store.index_overhead_bytes
        return {
            "fast_state_bytes": fast_b,
            "fast_value_capacity_bytes": self.fast.fast_value_capacity_bytes,
            "byte_measure": "logical capacity and payload estimates, excluding Python object overhead",
            "evidence_store_bytes": store_b,
            "index_overhead_bytes": idx_b,
            "total_dual_bytes": fast_b + store_b + idx_b,
            "fast_overwrites": self.fast.overwrites,
            "store_scan_ops": self.store.scan_ops,
            "fast_scan_ops": self.fast.scan_ops,
            "empirical_corruption_rate": (
                self.corruption_failures / self.corruption_checks if self.corruption_checks else 0.0
            ),
        }
