"""Mechanism laboratory for Nano Foundation Intelligence (X03–X05, X07).

Keep package top-level imports pure Python stdlib so CI environments without
PyTorch can import and test ``dual_memory`` cleanly. PyTorch-dependent sequence
cells live in ``nano.mechanisms.recurrent_cells`` and are imported on demand.
"""

from __future__ import annotations

from nano.mechanisms.dual_memory import (
    CompressedFastSketch,
    DeletionReceipt,
    DualMemoryController,
    EvidenceRecord,
    ExactEvidenceStore,
    MemoryQueryResult,
)

__all__ = [
    "CompressedFastSketch",
    "DeletionReceipt",
    "DualMemoryController",
    "EvidenceRecord",
    "ExactEvidenceStore",
    "MemoryQueryResult",
]
