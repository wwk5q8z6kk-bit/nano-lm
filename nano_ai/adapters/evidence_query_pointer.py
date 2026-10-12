"""Verification-gated adapter for Nano's H3 evidence-query predictions.

H3 changes only the trained state/pointer mechanism.  Its independent verifier
boundary is intentionally identical to H2: the model owns every proposed state
and evidence span, while the verifier may only accept the proposal or replace it
with an uncertain abstention.  It can never fill, repair, or substitute a value.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from nano_ai.adapters.pointer_span import PointerSpanSolver
from nano_ai.adapters.state_span import StateSpanProposal
from nano_ai.solver import SolverDescriptor, SolverKind
from nano_ai.training.evidence_query_model import (
    ARCHITECTURE_VERSION,
    NANO_EVIDENCE_QUERY_PARAMETER_COUNT,
)

EVIDENCE_QUERY_POINTER_PIPELINE_VERSION = ARCHITECTURE_VERSION


class EvidenceQueryPointerSolver(PointerSpanSolver):
    """Ground H3 proposals without changing any model-owned prediction."""

    def __init__(
        self,
        predict: Callable[[str], Sequence[StateSpanProposal]],
        *,
        solver_id: str = "candidate/nano-evidence-query-pointer-v1",
        version: str = EVIDENCE_QUERY_POINTER_PIPELINE_VERSION,
        artifact_bytes: int | None = None,
    ) -> None:
        super().__init__(
            predict,
            solver_id=solver_id,
            version=version,
            artifact_bytes=artifact_bytes,
        )
        self.descriptor = SolverDescriptor(
            solver_id=solver_id,
            kind=SolverKind.HYBRID,
            version=version,
            parameter_count=NANO_EVIDENCE_QUERY_PARAMETER_COUNT,
            artifact_bytes=artifact_bytes,
        )


__all__ = [
    "EVIDENCE_QUERY_POINTER_PIPELINE_VERSION",
    "EvidenceQueryPointerSolver",
]
