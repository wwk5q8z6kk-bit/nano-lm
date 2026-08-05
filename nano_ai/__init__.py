"""Nano's versioned, product-independent scribe AI contract."""

from .contract import (
    CONTRACT_VERSION,
    FIELD_ORDER,
    ContractValidationError,
    EvidenceSpan,
    FieldName,
    FieldOutput,
    FieldState,
    NanoInput,
    NanoOutput,
    normalize_value,
)
from .solver import (
    InferenceFailure,
    InferenceFailureCategory,
    InferenceResult,
    NanoSolver,
    SolverDescriptor,
    SolverKind,
    run_inference,
)

__all__ = [
    "CONTRACT_VERSION",
    "FIELD_ORDER",
    "ContractValidationError",
    "EvidenceSpan",
    "FieldName",
    "FieldOutput",
    "FieldState",
    "InferenceFailure",
    "InferenceFailureCategory",
    "InferenceResult",
    "NanoInput",
    "NanoOutput",
    "NanoSolver",
    "SolverDescriptor",
    "SolverKind",
    "normalize_value",
    "run_inference",
]
