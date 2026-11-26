"""Solver-neutral Nano inference boundary."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from .contract import ContractValidationError, NanoInput, NanoOutput


class SolverKind(str, Enum):
    TRAINED = "trained"
    HYBRID = "hybrid"
    REFERENCE = "reference"
    LEGACY_ADAPTER = "legacy_adapter"


@dataclass(frozen=True, slots=True)
class SolverDescriptor:
    """Stable identity and separately reported resource facts for a solver."""

    solver_id: str
    kind: SolverKind
    version: str = "0"
    parameter_count: int | None = None
    artifact_bytes: int | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.solver_id, str)
            or not self.solver_id
            or self.solver_id.strip() != self.solver_id
        ):
            raise ValueError("solver_id must be a non-empty, edge-trimmed string")
        if not isinstance(self.kind, SolverKind):
            raise TypeError("kind must be a SolverKind")
        if (
            not isinstance(self.version, str)
            or not self.version
            or self.version.strip() != self.version
        ):
            raise ValueError("version must be a non-empty, edge-trimmed string")
        for name, value in (
            ("parameter_count", self.parameter_count),
            ("artifact_bytes", self.artifact_bytes),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "solver_id": self.solver_id,
            "kind": self.kind.value,
            "version": self.version,
            "parameter_count": self.parameter_count,
            "artifact_bytes": self.artifact_bytes,
        }


@runtime_checkable
class NanoSolver(Protocol):
    @property
    def descriptor(self) -> SolverDescriptor: ...

    def infer(self, request: NanoInput) -> NanoOutput: ...


class InferenceFailureCategory(str, Enum):
    INVALID_REQUEST = "invalid_request"
    INVALID_SOLVER = "invalid_solver"
    SOLVER_EXCEPTION = "solver_exception"
    INVALID_OUTPUT_TYPE = "invalid_output_type"
    INVALID_DIAGNOSTICS = "invalid_diagnostics"
    SOLVER_ID_MISMATCH = "solver_id_mismatch"
    CONTRACT_VIOLATION = "contract_violation"


@dataclass(frozen=True, slots=True)
class InferenceFailure:
    category: InferenceFailureCategory
    message: str
    exception_type: str | None = None
    validation_code: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "category": self.category.value,
            "message": self.message,
            "exception_type": self.exception_type,
            "validation_code": self.validation_code,
        }


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Exactly one validated output or structured failure."""

    output: NanoOutput | None = None
    failure: InferenceFailure | None = None
    diagnostics: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if (self.output is None) == (self.failure is None):
            raise ValueError("exactly one of output or failure is required")
        if self.failure is not None and self.diagnostics is not None:
            raise ValueError("failed inference cannot carry diagnostics")
        if self.diagnostics is not None and not isinstance(self.diagnostics, Mapping):
            raise TypeError("diagnostics must be a mapping or None")

    @property
    def ok(self) -> bool:
        return self.output is not None


def _failure(
    category: InferenceFailureCategory,
    message: str,
    *,
    exception_type: str | None = None,
    validation_code: str | None = None,
) -> InferenceResult:
    return InferenceResult(
        failure=InferenceFailure(
            category=category,
            message=message,
            exception_type=exception_type,
            validation_code=validation_code,
        )
    )


class _InvalidDiagnostics(ValueError):
    """An optional diagnostic inference result is not JSON-safe."""


def _json_safe_copy(value: object, *, path: str = "$", depth: int = 0) -> Any:
    """Copy a value into the strict JSON data model or reject it.

    ``json.dumps`` accepts conveniences such as integer mapping keys, tuples, and
    non-finite floats.  Diagnostics cross an evidence boundary, so this validator
    accepts only actual JSON primitives and string-keyed mappings.  The copy also
    prevents a solver from mutating its result after validation.
    """

    if depth > 64:
        raise _InvalidDiagnostics(f"{path}: diagnostic value is nested too deeply")
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise _InvalidDiagnostics(f"{path}: diagnostic float must be finite")
        return value
    if type(value) is list:
        return [
            _json_safe_copy(item, path=f"{path}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise _InvalidDiagnostics(
                    f"{path}: diagnostic mapping keys must be strings"
                )
            result[key] = _json_safe_copy(
                item,
                path=f"{path}.{key}",
                depth=depth + 1,
            )
        return result
    raise _InvalidDiagnostics(
        f"{path}: {type(value).__name__} is not a JSON-safe diagnostic value"
    )


def _validated_diagnostics(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _InvalidDiagnostics("diagnostics must be a mapping")
    try:
        copied = _json_safe_copy(value)
    except _InvalidDiagnostics:
        raise
    except Exception as exc:
        raise _InvalidDiagnostics("diagnostics could not be read safely") from exc
    if not isinstance(copied, dict):  # pragma: no cover - guarded above.
        raise _InvalidDiagnostics("diagnostics must be a mapping")
    # Exercise the actual serializer as a final boundary check. ``allow_nan`` is
    # defensive even though the recursive validator already rejects NaN/Inf.
    try:
        json.dumps(copied, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise _InvalidDiagnostics("diagnostics are not JSON serializable") from exc
    return copied


def run_inference(
    solver: NanoSolver,
    request: NanoInput,
    *,
    expected_descriptor: SolverDescriptor | None = None,
) -> InferenceResult:
    """Run one inference and fail closed on every boundary violation.

    Failures are represented separately.  In particular, the runner never
    fabricates a valid-looking ``missing`` or other abstention output.
    """

    if not isinstance(request, NanoInput):
        return _failure(
            InferenceFailureCategory.INVALID_REQUEST,
            "request must be a NanoInput",
        )
    if expected_descriptor is not None and not isinstance(
        expected_descriptor, SolverDescriptor
    ):
        return _failure(
            InferenceFailureCategory.INVALID_SOLVER,
            "expected_descriptor must be a SolverDescriptor or None",
        )

    try:
        descriptor = solver.descriptor
    except Exception as exc:  # noqa: BLE001 - isolate untrusted solver code.
        return _failure(
            InferenceFailureCategory.INVALID_SOLVER,
            "solver descriptor could not be read",
            exception_type=type(exc).__name__,
        )
    try:
        infer = solver.infer
    except Exception as exc:  # noqa: BLE001 - isolate untrusted solver code.
        return _failure(
            InferenceFailureCategory.INVALID_SOLVER,
            "solver infer method could not be read",
            exception_type=type(exc).__name__,
        )
    try:
        infer_with_diagnostics = getattr(solver, "infer_with_diagnostics", None)
    except Exception as exc:  # noqa: BLE001 - isolate untrusted solver code.
        return _failure(
            InferenceFailureCategory.INVALID_SOLVER,
            "solver diagnostic inference method could not be read",
            exception_type=type(exc).__name__,
        )
    if not isinstance(descriptor, SolverDescriptor) or not callable(infer):
        return _failure(
            InferenceFailureCategory.INVALID_SOLVER,
            "solver must expose a SolverDescriptor and callable infer method",
        )
    if infer_with_diagnostics is not None and not callable(infer_with_diagnostics):
        return _failure(
            InferenceFailureCategory.INVALID_SOLVER,
            "solver infer_with_diagnostics attribute must be callable or absent",
        )
    if expected_descriptor is not None and descriptor != expected_descriptor:
        return _failure(
            InferenceFailureCategory.INVALID_SOLVER,
            "solver descriptor changed during the evaluation run",
        )

    diagnostics: dict[str, Any] | None = None
    try:
        if infer_with_diagnostics is None:
            output = infer(request)
        else:
            diagnostic_result = infer_with_diagnostics(request)
            if not isinstance(diagnostic_result, tuple) or len(diagnostic_result) != 2:
                return _failure(
                    InferenceFailureCategory.INVALID_DIAGNOSTICS,
                    "infer_with_diagnostics must return a two-item tuple",
                )
            output, raw_diagnostics = diagnostic_result
            diagnostics = _validated_diagnostics(raw_diagnostics)
    except ContractValidationError as exc:
        return _failure(
            InferenceFailureCategory.CONTRACT_VIOLATION,
            exc.message,
            exception_type=type(exc).__name__,
            validation_code=exc.code,
        )
    except _InvalidDiagnostics as exc:
        return _failure(
            InferenceFailureCategory.INVALID_DIAGNOSTICS,
            str(exc),
            exception_type=type(exc).__name__,
        )
    except Exception as exc:  # noqa: BLE001 - inference failures are report data.
        return _failure(
            InferenceFailureCategory.SOLVER_EXCEPTION,
            str(exc) or "solver raised an exception",
            exception_type=type(exc).__name__,
        )

    if not isinstance(output, NanoOutput):
        return _failure(
            InferenceFailureCategory.INVALID_OUTPUT_TYPE,
            "solver must return a NanoOutput",
        )
    try:
        final_descriptor = solver.descriptor
    except Exception as exc:  # noqa: BLE001 - isolate untrusted solver code.
        return _failure(
            InferenceFailureCategory.INVALID_SOLVER,
            "solver descriptor could not be re-read after inference",
            exception_type=type(exc).__name__,
        )
    if final_descriptor != descriptor:
        return _failure(
            InferenceFailureCategory.INVALID_SOLVER,
            "solver descriptor changed during inference",
        )
    if output.solver_id != descriptor.solver_id:
        return _failure(
            InferenceFailureCategory.SOLVER_ID_MISMATCH,
            "output solver_id does not match the solver descriptor",
        )
    try:
        output.validate_against(request)
    except ContractValidationError as exc:
        return _failure(
            InferenceFailureCategory.CONTRACT_VIOLATION,
            exc.message,
            exception_type=type(exc).__name__,
            validation_code=exc.code,
        )

    return InferenceResult(output=output, diagnostics=diagnostics)
