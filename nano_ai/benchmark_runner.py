"""Fail-closed execution of the frozen Nano scribe benchmark.

The runner deliberately keeps model preflight separate from fresh benchmark
access.  A sealed partition is not opened until artifacts are verified and the
solver has completed cold and warm inference on one historical smoke case.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any

RUN_SCHEMA_VERSION = "nano.benchmark-run.v0"
PROTOCOL_ID = "fresh-v0-cpu-single-thread-no-retry"
_REEXEC_MARKER = "NANO_BENCHMARK_PROTOCOL_REEXEC"
_REQUIRED_ENVIRONMENT = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}
_SOURCE_PATHS = (
    "nano_ai/__init__.py",
    "nano_ai/benchmark_runner.py",
    "nano_ai/benchmarking.py",
    "nano_ai/benchmarks/__init__.py",
    "nano_ai/benchmarks/fresh_v0.py",
    "nano_ai/contract.py",
    "nano_ai/evaluation.py",
    "nano_ai/fixtures.py",
    "nano_ai/solver.py",
    "nano_ai/adapters/__init__.py",
    "nano_ai/adapters/anchor_checkpoint.py",
    "nano_ai/adapters/deterministic_v0.py",
    "nano_ai/adapters/legacy_summary.py",
)


class BenchmarkRunError(RuntimeError):
    """The frozen execution protocol could not be completed safely."""


@dataclass(frozen=True, slots=True)
class RunnerDependencies:
    """Narrow dependency seam used by tests without loading real models/data."""

    configure_runtime: Callable[[], Mapping[str, Any]]
    build_solver: Callable[[str, Path, str], Any]
    verify_artifacts: Callable[[Any, Path], Mapping[str, Any]]
    load_historical_cases: Callable[[], Sequence[Any]]
    run_inference: Callable[..., Any]
    load_sealed_benchmark: Callable[..., Any]
    evaluate_solver: Callable[..., Any]
    aggregate_report: Callable[[Sequence[Any], Any], Any]
    attach_provenance: Callable[..., Any]
    source_hashes: Callable[[Path], Mapping[str, str]]
    git_state: Callable[[Path], Mapping[str, Any]]
    peak_rss_bytes: Callable[[], int]
    clock_ns: Callable[[], int] = perf_counter_ns


def _valid_sha256(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_git_oid(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value)
    )


def _configure_runtime() -> Mapping[str, Any]:
    mismatches = {
        name: os.environ.get(name)
        for name, expected in _REQUIRED_ENVIRONMENT.items()
        if os.environ.get(name) != expected
    }
    if mismatches:
        raise BenchmarkRunError(
            "benchmark process environment is not frozen; launch through the CLI"
        )

    try:
        import torch
    except ImportError as exc:
        raise BenchmarkRunError("Torch is required by the frozen protocol") from exc

    try:
        torch.set_num_interop_threads(1)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=False)
    except RuntimeError as exc:
        raise BenchmarkRunError(
            "Torch threading/determinism was initialized before protocol setup"
        ) from exc
    if (
        torch.get_num_interop_threads() != 1
        or torch.get_num_threads() != 1
        or not torch.are_deterministic_algorithms_enabled()
    ):
        raise BenchmarkRunError("Torch did not accept the frozen runtime settings")

    try:
        tokenizers_version = importlib.metadata.version("tokenizers")
    except importlib.metadata.PackageNotFoundError as exc:
        raise BenchmarkRunError(
            "tokenizers is required by the frozen protocol"
        ) from exc
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch": str(torch.__version__),
        "tokenizers": tokenizers_version,
        "environment": dict(_REQUIRED_ENVIRONMENT),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "torch_deterministic_algorithms": True,
    }


def _build_solver(tag: str, repository_root: Path, device: str) -> Any:
    if tag == "deterministic":
        from nano_ai.adapters.deterministic_v0 import DeterministicV0Solver

        return DeterministicV0Solver()
    from nano_ai.adapters.anchor_checkpoint import AnchorCheckpointSolver

    return AnchorCheckpointSolver.from_repository(
        tag,
        repository_root=repository_root,
        device=device,
    )


def _repository_relative(path: Path, repository_root: Path) -> str:
    try:
        return (
            path.resolve(strict=True)
            .relative_to(repository_root.resolve(strict=True))
            .as_posix()
        )
    except (OSError, ValueError) as exc:
        raise BenchmarkRunError("solver artifact is outside the repository") from exc


def _verify_artifacts(solver: Any, repository_root: Path) -> Mapping[str, Any]:
    generator = getattr(solver, "generator", None)
    if generator is None:
        descriptor = getattr(solver, "descriptor", None)
        if getattr(descriptor, "solver_id", None) != "reference/deterministic-v0":
            raise BenchmarkRunError("solver does not expose verifiable artifacts")
        return {"kind": "none", "total_bytes": 0, "files": {}}

    verified = generator.verify()
    checkpoint_bytes = verified.checkpoint_bytes
    tokenizer_bytes = verified.tokenizer_bytes
    return {
        "kind": "hash-gated-anchor",
        "identity": verified.artifact_identity,
        "total_bytes": len(checkpoint_bytes) + len(tokenizer_bytes),
        "files": {
            "checkpoint": {
                "path": _repository_relative(verified.checkpoint_path, repository_root),
                "sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
                "bytes": len(checkpoint_bytes),
            },
            "tokenizer": {
                "path": _repository_relative(verified.tokenizer_path, repository_root),
                "sha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
                "bytes": len(tokenizer_bytes),
            },
        },
    }


def _load_historical_cases() -> Sequence[Any]:
    from nano_ai.fixtures import load_contract_smoke

    return load_contract_smoke()


def _source_hashes(repository_root: Path) -> Mapping[str, str]:
    result: dict[str, str] = {}
    root = repository_root.resolve(strict=True)
    for relative in _SOURCE_PATHS:
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise BenchmarkRunError(f"source escapes repository: {relative}")
        cursor = root
        try:
            for part in relative_path.parts:
                cursor /= part
                if stat.S_ISLNK(os.lstat(cursor).st_mode):
                    raise BenchmarkRunError(f"source uses a symlink: {relative}")
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(cursor, flags)
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise BenchmarkRunError(f"source is not a regular file: {relative}")
                with os.fdopen(descriptor, "rb") as handle:
                    descriptor = -1
                    payload = handle.read()
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
        except BenchmarkRunError:
            raise
        except OSError as exc:
            raise BenchmarkRunError(f"cannot read source: {relative}") from exc
        result[relative] = hashlib.sha256(payload).hexdigest()
    return result


def _git_state(repository_root: Path) -> Mapping[str, Any]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
            cwd=repository_root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BenchmarkRunError("cannot capture Git provenance") from exc
    try:
        head_text = head.decode("ascii")
    except UnicodeDecodeError as exc:
        raise BenchmarkRunError("Git HEAD is not ASCII") from exc
    if not _valid_git_oid(head_text):
        raise BenchmarkRunError("Git HEAD is not a full object identity")
    return {
        "head": head_text,
        "object_format": "sha1" if len(head_text) == 40 else "sha256",
        "oid_hex_length": len(head_text),
        "dirty": bool(status),
        "porcelain_v1_z_sha256": hashlib.sha256(status).hexdigest(),
    }


def _peak_rss_bytes() -> int:
    observed = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(observed if sys.platform == "darwin" else observed * 1024)


def _default_dependencies() -> RunnerDependencies:
    from nano_ai.benchmarking import (
        aggregate_benchmark_report,
        attach_execution_provenance,
    )
    from nano_ai.benchmarks.fresh_v0 import load_sealed_benchmark
    from nano_ai.evaluation import evaluate_solver
    from nano_ai.solver import run_inference

    return RunnerDependencies(
        configure_runtime=_configure_runtime,
        build_solver=_build_solver,
        verify_artifacts=_verify_artifacts,
        load_historical_cases=_load_historical_cases,
        run_inference=run_inference,
        load_sealed_benchmark=load_sealed_benchmark,
        evaluate_solver=evaluate_solver,
        aggregate_report=aggregate_benchmark_report,
        attach_provenance=attach_execution_provenance,
        source_hashes=_source_hashes,
        git_state=_git_state,
        peak_rss_bytes=_peak_rss_bytes,
    )


def _successful_preflight(result: Any, *, phase: str) -> None:
    if not getattr(result, "ok", False):
        failure = getattr(result, "failure", None)
        detail = getattr(failure, "category", None)
        raise BenchmarkRunError(f"historical {phase} inference failed: {detail}")


def _historical_case(cases: Sequence[Any]) -> Any:
    for case in cases:
        provenance = getattr(case, "provenance", None)
        if isinstance(provenance, Mapping) and "historical_source" in provenance:
            return case
    raise BenchmarkRunError("historical preflight fixture is unavailable")


def _platform_environment() -> Mapping[str, Any]:
    uname = platform.uname()
    return {
        "system": uname.system,
        "release": uname.release,
        "version": uname.version,
        "machine": uname.machine,
        "processor": uname.processor,
        "logical_cpu_count": os.cpu_count(),
    }


def _write_create_only(path: Path, payload: bytes) -> None:
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir():
        raise BenchmarkRunError("output parent is not a directory")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def run_benchmark(
    *,
    solver_tag: str,
    manifest_path: Path,
    expected_manifest_sha256: str,
    output_path: Path,
    repository_root: Path,
    device: str = "cpu",
    dependencies: RunnerDependencies | None = None,
) -> Mapping[str, Any]:
    """Execute exactly one solver run and create one auditable result envelope."""

    if solver_tag not in {"deterministic", "nano", "scale"}:
        raise ValueError("solver_tag must be deterministic, nano, or scale")
    if device != "cpu":
        raise BenchmarkRunError("the frozen benchmark protocol permits only CPU")
    if not _valid_sha256(expected_manifest_sha256):
        raise ValueError("expected_manifest_sha256 must be a lowercase SHA-256")
    repository_root = Path(repository_root).resolve(strict=True)
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)

    deps = _default_dependencies() if dependencies is None else dependencies
    runtime = dict(deps.configure_runtime())
    sources_before = dict(deps.source_hashes(repository_root))
    git = dict(deps.git_state(repository_root))
    solver = deps.build_solver(solver_tag, repository_root, device)
    descriptor = solver.descriptor

    # Artifact verification and historical execution intentionally precede the
    # first call capable of opening the fresh sealed partition.
    artifacts = dict(deps.verify_artifacts(solver, repository_root))
    historical = _historical_case(tuple(deps.load_historical_cases()))
    preflight: dict[str, Any] = {"case_id": historical.case_id}
    for phase in ("cold", "warm"):
        started = deps.clock_ns()
        result = deps.run_inference(
            solver,
            historical.request,
            expected_descriptor=descriptor,
        )
        preflight[f"{phase}_inference_ms"] = (deps.clock_ns() - started) / 1_000_000
        _successful_preflight(result, phase=phase)

    sealed = deps.load_sealed_benchmark(
        manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
        repository_root=repository_root,
    )
    evaluation_started = deps.clock_ns()
    evaluation = deps.evaluate_solver(solver, sealed.cases, measure_latency=True)
    evaluation_elapsed_ms = (deps.clock_ns() - evaluation_started) / 1_000_000
    aggregate = deps.aggregate_report(sealed.cases, evaluation)

    sources_after = dict(deps.source_hashes(repository_root))
    if sources_after != sources_before:
        raise BenchmarkRunError("benchmark source changed during execution")
    operational = dict(evaluation.operational)
    environment = {
        "protocol_id": PROTOCOL_ID,
        "device": device,
        "runtime": runtime,
        "platform": _platform_environment(),
        "git": git,
        "preflight": preflight,
        "evaluation": {
            "elapsed_ms": evaluation_elapsed_ms,
            "latency_measured": operational.get("latency_measured"),
            "inference_latency_ms_p50": operational.get("inference_latency_ms_p50"),
            "inference_latency_ms_p95": operational.get("inference_latency_ms_p95"),
            "peak_rss_bytes": deps.peak_rss_bytes(),
        },
    }
    resources = {
        "solver": descriptor.to_dict(),
        "artifacts": artifacts,
        "benchmark": {
            "benchmark_id": sealed.benchmark_id,
            "status": sealed.status,
            "manifest_sha256": sealed.manifest_sha256,
            "partition_sha256": sealed.partition_sha256,
            "case_count": len(sealed.cases),
        },
        "source_sha256": sources_before,
    }
    report = deps.attach_provenance(
        aggregate,
        resources=resources,
        environment=environment,
    )
    envelope = {
        "schema_version": RUN_SCHEMA_VERSION,
        "benchmark_report": report.to_dict(),
        "evaluation": evaluation.to_dict(),
    }
    encoded = json.dumps(
        envelope,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    _write_create_only(output_path, encoded)
    return envelope


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solver", required=True, choices=("deterministic", "nano", "scale")
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    return parser


def _reexec_for_protocol(arguments: Sequence[str]) -> None:
    if all(
        os.environ.get(key) == value for key, value in _REQUIRED_ENVIRONMENT.items()
    ):
        return
    if os.environ.get(_REEXEC_MARKER) == "1":
        raise BenchmarkRunError("failed to establish the frozen process environment")
    environment = os.environ.copy()
    environment.update(_REQUIRED_ENVIRONMENT)
    environment[_REEXEC_MARKER] = "1"
    os.execvpe(
        sys.executable,
        [sys.executable, "-m", "nano_ai.benchmark_runner", *arguments],
        environment,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    _reexec_for_protocol(arguments)
    args = _parser().parse_args(arguments)
    run_benchmark(
        solver_tag=args.solver,
        manifest_path=args.manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
        output_path=args.output,
        repository_root=args.repository_root,
        device=args.device,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a process
    raise SystemExit(main())


__all__ = [
    "PROTOCOL_ID",
    "RUN_SCHEMA_VERSION",
    "BenchmarkRunError",
    "RunnerDependencies",
    "main",
    "run_benchmark",
]
