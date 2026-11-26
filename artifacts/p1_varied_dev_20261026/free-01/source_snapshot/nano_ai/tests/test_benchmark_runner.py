from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from nano_ai import benchmark_runner
from nano_ai.benchmark_runner import (
    RUN_SCHEMA_VERSION,
    BenchmarkRunError,
    RunnerDependencies,
    _git_state,
    _source_hashes,
    _write_create_only,
    run_benchmark,
)

_DIGEST = "a" * 64
_PARTITION_DIGEST = "b" * 64


@dataclass(frozen=True)
class _Descriptor:
    solver_id: str = "fake/solver"

    def to_dict(self) -> dict[str, Any]:
        return {
            "solver_id": self.solver_id,
            "kind": "reference",
            "version": "test",
            "parameter_count": 0,
            "artifact_bytes": 0,
        }


class _Evaluation:
    operational: ClassVar[dict[str, Any]] = {
        "latency_measured": True,
        "inference_latency_ms_p50": 1.5,
        "inference_latency_ms_p95": 2.5,
    }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "test-evaluation",
            "items": [{"case_id": "fresh-0001", "status": "failure"}],
        }


class _Report:
    def __init__(self, value: dict[str, Any]) -> None:
        self.value = value

    def to_dict(self) -> dict[str, Any]:
        return self.value


def _dependencies(
    events: list[str],
    *,
    inference_ok: bool = True,
    change_sources: bool = False,
) -> RunnerDependencies:
    descriptor = _Descriptor()
    solver = SimpleNamespace(descriptor=descriptor)
    historical = SimpleNamespace(
        case_id="historical-0001",
        request=object(),
        provenance={"historical_source": {"partition": "old"}},
    )
    fresh_case = SimpleNamespace(case_id="fresh-0001")
    sealed = SimpleNamespace(
        benchmark_id="nano-fresh-v0",
        status="sealed_unmeasured",
        manifest_sha256=_DIGEST,
        partition_sha256=_PARTITION_DIGEST,
        cases=(fresh_case,),
    )
    source_calls = 0

    def configure_runtime() -> dict[str, Any]:
        events.append("configure")
        return {"torch_num_threads": 1}

    def build_solver(tag: str, root: Path, device: str) -> Any:
        events.append("build")
        assert tag == "deterministic"
        assert device == "cpu"
        assert root.is_dir()
        return solver

    def verify_artifacts(candidate: Any, root: Path) -> dict[str, Any]:
        events.append("verify")
        assert candidate is solver
        return {"kind": "none", "total_bytes": 0, "files": {}}

    def load_historical_cases() -> tuple[Any, ...]:
        events.append("historical")
        return (historical,)

    def run_inference(candidate: Any, request: Any, **kwargs: Any) -> Any:
        events.append("inference")
        assert candidate is solver
        assert request is historical.request
        assert kwargs["expected_descriptor"] is descriptor
        failure = SimpleNamespace(category="synthetic")
        return SimpleNamespace(ok=inference_ok, failure=failure)

    def load_sealed(path: Path, **kwargs: Any) -> Any:
        events.append("sealed")
        assert events.count("inference") == 2
        assert kwargs == {
            "expected_manifest_sha256": _DIGEST,
            "repository_root": path.parent,
        }
        return sealed

    def evaluate(candidate: Any, cases: Any, **kwargs: Any) -> _Evaluation:
        events.append("evaluate")
        assert candidate is solver
        assert tuple(cases) == (fresh_case,)
        assert kwargs == {"measure_latency": True}
        return _Evaluation()

    def aggregate(cases: Any, evaluation: Any) -> _Report:
        events.append("aggregate")
        assert tuple(cases) == (fresh_case,)
        assert isinstance(evaluation, _Evaluation)
        return _Report({"aggregate": True})

    def attach(
        report: _Report,
        *,
        resources: dict[str, Any],
        environment: dict[str, Any],
    ) -> _Report:
        events.append("attach")
        assert report.to_dict() == {"aggregate": True}
        return _Report(
            {
                "aggregate": True,
                "execution_provenance": {
                    "resources": resources,
                    "environment": environment,
                },
            }
        )

    def source_hashes(root: Path) -> dict[str, str]:
        nonlocal source_calls
        source_calls += 1
        events.append("sources")
        digest = "d" * 64 if change_sources and source_calls == 2 else "c" * 64
        return {"nano_ai/fake.py": digest}

    ticks = iter(range(0, 100_000_000, 1_000_000))
    return RunnerDependencies(
        configure_runtime=configure_runtime,
        build_solver=build_solver,
        verify_artifacts=verify_artifacts,
        load_historical_cases=load_historical_cases,
        run_inference=run_inference,
        load_sealed_benchmark=load_sealed,
        evaluate_solver=evaluate,
        aggregate_report=aggregate,
        attach_provenance=attach,
        source_hashes=source_hashes,
        git_state=lambda root: {
            "head": "e" * 40,
            "dirty": True,
            "porcelain_v1_z_sha256": "f" * 64,
        },
        peak_rss_bytes=lambda: 1234,
        clock_ns=lambda: next(ticks),
    )


def _run(tmp_path: Path, dependencies: RunnerDependencies) -> dict[str, Any]:
    return dict(
        run_benchmark(
            solver_tag="deterministic",
            manifest_path=tmp_path / "manifest.json",
            expected_manifest_sha256=_DIGEST,
            output_path=tmp_path / "run.json",
            repository_root=tmp_path,
            dependencies=dependencies,
        )
    )


def test_run_preflights_before_sealed_load_and_preserves_evidence_rows(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    envelope = _run(tmp_path, _dependencies(events))

    assert envelope["schema_version"] == RUN_SCHEMA_VERSION
    assert envelope["evaluation"]["items"] == [
        {"case_id": "fresh-0001", "status": "failure"}
    ]
    provenance = envelope["benchmark_report"]["execution_provenance"]
    benchmark = provenance["resources"]["benchmark"]
    assert benchmark["manifest_sha256"] == _DIGEST
    assert benchmark["partition_sha256"] == _PARTITION_DIGEST
    assert provenance["environment"]["evaluation"]["peak_rss_bytes"] == 1234
    assert provenance["environment"]["preflight"] == {
        "case_id": "historical-0001",
        "cold_inference_ms": 1.0,
        "warm_inference_ms": 1.0,
    }
    assert events.index("verify") < events.index("historical")
    assert [event for event in events if event == "inference"] == [
        "inference",
        "inference",
    ]
    assert events.index("sealed") > max(
        index for index, event in enumerate(events) if event == "inference"
    )
    assert json.loads((tmp_path / "run.json").read_bytes()) == envelope


def test_failed_historical_preflight_never_opens_fresh_partition(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    with pytest.raises(BenchmarkRunError, match="historical cold inference failed"):
        _run(tmp_path, _dependencies(events, inference_ok=False))

    assert "sealed" not in events
    assert not (tmp_path / "run.json").exists()


def test_source_change_fails_closed_without_output(tmp_path: Path) -> None:
    events: list[str] = []
    with pytest.raises(BenchmarkRunError, match="source changed"):
        _run(tmp_path, _dependencies(events, change_sources=True))

    assert "sealed" in events
    assert "attach" not in events
    assert not (tmp_path / "run.json").exists()


def test_existing_output_is_rejected_before_execution(tmp_path: Path) -> None:
    output = tmp_path / "run.json"
    output.write_text("owned", encoding="utf-8")
    events: list[str] = []

    with pytest.raises(FileExistsError):
        _run(tmp_path, _dependencies(events))

    assert output.read_text(encoding="utf-8") == "owned"
    assert events == []


def test_protocol_rejects_non_cpu_and_unpinned_manifest(tmp_path: Path) -> None:
    events: list[str] = []
    dependencies = _dependencies(events)
    common = {
        "solver_tag": "deterministic",
        "manifest_path": tmp_path / "manifest.json",
        "output_path": tmp_path / "run.json",
        "repository_root": tmp_path,
        "dependencies": dependencies,
    }
    with pytest.raises(BenchmarkRunError, match="only CPU"):
        run_benchmark(
            **common,
            expected_manifest_sha256=_DIGEST,
            device="mps",
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        run_benchmark(
            **common,
            expected_manifest_sha256="unpinned",
        )
    assert events == []


def test_atomic_writer_never_clobbers_existing_output(tmp_path: Path) -> None:
    target = tmp_path / "result.json"
    target.write_bytes(b"original")

    with pytest.raises(FileExistsError):
        _write_create_only(target, b"replacement")

    assert target.read_bytes() == b"original"
    assert list(tmp_path.glob(".result.json.*.tmp")) == []


def test_git_state_accepts_and_labels_sha1_object_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(arguments: list[str], **kwargs: Any) -> Any:
        del kwargs
        stdout = b"1" * 40 + b"\n" if arguments[-1] == "HEAD" else b" M file\0"
        return SimpleNamespace(stdout=stdout)

    monkeypatch.setattr(benchmark_runner.subprocess, "run", fake_run)

    state = _git_state(tmp_path)

    assert state["head"] == "1" * 40
    assert state["object_format"] == "sha1"
    assert state["oid_hex_length"] == 40
    assert state["dirty"] is True


def test_source_hashing_rejects_symlinks_before_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target.py"
    target.write_text("pass\n", encoding="utf-8")
    (tmp_path / "source.py").symlink_to(target)
    monkeypatch.setattr(benchmark_runner, "_SOURCE_PATHS", ("source.py",))

    with pytest.raises(BenchmarkRunError, match="uses a symlink"):
        _source_hashes(tmp_path)


def test_source_provenance_covers_imported_package_initializers() -> None:
    assert {
        "nano_ai/__init__.py",
        "nano_ai/adapters/__init__.py",
        "nano_ai/benchmarks/__init__.py",
    }.issubset(benchmark_runner._SOURCE_PATHS)
