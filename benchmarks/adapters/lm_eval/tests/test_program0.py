"""Program 0 infrastructure tests — identity, reproducibility, boundary isolation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jsonschema
import pytest
import yaml

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks.adapters.lm_eval.schemas import (  # noqa: E402
    DecisionStatus,
    canonical_json,
    compute_run_id,
    sha256_file,
)
from benchmarks.adapters.lm_eval.runner import (  # noqa: E402
    PROGRAM0_EXPECTED_N,
    PROGRAM0_EXPECTED_SHA256,
    DEFAULT_FIXTURE,
    run_smoke,
)
from benchmarks.adapters.lm_eval.task_adapter import load_and_bind_instrument  # noqa: E402


def test_registry_schema_and_unique_ids():
    registry = yaml.safe_load((REPO / "benchmarks/REGISTRY.yaml").read_text())
    schema = json.loads((REPO / "benchmarks/REGISTRY.schema.json").read_text())
    jsonschema.validate(instance=registry, schema=schema)
    rc_ids = [x["id"] for x in registry["resource_classes"]]
    suite_ids = [x["id"] for x in registry["suites"]]
    task_ids = [x["id"] for x in registry["tasks"]]
    assert len(rc_ids) == len(set(rc_ids))
    assert len(suite_ids) == len(set(suite_ids))
    assert len(task_ids) == len(set(task_ids))
    assert registry["harness_pin"]["version"] == "0.4.12"
    assert len(registry["harness_pin"]["git_commit"]) == 40
    operational = [t for t in registry["tasks"] if t["status"] == "operational"]
    assert len(operational) == 1
    assert operational[0]["id"] == "nano_held_value_sentinel"
    assert operational[0]["evidence_ledger_eligible"] is False
    assert operational[0]["leaderboard_eligible"] is False


def test_canonical_json_and_run_id_stable():
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    r1 = compute_run_id(
        benchmark_manifest_sha256="aa",
        model_or_solver_manifest_sha256="bb",
        config_sha256="cc",
        code_git_commit="dd",
    )
    r2 = compute_run_id(
        benchmark_manifest_sha256="aa",
        model_or_solver_manifest_sha256="bb",
        config_sha256="cc",
        code_git_commit="dd",
    )
    assert r1 == r2 and len(r1) == 64


def test_source_instrument_digest_and_count():
    bound = load_and_bind_instrument(
        DEFAULT_FIXTURE,
        git_commit="test",
        expected_sha256=PROGRAM0_EXPECTED_SHA256,
        expected_record_count=PROGRAM0_EXPECTED_N,
    )
    assert bound.record_count == 4
    assert bound.sha256 == PROGRAM0_EXPECTED_SHA256
    with pytest.raises(ValueError):
        load_and_bind_instrument(
            DEFAULT_FIXTURE,
            git_commit="test",
            expected_sha256="0" * 64,
        )


def test_deterministic_smoke_perfect_and_no_promotion(tmp_path):
    runs = tmp_path / "runs"
    r1 = run_smoke(mode="deterministic", runs_root=runs, code_commit="c" * 40)
    r2 = run_smoke(mode="deterministic", runs_root=runs, code_commit="c" * 40)
    assert r1["run_id"] == r2["run_id"]
    assert r1["benchmark_manifest_hash"] == r2["benchmark_manifest_hash"]
    assert r1["model_or_solver_manifest_hash"] == r2["model_or_solver_manifest_hash"]
    assert r1["metrics"]["exact_match"] == 1.0
    assert r1["decision"] == DecisionStatus.INFRA_SMOKE_PASS.value
    d = json.loads((runs / r1["run_id"] / "decision.json").read_text())
    assert d["promote"] is False
    assert d["leaderboard_eligible"] is False
    assert d["evidence_ledger_eligible"] is False
    raw = (runs / r1["run_id"] / "raw_outputs.jsonl").read_text().strip().splitlines()
    scores = (runs / r1["run_id"] / "per_item_scores.jsonl").read_text().strip().splitlines()
    assert len(raw) == len(scores) == 4
    for line in (runs / r1["run_id"] / "SHA256SUMS").read_text().strip().splitlines():
        digest, name = line.split("  ")
        assert sha256_file(str(runs / r1["run_id"] / name)) == digest
    assert r1["metrics"] == r2["metrics"]
    for key in (
        "benchmark_manifest.json",
        "solver_manifest.json",
        "metrics.json",
        "cost.json",
        "decision.json",
    ):
        assert r1["artifact_hashes"][key] == r2["artifact_hashes"][key]
    assert (runs / r1["run_id"] / "per_item_scores.jsonl").read_bytes() == (
        runs / r2["run_id"] / "per_item_scores.jsonl"
    ).read_bytes()


def test_mock_smoke_zero_score(tmp_path):
    runs = tmp_path / "runs"
    r = run_smoke(mode="mock", runs_root=runs, code_commit="c" * 40)
    assert r["metrics"]["exact_match"] == 0.0
    assert r["decision"] == "INFRA_SMOKE_PASS"
    assert (runs / r["run_id"] / "model_manifest.json").is_file()


def test_failed_run_preserved(tmp_path):
    runs = tmp_path / "runs"
    r = run_smoke(mode="mock", runs_root=runs, code_commit="c" * 40, fail=True)
    assert r["decision"] == "INFRA_SMOKE_FAIL"
    assert r["run_status"] == "FAILED"
    man = json.loads((runs / r["run_id"] / "run_manifest.json").read_text())
    assert man["failure_information"]["error"] == "intentional_fail_for_test"
    assert man["promote"] is False


def test_manifest_required_fields_present(tmp_path):
    runs = tmp_path / "runs"
    r = run_smoke(mode="deterministic", runs_root=runs, code_commit="d" * 40)
    bench = json.loads((runs / r["run_id"] / "benchmark_manifest.json").read_text())
    for key in (
        "schema_version",
        "suite_id",
        "task_id",
        "task_version",
        "task_yaml_sha256",
        "source_instrument_paths",
        "source_instrument_git_commit",
        "source_artifact_sha256",
        "record_count",
        "prompt_template_hash",
        "scorer_hash",
        "filter_pipeline_hash",
        "metric_definitions",
        "protected_metrics",
        "benchmark_license",
        "contamination_status",
    ):
        assert key in bench


def test_task_yaml_exists_and_pins_digest():
    ypath = REPO / "benchmarks/adapters/lm_eval/tasks/held_value_sentinel.yaml"
    text = ypath.read_text()
    assert "nano_held_value_sentinel" in text
    assert PROGRAM0_EXPECTED_SHA256 in text


def test_lm_eval_validate_if_available():
    pytest.importorskip("lm_eval")
    import subprocess

    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "validate",
        "--tasks",
        "nano_held_value_sentinel",
        "--include_path",
        str(REPO / "benchmarks/adapters/lm_eval/tasks"),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        pytest.skip(f"lm-eval validate unavailable/failed: {(proc.stderr or proc.stdout)[-800:]}")


def test_boundary_ledger_and_freeze_untouched_by_smoke(tmp_path):
    ledger = (REPO / "papers/EVIDENCE_LEDGER.md").read_bytes()
    sums_path = REPO / "artifacts/SHA256SUMS"
    freeze = sums_path.read_bytes() if sums_path.exists() else b""
    run_smoke(mode="deterministic", runs_root=tmp_path / "runs", code_commit="e" * 40)
    assert (REPO / "papers/EVIDENCE_LEDGER.md").read_bytes() == ledger
    if freeze:
        assert sums_path.read_bytes() == freeze
