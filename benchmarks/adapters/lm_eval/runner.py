"""Content-addressed held-value regression runner with no claim promotion."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .model_adapter import (
    DeterministicTemplateSolver,
    MockedModel,
    Predictor,
    deterministic_solver_manifest,
    mocked_model_manifest,
)
from .schemas import (
    SCHEMA_VERSION,
    BenchmarkManifest,
    DecisionStatus,
    RunManifest,
    RunStatus,
    canonical_json,
    compute_run_id,
    sha256_file,
    sha256_hex,
)
from .task_adapter import (
    docs_from_instrument,
    exact_match,
    filter_pipeline_hash,
    load_and_bind_instrument,
    prompt_template_hash,
    scorer_hash,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
ADAPTER_ROOT = Path(__file__).resolve().parent
DEFAULT_FIXTURE = "benchmarks/adapters/lm_eval/fixtures/held_value_sentinel_n4.json"
DEFAULT_TASK_YAML = "benchmarks/adapters/lm_eval/tasks/held_value_sentinel.yaml"
DEFAULT_RUNS = (
    REPO_ROOT / "experiments/verification/sentinel/runs"
)

# Frozen sentinel binding; the fixture is authoritative for this regression only.
SENTINEL_EXPECTED_SHA256 = (
    "ed5e8171cf13a4e802ecc6635740e8ad3977064eea7e4149b331a20792dee0a2"
)
SENTINEL_EXPECTED_N = 4
TASK_ID = "nano_held_value_sentinel"
TASK_VERSION = "0.1.0"
SUITE_ID = "suite_sentinel"


def git_commit() -> str:
    env = os.environ.get("NANO_LM_CODE_GIT_COMMIT")
    if env:
        return env.strip()
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "UNKNOWN"


def build_benchmark_manifest(bound, task_yaml_path: Path) -> BenchmarkManifest:
    return BenchmarkManifest(
        schema_version=SCHEMA_VERSION,
        suite_id=SUITE_ID,
        task_id=TASK_ID,
        task_version=TASK_VERSION,
        task_yaml_sha256=sha256_file(str(task_yaml_path)),
        source_instrument_paths=[bound.repo_relative_path],
        source_instrument_git_commit=bound.git_commit,
        source_artifact_sha256=bound.sha256,
        record_count=bound.record_count,
        prompt_template_hash=prompt_template_hash(),
        scorer_hash=scorer_hash(),
        filter_pipeline_hash=filter_pipeline_hash(),
        metric_definitions=[
            {"name": "exact_match", "higher_is_better": True},
            {"name": "exact_match_held", "higher_is_better": True},
            {"name": "exact_match_seen", "higher_is_better": True},
        ],
        protected_metrics=["exact_match", "exact_match_held"],
        benchmark_license="research-internal-sentinel",
        contamination_status="not_applicable_synthetic_fixture",
        schema_id=bound.schema_version,
    )


def environment_payload() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "implementation": platform.python_implementation(),
        # Timestamps intentionally excluded from environment_hash inputs used for identity.
        "measurement_note": "timestamps live in run_manifest only",
    }


def score_docs(docs: list[dict[str, Any]], predictor: Predictor) -> tuple[list[dict], list[dict], dict]:
    raw_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    held_scores: list[float] = []
    seen_scores: list[float] = []
    all_scores: list[float] = []
    for doc in docs:
        pred = predictor.predict(doc)
        em = exact_match(pred, doc["doc_to_target"])
        all_scores.append(em)
        if doc["held_values"]:
            held_scores.append(em)
        else:
            seen_scores.append(em)
        raw_rows.append(
            {
                "item_id": doc["item_id"],
                "held_values": doc["held_values"],
                "prediction": pred,
                "target": doc["doc_to_target"],
            }
        )
        score_rows.append(
            {
                "item_id": doc["item_id"],
                "held_values": doc["held_values"],
                "exact_match": em,
            }
        )

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    metrics = {
        "exact_match": _mean(all_scores),
        "exact_match_held": _mean(held_scores),
        "exact_match_seen": _mean(seen_scores),
        "n": len(all_scores),
        "n_held": len(held_scores),
        "n_seen": len(seen_scores),
    }
    return raw_rows, score_rows, metrics


def write_json(path: Path, obj: Any) -> str:
    text = canonical_json(obj) if not isinstance(obj, str) else obj
    # Pretty for human report artifacts that are dicts — still hash via bytes on disk after write
    if isinstance(obj, (dict, list)):
        path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        path.write_text(str(obj), encoding="utf-8")
    return sha256_file(str(path))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    return sha256_file(str(path))


def run_smoke(
    *,
    mode: str,
    runs_root: Path | None = None,
    code_commit: str | None = None,
    fail: bool = False,
) -> dict[str, Any]:
    """Execute one regression smoke. ``mode`` is mock or deterministic."""
    runs_root = runs_root or DEFAULT_RUNS
    commit = code_commit or git_commit()
    task_yaml = REPO_ROOT / DEFAULT_TASK_YAML
    bound = load_and_bind_instrument(
        DEFAULT_FIXTURE,
        git_commit=commit,
        expected_sha256=SENTINEL_EXPECTED_SHA256,
        expected_record_count=SENTINEL_EXPECTED_N,
    )
    docs = docs_from_instrument(bound)
    bench = build_benchmark_manifest(bound, task_yaml)
    bench_hash = bench.digest()

    if mode == "mock":
        predictor: Predictor = MockedModel()
        solver_or_model = mocked_model_manifest()
        manifest_name = "model_manifest.json"
    elif mode == "deterministic":
        predictor = DeterministicTemplateSolver()
        solver_or_model = deterministic_solver_manifest()
        manifest_name = "solver_manifest.json"
    else:
        raise ValueError(f"unknown mode: {mode}")

    solver_hash = solver_or_model.digest()
    config = {
        "mode": mode,
        "task_id": TASK_ID,
        "task_version": TASK_VERSION,
        "suite_id": SUITE_ID,
        "program": "NANO_RUNTIME_REGRESSION",
        "promote": False,
        "leaderboard_eligible": False,
        "evidence_ledger_eligible": False,
    }
    config_hash = sha256_hex(canonical_json(config))
    env = environment_payload()
    env_hash = sha256_hex(canonical_json(env))

    run_id = compute_run_id(
        benchmark_manifest_sha256=bench_hash,
        model_or_solver_manifest_sha256=solver_hash,
        config_sha256=config_hash,
        code_git_commit=commit,
    )

    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    start = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    decision = DecisionStatus.INFRA_SMOKE_PASS
    run_status = RunStatus.COMPLETED
    failure: dict[str, Any] = {}
    raw_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}

    if fail:
        run_status = RunStatus.FAILED
        decision = DecisionStatus.INFRA_SMOKE_FAIL
        failure = {"error": "intentional_fail_for_test", "stage": "predict"}
    else:
        raw_rows, score_rows, metrics = score_docs(docs, predictor)

    end = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    hashes: dict[str, str] = {}
    hashes["benchmark_manifest.json"] = write_json(
        run_dir / "benchmark_manifest.json", bench.to_dict()
    )
    hashes[manifest_name] = write_json(
        run_dir / manifest_name, solver_or_model.to_dict()
    )
    hashes["config.yaml"] = write_json(run_dir / "config.yaml", config)
    # store config also as yaml-like json for simplicity; filename config.yaml content is JSON
    (run_dir / "config.yaml").write_text(
        "\n".join(f"{k}: {json.dumps(v)}" for k, v in config.items()) + "\n",
        encoding="utf-8",
    )
    hashes["config.yaml"] = sha256_file(str(run_dir / "config.yaml"))
    hashes["environment.json"] = write_json(run_dir / "environment.json", env)
    hashes["raw_outputs.jsonl"] = write_jsonl(run_dir / "raw_outputs.jsonl", raw_rows)
    hashes["per_item_scores.jsonl"] = write_jsonl(
        run_dir / "per_item_scores.jsonl", score_rows
    )
    hashes["metrics.json"] = write_json(run_dir / "metrics.json", metrics)
    cost = {
        "currency": "USD",
        "amount": 0.0,
        "measurement_state": "not_measured",
        "units_note": "Local CPU regression smoke; cost not measured",
    }
    hashes["cost.json"] = write_json(run_dir / "cost.json", cost)
    decision_obj = {
        "decision": decision.value,
        "promote": False,
        "leaderboard_eligible": False,
        "evidence_ledger_eligible": False,
        "mode": mode,
        "run_id": run_id,
    }
    hashes["decision.json"] = write_json(run_dir / "decision.json", decision_obj)

    report = (
        f"# Held-value regression report\n\n"
        f"- mode: `{mode}`\n"
        f"- run_id: `{run_id}`\n"
        f"- decision: `{decision.value}`\n"
        f"- promote: false\n"
        f"- metrics: `{json.dumps(metrics, sort_keys=True)}`\n"
        f"- evidence_ledger_eligible: false\n"
        f"- leaderboard_eligible: false\n"
    )
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    hashes["report.md"] = sha256_file(str(run_dir / "report.md"))

    run_manifest = RunManifest(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        benchmark_manifest_hash=bench_hash,
        model_or_solver_manifest_hash=solver_hash,
        config_hash=config_hash,
        environment_hash=env_hash,
        code_git_commit=commit,
        start_utc=start,
        end_utc=end,
        run_status=run_status.value,
        decision_status=decision.value,
        artifact_paths_and_hashes=hashes,
        cost_reference=cost,
        failure_information=failure,
        promote=False,
        leaderboard_eligible=False,
        evidence_ledger_eligible=False,
    )
    # Exclude start/end from identity; still stored.
    hashes["run_manifest.json"] = write_json(
        run_dir / "run_manifest.json", run_manifest.to_dict()
    )

    sums_lines = [f"{digest}  {name}" for name, digest in sorted(hashes.items())]
    (run_dir / "SHA256SUMS").write_text("\n".join(sums_lines) + "\n", encoding="utf-8")

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "decision": decision.value,
        "run_status": run_status.value,
        "metrics": metrics,
        "benchmark_manifest_hash": bench_hash,
        "model_or_solver_manifest_hash": solver_hash,
        "config_hash": config_hash,
        "artifact_hashes": hashes,
    }
