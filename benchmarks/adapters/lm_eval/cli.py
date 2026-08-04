"""CLI for validating and running the held-value regression sentinel."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .runner import DEFAULT_TASK_YAML, REPO_ROOT, run_smoke
from .task_adapter import load_and_bind_instrument
from .runner import (
    SENTINEL_EXPECTED_N,
    SENTINEL_EXPECTED_SHA256,
    DEFAULT_FIXTURE,
    git_commit,
)


def cmd_validate(_: argparse.Namespace) -> int:
    """Validate custom task via lm-eval if installed; else structural checks."""
    task_yaml = REPO_ROOT / DEFAULT_TASK_YAML
    if not task_yaml.is_file():
        print(f"FAIL: missing task yaml {task_yaml}", file=sys.stderr)
        return 1
    bound = load_and_bind_instrument(
        DEFAULT_FIXTURE,
        git_commit=git_commit(),
        expected_sha256=SENTINEL_EXPECTED_SHA256,
        expected_record_count=SENTINEL_EXPECTED_N,
    )
    print(
        json.dumps(
            {
                "structural_ok": True,
                "task_yaml": str(task_yaml.relative_to(REPO_ROOT)),
                "instrument": bound.repo_relative_path,
                "sha256": bound.sha256,
                "record_count": bound.record_count,
            },
            indent=2,
        )
    )
    lm_eval = shutil.which("lm-eval")
    if lm_eval is None:
        # try python -m
        try:
            import lm_eval  # noqa: F401
        except ImportError:
            print(
                "NOTE: lm-eval not installed; structural validation only. "
                "Install via requirements-bench.txt for harness validate.",
                file=sys.stderr,
            )
            return 0
        cmd = [
            sys.executable,
            "-m",
            "lm_eval",
            "validate",
            "--tasks",
            "nano_held_value_sentinel",
            "--include_path",
            str(REPO_ROOT / "benchmarks/adapters/lm_eval/tasks"),
        ]
    else:
        cmd = [
            lm_eval,
            "validate",
            "--tasks",
            "nano_held_value_sentinel",
            "--include_path",
            str(REPO_ROOT / "benchmarks/adapters/lm_eval/tasks"),
        ]
    print("Running:", " ".join(cmd), file=sys.stderr)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return proc.returncode


def cmd_smoke(args: argparse.Namespace) -> int:
    result = run_smoke(mode=args.mode, fail=args.fail)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["decision"] == "INFRA_SMOKE_PASS" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nano_lm_benchmark")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_val = sub.add_parser("validate", help="Validate held-value sentinel task")
    p_val.set_defaults(func=cmd_validate)
    p_smoke = sub.add_parser("smoke", help="Run the held-value regression")
    p_smoke.add_argument("--mode", choices=["mock", "deterministic"], required=True)
    p_smoke.add_argument("--fail", action="store_true", help="Force FAILED path")
    p_smoke.set_defaults(func=cmd_smoke)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
