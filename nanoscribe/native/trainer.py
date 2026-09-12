"""Minimal native scratch trainer for P1 factorial screening (30M)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nanoscribe.native.factorial import (
    FACTORIAL_CELLS,
    ArchFactor,
    FactorialCell,
    ObjectiveFactor,
    canonical_run_id,
    legacy_run_id,
)


@dataclass(frozen=True, slots=True)
class NativeTrainConfig:
    run_id: str
    cell: FactorialCell
    seed: int
    params_m: int = 30
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    max_seq: int = 512
    max_steps: int = 200
    batch_size: int = 16
    peak_lr: float = 3e-4
    dataset_path: str = "artifacts/campaign/p1_distill_train_v1.json"


def config_for_run(run_id: str) -> NativeTrainConfig:
    for cell in FACTORIAL_CELLS:
        for seed in cell.seeds:
            if run_id in {canonical_run_id(cell, seed), legacy_run_id(cell, seed)}:
                return NativeTrainConfig(
                    run_id=canonical_run_id(cell, seed),
                    cell=cell,
                    seed=seed,
                )
    raise ValueError(f"unknown run_id: {run_id}")


def estimate_param_count(cfg: NativeTrainConfig) -> int:
    """Rough param count for scratch GPT-style native model."""
    d, l, v = cfg.d_model, cfg.n_layers, 4098
    emb = v * d
    block = l * (4 * d * d + 2 * d * (4 * d))  # attn + mlp approx
    return emb + block


def build_runpod_command(cfg: NativeTrainConfig) -> str:
    """Exact command for GPU pod launch — no GPU until this is ready."""
    return (
        f"python3 -m nanoscribe.native.trainer "
        f"--run-id {cfg.run_id} "
        f"--dataset {cfg.dataset_path} "
        f"--max-steps {cfg.max_steps} "
        f"--seed {cfg.seed}"
    )


def trainer_manifest(path: Path | None = None) -> dict[str, Any]:
    configs = [config_for_run(rid) for cell in FACTORIAL_CELLS for rid in cell.run_ids()]
    runs = []
    for cfg in configs:
        runs.append(
            {
                "run_id": cfg.run_id,
                "cell": cfg.cell.cell_id,
                "arch": cfg.cell.arch.value,
                "objective": cfg.cell.objective.value,
                "seed": cfg.seed,
                "params_m_target": cfg.params_m,
                "params_m_estimated": round(estimate_param_count(cfg) / 1e6, 2),
                "max_steps": cfg.max_steps,
                "runpod_command": build_runpod_command(cfg),
            }
        )
    payload = {
        "schema": "nano.native.trainer.v0",
        "timestamp": datetime.now(UTC).isoformat(),
        "status": "READY_FOR_GPU_LAUNCH",
        "backend": "scratch_from_random_init",
        "n_runs": len(runs),
        "runs": runs,
    }
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def train_step_stub(cfg: NativeTrainConfig) -> dict[str, Any]:
    """CPU-safe stub — validates config without GPU. Real train uses torch on pod."""
    if not Path(cfg.dataset_path).is_file():
        return {"ok": False, "reason": f"dataset missing: {cfg.dataset_path}"}
    return {
        "ok": True,
        "run_id": cfg.run_id,
        "arch": cfg.cell.arch.value,
        "objective": cfg.cell.objective.value,
        "seed": cfg.seed,
        "params_m_est": round(estimate_param_count(cfg) / 1e6, 2),
        "command": build_runpod_command(cfg),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Native scratch trainer")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--dataset", default="artifacts/campaign/p1_distill_train_v1.json")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--purpose",
        default="architecture_screening",
        help="training purpose for corpus launch guard",
    )
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()

    if args.manifest_only:
        out = Path("artifacts/campaign/native_trainer_manifest.json")
        print(json.dumps(trainer_manifest(out), indent=2))
        return 0

    if not args.run_id:
        parser.error("--run-id is required unless --manifest-only")

    from nanoscribe.native.train import train_run_id

    try:
        import torch

        cpu_smoke = not torch.cuda.is_available()
    except ImportError:
        cpu_smoke = True

    if not cpu_smoke:
        from nanoscribe.native.corpus.registry import assert_corpus_launch_allowed

        assert_corpus_launch_allowed(args.dataset, purpose=args.purpose, cpu_smoke=False)

    result = train_run_id(args.run_id, cpu_smoke=cpu_smoke)
    print(json.dumps(result.to_dict(), indent=2))
    if result.steps_completed < 1:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
