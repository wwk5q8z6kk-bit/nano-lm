"""Factorial 2×2 native screen design — A/B/C/D × 2 seeds @ 30M."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class ArchFactor(str, Enum):
    EVIDENCE_BOTTLENECK = "evidence_bottleneck"
    SLOT_ROUTER = "slot_router"


class ObjectiveFactor(str, Enum):
    SPAN_PORT = "span_port"
    STRUCTURED_JSON = "structured_json"


@dataclass(frozen=True, slots=True)
class FactorialCell:
    cell_id: str
    arch: ArchFactor
    objective: ObjectiveFactor
    params_m: int = 30
    seeds: tuple[int, ...] = (0, 1)

    def run_ids(self) -> tuple[str, ...]:
        return tuple(f"{self.cell_id}_s{seed}" for seed in self.seeds)


FACTORIAL_CELLS: tuple[FactorialCell, ...] = (
    FactorialCell("A", ArchFactor.EVIDENCE_BOTTLENECK, ObjectiveFactor.SPAN_PORT),
    FactorialCell("B", ArchFactor.EVIDENCE_BOTTLENECK, ObjectiveFactor.STRUCTURED_JSON),
    FactorialCell("C", ArchFactor.SLOT_ROUTER, ObjectiveFactor.SPAN_PORT),
    FactorialCell("D", ArchFactor.SLOT_ROUTER, ObjectiveFactor.STRUCTURED_JSON),
)


def factorial_manifest() -> dict[str, Any]:
    runs = []
    for cell in FACTORIAL_CELLS:
        for run_id in cell.run_ids():
            runs.append(
                {
                    "run_id": run_id,
                    "cell": cell.cell_id,
                    "arch": cell.arch.value,
                    "objective": cell.objective.value,
                    "params_m": cell.params_m,
                    "seed": int(run_id.rsplit("s", 1)[-1]),
                    "dataset": "p1_distill_train_v1",
                    "gpu": "NVIDIA B200",
                    "max_steps_halving_round1": 200,
                    "max_steps_halving_round2": 500,
                    "successive_halving": True,
                    "launch_gate": "trainer_ready + budget_gate",
                }
            )
    return {
        "schema": "nano.campaign.native_factorial.v1",
        "timestamp": datetime.now(UTC).isoformat(),
        "status": "TRAINER_READY_NO_GPU",
        "design": "2x2_factorial_ABCD_x2_seeds_30M",
        "cells": [cell.cell_id for cell in FACTORIAL_CELLS],
        "n_runs": len(runs),
        "runs": runs,
        "halving_schedule": {
            "round1": "all 8 runs @ 200 steps — keep top 4 by val loss",
            "round2": "survivors @ 500 steps — pick winner",
        },
    }
