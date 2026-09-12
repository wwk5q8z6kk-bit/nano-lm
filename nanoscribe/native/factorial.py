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
        return tuple(canonical_run_id(self, seed) for seed in self.seeds)


def canonical_run_id(cell: FactorialCell, seed: int) -> str:
    """Unambiguous run ID: native30_{arch}_{objective}_s{seed}."""
    arch = "bottleneck" if cell.arch == ArchFactor.EVIDENCE_BOTTLENECK else "decoder"
    obj = "span" if cell.objective == ObjectiveFactor.SPAN_PORT else "struct"
    return f"native30_{arch}_{obj}_s{seed}"


def legacy_run_id(cell: FactorialCell, seed: int) -> str:
    return f"{cell.cell_id}_s{seed}"


FACTORIAL_CELLS: tuple[FactorialCell, ...] = (
    FactorialCell("A", ArchFactor.EVIDENCE_BOTTLENECK, ObjectiveFactor.SPAN_PORT),
    FactorialCell("B", ArchFactor.EVIDENCE_BOTTLENECK, ObjectiveFactor.STRUCTURED_JSON),
    FactorialCell("C", ArchFactor.SLOT_ROUTER, ObjectiveFactor.SPAN_PORT),
    FactorialCell("D", ArchFactor.SLOT_ROUTER, ObjectiveFactor.STRUCTURED_JSON),
)


@dataclass(frozen=True, slots=True)
class Round2Promotion:
    """Round-2 100M promotion arm — fresh random init, not warm-started from 30M."""

    run_prefix: str
    arch: ArchFactor
    objective: ObjectiveFactor
    params_m: int = 100
    seeds: tuple[int, ...] = (0, 1)


def round2_run_id(promotion: Round2Promotion, seed: int) -> str:
    return f"{promotion.run_prefix}_s{seed}"


ROUND2_PROMOTIONS: tuple[Round2Promotion, ...] = (
    Round2Promotion("native100_evidence_bottleneck", ArchFactor.EVIDENCE_BOTTLENECK, ObjectiveFactor.SPAN_PORT),
    Round2Promotion("native100_span_port", ArchFactor.SLOT_ROUTER, ObjectiveFactor.SPAN_PORT),
)


@dataclass(frozen=True, slots=True)
class RevalidationArm:
    """Wave-1 arm: re-screen 30M mechanisms on a REAL corpus.

    The original 30M ranking was produced on the 4,516-char fixture with the
    64-char tokenizer truncation active, so it is EXPLORATORY_SCREENING_RANKING
    only. Before promoting anything to 100M the provisional leaders must be
    re-screened against a decoder CONTROL on the real corpus — otherwise a
    successive-halving artefact gets promoted as an architectural conclusion.
    """

    run_prefix: str
    arch: ArchFactor
    objective: ObjectiveFactor
    params_m: int = 30
    seeds: tuple[int, ...] = (0, 1, 2)
    is_control: bool = False


def revalidation_run_id(arm: RevalidationArm, seed: int) -> str:
    return f"{arm.run_prefix}_s{seed}"


# Three arms x three seeds. The decoder control is mandatory: without it a
# mechanism that merely tracks capacity is indistinguishable from one that works.
REVALIDATION_ARMS: tuple[RevalidationArm, ...] = (
    RevalidationArm("reval30_decoder_control", ArchFactor.SLOT_ROUTER,
                    ObjectiveFactor.STRUCTURED_JSON, is_control=True),
    RevalidationArm("reval30_evidence_bottleneck", ArchFactor.EVIDENCE_BOTTLENECK,
                    ObjectiveFactor.SPAN_PORT),
    RevalidationArm("reval30_span_port", ArchFactor.SLOT_ROUTER,
                    ObjectiveFactor.SPAN_PORT),
)


def round2_manifest() -> dict[str, Any]:
    runs = []
    for promo in ROUND2_PROMOTIONS:
        for seed in promo.seeds:
            runs.append(
                {
                    "run_id": round2_run_id(promo, seed),
                    "arch": promo.arch.value,
                    "objective": promo.objective.value,
                    "params_m": promo.params_m,
                    "seed": seed,
                    "init": "fresh_random",
                    "max_steps": 200,
                }
            )
    return {
        "schema": "nano.campaign.native_round2.v1",
        "timestamp": datetime.now(UTC).isoformat(),
        "promotions": [p.run_prefix for p in ROUND2_PROMOTIONS],
        "n_runs": len(runs),
        "runs": runs,
    }


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
