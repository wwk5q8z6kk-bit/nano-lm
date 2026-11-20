"""Native Nano architecture scaffold for P1 screening (30M–100M).

Not a vanilla decoder default: hypotheses favor evidence-aware, bottlenecked designs
suited to span-port scribing rather than open-ended generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NativeHypothesis(str, Enum):
    """Screening hypotheses for 30M–100M native Nano models."""

    H1_EVIDENCE_BOTTLENECK = "h1_evidence_bottleneck"
    H2_SLOT_ROUTER = "h2_slot_router"
    H3_HYBRID_RETRIEVER = "h3_hybrid_retriever"


@dataclass(frozen=True, slots=True)
class NativeArchSpec:
    hypothesis: NativeHypothesis
    param_budget_m: int
    description: str
    key_mechanism: str
    training_objective: str


HYPOTHESES: tuple[NativeArchSpec, ...] = (
    NativeArchSpec(
        hypothesis=NativeHypothesis.H1_EVIDENCE_BOTTLENECK,
        param_budget_m=50,
        description=(
            "Small encoder over source turns + slot-conditioned quote head; "
            "decoder never emits offsets — only quote strings for ConstrainedSelector."
        ),
        key_mechanism="turn_encoder + per-slot quote pointer",
        training_objective="span-port SFT with quote-only targets",
    ),
    NativeArchSpec(
        hypothesis=NativeHypothesis.H2_SLOT_ROUTER,
        param_budget_m=80,
        description=(
            "Shared trunk with learned slot routers; each ClinicalAtom type gets "
            "a dedicated low-rank adapter head (STATED/DENIED/NOT_MENTIONED)."
        ),
        key_mechanism="MoE-style slot routing without full MoE scale",
        training_objective="multi-task span-port across atom types",
    ),
    NativeArchSpec(
        hypothesis=NativeHypothesis.H3_HYBRID_RETRIEVER,
        param_budget_m=100,
        description=(
            "Bi-encoder retrieval over turns + tiny cross-encoder reranker; "
            "generation conditioned on top-1 turn only (evidence-first)."
        ),
        key_mechanism="retrieve-then-label (no free generation)",
        training_objective="contrastive turn ranking + span-port labels",
    ),
)


def manifest() -> dict[str, object]:
    return {
        "schema": "nano.native_arch_hypotheses.v0",
        "status": "DESIGN_ONLY_NOT_TRAINING",
        "param_range_m": [30, 100],
        "reference_nano_3m": "sft/model_nano.py (~3M GPT, 6L/192d)",
        "hypotheses": [
            {
                "id": spec.hypothesis.value,
                "param_budget_m": spec.param_budget_m,
                "description": spec.description,
                "key_mechanism": spec.key_mechanism,
                "training_objective": spec.training_objective,
            }
            for spec in HYPOTHESES
        ],
        "rejected": [
            "vanilla_1b_decoder_default — too costly, wrong inductive bias for P1",
            "end_to_end_encounter_json — violates evidence transport contract",
            "immediate_1b_pretrain — screening should start 30M–100M",
        ],
    }
