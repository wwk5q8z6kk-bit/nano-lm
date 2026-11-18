"""NanoLM Autonomous Intelligence Architecture Discovery Program (`nanolm_discovery`).

Provides:
- Parameter-verified configuration and iso-parameter suite construction (strict < 4B limit).
- Hardware-efficient parallel & O(1)-recurrent operators:
  * Port-Hamiltonian Metriplectic Network (I1_PHMN) via exact nilpotent WY factorization
  * Natural-Parameter Epistemic Belief Network (I2_NEBN) via log-decay precision dual filter
  * Holographic Resonant Wave-Memory Network (I3_HRWN) via causal dyadic waves + SCAM
  * Cycle 2 Unified Metriplectic Precision-Resonant Network (G2_MPRN)
  * Controlled Transformer GQA (B0) and 3:1 Gated-DeltaNet Hybrid (B1) baselines
- MuonAdamW optimizer, JEP latent training objective, and 3-track lineage registry.
"""

from __future__ import annotations

from nanolm_discovery.config import (
    HARD_MAX_DEPLOYED_PARAMETERS,
    PREFERRED_FLAGSHIP_TARGET,
    ArchitectureConfig,
    ParameterBudgetError,
    ParameterCensus,
    ParadigmKind,
    compute_parameter_census,
    make_iso_param_suite,
)
from nanolm_discovery.models import ModelOutput, NanoLMDiscoveryModel
from nanolm_discovery.optim import MuonAdamW
from nanolm_discovery.registry import InnovationScore, LineageRegistry

__all__ = [
    "HARD_MAX_DEPLOYED_PARAMETERS",
    "PREFERRED_FLAGSHIP_TARGET",
    "ArchitectureConfig",
    "ParameterBudgetError",
    "ParameterCensus",
    "ParadigmKind",
    "compute_parameter_census",
    "make_iso_param_suite",
    "ModelOutput",
    "NanoLMDiscoveryModel",
    "MuonAdamW",
    "InnovationScore",
    "LineageRegistry",
]
