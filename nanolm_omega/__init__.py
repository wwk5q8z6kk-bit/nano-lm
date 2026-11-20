"""NANOLM Omega: Unified capacity-balanced wave-metriplectic architecture, EFE cognition, and symbolic integration."""

from nanolm_omega.backbone import (
    G4OmegaModel,
    OmegaBackboneConfig,
    OmegaCensus,
    OmegaVariant,
    X8DualUnembedModel,
    build_omega_model,
    compute_omega_census,
    make_omega_suite,
)
from nanolm_omega.cognition_efe import (
    CognitiveAgentModel,
    HardCausalSWEEnv,
    HardRepoEpisode,
    run_efe_cognition_experiment,
)
from nanolm_omega.unified_system import (
    OmegaBenchTask,
    OmegaComputeRouter,
    OmegaUnifiedCognitiveSystem,
    build_omega_bench_01,
    evaluate_omega_bench_01,
)

__all__ = [
    "CognitiveAgentModel",
    "G4OmegaModel",
    "HardCausalSWEEnv",
    "HardRepoEpisode",
    "OmegaBackboneConfig",
    "OmegaBenchTask",
    "OmegaCensus",
    "OmegaComputeRouter",
    "OmegaUnifiedCognitiveSystem",
    "OmegaVariant",
    "X8DualUnembedModel",
    "build_omega_bench_01",
    "build_omega_model",
    "compute_omega_census",
    "evaluate_omega_bench_01",
    "make_omega_suite",
    "run_efe_cognition_experiment",
]
