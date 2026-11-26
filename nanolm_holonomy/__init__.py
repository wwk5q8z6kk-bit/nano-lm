"""NanoLM Holonomic Spinor Predictive Architecture (HSPA) Discovery Package."""

from nanolm_holonomy.architecture import (
    ABSOLUTE_MAX_PARAMETERS,
    HSPA_2B_CONFIG,
    HSPA_4B_CONFIG,
    TRANSFORMER_4B_CONFIG,
    HSPAArchConfig,
    HSPAFamily,
    HSPAUnifiedLM,
    ParameterBudgetExceededError,
    build_suite_configs,
)
from nanolm_holonomy.registry import ExperimentRegistry, RegistryVersion
from nanolm_holonomy.spinor_algebra import (
    abelian_complex_scan,
    cayley_rotor,
    complex_cayley_rotor,
    complex_conjugate,
    complex_product,
    hamilton_product,
    parallel_holonomy_scan,
    sequential_holonomy_reference,
    spinor_conjugate,
    step_holonomy,
)

from nanolm_holonomy.compute_planner import (
    GPU_CATALOG,
    WorkloadSpec,
    estimate_workload,
    select_optimal_gpu,
)
from nanolm_holonomy.omega_architecture import (
    HSPAOmegaSidecar,
    SpectralNullspaceLoRALinear,
    apply_nullspace_lora,
    apply_omega_graft,
    so3_projective_adjoint,
    su2_lie_exponential,
)
from nanolm_holonomy.omega_system import classify_task_mode, omega_generate, verify_python_smoke_zero_oracle
from nanolm_holonomy.triton_holonomy import HolonomyScanFn, holonomy_scan_fast

__all__ = [
    "ABSOLUTE_MAX_PARAMETERS",
    "ExperimentRegistry",
    "GPU_CATALOG",
    "HSPAArchConfig",
    "HSPAFamily",
    "HSPAOmegaSidecar",
    "HSPAUnifiedLM",
    "HSPA_2B_CONFIG",
    "HSPA_4B_CONFIG",
    "HolonomyScanFn",
    "ParameterBudgetExceededError",
    "SpectralNullspaceLoRALinear",
    "holonomy_scan_fast",
    "RegistryVersion",
    "TRANSFORMER_4B_CONFIG",
    "WorkloadSpec",
    "abelian_complex_scan",
    "apply_nullspace_lora",
    "apply_omega_graft",
    "build_suite_configs",
    "cayley_rotor",
    "classify_task_mode",
    "complex_cayley_rotor",
    "complex_conjugate",
    "complex_product",
    "estimate_workload",
    "hamilton_product",
    "omega_generate",
    "parallel_holonomy_scan",
    "select_optimal_gpu",
    "sequential_holonomy_reference",
    "so3_projective_adjoint",
    "spinor_conjugate",
    "step_holonomy",
    "su2_lie_exponential",
    "verify_python_smoke_zero_oracle",
]
