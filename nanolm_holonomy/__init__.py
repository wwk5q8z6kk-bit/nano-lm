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

__all__ = [
    "ABSOLUTE_MAX_PARAMETERS",
    "ExperimentRegistry",
    "HSPAArchConfig",
    "HSPAFamily",
    "HSPAUnifiedLM",
    "HSPA_2B_CONFIG",
    "HSPA_4B_CONFIG",
    "ParameterBudgetExceededError",
    "RegistryVersion",
    "TRANSFORMER_4B_CONFIG",
    "abelian_complex_scan",
    "build_suite_configs",
    "cayley_rotor",
    "complex_cayley_rotor",
    "complex_conjugate",
    "complex_product",
    "hamilton_product",
    "parallel_holonomy_scan",
    "sequential_holonomy_reference",
    "spinor_conjugate",
    "step_holonomy",
]
