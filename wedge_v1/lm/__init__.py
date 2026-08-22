"""W6 — gated marginal LM value (admission + probe harness).

No external paid LM. Default path: admission check + stub constructive probe.
Real local backend: `MLXLlamaBackend` (mlx_llama32_3b_spanbound) — one proven
span-binding contract; no fan-out until this seam holds.
"""
from wedge_v1.lm.admission import evaluate_admission
from wedge_v1.lm.marginal import run_marginal_probe
from wedge_v1.lm.probe import ALLOWLIST_TASKS, StubLMBackend, get_backend, probe_eclass_task

__all__ = [
    "evaluate_admission",
    "run_marginal_probe",
    "ALLOWLIST_TASKS",
    "StubLMBackend",
    "get_backend",
    "probe_eclass_task",
]
