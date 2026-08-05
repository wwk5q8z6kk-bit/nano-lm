"""W6 marginal model-value admission and probe tools.

The default backend is a deterministic local stub. External backends must be
configured explicitly by the caller.
"""
from wedge_v1.lm.admission import evaluate_admission
from wedge_v1.lm.marginal import run_marginal_probe
from wedge_v1.lm.probe import ALLOWLIST_TASKS, StubLMBackend, probe_eclass_task

__all__ = [
    "evaluate_admission",
    "run_marginal_probe",
    "ALLOWLIST_TASKS",
    "StubLMBackend",
    "probe_eclass_task",
]
