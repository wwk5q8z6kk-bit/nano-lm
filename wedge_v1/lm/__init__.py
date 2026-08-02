"""W6 — gated marginal LM value (admission + probe harness).

No external LM calls unless `AUTHORIZE_WEDGE_V1_PHASE3_LM_PROBE` + execute mode.
Default path: admission check + stub constructive-faithfulness probe.
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
