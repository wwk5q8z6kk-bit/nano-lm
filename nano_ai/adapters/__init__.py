"""Transcript-only solver adapters for the Nano scribe contract.

The deterministic solver is a diagnostic floor for the repository's closed
synthetic grammar.  It is deliberately not presented as Nano's trained AI.
"""

from .anchor_checkpoint import (
    ANCHOR_SPECS,
    FROZEN_NANO_V01,
    FROZEN_SCALE_V01,
    AnchorCheckpointSolver,
    verify_anchor_artifacts,
)
from .deterministic_v0 import DeterministicV0Solver
from .legacy_summary import (
    LEGACY_DIAGNOSTICS_VERSION,
    LegacySummaryFormatError,
    LegacySummarySolver,
)
from .state_checkpoint import (
    STATE_CHECKPOINT_PIPELINE_VERSION,
    STATE_GROUNDING_VERIFIER_ID,
    STATE_PROMPT_TEMPLATE_ID,
    STATE_SPAN_PROMPT_INSTRUCTION,
    StateCheckpointSolver,
    build_state_span_prompt,
)
from .state_span import (
    STATE_SPAN_DIAGNOSTICS_VERSION,
    StateSpanFormatError,
    StateSpanSolver,
    parse_state_span_summary,
)

__all__ = [
    "ANCHOR_SPECS",
    "FROZEN_NANO_V01",
    "FROZEN_SCALE_V01",
    "LEGACY_DIAGNOSTICS_VERSION",
    "STATE_CHECKPOINT_PIPELINE_VERSION",
    "STATE_GROUNDING_VERIFIER_ID",
    "STATE_PROMPT_TEMPLATE_ID",
    "STATE_SPAN_DIAGNOSTICS_VERSION",
    "STATE_SPAN_PROMPT_INSTRUCTION",
    "AnchorCheckpointSolver",
    "DeterministicV0Solver",
    "LegacySummaryFormatError",
    "LegacySummarySolver",
    "StateCheckpointSolver",
    "StateSpanFormatError",
    "StateSpanSolver",
    "build_state_span_prompt",
    "parse_state_span_summary",
    "verify_anchor_artifacts",
]
