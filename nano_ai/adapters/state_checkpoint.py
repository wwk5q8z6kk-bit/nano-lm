"""Integrity-gated checkpoint adapter for native state/span candidates.

The candidate artifact identity is supplied by the caller.  No checkpoint is
declared a winner here, and the frozen Nano v0.1 registration remains untouched.
The existing own-stack loader supplies the exact architecture, greedy decoding,
lazy dependency loading, and hash-before-load behavior.
"""

from __future__ import annotations

from pathlib import Path

from nano_ai.contract import NanoInput, NanoOutput
from nano_ai.solver import SolverDescriptor, SolverKind

from .anchor_checkpoint import (
    DECODE_POLICY_ID,
    FROZEN_NANO_V01,
    AnchorArtifactSpec,
    AnchorSummaryGenerator,
    RuntimeLoader,
)
from .state_span import StateSpanSolver

STATE_CHECKPOINT_PIPELINE_VERSION = "native-state-span-checkpoint-v0"
STATE_PROMPT_TEMPLATE_ID = "chatml-native-state-span-scribe-v0"
STATE_GROUNDING_VERIFIER_ID = "state-span-evidence-v0"

STATE_SPAN_PROMPT_INSTRUCTION = """Return exactly five ordered fields in this grammar:
CC:<state>|DUR:<state>|SEV:<state>|MED:<state>|ALG:<state>
Use only S[exact Patient span], A[exact Patient denial span], M, U[exact Patient span], U[], or C[exact Patient span;exact Patient span]. Copy every span verbatim. Add no other text.
Summarize the visit."""


def build_state_span_prompt(transcript: str) -> str:
    """Build the exact candidate prompt shared by training and inference."""

    if not isinstance(transcript, str):
        raise TypeError("transcript must be text")
    if not transcript.strip():
        raise ValueError("transcript must contain text")
    return f"{transcript.rstrip()}\n{STATE_SPAN_PROMPT_INSTRUCTION}"


def _nano_architecture(spec: AnchorArtifactSpec) -> bool:
    return spec.architecture_identity == FROZEN_NANO_V01.architecture_identity


class StateCheckpointSolver:
    """Verification-gated runtime for one hash-identified H1 candidate."""

    def __init__(
        self,
        spec: AnchorArtifactSpec,
        checkpoint_path: str | Path,
        tokenizer_path: str | Path,
        *,
        runtime_loader: RuntimeLoader | None = None,
        runtime_id: str | None = None,
        device: str = "cpu",
        max_new_tokens: int = 128,
    ) -> None:
        if not isinstance(spec, AnchorArtifactSpec):
            raise TypeError("spec must be an AnchorArtifactSpec")
        if not _nano_architecture(spec):
            raise ValueError(
                "native-state v0 candidates must retain Nano's frozen "
                "3,148,608-parameter architecture"
            )
        self.generator = AnchorSummaryGenerator(
            spec,
            checkpoint_path,
            tokenizer_path,
            runtime_loader=runtime_loader,
            runtime_id=runtime_id,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        runtime_identity = self.generator.runtime_id
        if self.generator.runtime_kind is SolverKind.LEGACY_ADAPTER:
            runtime_identity = f"injected-{runtime_identity}"
        solver_id = (
            f"{spec.solver_id}"
            f"/arch-{spec.architecture_identity}"
            f"/pipeline-{STATE_CHECKPOINT_PIPELINE_VERSION}"
            f"/runtime-{runtime_identity}"
            f"/prompt-{STATE_PROMPT_TEMPLATE_ID}"
            f"/decode-{DECODE_POLICY_ID}"
            f"/max-new-{self.generator.max_new_tokens}"
            f"/grounding-{STATE_GROUNDING_VERIFIER_ID}"
            f"/device-{self.generator.device_identity}"
        )
        self.descriptor = SolverDescriptor(
            solver_id=solver_id,
            kind=self.generator.runtime_kind,
            version=STATE_CHECKPOINT_PIPELINE_VERSION,
            parameter_count=(
                spec.parameter_count
                if self.generator.runtime_kind is SolverKind.HYBRID
                else None
            ),
            artifact_bytes=None,
        )
        self.solver_id = solver_id
        self._state_adapter = StateSpanSolver(
            self._generate_state_summary,
            solver_id=solver_id,
            version=STATE_CHECKPOINT_PIPELINE_VERSION,
            parameter_count=self.descriptor.parameter_count,
        )

    def _generate_state_summary(self, transcript: str) -> str:
        return self.generator.generate(build_state_span_prompt(transcript))

    def infer(self, request: NanoInput) -> NanoOutput:
        return self._state_adapter.infer(request)

    def infer_with_state_diagnostics(
        self, request: NanoInput
    ) -> tuple[NanoOutput, dict[str, object]]:
        """Expose the native-state trace outside the legacy metric protocol."""

        return self._state_adapter.infer_with_state_diagnostics(request)


__all__ = [
    "STATE_CHECKPOINT_PIPELINE_VERSION",
    "STATE_GROUNDING_VERIFIER_ID",
    "STATE_PROMPT_TEMPLATE_ID",
    "STATE_SPAN_PROMPT_INSTRUCTION",
    "StateCheckpointSolver",
    "build_state_span_prompt",
]
