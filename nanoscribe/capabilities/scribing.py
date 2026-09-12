"""Scribing capability — extends P1 CandidateAtom path."""

from __future__ import annotations

from typing import Any

from nanoscribe.adapt import ModelCandidate
from nanoscribe.artifacts import Artifact, ArtifactMetadata, ArtifactType
from nanoscribe.candidate_schema import candidate_batch_parameters_schema
from nanoscribe.capabilities.registry import CapabilityId, SUBMIT_CANDIDATE_ATOMS_TOOL, get_capability
from nanoscribe.tool_calling import ToolDefinition


def submit_candidate_atoms_definition() -> ToolDefinition:
    spec = get_capability(CapabilityId.SCRIBE)
    return ToolDefinition(
        name=SUBMIT_CANDIDATE_ATOMS_TOOL,
        description=(
            "Submit clinical fact candidates extracted from a transcript. "
            "Quote-only evidence — never emit offsets, evidence_id, or normalized_value."
        ),
        parameters=candidate_batch_parameters_schema(),
    )


def scribing_tool_definitions() -> tuple[ToolDefinition, ...]:
    return (submit_candidate_atoms_definition(),)


def scribing_tools_openai() -> list[dict[str, Any]]:
    return [tool.to_openai_tool() for tool in scribing_tool_definitions()]


def artifact_from_scribing(candidate: ModelCandidate, *, producer: str = "") -> Artifact:
    spec = get_capability(CapabilityId.SCRIBE)
    return Artifact(
        artifact_type=ArtifactType.CANDIDATE_BATCH,
        schema_version=spec.schema_version,
        data=candidate.to_dict(),
        metadata=ArtifactMetadata(capability_id=CapabilityId.SCRIBE.value, producer=producer),
    )
