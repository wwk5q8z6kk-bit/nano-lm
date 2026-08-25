"""Capability registry — maps tools to artifact contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from nanoscribe.artifacts import (
    Artifact,
    ArtifactMetadata,
    ArtifactType,
    CHART_SCHEMA_VERSION,
    DIAGRAM_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    TABLE_SCHEMA_VERSION,
)
from nanoscribe.adapt import CANDIDATE_SCHEMA_VERSION, ModelCandidate

SUBMIT_CANDIDATE_ATOMS_TOOL = "submit_candidate_atoms"
SUBMIT_SUMMARY_TOOL = "submit_summary"
SUBMIT_TABLE_TOOL = "submit_table"
SUBMIT_CHART_TOOL = "submit_chart"
SUBMIT_DIAGRAM_TOOL = "submit_diagram"
RUN_PYTHON_TOOL = "run_python"


class CapabilityId(str, Enum):
    SCRIBE = "scribe"
    SUMMARIZE = "summarize"
    TABLE = "table"
    CHART = "chart"
    DIAGRAM = "diagram"
    CODING = "coding"


class CapabilityStatus(str, Enum):
    ACTIVE = "active"
    STUB = "stub"
    PLANNED = "planned"


@dataclass(frozen=True, slots=True)
class CapabilitySpec:
    capability_id: CapabilityId
    label: str
    artifact_type: ArtifactType
    schema_version: str
    submit_tool_name: str
    status: CapabilityStatus
    trajectory_lane: str = "p8_compiler"

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id.value,
            "label": self.label,
            "artifact_type": self.artifact_type.value,
            "schema_version": self.schema_version,
            "submit_tool_name": self.submit_tool_name,
            "status": self.status.value,
            "trajectory_lane": self.trajectory_lane,
        }


CAPABILITY_REGISTRY: dict[CapabilityId, CapabilitySpec] = {
    CapabilityId.SCRIBE: CapabilitySpec(
        capability_id=CapabilityId.SCRIBE,
        label="Clinical scribing (CandidateAtoms)",
        artifact_type=ArtifactType.CANDIDATE_BATCH,
        schema_version=CANDIDATE_SCHEMA_VERSION,
        submit_tool_name=SUBMIT_CANDIDATE_ATOMS_TOOL,
        status=CapabilityStatus.ACTIVE,
        trajectory_lane="p1_scribing",
    ),
    CapabilityId.SUMMARIZE: CapabilitySpec(
        capability_id=CapabilityId.SUMMARIZE,
        label="Structured summarization",
        artifact_type=ArtifactType.SUMMARY,
        schema_version=SUMMARY_SCHEMA_VERSION,
        submit_tool_name=SUBMIT_SUMMARY_TOOL,
        status=CapabilityStatus.ACTIVE,
        trajectory_lane="p2_summarize",
    ),
    CapabilityId.TABLE: CapabilitySpec(
        capability_id=CapabilityId.TABLE,
        label="Canonical tables",
        artifact_type=ArtifactType.TABLE,
        schema_version=TABLE_SCHEMA_VERSION,
        submit_tool_name=SUBMIT_TABLE_TOOL,
        status=CapabilityStatus.ACTIVE,
        trajectory_lane="p2_tables",
    ),
    CapabilityId.CHART: CapabilitySpec(
        capability_id=CapabilityId.CHART,
        label="Charts (stub)",
        artifact_type=ArtifactType.CHART,
        schema_version=CHART_SCHEMA_VERSION,
        submit_tool_name=SUBMIT_CHART_TOOL,
        status=CapabilityStatus.STUB,
        trajectory_lane="p3_charts",
    ),
    CapabilityId.DIAGRAM: CapabilitySpec(
        capability_id=CapabilityId.DIAGRAM,
        label="Diagrams (stub)",
        artifact_type=ArtifactType.DIAGRAM,
        schema_version=DIAGRAM_SCHEMA_VERSION,
        submit_tool_name=SUBMIT_DIAGRAM_TOOL,
        status=CapabilityStatus.STUB,
        trajectory_lane="p3_diagrams",
    ),
    CapabilityId.CODING: CapabilitySpec(
        capability_id=CapabilityId.CODING,
        label="Coding sandbox (stub)",
        artifact_type=ArtifactType.CANDIDATE_BATCH,
        schema_version=CANDIDATE_SCHEMA_VERSION,
        submit_tool_name=RUN_PYTHON_TOOL,
        status=CapabilityStatus.STUB,
        trajectory_lane="p8_compiler",
    ),
}

TOOL_NAME_TO_CAPABILITY: dict[str, CapabilityId] = {
    spec.submit_tool_name: spec.capability_id for spec in CAPABILITY_REGISTRY.values()
}


def get_capability(capability_id: CapabilityId) -> CapabilitySpec:
    return CAPABILITY_REGISTRY[capability_id]


def capability_for_tool(tool_name: str) -> CapabilitySpec | None:
    capability_id = TOOL_NAME_TO_CAPABILITY.get(tool_name)
    if capability_id is None:
        return None
    return CAPABILITY_REGISTRY[capability_id]


def list_capabilities(*, status: CapabilityStatus | None = None) -> tuple[CapabilitySpec, ...]:
    specs = tuple(CAPABILITY_REGISTRY.values())
    if status is None:
        return specs
    return tuple(spec for spec in specs if spec.status == status)


def artifact_from_candidate(candidate: ModelCandidate, *, producer: str = "") -> Artifact:
    """Wrap a validated ModelCandidate in the normalized artifact envelope."""
    spec = get_capability(CapabilityId.SCRIBE)
    return Artifact(
        artifact_type=spec.artifact_type,
        schema_version=spec.schema_version,
        data=candidate.to_dict(),
        metadata=ArtifactMetadata(capability_id=CapabilityId.SCRIBE.value, producer=producer),
    )
