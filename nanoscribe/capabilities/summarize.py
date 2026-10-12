"""Summarization capability — structured summary sections."""

from __future__ import annotations

from typing import Any

from nanoscribe.artifacts import Artifact, ArtifactMetadata, ArtifactType, SummarySpec
from nanoscribe.artifacts.summary_spec import SUMMARY_SCHEMA_VERSION
from nanoscribe.capabilities.registry import CapabilityId, SUBMIT_SUMMARY_TOOL, get_capability
from nanoscribe.tool_calling import ToolDefinition


def summary_parameters_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "schema_version": {"type": "string", "enum": [SUMMARY_SCHEMA_VERSION]},
            "title": {"type": "string"},
            "sections": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "heading": {"type": "string"},
                        "bullets": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["heading", "bullets"],
                    "additionalProperties": False,
                },
            },
            "source_atom_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["schema_version", "title", "sections"],
        "additionalProperties": False,
    }


def submit_summary_definition() -> ToolDefinition:
    return ToolDefinition(
        name=SUBMIT_SUMMARY_TOOL,
        description="Submit a structured summary with section headings and bullet points.",
        parameters=summary_parameters_schema(),
    )


def validate_summary_payload(mapping: dict[str, Any]) -> SummarySpec:
    return SummarySpec.from_dict(mapping)


def artifact_from_summary(spec: SummarySpec, *, producer: str = "") -> Artifact:
    capability = get_capability(CapabilityId.SUMMARIZE)
    return Artifact(
        artifact_type=ArtifactType.SUMMARY,
        schema_version=capability.schema_version,
        data=spec.to_dict(),
        metadata=ArtifactMetadata(capability_id=CapabilityId.SUMMARIZE.value, producer=producer),
    )
