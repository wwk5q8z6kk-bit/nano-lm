"""Table capability — canonical table representation."""

from __future__ import annotations

from typing import Any

from nanoscribe.artifacts import Artifact, ArtifactMetadata, ArtifactType, TableSpec
from nanoscribe.artifacts.table_spec import TABLE_SCHEMA_VERSION
from nanoscribe.capabilities.registry import CapabilityId, SUBMIT_TABLE_TOOL, get_capability
from nanoscribe.tool_calling import ToolDefinition


def table_parameters_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "schema_version": {"type": "string", "enum": [TABLE_SCHEMA_VERSION]},
            "title": {"type": ["string", "null"]},
            "columns": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "label": {"type": "string"},
                        "data_type": {
                            "type": "string",
                            "enum": ["string", "number", "boolean"],
                        },
                    },
                    "required": ["key", "label"],
                    "additionalProperties": False,
                },
            },
            "rows": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
        "required": ["schema_version", "columns", "rows"],
        "additionalProperties": False,
    }


def submit_table_definition() -> ToolDefinition:
    return ToolDefinition(
        name=SUBMIT_TABLE_TOOL,
        description="Submit a canonical table with named columns and string cell rows.",
        parameters=table_parameters_schema(),
    )


def validate_table_payload(mapping: dict[str, Any]) -> TableSpec:
    return TableSpec.from_dict(mapping)


def artifact_from_table(spec: TableSpec, *, producer: str = "") -> Artifact:
    capability = get_capability(CapabilityId.TABLE)
    return Artifact(
        artifact_type=ArtifactType.TABLE,
        schema_version=capability.schema_version,
        data=spec.to_dict(),
        metadata=ArtifactMetadata(capability_id=CapabilityId.TABLE.value, producer=producer),
    )
