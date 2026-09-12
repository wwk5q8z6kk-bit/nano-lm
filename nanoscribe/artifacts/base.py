"""Normalized artifact envelope — type, schema_version, data, metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from nanoscribe.artifacts.errors import ArtifactError, artifact_fail
from nanoscribe.artifacts.validation import require_mapping, require_nonempty_string, require_schema_version


class ArtifactType(str, Enum):
    CANDIDATE_BATCH = "candidate_batch"
    SUMMARY = "summary"
    TABLE = "table"
    CHART = "chart"
    DIAGRAM = "diagram"


ARTIFACT_ENVELOPE_VERSION = "nano.artifact.v0"


@dataclass(frozen=True, slots=True)
class ArtifactMetadata:
    capability_id: str
    producer: str = ""
    source_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "producer": self.producer,
            "source_ids": list(self.source_ids),
        }

    @classmethod
    def from_dict(cls, data: object, *, path: str = "$.metadata") -> ArtifactMetadata:
        mapping = require_mapping(data, path)
        source_ids = mapping.get("source_ids", [])
        if not isinstance(source_ids, list):
            artifact_fail("type_error", "source_ids must be an array", f"{path}.source_ids")
        if any(not isinstance(item, str) for item in source_ids):
            artifact_fail("type_error", "source_ids must be strings", f"{path}.source_ids")
        producer = mapping.get("producer", "")
        if not isinstance(producer, str):
            artifact_fail("type_error", "producer must be a string", f"{path}.producer")
        return cls(
            capability_id=require_nonempty_string(mapping.get("capability_id", ""), f"{path}.capability_id"),
            producer=producer,
            source_ids=tuple(source_ids),
        )


@dataclass(frozen=True, slots=True)
class Artifact:
    """Validated capability output — converges JSON and tool-call paths."""

    artifact_type: ArtifactType
    schema_version: str
    data: Mapping[str, Any]
    metadata: ArtifactMetadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "envelope_version": ARTIFACT_ENVELOPE_VERSION,
            "artifact_type": self.artifact_type.value,
            "schema_version": self.schema_version,
            "data": dict(self.data),
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: object, *, path: str = "$") -> Artifact:
        mapping = require_mapping(data, path)
        envelope = mapping.get("envelope_version")
        if envelope is not None and envelope != ARTIFACT_ENVELOPE_VERSION:
            artifact_fail(
                "envelope_version",
                f"envelope_version must be {ARTIFACT_ENVELOPE_VERSION}",
                f"{path}.envelope_version",
            )
        artifact_type_raw = mapping.get("artifact_type")
        if not isinstance(artifact_type_raw, str):
            artifact_fail("type_error", "artifact_type must be a string", f"{path}.artifact_type")
        try:
            artifact_type = ArtifactType(artifact_type_raw)
        except ValueError:
            allowed = ", ".join(member.value for member in ArtifactType)
            artifact_fail("invalid_enum", f"expected one of: {allowed}", f"{path}.artifact_type")
        schema_version = require_nonempty_string(mapping.get("schema_version", ""), f"{path}.schema_version")
        data_block = mapping.get("data")
        data_mapping = require_mapping(data_block, f"{path}.data")
        metadata = ArtifactMetadata.from_dict(mapping.get("metadata", {}), path=f"{path}.metadata")
        return cls(
            artifact_type=artifact_type,
            schema_version=schema_version,
            data=data_mapping,
            metadata=metadata,
        )

    @classmethod
    def from_json(cls, raw: str) -> Artifact:
        from nanoscribe.artifacts.validation import load_json

        return cls.from_dict(load_json(raw))


def validate_artifact_envelope(data: object) -> Artifact:
    """Parse and validate a full artifact envelope."""
    return Artifact.from_dict(data)
