# Artifact spec validation tests.
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoscribe.artifacts import (
    Artifact,
    ArtifactError,
    ArtifactMetadata,
    ArtifactType,
    ChartSpec,
    DiagramSpec,
    SummarySpec,
    TableSpec,
    validate_artifact_envelope,
)
from nanoscribe.capabilities.scribing import artifact_from_scribing
from nanoscribe.adapt import ModelCandidate, CandidateAtom


def test_table_spec_roundtrip() -> None:
    spec = TableSpec.from_dict(
        {
            "schema_version": "nano.table.v0",
            "title": "Labs",
            "columns": [
                {"key": "test", "label": "Test", "data_type": "string"},
                {"key": "value", "label": "Value", "data_type": "number"},
            ],
            "rows": [["WBC", "7.2"]],
        }
    )
    restored = TableSpec.from_dict(spec.to_dict())
    assert restored.title == "Labs"
    assert len(restored.columns) == 2
    assert restored.rows[0] == ("WBC", "7.2")


def test_table_spec_rejects_row_width_mismatch() -> None:
    try:
        TableSpec.from_dict(
            {
                "schema_version": "nano.table.v0",
                "columns": [{"key": "a", "label": "A"}],
                "rows": [["x", "y"]],
            }
        )
        raise AssertionError("expected invalid_row_width")
    except ArtifactError as exc:
        assert exc.code == "invalid_row_width"


def test_summary_spec_roundtrip() -> None:
    spec = SummarySpec.from_dict(
        {
            "schema_version": "nano.summary.v0",
            "title": "Summary",
            "sections": [{"heading": "Symptoms", "bullets": ["Neck pain"]}],
            "source_atom_ids": ["atom-neck"],
        }
    )
    assert spec.source_atom_ids == ("atom-neck",)
    assert SummarySpec.from_dict(spec.to_dict()).title == "Summary"


def test_chart_and_diagram_stubs() -> None:
    chart = ChartSpec.from_dict(
        {
            "schema_version": "nano.chart.v0",
            "title": "Trend",
            "chart_kind": "stub",
        }
    )
    diagram = DiagramSpec.from_dict(
        {
            "schema_version": "nano.diagram.v0",
            "title": "Flow",
            "diagram_kind": "stub",
            "source_text": "A --> B",
        }
    )
    assert chart.chart_kind == "stub"
    assert diagram.notation == "mermaid"


def test_artifact_envelope_from_scribing() -> None:
    candidate = ModelCandidate(
        atoms=(
            CandidateAtom(atom_id="atom-neck", abstained=True),
        )
    )
    artifact = artifact_from_scribing(candidate, producer="test")
    envelope = validate_artifact_envelope(artifact.to_dict())
    assert envelope.artifact_type is ArtifactType.CANDIDATE_BATCH
    assert envelope.metadata.capability_id == "scribe"


def test_artifact_envelope_parse() -> None:
    artifact = Artifact(
        artifact_type=ArtifactType.TABLE,
        schema_version="nano.table.v0",
        data={"schema_version": "nano.table.v0", "columns": [], "rows": []},
        metadata=ArtifactMetadata(capability_id="table"),
    )
    parsed = Artifact.from_dict(artifact.to_dict())
    assert parsed.artifact_type is ArtifactType.TABLE
