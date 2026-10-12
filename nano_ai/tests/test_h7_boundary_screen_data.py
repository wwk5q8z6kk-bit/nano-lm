from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from nano_ai.adapters.state_span import parse_state_span_summary
from nano_ai.contract import FIELD_ORDER, FieldState
from nano_ai.training.h7_boundary_screen_data import (
    SCREEN_RECORDS,
    SCREEN_WORLDS,
    build_manifest,
    generate_screen,
    records_bytes,
    write_screen,
)

_CONTROL_SHA256 = "07547ec5c01b46fa87e8d53a40497a7c4b4a6bc92b596dddd67f4be4d18c89e0"


@pytest.fixture(scope="module")
def screen():  # type: ignore[no-untyped-def]
    return generate_screen()


def test_h7_screen_is_complete_balanced_and_deterministic(screen) -> None:  # type: ignore[no-untyped-def]
    assert len(screen) == SCREEN_RECORDS
    assert records_bytes(screen) == records_bytes(generate_screen())
    by_world: dict[str, list[str]] = defaultdict(list)
    targets: Counter[str] = Counter()
    for record in screen:
        by_world[record.world_id].append(record.variant)
        targets[record.target_field.value] += 1
    assert len(by_world) == SCREEN_WORLDS
    assert all(
        variants == ["normal", "missing", "uncertain", "conflicting"]
        for variants in by_world.values()
    )
    assert targets == Counter({field.value: 200 for field in FIELD_ORDER})


def test_h7_targets_parse_to_exact_patient_spans_and_expected_states(screen) -> None:  # type: ignore[no-untyped-def]
    for record in screen:
        proposals = parse_state_span_summary(record.target, record.transcript)
        assert tuple(proposal.field for proposal in proposals) == FIELD_ORDER
        for proposal in proposals:
            for span in proposal.spans:
                assert record.transcript[span.start : span.end] == span.text
        target = proposals[FIELD_ORDER.index(record.target_field)]
        if record.variant == "normal":
            assert record.target_state is None
        else:
            assert target.state is record.target_state
        if record.variant == "conflicting":
            assert target.state is FieldState.CONFLICTING
            assert len(target.spans) == 2


def test_h7_manifest_proves_source_isolation(screen) -> None:  # type: ignore[no-untyped-def]
    generator_sha = hashlib.sha256(b"test-generator").hexdigest()
    manifest = build_manifest(
        screen,
        generator_sha256=generator_sha,
        control_checkpoint_sha256=_CONTROL_SHA256,
    )
    assert manifest["screen"]["records"] == SCREEN_RECORDS
    assert manifest["screen"]["worlds"] == SCREEN_WORLDS
    assert manifest["screen"]["sha256"] == hashlib.sha256(records_bytes(screen)).hexdigest()
    assert manifest["isolation"] == {
        "audited_source_families": [
            "legacy-train",
            "legacy-dev",
            "h4-fit",
            "h4-calibration",
        ],
        "exact_transcript_overlap": 0,
        "normalized_line_component_overlap": 0,
        "open_value_overlap_excluding_fixed_severity": 0,
        "new_world_namespace": True,
        "evaluation_only": True,
        "gold_built_from_annotated_patient_turns": True,
        "private_or_phi_records": 0,
    }


def test_h7_writer_is_no_clobber(tmp_path: Path) -> None:
    output = tmp_path / "h7-screen"
    manifest = write_screen(output, control_checkpoint_sha256=_CONTROL_SHA256)
    assert (output / "screen.jsonl").is_file()
    assert (output / "manifest.json").is_file()
    assert manifest["screen"]["records"] == SCREEN_RECORDS
    with pytest.raises(FileExistsError, match="already exists"):
        write_screen(output, control_checkpoint_sha256=_CONTROL_SHA256)
