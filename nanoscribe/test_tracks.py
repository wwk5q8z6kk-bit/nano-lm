# Adversarial pins for tracks configuration.
# Run: python3 nanoscribe/test_tracks.py
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoscribe.harness import HarnessCase, ModelTrack, P1TestSet, TrackConfig
from nanoscribe.tracks import (
    api_teacher_track,
    compact_track,
    fixture_track,
    frontier_track,
    serverless_strong_control_track,
    student_track,
    tiny_fixture_case,
)
from nanoscribe.adapters import (
    ApiTeacherAdapter,
    Qwen25BaselineAdapter,
    ServerlessQwen38Adapter,
)


def test_tiny_fixture_case() -> None:
    case = tiny_fixture_case()
    assert isinstance(case, HarnessCase)
    assert case.test_set == P1TestSet.TINY_FIXTURE
    assert case.gold is not None
    assert case.model_input is not None
    assert case.atom_specs is not None
    assert len(case.atom_specs) > 0


def test_fixture_track() -> None:
    track = fixture_track()
    assert isinstance(track, TrackConfig)
    assert track.track == ModelTrack.FIXTURE
    adapter = track.adapter_factory()
    assert adapter is not None


def test_compact_track() -> None:
    track = compact_track()
    assert isinstance(track, TrackConfig)
    assert track.track == ModelTrack.COMPACT
    adapter = track.adapter_factory()
    assert isinstance(adapter, Qwen25BaselineAdapter)


def test_serverless_strong_control_track() -> None:
    track = serverless_strong_control_track()
    assert isinstance(track, TrackConfig)
    assert track.track == ModelTrack.FRONTIER
    adapter = track.adapter_factory()
    assert isinstance(adapter, ServerlessQwen38Adapter)


def test_api_teacher_track() -> None:
    track = api_teacher_track()
    assert isinstance(track, TrackConfig)
    assert track.track == ModelTrack.FRONTIER
    adapter = track.adapter_factory()
    assert isinstance(adapter, ApiTeacherAdapter)


def test_student_track() -> None:
    track = student_track()
    assert isinstance(track, TrackConfig)
    assert track.track == ModelTrack.FRONTIER
    adapter = track.adapter_factory()
    assert isinstance(adapter, Qwen25BaselineAdapter)


def test_frontier_track() -> None:
    track = frontier_track()
    assert isinstance(track, TrackConfig)
    assert track.track == ModelTrack.FRONTIER
    adapter = track.adapter_factory()
    assert isinstance(adapter, Qwen25BaselineAdapter)


if __name__ == "__main__":
    test_tiny_fixture_case()
    test_fixture_track()
    test_compact_track()
    test_serverless_strong_control_track()
    test_api_teacher_track()
    test_student_track()
    test_frontier_track()
    print("All track configuration tests passed!")
