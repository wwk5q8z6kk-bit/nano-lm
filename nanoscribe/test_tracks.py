from nanoscribe.tracks import (
    tiny_fixture_case,
    fixture_track,
    compact_track,
    serverless_strong_control_track,
    api_teacher_track,
    student_track,
    frontier_track,
    COMPACT_MODEL,
    STUDENT_MODEL,
    FRONTIER_MODEL,
    API_TEACHER_MODEL,
    SERVERLESS_STRONG_MODEL,
    SERVERLESS_ENDPOINT_ID,
)
from nanoscribe.adapters import ApiTeacherAdapter, Qwen25BaselineAdapter, ServerlessQwen38Adapter
from nanoscribe.harness import HarnessCase, TrackConfig, ModelTrack, P1TestSet


def test_tiny_fixture_case() -> None:
    case = tiny_fixture_case()
    assert isinstance(case, HarnessCase)
    assert case.test_set is P1TestSet.TINY_FIXTURE
    assert case.encounter_id == "enc-1"
    assert case.gold is not None
    assert case.model_input is not None
    assert case.atom_specs is not None
    assert len(case.atom_specs) > 0


def test_fixture_track() -> None:
    track = fixture_track()
    assert isinstance(track, TrackConfig)
    assert track.track is ModelTrack.FIXTURE
    assert track.model_id == "fixture/qwen2.5-1.5b-span-port"
    assert track.cost_class == "zero_local"
    assert "CI" in track.notes


def test_compact_track() -> None:
    track = compact_track()
    assert isinstance(track, TrackConfig)
    assert track.track is ModelTrack.COMPACT
    assert track.model_id == COMPACT_MODEL
    assert track.cost_class == "routine_runpod_4090"

    for builder in (compact_track, student_track, frontier_track):
        custom = builder("my-custom-model")
        adapter = custom.adapter_factory()
        assert isinstance(adapter, Qwen25BaselineAdapter)
        assert custom.model_id == adapter.model_id == adapter.weights_path == "my-custom-model"


def test_serverless_strong_control_track() -> None:
    track = serverless_strong_control_track()
    assert isinstance(track, TrackConfig)
    assert track.track is ModelTrack.SERVERLESS
    assert track.model_id == "serverless/qwen3.8-27b-strong-control"
    assert track.cost_class == "serverless_strong_control"
    adapter = serverless_strong_control_track("test-endpoint", "test-model").adapter_factory()
    assert isinstance(adapter, ServerlessQwen38Adapter)
    assert adapter.endpoint_id == "test-endpoint"
    assert adapter.api_model == "test-model"


def test_api_teacher_track() -> None:
    track = api_teacher_track()
    assert isinstance(track, TrackConfig)
    assert track.track is ModelTrack.FRONTIER
    assert track.model_id == f"api/{API_TEACHER_MODEL}-span-port"
    assert track.cost_class == "api_teacher_low"

    # Test custom api model
    track_custom = api_teacher_track("gpt-4-turbo")
    assert track_custom.model_id == "api/gpt-4-turbo-span-port"
    adapter = track_custom.adapter_factory()
    assert isinstance(adapter, ApiTeacherAdapter)
    assert adapter.api_model == "gpt-4-turbo"


def test_student_track() -> None:
    track = student_track()
    assert isinstance(track, TrackConfig)
    assert track.track is ModelTrack.FRONTIER
    assert track.model_id == STUDENT_MODEL
    assert track.cost_class == "experiment_scoped_a100_80gb"


def test_frontier_track() -> None:
    track = frontier_track()
    assert isinstance(track, TrackConfig)
    assert track.track is ModelTrack.FRONTIER
    assert track.model_id == FRONTIER_MODEL
    assert track.cost_class == "experiment_scoped_a100_80gb"


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in fns:
        fn()
        print(f"  PASS {name}")
    print(f"tracks pins: {len(fns)}/{len(fns)} PASS")
