# Adversarial contract pins for Track configs.
# Run: python3 nanoscribe/test_tracks.py
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoscribe.tracks import api_teacher_track, API_TEACHER_MODEL
from nanoscribe.harness import ModelTrack

def expect(code: str, fn) -> None:
    try:
        fn()
    except Exception as e:
        if code not in str(e):
            raise AssertionError(f"Expected {code} in {e!r}")
        return
    raise AssertionError(f"Expected {code} but no error was raised")

def test_api_teacher_track_fields() -> None:
    tc = api_teacher_track()
    assert tc.track == ModelTrack.FRONTIER, f"Expected FRONTIER, got {tc.track}"
    assert tc.model_id == f"api/{API_TEACHER_MODEL}-span-port", f"Expected api/{API_TEACHER_MODEL}-span-port, got {tc.model_id}"
    assert tc.cost_class == "api_teacher_low", f"Expected api_teacher_low, got {tc.cost_class}"
    adapter = tc.adapter_factory()
    assert adapter.api_model == API_TEACHER_MODEL, f"Expected {API_TEACHER_MODEL}, got {adapter.api_model}"

    tc_custom = api_teacher_track("gpt-4-custom")
    assert tc_custom.model_id == "api/gpt-4-custom-span-port"
    adapter_custom = tc_custom.adapter_factory()
    assert adapter_custom.api_model == "gpt-4-custom"

def main() -> None:
    test_api_teacher_track_fields()
    print("All tests passed!")

if __name__ == "__main__":
    main()
