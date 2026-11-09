# Adversarial contract pins for prompt templates.
# Run: python3 nanoscribe/test_prompt.py
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoscribe.prompt import _format_transcript
from nanoscribe.encounter import assemble_source, Speaker

def test_format_transcript() -> None:
    source = assemble_source(
        "src-1",
        (
            (Speaker.CLINICIAN, "What brings you in today?"),
            (Speaker.PATIENT, "My neck has been hurting."),
            (Speaker.CLINICIAN, "How long?"),
            (Speaker.PATIENT, "About 6 days."),
        )
    )
    transcript = _format_transcript(source)
    assert transcript == (
        "clinician: What brings you in today?\n"
        "patient: My neck has been hurting.\n"
        "clinician: How long?\n"
        "patient: About 6 days."
    )
    print("  PASS test_format_transcript")

if __name__ == '__main__':
    test_format_transcript()
    print("prompt pins: 1/1 PASS")
