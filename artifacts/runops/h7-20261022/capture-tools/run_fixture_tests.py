"""Run the preserved synthetic fixtures against the preserved evaluation helper.

The original fixture imports an absolute /private/tmp path. Redirect only that
import, leaving the archived fixture and helper bytes unchanged.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import runpy
import sys
from unittest.mock import patch


def main() -> None:
    directory = Path(__file__).resolve().parent
    original_spec = importlib.util.spec_from_file_location

    def preserved_spec(name, location, *args, **kwargs):
        if str(location) == "/private/tmp/nano_h7_capture_evaluation.py":
            location = directory / "nano_h7_capture_evaluation.py"
        return original_spec(name, location, *args, **kwargs)

    fixture = directory / "test_nano_h7_capture_evaluation.py"
    sys.argv[0] = str(fixture)
    with patch.object(importlib.util, "spec_from_file_location", preserved_spec):
        runpy.run_path(str(fixture), run_name="__main__")


if __name__ == "__main__":
    main()
