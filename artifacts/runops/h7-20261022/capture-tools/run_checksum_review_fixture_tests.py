"""Run preserved checksum-review fixtures without importing temporary files."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import runpy
import sys
from unittest.mock import patch


def main() -> None:
    directory = Path(__file__).resolve().parent
    original_spec = importlib.util.spec_from_file_location
    preserved = {
        f"/private/tmp/{name}": directory / name
        for name in (
            "nano_h7_capture_evaluation.py",
            "nano_h7_capture_evaluation_checksum_review.py",
            "test_nano_h7_capture_evaluation.py",
        )
    }

    def preserved_spec(name, location, *args, **kwargs):
        path = str(location)
        if path in preserved:
            location = preserved[path]
        elif path.startswith("/private/tmp/"):
            raise RuntimeError(f"Unexpected temporary import: {path}")
        return original_spec(name, location, *args, **kwargs)

    fixture = directory / "test_nano_h7_capture_evaluation_checksum_review.py"
    sys.argv[0] = str(fixture)
    with patch.object(importlib.util, "spec_from_file_location", preserved_spec):
        runpy.run_path(str(fixture), run_name="__main__")


if __name__ == "__main__":
    main()
