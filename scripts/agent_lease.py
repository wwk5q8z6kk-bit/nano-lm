#!/usr/bin/env python3
"""Backwards-compatible wrapper for scripts/agent_coord.py lease commands."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent_coord  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(agent_coord.main())
