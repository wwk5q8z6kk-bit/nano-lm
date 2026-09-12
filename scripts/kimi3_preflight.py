#!/usr/bin/env python3
"""Fail-fast Kimi K3 RunPod Public Endpoint probe."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanoscribe.kimi_teacher import kimi_preflight


def main() -> int:
    result = kimi_preflight()
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
