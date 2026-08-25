#!/usr/bin/env python3
"""Generate campaign B2 local span-transport artifact (no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoscribe.span_transport import write_span_transport_v2


def main() -> None:
    path = write_span_transport_v2()
    print(path)


if __name__ == "__main__":
    main()
