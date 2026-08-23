#!/usr/bin/env python3
"""RunPod Serverless endpoint worker scaling helpers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanoscribe.serverless_config import (
    configure_burst,
    configure_pause,
    fetch_health,
    get_endpoint,
)
from nanoscribe.tracks import SERVERLESS_ENDPOINT_ID


def main() -> int:
    parser = argparse.ArgumentParser(description="RunPod serverless worker config")
    parser.add_argument("--endpoint", default=SERVERLESS_ENDPOINT_ID)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("get", help="show endpoint config")
    sub.add_parser("health", help="show endpoint health")

    burst = sub.add_parser("burst", help="set burst workers (default max=10)")
    burst.add_argument("--max-workers", type=int, default=10)

    sub.add_parser("pause", help="scale workers to 0")

    args = parser.parse_args()
    endpoint = args.endpoint

    if args.cmd == "get":
        print(json.dumps(get_endpoint(endpoint), indent=2))
        return 0
    if args.cmd == "health":
        print(json.dumps(fetch_health(endpoint), indent=2))
        return 0
    if args.cmd == "burst":
        print(json.dumps(configure_burst(endpoint, max_workers=args.max_workers), indent=2))
        return 0
    if args.cmd == "pause":
        print(json.dumps(configure_pause(endpoint), indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
