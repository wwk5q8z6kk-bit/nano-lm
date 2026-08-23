#!/usr/bin/env python3
"""Smoke test a RunPod Serverless vLLM endpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanoscribe.serverless_inference import (
    _openai_client,
    endpoint_native_urls,
    parse_endpoint_id,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="RunPod serverless smoke test")
    parser.add_argument("--endpoint", required=True, help="endpoint id or URL")
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    args = parser.parse_args()

    endpoint_id = parse_endpoint_id(args.endpoint)
    client = _openai_client(endpoint_id=endpoint_id)
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": "Reply with exactly one line."},
            {"role": "user", "content": 'Reply with exactly: STATED: "neck"'},
        ],
        temperature=0,
        max_tokens=32,
    )
    content = response.choices[0].message.content or ""
    result = {
        "endpoint_id": endpoint_id,
        "urls": endpoint_native_urls(endpoint_id),
        "model": args.model,
        "success": bool(content.strip()),
        "content": content.strip(),
        "usage": response.usage.model_dump() if response.usage else None,
    }
    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
