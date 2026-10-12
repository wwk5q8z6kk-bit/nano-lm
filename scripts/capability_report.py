#!/usr/bin/env python3
"""Render the Nano capability specification as a matrix + status summary."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nano.capabilities import CAPABILITIES, by_domain, coverage  # noqa: E402

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--markdown", type=Path, default=None)
    args = ap.parse_args()
    cv = coverage()
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            {"coverage": cv, "capabilities": [c.to_dict() for c in CAPABILITIES]},
            indent=2) + "\n")
    if args.markdown:
        rows = ["# Nano capability specification", "",
                f"{cv['total']} capabilities across {cv['domains_covered']} domains. "
                f"Status: " + ", ".join(f"{k} {v}" for k, v in cv["by_status"].items()),
                "", "| id | domain | capability | stage | status | evidence |",
                "|---|---|---|---|---|---|"]
        for c in CAPABILITIES:
            rows.append(f"| `{c.capability_id}` | {c.domain} | {c.capability} | "
                        f"{c.implementation_stage.value} | **{c.status.value}** | "
                        f"{c.evidence or '—'} |")
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text("\n".join(rows) + "\n")
    print(f"capabilities: {cv['total']}  domains: {cv['domains_covered']}/{cv['domains_required']}")
    for k, v in cv["by_status"].items():
        print(f"  {k:12s} {v:3d}")
    print("\nby domain:")
    for dom, caps in by_domain().items():
        built = sum(1 for c in caps if c.status.value == "IMPLEMENTED")
        print(f"  {dom:22s} {len(caps):2d} capabilities, {built} implemented")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
