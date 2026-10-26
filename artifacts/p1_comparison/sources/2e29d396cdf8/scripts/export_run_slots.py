#!/usr/bin/env python3
"""Export per-slot payloads from orx run logs into committed JSON.

Why this exists: the eight-cell result shipped as 161 lines of aggregates while
the per-slot generations lived only in orx run logs, outside git. Every prior
result in trajectory/ ships as per-run JSON; that one did not, so the headline
could not be re-derived or re-interrogated from the versioned trail. Anyone
without the logs still open had to take the aggregates on trust — which is
exactly how dc3b310's "83%" hardened into a citation.

One row per probed slot, carrying what is needed to recompute every aggregate
and to ask questions the original analysis did not: the prompt as sent, the raw
model line, the parsed quote, the gold value, and every per-slot counter.

Usage:
  python3 scripts/export_run_slots.py --run <runId> --cell L000 --out artifacts/slots
  python3 scripts/export_run_slots.py --manifest artifacts/slots
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from nanoscribe.adapt import parse_label_and_quotes


def fetch_run_payload(run_id: str) -> dict:
    """Pull a run's stdout from orx and extract the JSON report."""
    out = subprocess.run(
        ["orx", "logs", run_id, "--head", "--bytes", "8000000"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    start = out.index("{")
    body = out[start:]
    return json.loads(body[: body.rindex("}") + 1])


def gold_value_for(encounter_id: str, atom_id: str) -> str | None:
    """Gold surface string for a slot, from the versioned suite definition."""
    from nanoscribe.campaign_datasets import case_for
    from nanoscribe.campaign_instances import split_encounter_id

    base, instance_id = split_encounter_id(encounter_id)
    case = case_for(base, instance_id)
    for spec in case.atom_specs:
        if spec.atom_id == atom_id:
            return spec.raw_value
    return None


def slot_rows(payload: dict, cell: str, run_id: str) -> list[dict]:
    rows: list[dict] = []
    for enc in payload["encounters"]:
        for atom_id, rec in sorted(enc["per_atom"].items()):
            # E-DELIMIT runs carry raw_line on the per-atom record and omit the
            # encounter-level raw_lines/prompts maps, so fall back rather than
            # KeyError on a schema that is otherwise compatible.
            raw_line = (enc.get("raw_lines") or {}).get(atom_id) or rec.get("raw_line", "")
            label, quotes = parse_label_and_quotes(raw_line)
            rows.append(
                {
                    "cell": cell,
                    "run_id": run_id,
                    "instance_id": enc["instance_id"],
                    "encounter_id": enc["encounter_id"],
                    "base_encounter_id": enc["base_encounter_id"],
                    "atom_id": atom_id,
                    "prompt": (enc.get("prompts") or {}).get(atom_id),
                    "raw_line": raw_line,
                    "parsed_label": label,
                    "parsed_quotes": list(quotes),
                    "gold_value": gold_value_for(enc["encounter_id"], atom_id),
                    "gold_present": rec["gold_present"],
                    "joint_cell": rec["cell"],
                    "exact_gold_span": rec["exact_gold_span"],
                    "span_character_f1": rec["span_character_f1"],
                    "assertion_state_correct": rec["assertion_state_correct"],
                    "abstained": rec["abstained"],
                    "unbound_assertion": rec["unbound_assertion"],
                    "malformed": rec["malformed"],
                    "gold_value_in_answer_template": (
                        enc.get("gold_value_in_answer_template") or {}
                    ).get(atom_id),
                    "gold_value_in_question": (
                        enc.get("gold_value_in_question") or {}
                    ).get(atom_id),
                    "quote_absent": (enc.get("quote_absent") or {}).get(atom_id),
                }
            )
    return rows


def export(run_id: str, cell: str, out_dir: Path) -> Path:
    payload = fetch_run_payload(run_id)
    rows = slot_rows(payload, cell, run_id)
    doc = {
        "schema": "nano.p1.run_slots.v1",
        "cell": cell,
        "run_id": run_id,
        "suite": payload.get("suite"),
        "dataset_revision": payload.get("dataset_revision"),
        "condition": payload["condition"],
        "leakage_config": payload.get("leakage_config"),
        "delimit_config": payload.get("delimit_config"),
        "arm": payload.get("arm"),
        "weights": payload["weights"],
        "interpreter": payload.get("interpreter"),
        "suite_aggregate": payload["suite_aggregate"],
        "per_instance_aggregate": payload.get("per_instance_aggregate"),
        "n_slots": len(rows),
        "slots": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"slots-{cell}.json"
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return path


def write_manifest(out_dir: Path) -> Path:
    """SHA256 over every exported file, so tampering or drift is detectable."""
    entries = {}
    for path in sorted(out_dir.glob("slots-*.json")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries[path.name] = digest
    manifest = out_dir / "SHA256SUMS"
    manifest.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in entries.items())
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--cell")
    ap.add_argument("--out", default="artifacts/slots")
    ap.add_argument("--manifest", help="write SHA256SUMS for this directory and exit")
    args = ap.parse_args(argv)

    if args.manifest:
        path = write_manifest(Path(args.manifest))
        print(f"wrote {path}")
        return 0
    if not (args.run and args.cell):
        ap.error("--run and --cell are required unless --manifest is given")
    path = export(args.run, args.cell, Path(args.out))
    doc = json.loads(path.read_text())
    print(f"{args.cell}: {doc['n_slots']} slots -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
