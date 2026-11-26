#!/usr/bin/env python3
"""Posthoc literal state-label diagnostic; never invokes a model or strict parser.

Run from any directory. Existing output is never overwritten. To verify the
saved result without writing, use --output - and compare the JSON in memory.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


BASE = Path(__file__).resolve().parent
LABELS = ("CC", "DUR", "SEV", "MED", "ALG")
STATES = "SAMUC"
PATTERN = r"(?:^|\|)(CC|DUR|SEV|MED|ALG):([SAMUC])(?=\[|\||$)"
LABEL_RE = re.compile(PATTERN)
EXPECTED_CASES = 24
EXPECTED_FIELDS = 120


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def indexed(rows, name):
    result = {}
    for row in rows:
        case_id = row["case_id"]
        if case_id in result:
            raise ValueError(f"Duplicate {name} case: {case_id}")
        result[case_id] = row
    if len(result) != EXPECTED_CASES:
        raise ValueError(f"Expected {EXPECTED_CASES} {name} cases, got {len(result)}")
    return result


def summarize(fields):
    correct = sum(field["label_correct"] for field in fields)
    available = sum(field["predicted_label"] is not None for field in fields)
    confusion = defaultdict(Counter)
    for field in fields:
        confusion[field["gold_state"]][field["predicted_label"] or "unavailable"] += 1
    return {
        "denominator": len(fields),
        "available_labels": available,
        "unavailable_labels": len(fields) - available,
        "correct_labels": correct,
        "label_match_fraction_all_fields": correct / len(fields),
        "gold_to_label_confusion": {key: dict(value) for key, value in confusion.items()},
    }


def analyze_run(run_name, gold):
    if Path(run_name).name != run_name or run_name in (".", ".."):
        raise ValueError("Run names must be direct child directory names")
    run_dir = BASE / run_name
    provenance_path = run_dir / "provenance.json"
    provenance = read_json(provenance_path)
    if provenance.get("status") != "complete":
        raise ValueError(f"Run is not complete: {run_name}")
    if provenance.get("input_lock_verified_after_inference") is not True:
        raise ValueError(f"Post-inference lock verification absent: {run_name}")
    for name in ("gold.json", "sources.json"):
        if sha256(BASE / name) != provenance["input_hashes"][name]:
            raise ValueError(f"Run input hash mismatch: {run_name}/{name}")
    # Completed provenance is checked before any generated output is read.
    encounters_path = run_dir / "encounters.json"
    encounters = indexed(read_json(encounters_path), "output")
    if set(encounters) != set(gold):
        raise ValueError(f"Output/gold case IDs differ: {run_name}")
    cases = []
    all_fields = []
    unavailable = []
    ordered_label_cases = 0
    for case_id, annotation in gold.items():
        encounter = encounters[case_id]
        raw = encounter.get("generation", {}).get("raw")
        if raw is not None and not isinstance(raw, str):
            raise ValueError(f"Raw output is not text: {run_name}/{case_id}")
        matches = list(LABEL_RE.finditer(raw or ""))
        ordered_label_cases += [match.group(1) for match in matches] == list(LABELS)
        fields = []
        for label in LABELS:
            occurrences = [match for match in matches if match.group(1) == label]
            predicted = occurrences[0].group(2) if len(occurrences) == 1 else None
            reason = None if predicted is not None else (
                "duplicate_recognized_label" if occurrences else "no_recognized_label"
            )
            field = {
                "case_id": case_id,
                "field": label,
                "gold_state": annotation["fields"][label]["state"],
                "predicted_label": predicted,
                "unavailable_reason": reason,
                "recognized_occurrences": [
                    {"state": match.group(2), "label_start_character": match.start(1)}
                    for match in occurrences
                ],
                "label_correct": predicted == annotation["fields"][label]["state"],
            }
            fields.append(field)
            all_fields.append(field)
            if reason:
                unavailable.append(field)
        cases.append({
            "case_id": case_id,
            "stratum": encounter.get("stratum"),
            "raw_output": raw,
            "primary_raw_parse_error": encounter.get("raw_parse_error"),
            "fields": fields,
        })
    if len(all_fields) != EXPECTED_FIELDS:
        raise ValueError("Diagnostic denominator changed")
    return {
        "run": run_name,
        "completed_utc": provenance.get("finished_utc"),
        "decoding": provenance.get("decoding"),
        "input_hashes": {
            "gold.json": sha256(BASE / "gold.json"),
            "sources.json": sha256(BASE / "sources.json"),
            f"{run_name}/provenance.json": sha256(provenance_path),
            f"{run_name}/encounters.json": sha256(encounters_path),
        },
        "summary": summarize(all_fields),
        "by_field": {
            label: summarize([field for field in all_fields if field["field"] == label])
            for label in LABELS
        },
        "cases_with_exactly_five_recognized_labels_in_canonical_order": ordered_label_cases,
        "unavailable_label_details": unavailable,
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=["free-01", "constrained-01"])
    parser.add_argument("--output", default=str(BASE / "posthoc_state_labels.json"),
                        help="New output path, or '-' for stdout; never overwrites")
    args = parser.parse_args()
    if len(set(args.runs)) != len(args.runs):
        raise ValueError("Duplicate run names")
    gold = indexed(read_json(BASE / "gold.json"), "gold")
    for case_id, annotation in gold.items():
        if set(annotation["fields"]) != set(LABELS):
            raise ValueError(f"Gold fields differ from canonical labels: {case_id}")
        if any(field["state"] not in set(STATES) for field in annotation["fields"].values()):
            raise ValueError(f"Unknown gold state: {case_id}")
    report = {
        "schema_version": "p1.posthoc-state-labels.v1",
        "analysis_status": "posthoc_diagnostic_not_preplanned",
        "primary_scores_replaced": False,
        "script_sha256": sha256(Path(__file__)),
        "method": {
            "purpose": "Separate literal state-label agreement from the primary full exact-source parser score.",
            "pattern": PATTERN,
            "canonical_labels": list(LABELS),
            "rule": "Find canonical LABEL:STATE strings at start or after a pipe, followed by '[' or a pipe or end of text. A field is available only with exactly one recognized occurrence; zero or multiple occurrences are unavailable and incorrect.",
            "denominator": "All 120 gold fields in all 24 cases per run; unavailable labels remain in the denominator. No case or field is excluded.",
            "payload_policy": "Payloads are ignored. No bracket, slot, span, source, referent, temporal, evidence, or clinical validity is checked or claimed. A pipe within malformed payload text can act as a label boundary. Canonical label order does not establish valid output syntax.",
            "interpretation": "This is literal label agreement, not standalone validated semantic-state accuracy, grounded extraction, complete evidence accuracy, or useful note yield.",
            "primary_rule": "The preplanned primary score marks all fields incorrect when the full raw exact-source parser fails; this diagnostic does not change that rule or any primary artifact.",
            "limitations": "Posthoc on a small synthetic development set with agent-applied annotations; not clinical, human, or confirmatory validation. Unlabeled state tokens are not inferred or repaired.",
            "reproduction": "Standard library only; no model calls or product imports. Completed provenance and source/gold hashes are verified before output analysis. Use --output - for a read-only reproduction.",
        },
        "runs": [analyze_run(name, gold) for name in args.runs],
    }
    serialized = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output == "-":
        print(serialized, end="")
    else:
        with Path(args.output).open("x", encoding="utf-8") as output:
            output.write(serialized)
        print(f"Created {args.output}")


if __name__ == "__main__":
    main()
