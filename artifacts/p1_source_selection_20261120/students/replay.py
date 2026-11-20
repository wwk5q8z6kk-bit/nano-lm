"""Replay archived model proposals through NanoScribe and score exact source boundaries."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path, help="Nano LM checkout that supplies the verifier")
    args = parser.parse_args()
    sys.path.insert(0, str(args.checkout.resolve()))
    from nano_ai.contract import FIELD_ORDER
    from nanoscribe.compact import compact_solver
    from nanoscribe.scribe import scribe

    data = Path(__file__).resolve().parent
    cases = {(case["cohort"], case["id"]): case for case in load_rows(data / "cases.jsonl")}
    rows = load_rows(data / "proposals.jsonl")
    metrics = defaultdict(Counter)
    seen = set()
    for row in rows:
        arm, key = row["arm"], (row["cohort"], row["id"])
        if (arm, key) in seen:
            raise ValueError(f"Duplicate prediction for {arm}, {key}")
        seen.add((arm, key))
        case = cases[key]
        note = scribe(
            case["transcript"],
            solver=compact_solver(lambda _, slots=row["slots"]: slots, binder="grounded"),
            context_policy="grounded",
        )
        if not note["verification"]["passed"]:
            raise ValueError(f"Verification failed for {arm}, {key}")
        evidence = {item["evidence_id"]: item for item in note["encounter"]["evidence"]}
        counts = metrics[arm]
        for field, claim in zip(FIELD_ORDER, note["note"]["claims"], strict=True):
            gold = case["gold"][field.value]
            expected = gold["state"] in ("supported", "absent")
            asserted = claim["state"] in ("supported", "absent")
            refs = [evidence[k] for k in claim["evidence_ids"]]
            exact_span = (
                expected and asserted and claim["state"] == gold["state"] and len(refs) == 1
                and (refs[0]["start"], refs[0]["end"], refs[0]["text"])
                == (gold["start"], gold["end"], gold["quote"])
            )
            counts["fields"] += 1
            counts["exact"] += int(exact_span or (not asserted and not expected))
            counts["expected_assertions"] += expected
            counts["exact_assertions"] += int(exact_span)
            counts["emitted"] += asserted
            counts["overclaims"] += asserted and not expected
            counts["omissions"] += expected and not asserted
    if len(seen) != len(metrics) * len(cases):
        raise ValueError("Missing model or control predictions")
    print(json.dumps(dict(sorted(metrics.items())), indent=2))

if __name__ == "__main__":
    main()
