"""Fetch a pinned public ACI-Bench CSV or read a local copy and replay saved proposals."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import urllib.request

HERE = Path(__file__).resolve().parent

def load_sources(path):
    source = json.loads((HERE / "source.json").read_text())
    if path is None:
        with urllib.request.urlopen(source["url"], timeout=60) as response:
            data = response.read()
    else:
        data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != source["csv_sha256"]:
        raise ValueError("The public source differs from the recorded revision")
    transcripts = {}
    ids = {item["id"] for item in source["selected"]}
    for row in csv.DictReader(io.StringIO(data.decode("utf-8-sig"))):
        if row["id"] in ids:
            parts = re.split(r"\[(doctor|patient)\]\s*", row["dialogue"], flags=re.I)
            if parts[0].strip() or len(parts) < 3:
                raise ValueError("Unrecognized speaker format")
            turns = [f"{role.capitalize()}: {' '.join(turn.split())}"
                     for role, turn in zip(parts[1::2], parts[2::2], strict=True)]
            transcripts[row["id"]] = "\n".join(turns)
    if transcripts.keys() != ids:
        raise ValueError("Missing a selected encounter")
    return transcripts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.checkout.resolve()))
    sys.path.insert(0, str(HERE.parent / "teachers"))
    from score import checked_note, raw_note

    transcripts = load_sources(args.csv)
    targets = json.loads((HERE / "reference.json").read_text())
    proposals = json.loads((HERE / "proposals.json").read_text())
    results = {}
    for arm, predictions in proposals.items():
        if predictions.keys() != transcripts.keys():
            raise ValueError(f"Case coverage differs for {arm}")
        routes = {"checked": checked_note}
        if arm in {"flash", "pro"}:
            routes["raw"] = raw_note
        for route, evaluate in routes.items():
            counts = dict(fields=0, exact=0, expected_assertions=0, positive_exact=0, omissions=0,
                          nonexact_assertions=0, unsupported=0, unusable_fields=0, malformed_fields=0, excluded=0)
            for key, transcript in transcripts.items():
                reference = {row["field"]: row for row in targets if row["id"] == key}
                note = evaluate(dict(id=key, transcript=transcript, gold=reference), predictions[key])
                for field, target in reference.items():
                    if target["status"] == "exclude":
                        counts["excluded"] += 1
                        continue
                    counts["fields"] += 1
                    positive = target["status"] == "supported"
                    counts["expected_assertions"] += positive
                    claim = note.claims[field]
                    asserted = claim.state in {"supported", "absent"}
                    if not note.usable:
                        counts["unusable_fields"] += 1
                    elif claim.malformed:
                        counts["malformed_fields"] += 1
                    elif positive:
                        exact = (claim.state == "supported" and len(claim.spans) == 1 and
                                 any(claim.spans[0][:2] == (opt["start"], opt["end"]) for opt in target["options"]))
                        counts["positive_exact"] += exact
                        counts["exact"] += exact
                        counts["omissions"] += not asserted
                        counts["nonexact_assertions"] += asserted and not exact
                    else:
                        counts["exact"] += not asserted
                        counts["unsupported"] += asserted
            results[f"{arm}-{route}"] = counts
    print(json.dumps(results, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
