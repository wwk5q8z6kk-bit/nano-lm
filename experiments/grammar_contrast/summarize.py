"""Summarize per-pair margins into the metrics fixed in PROTOCOL.md."""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
ARMS = ("unigram", "S2", "M0", "M2", "pythia-step256", "pythia-final")


def summarize(record):
    by_paradigm, by_family = defaultdict(list), defaultdict(list)
    equal, ties = [], 0
    for paradigm, _pair, family, good_len, bad_len, margin in record["rows"]:
        win = margin > 0  # a tie counts as zero wins
        ties += margin == 0
        by_paradigm[paradigm].append(win)
        by_family[family].append(win)
        if good_len == bad_len:
            equal.append(win)
    mean = lambda xs: sum(xs) / len(xs)
    wins = sum(map(sum, by_paradigm.values()))
    return {"pairs": record["pairs"], "wins": wins, "ties": ties,
            "pair_accuracy": wins / record["pairs"],
            "paradigm_macro_accuracy": mean([mean(v) for v in by_paradigm.values()]),
            "equal_length_pairs": len(equal), "equal_length_accuracy": mean(equal),
            "families": {k: mean(v) for k, v in sorted(by_family.items())}}


def main():
    records = {arm: json.loads((ROOT / "results" / f"{arm}.json").read_text()) for arm in ARMS}
    panels = {r["panel"] for r in records.values()}
    pin = json.loads((ROOT / "panel.json").read_text())
    if len(panels) != 1 or any(len(r["rows"]) != 2144 or len({x[0] for x in r["rows"]}) != 67 for r in records.values()):
        raise ValueError("Arms were not scored on one complete panel")
    print(json.dumps({"panel": panels.pop(), "pinned_panel": pin, "arms": {a: summarize(r) for a, r in records.items()}}, indent=2))


if __name__ == "__main__":
    main()
