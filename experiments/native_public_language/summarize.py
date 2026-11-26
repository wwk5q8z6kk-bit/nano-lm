"""Recompute the descriptive results and paired document interval from the saved loss totals."""

import argparse
import json
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ARMS = ("S2", "M0", "M2")


def summarize(directory):
    docs = json.loads((HERE / "sources.json").read_text())
    control = json.loads((directory / "control.json").read_text())
    inputs = json.loads((HERE / "inputs.json").read_text())
    arms = {arm: json.loads((directory / f"{arm}.json").read_text()) for arm in ARMS}
    names = {doc["doc"] for doc in docs}
    if len(names) != len(docs) or names != set(control["rows"]):
        raise ValueError("Missing or duplicate document")
    for name, arm in arms.items():
        if (set(arm["rows"]) != names or arm["arm"] != name
                or arm["checkpoint_sha256"] != inputs["arms"][name]["sha256"]):
            raise ValueError(f"Unexpected score set for arm {name}")
    retained = [doc for doc in docs if not control["rows"][doc["doc"]]["exact_match"]]
    groups = {k: [doc for doc in retained if doc["source"] == k]
              for k in sorted({doc["source"] for doc in docs})}
    if not all(groups.values()):
        raise ValueError("At least one source has no eligible documents")

    def scores(rows):
        total = sum(doc["bytes"] for doc in rows)
        out = dict(documents=len(rows), bytes=total, tokens=sum(doc["tokens"] for doc in rows))
        out["unigram"] = sum(control["rows"][doc["doc"]]["unigram_bits"] for doc in rows) / total
        for name, arm in arms.items():
            out[name] = {cond: sum(arm["rows"][doc["doc"]][cond + "_bits"] for doc in rows) / total
                         for cond in ("ordered", "shuffled")}
        out["M2_minus_M0"] = out["M2"]["ordered"] - out["M0"]["ordered"]
        return out

    ordered_delta = {doc["doc"]: (arms["M2"]["rows"][doc["doc"]]["ordered_bits"]
                                  - arms["M0"]["rows"][doc["doc"]]["ordered_bits"]) for doc in retained}
    rng = np.random.Generator(np.random.PCG64(20261120))
    deltas = []
    for _ in range(2000):
        draw = [rows[i] for rows in groups.values() for i in rng.integers(len(rows), size=len(rows))]
        deltas.append(sum(ordered_delta[row["doc"]] for row in draw) / sum(row["bytes"] for row in draw))
    if not all(map(math.isfinite, deltas)):
        raise ValueError("Nonfinite bootstrap estimate")
    return dict(pooled=scores(retained), by_source={k: scores(v) for k, v in groups.items()},
                M2_lower_documents=sum(v < 0 for v in ordered_delta.values()),
                M2_minus_M0_paired_document_bootstrap_95=np.percentile(deltas, [2.5, 97.5]).tolist(),
                excluded_full_duplicates=[row["doc"] for row in docs if row not in retained],
                anchor_matches=sum(sum(row["anchors"]) for row in control["rows"].values()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=HERE / "results")
    print(json.dumps(summarize(ap.parse_args().results), indent=2, sort_keys=True, allow_nan=False))
