"""Recompute document/pooled reference losses and paired contrasts from the chunk losses."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ARMS = ("S2", "M0", "M2", "Pythia-step256", "Pythia-final")


def summarize(results):
    sources = json.loads((HERE.parent / "sources.json").read_text())
    chunks = json.loads((HERE / "chunks.json").read_text())
    native = json.loads((HERE.parent / "inputs.json").read_text())["arms"]
    reference = json.loads((HERE / "inputs.json").read_text())
    digests = {arm: spec["sha256"] for arm, spec in native.items()}
    digests.update({arm: spec["files"]["model.safetensors"] for arm, spec in reference.items()})

    source_docs = {row["doc"]: row for row in sources}
    ends = {doc: 0 for doc in source_docs}
    coordinates = set()
    for c in chunks:
        doc, start, end = c["doc"], c["start"], c["end"]
        if not start == ends[doc] < end <= source_docs[doc]["bytes"]:
            raise ValueError("Chunk manifest leaves a gap or overlaps")
        coordinates.add((doc, start))
        ends[doc] = end
    if any(ends[doc] != row["bytes"] for doc, row in source_docs.items()):
        raise ValueError("Incomplete document coverage")

    bits = {arm: defaultdict(float) for arm in ARMS}
    for arm in ARMS:
        saved = json.loads((results / f"{arm}.json").read_text())
        actual = {(c["doc"], c["start"]) for c in saved["chunks"]}
        if (saved["arm"] != arm or saved["checkpoint_sha256"] != digests[arm]
                or len(saved["chunks"]) != len(coordinates) or actual != coordinates):
            raise ValueError(f"Incorrect checkpoint or incomplete results for {arm}")
        for c in saved["chunks"]:
            if not math.isfinite(c["bits"]) or c["bits"] < 0:
                raise ValueError(f"Invalid loss for {arm}")
            bits[arm][c["doc"]] += c["bits"]

    groups = defaultdict(list)
    for row in sources:
        groups[row["source"]].append(row["doc"])

    def aggregate(docs):
        size = sum(ends[doc] for doc in docs)
        return dict(bytes=size, documents=len(docs),
                    bpb={arm: sum(bits[arm][d] for d in docs) / size for arm in ARMS})

    comparisons = {}
    for ref in ("Pythia-final", "Pythia-step256"):
        diff = {d: bits[ref][d] - bits["M2"][d] for d in source_docs}
        rng = np.random.Generator(np.random.PCG64(20261120))
        draws = []
        for _ in range(2000):
            sample = [docs[i] for _, docs in sorted(groups.items()) for i in rng.integers(len(docs), size=len(docs))]
            draws.append(sum(diff[d] for d in sample) / sum(ends[d] for d in sample))
        comparisons[ref + "_minus_M2"] = dict(bpb=sum(diff.values()) / sum(ends.values()),
            lower_documents=sum(v < 0 for v in diff.values()),
            paired_bootstrap_95=np.percentile(draws, [2.5, 97.5]).tolist())
    return dict(chunks=len(chunks), pooled=aggregate(list(source_docs)),
                by_source={name: aggregate(docs) for name, docs in groups.items()},
                by_document={d: aggregate([d])["bpb"] for d in source_docs}, contrasts=comparisons)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=HERE / "results")
    print(json.dumps(summarize(ap.parse_args().results), indent=2, sort_keys=True, allow_nan=False))
