"""Nano LM research: constructed representation-transport anchor diagnostic.
This synthetic linear numerical check DOES NOT train a language model.
Run: python checks/transport_anchor_pilot.py
Output: results/transport_anchor_pilot_v0.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def run(seed: int, d: int = 16, classes: int = 4, n: int = 5000):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    W = rng.normal(size=(classes, d))
    X = rng.normal(size=(n + 100, d))
    Z = X @ Q.T
    labels = (X @ W.T).argmax(-1)
    x_val, z_val, y_val = X[100:], Z[100:], labels[100:]

    def accuracy(logits):
        return float((logits.argmax(-1) == y_val).mean())

    res = {"seed": seed, "dim": d, "n_test": len(y_val),
           "reference": accuracy(x_val @ W.T),
           "without_transport": accuracy(z_val @ W.T),
           "known_transform": accuracy((z_val @ Q) @ W.T)}
    res["estimate_by_anchor_count"] = {}
    for m in (4, 8, 12, 16, 24, 32, 64):
        estimated_inverse = np.linalg.pinv(Z[:m]) @ X[:m]
        res["estimate_by_anchor_count"][str(m)] = accuracy((z_val @ estimated_inverse) @ W.T)
    return res


def main():
    rows = [run(1100 + i) for i in range(5)]
    ks = list(rows[0]["estimate_by_anchor_count"])
    summary = {
        "reference": round(np.mean([r["reference"] for r in rows]), 4),
        "without_transport": round(np.mean([r["without_transport"] for r in rows]), 4),
        "known_transform": round(np.mean([r["known_transform"] for r in rows]), 4),
        "estimated": {
            k: {"mean": round(np.mean([r["estimate_by_anchor_count"][k] for r in rows]), 4),
                "min": round(min(r["estimate_by_anchor_count"][k] for r in rows), 4),
                "max": round(max(r["estimate_by_anchor_count"][k] for r in rows), 4)}
            for k in ks
        },
    }
    result = {
        "experiment": "transport_anchor_pilot_v0",
        "n_seeds": len(rows),
        "d": 16,
        "hypothesis": "generic unknown full-rank rotation needs independent paired anchors to reconstruct globally",
        "claim_limit": "synthetic linear features only; no LM learning, no energy measurement",
        "rows": rows,
        "summary": summary,
    }
    out = Path(__file__).resolve().parents[1] / "results" / "transport_anchor_pilot_v0.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
