"""Additional exposed development smoke cases; no private data or clinical claim.

Usage: python artifacts/p1_pilot_v1/challenge-01/run.py /pinned/checkpoint
Outputs are append-only to a new results.json (refuses overwrite).
"""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from nanoscribe.compact import LocalCompactPredictor, STRUCTURED_PROMPT, compact_solver
from nanoscribe.scribe import scribe

HERE = Path(__file__).parent
source_path = HERE / "cases.json"
output = HERE / "results.json"
if output.exists():
    raise SystemExit("Refusing to overwrite an existing challenge run")
sources = ["nanoscribe/compact.py", "nanoscribe/context.py", "nanoscribe/context_solver.py", "nanoscribe/check_note.py", "nanoscribe/scribe.py", "nano_ai/adapters/state_span.py"]
for source in sources:
    (HERE / (Path(source).name + ".txt")).write_bytes((ROOT / source).read_bytes())
(HERE / "runner.py.txt").write_bytes(Path(__file__).read_bytes())
rows = json.loads(source_path.read_text())
predictor = LocalCompactPredictor(sys.argv[1], decoding="constrained")
solver = compact_solver(predictor, binder="context")
results = []
start = time.perf_counter()
for row in rows:
    draft = scribe(row["transcript"], encounter_id=row["id"], solver=solver, context_policy="context")
    claims = draft["note"]["claims"] if draft.get("note") else []
    states = {c["field"]: c["state"] for c in claims}
    actual = {c["field"]: c["value"] for c in claims if c["state"] == "supported"}
    passed = (draft["status"] == "draft" and draft["disposition"] == row["disposition"]
              and actual == row["supported_claims"]
              and all(states.get(k) == v for k, v in row.get("states", {}).items()))
    results.append({"id": row["id"], "passed": passed, "draft": draft,
                    "generation": predictor.records.get(row["transcript"], {})})
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(row["id"], passed, draft["disposition"], flush=True)
provenance = {"checkpoint": str(Path(sys.argv[1]).resolve()), "paid_compute_usd": 0,
    "development_only": True, "total_seconds": time.perf_counter() - start,
    "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    "prompt_sha256": hashlib.sha256(STRUCTURED_PROMPT.encode()).hexdigest(),
    "code_hashes": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
        sources},
    "passed": sum(r["passed"] for r in results), "total": len(results)}
(HERE / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
