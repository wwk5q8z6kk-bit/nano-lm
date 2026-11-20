"""Instrument admission must fail before loading any model."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.run_p1_pilot import DATA, Replay, admit, digest, preflight, proposal_fields
from nano_ai.contract import NanoOutput
from nano_ai.evaluation import evaluate_solver


def snapshot_replay(tmp_path, path, provenance, inputs, *, legacy=False):
    """Replay recorded proposals with authenticated historical source files.

    These artifacts pin only the modules they snapshot. Unpinned dependencies
    still come from the repository; the isolated interpreter prevents imports
    of today's development modules from overriding the recorded snapshots.
    No model, targets, or saved drafts enter the replay process.
    """
    root = Path(__file__).resolve().parents[1]
    overlay = tmp_path / path.name
    snapshots = {}
    for source, expected in provenance["code_hashes"].items():
        name = "runner.py.txt" if source == "scripts/run_p1_pilot.py" else Path(source).name + ".txt"
        snapshot = path / name
        assert digest(snapshot) == expected, f"snapshot hash mismatch: {snapshot}"
        snapshots[source] = snapshot.read_bytes()
    # Authenticate every snapshot before constructing or executing the overlay.
    for source, contents in snapshots.items():
        target = overlay / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)
    for package in ("nanoscribe", "nano_ai", "nano_ai/adapters"):
        target = overlay / package / "__init__.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            f"__path__.append({str(root / package)!r})\n"
            + (root / package / "__init__.py").read_text()
        )
    program = """
import importlib
import json
from pathlib import Path
import sys

overlay, root = map(Path, sys.argv[1:])
sys.path[:0] = [str(overlay), str(root)]
payload = json.load(sys.stdin)
for source in payload['modules']:
    module = importlib.import_module(source[:-3].replace('/', '.'))
    assert Path(module.__file__).resolve() == (overlay / source).resolve(), source

from nano_ai.contract import NanoInput
from nano_ai.solver import run_inference
from nanoscribe.compact import compact_solver
from nanoscribe.scribe import scribe

options = payload['options']
drafts, legacy_outputs = [], []
for case in payload['inputs']:
    def predict(transcript):
        assert transcript == case['transcript']
        return case['raw']
    solver = compact_solver(predict, binder='context', **options)
    drafts.append(scribe(case['transcript'], encounter_id=case['id'],
                         solver=solver, context_policy='context'))
    if payload['legacy']:
        solver = compact_solver(predict, **options)
        result = run_inference(solver, NanoInput(case['id'], case['transcript']))
        legacy_outputs.append(result.output.to_dict() if result.output else None)
print(json.dumps({'drafts': drafts, 'legacy_outputs': legacy_outputs}))
"""
    payload = {
        "modules": [source for source in snapshots if not source.startswith("scripts/")],
        "inputs": inputs,
        "options": ({"version": provenance["checkpoint_revision"]}
                    if "checkpoint_revision" in provenance else {}),
        "legacy": legacy,
    }
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program, str(overlay), str(root)],
        input=json.dumps(payload), text=True, capture_output=True,
        cwd=tmp_path, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def changed_data(tmp_path, mutate):
    rows = json.loads((DATA / "dialogues.json").read_text())
    spec = json.loads((DATA / "spec.json").read_text())
    mutate(rows, spec)
    (tmp_path / "dialogues.json").write_text(json.dumps(rows))
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    return tmp_path


def test_pilot_controls_and_world_pairs():
    rows, cases, controls = preflight()
    assert len(rows) == len(cases) == 20
    assert controls["oracle"]["quality"]["grounded_exact_field_accuracy"] == 1
    assert controls["all_missing"]["quality"]["state_accuracy"] == .12


@pytest.mark.parametrize("mutate", [
    lambda r, s: r[0].update(split="train"),
    lambda r, s: r[0].update(world_id="different-world"),
    lambda r, s: r[0].update(transcript=r[1]["transcript"]),
    lambda r, s: r[0].update(target=r[0]["target"].replace("jaw pain", "invented")),
    lambda r, s: s["expected_counts"]["medication"].update(absent=7),
])
def test_instrument_rejects_leakage_source_or_quota_mutations(tmp_path, mutate):
    with pytest.raises(ValueError):
        admit(changed_data(tmp_path, mutate))


@pytest.mark.parametrize("run", ["run-02", "run-03", "run-08", "run-09"])
def test_stored_results_recompute_without_model_or_gold_at_inference(run):
    rows, cases = admit()
    path = DATA / run
    records = json.loads((path / "encounters.json").read_text())
    assert [r["example_id"] for r in records] == [r["example_id"] for r in rows]
    provenance = json.loads((path / "provenance.json").read_text())
    controls = json.loads((path / "controls.json").read_text())
    assert all(digest(DATA / name) == expected for name, expected in controls["source_hashes"].items())
    assert digest(path / "compact.py.txt") == provenance["code_hashes"]["nanoscribe/compact.py"]
    assert digest(path / "runner.py.txt") == provenance["code_hashes"]["scripts/run_p1_pilot.py"]
    prompt = json.loads((path / "prompt.json").read_text())["system"]
    import hashlib
    assert hashlib.sha256(prompt.encode()).hexdigest() == provenance["prompt_sha256"]
    raw, checked = {}, {}
    for row, record in zip(rows, records, strict=True):
        item_id = row["example_id"]
        try:
            raw[item_id] = proposal_fields(record["generation"]["raw"], row["transcript"])
        except ValueError as exc:
            raw[item_id] = exc
        output = record["draft"]["extraction"]
        checked[item_id] = NanoOutput.from_dict(output).fields if output else ValueError("stored inference failure")
        if record["draft"]["status"] == "failed":
            assert record["draft"]["note"] is None
    for name, outputs in (("raw_evaluation", raw), ("checked_evaluation", checked)):
        stored = json.loads((path / (name + ".json")).read_text())
        replay = evaluate_solver(Replay(outputs, "stored-output-audit"), cases)
        assert replay.quality == stored["quality"]
        assert replay.failures == stored["failures"]


@pytest.mark.parametrize("run", ["run-08", "run-09"])
def test_context_run_provenance_and_checked_path_replay(run, tmp_path):
    path = DATA / run
    records = json.loads((path / "encounters.json").read_text())
    provenance = json.loads((path / "provenance.json").read_text())
    assert provenance["decoding"] == "constrained"
    assert provenance["binder"] == "context"
    raw = {row["example_id"]: row["generation"]["raw"] for row in records}
    rows, cases = admit()
    inputs = [{"id": row["example_id"], "transcript": row["transcript"],
               "raw": raw[row["example_id"]]} for row in rows]
    replay = snapshot_replay(tmp_path, path, provenance, inputs, legacy=True)
    for draft, saved in zip(replay["drafts"], records, strict=True):
        assert draft == saved["draft"]
    legacy = {row["example_id"]: (NanoOutput.from_dict(output).fields if output
                                else ValueError("historical replay inference failure"))
              for row, output in zip(rows, replay["legacy_outputs"], strict=True)}
    evaluated = evaluate_solver(Replay(legacy, "historical-legacy-replay"), cases)
    saved = json.loads((path / "legacy_evaluation.json").read_text())
    assert evaluated.quality == saved["quality"]
    assert evaluated.failures == saved["failures"]


def test_additional_model_challenge_replays_and_preserves_failure(tmp_path):
    path = DATA / "challenge-01"
    rows = json.loads((path / "cases.json").read_text())
    results = json.loads((path / "results.json").read_text())
    provenance = json.loads((path / "provenance.json").read_text())
    assert digest(path / "cases.json") == provenance["source_sha256"]
    assert len(rows) == len(results) == 6
    inputs = [{"id": row["id"], "transcript": row["transcript"], "raw": saved["generation"]["raw"]}
              for row, saved in zip(rows, results, strict=True)]
    replay = snapshot_replay(tmp_path, path, provenance, inputs)
    for row, saved, draft in zip(rows, results, replay["drafts"], strict=True):
        assert draft == saved["draft"]
        claims = draft["note"]["claims"]
        actual = {c["field"]: c["value"] for c in claims if c["state"] == "supported"}
        states = {c["field"]: c["state"] for c in claims}
        passed = (draft["status"] == "draft" and draft["disposition"] == row["disposition"]
                  and actual == row["supported_claims"]
                  and all(states.get(k) == v for k, v in row.get("states", {}).items()))
        assert passed == saved["passed"]
    assert sum(r["passed"] for r in results) == provenance["passed"] == 5
    conflict = next(r for r in results if r["id"] == "medication-conflict")
    assert not conflict["passed"]
    assert conflict["draft"]["disposition"] == "abstain"
    assert all(c["state"] not in ("supported", "absent") for c in conflict["draft"]["note"]["claims"])


def test_generation_failure_replays_for_all_pilot_binders():
    from scripts.run_p1_pilot import PilotGeneration

    calls = []
    def transient(transcript):
        calls.append(transcript)
        if len(calls) == 1:
            raise ValueError("initial decode failed")
        return "unobserved"

    generate = PilotGeneration(transient)
    for _ in range(3):
        with pytest.raises(ValueError, match="initial decode failed"):
            generate("same transcript")
    assert calls == ["same transcript"]
