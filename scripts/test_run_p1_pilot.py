"""Instrument admission must fail before loading any model."""

import json
from pathlib import Path

import pytest

from scripts.run_p1_pilot import DATA, Replay, admit, digest, preflight, proposal_fields
from nano_ai.contract import NanoOutput
from nano_ai.evaluation import evaluate_solver


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


@pytest.mark.parametrize("run", ["run-02", "run-03"])
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
