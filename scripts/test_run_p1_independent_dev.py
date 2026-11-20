"""Development harness tests use only independent toy data, never the fresh set."""

import copy
import json

import pytest

from scripts import run_p1_independent_dev as runner


SOURCE = "Doctor: What is bothering you?\nPatient: I've been having hiccups.\nDoctor: What medication?\nPatient: I haven't taken anything."
RAW = "CC:S[hiccups]|DUR:M|SEV:M|MED:A[I haven't taken anything.]|ALG:M"


def write_data(tmp_path, *, source=SOURCE, gold=None):
    data = tmp_path / "data"
    data.mkdir()
    rows = [{"case_id": "toy-1", "stratum": "harness-toy", "transcript": source}]
    if gold is None:
        fields = {label: {"state": "M", "spans": []} for label in runner.LABELS}
        fields["CC"] = {"state": "S", "spans": ["hiccups"]}
        fields["MED"] = {"state": "A", "spans": ["I haven't taken anything."]}
        gold = [{"case_id": "toy-1", "fields": fields}]
    (data / "sources.json").write_text(json.dumps(rows))
    (data / "gold.json").write_text(json.dumps(gold))
    relock(data)
    return data


def relock(data):
    (data / "annotation_lock.json").write_text(json.dumps({"files": {
        name: runner.digest(data / name) for name in ("sources.json", "gold.json")}}))


def mutate(data, name, edit):
    value = json.loads((data / name).read_text())
    edit(value)
    (data / name).write_text(json.dumps(value))
    relock(data)


def test_preflight_direct_gold_and_data_dependent_negative_control(tmp_path):
    data = write_data(tmp_path)
    _, cases, controls = runner.preflight(data)
    assert cases[0].fixture is not None
    assert controls["primary_oracle"]["metrics"]["raw"]["grounded_exact_field_accuracy"] == 1
    assert controls["primary_all_missing"]["metrics"]["raw"]["state_accuracy"] == .6
    assert controls["secondary_transportable_case_count"] == 1


def test_natural_denial_outside_contract_remains_gold_and_raw_truth(tmp_path):
    source = SOURCE.replace("I haven't taken anything.", "I have used no medicines whatsoever.")
    data = write_data(tmp_path)
    mutate(data, "sources.json", lambda rows: rows[0].update(transcript=source))
    mutate(data, "gold.json", lambda rows: rows[0]["fields"]["MED"].update(spans=["I have used no medicines whatsoever."]))
    _, cases, controls = runner.preflight(data)
    assert cases[0].fixture is None
    assert controls["contract_transport_errors"][0]["field"] == "medication"
    assert controls["primary_oracle"]["fields"] == 5
    assert controls["primary_oracle"]["metrics"]["raw"]["state_accuracy"] == 1
    fields = runner.raw_fields(RAW.replace("I haven't taken anything.", "I have used no medicines whatsoever."), source)
    assert runner.field_score(fields[3], cases[0].fields[3])["grounded_exact_correct"]


@pytest.mark.parametrize("mutation,match", [
    (lambda rows: rows.append(copy.deepcopy(rows[0])), "duplicate annotation case ID"),
    (lambda rows: rows[0]["fields"].pop("ALG"), "exactly CC"),
    (lambda rows: rows[0]["fields"]["CC"].update(spans=["invented"]), "exactly once"),
    (lambda rows: rows[0]["fields"]["CC"].update(spans=["What is bothering you?"]), "not inside a Patient"),
    (lambda rows: rows[0]["fields"]["CC"].update(state="M"), "cannot have evidence"),
    (lambda rows: rows[0]["fields"]["CC"].update(state="S", spans=[]), "require source evidence"),
    (lambda rows: rows[0]["fields"]["CC"].update(state="C"), "two textually distinct"),
    (lambda rows: rows[0]["fields"]["CC"].update(state="A"), "restricted to MED/ALG"),
    (lambda rows: rows[0]["fields"]["CC"].update(spans=["hiccups", "hiccups"]), "duplicate annotation evidence"),
    (lambda rows: rows[0].update(case_id="wrong"), "IDs must match"),
])
def test_bad_annotations_rejected_before_prediction(tmp_path, mutation, match):
    data = write_data(tmp_path)
    mutate(data, "gold.json", mutation)
    with pytest.raises(ValueError, match=match):
        runner.admit(data)


def test_ambiguous_quotes_require_explicit_offsets(tmp_path):
    source = SOURCE + "\nDoctor: Repeat the symptom.\nPatient: hiccups"
    data = write_data(tmp_path, source=source)
    with pytest.raises(ValueError, match="exactly once"):
        runner.admit(data)
    mutate(data, "gold.json", lambda rows: rows[0]["fields"]["CC"].update(
        spans=[{"text": "hiccups", "start": source.index("hiccups")}]))
    assert runner.admit(data)[1][0].fields[0]["evidence"][0]["start"] == source.index("hiccups")
    mutate(data, "gold.json", lambda rows: rows[0]["fields"]["CC"]["spans"][0].update(start=0))
    with pytest.raises(ValueError, match="does not match"):
        runner.admit(data)


def test_lock_detects_changed_gold_and_changed_lock(tmp_path):
    data = write_data(tmp_path)
    admitted = runner.verify_lock(data)
    (data / "gold.json").write_text((data / "gold.json").read_text() + " ")
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.verify_lock(data)
    relock(data)
    with pytest.raises(ValueError, match="changed after admission"):
        runner.verify_lock(data, admitted)


def test_lock_rejects_path_escape_and_duplicate_json_keys(tmp_path):
    data = write_data(tmp_path)
    lock = json.loads((data / "annotation_lock.json").read_text())
    lock["files"]["../outside.json"] = "0" * 64
    (data / "annotation_lock.json").write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="invalid annotation lock path"):
        runner.verify_lock(data)
    (data / "annotation_lock.json").write_text('{"files":{},"files":{}}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        runner.verify_lock(data)


@pytest.mark.parametrize("alias", ["absolute", "./sources.json", "nested/../sources.json", "nested//source.json", "sources.json/"])
def test_lock_requires_canonical_relative_paths(tmp_path, alias):
    data = write_data(tmp_path)
    name = str(data / "sources.json") if alias == "absolute" else alias
    lock = json.loads((data / "annotation_lock.json").read_text())
    lock["files"][name] = runner.digest(data / "sources.json")
    (data / "annotation_lock.json").write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="invalid annotation lock path"):
        runner.verify_lock(data)


def test_collect_invokes_predictor_once_with_transcript_only_and_retains_raw(tmp_path):
    data = write_data(tmp_path)
    _, cases, _ = runner.preflight(data)
    calls = []
    def predictor(transcript):
        calls.append(transcript)
        assert isinstance(transcript, str)
        assert "state" not in transcript and "toy-1" not in transcript
        return RAW
    encounters = runner.collect(cases, predictor, tmp_path / "run")
    assert calls == [SOURCE]
    assert encounters[0]["generation"]["raw"] == RAW
    assert encounters[0]["raw_parse_error"] is None
    comparison, details = runner.compare(cases, encounters)
    assert comparison["metrics"]["raw"]["grounded_exact_field_accuracy"] == 1
    assert len(details) == 5
    assert comparison["final_note"]["emitted_asserted_fields"] == 2
    assert comparison["final_note"]["correct_asserted_fields_emitted"] == 2
    assert comparison["by_field"]["medication"]["raw"]["state_accuracy"] == 1
    assert comparison["by_stratum"]["harness-toy"]["raw"]["fields"] == 5
    assert json.loads((tmp_path / "run" / "encounters.json").read_text())[0]["generation"]["raw"] == RAW


def test_malformed_outputs_count_all_fields_and_whole_note_abstentions(tmp_path):
    data = write_data(tmp_path)
    _, cases, _ = runner.preflight(data)
    encounters = runner.collect(cases, lambda _: "bad output", tmp_path / "run")
    comparison, _ = runner.compare(cases, encounters)
    assert comparison["malformed_raw_cases"] == 1
    assert comparison["fields"] == 5
    assert comparison["metrics"]["raw"]["state_accuracy"] == 0
    assert comparison["metrics"]["checked"]["state_accuracy"] == 0
    assert comparison["whole_note_abstention_with_gold_usable_content"] == 1
    assert comparison["final_note"]["emitted_asserted_fields"] == 0


def test_paired_loss_and_salvage_metrics(tmp_path):
    data = write_data(tmp_path)
    _, cases, _ = runner.preflight(data)
    gold = list(cases[0].fields)
    raw, checked = copy.deepcopy(gold), copy.deepcopy(gold)
    raw[1] = {"field": "duration", "state": "uncertain", "value": None, "evidence": []}
    checked[0] = {"field": "chief_complaint", "state": "uncertain", "value": None, "evidence": []}
    summary, _ = runner.compare(cases, [{"case_id": "toy-1", "raw_fields": raw, "raw_parse_error": None,
                                        "draft": {"extraction": {"fields": checked}, "disposition": "review"}}])
    assert summary["paired_changes"]["exact_fields_salvaged"] == 1
    assert summary["paired_changes"]["exact_fields_lost"] == 1
    assert summary["paired_changes"]["correct_asserted_fields_lost"] == 1
    assert summary["paired_changes"]["correct_asserted_fields_retained"] == 1
    assert summary["metrics"]["raw"]["conflict_state_recall"] is None
    with pytest.raises(ValueError, match="exactly one encounter"):
        runner.compare(cases, [])


def test_prediction_field_order_is_validated(tmp_path):
    data = write_data(tmp_path)
    _, cases, _ = runner.preflight(data)
    fields = list(reversed(cases[0].fields))
    with pytest.raises(ValueError, match="canonical order"):
        runner.compare(cases, [{"case_id": "toy-1", "raw_fields": fields, "raw_parse_error": None,
                               "draft": {"extraction": None, "disposition": "abstain"}}])


def test_conflict_recall_and_full_evidence_distinguish_state_from_support(tmp_path):
    source = SOURCE + "\nDoctor: What is bothering you?\nPatient: I came in because of a rash."
    data = write_data(tmp_path, source=source)
    mutate(data, "gold.json", lambda rows: rows[0]["fields"]["CC"].update(state="C", spans=["hiccups", "a rash"]))
    _, cases, _ = runner.preflight(data)
    raw = runner.raw_fields(RAW, source)
    comparison, _ = runner.compare(cases, [{"case_id": "toy-1", "raw_fields": raw, "raw_parse_error": None,
                                           "draft": {"extraction": {"fields": cases[0].fields}, "disposition": "review"}}])
    assert comparison["metrics"]["raw"]["conflict_state_recall"] == 0
    assert comparison["metrics"]["checked"]["conflict_exact_recall"] == 1
    assert comparison["paired_changes"]["asserted_normalized_content_mismatches_removed"] == 1
    assert comparison["paired_changes"]["asserted_exact_mismatches_removed"] == 1
    uncertain = {**cases[0].fields[0], "state": "uncertain"}
    score = runner.field_score({**uncertain, "evidence": []}, uncertain)
    assert score["grounded_exact_correct"]
    assert not score["complete_evidence_exact_correct"]


def test_conflict_requires_all_gold_observations_including_denial(tmp_path):
    source = SOURCE + "\nDoctor: What medication?\nPatient: I've been taking ibuprofen.\nDoctor: Which medication?\nPatient: Just naproxen."
    data = write_data(tmp_path, source=source)
    mutate(data, "gold.json", lambda rows: rows[0]["fields"]["MED"].update(
        state="C", spans=["ibuprofen", "naproxen", "I haven't taken anything."]))
    _, cases, _ = runner.preflight(data)
    gold = cases[0].fields[3]
    compatible_pair_only = {**gold, "evidence": gold["evidence"][:2]}
    score = runner.field_score(compatible_pair_only, gold)
    assert score["state_correct"]
    assert not score["exact_evidence_aligned"]
    assert not score["grounded_exact_correct"]
    assert runner.field_score(gold, gold)["grounded_exact_correct"]


@pytest.mark.parametrize("failure_stage", ["constructor", "checkpoint_hashing", "source_snapshot"])
def test_setup_failures_have_terminal_provenance(tmp_path, monkeypatch, failure_stage):
    data = write_data(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    class Predictor:
        def __init__(self, path, **kwargs):
            if failure_stage == "constructor":
                raise OSError("toy setup failure")
            self.checkpoint, self.max_new_tokens = path, 256
    original_digest = runner.digest
    def digest(path):
        if failure_stage == "checkpoint_hashing" and path == checkpoint / "config.json":
            raise OSError("toy setup failure")
        return original_digest(path)
    def source_snapshot(_):
        if failure_stage == "source_snapshot":
            raise OSError("toy setup failure")
        return {}
    monkeypatch.setattr(runner, "LocalCompactPredictor", Predictor)
    monkeypatch.setattr(runner, "digest", digest)
    monkeypatch.setattr(runner, "source_snapshot", source_snapshot)
    output = tmp_path / "run"
    with pytest.raises(OSError, match="toy setup failure"):
        runner.main(["--data", str(data), "--checkpoint", str(checkpoint), "--output", str(output)])
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["status"] == "failed"
    assert "toy setup failure" in provenance["error"]
    assert provenance["total_seconds"] >= 0
    assert "finished_utc" in provenance
    assert not (output / "encounters.json").exists()


def test_run_fails_if_inputs_change_during_generation(tmp_path, monkeypatch):
    data = write_data(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    class Predictor:
        def __init__(self, path, **kwargs):
            self.checkpoint, self.max_new_tokens = path, 256
        def __call__(self, transcript):
            (data / "gold.json").write_text((data / "gold.json").read_text() + " ")
            return RAW
    monkeypatch.setattr(runner, "LocalCompactPredictor", Predictor)
    monkeypatch.setattr(runner, "source_snapshot", lambda _: {})
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.main(["--data", str(data), "--checkpoint", str(checkpoint), "--output", str(output)])
    assert json.loads((output / "provenance.json").read_text())["status"] == "failed"
    assert (output / "encounters.json").exists()
    assert not (output / "comparison.json").exists()


def test_preflight_never_overwrites_prior_run(tmp_path):
    data = write_data(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    (output / "sentinel").write_text("preserved")
    with pytest.raises(FileExistsError):
        runner.main(["--data", str(data), "--output", str(output), "--preflight"])
    assert (output / "sentinel").read_text() == "preserved"


def test_grounded_development_scoring_preserves_fractional_decimal_content(tmp_path):
    source = "Patient: The pain is .5/10."
    fields = {label: {"state": "M", "spans": []} for label in runner.LABELS}
    fields["SEV"] = {"state": "S", "spans": [".5/10"]}
    data = write_data(tmp_path, source=source, gold=[{"case_id":"toy-1", "fields":fields}])
    _, cases, _ = runner.admit(data)
    gold = cases[0].fields[2]
    raw = runner.raw_fields("CC:M|DUR:M|SEV:S[.5/10]|MED:M|ALG:M", source, policy="grounded")[2]
    assert raw["value"] == gold["value"] == ".5/10"
    assert runner.field_score(raw, gold)["grounded_exact_correct"]
    corrupted = {**raw, "value": "5/10"}
    score = runner.field_score(corrupted, gold)
    assert score["exact_evidence_aligned"]
    assert not score["normalized_content_correct"] and not score["grounded_exact_correct"]


def test_two_decimal_disagreement_annotations_remain_distinct():
    source = "Patient: .5/10. Patient: 5/10."
    annotation = {"state": "C", "spans": [{"start": 9, "text": ".5/10"},
                                          {"start": source.rindex("5/10"), "text": "5/10"}]}
    assert runner._field_annotation("SEV", annotation, source)["state"] == "conflicting"
