"""Model-free integration regressions for the compact checked path."""

import pytest

from nanoscribe.compact import LocalCompactPredictor, compact_solver
from nanoscribe.scribe import scribe


def test_compact_transcript_only_proposal_reaches_checked_note():
    transcript = "Doctor: What brings you to the clinic today?\nPatient: I've been having jaw pain."
    seen = []
    def predict(source):
        seen.append(source)
        return "CC:S[jaw pain]|DUR:M|SEV:M|MED:M|ALG:M"
    result = scribe(transcript, solver=compact_solver(predict))
    assert seen == [transcript]
    assert result["status"] == "draft"
    assert result["disposition"] == "review"
    assert result["note"]["claims"][0]["value"] == "jaw pain"


@pytest.mark.parametrize("raw", [
    "CC:S[invented]|DUR:M|SEV:M|MED:M|ALG:M",
    "CC:S[jaw pain]|DUR:M|SEV:M|MED:M|ALG:M\nExplanation",
])
def test_invalid_compact_generation_withholds_note(raw):
    result = scribe("Patient: jaw pain", solver=compact_solver(lambda _: raw))
    assert result["status"] == "failed"
    assert result.get("note") is None


def test_novel_question_cannot_bypass_binder():
    result = scribe("Doctor: Describe the problem that brought you here.\nPatient: I've been having jaw pain.",
                    solver=compact_solver(lambda _: "CC:S[jaw pain]|DUR:M|SEV:M|MED:M|ALG:M"))
    assert result["disposition"] == "abstain"
    assert result["note"]["claims"][0]["value"] is None


def test_compact_loader_requires_local_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        LocalCompactPredictor(str(tmp_path / "missing"))
    with pytest.raises(ValueError):
        LocalCompactPredictor(str(tmp_path), max_new_tokens=0)


def test_cli_selects_compact_prompt_without_passing_gold(tmp_path, monkeypatch, capsys):
    from nanoscribe import compact
    from nanoscribe.scribe import main
    transcript = "Doctor: What brings you to the clinic today?\nPatient: I've been having jaw pain."
    source = tmp_path / "source.txt"
    source.write_text(transcript)
    def load(checkpoint, *, system_prompt):
        assert checkpoint == str(tmp_path)
        assert system_prompt == compact.FEWSHOT_PROMPT
        def predict(text):
            assert text == transcript
            return "CC:S[jaw pain]|DUR:M|SEV:M|MED:M|ALG:M"
        return predict
    monkeypatch.setattr(compact, "LocalCompactPredictor", load)
    assert main(["--transcript", str(source), "--compact-checkpoint", str(tmp_path)]) == 0
    assert '"disposition": "review"' in capsys.readouterr().out
