"""Gate pins for corpus/context-window compatibility.

The 2026-08-24 native30 wave trained on a corpus whose every prompt exceeded
max_seq, and no build gate noticed. `sequence_budget` closes the half of that
class the corpus can own: whether a row is learnable inside the window at all.

Full diagnosis: artifacts/campaign/reval_results/FALSE_NULL_DIAGNOSIS.md
"""

from __future__ import annotations

import types

from nanoscribe.native.corpus.validate import sequence_budget


def _ex(prompt: str, target: str, raw_value: str, atom_id: str = "a0"):
    return types.SimpleNamespace(
        prompt=prompt, target=target, raw_value=raw_value, atom_id=atom_id
    )


def test_short_rows_pass_cleanly() -> None:
    r = sequence_budget([_ex("Transcript: sore throat here", "ASSERTED: sore throat",
                             "sore throat")], max_seq=512)
    assert r["pass"]
    assert r["n_over_budget"] == 0
    assert r["n_prompt_truncated"] == 0
    assert r["relies_on_left_truncation"] is False


def test_span_scrolling_out_of_the_window_fails_the_gate() -> None:
    """The failure mode left-truncation introduces: gold outside the model's view."""
    prompt = "sore throat" + "X" * 900  # span lives at the far head, gets cut
    r = sequence_budget([_ex(prompt, "ASSERTED: sore throat", "sore throat")],
                        max_seq=512)
    assert not r["pass"]
    assert not r["spans_visible"]
    assert r["n_span_lost_to_truncation"] == 1
    assert r["span_lost_examples"] == ["a0"]


def test_span_surviving_truncation_passes_but_flags_the_dependency() -> None:
    """Over-budget is tolerable when the span is still visible -- but it is a
    dependency on the trainer's left-truncation contract, and must be flagged."""
    prompt = "X" * 900 + " patient reports sore throat today"
    r = sequence_budget([_ex(prompt, "ASSERTED: sore throat", "sore throat")],
                        max_seq=512)
    assert r["pass"]
    assert r["n_over_budget"] == 1
    assert r["n_prompt_truncated"] == 1
    assert r["relies_on_left_truncation"] is True


def test_oversized_target_fails_the_gate() -> None:
    r = sequence_budget([_ex("short prompt", "ASSERTED: " + "y" * 600, "y" * 600)],
                        max_seq=512)
    assert not r["pass"]
    assert not r["target_fits"]


def test_abstention_rows_need_no_span() -> None:
    """NOT_MENTIONED rows carry no raw_value and must not be counted as lost."""
    r = sequence_budget([_ex("X" * 900, "NOT_MENTIONED", "")], max_seq=512)
    assert r["pass"]
    assert r["n_span_lost_to_truncation"] == 0


def test_gate_accepts_the_real_screen_corpus() -> None:
    """The shipped corpus is learnable under the fixed trainer: all 19,194 rows
    are over budget, yet every gold span survives left-truncation."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / (
        "artifacts/campaign/native_corpus_screen_v1_train.json"
    )
    if not path.is_file():  # untracked artifact; skip where absent
        import pytest

        pytest.skip("native_corpus_screen_v1_train.json not present")

    entries = json.loads(path.read_text())["entries"]
    rows = [_ex(e["prompt"], e["target"], e.get("raw_value", ""), e["atom_id"])
            for e in entries]
    r = sequence_budget(rows, max_seq=512)
    assert r["pass"]
    assert r["n_over_budget"] == len(rows)
    assert r["n_span_lost_to_truncation"] == 0
    assert r["relies_on_left_truncation"] is True
