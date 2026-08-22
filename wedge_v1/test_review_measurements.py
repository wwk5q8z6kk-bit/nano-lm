"""Structured Priority-5 review measurement regressions."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from wedge_v1.review import (
    PROVENANCE_SCHEMA,
    apply_label,
    batch_label,
    interactive_review,
    label_summary,
    merge_prior_labels,
)


def _card(
    card_id: str,
    *,
    result_fingerprint: str | None = None,
) -> dict:
    result_fingerprint = result_fingerprint or f"result-{card_id}"
    return {
        "id": card_id,
        "task_id": card_id.upper(),
        "task_class": "ask",
        "query": f"Question {card_id}?",
        "answer_status": "SUPPORTED",
        "document": f"doc-{card_id}",
        "evidence_span": f"Evidence {card_id}",
        "solver_used": ["test"],
        "latency_s": 0.01,
        "verifier_outcome": "SUPPORTED",
        "usefulness_label": None,
        "failure_reason": None,
        "suggested_correction": None,
        "correction_reason": None,
        "review_elapsed_s": None,
        "provenance": {
            "schema": PROVENANCE_SCHEMA,
            "corpus_digest": "corpus-v1",
            "task_fingerprint": f"task-{card_id}",
            "result_fingerprint": result_fingerprint,
        },
    }


def _state() -> dict:
    return {"schema": "nano-lm.wedge_v1.review.v1", "cards": {}, "labels": {}}


def _clock(*values: float):
    readings = iter(values)
    return lambda: next(readings)


def test_interactive_review_times_valid_label_without_resetting_after_invalid_input(
    tmp_path: Path,
):
    first = _card("first")
    second = _card("second")
    state = _state()
    stdout = io.StringIO()

    interactive_review(
        [first, second],
        state,
        stdin=io.StringIO(
            "not-a-label\n"
            "n needs a literal citation | NOT_USEFUL | cite the exact source\n"
            "u\n"
        ),
        stdout=stdout,
        state_path=tmp_path / "review.json",
        reviewer_kind="owner",
        clock=_clock(10.0, 12.0, 17.5, 30.0, 31.25),
    )

    assert state["labels"] == {
        first["id"]: "NOT_USEFUL",
        second["id"]: "USEFUL",
    }
    assert state["cards"][first["id"]]["review_elapsed_s"] == 7.5
    assert state["cards"][second["id"]]["review_elapsed_s"] == 1.25
    assert state["cards"][first["id"]]["correction_reason"] == "needs a literal citation"
    assert state["cards"][first["id"]]["failure_reason"] == "needs a literal citation"
    assert state["cards"][second["id"]]["correction_reason"] is None
    assert "unknown label NOT-A-LABEL" in stdout.getvalue()
    assert json.loads((tmp_path / "review.json").read_text(encoding="utf-8"))["labels"] == state["labels"]


def test_label_summary_reports_current_provenance_valid_measurements_only():
    first = _card("first")
    second = _card("second")
    stale = _card("stale", result_fingerprint="old-result")
    state = _state()
    apply_label(
        state,
        first,
        "NOT_USEFUL",
        correction_reason="missing literal support",
        review_elapsed_s=4.0,
        reviewer_kind="owner",
    )
    # Old callers and old state remain usable during the field-name migration.
    apply_label(
        state,
        second,
        "WRONG_EVIDENCE",
        failure_reason="wrong source selected",
        review_seconds=6.0,
        reviewer_kind="owner",
    )
    apply_label(
        state,
        stale,
        "NOT_USEFUL",
        correction_reason="must not leak",
        review_elapsed_s=100.0,
        reviewer_kind="owner",
    )
    current_stale = _card("stale", result_fingerprint="new-result")

    summary = label_summary(state, cards=[first, second, current_stale])

    assert summary["n_labeled"] == 2
    assert summary["n_timed"] == 2
    assert summary["total_review_elapsed_s"] == 10.0
    assert summary["median_review_elapsed_s"] == 5.0
    assert summary["n_with_correction_reason"] == 2


def test_label_summary_empty_measurements_are_explicit_and_json_safe(tmp_path: Path):
    card = _card("batch")
    state = _state()
    batch_label(state, [card], ["BATCH:USEFUL"], state_path=tmp_path / "review.json")

    summary = label_summary(state, cards=[card])

    assert summary["n_timed"] == 0
    assert summary["total_review_elapsed_s"] is None
    assert summary["median_review_elapsed_s"] is None
    assert summary["n_with_correction_reason"] == 0
    json.dumps(summary, allow_nan=False)


def test_suggested_action_is_not_counted_as_a_correction_reason():
    card = _card("suggestion-only")
    state = _state()

    apply_label(
        state,
        card,
        "WRONG_EVIDENCE",
        suggested_correction="select doc-b",
        reviewer_kind="owner",
    )

    stored = state["cards"][card["id"]]
    assert stored["suggested_correction"] == "select doc-b"
    assert stored["correction_reason"] is None
    assert label_summary(state, cards=[card])["n_with_correction_reason"] == 0


def test_measurements_restore_only_when_provenance_matches():
    prior = _card("same", result_fingerprint="result-v1")
    state = _state()
    apply_label(
        state,
        prior,
        "NOT_USEFUL",
        correction_reason="unsupported synthesis",
        review_elapsed_s=3.25,
        reviewer_kind="owner",
    )

    restored = merge_prior_labels(
        [_card("same", result_fingerprint="result-v1")], state
    )[0]
    invalidated = merge_prior_labels(
        [_card("same", result_fingerprint="result-v2")], state
    )[0]

    assert restored["prior_label_status"] == "RESTORED"
    assert restored["correction_reason"] == "unsupported synthesis"
    assert restored["review_elapsed_s"] == 3.25
    assert invalidated["prior_label_status"] == "IGNORED_RESULT_CHANGED"
    assert invalidated["usefulness_label"] is None
    assert invalidated["correction_reason"] is None
    assert invalidated["review_elapsed_s"] is None


def test_summary_rejects_malformed_provenance_matched_review_state():
    current = _card("malformed-summary")
    state = _state()
    apply_label(
        state,
        current,
        "WRONG_EVIDENCE",
        correction_reason="must not leak",
        reviewer_kind="owner",
        review_elapsed_s=9.5,
    )
    state["cards"][current["id"]]["reviewer_kind"] = "not-a-reviewer"

    summary = label_summary(state, cards=[current])

    assert summary["n_labeled"] == 0
    assert summary["by_label"] == {}
    assert summary["n_timed"] == 0
    assert summary["total_review_elapsed_s"] is None
    assert summary["median_review_elapsed_s"] is None
    assert summary["n_with_correction_reason"] == 0


def test_provenance_mismatch_clears_all_incoming_review_metadata():
    prior = _card("prepopulated", result_fingerprint="result-v1")
    state = _state()
    apply_label(
        state,
        prior,
        "WRONG_EVIDENCE",
        failure_reason="prior reason",
        suggested_correction="prior suggestion",
        reviewer_kind="owner",
        review_elapsed_s=2.0,
    )
    incoming = _card("prepopulated", result_fingerprint="result-v2")
    incoming.update(
        {
            "usefulness_label": "NOT_USEFUL",
            "failure_reason": "incoming reason",
            "suggested_correction": "incoming suggestion",
            "correction_reason": "incoming correction reason",
            "failure_class": "NOT_USEFUL",
            "reviewer_kind": "owner",
            "review_elapsed_s": 77.0,
            "review_seconds": 77.0,
            "labeled_at": "2026-01-01T00:00:00+00:00",
        }
    )

    invalidated = merge_prior_labels([incoming], state)[0]

    assert invalidated["prior_label_status"] == "IGNORED_RESULT_CHANGED"
    for key in (
        "usefulness_label",
        "failure_reason",
        "suggested_correction",
        "correction_reason",
        "failure_class",
        "reviewer_kind",
        "review_elapsed_s",
        "review_seconds",
        "labeled_at",
    ):
        assert invalidated[key] is None


def test_legacy_measurements_restore_into_canonical_fields():
    current = _card("legacy")
    prior = dict(current)
    prior.update(
        {
            "usefulness_label": "WRONG_EVIDENCE",
            "failure_reason": "legacy wrong source",
            "suggested_correction": "select doc-b",
            "failure_class": "WRONG_EVIDENCE",
            "review_seconds": 2.5,
            "reviewer_kind": "owner",
        }
    )
    state = {
        "schema": "nano-lm.wedge_v1.review.v1",
        "cards": {current["id"]: prior},
        "labels": {current["id"]: "WRONG_EVIDENCE"},
    }

    restored = merge_prior_labels([current], state)[0]

    assert restored["correction_reason"] == "legacy wrong source"
    assert restored["review_elapsed_s"] == 2.5
    assert restored["failure_reason"] == "legacy wrong source"
    assert restored["suggested_correction"] == "select doc-b"


@pytest.mark.parametrize(
    "mutation",
    (
        "unknown_label",
        "label_mismatch",
        "unknown_reviewer",
        "unknown_failure_class",
        "missing_failure_class",
        "nonfinite_elapsed",
        "nonfinite_legacy_elapsed",
        "mismatched_elapsed_aliases",
    ),
)
def test_merge_rejects_malformed_prior_review_state(mutation: str):
    current = _card(f"invalid-{mutation}")
    state = _state()
    apply_label(
        state,
        current,
        "WRONG_EVIDENCE",
        correction_reason="wrong source",
        reviewer_kind="owner",
        review_elapsed_s=2.5,
    )
    prior = state["cards"][current["id"]]

    if mutation == "unknown_label":
        prior["usefulness_label"] = "NOT_A_LABEL"
        state["labels"][current["id"]] = "NOT_A_LABEL"
    elif mutation == "label_mismatch":
        state["labels"][current["id"]] = "USEFUL"
    elif mutation == "unknown_reviewer":
        prior["reviewer_kind"] = "automated_human"
    elif mutation == "unknown_failure_class":
        prior["failure_class"] = "MADE_UP_FAILURE"
    elif mutation == "missing_failure_class":
        prior["failure_class"] = None
    elif mutation == "nonfinite_elapsed":
        prior["review_elapsed_s"] = float("inf")
    elif mutation == "nonfinite_legacy_elapsed":
        prior["review_seconds"] = float("nan")
    else:
        prior["review_seconds"] = 3.0

    restored = merge_prior_labels([current], state)[0]

    assert restored["prior_label_status"] == "IGNORED_INVALID_REVIEW_STATE"
    assert restored["usefulness_label"] is None
    assert restored["reviewer_kind"] is None
    assert restored["failure_class"] is None
    assert restored["review_elapsed_s"] is None


@pytest.mark.parametrize("elapsed", [-0.01, float("inf"), float("nan")])
def test_apply_label_rejects_invalid_review_elapsed(elapsed: float):
    with pytest.raises(ValueError, match="review_elapsed_s must be finite and non-negative"):
        apply_label(
            _state(),
            _card("invalid-time"),
            "USEFUL",
            review_elapsed_s=elapsed,
        )
