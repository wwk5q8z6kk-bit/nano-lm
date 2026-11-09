from __future__ import annotations

from typing import Any

from nanoscribe.decompose import classify_report
from nanoscribe.evaluate import AtomEval, EvalReport, SupportRelation

def _make_report(**kwargs: Any) -> EvalReport:
    # Set up defaults for all the required fields of EvalReport
    defaults: dict[str, Any] = {
        "exact_gold_span": 0,
        "span_character_f1": 0.0,
        "assertion_state_correct": 0,
        "support_direct_exact": 0,
        "support_normalized": 0,
        "support_semantically_supported": 0,
        "support_unsupported": 0,
        "support_contradicted": 0,
        "support_review_required": 0,
        "invalid_span": 0,
        "wrong_source": 0,
        "wrong_mention": 0,
        "ambiguity": 0,
        "omission": 0,
        "correct_abstention": 0,
        "unnecessary_abstention": 0,
        "malformed": 0,
        "critical_error": 0,
        "spurious_atom": 0,
        "coverage": 0.0,
        "latency_s": 0.0,
        "memory_bytes": 0,
        "atom_results": (),
    }
    defaults.update(kwargs)
    return EvalReport(**defaults)

def _make_atom(**kwargs: Any) -> AtomEval:
    # Set up defaults for AtomEval
    defaults: dict[str, Any] = {
        "atom_id": "test-atom",
        "exact_gold_span": False,
        "span_character_f1": 0.0,
        "support_relation": None,
        "assertion_state_correct": False,
        "invalid_span": False,
        "wrong_source": False,
        "wrong_mention": False,
        "malformed": False,
        "critical_error": False,
        "omitted": False,
        "abstained": False,
        "spurious_atom": False,
    }
    defaults.update(kwargs)
    return AtomEval(**defaults)


def test_classify_report_all_zeros() -> None:
    report = _make_report()
    result = classify_report(report)

    assert result["layers"]["transport"] == 0
    assert result["layers"]["support"] == 0
    assert result["layers"]["state"] == 0
    assert result["layers"]["abstention"] == 0
    assert result["layers"]["commission"] == 0
    assert result["layers"]["malformed"] == 0

    for _, count in result["support_mix"].items():
        assert count == 0

    assert result["coverage"] == 0.0
    assert result["exact_gold_span"] == 0
    assert result["span_character_f1"] == 0.0


def test_classify_report_layer_sums() -> None:
    report = _make_report(
        invalid_span=1,
        wrong_source=2,
        wrong_mention=3,
        support_unsupported=4,
        support_contradicted=5,
        omission=6,
        unnecessary_abstention=7,
        spurious_atom=8,
        malformed=9,
        critical_error=10,
        exact_gold_span=11,
        span_character_f1=12.34567,
        coverage=0.99,
        support_direct_exact=101,
        support_normalized=102,
        support_semantically_supported=103,
        support_review_required=104,
    )
    result = classify_report(report)

    assert result["layers"]["transport"] == 1 + 2 + 3
    assert result["layers"]["support"] == 4 + 5
    assert result["layers"]["abstention"] == 6 + 7
    assert result["layers"]["commission"] == 8
    assert result["layers"]["malformed"] == 9 + 10

    # State errors requires atom_results, which is empty here, so 0
    assert result["layers"]["state"] == 0

    assert result["support_mix"][SupportRelation.DIRECT_EXACT.value] == 101
    assert result["support_mix"][SupportRelation.NORMALIZED.value] == 102
    assert result["support_mix"][SupportRelation.SEMANTICALLY_SUPPORTED.value] == 103
    assert result["support_mix"][SupportRelation.UNSUPPORTED.value] == 4
    assert result["support_mix"][SupportRelation.CONTRADICTED.value] == 5
    assert result["support_mix"][SupportRelation.REVIEW_REQUIRED.value] == 104

    assert result["exact_gold_span"] == 11
    assert result["coverage"] == 0.99
    assert result["span_character_f1"] == 12.3457  # round(12.34567, 4)


def test_classify_report_state_errors() -> None:
    atom_correct = _make_atom(assertion_state_correct=True)
    atom_omitted = _make_atom(assertion_state_correct=False, omitted=True)
    atom_abstained = _make_atom(assertion_state_correct=False, abstained=True)
    atom_state_error_1 = _make_atom(assertion_state_correct=False, omitted=False, abstained=False)
    atom_state_error_2 = _make_atom(assertion_state_correct=False, omitted=False, abstained=False)

    report = _make_report(
        atom_results=(atom_correct, atom_omitted, atom_abstained, atom_state_error_1, atom_state_error_2)
    )

    result = classify_report(report)

    # Only atom_state_error_1 and atom_state_error_2 should be counted as state errors
    assert result["layers"]["state"] == 2


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in fns:
        fn()
        print(f"  PASS {name}")
    print(f"decompose pins: {len(fns)}/{len(fns)} PASS")
