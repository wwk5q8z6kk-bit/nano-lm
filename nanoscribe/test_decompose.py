from nanoscribe.decompose import classify_report
from nanoscribe.evaluate import AtomEval, EvalReport, SupportRelation


def test_classify_report_empty():
    report = EvalReport(
        exact_gold_span=0,
        span_character_f1=0.0,
        assertion_state_correct=0,
        support_direct_exact=0,
        support_normalized=0,
        support_semantically_supported=0,
        support_unsupported=0,
        support_contradicted=0,
        support_review_required=0,
        invalid_span=0,
        wrong_source=0,
        wrong_mention=0,
        ambiguity=0,
        omission=0,
        correct_abstention=0,
        unnecessary_abstention=0,
        malformed=0,
        critical_error=0,
        spurious_atom=0,
        coverage=0.0,
        latency_s=0.0,
        memory_bytes=0,
        atom_results=(),
    )
    result = classify_report(report)
    expected = {
        "layers": {
            "transport": 0,
            "support": 0,
            "state": 0,
            "abstention": 0,
            "commission": 0,
            "malformed": 0,
        },
        "support_mix": {
            SupportRelation.DIRECT_EXACT.value: 0,
            SupportRelation.NORMALIZED.value: 0,
            SupportRelation.SEMANTICALLY_SUPPORTED.value: 0,
            SupportRelation.UNSUPPORTED.value: 0,
            SupportRelation.CONTRADICTED.value: 0,
            SupportRelation.REVIEW_REQUIRED.value: 0,
        },
        "coverage": 0.0,
        "exact_gold_span": 0,
        "span_character_f1": 0.0,
    }
    assert result == expected


def test_classify_report_populated():
    atom1 = AtomEval(
        atom_id="1", omitted=False, abstained=False, assertion_state_correct=False
    )
    atom2 = AtomEval(
        atom_id="2", omitted=True, abstained=False, assertion_state_correct=False
    )
    atom3 = AtomEval(
        atom_id="3", omitted=False, abstained=True, assertion_state_correct=False
    )
    atom4 = AtomEval(
        atom_id="4", omitted=False, abstained=False, assertion_state_correct=True
    )

    report = EvalReport(
        exact_gold_span=1,
        span_character_f1=0.85,
        assertion_state_correct=1,
        support_direct_exact=2,
        support_normalized=3,
        support_semantically_supported=4,
        support_unsupported=5,
        support_contradicted=6,
        support_review_required=7,
        invalid_span=8,
        wrong_source=9,
        wrong_mention=10,
        ambiguity=11,
        omission=12,
        correct_abstention=13,
        unnecessary_abstention=14,
        malformed=15,
        critical_error=16,
        spurious_atom=17,
        coverage=0.9,
        latency_s=1.5,
        memory_bytes=1024,
        atom_results=(atom1, atom2, atom3, atom4),
    )
    result = classify_report(report)
    expected = {
        "layers": {
            "transport": 8 + 9 + 10,  # invalid_span + wrong_source + wrong_mention
            "support": 5 + 6,  # support_unsupported + support_contradicted
            "state": 1,  # only atom1 meets: not omitted, not abstained, not assertion_state_correct
            "abstention": 12 + 14,  # omission + unnecessary_abstention
            "commission": 17,  # spurious_atom
            "malformed": 15 + 16,  # malformed + critical_error
        },
        "support_mix": {
            SupportRelation.DIRECT_EXACT.value: 2,
            SupportRelation.NORMALIZED.value: 3,
            SupportRelation.SEMANTICALLY_SUPPORTED.value: 4,
            SupportRelation.UNSUPPORTED.value: 5,
            SupportRelation.CONTRADICTED.value: 6,
            SupportRelation.REVIEW_REQUIRED.value: 7,
        },
        "coverage": 0.9,
        "exact_gold_span": 1,
        "span_character_f1": 0.85,
    }
    assert result == expected


def test_state_errors_calculation():
    # Only atom1 should be counted.
    # atom2 is omitted
    # atom3 is abstained
    # atom4 is assertion_state_correct
    # atom5 is omitted and abstained
    atoms = (
        AtomEval("1", omitted=False, abstained=False, assertion_state_correct=False),
        AtomEval("2", omitted=True, abstained=False, assertion_state_correct=False),
        AtomEval("3", omitted=False, abstained=True, assertion_state_correct=False),
        AtomEval("4", omitted=False, abstained=False, assertion_state_correct=True),
        AtomEval("5", omitted=True, abstained=True, assertion_state_correct=False),
    )
    report = EvalReport(
        exact_gold_span=0,
        span_character_f1=0.0,
        assertion_state_correct=0,
        support_direct_exact=0,
        support_normalized=0,
        support_semantically_supported=0,
        support_unsupported=0,
        support_contradicted=0,
        support_review_required=0,
        invalid_span=0,
        wrong_source=0,
        wrong_mention=0,
        ambiguity=0,
        omission=0,
        correct_abstention=0,
        unnecessary_abstention=0,
        malformed=0,
        critical_error=0,
        spurious_atom=0,
        coverage=0.0,
        latency_s=0.0,
        memory_bytes=0,
        atom_results=atoms,
    )
    result = classify_report(report)
    assert result["layers"]["state"] == 1
