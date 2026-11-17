"""Unit tests for the 8-pillar (132-item) deterministic evaluation & 3-leaderboard harness."""

from lab.trackb.eval_capabilities import (
    build_coding_suite,
    build_diagrams_charts_suite,
    build_epistemic_reliability_suite,
    build_long_context_suite,
    build_math_logic_suite,
    build_structured_json_suite,
    build_summarization_suite,
    build_tool_calling_suite,
    evaluate_all_pillars,
    parse_tool_calls,
)


def test_suite_counts_and_uniqueness():
    all_suites = (
        build_tool_calling_suite()
        + build_coding_suite()
        + build_math_logic_suite()
        + build_structured_json_suite()
        + build_summarization_suite()
        + build_diagrams_charts_suite()
        + build_long_context_suite()
        + build_epistemic_reliability_suite()
    )
    assert len(all_suites) == 132
    ids = [x["id"] for x in all_suites]
    assert len(ids) == len(set(ids))


def test_tool_call_parser_variants():
    calls, status = parse_tool_calls('<think>plan</think>\n```json\n{"tool_calls": [{"name": "f", "arguments": {"x": 2}}]}\n```')
    assert status == "ok" and calls == [{"name": "f", "arguments": {"x": 2}}]
    calls2, status2 = parse_tool_calls('<tool_call>{"name": "g", "arguments": {"a": true}}</tool_call>')
    assert status2 == "ok" and calls2 == [{"name": "g", "arguments": {"a": True}}]
    calls3, _ = parse_tool_calls('{"tool_calls": [], "response": "No tool needed."}')
    assert calls3 == []


def test_end_to_end_null_model_report():
    rep = evaluate_all_pillars(lambda prompts: ['{"tool_calls": [], "response": "FINAL_ANSWER: INSUFFICIENT_EVIDENCE"}'] * len(prompts),
                               model_id="null-baseline",
                               model_meta={"total_params": 2_000_000_000, "active_params": 2_000_000_000})
    assert rep["total_items"] == 132
    # Abstention (8/8) and unanswerable reliability items (8/12) pass on pure abstention; active tasks fail!
    assert rep["leaderboards"]["B_reliability"]["tool_abstention_when2call"] == 1.0
    assert rep["leaderboards"]["A_intelligence"]["coding"] == 0.0
    assert rep["leaderboards"]["C_efficiency"]["intelligence_per_B_active_params"] is not None
