#!/usr/bin/env python3
"""Baseline eval runner: adapter → pipeline → PR2 metrics + failure layers.

Run:
  python3 nanoscribe/run_eval.py [--fixture-only]
  python3 nanoscribe/run_eval.py --suite campaign_v1 [--fixture-only]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_repo_root = str(Path(__file__).resolve().parents[1])
_script_dir = str(Path(__file__).resolve().parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
sys.path[:] = [p for p in sys.path if p != _script_dir]

from nanoscribe.adapt import parse_label_and_quotes, run_pipeline
from nanoscribe.adapters import (
    DEFAULT_BASELINE_LINES,
    FixtureSpanPortAdapter,
    Qwen25BaselineAdapter,
    default_baseline_specs,
)
from nanoscribe.campaign_datasets import (
    CAMPAIGN_DATASET_REVISION,
    CAMPAIGN_V1_ENCOUNTERS,
    campaign_cases,
    dataset_revision_for,
    fixture_lines_for_encounter,
    suite_manifest,
)
from nanoscribe.decompose import classify_report
from nanoscribe.harness import FailureTaxonomy, HarnessCase
from nanoscribe.leakage import condition_label, leakage_config
from nanoscribe.prompt import build_span_port_prompt, span_port_system_prompt
from nanoscribe.test_adapt import _gold, _model_input


@contextmanager
def _without_qwen_weights_env():
    saved = os.environ.pop("NANOSCIBE_QWEN_WEIGHTS", None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ["NANOSCIBE_QWEN_WEIGHTS"] = saved


def _aggregate_from_report(report) -> dict[str, Any]:
    return {
        "exact_gold_span": report.exact_gold_span,
        "span_character_f1": round(report.span_character_f1, 4),
        "assertion_state_correct": report.assertion_state_correct,
        "support_direct_exact": report.support_direct_exact,
        "support_normalized": report.support_normalized,
        "support_review_required": report.support_review_required,
        "wrong_source": report.wrong_source,
        "wrong_mention": report.wrong_mention,
        "invalid_span": report.invalid_span,
        "omission": report.omission,
        "correct_abstention": report.correct_abstention,
        "unnecessary_abstention": report.unnecessary_abstention,
        "unbound_assertion": report.unbound_assertion,
        "spurious_atom": report.spurious_atom,
        "malformed": report.malformed,
        "critical_error": report.critical_error,
        "coverage": round(report.coverage, 4),
    }


def _per_atom_from_report(report) -> dict[str, dict[str, Any]]:
    return {
        item.atom_id: {
            "exact_gold_span": item.exact_gold_span,
            "span_character_f1": round(item.span_character_f1, 4),
            "support_relation": (
                item.support_relation.value if item.support_relation else None
            ),
            "assertion_state_correct": item.assertion_state_correct,
            "abstained": item.abstained,
            "malformed": item.malformed,
            "omitted": item.omitted,
        }
        for item in report.atom_results
    }


def _payload_from_report(
    *,
    report,
    batch,
    adapter_model_id: str,
    fixture_only: bool,
    experiment: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment": experiment,
        "fixture_only": fixture_only,
        "adapter": adapter_model_id,
        "aggregate": _aggregate_from_report(report),
        "layers": classify_report(report),
        "per_atom": _per_atom_from_report(report),
        "latency_s": round(batch.latency_s, 4),
        "memory_bytes": batch.memory_bytes,
    }
    if extra:
        payload.update(extra)
    return payload


def run_baseline_eval(*, fixture_only: bool = False) -> dict[str, Any]:
    """Run the deterministic baseline encounter through adapter → eval → layers."""
    specs = default_baseline_specs()
    gold = _gold()
    model_input = _model_input(gold.sources[0])

    def _run() -> dict[str, Any]:
        adapter = Qwen25BaselineAdapter(fixture_lines=DEFAULT_BASELINE_LINES)
        batch = adapter.propose(model_input, specs)
        _, report = run_pipeline(model_input, batch, gold=gold)
        assert report is not None
        return _payload_from_report(
            report=report,
            batch=batch,
            adapter_model_id=adapter.model_id,
            fixture_only=fixture_only,
            experiment="p1_baseline_eval_v0",
        )

    if fixture_only:
        with _without_qwen_weights_env():
            return _run()
    return _run()


def _adapter_for_case(
    case: HarnessCase,
    *,
    fixture_only: bool,
    raw_line_sink: dict[str, str] | None = None,
) -> FixtureSpanPortAdapter | Qwen25BaselineAdapter:
    lines = fixture_lines_for_encounter(case.encounter_id)
    if fixture_only:
        return FixtureSpanPortAdapter(
            model_id="fixture/campaign-span-port",
            lines=lines,
            raw_line_sink=raw_line_sink,
        )
    return Qwen25BaselineAdapter(fixture_lines=lines, raw_line_sink=raw_line_sink)


def _quote_absent(raw_line: str) -> bool:
    """Model emitted a label but no quoted evidence — where channel C2 fires."""
    label, quotes = parse_label_and_quotes(raw_line)
    return label is not None and label != "NOT_MENTIONED" and not quotes


def _gold_value_in_prompt(spec, user_prompt: str) -> bool:
    """Mechanical C1 measure: is the slot's gold value visible in what we sent?

    Independent of model behaviour — it reads the prompt text, so it reports
    leakage even on a run where the model happens to answer correctly anyway.
    """
    value = (spec.raw_value or "").strip().casefold()
    if not value:
        return False
    shown = (span_port_system_prompt() + "\n" + user_prompt).casefold()
    # Discount the transcript itself: the value legitimately occurs there when
    # the atom is present. Only the instruction text counts as leakage.
    instructions = shown.split("\n\n", 1)[-1]
    return value in instructions or value in span_port_system_prompt().casefold()


def _run_campaign_case(case: HarnessCase, *, fixture_only: bool) -> dict[str, Any]:
    raw_lines: dict[str, str] = {}
    prompts = {
        spec.atom_id: build_span_port_prompt(case.model_input.source, spec)
        for spec in case.atom_specs
    }
    adapter = _adapter_for_case(
        case, fixture_only=fixture_only, raw_line_sink=raw_lines
    )
    batch = adapter.propose(case.model_input, case.atom_specs)
    _, report = run_pipeline(case.model_input, batch, gold=case.gold)
    assert report is not None
    failures = FailureTaxonomy.from_report(report)
    return {
        "encounter_id": case.encounter_id,
        "test_set": case.test_set.value,
        "adapter": adapter.model_id,
        "aggregate": _aggregate_from_report(report),
        "layers": classify_report(report),
        "per_atom": _per_atom_from_report(report),
        "failure_taxonomy": failures.to_dict(),
        # Primary leakage evidence: what the model was shown, what it said back.
        "prompts": prompts,
        "gold_value_in_prompt": {
            spec.atom_id: _gold_value_in_prompt(spec, prompts[spec.atom_id])
            for spec in case.atom_specs
        },
        "raw_lines": dict(raw_lines),
        "quote_absent": {
            atom_id: _quote_absent(line) for atom_id, line in raw_lines.items()
        },
        "latency_s": round(batch.latency_s, 4),
        "memory_bytes": batch.memory_bytes,
    }


def _suite_aggregate(encounters: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "exact_gold_span": 0,
        "assertion_state_correct": 0,
        "support_direct_exact": 0,
        "malformed": 0,
        "critical_error": 0,
        "spurious_atom": 0,
        "omission": 0,
        "correct_abstention": 0,
        "unnecessary_abstention": 0,
        "unbound_assertion": 0,
        "invalid_span": 0,
        "quote_absent": 0,
        "gold_value_in_prompt": 0,
        "atoms": 0,
        "encounters": len(encounters),
    }
    f1s: list[float] = []
    coverages: list[float] = []
    layer_totals = {
        "transport": 0,
        "support": 0,
        "state": 0,
        "abstention": 0,
        "commission": 0,
        "malformed": 0,
    }
    for item in encounters:
        agg = item["aggregate"]
        totals["exact_gold_span"] += agg["exact_gold_span"]
        totals["assertion_state_correct"] += agg["assertion_state_correct"]
        totals["support_direct_exact"] += agg["support_direct_exact"]
        totals["malformed"] += agg["malformed"]
        totals["critical_error"] += agg["critical_error"]
        totals["spurious_atom"] += agg["spurious_atom"]
        totals["omission"] += agg["omission"]
        totals["correct_abstention"] += agg["correct_abstention"]
        totals["unnecessary_abstention"] += agg["unnecessary_abstention"]
        totals["unbound_assertion"] += agg["unbound_assertion"]
        totals["invalid_span"] += agg["invalid_span"]
        totals["quote_absent"] += sum(1 for v in item["quote_absent"].values() if v)
        totals["gold_value_in_prompt"] += sum(
            1 for v in item["gold_value_in_prompt"].values() if v
        )
        # Denominator = slots probed (one prompt per spec), stable across outcomes.
        totals["atoms"] += len(item["prompts"])
        f1s.append(agg["span_character_f1"])
        coverages.append(agg["coverage"])
        layers = item["layers"]["layers"]
        for key in layer_totals:
            layer_totals[key] += layers[key]
    return {
        **totals,
        "mean_span_character_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "mean_coverage": round(sum(coverages) / len(coverages), 4) if coverages else 0.0,
        "layers": layer_totals,
    }


def run_campaign_eval(suite: str, *, fixture_only: bool = False) -> dict[str, Any]:
    cases = campaign_cases(suite)

    def _run() -> dict[str, Any]:
        encounters = [
            _run_campaign_case(case, fixture_only=fixture_only) for case in cases
        ]
        v1_subset = [
            item for item in encounters if item["encounter_id"] in CAMPAIGN_V1_ENCOUNTERS
        ]
        added = [
            item
            for item in encounters
            if item["encounter_id"] not in CAMPAIGN_V1_ENCOUNTERS
        ]
        payload: dict[str, Any] = {
            "experiment": "p1_campaign_eval_v0",
            "suite": suite,
            "dataset_revision": dataset_revision_for(suite),
            "fixture_only": fixture_only,
            "leakage_config": leakage_config(),
            "condition": condition_label(),
            "manifest": suite_manifest(),
            "suite_aggregate": _suite_aggregate(encounters),
            "encounters": encounters,
        }
        if added:
            # Keep the prior claim's denominator readable next to the new cases.
            payload["campaign_v1_subset_aggregate"] = _suite_aggregate(v1_subset)
            payload["added_cases_aggregate"] = _suite_aggregate(added)
        return payload

    if fixture_only:
        with _without_qwen_weights_env():
            return _run()
    return _run()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P1 baseline eval with layer decomposition")
    parser.add_argument(
        "--fixture-only",
        action="store_true",
        help="Use span-port fixture even if NANOSCIBE_QWEN_WEIGHTS is set",
    )
    parser.add_argument(
        "--suite",
        default="default",
        help="Evaluation suite: default | tiny_fixture | p1_core | p1_adversarial | campaign_v1",
    )
    args = parser.parse_args(argv)
    if args.suite == "default":
        result = run_baseline_eval(fixture_only=args.fixture_only)
    else:
        result = run_campaign_eval(args.suite, fixture_only=args.fixture_only)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.suite != "default":
        print(_summary_block(result))
    return 0


def _agg_line(name: str, agg: dict[str, Any]) -> str:
    n = agg["atoms"]
    return (
        f"  {name:<20} slots={n:<3} "
        f"exact_span={agg['exact_gold_span']}/{n} "
        f"state_ok={agg['assertion_state_correct']}/{n} "
        f"coverage={agg['mean_coverage']:.3f} "
        f"correct_abstain={agg['correct_abstention']} "
        f"unbound_assert={agg['unbound_assertion']} "
        f"spurious={agg['spurious_atom']} "
        f"critical={agg['critical_error']} "
        f"malformed={agg['malformed']} "
        f"no_quote={agg['quote_absent']} "
        f"leaked_prompts={agg['gold_value_in_prompt']}/{n}"
    )


def _summary_block(result: dict[str, Any]) -> str:
    """Compact, greppable tail block — the run log is the evidence channel."""
    cfg = result.get("leakage_config", {})
    lines = [
        "",
        "=" * 78,
        "P1 SPAN-PORT LEAKAGE ABLATION — SUMMARY",
        "=" * 78,
        f"  suite              {result['suite']}",
        f"  dataset_revision   {result['dataset_revision']}",
        f"  condition          {result.get('condition', 'n/a')}",
        f"  adapter            {result['encounters'][0]['adapter']}",
        f"  fixture_only       {result['fixture_only']}",
        f"  C1 prompt_includes_gold_value   {cfg.get('prompt_includes_gold_value')}",
        f"  C2 parser_raw_value_fallback    {cfg.get('parser_raw_value_fallback')}",
        "-" * 78,
        _agg_line("ALL", result["suite_aggregate"]),
    ]
    if "campaign_v1_subset_aggregate" in result:
        lines.append(_agg_line("campaign_v1 subset", result["campaign_v1_subset_aggregate"]))
        lines.append(_agg_line("added cases", result["added_cases_aggregate"]))
    lines.append("-" * 78)
    for item in result["encounters"]:
        for atom_id, line in sorted(item["raw_lines"].items()):
            flat = " ".join(line.split())
            lines.append(f"  {item['encounter_id']}/{atom_id:<20} -> {flat}")
    lines.append("=" * 78)
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
