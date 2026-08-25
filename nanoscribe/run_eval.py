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

from nanoscribe.venv_boot import ensure_venv, interpreter_provenance

ensure_venv(Path(_repo_root))

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
from nanoscribe.campaign_instances import INSTANCE_IDS, split_encounter_id
from nanoscribe.decompose import classify_report
from nanoscribe.harness import FailureTaxonomy, HarnessCase
from nanoscribe.leakage import condition_label, leakage_config
from nanoscribe.prompt import (
    answer_hint_for_spec,
    build_span_port_prompt,
    span_port_system_prompt,
    topic_for_spec,
)
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


def _gold_in_answer_template(spec) -> bool:
    """Channel C1, measured on the instruction text — not on model behaviour.

    True when the slot's gold value is visible in the answer template or in the
    system prompt's format examples, i.e. when the model is told the exact
    string to emit.
    """
    value = (spec.raw_value or "").strip().casefold()
    if not value:
        return False
    shown = (answer_hint_for_spec(spec) + "\n" + span_port_system_prompt()).casefold()
    return value in shown


def _gold_in_question(spec) -> bool:
    """Channel Q — the value names the concept the question asks about."""
    value = (spec.raw_value or "").strip().casefold()
    if not value:
        return False
    return value in topic_for_spec(spec).casefold()


JOINT_CELLS = (
    "asserted_grounded",
    "asserted_unbound",
    "asserted_bound_wrong",
    "abstained_correct",
    "abstained_incorrect",
)


def _joint_table(case, predicted, report, raw_lines: dict[str, str]) -> dict[str, Any]:
    """Model action x evidence status, partitioned so the cells sum to n_presented.

    The primary endpoint is this discrimination, not a single accuracy scalar:
    a prompt-parroting model and a transcript-reading model can score the same
    exact_gold_span while landing in completely different cells here.

    Derived from the model's own line (did it decline?) plus the bound
    prediction (did its quote reach the source?), so an unbound assertion is
    never silently folded into abstention.
    """
    cells = dict.fromkeys(JOINT_CELLS, 0)
    for record in _per_slot(case, predicted, report, raw_lines).values():
        cells[record["cell"]] += 1

    n_presented = len(case.atom_specs)
    assert sum(cells.values()) == n_presented, (cells, n_presented)
    asserted = (
        cells["asserted_grounded"]
        + cells["asserted_unbound"]
        + cells["asserted_bound_wrong"]
    )
    return {
        **cells,
        "n_presented": n_presented,
        "asserted": asserted,
        # Reported alongside the table, never instead of it: there is no
        # per-slot confidence score to threshold on, so coverage cannot be
        # equalised across cells and this risk is coverage-confounded.
        "observed_coverage": round(asserted / n_presented, 4) if n_presented else 0.0,
        "selective_risk": (
            round(
                (cells["asserted_unbound"] + cells["asserted_bound_wrong"]) / asserted, 4
            )
            if asserted
            else None
        ),
    }


def _per_slot(case, predicted, report, raw_lines: dict[str, str]) -> dict[str, Any]:
    """One record per PROBED SLOT, not per scored atom.

    report.atom_results omits slots that were correctly abstained on (no gold
    atom, nothing to score), so keying off it silently drops those slots and
    breaks the within-item pairing that paired tests need. Iterating the specs
    keeps every slot present in every cell, which is what makes cells
    comparable slot-by-slot.
    """
    gold_atom_ids = {atom.atom_id for atom in case.gold.atoms}
    pred_by_id = {atom.atom_id: atom for atom in predicted.atoms}
    eval_by_id = {item.atom_id: item for item in report.atom_results}
    out: dict[str, dict[str, Any]] = {}
    for spec in case.atom_specs:
        atom = pred_by_id.get(spec.atom_id)
        item = eval_by_id.get(spec.atom_id)
        out[spec.atom_id] = {
            "cell": _slot_cell(spec, atom, item, gold_atom_ids, raw_lines),
            "gold_present": spec.atom_id in gold_atom_ids,
            "exact_gold_span": bool(item and item.exact_gold_span),
            "span_character_f1": round(item.span_character_f1, 4) if item else 0.0,
            "assertion_state_correct": bool(item and item.assertion_state_correct),
            "abstained": bool(atom and atom.abstained),
            "unbound_assertion": bool(atom and atom.unbound_assertion),
            "malformed": bool(item and item.malformed),
            "raw_line": raw_lines.get(spec.atom_id, ""),
        }
    return out


def _slot_cell(spec, atom, item, gold_atom_ids, raw_lines: dict[str, str]) -> str:
    label, _quotes = parse_label_and_quotes(raw_lines.get(spec.atom_id, ""))
    gold_present = spec.atom_id in gold_atom_ids
    declined = label == "NOT_MENTIONED" or (
        atom is not None and atom.abstained and not atom.unbound_assertion
    )
    if declined:
        return "abstained_correct" if not gold_present else "abstained_incorrect"
    if atom is not None and atom.unbound_assertion:
        return "asserted_unbound"
    grounded = (
        gold_present
        and item is not None
        and item.exact_gold_span
        and item.assertion_state_correct
    )
    return "asserted_grounded" if grounded else "asserted_bound_wrong"


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
    predicted, report = run_pipeline(case.model_input, batch, gold=case.gold)
    assert report is not None
    failures = FailureTaxonomy.from_report(report)
    base_id, instance_id = split_encounter_id(case.encounter_id)
    return {
        "encounter_id": case.encounter_id,
        "base_encounter_id": base_id,
        "instance_id": instance_id,
        "test_set": case.test_set.value,
        "adapter": adapter.model_id,
        "aggregate": _aggregate_from_report(report),
        "layers": classify_report(report),
        "per_atom": _per_slot(case, predicted, report, raw_lines),
        "failure_taxonomy": failures.to_dict(),
        "joint_table": _joint_table(case, predicted, report, raw_lines),
        # Primary leakage evidence: what the model was shown, what it said back.
        "prompts": prompts,
        "gold_value_in_answer_template": {
            spec.atom_id: _gold_in_answer_template(spec) for spec in case.atom_specs
        },
        "gold_value_in_question": {
            spec.atom_id: _gold_in_question(spec) for spec in case.atom_specs
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
        "asserted_grounded": 0,
        "asserted_unbound": 0,
        "asserted_bound_wrong": 0,
        "abstained_correct": 0,
        "abstained_incorrect": 0,
        "asserted": 0,
        "n_presented": 0,
        "gold_in_answer_template": 0,
        "gold_in_question": 0,
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
        for cell in ("asserted_grounded", "asserted_unbound", "asserted_bound_wrong",
                     "abstained_correct", "abstained_incorrect", "asserted",
                     "n_presented"):
            totals[cell] += item["joint_table"][cell]
        totals["gold_in_answer_template"] += sum(
            1 for v in item["gold_value_in_answer_template"].values() if v
        )
        totals["gold_in_question"] += sum(
            1 for v in item["gold_value_in_question"].values() if v
        )
        # Denominator = slots probed (one prompt per spec), stable across outcomes.
        totals["atoms"] += len(item["prompts"])
        f1s.append(agg["span_character_f1"])
        coverages.append(agg["coverage"])
        layers = item["layers"]["layers"]
        for key in layer_totals:
            layer_totals[key] += layers[key]
    asserted = totals["asserted"]
    return {
        **totals,
        "observed_coverage": (
            round(asserted / totals["n_presented"], 4) if totals["n_presented"] else 0.0
        ),
        "selective_risk": (
            round(
                (totals["asserted_unbound"] + totals["asserted_bound_wrong"]) / asserted,
                4,
            )
            if asserted
            else None
        ),
        "mean_span_character_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "mean_coverage": round(sum(coverages) / len(coverages), 4) if coverages else 0.0,
        "layers": layer_totals,
    }


ACROSS_INSTANCE_METRICS = (
    "asserted_grounded",
    "asserted_unbound",
    "asserted_bound_wrong",
    "abstained_correct",
    "abstained_incorrect",
    "observed_coverage",
    "exact_gold_span",
    "assertion_state_correct",
    "quote_absent",
    "mean_span_character_f1",
)


def _mean_sd(values: list[float]) -> dict[str, Any]:
    """Sample mean, SD and SEM across instance draws (ddof=1).

    ddof=1 because the instances are a sample of the surface-value population
    the instrument draws from, not the population itself. `values` stays in the
    payload deliberately: the interaction contrast is estimated per instance
    and averaged over paired estimates, and that pairing cannot be recovered
    from summary statistics alone.
    """
    n = len(values)
    mean = sum(values) / n if n else 0.0
    if n < 2:
        return {"mean": round(mean, 4), "sd": None, "sem": None, "n": n, "values": values}
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    sd = var ** 0.5
    return {
        "mean": round(mean, 4),
        "sd": round(sd, 4),
        "sem": round(sd / (n ** 0.5), 4),
        "n": n,
        "values": values,
    }


def _across_instances(per_instance: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ordered = [per_instance[iid] for iid in INSTANCE_IDS if iid in per_instance]
    return {
        metric: _mean_sd([float(agg[metric]) for agg in ordered])
        for metric in ACROSS_INSTANCE_METRICS
    }


def _weights_provenance(fixture_only: bool) -> dict[str, Any]:
    """What the run actually loaded — pinned revision included, or why not.

    A run whose weights are not pinned in its own artifact is not
    re-measurable, which is the defect the ledger audit found in the claims
    this round exists to re-measure.
    """
    from nanoscribe.qwen_inference import resolve_weights_path, revision_for

    if fixture_only:
        return {"mode": "fixture", "weights_path": None, "revision": None}
    resolved = resolve_weights_path(None)
    if resolved is None:
        return {"mode": "fixture_fallback", "weights_path": None, "revision": None}
    return {
        "mode": "weights",
        "weights_path": resolved,
        "revision": revision_for(resolved),
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
        by_instance: dict[str, list[dict[str, Any]]] = {}
        for item in encounters:
            by_instance.setdefault(item["instance_id"], []).append(item)
        per_instance = {
            iid: _suite_aggregate(items) for iid, items in by_instance.items()
        }
        per_atom = {
            f"{item['instance_id']}/{atom_id}": data
            for item in encounters
            for atom_id, data in item["per_atom"].items()
        }
        payload: dict[str, Any] = {
            "experiment": "p1_campaign_eval_v0",
            "suite": suite,
            "weights": _weights_provenance(fixture_only),
            "interpreter": interpreter_provenance(Path(_repo_root)),
            "dataset_revision": dataset_revision_for(suite),
            "fixture_only": fixture_only,
            "leakage_config": leakage_config(),
            "condition": condition_label(),
            "manifest": suite_manifest(),
            "suite_aggregate": _suite_aggregate(encounters),
            "pooled_aggregate_warning": (
                "suite_aggregate pools all instances; it inflates apparent n and "
                "destroys the across-instance SD. Report per_instance / "
                "across_instance instead."
            ),
            "encounters": encounters,
        }
        if len(per_instance) > 1:
            payload["per_instance_aggregate"] = per_instance
            payload["across_instance"] = _across_instances(per_instance)
            payload["per_atom"] = per_atom
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
        f"gold_in_template={agg['gold_in_answer_template']}/{n} "
        f"gold_in_question={agg['gold_in_question']}/{n}"
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
        f"  C1 answer_template_gold_value   {cfg.get('prompt_answer_template_gold_value')}",
        f"  C2 parser_raw_value_fallback    {cfg.get('parser_raw_value_fallback')}",
        f"  Q  question_names_concept       {cfg.get('prompt_question_names_concept')}",
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
