"""Local span-transport benchmark for campaign B2 — no GPU required."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nanoscribe.adapt import run_pipeline
from nanoscribe.adapters import AtomSpec, FixtureSpanPortAdapter
from nanoscribe.campaign_datasets import SMOKE_SUITE_REVISION, campaign_cases
from nanoscribe.harness import HarnessCase, HarnessResult, ModelTrack, TrackConfig, _report_aggregate, aggregate_suite_metrics, FailureTaxonomy
from nanoscribe.select import ConstrainedSelector, relocate


@dataclass(frozen=True, slots=True)
class TransportArmResult:
    arm: str
    suite: str
    n_cases: int
    n_atoms: int
    exact_gold_span_rate: float
    support_direct_exact_rate: float
    assertion_state_correct_rate: float
    transport_abstain_rate: float


def _oracle_line(case: HarnessCase, spec: AtomSpec) -> str:
    try:
        atom = case.gold.atom(spec.atom_id)
    except Exception:
        return "NOT_MENTIONED"
    if not atom.evidence_ids:
        return "NOT_MENTIONED"
    span = case.gold.span(atom.evidence_ids[0])
    label = atom.assertion_state.value.upper()
    if label == "ASSERTED":
        label = "STATED"
    return f'{label}: "{span.text}"'


def _degraded_line(case: HarnessCase, spec: AtomSpec) -> str:
    """Surface perturbations that break exact-only relocate but not snap/variants."""
    oracle = _oracle_line(case, spec)
    if oracle == "NOT_MENTIONED":
        return oracle

    def upper_quote(match: re.Match[str]) -> str:
        return f'"{match.group(1).upper()}"'

    degraded = re.sub(r'"([^"]+)"', upper_quote, oracle)
    return degraded.replace(": ", ":  ")


def _fixture_lines(case: HarnessCase, *, degraded: bool) -> dict[str, str]:
    builder = _degraded_line if degraded else _oracle_line
    return {spec.atom_id: builder(case, spec) for spec in case.atom_specs}


def _run_arm(
    cases: list[HarnessCase],
    *,
    arm: str,
    selector: ConstrainedSelector | None,
    degraded: bool,
) -> TransportArmResult:
    results: list[HarnessResult] = []
    transport_abstains = 0
    atom_total = 0
    for case in cases:
        adapter = FixtureSpanPortAdapter(
            model_id=f"fixture/{arm}",
            lines=_fixture_lines(case, degraded=degraded),
        )
        batch = adapter.propose(case.model_input, case.atom_specs)
        predicted, report = run_pipeline(
            case.model_input,
            batch,
            selector=selector,
            gold=case.gold,
        )
        assert report is not None
        for pred in predicted.atoms:
            atom_total += 1
            gold_atom = next((a for a in case.gold.atoms if a.atom_id == pred.atom_id), None)
            gold_abstained = gold_atom is None or not gold_atom.evidence_ids
            if pred.abstained and not gold_abstained:
                transport_abstains += 1
        agg = _report_aggregate(report)
        track = TrackConfig(
            track=ModelTrack.FIXTURE,
            model_id=adapter.model_id,
            adapter_factory=lambda: adapter,
            cost_class="zero",
        )
        results.append(
            HarnessResult(
                track=track.track,
                model_id=track.model_id,
                test_set=case.test_set,
                encounter_id=case.encounter_id,
                cost_class=track.cost_class,
                aggregate=agg,
                failures=FailureTaxonomy(),
                per_atom={},
                latency_s=0.0,
                memory_bytes=0,
            )
        )
    suite = aggregate_suite_metrics(results)
    abstain_rate = transport_abstains / atom_total if atom_total else 0.0
    return TransportArmResult(
        arm=arm,
        suite=SMOKE_SUITE_REVISION,
        n_cases=len(cases),
        n_atoms=atom_total,
        exact_gold_span_rate=float(suite.get("exact_gold_span_rate", 0.0)),
        support_direct_exact_rate=float(suite.get("support_direct_exact_rate", 0.0)),
        assertion_state_correct_rate=float(suite.get("assertion_state_correct_rate", 0.0)),
        transport_abstain_rate=abstain_rate,
    )


def _reference_metrics() -> dict[str, Any]:
    path = Path("artifacts/campaign/student_gap_v1.json")
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    managed = payload.get("managed_reference", {}).get("c2_screening", {})
    student = payload.get("student_a", {}).get("c2_screening", {})
    return {
        "managed_ref_c2": {
            "exact_gold_span_rate": managed.get("exact_gold_span_rate"),
            "support_direct_exact_rate": managed.get("support_direct_exact_rate"),
            "assertion_state_correct_rate": managed.get("assertion_state_correct_rate"),
        },
        "student_a_c2": {
            "exact_gold_span_rate": student.get("exact_gold_span_rate"),
            "support_direct_exact_rate": student.get("support_direct_exact_rate"),
            "assertion_state_correct_rate": student.get("assertion_state_correct_rate"),
        },
    }


def run_b2_local_benchmark(*, suite: str = "p1_contract_smoke_v1") -> dict[str, Any]:
    cases = campaign_cases(suite)
    oracle_v2 = _run_arm(
        cases,
        arm="selector_v2_oracle_fixture",
        selector=ConstrainedSelector(),
        degraded=False,
    )
    baseline_degraded = _run_arm(
        cases,
        arm="selector_v1_degraded_fixture",
        selector=_LegacyExactSelector(),
        degraded=True,
    )
    improved_degraded = _run_arm(
        cases,
        arm="selector_v2_degraded_fixture",
        selector=ConstrainedSelector(),
        degraded=True,
    )
    refs = _reference_metrics()
    interim_gate = {
        "c2_exact_gold_span_rate_min": 0.25,
        "oracle_fixture_exact_gold_span_rate": oracle_v2.exact_gold_span_rate,
        "degraded_fixture_v1_exact_gold_span_rate": baseline_degraded.exact_gold_span_rate,
        "degraded_fixture_v2_exact_gold_span_rate": improved_degraded.exact_gold_span_rate,
        "local_smoke_passes_oracle_fixture": oracle_v2.exact_gold_span_rate >= 0.99,
        "local_smoke_v2_beats_v1_on_degraded": (
            improved_degraded.exact_gold_span_rate > baseline_degraded.exact_gold_span_rate
        ),
        "note": "Degraded fixture uses case-folded quotes + extra whitespace; C2 serverless re-run still required for B2 close.",
    }
    return {
        "schema": "nano.campaign.span_transport.v2",
        "campaign_id": "accelerated_research_campaign_v2",
        "task_id": "B2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite": suite,
        "arms": {
            oracle_v2.arm: asdict(oracle_v2),
            baseline_degraded.arm: asdict(baseline_degraded),
            improved_degraded.arm: asdict(improved_degraded),
        },
        "reference_c2": refs,
        "interim_gate": interim_gate,
        "status": "local_smoke_complete_serverless_pending",
    }


class _LegacyExactSelector(ConstrainedSelector):
    """Pre-v2 behavior: exact relocate only (no snap / variant fallbacks)."""

    def select_quote(
        self,
        source,
        quote: str,
        *,
        evidence_id: str,
        raw_value: str | None = None,
    ):
        del raw_value
        return relocate(source, quote, evidence_id=evidence_id)


def write_span_transport_v2(path: Path | None = None) -> Path:
    out = path or Path("artifacts/campaign/span_transport_v2.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = run_b2_local_benchmark()
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out
