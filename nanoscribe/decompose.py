"""Failure decomposition for PR2 eval reports — stdlib only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nanoscribe.evaluate import EvalReport, SupportRelation


@dataclass(frozen=True, slots=True)
class LayerCounts:
    transport: int = 0
    support: int = 0
    state: int = 0
    abstention: int = 0
    commission: int = 0
    malformed: int = 0


def classify_report(report: EvalReport) -> dict[str, Any]:
    """Map an EvalReport into P1 failure layers for dashboards and regression tracking."""
    state_errors = sum(
        1
        for item in report.atom_results
        if not item.omitted and not item.abstained and not item.assertion_state_correct
    )
    layers = LayerCounts(
        transport=report.invalid_span + report.wrong_source + report.wrong_mention,
        support=(
            report.support_unsupported
            + report.support_contradicted
            + (1 if report.support_review_required else 0)
        ),
        state=state_errors,
        abstention=report.omission + report.unnecessary_abstention,
        commission=report.spurious_atom,
        malformed=report.malformed + report.critical_error,
    )
    support_mix = {
        SupportRelation.DIRECT_EXACT.value: report.support_direct_exact,
        SupportRelation.NORMALIZED.value: report.support_normalized,
        SupportRelation.SEMANTICALLY_SUPPORTED.value: report.support_semantically_supported,
        SupportRelation.UNSUPPORTED.value: report.support_unsupported,
        SupportRelation.CONTRADICTED.value: report.support_contradicted,
        SupportRelation.REVIEW_REQUIRED.value: report.support_review_required,
    }
    return {
        "layers": {
            "transport": layers.transport,
            "support": layers.support,
            "state": layers.state,
            "abstention": layers.abstention,
            "commission": layers.commission,
            "malformed": layers.malformed,
        },
        "support_mix": support_mix,
        "coverage": report.coverage,
        "exact_gold_span": report.exact_gold_span,
        "span_character_f1": round(report.span_character_f1, 4),
    }
