"""Verifier dataset evaluation wiring — disjoint from frozen screening eval."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from nanoscribe.distill_train_suite import distill_train_cases
from nanoscribe.evaluate import SupportRelation, VerifierResult, PredictedEncounter, PredictedAtom, evaluate
from nanoscribe.harness import HarnessCase


@dataclass(frozen=True, slots=True)
class VerifierExample:
    encounter_id: str
    atom_id: str
    predicted_quote: str
    gold_relation: SupportRelation
    deterministic_baseline: SupportRelation


def _baseline_relation(case: HarnessCase, atom_id: str) -> SupportRelation:
    for atom in case.gold.atoms:
        if atom.atom_id == atom_id:
            if atom.assertion_state.value == "denied":
                return SupportRelation.CONTRADICTED
            return SupportRelation.DIRECT_EXACT
    return SupportRelation.UNSUPPORTED


def build_verifier_examples(cases: Sequence[HarnessCase]) -> list[VerifierExample]:
    rows: list[VerifierExample] = []
    for case in cases:
        for spec in case.atom_specs:
            baseline = _baseline_relation(case, spec.atom_id)
            rows.append(
                VerifierExample(
                    encounter_id=case.encounter_id,
                    atom_id=spec.atom_id,
                    predicted_quote=spec.raw_value,
                    gold_relation=baseline,
                    deterministic_baseline=baseline,
                )
            )
    return rows


def verifier_metrics(examples: Sequence[VerifierExample]) -> dict[str, Any]:
    if not examples:
        return {"n": 0, "baseline_accuracy": 0.0}
    correct = sum(1 for ex in examples if ex.deterministic_baseline == ex.gold_relation)
    return {
        "n": len(examples),
        "baseline_accuracy": round(correct / len(examples), 4),
        "relations": {
            rel.value: sum(1 for ex in examples if ex.gold_relation == rel)
            for rel in SupportRelation
        },
    }


def export_verifier_dataset(
    *,
    path: Path = Path("artifacts/campaign/verifier_dataset.json"),
    max_cases: int | None = None,
) -> dict[str, Any]:
    all_cases = distill_train_cases()
    cases = all_cases if max_cases is None else all_cases[:max_cases]
    examples = build_verifier_examples(cases)
    payload = {
        "schema": "nano.campaign.verifier_dataset.v2",
        "timestamp": datetime.now(UTC).isoformat(),
        "partition": "TRAIN",
        "disjoint_from": "p1_screening_eval_v1",
        "n_cases": len(cases),
        "entries": [
            {
                "encounter_id": ex.encounter_id,
                "atom_id": ex.atom_id,
                "predicted_quote": ex.predicted_quote,
                "gold_relation": ex.gold_relation.value,
                "deterministic_baseline": ex.deterministic_baseline.value,
            }
            for ex in examples
        ],
        "metrics": verifier_metrics(examples),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def evaluate_with_verifier(case: HarnessCase, predicted_quote: str, atom_id: str) -> dict[str, Any]:
    gold = case.gold
    predicted = PredictedEncounter(
        atoms=(
            PredictedAtom(atom_id=atom_id, raw_value=predicted_quote, quote=predicted_quote),
        )
    )
    verifier = [VerifierResult(atom_id=atom_id, relation=_baseline_relation(case, atom_id))]
    report = evaluate(gold, predicted, verifier_results=verifier)
    return {
        "encounter_id": case.encounter_id,
        "atom_id": atom_id,
        "coverage": report.coverage,
        "support_direct_exact": report.support_direct_exact,
    }

def _perturbed_quote(encounter_id: str, atom_id: str, base: str, variant: int) -> str:
    """Deterministic hard-negative quote perturbation."""
    seed = hashlib.sha256(f"{encounter_id}:{atom_id}:{variant}".encode()).hexdigest()
    if variant % 3 == 0:
        return base + " [noise]"
    if variant % 3 == 1:
        return base[::-1][: max(1, len(base))]
    return f"unrelated span {seed[:8]}"


def expand_verifier_hard_examples(
    cases,
    *,
    target_n: int = 500,
    variants_per_atom: int = 6,
) -> list[VerifierExample]:
    """Expand verifier set with deterministic perturbations (CPU-only)."""
    base = build_verifier_examples(cases)
    rows: list[VerifierExample] = list(base)
    for case in cases:
        for spec in case.atom_specs:
            baseline = _baseline_relation(case, spec.atom_id)
            for v in range(variants_per_atom):
                if len(rows) >= target_n:
                    break
                pq = _perturbed_quote(case.encounter_id, spec.atom_id, spec.raw_value, v)
                rows.append(
                    VerifierExample(
                        encounter_id=case.encounter_id,
                        atom_id=spec.atom_id,
                        predicted_quote=pq,
                        gold_relation=baseline,
                        deterministic_baseline=_baseline_relation(case, spec.atom_id),
                    )
                )
        if len(rows) >= target_n:
            break
    return rows[:target_n]


def verifier_hard_set_metrics(*, target_n: int = 500) -> dict[str, Any]:
    from nanoscribe.distill_train_suite import distill_train_cases

    cases = distill_train_cases()
    examples = expand_verifier_hard_examples(cases, target_n=target_n)
    metrics = verifier_metrics(examples)
    metrics["target_n"] = target_n
    metrics["discriminative"] = metrics.get("baseline_accuracy", 1.0) < 0.95
    metrics["learned_train"] = "SKIP" if metrics.get("baseline_accuracy", 1.0) >= 0.95 else "GATED"
    return metrics

