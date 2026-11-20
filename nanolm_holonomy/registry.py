"""Three-version evolutionary experiment registry (BASELINE, BEST, EXPERIMENTAL).

Enforces disciplined scientific promotion: an EXPERIMENTAL candidate replaces BEST
only when it demonstrates a verified multi-seed improvement over BEST while obeying
parameter (<= 4B) and compute/stability invariants.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RegistryVersion:
    name: str
    family: str
    total_parameters: int
    mean_primary_metric: float
    seed_scores: list[float]
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


class ExperimentRegistry:
    """Tracks BASELINE, BEST, and EXPERIMENTAL model generations with audit log."""

    def __init__(self, *, higher_is_better: bool = False) -> None:
        self.higher_is_better = higher_is_better
        self.baseline: RegistryVersion | None = None
        self.best: RegistryVersion | None = None
        self.experimental: RegistryVersion | None = None
        self.audit_log: list[dict[str, Any]] = []

    def register_baseline(self, version: RegistryVersion) -> None:
        self.baseline = version
        if self.best is None:
            self.best = version
        self.audit_log.append(
            {"event": "register_baseline", "name": version.name, "metric": version.mean_primary_metric}
        )

    def _is_better(self, cand: float, incumbent: float, *, min_delta: float = 0.0) -> bool:
        if self.higher_is_better:
            return cand > incumbent + min_delta
        return cand < incumbent - min_delta

    def evaluate_candidate(
        self,
        candidate: RegistryVersion,
        *,
        min_seeds: int = 2,
        min_delta: float = 0.0,
        max_param_limit: int = 4_000_000_000,
    ) -> dict[str, Any]:
        """Set candidate as EXPERIMENTAL and promote to BEST only if strictly validated."""
        self.experimental = candidate
        if candidate.total_parameters > max_param_limit:
            decision = {
                "candidate": candidate.name,
                "promoted": False,
                "reason": f"exceeds_parameter_cap:{candidate.total_parameters}>{max_param_limit}",
            }
            self.audit_log.append(decision)
            return decision

        if len(candidate.seed_scores) < min_seeds:
            decision = {
                "candidate": candidate.name,
                "promoted": False,
                "reason": f"insufficient_seeds:{len(candidate.seed_scores)}<{min_seeds}",
            }
            self.audit_log.append(decision)
            return decision

        assert self.best is not None, "Baseline must be registered before evaluating candidates"
        if self._is_better(candidate.mean_primary_metric, self.best.mean_primary_metric, min_delta=min_delta):
            prev_best = self.best.name
            prev_val = self.best.mean_primary_metric
            self.best = candidate
            decision = {
                "candidate": candidate.name,
                "promoted": True,
                "reason": f"beat_best:{prev_best}({prev_val:.4f}->{candidate.mean_primary_metric:.4f})",
            }
        else:
            decision = {
                "candidate": candidate.name,
                "promoted": False,
                "reason": f"did_not_beat_best:{candidate.mean_primary_metric:.4f}_vs_{self.best.mean_primary_metric:.4f}",
            }
        self.audit_log.append(decision)
        return decision

    def to_dict(self) -> dict[str, Any]:
        return {
            "higher_is_better": self.higher_is_better,
            "baseline": asdict(self.baseline) if self.baseline else None,
            "best": asdict(self.best) if self.best else None,
            "experimental": asdict(self.experimental) if self.experimental else None,
            "audit_log": self.audit_log,
        }

    def save_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
