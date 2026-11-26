"""Three-track model lineage registry (BASELINE / BEST / EXPERIMENTAL) with non-regression gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class InnovationScore:
    """Five-axis evaluation rubric (0..10 per axis)."""

    originality: float
    mathematical_coherence: float
    experimental_feasibility: float
    computational_efficiency: float
    capability_gain: float

    @property
    def mean_score(self) -> float:
        return (
            self.originality
            + self.mathematical_coherence
            + self.experimental_feasibility
            + self.computational_efficiency
            + self.capability_gain
        ) / 5.0

    def to_dict(self) -> dict[str, float]:
        d = asdict(self)
        d["mean_score"] = self.mean_score
        return d


class LineageRegistry:
    """Immutable-baseline 3-track evolution registry (`baseline`, `best`, `experimental`)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.data: dict[str, Any] = {
            "schema": "nanolm.discovery.lineage.v1",
            "baseline": None,
            "best": None,
            "experimental": None,
            "promoted_history": [],
            "rejected": [],
        }
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def register_baseline(self, record: dict[str, Any]) -> None:
        if self.data["baseline"] is not None:
            raise RuntimeError("BASELINE is immutable once registered and cannot be overwritten")
        self.data["baseline"] = dict(record)
        if self.data["best"] is None:
            self.data["best"] = dict(record)
        self.save()

    def submit_experimental(
        self,
        record: dict[str, Any],
        *,
        promote_min_bpb_gain: float = 0.002,
        tie_bpb_tolerance: float = 0.015,
        min_ood_gain: float = 0.05,
    ) -> dict[str, Any]:
        """Record an EXPERIMENTAL model and promote to BEST iff strict evidence gates hold."""
        self.data["experimental"] = dict(record)
        best = self.data["best"]
        if best is None:
            self.data["best"] = dict(record)
            verdict = {"decision": "PROMOTED_INITIAL_BEST", "reason": "No prior BEST existed"}
        else:
            cand_bpb = float(record["composite_bpb"])
            best_bpb = float(best["composite_bpb"])
            bpb_delta = best_bpb - cand_bpb  # positive = candidate is lower (better) BPB
            cand_ood = float(record.get("ood_composite_acc", 0.0))
            best_ood = float(best.get("ood_composite_acc", 0.0))
            ood_delta = cand_ood - best_ood

            if bpb_delta >= promote_min_bpb_gain and ood_delta >= -0.02:
                self.data["promoted_history"].append(best)
                self.data["best"] = dict(record)
                verdict = {
                    "decision": "PROMOTED_TO_BEST",
                    "reason": f"Improved real BPB by {bpb_delta:+.4f} without OOD regression (ood_delta={ood_delta:+.4f})",
                }
            elif bpb_delta >= -tie_bpb_tolerance and ood_delta >= min_ood_gain:
                self.data["promoted_history"].append(best)
                self.data["best"] = dict(record)
                verdict = {
                    "decision": "PROMOTED_TO_BEST",
                    "reason": f"Matched real BPB within {tie_bpb_tolerance:.3f} ({bpb_delta:+.4f}) with major OOD/memory gain ({ood_delta:+.4f})",
                }
            else:
                rejected_entry = dict(record)
                rejected_entry["rejection_reason"] = (
                    f"Failed promotion gate: bpb_delta={bpb_delta:+.4f}, ood_delta={ood_delta:+.4f}"
                )
                self.data["rejected"].append(rejected_entry)
                verdict = {"decision": "REJECTED", "reason": rejected_entry["rejection_reason"]}

        self.save()
        return verdict

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")
