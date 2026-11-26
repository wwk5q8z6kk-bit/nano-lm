"""P1 acceleration campaign spend tracker and budget gates."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "nano.campaign.spend.v0"
DEFAULT_LEDGER = Path("artifacts/campaign/spend.json")

# Mandate v1.0 envelopes (USD).
ABSOLUTE_MAX_USD = 200.0
NORMAL_ENVELOPE_USD = 180.0
SAFETY_MARGIN_USD = 20.0

# Review / kill thresholds on campaign_spend_actual + campaign_spend_committed.
THRESHOLD_REVIEW_USD = 90.0
THRESHOLD_KILL_MARGINAL_USD = 130.0
THRESHOLD_ARCH_ONLY_USD = 150.0
THRESHOLD_NO_SPECULATIVE_USD = 170.0


@dataclass
class SpendEntry:
    lane: str
    description: str
    amount_usd: float
    status: str  # committed | actual | released
    pod_id: str | None = None
    gpu: str | None = None
    rate_per_hr: float | None = None
    started_at: str | None = None
    ended_at: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane": self.lane,
            "description": self.description,
            "amount_usd": round(self.amount_usd, 4),
            "status": self.status,
            "pod_id": self.pod_id,
            "gpu": self.gpu,
            "rate_per_hr": self.rate_per_hr,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "notes": self.notes,
        }


@dataclass
class CampaignLedger:
    campaign_id: str = "p1_acceleration_campaign_v0"
    entries: list[SpendEntry] = field(default_factory=list)
    updated_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )

    @property
    def campaign_spend_actual(self) -> float:
        return sum(
            e.amount_usd for e in self.entries if e.status == "actual"
        )

    @property
    def campaign_spend_committed(self) -> float:
        return sum(
            e.amount_usd for e in self.entries if e.status == "committed"
        )

    @property
    def campaign_spend_remaining(self) -> float:
        used = self.campaign_spend_actual + self.campaign_spend_committed
        return max(0.0, NORMAL_ENVELOPE_USD - used)

    def total_exposure(self) -> float:
        return self.campaign_spend_actual + self.campaign_spend_committed

    def budget_gate(self, proposed_usd: float) -> tuple[bool, str]:
        """Return (allowed, reason) for a new spend commitment."""
        exposure = self.total_exposure() + proposed_usd
        if exposure > ABSOLUTE_MAX_USD:
            return False, f"would exceed absolute max ${ABSOLUTE_MAX_USD:.0f}"
        if exposure > NORMAL_ENVELOPE_USD:
            return (
                False,
                f"would exceed normal envelope ${NORMAL_ENVELOPE_USD:.0f} "
                f"(safety margin reserved)",
            )
        if exposure > THRESHOLD_NO_SPECULATIVE_USD:
            return False, f"above ${THRESHOLD_NO_SPECULATIVE_USD:.0f}: no speculative paid"
        if exposure > THRESHOLD_ARCH_ONLY_USD:
            return False, f"above ${THRESHOLD_ARCH_ONLY_USD:.0f}: architecture-changing only"
        if exposure > THRESHOLD_KILL_MARGINAL_USD:
            return False, f"above ${THRESHOLD_KILL_MARGINAL_USD:.0f}: kill marginal spend"
        return True, "ok"

    def posture(self) -> str:
        exposure = self.total_exposure()
        if exposure >= THRESHOLD_NO_SPECULATIVE_USD:
            return "NO_SPECULATIVE_PAID"
        if exposure >= THRESHOLD_ARCH_ONLY_USD:
            return "ARCH_ONLY"
        if exposure >= THRESHOLD_KILL_MARGINAL_USD:
            return "KILL_MARGINAL"
        if exposure >= THRESHOLD_REVIEW_USD:
            return "REVIEW"
        return "NORMAL"

    def commit(
        self,
        lane: str,
        description: str,
        amount_usd: float,
        *,
        pod_id: str | None = None,
        gpu: str | None = None,
        rate_per_hr: float | None = None,
        notes: str = "",
    ) -> SpendEntry:
        allowed, reason = self.budget_gate(amount_usd)
        if not allowed:
            raise BudgetGateError(reason)
        entry = SpendEntry(
            lane=lane,
            description=description,
            amount_usd=amount_usd,
            status="committed",
            pod_id=pod_id,
            gpu=gpu,
            rate_per_hr=rate_per_hr,
            started_at=datetime.now(UTC).isoformat(),
            notes=notes,
        )
        self.entries.append(entry)
        self.updated_at = datetime.now(UTC).isoformat()
        return entry

    def actualize(
        self,
        entry: SpendEntry,
        amount_usd: float,
        *,
        ended_at: str | None = None,
    ) -> None:
        entry.status = "actual"
        entry.amount_usd = amount_usd
        entry.ended_at = ended_at or datetime.now(UTC).isoformat()

    def release(self, entry: SpendEntry) -> None:
        entry.status = "released"
        entry.ended_at = datetime.now(UTC).isoformat()

    def summary(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "campaign_id": self.campaign_id,
            "updated_at": self.updated_at,
            "campaign_spend_actual": round(self.campaign_spend_actual, 4),
            "campaign_spend_committed": round(self.campaign_spend_committed, 4),
            "campaign_spend_remaining": round(self.campaign_spend_remaining, 4),
            "total_exposure": round(self.total_exposure(), 4),
            "posture": self.posture(),
            "envelopes": {
                "absolute_max_usd": ABSOLUTE_MAX_USD,
                "normal_envelope_usd": NORMAL_ENVELOPE_USD,
                "safety_margin_usd": SAFETY_MARGIN_USD,
            },
            "entries": [e.to_dict() for e in self.entries],
        }

    def save(self, path: Path = DEFAULT_LEDGER) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Path = DEFAULT_LEDGER) -> CampaignLedger:
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        ledger = cls(campaign_id=data.get("campaign_id", "p1_acceleration_campaign_v0"))
        ledger.updated_at = data.get("updated_at", ledger.updated_at)
        for raw in data.get("entries", []):
            ledger.entries.append(SpendEntry(**raw))
        return ledger


class BudgetGateError(RuntimeError):
    """Raised when a spend would violate campaign budget gates."""


def estimate_pod_cost(rate_per_hr: float, hours: float) -> float:
    return round(rate_per_hr * hours, 4)
