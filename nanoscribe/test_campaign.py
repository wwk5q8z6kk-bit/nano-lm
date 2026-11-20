from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoscribe.campaign import (
    ABSOLUTE_MAX_USD,
    NORMAL_ENVELOPE_USD,
    THRESHOLD_ARCH_ONLY_USD,
    THRESHOLD_KILL_MARGINAL_USD,
    THRESHOLD_NO_SPECULATIVE_USD,
    CampaignLedger,
    SpendEntry,
)


def test_budget_gate() -> None:
    # Initialize a ledger with some existing exposure (actual + committed)
    ledger = CampaignLedger()
    ledger.entries.append(
        SpendEntry(lane="compute", description="A", amount_usd=50.0, status="actual")
    )
    ledger.entries.append(
        SpendEntry(lane="compute", description="B", amount_usd=50.0, status="committed")
    )

    # Total exposure is now 100.0 USD
    assert ledger.total_exposure() == 100.0

    # 1. OK
    # 100 + 30 = 130 <= THRESHOLD_KILL_MARGINAL_USD (130)
    allowed, reason = ledger.budget_gate(30.0)
    assert allowed is True
    assert reason == "ok"

    # 2. Kill marginal spend
    # 100 + 31 = 131 > THRESHOLD_KILL_MARGINAL_USD (130)
    allowed, reason = ledger.budget_gate(31.0)
    assert allowed is False
    assert "kill marginal spend" in reason

    # 3. Architecture-changing only
    # 100 + 51 = 151 > THRESHOLD_ARCH_ONLY_USD (150)
    allowed, reason = ledger.budget_gate(51.0)
    assert allowed is False
    assert "architecture-changing only" in reason

    # 4. No speculative paid
    # 100 + 71 = 171 > THRESHOLD_NO_SPECULATIVE_USD (170)
    allowed, reason = ledger.budget_gate(71.0)
    assert allowed is False
    assert "no speculative paid" in reason

    # 5. Normal envelope
    # 100 + 81 = 181 > NORMAL_ENVELOPE_USD (180)
    allowed, reason = ledger.budget_gate(81.0)
    assert allowed is False
    assert "would exceed normal envelope" in reason

    # 6. Absolute max
    # 100 + 101 = 201 > ABSOLUTE_MAX_USD (200)
    allowed, reason = ledger.budget_gate(101.0)
    assert allowed is False
    assert "would exceed absolute max" in reason


if __name__ == "__main__":
    test_budget_gate()
    print("test_campaign: PASS")
