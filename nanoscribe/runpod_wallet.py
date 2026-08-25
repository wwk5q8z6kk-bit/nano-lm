"""RunPod live wallet query — physical ceiling for campaign spend."""

from __future__ import annotations

import json
import subprocess
from typing import Any

from pathlib import Path

from nanoscribe.campaign import CampaignLedger, DEFAULT_LEDGER, NORMAL_ENVELOPE_USD

WALLET_FLOOR_USD = 10.0


def query_live_balance() -> dict[str, Any]:
    """Query RunPod account via runpodctl user -o json."""
    proc = subprocess.run(
        ["runpodctl", "user", "-o", "json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return {
            "ok": False,
            "error": proc.stderr.strip() or proc.stdout.strip() or "runpodctl user failed",
        }
    data = json.loads(proc.stdout)
    balance = float(data.get("clientBalance", 0.0))
    spend_per_hr = float(data.get("currentSpendPerHr", 0.0))
    return {
        "ok": True,
        "client_balance_usd": round(balance, 4),
        "current_spend_per_hr": round(spend_per_hr, 4),
        "user_id": data.get("id"),
        "email": data.get("email"),
    }


def effective_campaign_budget(
    *,
    ledger_path: Path = DEFAULT_LEDGER,
    authorized_envelope_usd: float = NORMAL_ENVELOPE_USD,
) -> dict[str, Any]:
    """campaign_remaining = min(authorized_remaining, live_balance - WALLET_FLOOR_USD)."""
    wallet = query_live_balance()
    ledger = CampaignLedger.load(ledger_path)
    authorized_remaining = ledger.campaign_spend_remaining
    live_balance = wallet.get("client_balance_usd", 0.0) if wallet.get("ok") else None
    if live_balance is not None:
        campaign_remaining = min(authorized_remaining, max(0.0, live_balance - WALLET_FLOOR_USD))
    else:
        campaign_remaining = authorized_remaining
    return {
        "wallet": wallet,
        "authorized_envelope_usd": authorized_envelope_usd,
        "authorized_remaining": round(authorized_remaining, 4),
        "live_runpod_balance": live_balance,
        "wallet_floor_usd": WALLET_FLOOR_USD,
        "campaign_remaining": round(campaign_remaining, 4),
        "ledger_actual": round(ledger.campaign_spend_actual, 4),
        "ledger_committed": round(ledger.campaign_spend_committed, 4),
        "posture": ledger.posture(),
    }


def budget_gate_with_wallet(
    proposed_usd: float,
    *,
    ledger_path: Path = DEFAULT_LEDGER,
) -> tuple[bool, str, dict[str, Any]]:
    """Ledger gate AND live wallet ceiling."""
    budget = effective_campaign_budget(ledger_path=ledger_path)
    ledger = CampaignLedger.load(ledger_path)
    allowed, reason = ledger.budget_gate(proposed_usd)
    if not allowed:
        return False, reason, budget
    remaining = budget["campaign_remaining"]
    if proposed_usd > remaining:
        return (
            False,
            f"would exceed campaign_remaining ${remaining:.2f} "
            f"(live balance ${budget.get('live_runpod_balance', '?')})",
            budget,
        )
    return True, "ok", budget
