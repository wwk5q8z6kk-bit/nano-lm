from nanoscribe.campaign import CampaignLedger, SpendEntry, BudgetGateError
import pytest
from dataclasses import replace

def test_release_spend_success():
    ledger = CampaignLedger()
    entry = SpendEntry(lane="compute", description="Test", amount_usd=100.0, status="committed")
    ledger.entries.append(entry)
    ledger.release(entry)
    assert entry.status == "released"
    assert entry.ended_at is not None

def test_release_spend_error():
    ledger = CampaignLedger()
    entry = SpendEntry(lane="compute", description="Test", amount_usd=100.0, status="actual")
    with pytest.raises(ValueError, match="Attempted to release a spend entry not in pending"):
        ledger.release(entry)

def test_release_rejects_equal_copy():
    ledger = CampaignLedger()
    owned = ledger.commit("compute", "Test", 1.0)
    with pytest.raises(ValueError, match="not in pending"):
        ledger.release(replace(owned))
    assert owned.status == "committed"


if __name__ == "__main__":
    pytest.main([__file__])
