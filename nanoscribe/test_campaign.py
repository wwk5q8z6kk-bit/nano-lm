from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoscribe.campaign import (
    CampaignLedger,
    SpendEntry,
    BudgetGateError,
    ABSOLUTE_MAX_USD,
    NORMAL_ENVELOPE_USD,
    THRESHOLD_NO_SPECULATIVE_USD,
    THRESHOLD_ARCH_ONLY_USD,
    THRESHOLD_KILL_MARGINAL_USD,
    THRESHOLD_REVIEW_USD,
    estimate_pod_cost,
)

def test_ledger_initialization():
    ledger = CampaignLedger()
    assert ledger.campaign_spend_actual == 0.0
    assert ledger.campaign_spend_committed == 0.0
    assert ledger.total_exposure() == 0.0
    assert ledger.posture() == "NORMAL"

def test_commit_valid_spend():
    ledger = CampaignLedger()
    entry = ledger.commit("dev", "test instance", 10.0)
    assert entry.status == "committed"
    assert entry.amount_usd == 10.0
    assert ledger.campaign_spend_committed == 10.0
    assert ledger.total_exposure() == 10.0

def test_actualize_spend():
    ledger = CampaignLedger()
    entry = ledger.commit("dev", "test instance", 10.0)
    ledger.actualize(entry, 12.0)
    assert entry.status == "actual"
    assert entry.amount_usd == 12.0
    assert ledger.campaign_spend_committed == 0.0
    assert ledger.campaign_spend_actual == 12.0
    assert ledger.total_exposure() == 12.0

def test_release_spend():
    ledger = CampaignLedger()
    entry = ledger.commit("dev", "test instance", 10.0)
    ledger.release(entry)
    assert entry.status == "released"
    assert ledger.campaign_spend_committed == 0.0
    assert ledger.total_exposure() == 0.0

def test_budget_gates():
    ledger = CampaignLedger()

    # Test absolute max
    allowed, reason = ledger.budget_gate(ABSOLUTE_MAX_USD + 1)
    assert not allowed
    assert "absolute max" in reason

    # Test normal envelope
    allowed, reason = ledger.budget_gate(NORMAL_ENVELOPE_USD + 1)
    assert not allowed
    assert "normal envelope" in reason

    # Test no speculative
    allowed, reason = ledger.budget_gate(THRESHOLD_NO_SPECULATIVE_USD + 1)
    assert not allowed
    assert "no speculative" in reason

    # Test arch only
    allowed, reason = ledger.budget_gate(THRESHOLD_ARCH_ONLY_USD + 1)
    assert not allowed
    assert "architecture-changing" in reason

    # Test kill marginal
    allowed, reason = ledger.budget_gate(THRESHOLD_KILL_MARGINAL_USD + 1)
    assert not allowed
    assert "kill marginal" in reason

def test_posture_changes():
    ledger = CampaignLedger()

    ledger.entries.append(SpendEntry(lane="dev", description="test", amount_usd=THRESHOLD_REVIEW_USD, status="actual"))
    assert ledger.posture() == "REVIEW"

    ledger = CampaignLedger()
    ledger.entries.append(SpendEntry(lane="dev", description="test", amount_usd=THRESHOLD_KILL_MARGINAL_USD, status="actual"))
    assert ledger.posture() == "KILL_MARGINAL"

    ledger = CampaignLedger()
    ledger.entries.append(SpendEntry(lane="dev", description="test", amount_usd=THRESHOLD_ARCH_ONLY_USD, status="actual"))
    assert ledger.posture() == "ARCH_ONLY"

    ledger = CampaignLedger()
    ledger.entries.append(SpendEntry(lane="dev", description="test", amount_usd=THRESHOLD_NO_SPECULATIVE_USD, status="actual"))
    assert ledger.posture() == "NO_SPECULATIVE_PAID"

def test_commit_violates_gate_raises_error():
    ledger = CampaignLedger()
    try:
        ledger.commit("dev", "too expensive", NORMAL_ENVELOPE_USD + 1)
        assert False, "Should have raised BudgetGateError"
    except BudgetGateError as e:
        assert "would exceed normal envelope" in str(e)

def test_save_load_ledger():
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger_path = Path(tmpdir) / "spend.json"

        ledger = CampaignLedger(campaign_id="test_campaign")
        ledger.commit("dev", "test run", 50.0, pod_id="pod-1", gpu="a100", rate_per_hr=2.5, notes="test notes")
        ledger.save(ledger_path)

        assert ledger_path.exists()

        loaded_ledger = CampaignLedger.load(ledger_path)
        assert loaded_ledger.campaign_id == "test_campaign"
        assert len(loaded_ledger.entries) == 1

        entry = loaded_ledger.entries[0]
        assert entry.lane == "dev"
        assert entry.description == "test run"
        assert entry.amount_usd == 50.0
        assert entry.status == "committed"
        assert entry.pod_id == "pod-1"
        assert entry.gpu == "a100"
        assert entry.rate_per_hr == 2.5
        assert entry.notes == "test notes"

        assert loaded_ledger.campaign_spend_committed == 50.0

def test_load_nonexistent_file_returns_empty_ledger():
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger_path = Path(tmpdir) / "does_not_exist.json"
        ledger = CampaignLedger.load(ledger_path)
        assert len(ledger.entries) == 0

def test_estimate_pod_cost():
    assert estimate_pod_cost(2.5, 10.0) == 25.0
    assert estimate_pod_cost(1.1234, 2.0) == 2.2468


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in fns:
        fn()
        print(f"  PASS {name}")
    print(f"campaign pins: {len(fns)}/{len(fns)} PASS")
