"""Architecture registry / trace / adversarial pins."""
from __future__ import annotations

from pathlib import Path

from wedge_v1.arch.failure_codes import FailureCode
from wedge_v1.arch.registry import COMPONENTS, LAYERS, registry_snapshot
from wedge_v1.eval.adversarial import ADV, run_adversarial_suite
from wedge_v1.runtime import ask


def test_registry_has_core_layers_and_components():
    snap = registry_snapshot()
    assert "L4" in snap["layers"] and "L16" in snap["layers"]
    assert "retrieve.bm25" in snap["components"]
    assert "obs.ask_trace" in snap["components"]
    assert len(LAYERS) >= 18
    assert "abstain.policy" in COMPONENTS


def test_ask_emits_trace_and_failure_codes():
    r = ask("What is the capital of Mars colonies in year 3100?", corpus_dir=ADV)
    assert r["answer_status"] == "ABSTAIN"
    assert "trace" in r
    assert r["trace"]["schema"] == "nano-lm.wedge_v1.ask_trace.v1"
    assert isinstance(r.get("failure_codes"), list)
    assert r["trace"]["layers_touched"]


def test_composition_gate_abstains():
    r = ask("What is the cache TTL and the metformin dose?", corpus_dir=ADV)
    assert r["answer_status"] == "ABSTAIN"
    assert FailureCode.UNSUPPORTED_COMPOSITION.value in (r.get("failure_codes") or [])


def test_ttl_paraphrase_not_fixture_tied():
    r = ask("How long before cached entries expire?", corpus_dir=ADV)
    assert r["answer_status"] == "SUPPORTED"
    blob = str(r).lower()
    assert "300" in blob
    assert "3000" not in blob or "300 seconds" in blob


def test_adversarial_suite_all_pass():
    out = run_adversarial_suite()
    assert out["n_ok"] == out["n_cases"]
    assert out["accuracy"] == 1.0
