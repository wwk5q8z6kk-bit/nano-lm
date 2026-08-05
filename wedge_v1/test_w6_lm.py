"""W6 — gated LM admission + marginal stub probe."""
from __future__ import annotations

from wedge_v1.classical.eclass_probes import lm_still_needed, probe_t35, probe_t36, probe_t39
from wedge_v1.classical.solvers import Claim
from wedge_v1.lm.admission import evaluate_admission
from wedge_v1.lm.marginal import run_marginal_probe
from wedge_v1.lm.probe import ALLOWLIST_TASKS, StubLMBackend, ablation_fails_support, probe_eclass_task
from wedge_v1.runtime import DEFAULT_CORPUS, load_corpus


def test_admission_not_indicated_on_clean_eclass():
    docs = load_corpus(DEFAULT_CORPUS)
    still = lm_still_needed([probe_t35(docs), probe_t36(docs), probe_t39(docs)])
    assert still is False
    out = evaluate_admission({"fine_counts": {"over_abstention": 0}}, eclass_lm_still_needed=still)
    assert out["verdict"] == "LM_PROBE_NOT_INDICATED"
    assert out["lm_probe_indicated"] is False
    assert any("E-class closed" in r for r in out["reasons"])


def test_admission_indicated_with_irreducible_abstain():
    gallery = {"fine_counts": {"over_abstention": 3, "correct_abstention": 1}}
    out = evaluate_admission(gallery, eclass_lm_still_needed=True, min_irreducible=2)
    assert out["verdict"] == "LM_PROBE_INDICATED"
    assert out["lm_probe_indicated"] is True


def test_probe_skips_non_allowlisted():
    docs = load_corpus(DEFAULT_CORPUS)
    row = probe_eclass_task("T01", docs, classical_status="ABSTAIN")
    assert row["skipped"] and row["lm_invoked"] is False


def test_probe_skips_when_classical_resolved():
    docs = load_corpus(DEFAULT_CORPUS)
    row = probe_eclass_task("T35", docs, classical_status="CONFIRMED")
    assert row["skipped"]


def test_stub_probe_t35_span_locked():
    docs = load_corpus(DEFAULT_CORPUS)
    row = probe_eclass_task("T35", docs, backend=StubLMBackend(), classical_status="ABSTAIN")
    assert row["lm_invoked"] is True
    assert row["claim_status"] in {"PRESENT", "CONFIRMED", "ABSTAIN"}
    if row["claim_status"] == "PRESENT":
        assert row["evidence_n"] >= 1


def test_ablation_fails_empty_evidence():
    c = Claim("T35", "x", "300 seconds", evidence=[{"start": 0, "end": 3, "text": "300"}], status="PRESENT")
    assert ablation_fails_support(c, {"x": "300 seconds TTL"})


def test_marginal_probe_clean_corpus_not_applicable():
    out = run_marginal_probe(persist=False, gallery={"fine_counts": {"over_abstention": 0}})
    assert out["eclass_lm_still_needed"] is False
    assert out["product_verdict"] == "NOT_APPLICABLE"
    assert out["admission"]["verdict"] == "LM_PROBE_NOT_INDICATED"
    assert out["dry_run"] is True
    assert all(t in ALLOWLIST_TASKS for t in out["eclass_classical"])


if __name__ == "__main__":
    test_admission_not_indicated_on_clean_eclass()
    test_admission_indicated_with_irreducible_abstain()
    test_probe_skips_non_allowlisted()
    test_probe_skips_when_classical_resolved()
    test_stub_probe_t35_span_locked()
    test_ablation_fails_empty_evidence()
    test_marginal_probe_clean_corpus_not_applicable()
    print("W6_LM_OK")
