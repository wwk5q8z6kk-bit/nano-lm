"""W6 — ONE real LMBackend end-to-end (MLX Llama, span-binding contract)."""
from __future__ import annotations

from wedge_v1.classical.solvers import Claim
from wedge_v1.lm.mlx_backend import MLXLlamaBackend, extract_quote, relocate_unique
from wedge_v1.lm.probe import ablation_fails_support, get_backend, probe_eclass_task
from wedge_v1.runtime import DEFAULT_CORPUS, load_corpus


def test_get_backend_stub_and_mlx():
    assert get_backend("stub").name == "stub_constructive"
    b = get_backend("mlx")
    assert isinstance(b, MLXLlamaBackend)
    assert b.name == "mlx_llama32_3b_spanbound"


def test_extract_quote_variants():
    assert extract_quote('"TTL as 300 seconds"') == "TTL as 300 seconds"
    assert extract_quote("Answer: TTL as 300 seconds") == "TTL as 300 seconds"
    assert extract_quote("ABSTAIN") == "ABSTAIN"
    assert extract_quote("") is None


def test_relocate_unique_requires_single_hit():
    docs = {
        "a": "We define cache TTL as 300 seconds.",
        "b": "Other text.",
    }
    hit = relocate_unique(docs, "TTL as 300 seconds")
    assert hit is not None
    doc_id, ev = hit
    assert doc_id == "a"
    assert ev["text"] == "TTL as 300 seconds"
    # Ambiguous → None
    docs2 = {"a": "TTL as 300 seconds", "b": "TTL as 300 seconds"}
    assert relocate_unique(docs2, "TTL as 300 seconds") is None


def test_mlx_backend_t35_end_to_end_span_bound():
    """Prove ONE real backend: retrieve → quote → unique relocate → verify."""
    docs = load_corpus(DEFAULT_CORPUS)
    backend = MLXLlamaBackend()
    row = probe_eclass_task("T35", docs, backend=backend, classical_status="ABSTAIN")
    assert row["lm_invoked"] is True
    assert row["backend"] == "mlx_llama32_3b_spanbound"
    assert row["claim_status"] in {"PRESENT", "ABSTAIN"}
    if row["claim_status"] == "PRESENT":
        assert row["evidence_n"] >= 1
        assert row["constructive_faithfulness"] is True
        # Rebuild claim shape for ablation check via propose
        claim = backend.propose("T35", docs)
        assert claim.status == "PRESENT"
        assert claim.evidence
        # ablation_fails_support True ⇒ empty evidence is rejected (contract holds)
        assert ablation_fails_support(claim, docs) is True
        ev = claim.evidence[0]
        body = docs[claim.doc_id]
        assert body[int(ev["start"]) : int(ev["end"])] == ev["text"]
        assert "300" in str(row.get("value")) or "600" in str(row.get("value"))


def test_mlx_backend_oos_abstains_or_fails_closed():
    docs = {"only": "The capital of Mars is not discussed here."}
    claim = MLXLlamaBackend().propose("T35", docs, query="How long before cached entries expire?")
    assert claim.status == "ABSTAIN"


if __name__ == "__main__":
    test_get_backend_stub_and_mlx()
    test_extract_quote_variants()
    test_relocate_unique_requires_single_hit()
    test_mlx_backend_oos_abstains_or_fails_closed()
    test_mlx_backend_t35_end_to_end_span_bound()
    print("W6_MLX_BACKEND_OK")
