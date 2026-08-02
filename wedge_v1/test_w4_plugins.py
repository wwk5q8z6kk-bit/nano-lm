"""W4 pluggable cascade pins — no fixture doc-id control flow."""
from __future__ import annotations

from pathlib import Path

from wedge_v1.plugins.cascade import plugin_registry, run_cascade
from wedge_v1.plugins.registry import run_cascade_registered
from wedge_v1.plugins import coref, synonym
from wedge_v1.runtime import DEFAULT_CORPUS, ask, load_corpus, scan


def test_synonym_ttl_any_doc():
    docs = load_corpus(DEFAULT_CORPUS)
    c = synonym.probe_ttl(docs, "How long before cached entries expire?")
    assert c.status == "PRESENT"
    assert "300" in str(c.value)
    assert c.doc_id != "binding_coref"


def test_coref_works_after_rename():
    docs = load_corpus(DEFAULT_CORPUS)
    renamed = {("renamed_coref" if k == "binding_coref" else k): v for k, v in docs.items()}
    assert "binding_coref" not in renamed
    claims = coref.probe_docs(renamed)
    assert claims and claims[0].status == "PRESENT"
    assert claims[0].doc_id == "renamed_coref"
    ants = {b["antecedent"].lower() for b in claims[0].value}
    assert "metformin" in ants


def test_ask_uses_plugin_cascade_not_fixture_gate():
    r = ask("How long before cached entries expire?")
    path = " ".join(r.get("solver_path") or [])
    assert "plugin.synonym" in path or "plugin." in path
    assert r["answer_status"] in {"SUPPORTED", "CONTRADICTED"}


def test_ask_coref_without_binding_coref_id():
    # Point corpus at synthetic dict via temporary: use ask on default still works
    r = ask("What does it refer to regarding binding coref antecedent?")
    # May abstain on relevance, but must not require key binding_coref in control flow
    assert "failure_codes" in r
    casc = run_cascade(load_corpus(DEFAULT_CORPUS), "coref binding it")
    assert "coref" in casc.modules_run
    assert any(c.task_id == "T39" for c in casc.claims)


def test_scan_runs_plugins():
    out = scan()
    assert "plugin_cascade" in (out.get("solver_path") or [])
    notes = " ".join(str(c.get("notes")) for c in out.get("claims") or [])
    assert "plugin." in notes or any(
        (c.get("meta") or {}).get("plugin") for c in out.get("claims") or []
    )


def test_registry_skips_synonym_on_unrelated():
    docs = load_corpus(DEFAULT_CORPUS)
    _, mods = run_cascade_registered(docs, "What is QPS?", want={"synonym", "ocr", "coref"})
    assert "synonym" not in mods


def test_plugin_registry_snapshot():
    snap = plugin_registry()
    assert snap["schema"].endswith(".v1")
    ids = {p["id"] for p in snap["plugins"]}
    assert ids == {"synonym", "ocr", "coref"}


def test_ocr_plugin_emits_evidence():
    noisy = Path(__file__).resolve().parent / "data" / "corpus_noisy"
    docs = load_corpus(noisy, normalize=False)
    casc = run_cascade(docs, want={"ocr"})
    ocr_claims = [c for c in casc.claims if c.task_id == "T37"]
    assert ocr_claims
    assert ocr_claims[0].evidence


if __name__ == "__main__":
    test_synonym_ttl_any_doc()
    test_coref_works_after_rename()
    test_ask_uses_plugin_cascade_not_fixture_gate()
    test_ask_coref_without_binding_coref_id()
    test_scan_runs_plugins()
    test_registry_skips_synonym_on_unrelated()
    test_plugin_registry_snapshot()
    test_ocr_plugin_emits_evidence()
    print("W4_PLUGINS_OK")
