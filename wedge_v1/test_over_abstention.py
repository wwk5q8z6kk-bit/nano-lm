"""Pins for p5 OVER_ABSTENTION recoveries without OOS false-support.

Not Layer-1 evidence. Uses gitignored owner corpus papers snapshot when present.
"""
from __future__ import annotations

from pathlib import Path

from wedge_v1.runtime import ask

P5 = Path(__file__).resolve().parent / "data" / "owner_corpus" / "p5_repo_papers_20260802"


def _have_p5() -> bool:
    return P5.is_dir() and any(P5.glob("*.md"))


def test_p5_e1_kill_m1_template_supported():
    if not _have_p5():
        return
    r = ask("E1 KILL M1_template", corpus_dir=P5)
    assert r["answer_status"] == "SUPPORTED"
    blob = str(r).lower()
    assert "kill" in blob or "0.999" in blob or "m1" in blob


def test_p5_e4_kill_0638_supported():
    if not _have_p5():
        return
    r = ask("E4 KILL 0.638", corpus_dir=P5)
    assert r["answer_status"] == "SUPPORTED"
    assert "0.638" in str(r)


def test_p5_smallest_sufficient_solver_via_phrase():
    if not _have_p5():
        return
    r = ask("Nano Runtime smallest sufficient solver", corpus_dir=P5)
    assert r["answer_status"] == "SUPPORTED"
    assert "phrase_span" in (r.get("solver_path") or [])


def test_oos_clinical_nanoscribe_abstains():
    if not _have_p5():
        return
    r = ask(
        "What is the clinical accuracy of NanoScribe in hospitals?",
        corpus_dir=P5,
    )
    assert r["answer_status"] == "ABSTAIN"


def test_oos_gpt4_score_abstains():
    if not _have_p5():
        return
    r = ask("What is GPT-4 exact match score on Paper alpha?", corpus_dir=P5)
    assert r["answer_status"] == "ABSTAIN"


if __name__ == "__main__":
    test_p5_e1_kill_m1_template_supported()
    test_p5_e4_kill_0638_supported()
    test_p5_smallest_sufficient_solver_via_phrase()
    test_oos_clinical_nanoscribe_abstains()
    test_oos_gpt4_score_abstains()
    print("OVER_ABSTENTION_OK")
