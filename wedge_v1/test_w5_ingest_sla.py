"""W5 ingest SLA / recover_gap pins."""
from __future__ import annotations

from pathlib import Path

from wedge_v1.ingest_sla import extract_fields, measure_ingest_sla, normalize_corpus
from wedge_v1.plugins.ocr import normalize_text
from wedge_v1.ingest import load_corpus
from wedge_v1.runtime import DEFAULT_CORPUS


def test_ocr_lexicon_recovers_ttl_glyphs():
    raw = "TTL  i5  300 secands"
    fixed, edits = normalize_text(raw)
    assert edits
    assert "seconds" in fixed
    assert "TTL as" in fixed or "TTL" in fixed


def test_field_extract_on_clean():
    docs = load_corpus(DEFAULT_CORPUS)
    fields = extract_fields(docs)
    assert any("ttl_seconds" in f for f in fields.values())


def test_normalize_improves_synthetic_noisy():
    out = measure_ingest_sla(clean_dir=DEFAULT_CORPUS, noisy_dir=Path("/nonexistent_noisy_dir"))
    assert out["synthetic_noisy"] is True
    assert out["fields_normalized"]["recovery_rate"] >= out["fields_raw"]["recovery_rate"]
    assert out["sla_field_ok"] is True
    assert out["verdict"] in {
        "FIELD_SLA_PASS",
        "NOISY_INGEST_NORMALIZE_SUFFICIENT",
    }


def test_auto_normalize_noisy_corpus():
    noisy = Path(__file__).resolve().parent / "data" / "corpus_noisy"
    raw = load_corpus(noisy, normalize=False)
    auto = load_corpus(noisy, normalize="auto")
    from wedge_v1.ingest import needs_ocr_normalize

    assert needs_ocr_normalize(raw.get("noisy_ocr_line", ""))
    assert "secands" not in auto.get("noisy_ocr_line", "")
    fields = extract_fields(auto)
    assert fields.get("noisy_ocr_line", {}).get("ttl_seconds") == "250"


def test_load_corpus_normalize_flag():
    docs = load_corpus(DEFAULT_CORPUS, normalize=True)
    assert docs
    # should still find TTL after normalize pass
    blob = "\n".join(docs.values())
    assert "TTL" in blob


def test_ingest_sla_cli_main(tmp_path: Path):
    from wedge_v1.ingest_sla import main

    output = tmp_path / "results_ingest_sla.json"
    rc = main(
        [
            "--clean",
            str(DEFAULT_CORPUS),
            "--noisy",
            "/nope",
            "--output",
            str(output),
        ]
    )
    assert rc == 0
    assert output.is_file()


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    test_ocr_lexicon_recovers_ttl_glyphs()
    test_field_extract_on_clean()
    test_auto_normalize_noisy_corpus()
    test_normalize_improves_synthetic_noisy()
    test_load_corpus_normalize_flag()
    with TemporaryDirectory() as tmp:
        test_ingest_sla_cli_main(Path(tmp))
    print("W5_INGEST_SLA_OK")
