"""Corpus ingest for wedge_v1 — md/txt always; PDF text-layer when pypdf available."""
from __future__ import annotations

from pathlib import Path

TEXT_SUFFIXES = {".md", ".txt", ".markdown"}


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf(path: Path) -> str | None:
    try:
        from pypdf import PdfReader
    except Exception:
        return None
    try:
        reader = PdfReader(str(path))
        parts: list[str] = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        body = "\n".join(parts).strip()
        return body or None
    except Exception:
        return None


def needs_ocr_normalize(text: str) -> bool:
    """Detect OCR corruption via frozen substitution table (W5)."""
    from wedge_v1.plugins.lexicon import ocr_subs

    for row in ocr_subs():
        src = str(row.get("from") or "")
        if src and src in text:
            return True
    return False


def load_corpus(corpus_dir: Path, *, normalize: bool | str = "auto") -> dict[str, str]:
    """Load documents from a folder into {doc_id: text}.

    doc_id = file stem. Later files of the same stem overwrite earlier ones
    in deterministic suffix order (.md, .txt, .pdf).
    """
    path = Path(corpus_dir)
    if not path.is_dir():
        return {}

    docs: dict[str, str] = {}
    pdf_skipped = 0

    # Prefer markdown/text first, then PDF fill-in for unique stems
    for p in sorted(path.rglob("*")):
        if not p.is_file():
            continue
        if any(part.startswith(".") for part in p.relative_to(path).parts):
            continue
        suf = p.suffix.lower()
        if suf in TEXT_SUFFIXES:
            docs[p.stem] = _read_text_file(p)

    for p in sorted(path.rglob("*.pdf")):
        if any(part.startswith(".") for part in p.relative_to(path).parts):
            continue
        if p.stem in docs:
            continue  # text wins over PDF for same stem
        body = _read_pdf(p)
        if body is None:
            pdf_skipped += 1
            continue
        docs[p.stem] = body

    if docs and normalize:
        from wedge_v1.plugins.ocr import normalize_text

        if normalize is True or normalize == "always":
            docs = {k: normalize_text(v)[0] for k, v in docs.items()}
        elif normalize == "auto":
            docs = {
                k: (normalize_text(v)[0] if needs_ocr_normalize(v) else v)
                for k, v in docs.items()
            }
    return docs


def corpus_stats(corpus_dir: Path) -> dict:
    docs = load_corpus(corpus_dir)
    n_chars = sum(len(v) for v in docs.values())
    path = Path(corpus_dir)
    n_pdf = len(list(path.rglob("*.pdf"))) if path.is_dir() else 0
    try:
        import pypdf  # noqa: F401

        pypdf_ok = True
    except Exception:
        pypdf_ok = False
    return {
        "corpus_dir": str(path.resolve()) if path.exists() else str(path),
        "n_docs": len(docs),
        "n_chars": n_chars,
        "doc_ids": sorted(docs),
        "n_pdf_files_on_disk": n_pdf,
        "pypdf_available": pypdf_ok,
        "note": None
        if pypdf_ok
        else "PDF text-layer ingest disabled until: pip install pypdf",
    }
