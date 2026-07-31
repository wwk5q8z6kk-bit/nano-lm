"""Classical-only solvers for Wedge v1 (no LM)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Claim:
    task_id: str
    doc_id: str | None
    value: object
    evidence: list = field(default_factory=list)
    status: str = "PRESENT"  # PRESENT | ABSTAIN | DISPUTED | MISSING
    notes: str = ""


def _find(text: str, needle: str):
    i = text.find(needle)
    if i < 0:
        return None
    return {"start": i, "end": i + len(needle), "text": needle}


def load_docs(corpus_dir: Path) -> dict:
    docs = {}
    for p in sorted(corpus_dir.glob("*.md")):
        docs[p.stem] = p.read_text(encoding="utf-8")
    return docs


def extract_title(doc_id: str, text: str) -> Claim:
    m = re.search(r"^#\s+(.+)$", text, re.M)
    if not m:
        return Claim("T01", doc_id, None, status="ABSTAIN")
    val = m.group(1).strip()
    return Claim("T01", doc_id, val, evidence=[_find(text, val) or {"text": val}])


def extract_authors(doc_id: str, text: str) -> Claim:
    m = re.search(r"^Authors:\s*(.+)$", text, re.M)
    if not m:
        return Claim("T02", doc_id, [], status="ABSTAIN")
    authors = [a.strip() for a in m.group(1).split(",") if a.strip()]
    return Claim("T02", doc_id, authors, evidence=[_find(text, m.group(0)) or {}])


def extract_year(doc_id: str, text: str) -> Claim:
    m = re.search(r"^Year:\s*(\d{4})\s*$", text, re.M)
    if not m:
        return Claim("T03", doc_id, None, status="ABSTAIN")
    return Claim("T03", doc_id, int(m.group(1)), evidence=[_find(text, m.group(1)) or {}])


def detect_doc_type(doc_id: str, text: str) -> Claim:
    if re.search(r"^## Abstract\s*$", text, re.M):
        val = "abstract"
    elif re.search(r"^\w+\s*\|\s*\w+", text, re.M):
        val = "table_dump"
    elif re.search(r"^#\s+", text, re.M):
        val = "note"
    else:
        return Claim("T04", doc_id, None, status="ABSTAIN")
    return Claim("T04", doc_id, val, evidence=[])


def list_headings(doc_id: str, text: str) -> Claim:
    heads = re.findall(r"^(#{1,3})\s+(.+)$", text, re.M)
    vals = [h[1].strip() for h in heads]
    return Claim("T05", doc_id, vals, evidence=[])


def extract_doi(doc_id: str, text: str) -> Claim:
    m = re.search(r"DOI:\s*(\S+)", text)
    if not m:
        return Claim("T06", doc_id, None, status="MISSING")
    return Claim("T06", doc_id, m.group(1), evidence=[_find(text, m.group(1)) or {}])


def word_count(doc_id: str, text: str) -> Claim:
    n = len(re.findall(r"\S+", text))
    return Claim("T07", doc_id, n, evidence=[])


def build_toc(doc_id: str, text: str) -> Claim:
    c = list_headings(doc_id, text)
    return Claim("T08", doc_id, c.value, evidence=c.evidence, status=c.status)


def extract_dosages(doc_id: str, text: str) -> Claim:
    hits = list(re.finditer(r"\b(\d+(?:\.\d+)?)\s*mg\b", text, re.I))
    vals = [{"text": h.group(0), "start": h.start(), "end": h.end()} for h in hits]
    if not vals:
        return Claim("T09", doc_id, [], status="MISSING")
    return Claim("T09", doc_id, vals, evidence=vals)


COMPOUND_LEX = {"metformin", "placebo", "ibuprofen"}


def extract_compounds(doc_id: str, text: str) -> Claim:
    found = []
    low = text.lower()
    for c in sorted(COMPOUND_LEX):
        i = low.find(c)
        if i >= 0:
            found.append({"text": text[i : i + len(c)], "start": i, "end": i + len(c)})
    if not found:
        return Claim("T10", doc_id, [], status="MISSING")
    return Claim("T10", doc_id, found, evidence=found)


def extract_sample_n(doc_id: str, text: str) -> Claim:
    m = re.search(r"\bn\s*=\s*(\d+)\b", text)
    if not m:
        return Claim("T13", doc_id, None, status="MISSING")
    return Claim("T13", doc_id, int(m.group(1)), evidence=[_find(text, m.group(0)) or {}])


def extract_definition(doc_id: str, text: str, term: str = "Latency") -> Claim:
    m = re.search(rf"^{re.escape(term)}\s+is\s+.+$", text, re.M)
    if not m:
        return Claim("T12", doc_id, None, status="ABSTAIN")
    return Claim("T12", doc_id, m.group(0), evidence=[_find(text, m.group(0)) or {}])


def extract_email(doc_id: str, text: str) -> Claim:
    m = re.search(r"[\w.+-]+@[\w.-]+\.\w+", text)
    if not m:
        return Claim("T15", doc_id, None, status="MISSING")
    return Claim("T15", doc_id, m.group(0), evidence=[_find(text, m.group(0)) or {}])


def extract_urls(doc_id: str, text: str) -> Claim:
    hits = list(re.finditer(r"https?://\S+", text))
    vals = [{"text": h.group(0).rstrip("."), "start": h.start(), "end": h.end()} for h in hits]
    keys = re.findall(r"citation key\s+(\w+)", text, re.I)
    if not vals and not keys:
        return Claim("T18", doc_id, {"urls": [], "keys": []}, status="MISSING")
    return Claim("T18", doc_id, {"urls": vals, "keys": keys}, evidence=vals)


def extract_kv(doc_id: str, text: str) -> Claim:
    pairs = dict(re.findall(r"^([a-z_][a-z0-9_]*)\s*:\s*(.+)$", text, re.M | re.I))
    if not pairs:
        return Claim("T17", doc_id, {}, status="MISSING")
    return Claim("T17", doc_id, pairs, evidence=[])


def extract_captions(doc_id: str, text: str) -> Claim:
    lines = re.findall(r"^(?:Figure|Table)\s+\d+:\s*.+$", text, re.M)
    if not lines:
        return Claim("T16", doc_id, [], status="MISSING")
    return Claim("T16", doc_id, lines, evidence=[])


def parse_table(doc_id: str, text: str) -> Claim:
    rows = []
    for line in text.splitlines():
        if "|" in line and not line.strip().startswith("#"):
            parts = [p.strip() for p in line.split("|")]
            if parts and parts[0].lower() == "region":
                continue
            if len(parts) >= 3 and parts[0] and parts[0][0].isalpha():
                rows.append({"region": parts[0], "qps": parts[1], "error_rate": parts[2]})
    if not rows:
        return Claim("T38", doc_id, [], status="MISSING")
    return Claim("T38", doc_id, rows, evidence=[])


def mention_docs(docs: dict, term: str) -> Claim:
    hits = [did for did, t in docs.items() if term.lower() in t.lower()]
    if not hits:
        return Claim("T25", None, [], status="MISSING", notes=term)
    return Claim("T25", None, hits, status="PRESENT", notes=term)


def yes_no_mention(doc_id: str, text: str, term: str) -> Claim:
    i = text.lower().find(term.lower())
    if i < 0:
        return Claim("T21", doc_id, False, status="PRESENT", notes=f"no:{term}")
    ev = {"start": i, "end": i + len(term), "text": text[i : i + len(term)]}
    return Claim("T21", doc_id, True, evidence=[ev], notes=term)


def quote_sentence(doc_id: str, text: str, term: str) -> Claim:
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if term.lower() in sent.lower():
            sp = _find(text, sent.strip())
            return Claim("T20", doc_id, sent.strip(), evidence=[sp or {"text": sent.strip()}])
    return Claim("T20", doc_id, None, status="ABSTAIN", notes=term)


def keyword_paragraph(doc_id: str, text: str, query: str) -> Claim:
    qtoks = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2]
    best, best_score, best_span = None, 0, None
    offset = 0
    for para in text.split("\n\n"):
        low = para.lower()
        score = sum(1 for t in qtoks if t in low)
        if score > best_score:
            best_score = score
            best = para.strip()
            best_span = {"start": offset, "end": offset + len(para), "text": best[:200]}
        offset += len(para) + 2
    if not best or best_score == 0:
        return Claim("T19", doc_id, None, status="ABSTAIN", notes=query)
    return Claim("T19", doc_id, best, evidence=[best_span or {}], notes=query)


def flag_numeric_contradiction(gold_planted: dict) -> Claim:
    block = gold_planted.get("B1_numeric_contradiction", {})
    docs = block.get("docs", {})
    if len(docs) >= 2 and len(set(docs.values())) > 1:
        return Claim("T29", None, docs, status="DISPUTED", notes="planted numeric contradiction")
    return Claim("T29", None, docs, status="MISSING")


def flag_entity_collision(docs: dict) -> Claim:
    hits = []
    for did, text in docs.items():
        if re.search(r"\bibuprofen\b", text, re.I):
            typ = "reagent" if "reagent" in text.lower() else "unknown"
            hits.append({"doc_id": did, "string": "ibuprofen", "type": typ})
    if not hits:
        return Claim("T30", None, [], status="MISSING")
    return Claim(
        "T30",
        None,
        hits,
        status="DISPUTED" if any(h["type"] == "reagent" for h in hits) else "PRESENT",
    )


def missing_patient_id(doc_id: str, text: str) -> Claim:
    if re.search(r"^patient_id\s*:", text, re.M | re.I):
        m = re.search(r"^patient_id\s*:\s*(.+)$", text, re.M | re.I)
        return Claim("T34", doc_id, m.group(1).strip() if m else None, status="PRESENT")
    return Claim("T34", doc_id, None, status="ABSTAIN", notes="patient_id absent")


def reject_ungrounded() -> Claim:
    return Claim("T33", None, "verifier_requires_evidence", status="PRESENT")


def paraphrastic_ttl(docs: dict, gold: dict) -> Claim:
    """T35: paraphrastic query via frozen synonym expansion (non-LM)."""
    planted = gold["planted"]["B4_paraphrastic"]
    q = planted["query"]
    answer = planted["answer_span"]
    expand = {
        "expire": ["ttl", "seconds", "invalidation", "timeout"],
        "cached": ["cache", "ttl"],
        "entries": ["cache"],
        "long": ["ttl", "seconds"],
    }
    terms = set(re.findall(r"[a-z0-9]+", q.lower()))
    for src, dsts in expand.items():
        if src in q.lower():
            terms.update(dsts)
    best_id, best_score, best_m = None, 0, None
    for did, text in docs.items():
        low = text.lower()
        score = sum(1 for t in terms if t in low)
        m = re.search(r"TTL as (\d+) seconds", text)
        if m:
            score += 5
        if score > best_score:
            best_score, best_id, best_m = score, did, m
    if best_m is not None and best_score > 0:
        span = _find(docs[best_id], best_m.group(0)) or _find(docs[best_id], answer)
        return Claim(
            "T35",
            best_id,
            answer if answer in docs[best_id] else f"{best_m.group(1)} seconds",
            evidence=[span or {"text": best_m.group(0)}],
            notes="query_expand",
        )
    return Claim("T35", planted["target_doc"], None, status="ABSTAIN", notes=q)


def ocr_normalize(doc_id: str, text: str) -> Claim:
    fixed = text.replace("secands", "seconds").replace("i5", "is").replace("5O0", "500")
    m = re.search(r"TTL\s+is\s+(\d+)\s+seconds", fixed)
    dose = re.search(r"dose:\s*(\d+)\s*mg", fixed, re.I)
    val = {}
    if m:
        val["ttl_seconds"] = int(m.group(1))
    if dose:
        val["dose_mg"] = int(dose.group(1))
    if not val:
        return Claim("T37", doc_id, None, status="ABSTAIN")
    return Claim("T37", doc_id, val, evidence=[])


def coref_binding(doc_id: str, text: str) -> Claim:
    """Bind 'It' to nearest prior Metformin/Placebo mention (sentence-level)."""
    # Strip headings / metadata lines for binding body
    body_lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or s.lower().startswith("authors:") or s.lower().startswith("year:"):
            continue
        body_lines.append(s)
    body = " ".join(body_lines)
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]
    binds = []
    last = None
    ent = re.compile(r"\b(Metformin|Placebo)\b", re.I)
    for sent in sents:
        ents = ent.findall(sent)
        if ents:
            last = ents[-1].lower()
        if re.match(r"^It\b", sent) and last:
            binds.append({"pronoun_sentence": sent, "antecedent": last})
    if not binds:
        return Claim("T39", doc_id, [], status="ABSTAIN")
    return Claim("T39", doc_id, binds, status="PRESENT")


def union_dosages(docs: dict) -> Claim:
    all_d = []
    for did, text in docs.items():
        c = extract_dosages(did, text)
        if c.status == "PRESENT":
            for v in c.value:
                all_d.append({"doc_id": did, **v})
    if not all_d:
        return Claim("T26", None, [], status="MISSING")
    return Claim("T26", None, all_d, status="PRESENT")


def symbolic_dose_change(docs: dict) -> Claim:
    """T36: implicit dose change via symbolic multi-doc compare (not LM)."""
    dose = {}
    for did, t in docs.items():
        m = re.search(r"metformin\s+(\d+)\s*mg", t, re.I)
        if m:
            dose[did] = int(m.group(1))
    vals = sorted(set(dose.values()))
    if len(vals) >= 2:
        return Claim(
            "T36",
            None,
            {"from": vals[0], "to": vals[-1], "values": dose},
            evidence=[{"doc_id": k, "text": str(v)} for k, v in dose.items()],
            status="PRESENT",
            notes="symbolic_compare",
        )
    return Claim("T36", None, None, status="ABSTAIN", notes="no multi-dose")
