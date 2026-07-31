"""Coref-lite plugin — any document; entity lexicon driven."""
from __future__ import annotations

import re

from wedge_v1.classical.solvers import Claim
from wedge_v1.plugins.lexicon import coref_config


def _entity_re() -> re.Pattern[str]:
    cfg = coref_config()
    ents = cfg.get("entities") or ["Metformin", "Placebo"]
    alt = "|".join(re.escape(e) for e in ents)
    # Build word-boundary pattern without broken escapes in source writers
    return re.compile(r"(?<![A-Za-z])(" + alt + r")(?![A-Za-z])", re.I)


def bind_doc(doc_id: str, text: str) -> Claim:
    """Bind sentence-initial pronouns to nearest prior lexicon entity."""
    cfg = coref_config()
    pronouns = cfg.get("pronouns") or ["It"]
    ent = _entity_re()
    body_lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or s.lower().startswith("authors:") or s.lower().startswith("year:"):
            continue
        body_lines.append(s)
    body = " ".join(body_lines)
    sents = [s.strip() for s in re.split(r"(?<=[.!?])" + r"\s+", body) if s.strip()]
    binds = []
    last = None
    last_span = None
    for sent in sents:
        ents = ent.findall(sent)
        if ents:
            last = ents[-1]
            i = text.find(last)
            last_span = {
                "doc_id": doc_id,
                "start": i if i >= 0 else None,
                "end": (i + len(last)) if i >= 0 else None,
                "text": last,
            }
        for pron in pronouns:
            if re.match(rf"^{re.escape(pron)}(?![A-Za-z])", sent) and last:
                binds.append({"pronoun_sentence": sent, "antecedent": last, "pronoun": pron})
                break
    if not binds:
        return Claim("T39", doc_id, [], status="ABSTAIN", notes="plugin.coref.none", meta={"plugin": "coref"})
    evidence = [last_span] if last_span else [{"doc_id": doc_id, "text": binds[0]["antecedent"]}]
    return Claim(
        "T39",
        doc_id,
        binds,
        evidence=evidence,
        status="PRESENT",
        notes="plugin.coref.lite",
        meta={"plugin": "coref"},
    )


def probe_docs(docs: dict[str, str]) -> list[Claim]:
    """Run coref on every doc that contains a configured pronoun — no fixture id gate."""
    cfg = coref_config()
    pronouns = cfg.get("pronouns") or ["It"]
    out = []
    for did, text in docs.items():
        if not any(re.search(rf"(?<![A-Za-z]){re.escape(p)}(?![A-Za-z])", text) for p in pronouns):
            continue
        c = bind_doc(did, text)
        if c.status != "ABSTAIN":
            out.append(c)
    return out
