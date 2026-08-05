"""W4 plugin registry — lexicon-driven routing, no fixture doc-id gates."""
from __future__ import annotations

import re
from typing import Callable

from wedge_v1.classical.solvers import Claim
from wedge_v1.plugins import coref, ocr, synonym
from wedge_v1.plugins.lexicon import coref_config, synonyms

PluginProbe = Callable[[dict[str, str], str], list[Claim]]


def _synonym_triggers(query: str, docs: dict[str, str]) -> bool:
    if not query.strip():
        return True
    ql = query.lower()
    raw = set(re.findall(r"[a-z0-9]+", ql))
    for src, dsts in synonyms().items():
        src_l = src.lower()
        if src_l in ql:
            return True
        if raw & {d.lower() for d in dsts}:
            return True
    expanded = synonym.expand_terms(query)
    return bool(expanded - raw) or bool(raw & {"ttl", "cache", "expire", "timeout", "cached"})


def _coref_triggers(query: str, docs: dict[str, str]) -> bool:
    cfg = coref_config()
    pronouns = cfg.get("pronouns") or ["It"]
    if not query.strip():
        return any(
            re.search(rf"(?<![A-Za-z]){re.escape(p)}(?![A-Za-z])", text)
            for text in docs.values()
            for p in pronouns
        )
    ql = query.lower()
    if any(k in ql for k in ("binding", "coref", "antecedent", "pronoun", "refer", "pronoun")):
        return True
    return any(
        re.search(rf"(?<![A-Za-z]){re.escape(p.lower())}(?![A-Za-z])", ql)
        for p in pronouns
    )


def _ocr_triggers(query: str, docs: dict[str, str]) -> bool:
    if not query.strip():
        return True
    ql = query.lower()
    if any(k in ql for k in ("ocr", "noisy", "typo", "normalize", "scan")):
        return True
    from wedge_v1.plugins.lexicon import ocr_subs

    subs = [str(r.get("from") or "") for r in ocr_subs() if r.get("from")]
    return any(any(s in text for s in subs) for text in docs.values())


def _run_synonym(docs: dict[str, str], query: str) -> list[Claim]:
    q = query.strip() or "How long before cached entries expire?"
    c = synonym.probe_paraphrase(docs, q)
    return [c] if c.status != "ABSTAIN" else []


def _run_ocr(docs: dict[str, str], query: str) -> list[Claim]:
    return ocr.probe_docs(docs)


def _run_coref(docs: dict[str, str], query: str) -> list[Claim]:
    return coref.probe_docs(docs)


PLUGINS: tuple[dict, ...] = (
    {
        "id": "synonym",
        "order": 10,
        "should_run": _synonym_triggers,
        "run": _run_synonym,
        "task_ids": ("T35",),
        "lexicon": "plugins/data/synonyms.json",
    },
    {
        "id": "ocr",
        "order": 20,
        "should_run": _ocr_triggers,
        "run": _run_ocr,
        "task_ids": ("T37",),
        "lexicon": "plugins/data/ocr_substitutions.json",
    },
    {
        "id": "coref",
        "order": 30,
        "should_run": _coref_triggers,
        "run": _run_coref,
        "task_ids": ("T39",),
        "lexicon": "plugins/data/coref_entities.json",
    },
)


def registry_snapshot() -> dict:
    return {
        "schema": "nano-lm.wedge_v1.plugin_registry.v1",
        "plugins": [
            {
                "id": p["id"],
                "order": p["order"],
                "task_ids": list(p["task_ids"]),
                "lexicon": p["lexicon"],
            }
            for p in PLUGINS
        ],
        "note": "Lexicon-driven classical plugins; no fixture doc-id control flow.",
    }


def run_cascade_registered(
    docs: dict[str, str],
    query: str = "",
    *,
    want: set[str] | None = None,
) -> tuple[list[Claim], list[str]]:
    want = want or {p["id"] for p in PLUGINS}
    claims: list[Claim] = []
    modules_run: list[str] = []
    for row in sorted(PLUGINS, key=lambda r: r["order"]):
        pid = row["id"]
        if pid not in want:
            continue
        should = row["should_run"]
        if not should(query, docs):
            continue
        modules_run.append(pid)
        claims.extend(row["run"](docs, query))
    return claims, modules_run
