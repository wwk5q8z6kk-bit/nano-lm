"""Replay verified-ask from corpus + query; compare to prior CoE payload."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def replay_ask(
    *,
    query: str,
    corpus_dir: Path,
    prior: dict | None = None,
    persist_coe: bool = False,
) -> dict:
    """Re-run ask and optionally compare claim values / statuses to prior."""
    from wedge_v1.runtime import ask, load_corpus
    from wedge_v1.coe.bind import bind_ask_payload
    from wedge_v1.coe.audit import audit_payload

    docs = load_corpus(corpus_dir)
    out = ask(query, corpus_dir=corpus_dir)
    out = bind_ask_payload(out, docs, persist=persist_coe)
    audit = audit_payload(out, docs)
    cmp: dict[str, Any] = {"matched": None}
    if prior is not None:
        cmp = {
            "matched_status": out.get("answer_status") == prior.get("answer_status"),
            "prior_status": prior.get("answer_status"),
            "replay_status": out.get("answer_status"),
            "prior_n_claims": len(prior.get("claims") or []),
            "replay_n_claims": len(out.get("claims") or []),
            "corpus_digest_prior": (prior.get("coe") or {}).get("corpus_digest"),
            "corpus_digest_replay": (out.get("coe") or {}).get("corpus_digest"),
            "digest_match": (prior.get("coe") or {}).get("corpus_digest")
            == (out.get("coe") or {}).get("corpus_digest"),
        }
    return {
        "schema": "nano-lm.wedge_v1.coe_replay.v1",
        "query": query,
        "corpus_dir": str(corpus_dir),
        "payload": out,
        "audit": audit,
        "comparison": cmp,
    }
