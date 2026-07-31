"""Load versioned lexicon JSON for plugins."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


@lru_cache(maxsize=16)
def load_json(name: str) -> dict:
    path = DATA / name
    return json.loads(path.read_text(encoding="utf-8"))


def synonyms() -> dict[str, list[str]]:
    return dict(load_json("synonyms.json").get("expand") or {})


def ocr_subs() -> list[dict]:
    rows = load_json("ocr_substitutions.json").get("substitutions") or []
    return [r for r in rows if r.get("enabled", True)]


def coref_config() -> dict:
    return load_json("coref_entities.json")
