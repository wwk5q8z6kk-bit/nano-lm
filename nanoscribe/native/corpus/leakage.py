"""Leakage gate wrapper — delegates to validate.check_leakage."""

from __future__ import annotations

from typing import Any

from nanoscribe.native.corpus.schema import CorpusExample
from nanoscribe.native.corpus.validate import check_leakage as _check_leakage


def check_leakage(examples: list[CorpusExample]) -> dict[str, Any]:
    return _check_leakage(examples)


def leakage_gate_pass(examples: list[CorpusExample]) -> bool:
    return bool(check_leakage(examples).get("pass"))
