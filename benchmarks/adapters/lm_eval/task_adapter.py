"""Map held-value instruments into the harness without duplicating ground truth."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schemas import sha256_file, sha256_hex

REPO_ROOT = Path(__file__).resolve().parents[3]

PROMPT_TEMPLATE = (
    "Given the clinical dialogue, emit the structured summary line "
    "exactly as: CC: <cc> | DUR: <dur> | SEV: <sev> | MED: <med> | ALG: <alg>"
)

SCORER_IMPL = "exact_match_normalized_v1"
FILTER_PIPELINE = "strip_whitespace_v1"


@dataclass(frozen=True)
class BoundInstrument:
    repo_relative_path: str
    absolute_path: Path
    git_commit: str
    sha256: str
    schema_version: str
    record_count: int
    records: list[dict[str, Any]]


def _normalize(text: str) -> str:
    return " ".join(text.strip().split())


def format_target(tuple_fields: dict[str, str]) -> str:
    return (
        f"CC: {tuple_fields['cc']} | DUR: {tuple_fields['dur']} | "
        f"SEV: {tuple_fields['sev']} | MED: {tuple_fields['med']} | "
        f"ALG: {tuple_fields['alg']}"
    )


def load_and_bind_instrument(
    repo_relative_path: str,
    *,
    git_commit: str,
    expected_sha256: str | None = None,
    expected_record_count: int | None = None,
    schema_version: str = "held_value_scribe_v1",
) -> BoundInstrument:
    path = REPO_ROOT / repo_relative_path
    if not path.is_file():
        raise FileNotFoundError(f"instrument missing: {repo_relative_path}")
    digest = sha256_file(str(path))
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"instrument digest mismatch for {repo_relative_path}: "
            f"expected {expected_sha256}, got {digest}"
        )
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("instrument must be a JSON list")
    if expected_record_count is not None and len(records) != expected_record_count:
        raise ValueError(
            f"record_count mismatch: expected {expected_record_count}, got {len(records)}"
        )
    return BoundInstrument(
        repo_relative_path=repo_relative_path,
        absolute_path=path,
        git_commit=git_commit,
        sha256=digest,
        schema_version=schema_version,
        record_count=len(records),
        records=records,
    )


def docs_from_instrument(bound: BoundInstrument) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for i, rec in enumerate(bound.records):
        user = rec["convo"][0]["content"]
        target = rec["convo"][1]["content"]
        docs.append(
            {
                "item_id": f"{bound.sha256[:12]}:{i}",
                "held_values": bool(rec.get("held_values")),
                "tuple": rec["tuple"],
                "doc_to_text": user,
                "doc_to_target": target,
                "prompt_prefix": PROMPT_TEMPLATE,
            }
        )
    return docs


def exact_match(pred: str, target: str) -> float:
    return 1.0 if _normalize(pred) == _normalize(target) else 0.0


def prompt_template_hash() -> str:
    return sha256_hex(PROMPT_TEMPLATE)


def scorer_hash() -> str:
    return sha256_hex(SCORER_IMPL)


def filter_pipeline_hash() -> str:
    return sha256_hex(FILTER_PIPELINE)


FIELD_PATTERNS = {
    "cc": re.compile(
        r"Patient: (?:Honestly, |It started as )(.+?)(?: has been troubling me\.| and hasn't stopped\.)",
        re.S,
    ),
}
