"""Pre-launch guard: a unit fixture may not back a scientific claim.

The 96-row / 4,516-char corpus produced a full round of architecture rankings
before anyone checked whether it could support them. This makes that class of
mistake fail loudly at launch instead of silently at conclusion time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

UNIT_FIXTURE = "NATIVE_UNIT_OVERFIT_FIXTURE"
REAL_CORPUS = "NATIVE_TRAINING_CORPUS"

#: Purposes a unit fixture is allowed to serve.
ALLOWED_UNIT_FIXTURE_PURPOSES: frozenset[str] = frozenset(
    {
        "forward_pass_test",
        "backward_pass_test",
        "finite_loss_test",
        "optimizer_step_test",
        "checkpoint_resume_test",
        "tiny_task_overfit",
        "parser_contract_regression",
        "qlora_compatibility_canary",
        "trainer_smoke",
    }
)

#: Purposes that require a real corpus, whatever the file is called.
SCIENTIFIC_PURPOSES: frozenset[str] = frozenset(
    {
        "architecture_ranking",
        "scale_comparison",
        "held_out_generalization",
        "winner_claim",
        "promotion",
        "student_adaptation_quality",
        "p1_capability_claim",
        "confirmatory",
    }
)

#: Below this, a corpus cannot support a scientific claim regardless of label.
MIN_SCIENTIFIC_TOKENS = 1_000_000


class CorpusGuardError(RuntimeError):
    """Raised when a corpus may not be used for the requested purpose."""


def classify(manifest: dict[str, Any]) -> str:
    tokens = (manifest.get("statistics") or {}).get("training_tokens_char_level")
    if tokens is None:
        tokens = manifest.get("training_tokens_char_level", 0)
    return REAL_CORPUS if (tokens or 0) >= MIN_SCIENTIFIC_TOKENS else UNIT_FIXTURE


def assert_usable(manifest: dict[str, Any], purpose: str) -> None:
    """Raise unless `manifest` may back `purpose`.

    Two independent checks, because a label alone is not evidence:
      1. an explicitly declared unit fixture may only serve fixture purposes;
      2. any corpus below MIN_SCIENTIFIC_TOKENS is treated as a fixture even if
         it claims otherwise — the 4.5KB corpus was never labelled a fixture, it
         was simply used as if it were a corpus.
    """
    declared = manifest.get("corpus_class") or classify(manifest)
    tokens = (manifest.get("statistics") or {}).get("training_tokens_char_level") or manifest.get(
        "training_tokens_char_level", 0
    )

    if declared == UNIT_FIXTURE and purpose not in ALLOWED_UNIT_FIXTURE_PURPOSES:
        raise CorpusGuardError(
            f"corpus '{manifest.get('corpus_id', '?')}' is {UNIT_FIXTURE} "
            f"({tokens:,} tokens) and may not be used for purpose '{purpose}'. "
            f"Allowed: {sorted(ALLOWED_UNIT_FIXTURE_PURPOSES)}"
        )

    if purpose in SCIENTIFIC_PURPOSES and (tokens or 0) < MIN_SCIENTIFIC_TOKENS:
        raise CorpusGuardError(
            f"purpose '{purpose}' requires at least {MIN_SCIENTIFIC_TOKENS:,} training tokens; "
            f"corpus '{manifest.get('corpus_id', '?')}' has {tokens:,}. "
            "This is the check that the 4,516-char corpus would have failed."
        )


def assert_manifest_path_usable(path: str | Path, purpose: str) -> None:
    payload = json.loads(Path(path).read_text())
    assert_usable(payload, purpose)
