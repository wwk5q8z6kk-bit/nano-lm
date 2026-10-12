"""Data registry lookup and RunPod launch guards for corpus classes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
REGISTRY_PATH = ROOT / "data" / "REGISTRY.json"

# Layer mixture for architecture-screen and promotion corpora (must sum to 1.0).
MIXTURE_V1: dict[str, float] = {
    "mechanism": 0.50,
    "adversarial": 0.30,
    "teacher_verified": 0.15,
    "rejected_buffer": 0.05,
}

TIER_TARGETS: dict[str, dict[str, int]] = {
    "R30": {
        "unique_tokens_char_level": 50_000_000,
        "exposure_tokens_char_level": 100_000_000,
    },
    "P100": {
        "unique_tokens_char_level_min": 200_000_000,
        "unique_tokens_char_level_max": 300_000_000,
    },
}

CORPUS_CLASS_UNIT_FIXTURE = "NATIVE_UNIT_OVERFIT_FIXTURE"
CORPUS_CLASS_ARCHITECTURE_SCREEN = "ARCHITECTURE_SCREEN"
CORPUS_CLASS_EVAL_FROZEN = "EVAL_FROZEN"

UNIT_FIXTURE_ALLOWED_PURPOSES: frozenset[str] = frozenset(
    {
        "trainer_smoke",
        "overfit_regression",
        "checkpoint_resume",
        "qlora_canary_fixture",
        "cpu_smoke",
    }
)

ARCHITECTURE_SCREEN_ALLOWED_PURPOSES: frozenset[str] = frozenset(
    {
        "architecture_screening",
        "architecture_ranking",
        "30m_revalidation",
        "100m_promotion",
        "extended_training",
    }
)

# Legacy dataset id retained for path resolution.
LEGACY_FIXTURE_IDS: frozenset[str] = frozenset(
    {"p1_distill_train_v1", "native_unit_overfit_fixture_v1", "distill_train"}
)


class CorpusLaunchGuardError(RuntimeError):
    """Raised when a paid GPU launch would train on the wrong corpus class."""


def load_registry(path: Path | None = None) -> dict[str, Any]:
    registry_path = path or REGISTRY_PATH
    return json.loads(registry_path.read_text())


def _datasets_by_id(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entry["dataset_id"]: entry for entry in registry.get("datasets", [])}


def resolve_dataset(
    dataset: str | Path,
    *,
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve a dataset id or on-disk path to a registry entry."""
    registry = registry or load_registry()
    by_id = _datasets_by_id(registry)
    key = str(dataset)
    if key in by_id:
        return by_id[key]
    if key in LEGACY_FIXTURE_IDS:
        for entry in registry.get("datasets", []):
            if entry.get("dataset_id") in LEGACY_FIXTURE_IDS:
                return entry
    path = Path(key)
    for entry in registry.get("datasets", []):
        if Path(entry.get("path", "")) == path or Path(entry.get("path", "")).name == path.name:
            return entry
    raise KeyError(f"dataset not in registry: {dataset}")


def corpus_class_for(dataset: str | Path, *, registry: dict[str, Any] | None = None) -> str:
    return str(resolve_dataset(dataset, registry=registry).get("corpus_class", "UNKNOWN"))


def allowed_purposes_for(corpus_class: str) -> frozenset[str]:
    if corpus_class == CORPUS_CLASS_UNIT_FIXTURE:
        return UNIT_FIXTURE_ALLOWED_PURPOSES
    if corpus_class == CORPUS_CLASS_ARCHITECTURE_SCREEN:
        return ARCHITECTURE_SCREEN_ALLOWED_PURPOSES
    return frozenset()


def assert_corpus_launch_allowed(
    dataset: str | Path,
    *,
    purpose: str,
    cpu_smoke: bool = False,
    registry: dict[str, Any] | None = None,
) -> None:
    """Fail before RunPod when UNIT_FIXTURE would be used for ranking/promotion."""
    if cpu_smoke:
        return
    entry = resolve_dataset(dataset, registry=registry)
    corpus_class = str(entry.get("corpus_class", "UNKNOWN"))
    allowed = allowed_purposes_for(corpus_class)
    if purpose in allowed:
        return
    disallowed = set(entry.get("disallowed_uses", []))
    if purpose in disallowed:
        raise CorpusLaunchGuardError(
            f"purpose {purpose!r} is disallowed for {entry.get('dataset_id')} "
            f"(corpus_class={corpus_class})"
        )
    if corpus_class == CORPUS_CLASS_UNIT_FIXTURE:
        raise CorpusLaunchGuardError(
            f"corpus_class={corpus_class} cannot launch GPU training for purpose={purpose!r}. "
            f"Allowed: {sorted(allowed)}. Build an architecture-screen corpus first."
        )
    if not allowed:
        raise CorpusLaunchGuardError(
            f"unknown corpus_class={corpus_class!r} for dataset {entry.get('dataset_id')}"
        )
    raise CorpusLaunchGuardError(
        f"purpose {purpose!r} not in allowed purposes for {corpus_class}: {sorted(allowed)}"
    )
