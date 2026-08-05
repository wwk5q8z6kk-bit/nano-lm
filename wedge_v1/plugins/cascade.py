"""Ordered classical plugin cascade (W4).

synonym → ocr → coref → (callers may add merge/symbolic).
No LM. No fixture doc-id switches. Routing via plugins/registry.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from wedge_v1.classical.solvers import Claim
from wedge_v1.plugins.registry import registry_snapshot, run_cascade_registered


@dataclass
class CascadeResult:
    claims: list[Claim] = field(default_factory=list)
    modules_run: list[str] = field(default_factory=list)

    def to_meta(self) -> dict:
        return {"modules_run": list(self.modules_run), "n_claims": len(self.claims)}


def run_cascade(docs: dict[str, str], query: str = "", *, want: set[str] | None = None) -> CascadeResult:
    """Run selected plugins. want ⊆ {synonym, ocr, coref}; default all."""
    claims, modules_run = run_cascade_registered(docs, query, want=want)
    return CascadeResult(claims=claims, modules_run=modules_run)


def plugin_registry() -> dict:
    return registry_snapshot()
