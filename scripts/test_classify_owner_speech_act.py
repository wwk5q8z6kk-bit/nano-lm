#!/usr/bin/env python3
"""Unit tests for B23 speech-act classifier (no network, no git mutation)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "classify_owner_speech_act", ROOT / "scripts" / "classify_owner_speech_act.py"
)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_continue_grants_no_bits():
    out = mod.classify("continue")
    assert out["force"] == "CONTINUE_SESSION"
    assert out["scope_bits"] == []
    assert out["may_mint_owner_marker"] is False


def test_authorize_commit():
    out = mod.classify("authorize commit")
    assert out["force"] == "AUTHORIZE_COMMIT"
    assert "commit" in out["scope_bits"]
    assert out["may_mint_owner_marker"] is True


def test_authorize_tag_requires_policy_field():
    out = mod.classify("authorize tag clean-lineage")
    assert out["force"] == "AUTHORIZE_TAG"
    assert out["tip_policy"] == "clean-lineage"


def test_proceed_is_authorize_commit():
    """Lab convention (OWNER_SPEECH_ACTS.md): bare proceed ⇒ commit-only, not push/tag/execute."""
    out = mod.classify("proceed")
    assert out["force"] == "AUTHORIZE_COMMIT"
    assert out["scope_bits"] == ["commit"]
    assert out["may_mint_owner_marker"] is True


def test_untyped():
    out = mod.classify("please just do the freeze thing")
    assert out["force"] == "UNTYPED"
    assert out["may_mint_owner_marker"] is False


def test_dispose_e4():
    assert mod.classify("PARK_AS_EXPLORATORY")["force"] == "DISPOSE_E4"


if __name__ == "__main__":
    test_continue_grants_no_bits()
    test_authorize_commit()
    test_authorize_tag_requires_policy_field()
    test_proceed_is_authorize_commit()
    test_untyped()
    test_dispose_e4()
    print("PASS")
