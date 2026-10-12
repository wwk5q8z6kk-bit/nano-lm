"""Test isolation for the nanoscribe package.

`NANO_NATIVE_TOKENIZER` selects which tokenizer a run uses, and
`integrity.assert_requested_tokenizer_honoured` compares that request against
the resolved config. Most tests build a config directly, bypassing the
factories, so they carry no launch request — if the ambient environment holds
one, the guard correctly reports a mismatch and the suite goes red for a reason
unconnected to the code under test.

That is the environment-dependent-test defect twice over in this session: first
`test_candidate_raw_value_when_quote_missing` reading the C2 flag from a module
default, then six tests here going red only when the suite was run under the
D3.3 launch configuration. A test that is green or red depending on how you
invoked it cannot serve as a pre-launch gate, so the isolation belongs in one
shared place rather than in each module.

Tests that need a request set it explicitly inside the test and restore it.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _isolate_tokenizer_request():
    saved = os.environ.pop("NANO_NATIVE_TOKENIZER", None)
    try:
        yield
    finally:
        os.environ.pop("NANO_NATIVE_TOKENIZER", None)
        if saved is not None:
            os.environ["NANO_NATIVE_TOKENIZER"] = saved
