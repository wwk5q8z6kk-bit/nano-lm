# Serverless inference pins — no live RunPod calls in CI.
# Run: pytest nanoscribe/test_serverless_inference.py -q
from __future__ import annotations

from dataclasses import replace
import hashlib
import hmac
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoscribe.adapters import ModelAdapter, ServerlessQwen38Adapter, default_baseline_specs
from nanoscribe.adapt import AdapterExecutionMode, ModelInput
from nanoscribe.egress import (
    EgressAuthorizationError,
    EGRESS_AUTHORIZATION_HMAC_KEY_ENV,
    EgressDataClassification,
    ExternalEgressAuthorization,
    ExternalEgressTarget,
    source_content_sha256,
)
from nanoscribe.serverless_inference import (
    endpoint_native_urls,
    endpoint_openai_url,
    parse_endpoint_id,
)
from nanoscribe.test_adapt import _model_input

_TEST_AUTHORIZATION_KEY = "test-only-egress-authorization-key"


def _authorized_model_input() -> ModelInput:
    model_input = _model_input()
    unsigned = ExternalEgressAuthorization(
        data_classification=EgressDataClassification.NON_PHI_AUTHORIZED,
        source_provenance_id="synthetic-fixture-v1",
        run_provenance_id="test-serverless-inference",
        source_id=model_input.source.source_id,
        source_sha256=source_content_sha256(model_input.source),
        authorized_targets=frozenset(
            {ExternalEgressTarget.runpod_serverless("tbnur4mac60i70")}
        ),
        signature="0" * 64,
    )
    authorization = replace(
        unsigned,
        signature=hmac.new(
            _TEST_AUTHORIZATION_KEY.encode("utf-8"),
            unsigned.signing_payload(),
            hashlib.sha256,
        ).hexdigest(),
    )
    return ModelInput(
        source=model_input.source,
        encounter_id=model_input.encounter_id,
        external_egress=authorization,
    )


def test_parse_endpoint_id_bare() -> None:
    assert parse_endpoint_id("tbnur4mac60i70") == "tbnur4mac60i70"


def test_parse_endpoint_id_url() -> None:
    url = "https://api.runpod.ai/v2/tbnur4mac60i70/openai/v1"
    assert parse_endpoint_id(url) == "tbnur4mac60i70"


def test_endpoint_urls() -> None:
    endpoint = "tbnur4mac60i70"
    assert endpoint_openai_url(endpoint).endswith("/openai/v1")
    urls = endpoint_native_urls(endpoint)
    assert urls["health"].endswith("/health")
    assert urls["openai_v1"].endswith("/openai/v1")


def test_serverless_adapter_implements_protocol() -> None:
    adapter = ServerlessQwen38Adapter(endpoint_id="tbnur4mac60i70")
    assert isinstance(adapter, ModelAdapter)


def test_serverless_adapter_propose_mocked() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content='STATED: "neck"'))]
    mock_client.chat.completions.create.return_value = mock_response

    with (
        patch.dict(
            os.environ,
            {EGRESS_AUTHORIZATION_HMAC_KEY_ENV: _TEST_AUTHORIZATION_KEY},
        ),
        patch(
            "nanoscribe.serverless_inference._openai_client",
            return_value=mock_client,
        ),
    ):
        adapter = ServerlessQwen38Adapter(endpoint_id="tbnur4mac60i70")
        batch = adapter.propose(
            _authorized_model_input(),
            (default_baseline_specs()[0],),
        )

    assert len(batch.atoms) == 1
    assert batch.atoms[0].quotes == ("neck",)
    assert batch.memory_bytes == 0
    assert batch.execution_mode is AdapterExecutionMode.EXTERNAL_API


def test_serverless_rejects_missing_egress_authorization_before_client() -> None:
    with patch("nanoscribe.serverless_inference._openai_client") as openai_client:
        adapter = ServerlessQwen38Adapter(endpoint_id="tbnur4mac60i70")
        try:
            adapter.propose(_model_input(), (default_baseline_specs()[0],))
        except EgressAuthorizationError as exc:
            assert exc.code == "authorization_required"
        else:
            raise AssertionError("expected external egress to be denied")

    assert not openai_client.called


def test_serverless_rejects_unapproved_endpoint_before_client() -> None:
    with (
        patch.dict(
            os.environ,
            {EGRESS_AUTHORIZATION_HMAC_KEY_ENV: _TEST_AUTHORIZATION_KEY},
        ),
        patch("nanoscribe.serverless_inference._openai_client") as openai_client,
    ):
        adapter = ServerlessQwen38Adapter(endpoint_id="otherendpoint123")
        try:
            adapter.propose(
                _authorized_model_input(),
                (default_baseline_specs()[0],),
            )
        except EgressAuthorizationError as exc:
            assert exc.code == "destination_not_authorized"
        else:
            raise AssertionError("expected external egress to be denied")

    assert not openai_client.called


def test_serverless_rejects_non_runpod_base_url_before_client() -> None:
    with (
        patch.dict(
            os.environ,
            {EGRESS_AUTHORIZATION_HMAC_KEY_ENV: _TEST_AUTHORIZATION_KEY},
        ),
        patch("nanoscribe.serverless_inference._openai_client") as openai_client,
    ):
        adapter = ServerlessQwen38Adapter(
            endpoint_id="tbnur4mac60i70",
            base_url="https://untrusted.example/v2/tbnur4mac60i70/openai/v1",
        )
        try:
            adapter.propose(
                _authorized_model_input(),
                (default_baseline_specs()[0],),
            )
        except ValueError as exc:
            assert "canonical api.runpod.ai HTTPS" in str(exc)
        else:
            raise AssertionError("expected non-RunPod endpoint to be denied")

    assert not openai_client.called
