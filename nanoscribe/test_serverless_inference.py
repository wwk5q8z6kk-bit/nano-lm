# Serverless inference pins — no live RunPod calls in CI.
# Run: pytest nanoscribe/test_serverless_inference.py -q
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoscribe.adapters import ModelAdapter, ServerlessQwen38Adapter, default_baseline_specs
from nanoscribe.serverless_inference import (
    endpoint_native_urls,
    endpoint_openai_url,
    parse_endpoint_id,
)
from nanoscribe.test_adapt import _model_input


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

    with patch("nanoscribe.serverless_inference._openai_client", return_value=mock_client):
        adapter = ServerlessQwen38Adapter(endpoint_id="tbnur4mac60i70")
        batch = adapter.propose(_model_input(), (default_baseline_specs()[0],))

    assert len(batch.atoms) == 1
    assert batch.atoms[0].quotes == ("neck",)
    assert batch.memory_bytes == 0
