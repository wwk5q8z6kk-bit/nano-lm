"""External API teacher authorization pins — no live API calls."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import hmac
import os
import sys
from types import ModuleType
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoscribe.adapters import default_baseline_specs
from nanoscribe.api_teacher import (
    OPENAI_API_BASE_URL,
    _openai_client,
    generate_api_span_port_lines,
)
from nanoscribe.encounter import Source, Speaker, Turn, assemble_source
from nanoscribe.egress import (
    EgressAuthorizationError,
    EGRESS_AUTHORIZATION_HMAC_KEY_ENV,
    EgressDataClassification,
    ExternalEgressAuthorization,
    ExternalEgressTarget,
    source_content_sha256,
)
from nanoscribe.test_adapt import _model_input

_TEST_AUTHORIZATION_KEY = "test-only-egress-authorization-key"


def _sign(authorization: ExternalEgressAuthorization) -> ExternalEgressAuthorization:
    return replace(
        authorization,
        signature=hmac.new(
            _TEST_AUTHORIZATION_KEY.encode("utf-8"),
            authorization.signing_payload(),
            hashlib.sha256,
        ).hexdigest(),
    )


def _authorized_model_input():
    model_input = _model_input()
    unsigned = ExternalEgressAuthorization(
        data_classification=EgressDataClassification.NON_PHI_AUTHORIZED,
        source_provenance_id="synthetic-fixture-v1",
        run_provenance_id="test-api-teacher",
        source_id=model_input.source.source_id,
        source_sha256=source_content_sha256(model_input.source),
        authorized_targets=frozenset({ExternalEgressTarget.openai_api()}),
        signature="0" * 64,
    )
    authorization = _sign(unsigned)
    return type(model_input)(
        source=model_input.source,
        encounter_id=model_input.encounter_id,
        external_egress=authorization,
    )


def test_api_teacher_rejects_missing_authorization_before_client() -> None:
    with patch("nanoscribe.api_teacher._openai_client") as openai_client:
        try:
            generate_api_span_port_lines(_model_input(), default_baseline_specs()[:1])
        except EgressAuthorizationError as exc:
            assert exc.code == "authorization_required"
        else:
            raise AssertionError("expected external egress to be denied")

    assert not openai_client.called


def test_api_teacher_pins_the_canonical_openai_host() -> None:
    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = MagicMock()
    with (
        patch.dict(sys.modules, {"openai": fake_openai}),
        patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-key",
                "OPENAI_BASE_URL": "https://untrusted.example/v1",
            },
        ),
    ):
        _openai_client()

    fake_openai.OpenAI.assert_called_once_with(
        api_key="test-key",
        base_url=OPENAI_API_BASE_URL,
    )


def test_api_teacher_requires_authorized_destination_before_client() -> None:
    model_input = _model_input()
    authorization = _sign(
        ExternalEgressAuthorization(
            data_classification=EgressDataClassification.NON_PHI_AUTHORIZED,
            source_provenance_id="synthetic-fixture-v1",
            run_provenance_id="test-api-teacher",
            source_id=model_input.source.source_id,
            source_sha256=source_content_sha256(model_input.source),
            authorized_targets=frozenset(
                {ExternalEgressTarget.runpod_serverless("tbnur4mac60i70")}
            ),
            signature="0" * 64,
        )
    )
    unauthorized_input = type(model_input)(
        source=model_input.source,
        encounter_id=model_input.encounter_id,
        external_egress=authorization,
    )

    with (
        patch.dict(
            os.environ,
            {EGRESS_AUTHORIZATION_HMAC_KEY_ENV: _TEST_AUTHORIZATION_KEY},
        ),
        patch("nanoscribe.api_teacher._openai_client") as openai_client,
    ):
        try:
            generate_api_span_port_lines(
                unauthorized_input,
                default_baseline_specs()[:1],
            )
        except EgressAuthorizationError as exc:
            assert exc.code == "destination_not_authorized"
        else:
            raise AssertionError("expected external egress to be denied")

    assert not openai_client.called


def test_api_teacher_requires_source_bound_provenance() -> None:
    authorized_input = _authorized_model_input()
    changed_source = assemble_source(
        "src-1",
        ((Speaker.PATIENT, "Different source content."),),
    )
    mismatched_input = type(authorized_input)(
        source=changed_source,
        encounter_id=authorized_input.encounter_id,
        external_egress=authorized_input.external_egress,
    )

    with (
        patch.dict(
            os.environ,
            {EGRESS_AUTHORIZATION_HMAC_KEY_ENV: _TEST_AUTHORIZATION_KEY},
        ),
        patch("nanoscribe.api_teacher._openai_client") as openai_client,
    ):
        try:
            generate_api_span_port_lines(
                mismatched_input,
                default_baseline_specs()[:1],
            )
        except EgressAuthorizationError as exc:
            assert exc.code == "source_provenance_mismatch"
            assert "Different source content." not in str(exc)
        else:
            raise AssertionError("expected external egress to be denied")

    assert not openai_client.called


def test_api_teacher_rejects_changed_turns_with_unchanged_source_text() -> None:
    authorized_input = _authorized_model_input()
    original = authorized_input.source
    first_turn = original.turns[0]
    changed_turn = Turn(
        turn_id=first_turn.turn_id,
        source_id=first_turn.source_id,
        speaker=first_turn.speaker,
        start=first_turn.start,
        end=first_turn.start + len("Sensitive replacement."),
        text="Sensitive replacement.",
    )
    mismatched_source = Source(
        source_id=original.source_id,
        text=original.text,
        turns=(changed_turn, *original.turns[1:]),
    )
    mismatched_input = type(authorized_input)(
        source=mismatched_source,
        encounter_id=authorized_input.encounter_id,
        external_egress=authorized_input.external_egress,
    )

    with (
        patch.dict(
            os.environ,
            {EGRESS_AUTHORIZATION_HMAC_KEY_ENV: _TEST_AUTHORIZATION_KEY},
        ),
        patch("nanoscribe.api_teacher._openai_client") as openai_client,
    ):
        try:
            generate_api_span_port_lines(
                mismatched_input,
                default_baseline_specs()[:1],
            )
        except EgressAuthorizationError as exc:
            assert exc.code == "source_provenance_mismatch"
            assert "Sensitive replacement." not in str(exc)
        else:
            raise AssertionError("expected changed outbound transcript to be denied")

    assert not openai_client.called


def test_api_teacher_rejects_unsigned_self_issued_authorization() -> None:
    model_input = _model_input()
    authorization = ExternalEgressAuthorization(
        data_classification=EgressDataClassification.NON_PHI_AUTHORIZED,
        source_provenance_id="invented-source",
        run_provenance_id="invented-run",
        source_id=model_input.source.source_id,
        source_sha256=source_content_sha256(model_input.source),
        authorized_targets=frozenset({ExternalEgressTarget.openai_api()}),
        signature="0" * 64,
    )
    unauthorized_input = type(model_input)(
        source=model_input.source,
        encounter_id=model_input.encounter_id,
        external_egress=authorization,
    )

    with (
        patch.dict(
            os.environ,
            {EGRESS_AUTHORIZATION_HMAC_KEY_ENV: _TEST_AUTHORIZATION_KEY},
        ),
        patch("nanoscribe.api_teacher._openai_client") as openai_client,
    ):
        try:
            generate_api_span_port_lines(
                unauthorized_input,
                default_baseline_specs()[:1],
            )
        except EgressAuthorizationError as exc:
            assert exc.code == "authorization_signature_invalid"
        else:
            raise AssertionError("expected unsigned authorization to be denied")

    assert not openai_client.called


def test_api_teacher_allows_explicit_authorized_egress() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content='STATED: "neck"'))]
    mock_client.chat.completions.create.return_value = mock_response

    with (
        patch.dict(
            os.environ,
            {EGRESS_AUTHORIZATION_HMAC_KEY_ENV: _TEST_AUTHORIZATION_KEY},
        ),
        patch("nanoscribe.api_teacher._openai_client", return_value=mock_client),
    ):
        lines, _, memory_bytes = generate_api_span_port_lines(
            _authorized_model_input(),
            default_baseline_specs()[:1],
        )

    assert lines == {"atom-neck": 'STATED: "neck"'}
    assert memory_bytes == 0
    assert mock_client.chat.completions.create.called
