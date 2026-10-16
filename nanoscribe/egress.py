"""Typed authorization gate for sending encounter sources to external services."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import string
from dataclasses import dataclass
from enum import Enum

from nanoscribe.encounter import Source
from nanoscribe.prompt import format_transcript

EGRESS_AUTHORIZATION_HMAC_KEY_ENV = "NANOSCRIBE_EGRESS_AUTHORIZATION_HMAC_KEY"


class EgressDataClassification(str, Enum):
    """Only classifications allowed to leave the local process."""

    NON_PHI_AUTHORIZED = "non_phi_authorized"


class ExternalEgressDestination(str, Enum):
    """External services that may receive model prompts."""

    OPENAI_API = "openai_api"
    RUNPOD_SERVERLESS = "runpod_serverless"


@dataclass(frozen=True, slots=True)
class ExternalEgressTarget:
    """One exact external service target approved for source egress."""

    destination: ExternalEgressDestination
    host: str
    endpoint_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.destination, ExternalEgressDestination):
            raise TypeError("destination must be an ExternalEgressDestination")
        if self.destination is ExternalEgressDestination.OPENAI_API:
            if self.host != "api.openai.com" or self.endpoint_id is not None:
                raise TypeError(
                    "OpenAI API authorization must target api.openai.com without endpoint_id"
                )
            return
        if self.host != "api.runpod.ai":
            raise TypeError("RunPod authorization must target api.runpod.ai")
        _require_endpoint_id(self.endpoint_id)

    @classmethod
    def openai_api(cls) -> ExternalEgressTarget:
        """Build the canonical OpenAI API target."""
        return cls(
            destination=ExternalEgressDestination.OPENAI_API,
            host="api.openai.com",
        )

    @classmethod
    def runpod_serverless(cls, endpoint_id: str) -> ExternalEgressTarget:
        """Build an exact RunPod Serverless endpoint target."""
        return cls(
            destination=ExternalEgressDestination.RUNPOD_SERVERLESS,
            host="api.runpod.ai",
            endpoint_id=endpoint_id,
        )


class EgressAuthorizationError(PermissionError):
    """A fail-closed denial that deliberately excludes source content."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"external transcript egress denied [{code}]")


def source_content_sha256(source: Source) -> str:
    """Hash the exact transcript serialization sent in a span-port prompt."""
    return hashlib.sha256(format_transcript(source).encode("utf-8")).hexdigest()


def _require_identifier(value: object, field: str) -> None:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise TypeError(f"{field} must be a non-empty, edge-trimmed string")


def _require_endpoint_id(value: object) -> None:
    if not isinstance(value, str) or not value.isascii() or not value.isalnum():
        raise TypeError("endpoint_id must be a non-empty ASCII alphanumeric string")


@dataclass(frozen=True, slots=True)
class ExternalEgressAuthorization:
    """Explicit, source-bound authorization for one external inference run.

    ``source_provenance_id`` and ``run_provenance_id`` must refer to the
    approved dataset/manifest and run record outside the transcript. The
    source ID and digest bind authorization to the exact transcript
    serialization sent in span-port prompts.
    """

    data_classification: EgressDataClassification
    source_provenance_id: str
    run_provenance_id: str
    source_id: str
    source_sha256: str
    authorized_targets: frozenset[ExternalEgressTarget]
    signature: str

    def __post_init__(self) -> None:
        if not isinstance(self.data_classification, EgressDataClassification):
            raise TypeError("data_classification must be an EgressDataClassification")
        _require_identifier(self.source_provenance_id, "source_provenance_id")
        _require_identifier(self.run_provenance_id, "run_provenance_id")
        _require_identifier(self.source_id, "source_id")
        if (
            not isinstance(self.source_sha256, str)
            or len(self.source_sha256) != 64
            or any(char not in string.hexdigits for char in self.source_sha256)
        ):
            raise TypeError("source_sha256 must be a SHA-256 hex digest")
        if not isinstance(self.authorized_targets, frozenset) or not self.authorized_targets:
            raise TypeError("authorized_targets must be a non-empty frozenset")
        if not all(
            isinstance(target, ExternalEgressTarget)
            for target in self.authorized_targets
        ):
            raise TypeError(
                "authorized_targets must contain ExternalEgressTarget values"
            )
        if (
            not isinstance(self.signature, str)
            or len(self.signature) != 64
            or any(char not in string.hexdigits for char in self.signature)
        ):
            raise TypeError("signature must be an HMAC-SHA256 hex digest")

    def signing_payload(self) -> bytes:
        """Serialize authorization claims for external approval verification."""
        return json.dumps(
            {
                "authorized_targets": [
                    {
                        "destination": target.destination.value,
                        "endpoint_id": target.endpoint_id,
                        "host": target.host,
                    }
                    for target in sorted(
                        self.authorized_targets,
                        key=lambda item: (
                            item.destination.value,
                            item.host,
                            item.endpoint_id or "",
                        ),
                    )
                ],
                "data_classification": self.data_classification.value,
                "run_provenance_id": self.run_provenance_id,
                "source_id": self.source_id,
                "source_provenance_id": self.source_provenance_id,
                "source_sha256": self.source_sha256,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    def _require_valid_signature(self) -> None:
        key = os.environ.get(EGRESS_AUTHORIZATION_HMAC_KEY_ENV)
        if not key:
            raise EgressAuthorizationError("authorization_verifier_unavailable")
        expected = hmac.new(
            key.encode("utf-8"),
            self.signing_payload(),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(self.signature, expected):
            raise EgressAuthorizationError("authorization_signature_invalid")

    def require_destination(
        self,
        source: Source,
        target: ExternalEgressTarget,
    ) -> None:
        """Fail closed unless the exact source and destination were authorized."""
        self._require_valid_signature()
        if self.data_classification is not EgressDataClassification.NON_PHI_AUTHORIZED:
            raise EgressAuthorizationError("non_phi_authorization_required")
        if target not in self.authorized_targets:
            raise EgressAuthorizationError("destination_not_authorized")
        if (
            self.source_id != source.source_id
            or self.source_sha256 != source_content_sha256(source)
        ):
            raise EgressAuthorizationError("source_provenance_mismatch")


def require_external_egress(
    model_input: object,
    target: ExternalEgressTarget,
) -> None:
    """Require typed, source-bound authorization before an external API call."""
    authorization = getattr(model_input, "external_egress", None)
    source = getattr(model_input, "source", None)
    if not isinstance(authorization, ExternalEgressAuthorization) or not isinstance(source, Source):
        raise EgressAuthorizationError("authorization_required")
    authorization.require_destination(source, target)
