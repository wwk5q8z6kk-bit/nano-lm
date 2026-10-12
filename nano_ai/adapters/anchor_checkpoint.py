"""Integrity-gated adapter for the frozen own-stack scribe checkpoints.

Artifact verification and runtime loading are deliberately lazy.  Importing this
module does not import Torch, tokenizers, or the historical experiment script.
The verified byte snapshots—not subsequently re-opened paths—are handed to the
runtime loader, closing the verification/load race.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import os
import re
import stat
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

from nano_ai.adapters.legacy_summary import LegacySummarySolver
from nano_ai.contract import NanoInput, NanoOutput
from nano_ai.solver import SolverDescriptor, SolverKind

TOKENIZER_SHA256 = "bae49648bfcc4904c50e2f006ee184bd26e74454ee170663e30a8e71640ce3c9"

# These identifiers are part of the observable solver identity.  Changing any
# corresponding behavior requires a new identifier rather than silently
# reusing historical benchmark results.
ANCHOR_PIPELINE_VERSION = "anchor-summary-grounding-v0"
NATIVE_RUNTIME_ID = "ownstack-torch-tokenizers-v0"
PROMPT_TEMPLATE_ID = "chatml-summarize-visit-v0"
DECODE_POLICY_ID = "greedy-argmax-until-im-end-v0"
GROUNDING_VERIFIER_ID = "legacy-summary-evidence-v0"

_SAFE_RUNTIME_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")


class AnchorArtifactError(RuntimeError):
    """Base error for an anchor artifact that cannot be trusted."""


class ArtifactUnavailableError(AnchorArtifactError):
    """A required artifact could not be read as a regular file."""


class ArtifactIntegrityError(AnchorArtifactError):
    """Artifact bytes do not match the registered identity."""


class AnchorRuntimeUnavailableError(RuntimeError):
    """The verified anchor cannot be loaded by the available runtime."""


class AnchorGenerationError(RuntimeError):
    """The anchor runtime could not return a legacy summary."""


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _validate_runtime_id(runtime_id: str) -> str:
    if (
        not isinstance(runtime_id, str)
        or _SAFE_RUNTIME_ID.fullmatch(runtime_id) is None
    ):
        raise ValueError(
            "runtime_id must be 1-128 lowercase ASCII letters, digits, '.', '_', "
            "or '-', beginning with a letter or digit"
        )
    return runtime_id


@dataclass(frozen=True, slots=True)
class AnchorArtifactSpec:
    """Registered content identity and architecture for one frozen anchor."""

    name: str
    release: str
    checkpoint_filename: str
    checkpoint_sha256: str
    tokenizer_sha256: str = TOKENIZER_SHA256
    model_width: int = 192
    layer_count: int = 6
    attention_heads: int = 6
    kv_heads: int = 2
    head_width: int = 32
    feed_forward_width: int = 512
    vocabulary_size: int = 4098
    sequence_length: int = 512

    def __post_init__(self) -> None:
        for label, value in (
            ("name", self.name),
            ("release", self.release),
            ("checkpoint_filename", self.checkpoint_filename),
        ):
            if not value or value.strip() != value:
                raise ValueError(f"{label} must be a non-empty, edge-trimmed string")
        for label, value in (
            ("checkpoint_sha256", self.checkpoint_sha256),
            ("tokenizer_sha256", self.tokenizer_sha256),
        ):
            if not _valid_sha256(value):
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")
        for label, value in (
            ("model_width", self.model_width),
            ("layer_count", self.layer_count),
            ("attention_heads", self.attention_heads),
            ("kv_heads", self.kv_heads),
            ("head_width", self.head_width),
            ("feed_forward_width", self.feed_forward_width),
            ("vocabulary_size", self.vocabulary_size),
            ("sequence_length", self.sequence_length),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if self.attention_heads % self.kv_heads:
            raise ValueError("attention_heads must be divisible by kv_heads")
        if self.attention_heads * self.head_width != self.model_width:
            raise ValueError("attention geometry must equal model_width")

    @property
    def parameter_count(self) -> int:
        d = self.model_width
        attention = (
            d * self.attention_heads * self.head_width
            + 2 * d * self.kv_heads * self.head_width
            + d * self.attention_heads * self.head_width
        )
        feed_forward = 3 * d * self.feed_forward_width
        block_norms = 2 * d
        return (
            self.vocabulary_size * d
            + self.layer_count * (attention + feed_forward + block_norms)
            + d
        )

    @property
    def artifact_identity(self) -> str:
        return (
            f"{self.name}@{self.release}"
            f";checkpoint=sha256:{self.checkpoint_sha256}"
            f";tokenizer=sha256:{self.tokenizer_sha256}"
        )

    @property
    def solver_id(self) -> str:
        # Full hashes keep identity exact even when filenames or release labels collide.
        return (
            f"ownstack/{self.name}@{self.release}"
            f"/ckpt-{self.checkpoint_sha256}"
            f"/tok-{self.tokenizer_sha256}"
        )

    @property
    def architecture_identity(self) -> str:
        """Exact native model configuration not encoded in checkpoint bytes."""

        return (
            f"d{self.model_width}-l{self.layer_count}"
            f"-h{self.attention_heads}-kv{self.kv_heads}-hd{self.head_width}"
            f"-ff{self.feed_forward_width}-v{self.vocabulary_size}"
            f"-ctx{self.sequence_length}"
        )


FROZEN_NANO_V01 = AnchorArtifactSpec(
    name="nano-scribe",
    release="v0.1",
    # Keep the frozen release anchor separate from the later E4 base that
    # historically reused the local filename ``scribe.pt``.
    checkpoint_filename="nano_v01_scribe.pt",
    checkpoint_sha256="0e4f348eea00c660236cfd9e5bc2d9a71274adfc4d738db6f664664c9a06725b",
)

FROZEN_SCALE_V01 = AnchorArtifactSpec(
    name="scale10m-scribe",
    release="v0.1",
    checkpoint_filename="scale10m_scribe.pt",
    checkpoint_sha256="f5aca5f04bd1045cc158d46a27b84024bb94baa349ed330933631c8b8d5acf0d",
    model_width=320,
    layer_count=8,
    attention_heads=8,
    kv_heads=2,
    head_width=40,
    feed_forward_width=864,
)

ANCHOR_SPECS = {
    "nano": FROZEN_NANO_V01,
    "scale": FROZEN_SCALE_V01,
}


@dataclass(frozen=True, slots=True)
class VerifiedAnchorArtifacts:
    """One immutable, hash-verified checkpoint/tokenizer snapshot."""

    spec: AnchorArtifactSpec
    checkpoint_path: Path
    tokenizer_path: Path
    checkpoint_bytes: bytes = field(repr=False)
    tokenizer_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.spec, AnchorArtifactSpec):
            raise TypeError("spec must be an AnchorArtifactSpec")
        for role, path in (
            ("checkpoint", self.checkpoint_path),
            ("tokenizer", self.tokenizer_path),
        ):
            if not isinstance(path, Path):
                raise TypeError(f"{role}_path must be a Path")
        for role, payload, expected in (
            ("checkpoint", self.checkpoint_bytes, self.spec.checkpoint_sha256),
            ("tokenizer", self.tokenizer_bytes, self.spec.tokenizer_sha256),
        ):
            # A mutable bytearray or memoryview would invalidate the meaning of
            # a verified snapshot after construction.
            if type(payload) is not bytes:
                raise TypeError(f"{role}_bytes must be immutable exact bytes")
            observed = hashlib.sha256(payload).hexdigest()
            if not hmac.compare_digest(observed, expected):
                raise ArtifactIntegrityError(
                    f"{role} SHA-256 mismatch for {self.spec.artifact_identity}: "
                    f"expected {expected}, observed {observed}"
                )

    @property
    def artifact_bytes(self) -> int:
        return len(self.checkpoint_bytes) + len(self.tokenizer_bytes)

    @property
    def artifact_identity(self) -> str:
        return self.spec.artifact_identity


def _read_regular_file(path: Path, *, role: str) -> bytes:
    try:
        with path.open("rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise ArtifactUnavailableError(f"{role} artifact is not a regular file")
            return handle.read()
    except ArtifactUnavailableError:
        raise
    except OSError as exc:
        raise ArtifactUnavailableError(f"{role} artifact is unavailable") from exc


def _verified_bytes(
    path: Path, expected_sha256: str, *, role: str, identity: str
) -> bytes:
    payload = _read_regular_file(path, role=role)
    observed = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(observed, expected_sha256):
        raise ArtifactIntegrityError(
            f"{role} SHA-256 mismatch for {identity}: expected "
            f"{expected_sha256}, observed {observed}; artifact was not loaded"
        )
    return payload


def verify_anchor_artifacts(
    spec: AnchorArtifactSpec,
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
) -> VerifiedAnchorArtifacts:
    """Read and verify both artifacts before any model dependency is loaded."""

    checkpoint = Path(checkpoint_path)
    tokenizer = Path(tokenizer_path)
    checkpoint_bytes = _verified_bytes(
        checkpoint,
        spec.checkpoint_sha256,
        role="checkpoint",
        identity=spec.artifact_identity,
    )
    tokenizer_bytes = _verified_bytes(
        tokenizer,
        spec.tokenizer_sha256,
        role="tokenizer",
        identity=spec.artifact_identity,
    )
    return VerifiedAnchorArtifacts(
        spec=spec,
        checkpoint_path=checkpoint,
        tokenizer_path=tokenizer,
        checkpoint_bytes=checkpoint_bytes,
        tokenizer_bytes=tokenizer_bytes,
    )


class SummaryRuntime(Protocol):
    def generate(self, transcript: str) -> str: ...


class RuntimeLoader(Protocol):
    def __call__(
        self,
        artifacts: VerifiedAnchorArtifacts,
        *,
        device: str,
        max_new_tokens: int,
    ) -> SummaryRuntime: ...


def _native_runtime_loader(
    artifacts: VerifiedAnchorArtifacts,
    *,
    device: str,
    max_new_tokens: int,
) -> SummaryRuntime:
    """Build the historical native runtime only after integrity verification."""

    try:
        import torch
        from tokenizers import Tokenizer
        from torch import nn
        from torch.nn import functional as torch_functional
    except ImportError as exc:
        raise AnchorRuntimeUnavailableError(
            "Torch and tokenizers are required to load an own-stack anchor"
        ) from exc

    spec = artifacts.spec
    sequence_length = spec.sequence_length

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            d = spec.model_width
            h = spec.attention_heads
            kv = spec.kv_heads
            hd = spec.head_width
            ff = spec.feed_forward_width
            self.h = h
            self.kv = kv
            self.hd = hd
            # Module names match the frozen checkpoints' native state_dict keys.
            self.n1 = nn.RMSNorm(d)
            self.n2 = nn.RMSNorm(d)
            self.q = nn.Linear(d, h * hd, bias=False)
            self.k = nn.Linear(d, kv * hd, bias=False)
            self.v = nn.Linear(d, kv * hd, bias=False)
            self.o = nn.Linear(h * hd, d, bias=False)
            self.g = nn.Linear(d, ff, bias=False)
            self.u = nn.Linear(d, ff, bias=False)
            self.dn = nn.Linear(ff, d, bias=False)

        def forward(self, values, cosine, sine):
            batch = values.shape[0]
            hidden = self.n1(values)
            query = (
                self.q(hidden)
                .view(batch, sequence_length, self.h, self.hd)
                .transpose(1, 2)
            )
            key = (
                self.k(hidden)
                .view(batch, sequence_length, self.kv, self.hd)
                .transpose(1, 2)
            )
            value = (
                self.v(hidden)
                .view(batch, sequence_length, self.kv, self.hd)
                .transpose(1, 2)
            )

            def rotate(tensor):
                first, second = tensor[..., 0::2], tensor[..., 1::2]
                return torch.stack(
                    [first * cosine - second * sine, first * sine + second * cosine],
                    dim=-1,
                ).flatten(-2)

            query, key = rotate(query), rotate(key)
            key = key.repeat_interleave(self.h // self.kv, 1)
            value = value.repeat_interleave(self.h // self.kv, 1)
            attention = torch_functional.scaled_dot_product_attention(
                query, key, value, is_causal=True
            )
            values = values + self.o(
                attention.transpose(1, 2).reshape(
                    batch, sequence_length, self.h * self.hd
                )
            )
            hidden = self.n2(values)
            return values + self.dn(
                torch_functional.silu(self.g(hidden)) * self.u(hidden)
            )

    class _GPT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hd = spec.head_width
            self.emb = nn.Embedding(spec.vocabulary_size, spec.model_width)
            self.blocks = nn.ModuleList(_Block() for _ in range(spec.layer_count))
            self.nf = nn.RMSNorm(spec.model_width)

        def _rope(self, target_device):
            positions = torch.arange(
                sequence_length, device=target_device, dtype=torch.float32
            )
            inverse = 1.0 / (
                10000
                ** (
                    torch.arange(0, spec.head_width, 2, device=target_device).float()
                    / spec.head_width
                )
            )
            frequency = torch.outer(positions, inverse)
            return frequency.cos()[None, None], frequency.sin()[None, None]

        def forward(self, token_ids):
            cosine, sine = self._rope(token_ids.device)
            hidden = self.emb(token_ids)
            for block in self.blocks:
                hidden = block(hidden, cosine, sine)
            return torch_functional.linear(self.nf(hidden), self.emb.weight)

    try:
        tokenizer_text = artifacts.tokenizer_bytes.decode("utf-8")
        tokenizer = Tokenizer.from_str(tokenizer_text)
        start_token = tokenizer.token_to_id("<|im_start|>")
        end_token = tokenizer.token_to_id("<|im_end|>")
        if start_token is None or end_token is None:
            raise ValueError("tokenizer is missing required ChatML tokens")

        state_dict = torch.load(
            io.BytesIO(artifacts.checkpoint_bytes),
            map_location="cpu",
            weights_only=True,
        )
        model = _GPT()
        model.load_state_dict(state_dict, strict=True)
        model.to(device).eval()
    except Exception as exc:
        raise AnchorRuntimeUnavailableError(
            "verified own-stack anchor could not be initialized"
        ) from exc

    class _NativeRuntime:
        def generate(self, transcript: str) -> str:
            prompt = transcript.rstrip()
            if not prompt.endswith("Summarize the visit."):
                prompt += "\nSummarize the visit."
            prompt_ids = (
                [start_token]
                + tokenizer.encode("user\n", add_special_tokens=False).ids
                + tokenizer.encode(prompt, add_special_tokens=False).ids
                + [end_token]
                + [start_token]
                + tokenizer.encode("assistant\n", add_special_tokens=False).ids
            )
            if len(prompt_ids) >= sequence_length:
                raise AnchorGenerationError(
                    "transcript exceeds the anchor's native context window"
                )
            generated = list(prompt_ids)
            try:
                with torch.no_grad():
                    for _ in range(max_new_tokens):
                        if len(generated) >= sequence_length:
                            break
                        padded = generated + [0] * (sequence_length - len(generated))
                        tensor = torch.tensor([padded], device=device)
                        next_token = int(
                            model(tensor)[0, len(generated) - 1].argmax().item()
                        )
                        if next_token == end_token:
                            break
                        generated.append(next_token)
                return tokenizer.decode(generated[len(prompt_ids) :]).strip()
            except AnchorGenerationError:
                raise
            except Exception as exc:
                raise AnchorGenerationError("own-stack generation failed") from exc

    return _NativeRuntime()


class AnchorSummaryGenerator:
    """Lazy legacy-summary generator backed by one exact anchor identity."""

    def __init__(
        self,
        spec: AnchorArtifactSpec,
        checkpoint_path: str | Path,
        tokenizer_path: str | Path,
        *,
        runtime_loader: RuntimeLoader | None = None,
        runtime_id: str | None = None,
        device: str = "cpu",
        max_new_tokens: int = 64,
    ) -> None:
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise TypeError("max_new_tokens must be a positive integer")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        if (
            not isinstance(device, str)
            or not device
            or device.strip() != device
            or device.casefold() == "auto"
        ):
            raise ValueError(
                "device must be an explicit, non-empty, edge-trimmed backend; "
                "use 'cpu' for the reproducible default"
            )
        if runtime_loader is None:
            if runtime_id is not None:
                raise ValueError(
                    "runtime_id is reserved for an explicitly injected runtime_loader"
                )
            resolved_runtime_id = NATIVE_RUNTIME_ID
            runtime_kind = SolverKind.HYBRID
        else:
            if not callable(runtime_loader):
                raise TypeError("runtime_loader must be callable")
            if runtime_id is None:
                raise ValueError(
                    "runtime_id is required when runtime_loader is explicitly injected"
                )
            resolved_runtime_id = _validate_runtime_id(runtime_id)
            runtime_kind = SolverKind.LEGACY_ADAPTER
        self.spec = spec
        self.checkpoint_path = Path(checkpoint_path)
        self.tokenizer_path = Path(tokenizer_path)
        self._runtime_loader = (
            _native_runtime_loader if runtime_loader is None else runtime_loader
        )
        self.runtime_id = resolved_runtime_id
        self.runtime_kind = runtime_kind
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._runtime: SummaryRuntime | None = None
        self._runtime_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._runtime is not None

    @property
    def artifact_identity(self) -> str:
        return self.spec.artifact_identity

    @property
    def device_identity(self) -> str:
        # Device/backend selection can affect floating-point generation, so the
        # explicit configured choice is part of solver provenance.
        return quote(self._device, safe="._-")

    @property
    def max_new_tokens(self) -> int:
        return self._max_new_tokens

    def verify(self) -> VerifiedAnchorArtifacts:
        return verify_anchor_artifacts(
            self.spec, self.checkpoint_path, self.tokenizer_path
        )

    def _get_runtime(self) -> SummaryRuntime:
        if self._runtime is not None:
            return self._runtime
        with self._runtime_lock:
            if self._runtime is not None:
                return self._runtime
            artifacts = self.verify()
            try:
                runtime = self._runtime_loader(
                    artifacts,
                    device=self._device,
                    max_new_tokens=self._max_new_tokens,
                )
            except AnchorArtifactError:
                raise
            except AnchorRuntimeUnavailableError:
                raise
            except (ImportError, ModuleNotFoundError) as exc:
                raise AnchorRuntimeUnavailableError(
                    "own-stack runtime dependencies are unavailable"
                ) from exc
            except Exception as exc:
                raise AnchorRuntimeUnavailableError(
                    "verified own-stack anchor could not be initialized"
                ) from exc
            if not callable(getattr(runtime, "generate", None)):
                raise AnchorRuntimeUnavailableError(
                    "runtime loader did not return a summary generator"
                )
            self._runtime = runtime
            return runtime

    def generate(self, transcript: str) -> str:
        if not isinstance(transcript, str) or not transcript.strip():
            raise AnchorGenerationError("transcript must contain text")
        try:
            summary = self._get_runtime().generate(transcript)
        except (
            AnchorArtifactError,
            AnchorRuntimeUnavailableError,
            AnchorGenerationError,
        ):
            raise
        except Exception as exc:
            raise AnchorGenerationError("own-stack generation failed") from exc
        if not isinstance(summary, str) or not summary.strip():
            raise AnchorGenerationError("own-stack runtime returned no summary")
        return summary.strip()


class AnchorCheckpointSolver:
    """NanoSolver-compatible, evidence-grounded own-stack checkpoint adapter."""

    def __init__(
        self,
        spec: AnchorArtifactSpec,
        checkpoint_path: str | Path,
        tokenizer_path: str | Path,
        *,
        runtime_loader: RuntimeLoader | None = None,
        runtime_id: str | None = None,
        device: str = "cpu",
        max_new_tokens: int = 64,
    ) -> None:
        self.generator = AnchorSummaryGenerator(
            spec,
            checkpoint_path,
            tokenizer_path,
            runtime_loader=runtime_loader,
            runtime_id=runtime_id,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        runtime_identity = self.generator.runtime_id
        if self.generator.runtime_kind is SolverKind.LEGACY_ADAPTER:
            runtime_identity = f"injected-{runtime_identity}"
        solver_id = (
            f"{spec.solver_id}"
            f"/arch-{spec.architecture_identity}"
            f"/pipeline-{ANCHOR_PIPELINE_VERSION}"
            f"/runtime-{runtime_identity}"
            f"/prompt-{PROMPT_TEMPLATE_ID}"
            f"/decode-{DECODE_POLICY_ID}"
            f"/max-new-{self.generator.max_new_tokens}"
            f"/grounding-{GROUNDING_VERIFIER_ID}"
            f"/device-{self.generator.device_identity}"
        )
        self.descriptor = SolverDescriptor(
            solver_id=solver_id,
            kind=self.generator.runtime_kind,
            version=ANCHOR_PIPELINE_VERSION,
            # An injected runtime may ignore or reinterpret the registered
            # checkpoint. Do not attribute its model size to that adapter.
            parameter_count=(
                spec.parameter_count
                if self.generator.runtime_kind is SolverKind.HYBRID
                else None
            ),
            # Report bytes only from a VerifiedAnchorArtifacts instance; a local
            # mismatched file must never be attributed to the registered anchor.
            artifact_bytes=None,
        )
        self._grounding_adapter = LegacySummarySolver(
            self.generator.generate,
            solver_id=self.descriptor.solver_id,
            version=self.descriptor.version,
            parameter_count=self.descriptor.parameter_count,
        )

    @classmethod
    def from_repository(
        cls,
        tag: str,
        *,
        repository_root: str | Path | None = None,
        runtime_loader: RuntimeLoader | None = None,
        runtime_id: str | None = None,
        device: str = "cpu",
        max_new_tokens: int = 64,
    ) -> AnchorCheckpointSolver:
        try:
            spec = ANCHOR_SPECS[tag]
        except KeyError as exc:
            raise ValueError(f"unknown anchor tag: {tag}") from exc
        root = (
            Path(repository_root)
            if repository_root is not None
            else Path(__file__).resolve().parents[2]
        )
        return cls(
            spec,
            root / "checkpoints" / "anchors" / spec.checkpoint_filename,
            root / "sft" / "tokenizer.json",
            runtime_loader=runtime_loader,
            runtime_id=runtime_id,
            device=device,
            max_new_tokens=max_new_tokens,
        )

    def infer(self, request: NanoInput) -> NanoOutput:
        return self._grounding_adapter.infer(request)

    def infer_with_diagnostics(
        self, request: NanoInput
    ) -> tuple[NanoOutput, dict[str, object]]:
        """Expose the grounding trace without changing pipeline identity."""

        return self._grounding_adapter.infer_with_diagnostics(request)


__all__ = [
    "ANCHOR_PIPELINE_VERSION",
    "ANCHOR_SPECS",
    "DECODE_POLICY_ID",
    "FROZEN_NANO_V01",
    "FROZEN_SCALE_V01",
    "GROUNDING_VERIFIER_ID",
    "NATIVE_RUNTIME_ID",
    "PROMPT_TEMPLATE_ID",
    "TOKENIZER_SHA256",
    "AnchorArtifactError",
    "AnchorArtifactSpec",
    "AnchorCheckpointSolver",
    "AnchorGenerationError",
    "AnchorRuntimeUnavailableError",
    "AnchorSummaryGenerator",
    "ArtifactIntegrityError",
    "ArtifactUnavailableError",
    "RuntimeLoader",
    "SummaryRuntime",
    "VerifiedAnchorArtifacts",
    "verify_anchor_artifacts",
]
