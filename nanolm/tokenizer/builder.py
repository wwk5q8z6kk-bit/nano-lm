"""Multi-domain Byte-Level BPE Tokenizer builder and compression analyzer for NanoLM."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

SPECIAL_TOKENS: tuple[str, ...] = (
    "<|bos|>",
    "<|eos|>",
    "<|pad|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|think|>",
    "<|end_think|>",
    "<|code|>",
    "<|json|>",
    "<|diagram|>",
    *tuple(f"<|reserved_{i:02d}|>" for i in range(22)),
)
assert len(SPECIAL_TOKENS) == 32


@dataclass(frozen=True, slots=True)
class DomainTokenStats:
    domain: str
    utf8_bytes: int
    tokens: int
    bytes_per_token: float
    tokens_per_1k_bytes: float


@dataclass(frozen=True, slots=True)
class TokenizerAuditReport:
    name: str
    vocab_size: int
    sha256: str
    special_token_count: int
    individual_digit_splitting: bool
    overall_bytes_per_token: float
    domains: dict[str, DomainTokenStats]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "vocab_size": self.vocab_size,
            "sha256": self.sha256,
            "special_token_count": self.special_token_count,
            "individual_digit_splitting": self.individual_digit_splitting,
            "overall_bytes_per_token": self.overall_bytes_per_token,
            "domains": {k: asdict(v) for k, v in self.domains.items()},
        }


def train_nanolm_tokenizer(
    texts: Iterable[str],
    vocab_size: int = 16_384,
    min_frequency: int = 2,
    split_digits: bool = True,
) -> Tokenizer:
    """Train a clean Byte-Level BPE tokenizer with digit splitting and 32 reserved special tokens."""
    if vocab_size < 512:
        raise ValueError("vocab_size must be >= 512")

    tok = Tokenizer(models.BPE(byte_fallback=False))
    steps = []
    if split_digits:
        steps.append(pre_tokenizers.Digits(individual_digits=True))
    steps.append(pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True))
    tok.pre_tokenizer = pre_tokenizers.Sequence(steps)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def save_tokenizer(tok: Tokenizer, path: str | Path) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(target))
    return hashlib.sha256(target.read_bytes()).hexdigest()


def load_tokenizer(path: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def format_chat_messages(messages: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
    """Render ChatML conversation using NanoLM's special tokens."""
    chunks: list[str] = []
    for msg in messages:
        chunks.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
    if add_generation_prompt:
        chunks.append("<|im_start|>assistant\n")
    return "".join(chunks)


def audit_tokenizer_efficiency(
    name: str,
    encode_fn,
    vocab_size: int,
    domain_samples: dict[str, list[str]],
    sha256: str = "external",
) -> TokenizerAuditReport:
    """Measure bytes-per-token across prose, code, math, reasoning, JSON, and diagrams."""
    domain_reports: dict[str, DomainTokenStats] = {}
    total_bytes = 0
    total_tokens = 0

    for domain, samples in domain_samples.items():
        d_bytes = 0
        d_tokens = 0
        for s in samples:
            d_bytes += len(s.encode("utf-8"))
            d_tokens += max(1, len(encode_fn(s)))
        total_bytes += d_bytes
        total_tokens += d_tokens
        bpt = d_bytes / max(1, d_tokens)
        domain_reports[domain] = DomainTokenStats(
            domain=domain,
            utf8_bytes=d_bytes,
            tokens=d_tokens,
            bytes_per_token=round(bpt, 4),
            tokens_per_1k_bytes=round(1000.0 / max(1e-9, bpt), 2),
        )

    digit_probe = encode_fn("9876543210")
    splits_digits = len(digit_probe) >= 10

    return TokenizerAuditReport(
        name=name,
        vocab_size=vocab_size,
        sha256=sha256,
        special_token_count=len(SPECIAL_TOKENS),
        individual_digit_splitting=splits_digits,
        overall_bytes_per_token=round(total_bytes / max(1, total_tokens), 4),
        domains=domain_reports,
    )
