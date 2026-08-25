"""Hash tokenization for native Nano — shared by training, inference, and eval."""

from __future__ import annotations


def hash_tokens(text: str, vocab_size: int, max_len: int | None = None) -> list[int]:
    """Map text to token ids as (ord(ch) % (vocab_size - 1)) + 1.

    Historically this hard-truncated to text[:64], which silently discarded ~88%
    of every span-port prompt (prompts are ~530 chars) and defeated the callers'
    own `[: cfg.max_seq]` cap of 512. With a 64-character window the gold value
    frequently fell outside the model's view entirely, making exact span emission
    impossible by construction. Truncation is now the caller's decision:
    `max_len=None` means no cap, and every call site already applies cfg.max_seq.
    """
    if max_len is not None:
        text = text[:max_len]
    tokens = [(ord(ch) % (vocab_size - 1)) + 1 for ch in text]
    if not tokens:
        tokens = [1]
    return tokens


def _char_for_token(token_id: int, vocab_size: int) -> str:
    """Exact inverse of hash_tokens for codepoints below the modulus.

    token_id = (ord(ch) % (vocab_size - 1)) + 1, so for ord(ch) < vocab_size - 1
    the inverse is simply chr(token_id - 1). The previous implementation scanned
    only range(32, 127), so '\\n', '\\t' and '\\r' had no preimage and decoded to
    '?' — corrupting the turn separators in 100% of corpus sources.
    """
    code = token_id - 1
    if 0 <= code < vocab_size - 1:
        try:
            return chr(code)
        except ValueError:  # pragma: no cover - defensive
            return "?"
    return "?"


def detokenize(token_ids: list[int], vocab_size: int) -> str:
    return "".join(_char_for_token(t, vocab_size) for t in token_ids if t > 0)
