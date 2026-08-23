"""Hash tokenization for native Nano — shared by training, inference, and eval."""

from __future__ import annotations


def hash_tokens(text: str, vocab_size: int) -> list[int]:
    tokens: list[int] = []
    for ch in text[:64]:
        tokens.append((ord(ch) % (vocab_size - 1)) + 1)
    if not tokens:
        tokens = [1]
    return tokens


def _char_for_token(token_id: int, vocab_size: int) -> str:
    for code in range(32, 127):
        ch = chr(code)
        if (ord(ch) % (vocab_size - 1)) + 1 == token_id:
            return ch
    return "?"


def detokenize(token_ids: list[int], vocab_size: int) -> str:
    return "".join(_char_for_token(t, vocab_size) for t in token_ids if t > 0)
