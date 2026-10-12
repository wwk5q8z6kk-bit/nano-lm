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


# ---------------------------------------------------------------------------
# Tokenizer registry (D3.3)
#
# `hash_tokens` / `detokenize` above are UNCHANGED and remain the default. Every
# result in the four frozen provenance classes was produced through them, so
# altering their behaviour would make those classes irreproducible. The BPE path
# is additive and selected explicitly via `NativeTrainConfig.tokenizer`, which
# rides along in `cfg.to_dict()` and therefore lands in every checkpoint and run
# report — so a result can always be traced to the instrument that produced it.
#
# Measured on the training corpus (artifacts/campaign/d33_prerun_verification_v1.json):
#   hash : median 551 prompt+target tokens, 96/96 over a 512 context
#   bpe  : median 216,                       0/96 over
#   round-trip 192/192 exact, including \n \t \r
# ---------------------------------------------------------------------------

BPE_TOKENIZER_PATH = "sft/tokenizer.json"
_BPE_CACHE: dict[str, object] = {}


def _load_bpe(path: str = BPE_TOKENIZER_PATH):
    cached = _BPE_CACHE.get(path)
    if cached is not None:
        return cached
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "the 'tokenizers' package is required for the bpe tokenizer "
            "(declared in requirements.txt)"
        ) from exc
    from pathlib import Path

    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path(__file__).resolve().parents[2] / path
    if not resolved.is_file():
        raise RuntimeError(f"bpe tokenizer asset not found at {resolved}")
    tok = Tokenizer.from_file(str(resolved))
    _BPE_CACHE[path] = tok
    return tok


def bpe_tokens(text: str, vocab_size: int, max_len: int | None = None) -> list[int]:
    """Own-stack BPE encoding. Same signature as `hash_tokens`.

    `vocab_size` is accepted for interface parity and validated rather than
    used: the asset's vocabulary is fixed at build time, and an id outside the
    model's embedding would be an out-of-bounds lookup at runtime rather than a
    visible error here.
    """
    ids = _load_bpe().encode(text).ids
    if ids and max(ids) >= vocab_size:
        raise ValueError(
            f"bpe produced token id {max(ids)} >= vocab_size {vocab_size}: "
            f"the model's embedding cannot represent it"
        )
    if max_len is not None:
        ids = ids[:max_len]
    return ids or [1]


def bpe_detokenize(token_ids: list[int], vocab_size: int) -> str:
    return _load_bpe().decode(list(token_ids))


# Stored by NAME, not by reference. Binding the functions here would snapshot
# them at import time, so a test that monkeypatches `tokenize.hash_tokens` to
# reconstruct D3.1 would silently stop biting — the registry would keep handing
# out the original. Resolving through module globals keeps every regression test
# that patches a tokenizer meaningful.
_TOKENIZERS = {
    "hash": ("hash_tokens", "detokenize"),
    "bpe": ("bpe_tokens", "bpe_detokenize"),
}


def resolve_tokenizer(kind: str = "hash"):
    """Return (encode, decode) for `kind`. Raises on an unknown name.

    Raising rather than falling back is deliberate: a silent fallback to the
    character tokenizer would produce a run that looks like a BPE run, reports
    itself as one, and is not — the exact shape of every defect in
    artifacts/DEFECT_INDEX.md.
    """
    try:
        enc_name, dec_name = _TOKENIZERS[kind]
    except KeyError:
        raise ValueError(
            f"unknown tokenizer {kind!r}; known: {sorted(_TOKENIZERS)}"
        ) from None
    g = globals()
    return g[enc_name], g[dec_name]
