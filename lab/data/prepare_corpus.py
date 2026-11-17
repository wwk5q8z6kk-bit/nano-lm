"""NanoLM discovery-lab corpus builder (LAB-DATA-01).

Builds a reproducible multi-domain pretraining corpus for small-scale
architecture experiments:

    fineweb-edu (sample-10BT)  -> general educational prose
    cosmopedia-v2              -> synthetic textbook prose
    finemath-4plus             -> mathematical text
    github-code-clean          -> code (permissive licences)

Outputs (under --out):
    tokenizer_32k.json         32,768-vocab byte-level BPE, digit-split
    train.bin / val_<domain>.bin   uint16 token streams, docs joined by <|eos|>
    manifest.json              provenance, sha256, token counts

Validation documents are drawn from different source shards than training
documents, and any val document whose 13-gram set overlaps training is
dropped (document-level split, not token-level).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

SPECIAL = ["<|bos|>", "<|eos|>", "<|pad|>", "<|im_start|>", "<|im_end|>", "<|think|>",
           "<|end_think|>", "<|tool_call|>", "<|tool_result|>", "<|json|>"] + [
    f"<|reserved_{i:02d}|>" for i in range(22)]

SOURCES = {
    # domain: (repo, file-prefix, text column, train-shard index, val-shard index)
    "web": ("HuggingFaceFW/fineweb-edu", "sample/10BT/", "text", 0, 1),
    "textbook": ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2/", "text", 0, 1),
    "math": ("HuggingFaceTB/finemath", "finemath-4plus/", "text", 0, 1),
    "code": ("codeparrot/github-code-clean", "data/", "code", 0, 1),  # permissive-licence filtered
}
MIX = {"web": 0.62, "textbook": 0.13, "math": 0.13, "code": 0.12}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def parquet_texts(repo: str, prefix: str, col: str, idx: int, cache: Path):
    files = sorted(f for f in list_repo_files(repo, repo_type="dataset")
                   if f.startswith(prefix) and f.endswith(".parquet"))
    path = hf_hub_download(repo, files[idx], repo_type="dataset", cache_dir=str(cache))
    pf = pq.ParquetFile(path)
    for rg in range(pf.num_row_groups):
        for t in pf.read_row_group(rg, columns=[col]).column(col).to_pylist():
            if t and 200 < len(t) < 100_000:
                yield t
    return


def ngrams13(text: str) -> set[int]:
    # Stable 64-bit hash (Python's hash() is salted per process -> non-reproducible filtering; fixed in v2).
    w = text.split()
    return {int.from_bytes(hashlib.blake2b(" ".join(w[i:i + 13]).encode(), digest_size=8).digest(), "little")
            for i in range(0, max(0, len(w) - 12), 7)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/Volumes/Express1M2/nanolm-lab/data/v1")
    ap.add_argument("--cache", default="/Volumes/Express1M2/nanolm-lab/hf-cache")
    ap.add_argument("--train-tokens", type=int, default=800_000_000)
    ap.add_argument("--val-tokens-per-domain", type=int, default=1_000_000)
    ap.add_argument("--tok-sample-mb", type=int, default=400)
    args = ap.parse_args()
    out, cache = Path(args.out), Path(args.cache)
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def log(*a):
        print(f"[{time.time() - t0:7.0f}s]", *a, flush=True)

    # ---------- 1. tokenizer ----------
    tok_path = out / "tokenizer_32k.json"
    if not tok_path.exists():
        log("training tokenizer")
        def sample():
            budget = args.tok_sample_mb * 1_000_000
            for dom, frac in MIX.items():
                b = 0
                it = parquet_texts(*SOURCES[dom][:3], SOURCES[dom][3], cache)
                for t in it:
                    yield t
                    b += len(t)
                    if b > budget * frac:
                        break
        tk = Tokenizer(models.BPE(byte_fallback=False))
        tk.normalizer = normalizers.NFC()
        tk.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)])
        tk.decoder = decoders.ByteLevel()
        tr = trainers.BpeTrainer(vocab_size=32768, special_tokens=SPECIAL, min_frequency=2,
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                 show_progress=False)
        tk.train_from_iterator(sample(), trainer=tr)
        tk.save(str(tok_path))
    tk = Tokenizer.from_file(str(tok_path))
    eos = tk.token_to_id("<|eos|>")
    assert tk.get_vocab_size() <= 65535
    log("tokenizer vocab", tk.get_vocab_size())

    def encode_stream(texts, limit, dst: Path, exclude: set[int] | None = None, collect=None):
        n, ndocs, nbytes, skipped = 0, 0, 0, 0
        buf: list[np.ndarray] = []
        batch: list[str] = []

        def flush():
            nonlocal n, ndocs, nbytes
            for enc, t in zip(tk.encode_batch(batch, add_special_tokens=False), batch):
                ids = np.asarray(enc.ids + [eos], dtype=np.uint16)
                buf.append(ids)
                n += len(ids)
                ndocs += 1
                nbytes += len(t.encode("utf-8"))
            batch.clear()

        for t in texts:
            if exclude is not None and ngrams13(t) & exclude:
                skipped += 1
                continue
            if collect is not None:
                collect |= ngrams13(t)
            batch.append(t)
            if len(batch) >= 512:
                flush()
                if n >= limit:
                    break
        if batch and n < limit:
            flush()
        arr = np.concatenate(buf)[:limit] if buf else np.zeros(0, np.uint16)
        arr.tofile(dst)
        return {"tokens": int(arr.size), "docs": ndocs, "bytes": nbytes,
                "bytes_per_token": round(nbytes / max(1, n), 4), "skipped_overlap": skipped,
                "sha256": sha256_file(dst)}

    manifest: dict = {"schema": "nanolm.lab.corpus.v1", "mix": MIX, "domains": {},
                      "tokenizer_sha256": sha256_file(tok_path), "vocab": tk.get_vocab_size()}

    # ---------- 2. validation (distinct shards) ----------
    val_grams: set[int] = set()
    for dom in MIX:
        dst = out / f"val_{dom}.bin"
        it = parquet_texts(*SOURCES[dom][:3], SOURCES[dom][4], cache)
        manifest["domains"].setdefault(dom, {})["val"] = encode_stream(
            it, args.val_tokens_per_domain, dst, collect=val_grams)
        log("val", dom, manifest["domains"][dom]["val"]["tokens"])

    # ---------- 3. training (per domain, val-overlap filtered) ----------
    parts = []
    for dom, frac in MIX.items():
        dst = out / f"train_{dom}.bin"
        lim = int(args.train_tokens * frac)
        repo, pre, col, tr_idx, _ = SOURCES[dom]
        def it_multi(repo=repo, pre=pre, col=col):
            for shard in [0, 2, 3, 4, 5, 6, 7, 8]:
                yield from parquet_texts(repo, pre, col, shard, cache)
        it = it_multi()
        manifest["domains"][dom]["train"] = encode_stream(it, lim, dst, exclude=val_grams)
        parts.append(np.fromfile(dst, dtype=np.uint16))
        log("train", dom, manifest["domains"][dom]["train"])

    # Interleave domains in 64k-token blocks so any prefix of train.bin is mix-representative.
    rng = np.random.default_rng(0)
    blocks = []
    B = 65536
    for p in parts:
        blocks += [p[i:i + B] for i in range(0, len(p), B)]
    order = rng.permutation(len(blocks))
    train = np.concatenate([blocks[i] for i in order])
    train.tofile(out / "train.bin")
    manifest["train_tokens_total"] = int(train.size)
    manifest["train_sha256"] = sha256_file(out / "train.bin")
    manifest["build_seconds"] = round(time.time() - t0, 1)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log("done", manifest["train_tokens_total"])


if __name__ == "__main__":
    main()
