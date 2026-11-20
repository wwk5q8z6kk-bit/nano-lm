"""Reconstruct the pinned Antigravity dataset locally from verified public parquet objects."""


from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from functools import lru_cache
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

SPECIAL = ["<|bos|>", "<|eos|>", "<|pad|>", "<|im_start|>", "<|im_end|>", "<|think|>",
           "<|end_think|>", "<|tool_call|>", "<|tool_result|>", "<|json|>"] + [
    f"<|reserved_{i:02d}|>" for i in range(22)]

SOURCES = {
    # domain: (repo, file-prefix, text column, train-shard index, val-shard index)
    "web": ("HuggingFaceFW/fineweb-edu", "sample/10BT/", "text", 0, 1),
    "textbook": ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2/", "text", 0, 1),
    "math": ("HuggingFaceTB/finemath", "finemath-4plus/", "text", 0, 1),
    "code": ("codeparrot/github-code-clean", "data/", "code", 0, 1),
}
MIX = {"web": 0.62, "textbook": 0.13, "math": 0.13, "code": 0.12}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


PINS = json.loads(Path(__file__).with_name("corpus.json").read_text())


@lru_cache(maxsize=8)
def verified_object(repo, prefix, idx, cache, download):
    source = PINS["sources"][repo]
    objects = source["cached_parquet_objects"]
    filename = sorted(f for f in objects if f.startswith(prefix))[idx]
    path = Path(hf_hub_download(repo, filename, revision=source["revision"],
                                repo_type="dataset", cache_dir=cache, local_files_only=not download))
    if sha256_file(path) != objects[filename]:
        raise ValueError(f"Source checksum mismatch: {repo}/{filename}")
    return path


def parquet_texts(repo, prefix, col, idx, cache, download):
    path = verified_object(repo, prefix, idx, cache, download)
    pf = pq.ParquetFile(path)
    for rg in range(pf.num_row_groups):
        for t in pf.read_row_group(rg, columns=[col]).column(col).to_pylist():
            if t and 200 < len(t) < 100_000:
                yield t
    return


def ngrams13(text: str) -> set[int]:
    w = text.split()
    return {hash(" ".join(w[i:i + 13])) for i in range(0, max(0, len(w) - 12), 7)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="Empty output directory")
    ap.add_argument("--cache", type=Path, required=True, help="Hugging Face object cache")
    ap.add_argument("--download", action="store_true", help="Fetch source objects missing from the local cache")
    ap.add_argument("--tokenizer", type=Path, help="Supply an original tokenizer with the recorded SHA-256")
    scope = ap.add_mutually_exclusive_group()
    scope.add_argument("--only-tokenizer", action="store_true", help="Stop after checking the tokenizer")
    scope.add_argument("--only-validation", action="store_true", help="Stop after rebuilding and checking all validation inputs")
    args = ap.parse_args()
    out, cache = Path(args.out), Path(args.cache)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Corpus reconstruction requires an empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def log(*a):
        print(f"[{time.time() - t0:7.0f}s]", *a, flush=True)

    # ---------- 1. tokenizer ----------
    tok_path = out / "tokenizer_32k.json"
    if args.tokenizer:
        shutil.copyfile(args.tokenizer, tok_path)
    else:
        log("training tokenizer")
        def sample():
            budget = 400_000_000
            for dom, frac in MIX.items():
                b = 0
                it = parquet_texts(*SOURCES[dom][:3], SOURCES[dom][3], cache, args.download)
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
    if sha256_file(tok_path) != PINS["manifest"]["tokenizer_sha256"]:
        raise ValueError("Tokenizer hash differs. Exact replay requires the original pinned tokenizer.")
    if args.only_tokenizer:
        log("tokenizer SHA-256 verified")
        return
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
                ids = np.asarray(enc.ids + [eos], dtype="<u2")
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
        arr = np.concatenate(buf)[:limit] if buf else np.zeros(0, dtype="<u2")
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
        it = parquet_texts(*SOURCES[dom][:3], SOURCES[dom][4], cache, args.download)
        manifest["domains"].setdefault(dom, {})["val"] = encode_stream(
            it, 1_000_000, dst, collect=val_grams)
        log("val", dom, manifest["domains"][dom]["val"]["tokens"])

    for dom in MIX:
        if manifest["domains"][dom]["val"] != PINS["manifest"]["domains"][dom]["val"]:
            raise ValueError(f"Rebuilt validation mismatch for {dom}")
    if args.only_validation:
        log("all validation inputs verified")
        return

    # ---------- 3. training (per domain, val-overlap filtered) ----------
    parts = []
    for dom, frac in MIX.items():
        dst = out / f"train_{dom}.bin"
        lim = int(800_000_000 * frac)
        repo, pre, col, tr_idx, _ = SOURCES[dom]
        it = parquet_texts(repo, pre, col, tr_idx, cache, args.download)
        manifest["domains"][dom]["train"] = encode_stream(it, lim, dst, exclude=val_grams)
        parts.append(np.fromfile(dst, dtype="<u2"))
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
    expected = PINS["manifest"]
    if any(manifest[k] != expected[k] for k in expected if k != "build_seconds"):
        raise ValueError("Rebuilt corpus differs from the archived pins")
    log("corpus and all component SHA-256 values verified", manifest["train_tokens_total"])


if __name__ == "__main__":
    main()
