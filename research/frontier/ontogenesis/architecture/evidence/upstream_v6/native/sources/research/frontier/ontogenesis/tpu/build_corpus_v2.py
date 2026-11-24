"""Corpus v2: the same four pinned public sources as corpus v1, many times more text.

Nothing about the recipe changes: same repositories and revisions, same text columns, same document
filter (200 to 100,000 characters), same tokenizer, same end-of-document token, same removal of any
document sharing a 13-word window with the held-out sets, same 64k-token interleaving of the domains.
The held-out sets are the v1 files, unchanged, so every earlier number stays comparable.
v1 took the first shard of each source up to a quota; v2 takes that shard and the following ones
(never the shard the held-out sets came from) up to a larger quota.

Sources and the licences stated on their dataset cards (checked 2026-11-21; not gated):
  web       HuggingFaceFW/fineweb-edu      sample/10BT/       odc-by
  textbook  HuggingFaceTB/smollm-corpus    cosmopedia-v2/     odc-by
  maths     HuggingFaceTB/finemath         finemath-4plus/    odc-by
  code      codeparrot/github-code-clean   data/              apache-2.0 on the card; files keep their own licences

Self-checks before any large work:
  1. the held-out sets rebuilt from the sources must hash to the v1 manifest values;
  2. (--check-prefix) the first tokens produced for the first web shard must equal v1's train_web.bin.

usage: build_corpus_v2.py --v1 <dir with v1 files> --out <dir> --cache <dir> [--total 10e9] [--workers 48] [--gcs gs://bucket/prefix]
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from multiprocessing import get_context
from pathlib import Path
import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import pyarrow.parquet as pq                                    # noqa: E402
from huggingface_hub import HfApi, hf_hub_download             # noqa: E402
from tokenizers import Tokenizer                               # noqa: E402

SOURCES = {  # domain: (repository, file prefix, text column, pinned revision)
    "web": ("HuggingFaceFW/fineweb-edu", "sample/10BT/", "text", "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"),
    "textbook": ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2/", "text", "3ba9d605774198c5868892d7a8deda78031a781f"),
    "math": ("HuggingFaceTB/finemath", "finemath-4plus/", "text", "e92b25a616738fe95dc186b64dfb19f9c8525594"),
    "code": ("codeparrot/github-code-clean", "data/", "code", "c48d40f9e70f0196f8236901ee35807f7d6c44c0"),
}
MIX = {"web": 0.62, "textbook": 0.13, "math": 0.13, "code": 0.12}
G = {}                                                          # filled in the parent before forking: tokenizer path, held-out n-grams, dirs


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""): h.update(chunk)
    return h.hexdigest()


def ngrams13(text):
    w = text.split()
    return {hash(" ".join(w[i:i + 13])) for i in range(0, max(0, len(w) - 12), 7)}


def fetch(repo, filename, rev, cache, tries=5):
    for k in range(tries):
        try: return hf_hub_download(repo, filename, revision=rev, repo_type="dataset", cache_dir=cache)
        except Exception:
            if k == tries - 1: raise
            time.sleep(5 * (k + 1))


def texts_of(path, col):
    pf = pq.ParquetFile(path)
    for rg in range(pf.num_row_groups):
        for t in pf.read_row_group(rg, columns=[col]).column(col).to_pylist():
            if t and 200 < len(t) < 100_000: yield t


def encode_shard(job):
    """Worker: one source shard -> one token file. Returns statistics. Deletes the downloaded shard afterwards unless told to keep it."""
    dom, filename, limit, keep = job
    repo, _, col, rev = SOURCES[dom]
    tk = Tokenizer.from_file(G["tokenizer"]); eos = tk.token_to_id("<|eos|>"); exclude = G["val_grams"]
    path = fetch(repo, filename, rev, G["cache"]); dst = Path(G["parts"]) / f"{dom}--{Path(filename).name}.bin"
    n = docs = nbytes = skipped = 0; batch = []
    with open(dst, "wb") as out:
        def flush():
            nonlocal n, docs, nbytes
            for enc, t in zip(tk.encode_batch(batch, add_special_tokens=False), batch):
                ids = np.asarray(enc.ids + [eos], dtype="<u2"); out.write(ids.tobytes()); n += len(ids); docs += 1; nbytes += len(t.encode("utf-8"))
            batch.clear()
        for t in texts_of(path, col):
            if ngrams13(t) & exclude: skipped += 1; continue
            batch.append(t)
            if len(batch) >= 512:
                flush()
                if limit and n >= limit: break
        if batch and not (limit and n >= limit): flush()
    if not keep:
        try: os.remove(os.path.realpath(path))
        except OSError: pass
    return {"domain": dom, "file": filename, "tokens": n, "docs": docs, "bytes": nbytes, "skipped_overlap": skipped, "part": str(dst)}


def rebuild_val(dom, files, v1, cache):
    """Exactly what the v1 builder did for the held-out set of one domain; returns its statistics and n-gram set."""
    repo, _, col, rev = SOURCES[dom]
    tk = Tokenizer.from_file(G["tokenizer"]); eos = tk.token_to_id("<|eos|>")
    path = fetch(repo, files[1], rev, cache); grams = set(); buf = []; batch = []; n = 0
    def flush():
        nonlocal n
        for enc in tk.encode_batch(batch, add_special_tokens=False):
            ids = np.asarray(enc.ids + [eos], dtype="<u2"); buf.append(ids); n += len(ids)
        batch.clear()
    for t in texts_of(path, col):
        grams |= ngrams13(t); batch.append(t)
        if len(batch) >= 512:
            flush()
            if n >= 1_000_000: break
    if batch and n < 1_000_000: flush()
    arr = np.concatenate(buf)[:1_000_000]
    same = hashlib.sha256(arr.tobytes()).hexdigest() == sha256_file(f"{v1}/val_{dom}.bin")
    return same, grams


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True); ap.add_argument("--out", required=True); ap.add_argument("--cache", required=True)
    ap.add_argument("--total", type=float, default=10e9); ap.add_argument("--workers", type=int, default=48); ap.add_argument("--gcs", default="")
    ap.add_argument("--check-prefix", type=int, default=0, help="only verify: produce this many web tokens from the first shard and compare with v1's train_web.bin")
    ap.add_argument("--keep-downloads", action="store_true")
    a = ap.parse_args(); t0 = time.time(); out = Path(a.out); (out / "parts").mkdir(parents=True, exist_ok=True)
    log = lambda *x: print(f"[{time.time() - t0:6.0f}s]", *x, flush=True)
    G.update(tokenizer=f"{a.v1}/tokenizer_32k.json", cache=a.cache, parts=str(out / "parts"), val_grams=set())
    v1m = json.load(open(f"{a.v1}/manifest.json"))
    assert sha256_file(G["tokenizer"]) == v1m["tokenizer_sha256"], "tokenizer differs from v1"
    api = HfApi(); files = {}
    for dom, (repo, pre, col, rev) in SOURCES.items():
        info = api.dataset_info(repo, revision=rev, files_metadata=True)
        files[dom] = sorted(s.rfilename for s in info.siblings if s.rfilename.startswith(pre) and s.rfilename.endswith(".parquet"))
        log(dom, "shards available", len(files[dom]))
    for dom in MIX:                                             # self-check 1 and the exclusion set
        same, grams = rebuild_val(dom, files[dom], a.v1, a.cache); G["val_grams"] |= grams
        log("held-out", dom, "reproduces v1:", same)
        if not same: raise SystemExit(f"held-out set for {dom} does not reproduce; refusing to build")
    if a.check_prefix:                                          # self-check 2
        st = encode_shard(("web", files["web"][0], a.check_prefix, True)); got = np.fromfile(st["part"], dtype="<u2")
        ref = np.fromfile(f"{a.v1}/train_web.bin", dtype="<u2", count=len(got)); k = min(len(got), len(ref))
        log("first web shard, first", k, "tokens equal v1's train_web.bin:", bool(np.array_equal(got[:k], ref[:k]))); sys.exit(0 if np.array_equal(got[:k], ref[:k]) else 1)
    quota = {dom: int(a.total * frac) for dom, frac in MIX.items()}
    order = {dom: [f for i, f in enumerate(files[dom]) if i != 1] for dom in MIX}          # never the held-out shard
    ctx = get_context("fork"); stats = {dom: [] for dom in MIX}
    with ctx.Pool(a.workers) as pool:
        first = pool.map(encode_shard, [(dom, order[dom][0], 0, a.keep_downloads) for dom in MIX])   # one shard each, to learn the yield
        for st in first: stats[st["domain"]].append(st); log("first shard", st["domain"], f"{st['tokens'] / 1e6:.0f}M tokens")
        jobs = []
        for dom in MIX:
            per = max(1, stats[dom][0]["tokens"]); need = int(np.ceil(max(0, quota[dom] - per) / per * 1.08))
            take = order[dom][1:1 + need]
            if len(take) < need: log("WARNING", dom, "source has too few shards for its quota; taking all", len(take))
            jobs += [(dom, f, 0, a.keep_downloads) for f in take]
        log("second wave:", len(jobs), "shards")
        for st in pool.imap_unordered(encode_shard, jobs):
            stats[st["domain"]].append(st); log("done", st["domain"], Path(st["file"]).name, f"{st['tokens'] / 1e6:.0f}M tokens")
    manifest = {"schema": "nanolm.lab.corpus.v2", "mix": MIX, "tokenizer_sha256": v1m["tokenizer_sha256"], "vocab": v1m["vocab"], "domains": {},
                "sources": {dom: {"repository": SOURCES[dom][0], "prefix": SOURCES[dom][1], "revision": SOURCES[dom][3]} for dom in MIX},
                "note": "held-out sets and tokenizer are the v1 files, unchanged; the held-out shard of each source is never used for training"}
    parts = []
    for dom in MIX:
        byfile = {s["file"]: s for s in stats[dom]}; arrs = []; n = 0; used = []
        for f in order[dom]:
            if f not in byfile or n >= quota[dom]: continue
            arr = np.fromfile(byfile[f]["part"], dtype="<u2"); arrs.append(arr[:quota[dom] - n]); n += len(arrs[-1]); used.append(Path(f).name)
        full = np.concatenate(arrs); parts.append(full)
        tot = {k: int(sum(s[k] for s in stats[dom] if Path(s["file"]).name in used)) for k in ("docs", "bytes", "skipped_overlap")}
        manifest["domains"][dom] = {"val": v1m["domains"][dom]["val"], "train": {"tokens": int(len(full)), "shards_used": used, **tot}}
        log("domain", dom, f"{len(full) / 1e9:.2f}B tokens from", len(used), "shards", "(short of quota)" if len(full) < quota[dom] else "")
    for dom in MIX:                                             # the token files are in memory now; free the disk before writing the result
        for st in stats[dom]:
            try: os.remove(st["part"])
            except OSError: pass
    rng = np.random.default_rng(0); B = 65536; blocks = []
    for p in parts: blocks += [p[i:i + B] for i in range(0, len(p), B)]
    perm = rng.permutation(len(blocks))
    with open(out / "train.bin", "wb") as f:
        for i in perm: f.write(blocks[i].tobytes())
    manifest["train_tokens_total"] = int(sum(len(p) for p in parts)); manifest["train_sha256"] = sha256_file(out / "train.bin"); manifest["build_seconds"] = round(time.time() - t0, 1)
    json.dump(manifest, open(out / "manifest.json", "w"), indent=2)
    for name in ["tokenizer_32k.json"] + [f"val_{d}.bin" for d in MIX]: subprocess.run(["cp", f"{a.v1}/{name}", str(out / name)])
    log("built", f"{manifest['train_tokens_total'] / 1e9:.2f}B tokens")
    if a.gcs:
        r = subprocess.run(["gcloud", "storage", "cp", str(out / "train.bin"), str(out / "manifest.json"), str(out / "tokenizer_32k.json")] + [str(out / f"val_{d}.bin") for d in MIX] + [a.gcs.rstrip("/") + "/"], capture_output=True, text=True)
        log("upload", "ok" if r.returncode == 0 else "FAILED " + r.stderr[-300:]); sys.exit(r.returncode)
