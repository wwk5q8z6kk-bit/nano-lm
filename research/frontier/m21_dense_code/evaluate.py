"""Evaluate all six finished fits on exact, within-document next-token targets."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from study import (ARMS, BATCH, LENGTH, PARAMS, SEEDS, STEPS, UnifiedBlockLM,
                   digest, make_model, sample_offsets, save, source_hashes,
                   synchronize, verified_corpus)


def chunks(docs):
    for i, row in enumerate(docs):
        for k in range(0, row["tokens"] - 1, LENGTH):
            yield i, row["token_offset"] + k, min(LENGTH, row["tokens"] - 1 - k)


def packed(tape, batch):
    rows = np.zeros((len(batch), LENGTH + 1), dtype=np.int64)
    for i, (_, start, n) in enumerate(batch):
        rows[i, :n + 1] = tape[start:start + n + 1]
    return rows


def pointer_available(x, y):
    idx, valid = UnifiedBlockLM._build_sparse_pointer_candidates(x, n_exact=8, n_local=24)
    rows = torch.arange(len(x), device=x.device)[:, None, None]
    labels = x[rows, (idx + 1).clamp_max(x.shape[1] - 1)]
    return (valid & (labels == y[..., None])).any(dim=-1)


def bits(logits, y):
    # Renormalize both models, including the probability floor on M21's native readout.
    chosen = logits.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    return (torch.logsumexp(logits, dim=-1) - chosen) / math.log(2)


def size_bin(n):
    if n <= 512:
        return "le512"
    if n <= 2048:
        return "513to2048"
    return "gt2048"


def corpus_accounting(docs, tape, tokenizer):
    from tokenizers import Tokenizer
    decoder = Tokenizer.from_file(str(tokenizer))
    byte_count = 0
    for row in docs:
        start, n = row["token_offset"], row["tokens"]
        raw = decoder.decode(tape[start:start + n - 1].tolist(), skip_special_tokens=False)
        byte_count += len(raw.encode("utf-8"))
    return {"documents": len(docs), "repositories": len({r["repo"].casefold() for r in docs}),
            "input_tokens_including_final_delimiters": len(tape), "decoded_utf8_bytes_excluding_final_delimiters": byte_count,
            "targets": sum(r["tokens"] - 1 for r in docs), "chunks": sum(1 for _ in chunks(docs))}


def empty():
    return {"targets": 0, "bits": 0.0}


def add(row, values):
    row["targets"] += values.size
    row["bits"] += float(values.sum(dtype=np.float64))


class Totals:
    def __init__(self, docs):
        groups = sorted({row["repo"].casefold() for row in docs})
        anon = {name: str(i) for i, name in enumerate(groups)}
        self.repos = [anon[row["repo"].casefold()] for row in docs]
        self.sums = {key: defaultdict(empty) for key in ("repo", "row_license", "document_tokens", "pointer_target", "all")}
        self.docs = docs

    def update(self, batch, values, hit):
        for i, (doc_id, _, n) in enumerate(batch):
            row = self.docs[doc_id]
            v = values[i, :n]
            if not np.isfinite(v).all() or (v < 0).any():
                raise ValueError("Invalid per-target bits")
            add(self.sums["repo"][self.repos[doc_id]], v)
            add(self.sums["row_license"][row["license"]], v)
            add(self.sums["document_tokens"][size_bin(row["tokens"])], v)
            add(self.sums["all"]["pooled"], v)
            available = hit[i, :n]
            add(self.sums["pointer_target"]["available"], v[available])
            add(self.sums["pointer_target"]["absent"], v[~available])

    def result(self, required):
        for keys in self.sums.values():
            if sum(v["targets"] for v in keys.values()) != required:
                raise ValueError("Incomplete stratified measurement")
        return {k: dict(v) for k,v in self.sums.items()}


@torch.inference_mode()
def evaluate(model, arm, docs, tape, device):
    model.eval()
    primary = Totals(docs)
    base = Totals(docs) if arm == "m21" else None
    parts = list(chunks(docs))
    for j in range(0, len(parts), BATCH):
        batch = parts[j:j + BATCH]
        rows = torch.from_numpy(packed(tape, batch)).to(device)
        x, y = rows[:, :-1], rows[:, 1:]
        available = pointer_available(x, y).cpu().numpy()
        if base:
            mixed, base_logits = model(x, return_base_logits=True)
            base.update(batch, bits(base_logits, y).cpu().numpy(), available)
            del base_logits
        else:
            mixed = model(x)
        primary.update(batch, bits(mixed, y).cpu().numpy(), available)
        del mixed
    n_targets = sum(row["tokens"] - 1 for row in docs)
    heads = {"primary": primary.result(n_targets)}
    if base:
        heads["base"] = base.result(n_targets)
    return heads


def finished(fits, train_docs):
    import hashlib
    cells = []
    for seed in SEEDS:
        offsets = sample_offsets(train_docs, seed)
        pin = hashlib.sha256(offsets.astype("<i8", copy=False).tobytes()).hexdigest()
        for arm in ARMS:
            cell = fits / f"seed-{seed}-{arm}"
            path = cell.with_suffix(".pt")
            r = json.loads(cell.with_suffix(".json").read_text())
            if (r["arm"], r["seed"], r["parameters"], r["training_tokens"], r["sample_offsets_sha256"],
                r["source_sha256"], r["weights_sha256"]) != (
                arm, seed, PARAMS[arm], STEPS * BATCH * LENGTH, pin, source_hashes(), digest(path)):
                raise ValueError(f"Fit does not satisfy the protocol: {cell.name}")
            cells.append((arm, seed, r, path))
    return cells


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "tokenizer", "fits", "output"):
        ap.add_argument("--" + key, type=Path, required=True)
    ap.add_argument("--device", choices=("cpu", "mps", "cuda"), required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    docs, streams = verified_corpus(args.data, args.tokenizer)
    cells = finished(args.fits, docs["train"])  # all six are mandatory before the first validation forward
    args.output.mkdir(parents=True, exist_ok=True)
    census = corpus_accounting(docs["val"], streams["val"], args.tokenizer)
    if census["targets"] != 516033 or census["repositories"] != 329 or census["documents"] != 335:
        raise ValueError("Unexpected validation population")
    save(args.output / "validation-accounting.json", census)
    for arm, seed, receipt, path in cells:
        out = args.output / f"eval-{seed}-{arm}.json"
        if out.exists():
            record = json.loads(out.read_text())
            if (record["weights_sha256"], record["eval_source_sha256"], record["device"]) != (digest(path), digest(Path(__file__)), args.device):
                raise ValueError(f"An incompatible evaluation result exists at {out}")
            continue
        model = make_model(arm, seed)
        model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"), strict=True)
        model = model.to(device)
        synchronize(device)
        start = time.perf_counter()
        heads = evaluate(model, arm, docs["val"], streams["val"], device)
        synchronize(device)
        seconds = time.perf_counter() - start
        save(out, {"seed": seed, "arm": arm, "weights_sha256": receipt["weights_sha256"],
                   "eval_source_sha256": digest(Path(__file__)), "device": args.device,
                   "seconds": seconds, "heads": heads})
        pooled = heads["primary"]["all"]["pooled"]
        print(f"{arm} seed={seed}, {pooled['targets']} targets, {pooled['bits']/pooled['targets']:.6f} bits/token, {seconds:.1f}s", flush=True)
        del model


if __name__ == "__main__":
    main()
