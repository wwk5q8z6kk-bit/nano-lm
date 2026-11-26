"""Train two size-matched next-token models on document-bounded sampled windows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from dense import Cfg, LM

ROOT = Path(__file__).resolve().parent
NATIVE = ROOT.parent / "u10_registers"
sys.path.insert(0, str(NATIVE))
import metrics
from model import UnifiedBlockLM

STEPS, BATCH, LENGTH = 4000, 8, 128
ARMS = ("m21", "gqa")
SEEDS = (0, 1, 2)
PARAMS = {"m21": 5_046_637, "gqa": 5_046_656}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def verified_corpus(data, tokenizer):
    reference = json.loads((ROOT / "corpus.json").read_text())
    if json.loads((data / "manifest.json").read_text()) != reference or digest(tokenizer) != reference["tokenizer_sha256"]:
        raise ValueError("Corpus manifest or tokenizer differs from the pinned study")
    rows, streams = {}, {}
    for split, item in reference["splits"].items():
        metadata, tape = (data / f"{split}_code.{ext}" for ext in ("jsonl", "bin"))
        if digest(metadata) != item["records_sha256"] or digest(tape) != item["tokens_sha256"]:
            raise ValueError(f"Input checksum mismatch in {split}")
        docs = [json.loads(line) for line in metadata.read_text().splitlines()]
        tokens = np.memmap(tape, dtype="<u2", mode="r")
        cursor = 0
        for doc in docs:
            size = doc["tokens"]
            if doc["token_offset"] != cursor or size < 2 or doc["license"] not in reference["allowed_row_labels"]:
                raise ValueError("Invalid document boundaries or label")
            cursor += size
            if tokens[cursor - 1] != reference["eos_id"]:
                raise ValueError("Missing document delimiter")
        if cursor != item["tokens"] or len(tokens) != cursor or len(docs) != item["accepted"]:
            raise ValueError("Corpus length mismatch")
        if len({r["source_sha256"] for r in docs}) != len(docs):
            raise ValueError("Duplicate document digest within a split")
        rows[split], streams[split] = docs, tokens
    for key in ("repo", "source_sha256"):
        groups = [{str(d[key]).casefold() for d in docs} for docs in rows.values()]
        if groups[0] & groups[1]:
            raise ValueError(f"Shared {key} across train and validation")
    return rows, streams


def sample_offsets(docs, seed, steps=STEPS):
    # Each rank corresponds to exactly one valid start across all files.
    sizes = np.array([r["tokens"] for r in docs], dtype=np.int64)
    cumulative = np.maximum(sizes - LENGTH, 0).cumsum()
    ranks = np.random.default_rng(70000 + 131 * seed).integers(cumulative[-1], size=(steps, BATCH), dtype=np.int64)
    files = cumulative.searchsorted(ranks, side="right")
    preceding = np.r_[0, cumulative[:-1]]
    file_offsets = np.array([r["token_offset"] for r in docs], dtype=np.int64)
    starts = file_offsets[files] + ranks - preceding[files]
    if np.any(starts < file_offsets[files]) or np.any(starts + LENGTH >= file_offsets[files] + sizes[files]):
        raise AssertionError("Training window crosses a document boundary")
    return starts


def make_model(arm, seed):
    torch.manual_seed(1000 + 97 * seed)
    if arm == "gqa":
        model = LM(Cfg(vocab=32768, d=128, n_head=4, n_kv=2, ffn=426, layout="AAAA", seq=128, tie=True))
    elif arm == "m21":
        model = UnifiedBlockLM(32768, 128, "M21_HSPA_ConceptSlot_EvidPtr", n_heads=4, ffn_hidden_override=460, hspa_registers=4)
    else:
        raise ValueError(f"Unknown arm {arm}")
    if sum(p.numel() for p in model.parameters()) != PARAMS[arm]:
        raise ValueError(f"Incorrect parameter count for {arm}")
    return model


def source_hashes():
    return {p.name: digest(p) for p in (ROOT / "study.py", ROOT / "dense.py", ROOT / "protocol.md",
                                         ROOT / "corpus.json", NATIVE / "model.py", NATIVE / "metrics.py")}


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def train_step(model, arm, optimizer, tokens, step):
    optimizer.param_groups[0]["lr"] = metrics._cosine_lr(step, STEPS, 400, 0.0015)
    x, targets = tokens[:, :-1], tokens[:, 1:]
    if arm == "m21":
        loss = metrics._compute_u08_loss(model, x, targets, metrics._annealed_aux_weight(step, STEPS))
    else:
        loss = model(x, tgt=targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return loss.detach()


def tensor_rows(stream, starts, device):
    sequences = np.stack([stream[s:s + LENGTH + 1] for s in starts]).astype(np.int64)
    return torch.from_numpy(sequences).to(device)


def fit(arm, seed, stream, offsets, output, device):
    name = f"seed-{seed}-{arm}"
    weights = output / f"{name}.pt"
    receipt = output / f"{name}.json"
    pin = hashlib.sha256(offsets.astype("<i8", copy=False).tobytes()).hexdigest()
    hashes = source_hashes()
    if weights.exists() and receipt.exists():
        row = json.loads(receipt.read_text())
        if (row["sample_offsets_sha256"], row["source_sha256"], row["weights_sha256"], row["device"],
            row["training_tokens"]) == (pin, hashes, digest(weights), str(device), STEPS * BATCH * LENGTH):
            print(f"{name} complete and verified", flush=True)
            return
        raise ValueError(f"Existing {name} has a different identity")
    if weights.exists() or receipt.exists():
        raise FileExistsError(f"Preserve partial {name} and use a clean output directory")
    model = make_model(arm, seed).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.01)
    synchronize(device)
    start = time.perf_counter()
    trace = []
    for step, starts in enumerate(offsets):
        loss = train_step(model, arm, opt, tensor_rows(stream, starts, device), step)
        if step == 0 or (step + 1) % 100 == 0:
            synchronize(device)
            seconds = time.perf_counter() - start
            val = float(loss.cpu())
            if not np.isfinite(val):
                raise FloatingPointError(f"Nonfinite loss for {name} on update {step + 1}")
            trace.append({"update": step + 1, "loss": val, "seconds": round(seconds, 4)})
            print(f"{name} {step + 1}/{STEPS} loss={val:.4f} tokens/s={(step+1)*BATCH*LENGTH/seconds:.0f}", flush=True)
    temp = weights.with_suffix(".pt.part")
    torch.save({key: t.detach().cpu() for key, t in model.state_dict().items()}, temp)
    temp.replace(weights)
    save(receipt, {"arm": arm, "seed": seed, "parameters": PARAMS[arm], "training_tokens": STEPS * BATCH * LENGTH,
                   "sample_offsets_sha256": pin, "source_sha256": hashes, "weights_sha256": digest(weights),
                   "device": str(device), "torch": str(torch.__version__), "numpy": np.__version__,
                   "python": platform.python_version(), "training_seconds": seconds, "trace": trace})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for arg in ("data", "tokenizer", "output"):
        ap.add_argument(f"--{arg}", type=Path, required=True)
    ap.add_argument("--device", choices=("mps", "cpu", "cuda"), required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    docs, streams = verified_corpus(args.data, args.tokenizer)
    # POSIX releases the single-run lock even after interruption.
    import fcntl
    with (args.output / ".training.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for seed in SEEDS:
            offsets = sample_offsets(docs["train"], seed)
            order = ARMS if seed % 2 == 0 else reversed(ARMS)
            for arm in order:
                fit(arm, seed, streams["train"], offsets, args.output, torch.device(args.device))


if __name__ == "__main__":
    main()
