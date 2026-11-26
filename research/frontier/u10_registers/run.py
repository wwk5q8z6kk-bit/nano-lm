"""Run the U10 matched next-token pilot on the pinned four-domain corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import torch

import metrics
from model import UnifiedBlockLM

ROOT = Path(__file__).resolve().parent
ARMS = {
    "m21": ("M21_HSPA_ConceptSlot_EvidPtr", 4, 460),
    "m22": ("M22_HSPA_FullCore_U09", 4, 460),
    "m22_wide": ("M22_HSPA_FullCore_U09", 11, 357),
}
BATCH, LENGTH, VOCAB = 8, 128, 32768
STEPS, WARMUP, LR = 1000, 100, 0.0015


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_corpus(data):
    manifest = json.loads((data / "manifest.json").read_text())
    pins = json.loads((ROOT / "corpus.json").read_text())
    if {k: v for k, v in manifest.items() if k != "build_seconds"} != {
        k: v for k, v in pins["manifest"].items() if k != "build_seconds"
    }:
        raise ValueError("Input manifest differs from the archived corpus manifest")
    files = {"train.bin": manifest["train_sha256"], "tokenizer_32k.json": manifest["tokenizer_sha256"]}
    for domain, item in manifest["domains"].items():
        files[f"val_{domain}.bin"] = item["val"]["sha256"]
    for name, expected in files.items():
        if digest(data / name) != expected:
            raise ValueError(f"Unexpected SHA-256 for {name}")
    return manifest


def make_model(name, seed):
    mechanism, registers, ffn = ARMS[name]
    torch.manual_seed(1000 + seed * 97)
    return UnifiedBlockLM(VOCAB, 128, mechanism, n_heads=4, ffn_hidden_override=ffn, hspa_registers=registers)


def paired_model(name, seed):
    # A wider layer consumes a different number of random draws. Share exact
    # initial values for every corresponding tensor whose shape stays the same.
    model = make_model(name, seed)
    reference = make_model("m22", seed)
    anchor = reference.state_dict()
    common = 0
    with torch.no_grad():
        for key, parameter in model.named_parameters():
            if key in anchor and anchor[key].shape == parameter.shape:
                parameter.copy_(anchor[key])
                common += parameter.numel()
    return model, common


def synchronize(dev):
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def run_arm(name, seed, data, output, dev, manifest):
    path = output / f"seed-{seed}-{name}.json"
    weight_file = output / f"seed-{seed}-{name}.pt"
    if path.exists() or weight_file.exists():
        raise FileExistsError(f"Use a new output directory for {path.name}")
    train = np.memmap(data / "train.bin", dtype="<u2", mode="r")
    rng = np.random.default_rng(70_000 + seed * 131)
    starts = rng.integers(0, len(train) - LENGTH - 1, size=(STEPS, BATCH), dtype=np.int64)
    model, shared = paired_model(name, seed)
    model = model.to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    model.train()
    losses = []
    synchronize(dev)
    t0 = time.perf_counter()
    for step, row in enumerate(starts):
        optimizer.param_groups[0]["lr"] = metrics._cosine_lr(step, STEPS, WARMUP, LR)
        sequences = np.stack([train[i:i + LENGTH + 1] for i in row]).astype(np.int64)
        tokens = torch.from_numpy(sequences).to(dev)
        auxiliary = metrics._annealed_aux_weight(step, STEPS)
        loss = metrics._compute_u08_loss(model, tokens[:, :-1], tokens[:, 1:], auxiliary)
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Nonfinite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0:
            losses.append([step + 1, float(loss.detach().cpu())])
            print(f"{name} seed={seed} update={step + 1}/{STEPS} loss={losses[-1][1]:.4f}", flush=True)
    synchronize(dev)
    seconds = time.perf_counter() - t0
    model.eval()
    bpb = metrics.eval_bpb_multi_horizon(model, manifest, dev, data)
    copy = metrics.eval_copy_probes(model, VOCAB, dev)
    recall = metrics.eval_real_haystack_mqar(model, VOCAB, dev, data)
    ranks = metrics.audit_layer_ranks_v2(model, VOCAB, dev, data)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, weight_file)
    receipt = {
        "arm": name, "mechanism": ARMS[name][0], "registers": ARMS[name][1], "ffn": ARMS[name][2],
        "seed": seed, "steps": STEPS, "batch": BATCH, "length": LENGTH, "train_tokens": STEPS * BATCH * LENGTH,
        "train_offsets_sha256": hashlib.sha256(starts.astype("<i8", copy=False).tobytes()).hexdigest(),
        "train_file_sha256": manifest["train_sha256"], "checkpoint_sha256": digest(weight_file),
        "script_sha256": {p: digest(ROOT / p) for p in ("model.py", "metrics.py", "run.py", "corpus.json")},
        "parameters": sum(p.numel() for p in model.parameters()), "shared_initial_parameters": shared,
        "device": str(dev), "torch": str(torch.__version__), "python": platform.python_version(),
        "numpy": np.__version__, "device_machine": platform.machine(), "training_seconds": round(seconds, 3),
        "training_tokens_per_second": round(STEPS * BATCH * LENGTH / seconds, 2),
        "train_loss_checkpoints": losses, "validation_bpb": bpb,
        "mean_512_bpb": statistics.mean(bpb["L_512"].values()), "copy": copy, "recall": recall, "rank": ranks,
    }
    save(path, receipt)
    print(f"{name} seed={seed} mean 512 BPB={receipt['mean_512_bpb']:.5f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, choices=[0, 1])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = verify_corpus(args.data)
    torch.set_num_threads(1)
    for seed in ([args.seed] if args.seed is not None else [0, 1]):
        # Counterbalance run order, because host load and heat can drift.
        order = list(ARMS) if seed == 0 else list(reversed(ARMS))
        for name in ([args.arm] if args.arm else order):
            run_arm(name, seed, args.data, args.output, torch.device(args.device), manifest)


if __name__ == "__main__":
    main()
