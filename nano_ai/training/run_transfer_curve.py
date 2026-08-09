"""P0 -- the transfer curve: smallest pretrained base that recovers transfer.

Preregistration: `papers/PREREG_TRANSFER_CURVE.md` (bar 75.0% held-out mean,
frozen before any point was measured). SCREENING, one seed per base; coarse
conclusions only.

Holds everything constant except the pretrained base: same fit partition, same
LoRA config, same arms, same documents, same balanced control block. Each point
is train-then-evaluate so a partial run still yields usable points.

    python3 -m nano_ai.training.run_transfer_curve \
        --out artifacts/nano_h6/transfer_curve
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Frozen in the preregistration.
BAR = 0.75
LORA = dict(num_layers=8, batch_size=4, iters=400, max_seq_length=768, seed=20260806)

BASES = [
    # All verified 2026-08-06: public, ungated, license apache-2.0 via the HF API.
    # mlx_lm converts HF safetensors on load, so the canonical repos are used
    # rather than mlx-community mirrors (the 4-bit mirror names I first guessed
    # do not exist; recorded so the next session does not repeat it).
    ("SmolLM2-135M", "HuggingFaceTB/SmolLM2-135M-Instruct", "Apache-2.0"),
    ("SmolLM2-360M", "HuggingFaceTB/SmolLM2-360M-Instruct", "Apache-2.0"),
    ("Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct", "Apache-2.0"),
    ("Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct", "Apache-2.0"),
    ("SmolLM2-1.7B", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "Apache-2.0"),
]


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("artifacts/nano_h6/lora_control/data"))
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--only", default=None, help="substring filter on base label")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    points = []
    for label, repo, license_ in BASES:
        if args.only and args.only.lower() not in label.lower():
            continue
        adapters = args.out / f"adapters_{label}"
        result_path = args.out / f"probe_{label}.json"
        print(f"\n=== {label} ({repo}, {license_}) ===", flush=True)

        started = time.time()
        code, log = _run([
            sys.executable, "-m", "mlx_lm", "lora",
            "--model", repo, "--train", "--data", str(args.data),
            "--fine-tune-type", "lora",
            "--num-layers", str(LORA["num_layers"]),
            "--batch-size", str(LORA["batch_size"]),
            "--iters", str(LORA["iters"]),
            "--max-seq-length", str(LORA["max_seq_length"]),
            "--seed", str(LORA["seed"]),
            "--steps-per-report", "100", "--steps-per-eval", "400",
            "--adapter-path", str(adapters),
        ])
        train_s = time.time() - started
        if code != 0:
            print(f"  TRAIN FAILED ({train_s:.0f}s): {log.strip().splitlines()[-1][:160]}")
            points.append({"base": label, "repo": repo, "license": license_,
                           "status": "train_failed", "error": log.strip()[-400:]})
            (args.out / "curve.json").write_text(
                json.dumps({"bar": BAR, "lora": LORA, "points": points},
                           indent=2, sort_keys=True) + "\n")
            continue
        val = [l for l in log.splitlines() if "Val loss" in l]
        print(f"  trained in {train_s:.0f}s   {val[-1].strip() if val else ''}")

        code, log = _run([
            sys.executable, "-m", "nano_ai.training.run_crossmodel_surface_probe",
            "--model", repo, "--adapter-path", str(adapters),
            "--limit", str(args.limit), "--mode", "direct",
            "--output", str(result_path),
        ])
        if code != 0:
            print(f"  PROBE FAILED: {log.strip().splitlines()[-1][:160]}")
            points.append({"base": label, "repo": repo, "license": license_,
                           "status": "probe_failed", "error": log.strip()[-400:]})
        else:
            data = json.loads(result_path.read_text())
            held = data["summary"]["nano_held_out_arms"]["mean"]
            ctrl = data["control"]
            interpretable = data["arm_accuracies_interpretable"]
            points.append({
                "base": label, "repo": repo, "license": license_,
                "status": "ok",
                "train_seconds": round(train_s, 1),
                "held_out_mean": held,
                "in_distribution_mean": data["summary"]["nano_in_distribution_arms"]["mean"],
                "dev_arm": data["summary"]["dev_arm"],
                "control_accuracy": ctrl["accuracy"],
                "worst_class_recall": ctrl["worst_class_recall"],
                "interpretable": interpretable,
                "clears_bar": bool(interpretable and held >= BAR),
            })
            p = points[-1]
            print(f"  held-out {held:.1%}  in-dist {p['in_distribution_mean']:.1%}  "
                  f"control {ctrl['accuracy']:.1%} (worst class {ctrl['worst_class_recall']:.2f})  "
                  f"interpretable={interpretable}  CLEARS BAR={p['clears_bar']}")

        (args.out / "curve.json").write_text(
            json.dumps({"bar": BAR, "lora": LORA,
                        "preregistration": "papers/PREREG_TRANSFER_CURVE.md",
                        "status": "SCREENING -- one seed per base; coarse conclusions only",
                        "points": points}, indent=2, sort_keys=True) + "\n")

    print(f"\n=== curve (bar {BAR:.0%}) ===")
    for p in points:
        if p.get("status") != "ok":
            print(f"  {p['base']:14s} {p['status']}")
        else:
            print(f"  {p['base']:14s} held-out {p['held_out_mean']:6.1%}  "
                  f"{'CLEARS' if p['clears_bar'] else 'below '} bar  "
                  f"{'' if p['interpretable'] else '(UNINTERPRETABLE)'}")
    print(f"\nwrote {args.out}/curve.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
