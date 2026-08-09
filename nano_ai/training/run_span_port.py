"""Train then probe the route-(b) span port on one pretrained base.

Preregistration: `papers/PREREG_SPAN_PORT_B.md`.

    python3 -m nano_ai.training.run_span_port \
        --out artifacts/nano_h6/span_port
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Frozen recipe — identical to papers/PREREG_TRANSFER_CURVE.md / span-port prereg.
LORA = dict(num_layers=8, batch_size=4, iters=400, max_seq_length=768, seed=20260806)
DEFAULT_BASE = ("Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct", "Apache-2.0")


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("artifacts/nano_h6/lora_control/data_spans_balanced"),
    )
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--model-label", default=DEFAULT_BASE[0])
    parser.add_argument("--model-repo", default=DEFAULT_BASE[1])
    parser.add_argument("--license", default=DEFAULT_BASE[2])
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="reuse existing adapters under --out",
    )
    args = parser.parse_args()

    if not (args.data / "train.jsonl").is_file():
        raise SystemExit(
            f"missing {args.data}/train.jsonl — build with:\n"
            "  python3 -m nano_ai.training.build_lora_control_data "
            f"--out {args.data} --with-spans --balance"
        )
    provenance = json.loads((args.data / "PROVENANCE.json").read_text())
    if provenance.get("target_format") != "state_and_span_text":
        raise SystemExit(
            f"{args.data} is not span-bearing "
            f"(target_format={provenance.get('target_format')!r})"
        )
    if provenance.get("prompt_format") != "state_and_span_line":
        raise SystemExit(
            f"{args.data} still uses the state-only prompt "
            f"(prompt_format={provenance.get('prompt_format')!r}); rebuild"
        )
    if not provenance.get("balanced"):
        raise SystemExit(
            f"{args.data} is not balanced — refusing to run the falsifier on "
            "the same minority-class confound the transfer curve hit"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    adapters = args.out / f"adapters_{args.model_label}"
    result_path = args.out / f"probe_{args.model_label}.json"
    summary_path = args.out / "SUMMARY.json"

    train_s = None
    if not args.skip_train:
        print(f"=== train {args.model_label} ({args.model_repo}) ===", flush=True)
        started = time.time()
        code, log = _run(
            [
                sys.executable,
                "-m",
                "mlx_lm",
                "lora",
                "--model",
                args.model_repo,
                "--train",
                "--data",
                str(args.data),
                "--fine-tune-type",
                "lora",
                "--num-layers",
                str(LORA["num_layers"]),
                "--batch-size",
                str(LORA["batch_size"]),
                "--iters",
                str(LORA["iters"]),
                "--max-seq-length",
                str(LORA["max_seq_length"]),
                "--seed",
                str(LORA["seed"]),
                "--steps-per-report",
                "100",
                "--steps-per-eval",
                "400",
                "--adapter-path",
                str(adapters),
            ]
        )
        train_s = round(time.time() - started, 1)
        if code != 0:
            summary_path.write_text(
                json.dumps(
                    {
                        "status": "train_failed",
                        "error": log.strip()[-800:],
                        "train_seconds": train_s,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            print(log.strip().splitlines()[-5:])
            return code
        val = [line for line in log.splitlines() if "Val loss" in line]
        print(f"  trained in {train_s}s   {val[-1].strip() if val else ''}")
    elif not adapters.is_dir():
        raise SystemExit(f"--skip-train set but adapters missing: {adapters}")

    print(f"=== probe {args.model_label} ===", flush=True)
    code, log = _run(
        [
            sys.executable,
            "-m",
            "nano_ai.training.run_span_port_probe",
            "--model",
            args.model_repo,
            "--adapter-path",
            str(adapters),
            "--limit",
            str(args.limit),
            "--output",
            str(result_path),
        ]
    )
    if code != 0:
        summary_path.write_text(
            json.dumps(
                {
                    "status": "probe_failed",
                    "error": log.strip()[-800:],
                    "train_seconds": train_s,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(log[-2000:])
        return code

    probe = json.loads(result_path.read_text())
    summary = {
        "schema": "nano.span-port-run.v1",
        "preregistration": "papers/PREREG_SPAN_PORT_B.md",
        "status": "ok",
        "base": args.model_label,
        "repo": args.model_repo,
        "license": args.license,
        "data": str(args.data),
        "data_provenance": {
            "balanced": provenance.get("balanced"),
            "target_format": provenance.get("target_format"),
            "prompt_format": provenance.get("prompt_format"),
            "label_mix": provenance.get("label_mix"),
        },
        "lora": LORA,
        "train_seconds": train_s,
        "interpretable": probe["interpretable"],
        "decision": probe["decision"],
        "span_relocation": probe["span_relocation"],
        "state_held_out_mean": (
            probe["state_summary"]["nano_held_out_arms"] or {}
        ).get("mean"),
        "control_worst_class_recall": probe["control"]["worst_class_recall"],
        "probe_path": str(result_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
