"""nano-h6 ONE-SHOT development evaluation on Kaggle.

Consumes the sealed development partition exactly once. Mirrors the frozen
RUN_H6_EVALUATE.sh: verify training reports -> hard-check development hashes ->
run the unchanged evaluator once -> emit results with checksums.

Inputs (all content-addressed, all hard-asserted before development is opened):
  - frozen package bundle          sha b1eff7c9...
  - training kernel output         (kernel_sources chain, preserves provenance)
  - sealed development partition   dev sha 9c893d8e..., manifest sha 47ee157a...

Device is CPU because training ran on CPU (Kaggle's P100 is sm_60, unsupported
by torch 2.10+cu128). The frozen protocol requires evaluation on the same
runtime as training; same-kernel-class CPU satisfies that by construction.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import platform
import subprocess
import sys

BUNDLE_SHA = "b1eff7c9ccb06f05e46f0639c88a7e74daeb0880dc8ecbb0641359d74b5ea505"
TRAIN_MANIFEST_SHA = "2569e7b27b53ef741fc92d1545ca323367155d960e77ff0e6852b32db0d32f31"
DEV_SHA = "9c893d8e64110287b433d567e0e9abb42c611ecba33b40de192741324d37e290"
DEV_MANIFEST_SHA = "47ee157ac037c0771100b8546c90da91dbd2006198700bb642f1561d2124c1a3"
REPORT_SHA = {
    "20260805": "9bf3d8281a50d8d85b01e7b72ff5180ba42a554e2f9092a42d37cdf395e572b8",
    "20260806": "87da86a5793e6ba04295bb52fe453a32aa924243711173eff54251bccaf06843",
}
WORK = "/kaggle/working"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd, **kw):
    print(f"+ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=True, **kw)


def find_one(patterns: list[str], what: str) -> str:
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    for root, dirs, files in os.walk("/kaggle/input"):
        print("INPUT:", root, dirs[:6], files[:6], flush=True)
    raise SystemExit(f"could not locate {what}")


# Stage 0: frozen package
bundle = find_one(["/kaggle/input/**/nano-h6-input.bin"], "frozen bundle")
assert sha256(bundle) == BUNDLE_SHA, "bundle sha mismatch"
os.chdir(WORK)
run(["tar", "-xzf", bundle, "-C", WORK])
ROOT = os.path.join(WORK, "nano-h6-runpod")
os.chdir(ROOT)
assert sha256("h6_data/manifest.json") == TRAIN_MANIFEST_SHA, "training manifest mismatch"

# Stage 1: training output from the training kernel (provenance chain)
train_sums = find_one(
    ["/kaggle/input/**/results/TRAINING_REPORT_SHA256SUMS"], "training results"
)
train_results = os.path.dirname(train_sums)
run(["cp", "-r", train_results, os.path.join(ROOT, "results")])
for seed, expected in REPORT_SHA.items():
    got = sha256(f"results/seed-{seed}/training_report.json")
    assert got == expected, f"training report {seed} mismatch: {got}"
print("training reports authenticated", flush=True)

# Stage 2: runtime (must match what training recorded)
run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--quiet",
     "pytest==9.0.2", "tokenizers==0.22.2"])
probe = subprocess.run(
    [sys.executable, "-c",
     "import json, torch, tokenizers;"
     "print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda,"
     "'tokenizers': tokenizers.__version__}))"],
    capture_output=True, text=True, check=True)
runtime = json.loads(probe.stdout.strip().splitlines()[-1])
runtime.update({"python": platform.python_version(), "platform": platform.platform(),
                "device": "cpu", "schema": "nano.h6.kaggle-eval-disclosure.v1"})
assert runtime["tokenizers"] == "0.22.2", runtime
print("EVAL_RUNTIME:", json.dumps(runtime), flush=True)

# Stage 3: DEVELOPMENT RELEASE — the one-shot boundary
dev_src = os.path.dirname(find_one(["/kaggle/input/**/dev.jsonl"], "development partition"))
os.makedirs("h2_development", exist_ok=False)
run(["cp", os.path.join(dev_src, "dev.jsonl"),
     os.path.join(dev_src, "manifest.json"), "h2_development/"])
assert sha256("h2_development/dev.jsonl") == DEV_SHA, "DEVELOPMENT SHA MISMATCH"
assert sha256("h2_development/manifest.json") == DEV_MANIFEST_SHA, "dev manifest mismatch"
print("development opened once, hashes authenticated", flush=True)

# Stage 4: the unchanged evaluator, exactly once
env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONHASHSEED="0")
assert not os.path.exists("results/development_evaluation.json")
proc = subprocess.run(
    [sys.executable, "-m", "nano_ai.training.evaluate_evidence_query_h6",
     "--training-data-dir", "h6_data",
     "--training-manifest-sha256", TRAIN_MANIFEST_SHA,
     "--development-data-dir", "h2_development",
     "--development-manifest-sha256", DEV_MANIFEST_SHA,
     "--tokenizer", "sft/tokenizer.json",
     "--training-report", "results/seed-20260805/training_report.json", REPORT_SHA["20260805"],
     "--training-report", "results/seed-20260806/training_report.json", REPORT_SHA["20260806"],
     "--output", "results/development_evaluation.json",
     "--device", "cpu", "--batch-size", "32"],
    env=env, capture_output=True, text=True)
print(proc.stdout[-4000:], flush=True)
if proc.returncode != 0:
    print("EVALUATOR FAILED:", proc.stderr[-4000:], flush=True)
    raise SystemExit("evaluation failed")

# Stage 5: seal the output
with open("results/EVAL_RUNTIME_DISCLOSURE.json", "w") as f:
    json.dump(runtime, f, indent=2, sort_keys=True)
out_files = sorted(
    p for p in glob.glob("results/**/*", recursive=True) if os.path.isfile(p)
)
with open("SHA256SUMS", "w") as f:
    for p in out_files:
        f.write(f"{sha256(p)}  {p}\n")
run(["cp", "-r", "results", os.path.join(WORK, "results")])
run(["cp", "SHA256SUMS", os.path.join(WORK, "SHA256SUMS")])
print("RESULT_SHA:", sha256("results/development_evaluation.json"), flush=True)
print("H6_ONE_SHOT_EVALUATION_COMPLETE", flush=True)
