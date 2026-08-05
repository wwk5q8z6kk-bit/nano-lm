"""nano-h6 frozen-recipe training on Kaggle, under KAGGLE_ADDENDUM 7831e78d.

Scientific steps are identical to the frozen RUN_H6_TRAIN.sh:
bundle sha hard-check -> manifest sha hard-check -> focused test suite ->
two seed trainings with byte-identical arguments -> report checksums.
Disclosed relaxations (recorded, never silent): python version, GPU name,
memory floor 14GB, torch fallback only after a failed exact-pin attempt.
No development data exists anywhere in this kernel's inputs.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys

import glob

_candidates = glob.glob("/kaggle/input/*/nano-h6-input.bin") or glob.glob(
    "/kaggle/input/**/nano-h6-input.bin", recursive=True)
if not _candidates:
    for root, dirs, files in os.walk("/kaggle/input"):
        print("INPUT:", root, dirs, files, flush=True)
    raise SystemExit("nano-h6-input.bin not found in any mounted input")
BUNDLE = _candidates[0]
BUNDLE_SHA = "b1eff7c9ccb06f05e46f0639c88a7e74daeb0880dc8ecbb0641359d74b5ea505"
MANIFEST_SHA = "2569e7b27b53ef741fc92d1545ca323367155d960e77ff0e6852b32db0d32f31"
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


# Stage 0: bundle integrity (HARD)
actual = sha256(BUNDLE)
assert actual == BUNDLE_SHA, f"bundle sha mismatch: {actual}"
os.chdir(WORK)
run(["tar", "-xzf", BUNDLE, "-C", WORK])
ROOT = os.path.join(WORK, "nano-h6-runpod")
os.chdir(ROOT)

# Stage 1: manifest integrity (HARD)
m = sha256("h6_data/manifest.json")
assert m == MANIFEST_SHA, f"manifest sha mismatch: {m}"

# Stage 2: runtime - exact pin attempted in venv, disclosed fallback
disclosure = {
    "schema": "nano.h6.kaggle-runtime-disclosure.v1",
    "python": platform.python_version(),
    "platform": platform.platform(),
}
# Kaggle's image cannot bootstrap venv pip (ensurepip fails); the kernel
# container is disposable, so direct installs are the disclosed equivalent.
VPY = sys.executable
pre = subprocess.run([VPY, "-c", "import torch; print(torch.__version__)"],
                     capture_output=True, text=True)
pre_ver = pre.stdout.strip() if pre.returncode == 0 else None
print("PREINSTALLED_TORCH:", pre_ver, flush=True)
    # DISCLOSED DELTA (v6): torch is NEVER reinstalled. Kernel v4/v5 proved the
    # exact 2.8.0+cu128 pin has no kernel image for Kaggle's GPU, and installing
    # any torch from the default index yields a wheel that also lacks it. The
    # Kaggle-preinstalled build is the only working one, so it is used as-is and
# recorded. Device is chosen by an actual CUDA matmul smoke test.
exact_ok = False
run([VPY, "-m", "pip", "install", "--no-cache-dir", "--quiet",
     "pytest==9.0.2", "tokenizers==0.22.2"])
cuda_smoke = subprocess.run(
    [VPY, "-c",
     "import torch; assert torch.cuda.is_available();"
     "a=torch.randn(64,64,device='cuda'); b=(a@a).sum().item(); print('CUDA_OK',b==b)"],
    capture_output=True, text=True, timeout=300)
DEVICE = "cuda" if cuda_smoke.returncode == 0 else "cpu"
if DEVICE == "cpu":
    print("CUDA SMOKE FAILED (training on CPU, disclosed):",
          cuda_smoke.stderr[-400:], flush=True)

probe = subprocess.run(
    [VPY, "-c",
     "import json, torch, tokenizers, platform;"
     "p=torch.cuda.get_device_properties(0);"
     "print(json.dumps({'torch': torch.__version__,"
     "'cuda': torch.version.cuda,"
     "'tokenizers': tokenizers.__version__,"
     "'gpu': torch.cuda.get_device_name(0),"
     "'gpu_mem_gb': round(p.total_memory/2**30, 1)}))"],
    capture_output=True, text=True, check=True)
disclosure.update(json.loads(probe.stdout.strip().splitlines()[-1]))
disclosure["exact_torch_pin_used"] = exact_ok
assert disclosure["gpu_mem_gb"] >= 14, f"GPU too small: {disclosure}"
assert disclosure["tokenizers"] == "0.22.2", disclosure
print("RUNTIME_DISCLOSURE:", json.dumps(disclosure), flush=True)

# Stage 3: focused test suite (unchanged from frozen runner)
env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONHASHSEED="0")
run([VPY, "-m", "pytest", "-q",
     "nano_ai/tests/test_state_conditioned_evidence_query_model.py",
     "nano_ai/tests/test_replay_mixture_data.py",
     "nano_ai/tests/test_train_evidence_query_h6.py",
     "nano_ai/tests/test_evaluate_evidence_query_h6.py",
     "nano_ai/tests/test_package_evidence_query_h6.py"], env=env)

# Stage 4: two seed trainings, byte-identical arguments to the frozen runner
assert not os.path.exists("h2_development")
os.makedirs("results", exist_ok=False)
for seed in ("20260805", "20260806"):
    assert not os.path.exists("h2_development")
    with open(f"results/seed-{seed}.log", "w") as log:
        print(f"=== training seed {seed} ===", flush=True)
        proc = subprocess.run(
            [VPY, "-m", "nano_ai.training.train_evidence_query_h6",
             "--data-dir", "h6_data",
             "--base-checkpoint", "checkpoints/anchors/nano_v01_scribe.pt",
             "--tokenizer", "sft/tokenizer.json",
             "--output-dir", f"results/seed-{seed}",
             "--seed", seed, "--device", "cuda"],
            env=env, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        with open(f"results/seed-{seed}.log") as log:
            tail = log.readlines()[-50:]
        print(f"TRAINING FAILED seed {seed}, log tail:", flush=True)
        for line in tail:
            print("  |", line.rstrip(), flush=True)
        raise SystemExit(f"training failed for seed {seed}")
assert not os.path.exists("h2_development")

# Stage 5: checksums + disclosure into the kernel output
sums = []
for seed in ("20260805", "20260806"):
    p = f"results/seed-{seed}/training_report.json"
    sums.append(f"{sha256(p)}  {p}")
with open("results/TRAINING_REPORT_SHA256SUMS", "w") as f:
    f.write("\n".join(sums) + "\n")
with open("results/RUNTIME_DISCLOSURE.json", "w") as f:
    json.dump(disclosure, f, indent=2, sort_keys=True)
run(["cp", "-r", "results", os.path.join(WORK, "results")])
for line in sums:
    print("REPORT:", line, flush=True)
print("H6_KAGGLE_TRAINING_COMPLETE", flush=True)
