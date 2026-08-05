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

BUNDLE = "/kaggle/input/nano-h6-frozen-input/nano-h6-input.bin"
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
run([sys.executable, "-m", "venv", "--system-site-packages", ".h6-venv"])
VPY = os.path.join(ROOT, ".h6-venv", "bin", "python")
exact_ok = False
try:
    run([VPY, "-m", "pip", "install", "--no-cache-dir", "--quiet",
         "torch==2.8.0", "--index-url", "https://download.pytorch.org/whl/cu128"])
    smoke = subprocess.run(
        [VPY, "-c",
         "import torch; assert torch.__version__=='2.8.0+cu128';"
         "assert torch.cuda.is_available();"
         "a=torch.randn(64,64,device='cuda');"
         "assert float((a@a).sum())==float((a@a).sum())"],
        capture_output=True, text=True, timeout=300)
    exact_ok = smoke.returncode == 0
    if not exact_ok:
        print("EXACT PIN SMOKE FAILED:", smoke.stderr[-500:], flush=True)
except Exception as e:  # noqa: BLE001
    print("EXACT PIN INSTALL FAILED:", e, flush=True)
if not exact_ok:
    # Disclosed fallback: rebuild venv on the Kaggle-preinstalled torch.
    run(["rm", "-rf", ".h6-venv"])
    run([sys.executable, "-m", "venv", "--system-site-packages", ".h6-venv"])
run([VPY, "-m", "pip", "install", "--no-cache-dir", "--quiet",
     "pytest==9.0.2", "tokenizers==0.22.2"])

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
        subprocess.run(
            [VPY, "-m", "nano_ai.training.train_evidence_query_h6",
             "--data-dir", "h6_data",
             "--base-checkpoint", "checkpoints/anchors/nano_v01_scribe.pt",
             "--tokenizer", "sft/tokenizer.json",
             "--output-dir", f"results/seed-{seed}",
             "--seed", seed, "--device", "cuda"],
            check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
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
