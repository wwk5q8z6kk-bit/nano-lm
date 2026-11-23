"""Small source-only witnesses. No checkpoint, tokenizer binary or training corpus."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1] / "evidence/upstream_v6/context"
SOURCE = ROOT / "sources" / "3a38b5f1" / "research" / "frontier" / "ontogenesis"
scorer = SOURCE / "tpu" / "eval_score.py"
tree = ast.parse(scorer.read_text())
entries_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "entries")
namespace = {"T": 512, "EOS": 1}
exec(compile(ast.Module(body=[entries_node], type_ignores=[]), str(scorer), "exec"), namespace)
entries = namespace["entries"]

normal = entries("lambada", [{"ctx": [10, 11], "conts": [[12, 13]]}])[0]
normal_targets = normal[3][1:][normal[4]:normal[5]]
long = entries("lambada", [{"ctx": list(range(2, 602)), "conts": [[700, 701]]}])[0]
long_targets = long[3][1:][long[4]:long[5]]
wino = entries("winogrande", [{"ctxs": [[10], [20]], "cont": [30, 31]}], nocontext=True)
empty = entries("lambada", [{"ctx": [10], "conts": [[]]}])[0]
empty_targets = empty[3][1:][empty[4]:empty[5]]
too_long_rejected = False
try:
    entries("lambada", [{"ctx": [10], "conts": [list(range(513))]}])
except AssertionError:
    too_long_rejected = True

native = json.loads(next((SOURCE / "results" / "scale2").glob("*.bench.json")).read_text())["tasks"]
native_nc = json.loads(next((SOURCE / "results" / "scale2").glob("*.bench-nocontext.json")).read_text())["tasks"]
smol_nc = json.loads((SOURCE / "results" / "lm" / "eval-reference-SmolLM2-135M-nocontext.json").read_text())["tasks"]

with tempfile.TemporaryDirectory(prefix="q13-source-witness-") as root:
    root = Path(root)
    run = {"name": "SYNTHETIC_RUN_A", "finished": True, "status": "finished", "steps": 1,
           "curve": [{"step": 1, "mix": .8049, "web": .8, "textbook": .8, "math": .8, "code": .8}]}
    bench = {"ckpt": "FOREIGN_SYNTHETIC_CHECKPOINT_B", "tasks": {
        "arc_easy": {"acc": .59}, "piqa": {"acc": .677}, "lambada": {"acc": .35, "ppl": 32}}}
    xl = {"tasks": {"arc_easy": {"acc": .55}, "piqa": {"acc": .66}}}
    paths = []
    for name, value in (("run.json", run), ("bench.json", bench), ("xl.json", xl)):
        path = root / name
        path.write_text(json.dumps(value))
        paths.append(str(path))
    process = subprocess.run([sys.executable, str(SOURCE / "tpu" / "score_q13.py"), *paths],
                             capture_output=True, text=True)
    verdict = json.loads(process.stdout) if process.returncode == 0 else None

report = {
    "classification": "SOURCE_ONLY_SYNTHETIC_DIAGNOSTICS_NOT_MODEL_RESULTS",
    "research_cutoff": "3a38b5f1529a718838b5bc83a010574bfa9a4c6e",
    "model_execution": "NOT_RUN",
    "actual_tokenizer_or_corpus_access": "NOT_AVAILABLE",
    "entries_witnesses": {
        "normal_target_alignment_correct": normal_targets == [12, 13],
        "left_crop_preserves_target_alignment": long_targets == [700, 701],
        "left_crop_removes_initial_EOS": long[3][0] != 1,
        "long_sequence_length": len(long[3]),
        "winogrande_no_context_identical_options": wino[0][3:] == wino[1][3:],
        "empty_continuation_is_accepted": empty_targets == [],
        "empty_greedy_all_is_true": all(empty_targets),
        "oversized_continuation_rejected": too_long_rejected,
    },
    "committed_aggregate_arithmetic": {
        "task_n": {name: value["n"] for name, value in native.items()},
        "sum_task_n": sum(value["n"] for value in native.values()),
        "investigation_narrative_n": 30328,
        "arc_easy_no_context_native_minus_smol135m": round(native_nc["arc_easy"]["acc"] - smol_nc["arc_easy"]["acc"], 4),
    },
    "q13_identity_binding_witness": {
        "synthetic_run_name": run["name"],
        "synthetic_benchmark_checkpoint": bench["ckpt"],
        "mismatched_identity_accepted": process.returncode == 0,
        "scorer_reported_run": verdict["run"] if verdict else None,
        "historical_misbinding_demonstrated": False,
        "interpretation": "The scorer lacks identity/hash binding. This synthetic witness does not establish a historical benchmark mismatch.",
    },
    "scope": [
        "AST-executed entries() only; no JAX import or model.",
        "Malformed/empty continuation witness is a schema robustness gap, not an observed real task row.",
        "Boundary tokenization discrepancy is a conditional code risk; actual tokenizer bytes were unavailable.",
        "No historical score changed by these diagnostics.",
    ],
}
assert all(report["entries_witnesses"][k] for k in (
    "normal_target_alignment_correct", "left_crop_preserves_target_alignment",
    "left_crop_removes_initial_EOS", "winogrande_no_context_identical_options",
    "empty_continuation_is_accepted", "empty_greedy_all_is_true", "oversized_continuation_rejected"))
assert report["committed_aggregate_arithmetic"]["sum_task_n"] == 22348
assert report["q13_identity_binding_witness"]["mismatched_identity_accepted"]
report["diagnostic_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
print(json.dumps(report, indent=2, allow_nan=False))
