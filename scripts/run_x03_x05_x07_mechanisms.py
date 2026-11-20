#!/usr/bin/env python3
"""Run local 3-seed $0 mechanisms suite (X03, X04, X05, X07) on CPU (`float32`).

Writes immutable results, SHA256SUMS, and an empirical summary report to
`artifacts/foundation_mech_x03_x07_v1/`.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nano.mechanisms.dual_memory import DualMemoryController  # noqa: E402
from nano.mechanisms.recurrent_cells import (  # noqa: E402
    ToySequenceModel,
    compute_transition_and_jacobian_spectrum,
)

SEEDS = (101, 202, 303)
OUT_DIR = ROOT / "artifacts" / "foundation_mech_x03_x07_v1"

# Precomputed S_3 Cayley table (6 elements; generators 0=id, 1=(01), 2=(12), 3=(02))
_S3_PERMS = [(0, 1, 2), (1, 0, 2), (0, 2, 1), (2, 1, 0), (1, 2, 0), (2, 0, 1)]
_S3_ID = {p: i for i, p in enumerate(_S3_PERMS)}
_S3_MUL = torch.zeros(6, 6, dtype=torch.long)
for _i, _p1 in enumerate(_S3_PERMS):
    for _j, _p2 in enumerate(_S3_PERMS):
        _S3_MUL[_i, _j] = _S3_ID[tuple(_p2[_p1[k]] for k in range(3))]


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def _mean_std(vals: list[float]) -> dict[str, Any]:
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return {"mean": round(m, 4), "std": round(s, 4), "values": [round(v, 4) for v in vals]}


def _make_group_batch(task: str, bsz: int, seq_len: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    if task == "z2_parity":
        ops = torch.randint(0, 2, (bsz, seq_len), generator=gen)
        return ops, ops.cumsum(dim=-1) % 2
    if task == "z4_rotation":
        # 0=stay, 1=+1, 2=+2, 3=-1 mod 4
        ops = torch.randint(0, 4, (bsz, seq_len), generator=gen)
        return ops, ops.cumsum(dim=-1) % 4
    if task == "s3_nonabelian":
        ops = torch.randint(0, 4, (bsz, seq_len), generator=gen)
        states = torch.zeros(bsz, seq_len, dtype=torch.long)
        cur = torch.zeros(bsz, dtype=torch.long)
        for t in range(seq_len):
            cur = _S3_MUL[cur, ops[:, t]]
            states[:, t] = cur
        return ops, states
    raise ValueError(f"unknown group task: {task}")


def run_x03_group_and_jacobians() -> dict[str, Any]:
    """X03: 3-seed Z_2, Z_4, and non-commutative S_3 extrapolation + Jacobian conditioning."""
    print("=== X03: Group State Tracking & Full-Jacobian Conditioning ===", flush=True)
    arms = (
        "contraction_delta",
        "householder_delta",
        "householder_rank2",
        "gdn2",
        "complex_roto_ssm",
        "geo_erase_write",
    )
    tasks = {
        "z2_parity": {"layers": 1, "vocab": 4, "classes": 2, "d": 16, "heads": 2, "steps": 200, "lr": 8e-3},
        "z4_rotation": {"layers": 1, "vocab": 4, "classes": 4, "d": 16, "heads": 2, "steps": 240, "lr": 8e-3},
        "s3_nonabelian": {"layers": 2, "vocab": 4, "classes": 6, "d": 16, "heads": 2, "steps": 260, "lr": 6e-3},
    }
    out: dict[str, Any] = {}
    for task_name, cfg in tasks.items():
        task_res: dict[str, Any] = {}
        for arm in arms:
            id_accs, ood4_accs, ood16_accs = [], [], []
            jac_maxs, a_maxs, param_count = [], [], 0
            for seed in SEEDS:
                _set_seed(seed)
                model = ToySequenceModel(
                    vocab_size=cfg["vocab"],
                    d_model=cfg["d"],
                    n_heads=cfg["heads"],
                    layer_names=(arm,) * cfg["layers"],
                    num_classes=cfg["classes"],
                    use_ffn=False,
                    learnable_init_state=True,
                )
                param_count = model.total_parameters()
                opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.0)
                gen = torch.Generator().manual_seed(seed + 1000)
                model.train()
                for _ in range(cfg["steps"]):
                    xb, yb = _make_group_batch(task_name, 48, 32, gen)
                    logits = model(xb)
                    loss = F.cross_entropy(logits.view(-1, cfg["classes"]), yb.view(-1))
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

                model.eval()
                eval_gen = torch.Generator().manual_seed(seed + 9000)
                with torch.no_grad():
                    for length, bucket in ((32, id_accs), (128, ood4_accs), (512, ood16_accs)):
                        xe, ye = _make_group_batch(task_name, 48, length, eval_gen)
                        pred = model(xe).argmax(dim=-1)
                        bucket.append(float((pred == ye).float().mean().item()))

                xe_probe, _ = _make_group_batch(task_name, 1, 8, eval_gen)
                h_probe = model.norms[0](model.emb(xe_probe[:, 0]))
                s_probe = model.init_states["0"]
                spec = compute_transition_and_jacobian_spectrum(model.mixers[0], h_probe, s_probe)
                jac_maxs.append(spec["full_step_jacobian_sigma_max"])
                if "homogeneous_A_sigma_max" in spec:
                    a_maxs.append(spec["homogeneous_A_sigma_max"])

            task_res[arm] = {
                "params": param_count,
                "id_T32_acc": _mean_std(id_accs),
                "ood_4x_T128_acc": _mean_std(ood4_accs),
                "ood_16x_T512_acc": _mean_std(ood16_accs),
                "full_jacobian_sigma_max": _mean_std(jac_maxs),
                "homogeneous_A_sigma_max": _mean_std(a_maxs) if a_maxs else None,
            }
            print(
                f"  {task_name:13s} | {arm:18s} | params={param_count:5d} | "
                f"ID={task_res[arm]['id_T32_acc']['mean']:.4f} | "
                f"4x={task_res[arm]['ood_4x_T128_acc']['mean']:.4f} | "
                f"16x={task_res[arm]['ood_16x_T512_acc']['mean']:.4f} | "
                f"J_max={task_res[arm]['full_jacobian_sigma_max']['mean']:.3f}",
                flush=True,
            )
        out[task_name] = task_res

    # Zero-init vs learnable-S0 ablation on 1-layer Z_4 for ComplexRotoSSM (documents why S0 != 0 matters)
    z4_zero_init: list[float] = []
    for seed in SEEDS:
        _set_seed(seed)
        model = ToySequenceModel(
            vocab_size=4, d_model=16, n_heads=2, layer_names=("complex_roto_ssm",),
            num_classes=4, use_ffn=False, learnable_init_state=False,
        )
        opt = torch.optim.AdamW(model.parameters(), lr=8e-3, weight_decay=0.0)
        gen = torch.Generator().manual_seed(seed + 1000)
        model.train()
        for _ in range(240):
            xb, yb = _make_group_batch("z4_rotation", 48, 32, gen)
            loss = F.cross_entropy(model(xb).view(-1, 4), yb.view(-1))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            xe, ye = _make_group_batch("z4_rotation", 48, 128, torch.Generator().manual_seed(seed + 9000))
            z4_zero_init.append(float((model(xe).argmax(-1) == ye).float().mean().item()))
    out["z4_complex_zero_s0_control"] = {"ood_4x_T128_acc": _mean_std(z4_zero_init)}
    print(f"  z4_zero_s0    | complex_roto_ssm   | 4x={out['z4_complex_zero_s0_control']['ood_4x_T128_acc']['mean']:.4f}", flush=True)
    return out


def _make_edit_batch(
    bsz: int, seq_len: int, gen: torch.Generator, *, n_keys: int = 6, n_vals: int = 6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Delayed multi-key write, overwrite, selective erase, distractor gap, and query stream."""
    # Vocab: 0..35=WRITE(k,v), 36..41=ERASE(k), 42=NOISE, 43..48=QUERY(k)
    n_write = n_keys * n_vals
    erase_base = n_write
    noise_tok = erase_base + n_keys
    query_base = noise_tok + 1

    xb = torch.full((bsz, seq_len), noise_tok, dtype=torch.long)
    yb = torch.full((bsz, seq_len), -100, dtype=torch.long)
    p1_end = int(seq_len * 0.45)
    p2_end = int(seq_len * 0.80)
    for b in range(bsz):
        mem = [0] * n_keys  # 0 = erased/missing, 1..n_vals = active value
        # Phase 1: write, overwrite, and selective erase
        for t in range(p1_end):
            if t < n_keys or float(torch.rand(1, generator=gen).item()) < 0.72:
                k = int(torch.randint(0, n_keys, (1,), generator=gen).item())
                v = int(torch.randint(0, n_vals, (1,), generator=gen).item())
                xb[b, t] = k * n_vals + v
                mem[k] = v + 1
            else:
                k = int(torch.randint(0, n_keys, (1,), generator=gen).item())
                xb[b, t] = erase_base + k
                mem[k] = 0
        # Phase 2: pure distractor delay ([p1_end .. p2_end))
        # Phase 3: delayed queries across all keys
        for t in range(p2_end, seq_len):
            k = int(torch.randint(0, n_keys, (1,), generator=gen).item())
            xb[b, t] = query_base + k
            yb[b, t] = mem[k]
    return xb, yb


def run_x04_erase_write_benchmark() -> dict[str, Any]:
    """X04: 3-seed delayed multi-key overwrite & selective erasure (`T=40` ID -> `T=120` OOD)."""
    print("\n=== X04: Delayed Multi-Key Overwrite & Selective Erasure ===", flush=True)
    arms = ("contraction_delta", "householder_delta", "gdn2", "complex_roto_ssm", "geo_erase_write")
    out: dict[str, Any] = {}
    for arm in arms:
        id_accs, ood_accs = [], []
        param_count = 0
        for seed in SEEDS:
            _set_seed(seed)
            model = ToySequenceModel(
                vocab_size=49,
                d_model=16,
                n_heads=2,
                layer_names=(arm, arm),
                num_classes=7,
                use_ffn=True,
            )
            param_count = model.total_parameters()
            opt = torch.optim.AdamW(model.parameters(), lr=6e-3, weight_decay=0.01)
            gen = torch.Generator().manual_seed(seed + 2000)
            model.train()
            for _ in range(220):
                xb, yb = _make_edit_batch(48, 40, gen)
                mask = yb != -100
                loss = F.cross_entropy(model(xb)[mask], yb[mask])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            model.eval()
            eval_gen = torch.Generator().manual_seed(seed + 8000)
            with torch.no_grad():
                for length, bucket in ((40, id_accs), (120, ood_accs)):
                    xe, ye = _make_edit_batch(48, length, eval_gen)
                    mask = ye != -100
                    pred = model(xe).argmax(dim=-1)
                    bucket.append(float((pred[mask] == ye[mask]).float().mean().item()))

        out[arm] = {
            "params": param_count,
            "id_T40_acc": _mean_std(id_accs),
            "ood_3x_T120_acc": _mean_std(ood_accs),
        }
        print(
            f"  {arm:18s} | params={param_count:5d} | "
            f"ID(T=40)={out[arm]['id_T40_acc']['mean']:.4f}±{out[arm]['id_T40_acc']['std']:.4f} | "
            f"OOD(T=120)={out[arm]['ood_3x_T120_acc']['mean']:.4f}±{out[arm]['ood_3x_T120_acc']['std']:.4f}",
            flush=True,
        )
    return out


def _make_needle_batch(
    bsz: int, seq_len: int, num_needles: int, gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Place `num_needles` unique (key, val) pairs amidst distractors, then query one (`8` values)."""
    # Vocab: 0..5 distractors, 8..23 keys (16 keys), 24..31 values (8 classes -> 0..7), 39 QUERY
    xb = torch.randint(0, 6, (bsz, seq_len), generator=gen)
    targets = torch.zeros(bsz, dtype=torch.long)
    for b in range(bsz):
        keys = torch.randperm(16, generator=gen)[:num_needles] + 8
        vals = torch.randint(24, 32, (num_needles,), generator=gen)
        slots = torch.randperm((seq_len - 4) // 2, generator=gen)[:num_needles] * 2
        for pos, k, v in zip(slots.tolist(), keys.tolist(), vals.tolist()):
            xb[b, pos] = k
            xb[b, pos + 1] = v
        q_idx = int(torch.randint(0, num_needles, (1,), generator=gen).item())
        xb[b, -2] = 39
        xb[b, -1] = keys[q_idx]
        targets[b] = vals[q_idx] - 24
    return xb, targets


def run_x05_x07_retrieval_and_dual_memory() -> dict[str, Any]:
    """X05 & X07: Multi-needle distractor scaling vs DualMemoryController with total byte accounting."""
    print("\n=== X05 & X07: Multi-Needle Scaling & Dual-Memory Baseline ===", flush=True)
    configs = {
        "dense_gqa": ("dense_gqa", "dense_gqa", "dense_gqa", "dense_gqa"),
        "diff_gqa": ("diff_gqa", "diff_gqa", "diff_gqa", "diff_gqa"),
        "pure_geo_delta": ("geo_erase_write", "geo_erase_write", "geo_erase_write", "geo_erase_write"),
        "hybrid_3to1_geo": ("geo_erase_write", "geo_erase_write", "geo_erase_write", "dense_gqa"),
    }
    eval_grid = ((48, 1), (48, 4), (48, 8), (192, 8), (192, 16))
    neural_res: dict[str, Any] = {}

    for arm, layout in configs.items():
        by_regime: dict[str, list[float]] = {f"T{t}_N{n}": [] for (t, n) in eval_grid}
        param_count = 0
        mem_prof: dict[str, Any] = {}
        for seed in SEEDS:
            _set_seed(seed)
            model = ToySequenceModel(
                vocab_size=40,
                d_model=24,
                n_heads=2,
                layer_names=layout,
                num_classes=8,
                use_ffn=True,
            )
            param_count = model.total_parameters()
            mem_prof = {
                "T48": model.memory_profile_bytes(48),
                "T192": model.memory_profile_bytes(192),
                "T384": model.memory_profile_bytes(384),
            }
            opt = torch.optim.AdamW(model.parameters(), lr=6e-3, weight_decay=0.01)
            gen = torch.Generator().manual_seed(seed + 3000)
            model.train()
            for step in range(240):
                n_train = (1, 4, 8)[step % 3]
                xb, yb = _make_needle_batch(48, 48, n_train, gen)
                loss = F.cross_entropy(model(xb)[:, -1, :], yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            model.eval()
            eval_gen = torch.Generator().manual_seed(seed + 7000)
            with torch.no_grad():
                for seq_len, n_needles in eval_grid:
                    xe, ye = _make_needle_batch(64, seq_len, n_needles, eval_gen)
                    pred = model(xe)[:, -1, :].argmax(dim=-1)
                    by_regime[f"T{seq_len}_N{n_needles}"].append(float((pred == ye).float().mean().item()))

        neural_res[arm] = {
            "params": param_count,
            "memory_profile_bytes": mem_prof,
            "regimes": {k: _mean_std(v) for k, v in by_regime.items()},
        }
        summary = " | ".join(f"{k}={neural_res[arm]['regimes'][k]['mean']:.3f}" for k in by_regime)
        print(f"  {arm:16s} | params={param_count:5d} | {summary}", flush=True)

    # X07: Direct comparison of fixed-byte fast sketch vs DualMemoryController on 16-needle & contradiction suites
    dual_trials, sketch_trials, dual_bytes_samples = [], [], []
    for trial in range(12):
        ctrl = DualMemoryController(fast_slots=8, slot_bytes=16)
        keys_vals = [(f"k{i:02d}", f"val-rx-{trial}-{i:02d}-exact") for i in range(16)]
        for i, (k, v) in enumerate(keys_vals):
            ctrl.commit(
                subject_id="s0", key=k, verbatim_value=v, source_id="src",
                start=i * 32, end=i * 32 + len(v), valid_time=i, observed_time=i,
            )
        dual_hits = sum(ctrl.verify_no_corruption(subject_id="s0", key=k, expected_verbatim=v) for k, v in keys_vals)
        sketch_hits = sum(ctrl.query_fast_only(subject_id="s0", key=k).values == (v,) for k, v in keys_vals)
        dual_trials.append(dual_hits / 16.0)
        sketch_trials.append(sketch_hits / 16.0)
        dual_bytes_samples.append(ctrl.resource_accounting()["total_dual_bytes"])

    dual_summary = {
        "fast_sketch_only_16needle_exact": _mean_std(sketch_trials),
        "dual_memory_16needle_exact": _mean_std(dual_trials),
        "fast_sketch_bytes": 8 * 16,
        "dual_total_bytes_mean": round(statistics.mean(dual_bytes_samples), 1),
    }
    print(
        f"  X07 DualMemory  | FastSketch(128B)={dual_summary['fast_sketch_only_16needle_exact']['mean']:.3f} | "
        f"DualStore({dual_summary['dual_total_bytes_mean']}B)={dual_summary['dual_memory_16needle_exact']['mean']:.3f}",
        flush=True,
    )
    return {"neural": neural_res, "dual_memory_x07": dual_summary}


def _write_report(results: dict[str, Any]) -> None:
    x03 = results["x03"]
    x04 = results["x04"]
    x05 = results["x05_x07"]["neural"]
    x07 = results["x05_x07"]["dual_memory_x07"]

    lines = [
        "# Mechanism Lab Report: X03, X04, X05 & X07 (3-Seed Local CPU Suite)",
        "",
        f"- **Device / dtype:** `cpu` / `float32` · **Seeds:** `{list(SEEDS)}` · **Paid compute:** `$0.00` · **Wall time:** `{results['wall_seconds']:.2f}s`",
        "",
        "## 1. X03 — Group Tracking (`Z_2`, `Z_4`, `S_3`) & Jacobian Diagnostics",
        "",
        "| Task | Arm | Params | ID (`T=32`) | 4x OOD (`T=128`) | 16x OOD (`T=512`) | `||A_t||_2` | Full `||dS_t/dS_{t-1}||_2` |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task_name, task_res in x03.items():
        if task_name == "z4_complex_zero_s0_control":
            continue
        for arm, row in task_res.items():
            a_str = f"{row['homogeneous_A_sigma_max']['mean']:.3f}" if row["homogeneous_A_sigma_max"] else "N/A"
            lines.append(
                f"| `{task_name}` | `{arm}` | {row['params']} | "
                f"{row['id_T32_acc']['mean']:.4f}±{row['id_T32_acc']['std']:.4f} | "
                f"{row['ood_4x_T128_acc']['mean']:.4f}±{row['ood_4x_T128_acc']['std']:.4f} | "
                f"{row['ood_16x_T512_acc']['mean']:.4f}±{row['ood_16x_T512_acc']['std']:.4f} | "
                f"{a_str} | {row['full_jacobian_sigma_max']['mean']:.3f}±{row['full_jacobian_sigma_max']['std']:.3f} |"
            )

    z0 = x03["z4_complex_zero_s0_control"]["ood_4x_T128_acc"]
    lines.append("")
    lines.append(
        f"- **Zero-init control (`z4_rotation`, `complex_roto_ssm`, `S_0=0`):** `4x OOD = {z0['mean']:.4f}±{z0['std']:.4f}` (vs learnable $S_0$ above)."
    )

    lines.extend([
        "",
        "## 2. X04 — Delayed Multi-Key Overwrite & Selective Erasure",
        "",
        "| Arm | Params | ID (`T=40`) | 3x OOD (`T=120`) |",
        "|---|---:|---:|---:|",
    ])
    for arm, row in x04.items():
        lines.append(
            f"| `{arm}` | {row['params']} | "
            f"{row['id_T40_acc']['mean']:.4f}±{row['id_T40_acc']['std']:.4f} | "
            f"{row['ood_3x_T120_acc']['mean']:.4f}±{row['ood_3x_T120_acc']['std']:.4f} |"
        )

    lines.extend([
        "",
        "## 3. X05 & X07 — Multi-Needle Retrieval & Dual-Memory Accounting",
        "",
        "| Arm | Params | `T48_N1` | `T48_N4` | `T48_N8` | `T192_N8` | `T192_N16` | Seq Mem @ `T=384` (B) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for arm, row in x05.items():
        r = row["regimes"]
        mem384 = row["memory_profile_bytes"]["T384"]["total_sequence_memory_bytes"]
        lines.append(
            f"| `{arm}` | {row['params']} | "
            f"{r['T48_N1']['mean']:.3f}±{r['T48_N1']['std']:.3f} | "
            f"{r['T48_N4']['mean']:.3f}±{r['T48_N4']['std']:.3f} | "
            f"{r['T48_N8']['mean']:.3f}±{r['T48_N8']['std']:.3f} | "
            f"{r['T192_N8']['mean']:.3f}±{r['T192_N8']['std']:.3f} | "
            f"{r['T192_N16']['mean']:.3f}±{r['T192_N16']['std']:.3f} | {mem384:,} |"
        )

    lines.extend([
        "",
        f"- **X07 Dual Memory (`16` exact needles):** `CompressedFastSketch(128B)` = `{x07['fast_sketch_only_16needle_exact']['mean']:.3f}` vs "
        f"`DualMemoryController({x07['dual_total_bytes_mean']}B total)` = `{x07['dual_memory_16needle_exact']['mean']:.3f}` (zero silent corruption; explicit byte & index overhead accounted).",
    ])
    (OUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    results = {
        "schema": "nano.mechanisms_x03_x07.v1",
        "device": "cpu",
        "dtype": "float32",
        "seeds": list(SEEDS),
        "paid_compute_usd": 0.0,
        "x03": run_x03_group_and_jacobians(),
        "x04": run_x04_erase_write_benchmark(),
        "x05_x07": run_x05_x07_retrieval_and_dual_memory(),
    }
    results["wall_seconds"] = round(time.perf_counter() - t0, 2)
    results_path = OUT_DIR / "results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    _write_report(results)

    sums = []
    for fname in ("results.json", "REPORT.md"):
        digest = hashlib.sha256((OUT_DIR / fname).read_bytes()).hexdigest()
        sums.append(f"{digest}  {fname}")
    (OUT_DIR / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(f"\nSaved immutable X03-X07 artifacts to {OUT_DIR} in {results['wall_seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
