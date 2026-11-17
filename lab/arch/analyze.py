"""Aggregate Track A (LAB-ARCH-02, LAB-ARCH-03 & LAB-SCALE-01) results into multi-scale tables.

Usage:
  python -m lab.arch.analyze [/Volumes/Express1M2/nanolm-lab/runs] > lab/reports/LAB-ARCH-02_03_FINAL_REPORT_TABLES.md
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

DOMS = ["web", "textbook", "math", "code"]


def load(root: Path) -> list[dict]:
    rows = []
    for f in sorted(root.glob("*/result.json")):
        r = json.loads(f.read_text())
        if r.get("final") is None or "smoke" in r["run"]:
            continue
        rows.append(r)
    return rows


def _mean_spread(vals: list[float]) -> tuple[float, float]:
    return (st.mean(vals), (max(vals) - min(vals)) / 2 if len(vals) > 1 else float("nan"))


def _render_scale_table(
    title: str,
    rows: list[dict],
    ref_key: tuple[str, str, str],
) -> None:
    if not rows:
        return
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["arm"], r["optim"], r.get("match", "tokens"))].append(r)

    ref = groups.get(ref_key)
    ref_c = st.mean(r["final"]["bpb_composite"] for r in ref) if ref else None
    ref_d = {d: st.mean(r["final"]["bpb"][d] for r in ref) for d in DOMS} if ref else None
    ref_spread = _mean_spread([r["final"]["bpb_composite"] for r in ref])[1] if ref and len(ref) > 1 else None

    print(f"## {title}\n")
    print(
        f"Runs: {len(rows)}. Values = mean over seeds; ± = half-range across seeds. "
        f"Δ% is relative to `{ref_key[0]}/{ref_key[1]}/{ref_key[2]}` (negative = better).\n"
    )
    print(
        "| Arm | Optim | Match | n | Tokens | Train TFLOP·s | KV B/tok | Composite BPB | Δ% vs Ref | "
        "web | textbook | math | code | copy loss | ctx gain (nats) | tok/s | Decision / Notes |"
    )
    print("|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    order = sorted(groups, key=lambda k: st.mean(r["final"]["bpb_composite"] for r in groups[k]))
    for k in order:
        g = groups[k]
        c, cs = _mean_spread([r["final"]["bpb_composite"] for r in g])
        doms = {d: st.mean(r["final"]["bpb"][d] for r in g) for d in DOMS}
        copy = st.mean(r["final"]["copy_loss_second_half"] for r in g)
        ctx = st.mean(r["final"]["web_loss_pos_0_64"] - r["final"]["web_loss_pos_last_256"] for r in g)
        tps = st.mean(r["train_tok_s"] for r in g)
        flops = st.mean(r.get("train_flops_total", 0) for r in g) / 1e12
        toks = st.mean(r["tokens"] for r in g) / 1e6
        kv_b = int(st.mean(r.get("global_attn_kv_bytes_per_token", 0) for r in g))
        delta = 100 * (c / ref_c - 1) if ref_c else float("nan")
        dec = ""
        if ref_c and k != ref_key:
            worst = max(100 * (doms[d] / ref_d[d] - 1) for d in DOMS)
            consistent = (
                all(r["final"]["bpb_composite"] < ref_c for r in g)
                if delta < 0
                else all(r["final"]["bpb_composite"] > ref_c for r in g)
            )
            if len(g) < 2:
                dec = ("promising (< -1%, 1 seed)" if delta <= -1.0 else "1-seed check")
            elif abs(delta) < 1.0:
                dec = "no detectable difference (<1%)"
            elif delta <= -1.0 and consistent and worst <= 1.0:
                dec = "**WIN** (prereg rule met)"
            elif delta <= -1.0:
                dec = (
                    "better on mean; rule not met"
                    + ("" if consistent else " (seed-inconsistent)")
                    + ("" if worst <= 1.0 else f" (domain {worst:+.1f}%)")
                )
            else:
                dec = "worse"
        if "expected_loops" in g[0]["final"]:
            el = st.mean(r["final"]["expected_loops"] for r in g)
            hg = st.mean(r["final"]["halt_gamma"] for r in g)
            dec = f"{dec}; SGRD loops={el:.2f}, γ={hg:.3f}".lstrip("; ")
        spread_str = f" ± {cs:.4f}" if len(g) > 1 else ""
        print(
            f"| {k[0]} | {k[1]} | {k[2]} | {len(g)} | {toks:.0f}M | {flops:.3g} | {kv_b:,} | "
            f"{c:.4f}{spread_str} | {delta:+.2f}% | "
            + " | ".join(f"{doms[d]:.4f}" for d in DOMS)
            + f" | {copy:.3f} | {ctx:+.3f} | {tps:,.0f} | {dec} |"
        )
    if ref_spread is not None and ref_c:
        print(f"\nReference `{ref_key[0]}` seed half-range on composite BPB: {ref_spread:.4f} ({100 * ref_spread / ref_c:.2f}%).\n")


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/Volumes/Express1M2/nanolm-lab/runs")
    rows = load(root)
    rows_46m = [r for r in rows if r.get("params", {}).get("non_embedding", 0) < 60_000_000]
    rows_125m = [r for r in rows if r.get("params", {}).get("non_embedding", 0) >= 60_000_000]

    print("# Track A Empirical Architecture, Optimizer & Scaling Results (Auto-Generated)\n")
    _render_scale_table(
        "1. Iso-Parameter (~29.5M Non-Emb / ~46.2M Total) Architecture & Optimizer Grid",
        rows_46m,
        ("T0_dense", "adamw", "tokens"),
    )
    if rows_125m:
        ref_125 = ("S0_dense_125m", "muon", "tokens")
        if not any((r["arm"], r["optim"], r.get("match", "tokens")) == ref_125 for r in rows_125m):
            ref_125 = ("S0_dense_125m", "adamw", "tokens")
        _render_scale_table(
            "2. Multi-Scale Transfer (`LAB-SCALE-01`, ~96.3M Non-Emb / ~121M Total)",
            rows_125m,
            ref_125,
        )
    print("`ctx gain` = loss(positions 0–64) − loss(last 256 positions) on web val (larger = better long-context utilization).")


if __name__ == "__main__":
    main()
