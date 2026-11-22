"""Plot the saved native comparison summary without training or model imports.

Seed ranges are descriptive minima/maxima, never confidence intervals.
An explicit --out-dir is required; both output files are created exclusively.
The summary digest, raw receipt identity and observations are embedded in each
artifact. Full receipts are gzip JSON; this renderer reads the summary only.
"""
from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
from statistics import mean

SEEDS = [9201, 9202, 9203]
PROTOCOL_SHA256 = "4cd9bdabedd2b507663f5d5d368dab34c037c2b078b386d872d9238b23f18425"
ARMS = [
    ("current_only", "Current only", "#2E5C88"),
    ("replay_matched_total", "Replay, fixed total", "#447C60"),
    ("replay_additive", "Replay, additive", "#B9692C"),
    ("frozen", "Frozen reference", "#6C7582"),
]
METRICS = [
    ("assignment_horizon1_accuracy", "A  Correction accuracy at horizon 1",
     "Primary endpoint; eligible assignments per seed: 6, 7, 9"),
    ("new_fact_immediate_accuracy", "B  Immediate new-fact acquisition",
     "Secondary endpoint; 24 first observations per seed"),
    ("final_current_observed_accuracy", "C  Final current-fact accuracy",
     "Secondary endpoint; 28 current facts per seed"),
]


def load_summary(path):
    raw = path.read_bytes()
    report = json.loads(raw)
    if (report["experiment_id"] != "native-update-allocation-v1"
            or report["protocol_sha256"] != PROTOCOL_SHA256
            or report["protocol"]["seeds"] != SEEDS
            or [run["seed"] for run in report["per_seed"]] != SEEDS):
        raise ValueError("Expected the frozen native comparison and all three seeds")
    observations, exposures = {}, {}
    for arm, _, _ in ARMS:
        observations[arm] = {}
        for metric, _, _ in METRICS:
            row = report["descriptive_seed_summary"][arm][metric]
            values = row["values"]
            if (len(values) != 3 or not row["complete"] or row["available_seed_count"] != 3
                    or not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                               and math.isfinite(x) and 0 <= x <= 1 for x in values)):
                raise ValueError("A plotted metric must retain all three finite seed values")
            for name, actual in (("mean", mean(values)), ("min", min(values)), ("max", max(values))):
                if not math.isclose(row[name], actual, rel_tol=0, abs_tol=1e-12):
                    raise ValueError(f"Summary statistic differs from observations: {arm}/{metric}/{name}")
            observations[arm][metric] = values
        counts = [run["arms"][arm]["counts"] for run in report["per_seed"]]
        exposures[arm] = {key: counts[0][key] for key in (
            "current_target_presentations", "replay_target_presentations", "optimizer_steps")}
        if any(any(counts[i][key] != value for key, value in exposures[arm].items())
               for i in range(3)):
            raise ValueError("The fixed update recipe must match across seeds")
        if exposures[arm]["optimizer_steps"] != (
                exposures[arm]["current_target_presentations"] + exposures[arm]["replay_target_presentations"]):
            raise ValueError("Post-prefix targets and optimizer calls disagree")
        for i, run in enumerate(report["per_seed"]):
            data = run["arms"][arm]
            h1 = data["correction_metrics"]["cohorts"]["all_assignments"]["phase_end_horizons"]["1"]
            if (h1["missing_measurement"] or h1["measured"] != h1["eligible"]
                    or h1["correct"] / h1["eligible"] != observations[arm]["assignment_horizon1_accuracy"][i]
                    or data["post_prefix_new_fact_acquisition"]["accuracy"]
                    != observations[arm]["new_fact_immediate_accuracy"][i]):
                raise ValueError("Plotted values differ from retained endpoint denominators")
    metadata = {
        "source_file": path.name, "source_sha256": hashlib.sha256(raw).hexdigest(),
        "protocol_sha256": PROTOCOL_SHA256, "full_receipt": report["full_receipt"],
        "seeds": SEEDS, "aggregation": "equal-seed mean",
        "interval": "observed three-seed minimum and maximum; not a confidence interval",
        "scope": report["scope"], "observations": observations, "exposures_per_seed": exposures,
    }
    return observations, exposures, metadata


def render(observations, exposures, metadata):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.labelcolor": "#344657", "xtick.color": "#52606F", "ytick.color": "#26384A",
        "svg.fonttype": "none", "svg.hashsalt": "nanolm-v5-native-update-comparison",
    })
    fig, axes = plt.subplots(2, 2, figsize=(14.4, 10.8))
    fig.subplots_adjust(left=.172, right=.975, bottom=.20, top=.825, wspace=.47, hspace=.55)
    fig.patch.set_facecolor("white")
    fig.suptitle("Native update allocation produced mixed descriptive outcomes",
                 x=.04, y=.97, ha="left", fontsize=18, weight="bold", color="#172C40")
    fig.text(.04, .925, "Three fixed seeds  ·  Identical model and Adam branch state  ·  Synthetic facts only",
             fontsize=11.5, color="#465B6D")
    fig.text(.04, .887, "Both replay arms lowered mean correction H1; secondary new-learning measures increased.",
             fontsize=11, color="#344657")
    offsets, markers = [-.11, 0, .11], ["o", "s", "^"]
    for axis, (metric, title, subtitle) in zip(axes.flat, METRICS):
        axis.set_title(title, loc="left", fontsize=12.5, weight="bold", pad=29, color="#172C40")
        axis.text(0, 1.05, subtitle, transform=axis.transAxes, fontsize=9.4, color="#52606F")
        axis.set_xlim(-3, 115)
        axis.set_xticks([0, 25, 50, 75, 100])
        axis.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        axis.set_ylim(3.65, -.5)
        axis.set_yticks(range(4), [label for _, label, _ in ARMS])
        axis.tick_params(axis="y", length=0, pad=11, labelsize=10)
        axis.tick_params(axis="x", length=0, pad=8)
        axis.grid(axis="x", color="#E3E9ED", linewidth=.8)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_visible(False)
        for i, (arm, _, color) in enumerate(ARMS):
            values = [100 * value for value in observations[arm][metric]]
            axis.axhline(i, color="#F2F4F6", linewidth=1, zorder=0)
            axis.hlines(i, min(values), max(values), color=color, alpha=.65, linewidth=3)
            axis.vlines([min(values), max(values)], i-.07, i+.07, color=color, alpha=.65)
            for value, offset, marker in zip(values, offsets, markers):
                axis.scatter([value], [i+offset], marker=marker, s=34, facecolor="white",
                             edgecolor=color, linewidth=1.2, zorder=3)
            axis.scatter([mean(values)], [i], marker="D", s=67, color=color,
                         edgecolor="white", linewidth=.8, zorder=4)
            axis.text(113, i, f"{mean(values):.2f}%", ha="right", va="center",
                      color=color, weight="bold", fontsize=10.5)
        axis.text(1, 1.0, "Mean", transform=axis.transAxes, ha="right", va="bottom",
                  color="#52606F", fontsize=9)

    axis = axes[1, 1]
    axis.set_title("D  Post-prefix update allocation per seed", loc="left",
                   fontsize=12.5, weight="bold", pad=29, color="#172C40")
    axis.text(0, 1.05, "One scored target per update; these are counts, not FLOPs or time",
              transform=axis.transAxes, fontsize=9.4, color="#52606F")
    axis.set_xlim(0, 132)
    axis.set_ylim(3.65, -.5)
    axis.set_yticks(range(4), [label for _, label, _ in ARMS])
    axis.set_xticks([0, 36, 72, 108])
    axis.tick_params(axis="both", length=0, pad=9)
    axis.grid(axis="x", color="#E3E9ED", linewidth=.8)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(False)
    for i, (arm, _, _) in enumerate(ARMS):
        current, replay = (exposures[arm][key] for key in
                           ("current_target_presentations", "replay_target_presentations"))
        axis.barh(i, current, height=.45, color="#2E5C88")
        axis.barh(i, replay, left=current, height=.45, color="#B9692C")
        for start, count in ((0, current), (current, replay)):
            if count:
                axis.text(start+count/2, i, str(count), ha="center", va="center",
                          color="white", weight="bold", fontsize=10)
        label = f"{current+replay}" + ("  (+50%)" if arm == "replay_additive" else "")
        axis.text(current+replay+3, i, label, va="center", color="#26384A", fontsize=10)
    axis.set_xlabel("Optimizer calls / scored target presentations", fontsize=10, labelpad=7)
    axis.legend(handles=[Patch(facecolor="#2E5C88", label="Current event"),
                         Patch(facecolor="#B9692C", label="Replay")],
                loc="upper left", bbox_to_anchor=(0, -.21), ncol=2, frameon=False, fontsize=9.5)

    legend = [
        Line2D([], [], marker="D", color="#344657", linestyle="None", markersize=7, label="Equal-seed mean"),
        Line2D([], [], color="#798D9C", linewidth=3, label="Observed seed minimum–maximum"),
    ] + [Line2D([], [], marker=m, markerfacecolor="white", color="#52606F", linestyle="None",
                markersize=5, label=str(s)) for m, s in zip(markers, SEEDS)]
    fig.legend(handles=legend, loc="lower left", bbox_to_anchor=(.16, .105),
               ncol=5, frameon=False, fontsize=10, columnspacing=1.4)
    fig.text(.04, .075, "Ranges describe these three seeds; they are not confidence intervals. No significance or winner claim.",
             fontsize=10, color="#344657")
    fig.text(.04, .046, "Common prefix: 40 batch updates and 160 scored targets per seed, executed once then cloned into all arms.",
             fontsize=10, color="#52606F")
    fig.text(.04, .019, "Receipt view only; no training rerun. Fixed-total replay replaces one current exposure; additive replay spends 50% more post-prefix updates.",
             fontsize=9.5, color="#52606F")
    description = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
    outputs = {}
    for suffix in ("svg", "png"):
        stream = BytesIO()
        common = {"Title": "Nano LM V5 native update comparison", "Description": description}
        meta = {**common, "Creator": "Nano LM V5 receipt renderer", "Date": None} if suffix == "svg" else {
            **common, "Software": "Nano LM V5 receipt renderer"}
        fig.savefig(stream, format=suffix, dpi=180, metadata=meta, facecolor="white")
        outputs[suffix] = stream.getvalue()
    plt.close(fig)
    return outputs


def save_new(outputs, out_dir):
    paths = {suffix: out_dir / f"native_update_comparison.{suffix}" for suffix in outputs}
    if any(path.exists() or path.is_symlink() for path in paths.values()):
        raise FileExistsError("Existing native comparison figures are preserved")
    out_dir.mkdir(parents=True, exist_ok=True)
    handles = {}
    try:
        for suffix, path in paths.items():
            handles[suffix] = path.open("xb")
        for suffix, handle in handles.items():
            handle.write(outputs[suffix])
    except BaseException:
        for suffix, handle in handles.items():
            handle.close()
            paths[suffix].unlink()
        raise
    finally:
        for handle in handles.values():
            handle.close()
    return {suffix: {"path": str(paths[suffix]), "bytes": len(content),
                     "sha256": hashlib.sha256(content).hexdigest()}
            for suffix, content in outputs.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1]
                        / "results/native_update_comparison_summary_v5.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    for suffix in ("svg", "png"):
        path = args.out_dir / f"native_update_comparison.{suffix}"
        if path.exists() or path.is_symlink():
            parser.error(f"Existing figure is preserved: {path}")
    observations, exposures, metadata = load_summary(args.source)
    artifacts = save_new(render(observations, exposures, metadata), args.out_dir)
    print(json.dumps({"source_sha256": metadata["source_sha256"],
                      "interval": metadata["interval"], "artifacts": artifacts}, indent=2))


if __name__ == "__main__":
    main()
