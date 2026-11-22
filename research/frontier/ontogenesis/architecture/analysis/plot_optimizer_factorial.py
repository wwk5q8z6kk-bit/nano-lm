"""Plot frozen V5 factorial results without rerunning the model.

An explicit --out-dir is required. Both artifact paths are checked and opened
exclusively; existing outputs are never overwritten. The plotted observations
and source digest are embedded as metadata in the SVG and PNG.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

CELLS = [
    ("original/adam", "Original · Adam carried"),
    ("original/adam_reset", "Original · Adam reset"),
    ("hash/adam", "Hashed · Adam carried"),
    ("hash/adam_reset", "Hashed · Adam reset"),
    ("original/adam_b2_09", "Original · Adam β₂ = 0.9"),
    ("original/sgd", "Original · SGD, rate 1"),
]
SEEDS = [3101, 3102, 3103, 3104, 3105]
COMMIT = "767857e72bfa51e3fa6f32d047507aabef077fde"


def load_observations(path):
    raw = path.read_bytes()
    report = json.loads(raw)
    if report["provenance"]["cutoffs"]["factorial"] != COMMIT:
        raise ValueError("Expected the pinned V5 factorial result commit")
    result = {}
    for key, _ in CELLS:
        cell = report["factorial"]["cells"][key]
        rows = cell["per_seed"]
        if [row["seed"] for row in rows] != SEEDS:
            raise ValueError(f"Expected all five seeds in canonical order: {key}")
        result[key] = {}
        for metric in ["late_acquire", "late_retain"]:
            values = [row[metric] for row in rows]
            if not all(math.isfinite(x) and 0 <= x <= 1 for x in values):
                raise ValueError(f"Invalid plotted accuracy: {key}/{metric}")
            saved = cell["metrics"][metric]
            for name, computed in [("mean", mean(values)), ("min", min(values)), ("max", max(values))]:
                if not math.isclose(saved[name], computed, rel_tol=0, abs_tol=1e-12):
                    raise ValueError(f"Receipt statistic disagrees with its observations: {key}/{metric}/{name}")
            if saved["values"] != values:
                raise ValueError(f"Receipt observation order disagrees: {key}/{metric}")
            result[key][metric] = values
    return result, {
        "source_file": path.name,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_commit": COMMIT,
        "seeds": SEEDS,
        "unit_of_analysis": "seed",
        "endpoint_window": "phases 46 through 60; zero-based log phases 45 through 59",
        "aggregation": "equal-seed mean of per-seed phase means",
        "interval": "observed minimum and maximum of five seeds; descriptive, not a confidence interval",
        "data_in_accuracy_units": result,
    }


def render(observations, metadata):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.edgecolor": "#D9E1E5", "axes.labelcolor": "#334451",
        "xtick.color": "#52616B", "ytick.color": "#253746",
        "svg.fonttype": "none", "svg.hashsalt": "nanolm-v5-factorial",
    })
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(13.2, 7.2))
    fig.subplots_adjust(left=.232, right=.964, bottom=.226, top=.755, wspace=.16)
    fig.set_facecolor("white")
    fig.suptitle("Address and optimizer choices change acquisition and retention",
                 x=.055, y=.965, ha="left", fontsize=18, weight="bold", color="#182C3A")
    fig.text(.055, .906, "Five paired seeds  ·  Phases 46–60  ·  200 updates per phase",
             fontsize=11.5, color="#52616B")
    metric_specs = [
        ("late_acquire", "Late acquisition", "Primary endpoint"),
        ("late_retain", "Late retention", "Secondary endpoint"),
    ]
    offsets = [-.10, -.05, 0, .05, .10]
    line_color, mean_color, seed_color = "#4F8B95", "#123F58", "#879FA9"
    for axis, (metric, title, subtitle) in zip(axes, metric_specs):
        axis.set_title(title, loc="left", fontsize=14, weight="bold", pad=38, color="#123F58")
        axis.text(0, 1.055, subtitle, transform=axis.transAxes, fontsize=10.5, color="#52616B")
        axis.set_xlim(-3, 104)
        axis.set_xticks([0, 25, 50, 75, 100])
        axis.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        axis.set_ylim(5.6, -.58)
        axis.set_yticks(range(6), [label for _, label in CELLS])
        axis.tick_params(axis="y", length=0, pad=11, labelsize=10.5)
        axis.tick_params(axis="x", length=0, pad=10)
        axis.grid(axis="x", color="#E5EAED", linewidth=.8, zorder=0)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_visible(False)
        for index, (key, _) in enumerate(CELLS):
            values = [100 * x for x in observations[key][metric]]
            center, lo, hi = mean(values), min(values), max(values)
            axis.axhline(index, color="#F0F3F5", linewidth=1, zorder=0)
            axis.hlines(index, lo, hi, color=line_color, linewidth=3, zorder=2)
            axis.vlines([lo, hi], index - .07, index + .07, color=line_color, linewidth=1.3, zorder=2)
            axis.scatter(values, [index + d for d in offsets], s=28,
                         color=seed_color, alpha=.8, edgecolors="white", linewidth=.55, zorder=3)
            axis.scatter([center], [index], marker="D", s=65, color=mean_color,
                         edgecolors="white", linewidth=.8, zorder=4)
            align = "left" if center < 10 else "right" if center > 94 else "center"
            axis.text(center, index - .255, f"{center:.2f}%", ha=align, va="center",
                      fontsize=10.5, weight="bold", color=mean_color)
    legend_handles = [
        Line2D([], [], color=mean_color, marker="D", markersize=7, linestyle="None", label="Equal-seed mean"),
        Line2D([], [], color=seed_color, marker="o", markersize=5, linestyle="None", label="Individual seeds"),
        Line2D([], [], color=line_color, linewidth=3, label="Observed seed minimum–maximum"),
    ]
    fig.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(.23, .112),
               frameon=False, ncol=3, fontsize=10, handletextpad=.7, columnspacing=1.5)
    fig.text(.055, .078,
             "Ranges describe these five seeds; they are not confidence intervals. Labels show means rounded to two decimals.",
             fontsize=10, color="#425866")
    fig.text(.055, .043,
             "Synthetic fact-table task · Frozen result 767857e72bfa · Phase-log reanalysis only; no neural rerun.",
             fontsize=9.5, color="#647782")
    description = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
    output = {}
    for suffix in ["svg", "png"]:
        stream = BytesIO()
        common = {"Title": "Nano LM V5 address × optimizer factorial", "Description": description}
        meta = {**common, "Creator": "Nano LM V5 read-only reanalysis", "Date": None} if suffix == "svg" else {
            **common, "Software": "Nano LM V5 read-only reanalysis"}
        fig.savefig(stream, format=suffix, dpi=180, metadata=meta, facecolor="white")
        output[suffix] = stream.getvalue()
    plt.close(fig)
    return output


def save_new(outputs, out_dir):
    paths = {suffix: out_dir / f"optimizer_factorial.{suffix}" for suffix in outputs}
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    handles = {}
    try:
        for suffix, path in paths.items():
            handles[suffix] = path.open("xb")
        for suffix, stream in handles.items():
            stream.write(outputs[suffix])
    except BaseException:
        for suffix, stream in handles.items():
            stream.close()
            paths[suffix].unlink()
        raise
    finally:
        for stream in handles.values():
            stream.close()
    return {suffix: {"path": str(path), "bytes": len(outputs[suffix]),
                     "sha256": hashlib.sha256(outputs[suffix]).hexdigest()}
            for suffix, path in paths.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results/optimizer_factorial_reanalysis_v5.json")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Explicit destination; both output filenames must be absent")
    args = parser.parse_args()
    # Reject existing destination files before the rendering work as well.
    for suffix in ["svg", "png"]:
        if (args.out_dir / f"optimizer_factorial.{suffix}").exists():
            parser.error(f"Output already exists: optimizer_factorial.{suffix}")
    observations, metadata = load_observations(args.source)
    artifacts = save_new(render(observations, metadata), args.out_dir)
    print(json.dumps({"source_sha256": metadata["source_sha256"],
                      "seed_count": len(SEEDS), "interval": metadata["interval"],
                      "artifacts": artifacts}, indent=2))


if __name__ == "__main__":
    main()
