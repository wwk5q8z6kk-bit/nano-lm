"""Plot the fixed V4 margin summaries; error bars are seed ranges, not CIs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    targets = [args.output_dir / ("margin_tradeoffs." + ext) for ext in ("svg", "png")]
    for target in targets:
        if target.exists():
            raise FileExistsError(f"Existing figure is preserved: {target}")
    data = json.loads((ROOT / "results/margin_reanalysis_v4.json").read_text())
    arms = ["none_m1", "half_m1", "half_m3", "half_m6", "keep_m3"]
    labels = ["Minimum L1\nmargin 1", "Half kept\nmargin 1", "Half kept\nmargin 3",
              "Half kept\nmargin 6", "90% kept\nmargin 3"]
    colors = ["#7e899b", "#3993bd", "#2378a6", "#15587f", "#cc8550"]
    metrics = [
        ("edit_l1", "Address movement", "Mean edit L1", 1),
        ("collateral_now", "Neighbor facts at phase end", "Accuracy (%)", 100),
        ("rule_old_corrected", "Old regular facts on corrected entities", "Accuracy (%)", 100),
        ("corrected_old", "Final eligible old corrections", "Accuracy (%)", 100),
    ]
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), layout="constrained")
    for ax, (key, title, ylabel, factor) in zip(axes.flat, metrics):
        records = [data["arms"][a]["metrics"][key] for a in arms]
        means = np.array([r["mean"] * factor for r in records])
        lows = np.array([r["min"] * factor for r in records])
        highs = np.array([r["max"] * factor for r in records])
        ax.bar(range(5), means, color=colors, width=.67, zorder=3)
        ax.errorbar(range(5), means, yerr=[means - lows, highs - means],
                    fmt="none", ecolor="#253248", capsize=4, linewidth=1.4, zorder=4)
        for i, value in enumerate(means):
            ax.text(i, highs[i] + (2 if factor == 100 else 3), f"{value:.2f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(5), labels)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=10)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 111 if factor == 100 else max(highs) * 1.22)
        ax.grid(axis="y", alpha=.2, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Correction margins trade off locality and later endpoints",
                 fontsize=17, fontweight="bold")
    fig.supxlabel("Means of three rounded seed summaries; bars show min–max, not confidence intervals.\n"
                  "Different endpoint cohorts. Bilinear pilot at commit 9d88a2c; no native LM result.",
                  fontsize=10)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for target in targets:
        with target.open("xb") as stream:
            fig.savefig(stream, format=target.suffix[1:], dpi=180,
                        metadata={"Creator": "Nano LM V4 evidence analysis"})
    plt.close(fig)
    print(json.dumps({"figures": [str(p) for p in targets],
                      "source": "results/margin_reanalysis_v4.json"}))


if __name__ == "__main__":
    main()
