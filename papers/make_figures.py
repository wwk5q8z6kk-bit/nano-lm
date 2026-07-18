# Paper 1 figures — generated from the committed result JSONs (no re-measurement).
#   fig1_gap_vs_scale.{png,pdf}      — the collapse, one consistent instrument, stack change marked
#   fig2_instance_difficulty.{png,pdf} — inst0 (public instance) is a hard draw at every rung
# Usage: python papers/make_figures.py
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = os.path.join(REPO, "trajectory")
OUT = os.path.join(REPO, "papers", "figures"); os.makedirs(OUT, exist_ok=True)

def load(fn): return json.load(open(os.path.join(T, fn)))

# ---- assemble the ladder from committed JSONs (params in millions) ----
RUNGS = [
    dict(name="nano",        params=3.15, stack="own",    file="results_anchors_v2_nano.json"),
    dict(name="scale",       params=10.0, stack="own",    file="results_anchors_v2_scale.json"),
    dict(name="pythia-160m", params=160., stack="Pythia", file="results_arm1_v2_pythia-160m.json"),
    dict(name="pythia-410m", params=410., stack="Pythia", file="results_arm1_v2_pythia-410m.json"),
]
for r in RUNGS:
    d = load(r["file"])
    r["gap_mean"] = d["gap_mean"]; r["gap_sd"] = d["gap_sd"]
    r["fresh_gaps"] = d["fresh_gaps"]; r["inst0"] = d["inst0"]["gap_pts"]

# 1B: reported as an interval [0,5] (training-run bounded), not a powered point
ONE_B = dict(name="pythia-1b", params=1000., stack="Pythia", interval=(0.0, 5.0),
             runs=(5.0, 0.0))  # the two training runs behind the interval

OWN = "#1b3a6b"; PY = "#c0392b"; GREY = "#7f8c8d"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#333",
                     "savefig.bbox": "tight", "figure.dpi": 150})

# =====================================================================
# Figure 1 — held-out copying gap vs scale (the collapse)
# =====================================================================
fig, ax = plt.subplots(figsize=(7.2, 4.6))
# stack-change band between 10M and 160M (visual: this axis is NOT pure scale)
ax.axvspan(10, 160, color=GREY, alpha=0.07, zorder=0)
ax.text(40, 20.6, "stack change\n(own → Pythia)", ha="center", va="top",
        fontsize=8.5, color=GREY, style="italic")

for stack, col, mk in [("own", OWN, "o"), ("Pythia", PY, "s")]:
    xs = [r["params"] for r in RUNGS if r["stack"] == stack]
    ys = [r["gap_mean"] for r in RUNGS if r["stack"] == stack]
    es = [r["gap_sd"] for r in RUNGS if r["stack"] == stack]
    lab = "own stack (3.15M, 10M)" if stack == "own" else "Pythia open-weight (160M–1B)"
    ax.errorbar(xs, ys, yerr=es, fmt=mk, color=col, ms=8, capsize=4, lw=1.6,
                elinewidth=1.6, label=lab, zorder=3)

# 1B interval
lo, hi = ONE_B["interval"]
ax.plot([ONE_B["params"], ONE_B["params"]], [lo, hi], color=PY, lw=6, alpha=0.35, solid_capstyle="butt", zorder=2)
ax.plot([ONE_B["params"]], [(lo + hi) / 2], "s", color=PY, ms=8, mfc="white", mew=1.6, zorder=3)
ax.annotate("1B: [0, 5]\ntraining-run bounded", (ONE_B["params"], (lo + hi) / 2), xytext=(560, 7.4),
            ha="right", fontsize=8.5, color=PY,
            arrowprops=dict(arrowstyle="-", color=PY, lw=0.8, alpha=0.6))

# annotate the own-stack near-identity
ax.annotate("3M ≈ 10M\n(Δ0.4, inside 1 SD)", (6.0, 18.5), xytext=(4.0, 13.0),
            ha="center", fontsize=8.5, color=OWN,
            arrowprops=dict(arrowstyle="-", color=OWN, lw=0.8, alpha=0.6))

ax.set_xscale("log")
ax.set_xticks([3.15, 10, 160, 410, 1000])
ax.set_xticklabels(["3.15M", "10M", "160M", "410M", "1B"])
ax.set_xlabel("Parameters (log scale)")
ax.set_ylabel("Held-out copying gap  (seen − held recall, pts)")
ax.set_ylim(-1.5, 22)
ax.axhline(0, color="#bbb", lw=0.8, zorder=1)
ax.set_title("Held-out copying gap collapses by Pythia scale — one consistent instrument",
             fontsize=11.5)
ax.text(0.985, 0.52, "gap = mean ± across-instance SD\n(5 instances × 100 held)",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#555")
ax.legend(loc="upper right", frameon=False, fontsize=9)
ax.grid(axis="y", ls=":", alpha=0.4)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.savefig(os.path.join(OUT, "fig1_gap_vs_scale.png"))
fig.savefig(os.path.join(OUT, "fig1_gap_vs_scale.pdf"))
plt.close(fig)
print("wrote fig1_gap_vs_scale.{png,pdf}")

# =====================================================================
# Figure 2 — the public instance (inst0) is a hard draw at every rung
# =====================================================================
fig, ax = plt.subplots(figsize=(7.2, 4.6))
labels = [r["name"] for r in RUNGS] + ["pythia-1b"]
xpos = np.arange(len(labels))
rng = np.random.default_rng(0)

for i, r in enumerate(RUNGS):
    fg = r["fresh_gaps"]
    # fresh instances: jittered dots + mean±SD bar
    jit = (rng.random(len(fg)) - 0.5) * 0.18
    col = OWN if r["stack"] == "own" else PY
    ax.scatter(np.full(len(fg), i) + jit, fg, s=22, color=col, alpha=0.55, zorder=2,
               label="fresh instances (m0–m4)" if i == 0 else None)
    ax.errorbar(i, r["gap_mean"], yerr=r["gap_sd"], fmt="_", color=col, ms=22, capsize=5,
                elinewidth=1.6, mew=2, zorder=3,
                label="multi-instance mean ± SD" if i == 0 else None)
    # inst0 marker
    ax.scatter(i, r["inst0"], marker="X", s=95, color="black", zorder=4,
               label="public instance (inst0)" if i == 0 else None)
    ax.annotate(f"{r['inst0']:.1f}", (i, r["inst0"]), xytext=(i + 0.16, r["inst0"]),
                fontsize=8, va="center", color="black")

# 1B: everything ~0, training-variance dominated — shown honestly, not as a hard draw
i = len(RUNGS)
ax.plot([i, i], [0, 5], color=PY, lw=6, alpha=0.3, solid_capstyle="butt", zorder=2)
ax.scatter([i], [ONE_B["runs"][0]], marker="X", s=95, color="black", zorder=4)
ax.scatter([i], [ONE_B["runs"][1]], marker="X", s=95, color="black", zorder=4)
ax.annotate("training-run\nvariance\n(0 and 5)", (i, 5), xytext=(i, 8.5), ha="center",
            fontsize=8, color=PY)

ax.set_xticks(xpos); ax.set_xticklabels(labels, rotation=15)
ax.set_ylabel("Held-out copying gap (pts)")
ax.set_title("The public instance is a systematically hard draw (inst0 ≥ mean at every rung)",
             fontsize=11.5)
ax.set_ylim(-1.5, 26)
ax.axhline(0, color="#bbb", lw=0.8)
ax.grid(axis="y", ls=":", alpha=0.4)
for s in ("top", "right"): ax.spines[s].set_visible(False)
handles = [Line2D([], [], marker="X", color="black", ls="", ms=9, label="public instance (inst0)"),
           Line2D([], [], marker="_", color=GREY, ls="", ms=16, mew=2, label="multi-instance mean ± SD"),
           Line2D([], [], marker="o", color=GREY, ls="", ms=6, alpha=0.6, label="fresh instances (m0–m4)")]
ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)
fig.savefig(os.path.join(OUT, "fig2_instance_difficulty.png"))
fig.savefig(os.path.join(OUT, "fig2_instance_difficulty.pdf"))
plt.close(fig)
print("wrote fig2_instance_difficulty.{png,pdf}")

# ---- also emit a LaTeX booktabs table for §6.1 ----
tex = r"""\begin{table}[t]\centering
\begin{tabular}{lrlrr}
\toprule
Model & Params & Stack & Gap (5$\times$100 held) & inst0 \\
\midrule
nano        & 3.15M & own    & $18.3 \pm 1.3$ & 22.4 \\
scale       & 10M   & own    & $18.7 \pm 1.5$ & 23.0 \\
pythia-160m & 160M  & Pythia & $3.5 \pm 0.7$  & 7.0 \\
pythia-410m & 410M  & Pythia & $4.2 \pm 0.9$  & 8.0 \\
pythia-1b   & 1B    & Pythia & $[0, 5]$\,$^\dagger$ & 5.0/0.0 \\
\bottomrule
\end{tabular}
\caption{Held-out copying gap across the ladder on one multi-instance instrument
(mean $\pm$ across-instance SD over five 100-held instances). inst0 is the single
public instance. $^\dagger$1B is bounded by training-run nondeterminism, not eval noise.}
\label{tab:gap-ladder}
\end{table}
"""
open(os.path.join(OUT, "table_gap_ladder.tex"), "w").write(tex)
print("wrote table_gap_ladder.tex")
