"""Render exact reported correction summaries; requires an explicit new output directory."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', required=True, type=Path)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = [args.out_dir / ('correction_locality_durability.' + ext) for ext in ('svg', 'png')]
    if any(path.exists() for path in paths):
        raise FileExistsError('Choose a new output path; existing figures are preserved')
    data = json.loads((ROOT / 'results/repository_reanalysis_v3.json').read_text())['correction_results']
    names = [('single', 'Single-fact write'), ('entity_replay', 'Same-entity replay'),
             ('random_replay', 'Other-entity replay*'), ('frozen_address', 'Frozen address'),
             ('constrained', 'Computed address edit'), ('store', 'Store arm: model alone')]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.4), sharey=True)
    fig.subplots_adjust(left=.20, right=.98, top=.80, bottom=.25, wspace=.15)
    for ax, metric, title, color in zip(axes,
        ['collateral_now', 'corrected_old'],
        ['Immediate accuracy on neighbouring facts', 'Final accuracy on old corrections'],
        ['#226B9B', '#BB5A30']):
        ys = np.arange(len(names))
        mean = np.array([data[k]['means'][metric] * 100 for k, _ in names])
        low = np.array([data[k]['ranges'][metric][0] * 100 for k, _ in names])
        high = np.array([data[k]['ranges'][metric][1] * 100 for k, _ in names])
        ax.barh(ys, mean, color=color, height=.55, alpha=.9)
        ax.errorbar(mean, ys, xerr=[np.maximum(0, mean-low), np.maximum(0, high-mean)],
                    fmt='none', color='#1D2939', capsize=3, lw=1.2)
        for y, m, h in zip(ys, mean, high):
            ax.text(min(103.0, h+1.4), y, f'{m:.1f}', va='center', fontsize=10)
        ax.set_xlim(0, 113)
        ax.set_xticks([0,25,50,75,100])
        ax.set_xlabel('Accuracy (%)')
        ax.set_title(title, fontsize=12, loc='left', pad=12)
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#E3E8EF', linewidth=.8)
    axes[0].set_yticks(np.arange(len(names)), [label for _, label in names])
    axes[0].invert_yaxis()
    fig.text(.02,.96,'Nano LM | Locality and durability are separate outcomes', fontsize=18,
             weight='bold', color='#16324F', va='top')
    fig.text(.02,.905,'Existing bilinear-model pilot • 3 seeds • bars: mean • whiskers: seed range, not confidence intervals',
             color='#475467', fontsize=11)
    fig.text(.02,.155,'The panels score different fact cohorts. Their difference is not a paired forgetting rate.',
             fontsize=11, weight='bold', color='#344054')
    fig.text(.02,.105,'* Other-entity replay advances the correction RNG; future event streams differ. Same-entity and single-write streams match.',
             fontsize=10, color='#475467')
    fig.text(.02,.055,'Source: pinned repository commit 2ef9ce5, results/correction_interference. Reanalysis only; no new training.',
             fontsize=10, color='#475467')
    for path in paths:
        fig.savefig(path, dpi=170, facecolor='white')
    plt.close(fig)
    print(json.dumps({'figures': [str(p) for p in paths], 'scope': 'Descriptive existing three-seed results'}))


if __name__ == '__main__':
    main()
