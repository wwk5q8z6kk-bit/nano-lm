"""Score results/memory_saturation_factorial/ against F1-F6 of PROTOCOL-MEMSAT-FACTORIAL.md.

Read-only. Prints per-seed values for every cell and the verdict of each prediction by the
registered rule ("holds" = at least four of the five seeds). Missing runs are listed, not skipped.
"""
import json, pathlib, sys

ROOT = pathlib.Path(__file__).parent / "results" / "memory_saturation_factorial"
SEEDS = [3101, 3102, 3103, 3104, 3105]
CELLS = [("original", "adam"), ("original", "adam_reset"), ("hash", "adam"), ("hash", "adam_reset"),
         ("original", "adam_b2_09"), ("original", "sgd")]


def load(address, opt):
    out = {}
    for s in SEEDS:
        f = ROOT / f"{address}-{opt}-s{s}.json"
        if f.exists() and f.stat().st_size:
            d = json.load(open(f)); out[s] = (d["late_acquire"], d["late_retain"], d["early_acquire"])
    return out


def count(cell, test):
    return sum(1 for s in SEEDS if s in cell and test(cell[s][0]))


def main():
    data = {c: load(*c) for c in CELLS}
    print("| Cell | " + " | ".join(f"s{s}" for s in SEEDS) + " | mean late acquire | mean late retain |")
    print("|---|" + "---|" * (len(SEEDS) + 2))
    for c in CELLS:
        d = data[c]
        row = [f"{d[s][0]:.3f}" if s in d else "missing" for s in SEEDS]
        ma = sum(v[0] for v in d.values()) / len(d) if d else float("nan")
        mr = sum(v[1] for v in d.values()) / len(d) if d else float("nan")
        print(f"| {c[0]}/{c[1]} | " + " | ".join(row) + f" | {ma:.3f} | {mr:.3f} |")
    sel = ROOT / "dev-sgd-selection.json"
    selection = json.load(open(sel)) if sel.exists() else None
    print("\nSGD selection:", selection)

    oa, orr, ha, hr = (data[c] for c in CELLS[:4]); b2, sgd = data[CELLS[4]], data[CELLS[5]]
    verdicts = {}
    verdicts["F1 original/adam < 0.20"] = (count(oa, lambda v: v < 0.20), len(oa))
    verdicts["F2 hash/adam > 0.90"] = (count(ha, lambda v: v > 0.90), len(ha))
    verdicts["F3 original/adam_reset >= 0.70"] = (count(orr, lambda v: v >= 0.70), len(orr))
    both = [s for s in SEEDS if s in ha and s in hr]
    verdicts["F4 |hash/reset - hash/adam| <= 0.05 (every seed)"] = (sum(abs(hr[s][0] - ha[s][0]) <= 0.05 for s in both), len(both))
    verdicts["F5 original/adam_b2_09 >= 0.70"] = (count(b2, lambda v: v >= 0.70), len(b2))
    verdicts["F5 refuted if < 0.30"] = (count(b2, lambda v: v < 0.30), len(b2))
    verdicts["F6 original/sgd >= 0.70"] = (count(sgd, lambda v: v >= 0.70), len(sgd))
    print()
    for k, (n, total) in verdicts.items():
        need = total if "every seed" in k else 4
        state = "n/a (no runs)" if total == 0 else ("holds" if n >= need and total == len(SEEDS) else ("incomplete" if total < len(SEEDS) else "does not hold"))
        print(f"{k}: {n}/{total} seeds -> {state}")
    missing = [(c, s) for c in CELLS for s in SEEDS if s not in data[c]]
    if missing:
        print("\nMissing runs:", missing); sys.exit(2)


if __name__ == "__main__":
    main()
