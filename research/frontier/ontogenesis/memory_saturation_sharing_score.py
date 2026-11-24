"""Score results/memory_saturation_sharing/ against G1-G4 of PROTOCOL-MEMSAT-SHARING.md. Read-only.

M = 0 is read from the factorial's original/adam_reset files, as the protocol states.
"""
import json, pathlib, sys

HERE = pathlib.Path(__file__).parent / "results"
SEEDS = [3101, 3102, 3103, 3104, 3105]
DOSES = [0, 2, 4, 6, 8]


def load(m, s):
    f = HERE / "memory_saturation_factorial" / f"original-adam_reset-s{s}.json" if m == 0 else HERE / "memory_saturation_sharing" / f"m{m}-adam_reset-s{s}.json"
    return json.load(open(f)) if f.exists() and f.stat().st_size else None


def main():
    d = {(m, s): load(m, s) for m in DOSES for s in SEEDS}
    missing = [k for k, v in d.items() if v is None]
    acq = lambda m, s: d[(m, s)]["late_acquire"] if d[(m, s)] else None
    print("| M | " + " | ".join(f"s{s}" for s in SEEDS) + " | mean acquire | mean retain | effective slots (range) |")
    print("|---|" + "---|" * (len(SEEDS) + 3))
    for m in DOSES:
        rows = [d[(m, s)] for s in SEEDS if d[(m, s)]]
        cells = [f"{acq(m, s):.3f}" if acq(m, s) is not None else "missing" for s in SEEDS]
        eff = [r["address_stats"]["effective_slots"] for r in rows if "address_stats" in r]
        ma = sum(r["late_acquire"] for r in rows) / len(rows) if rows else float("nan")
        mr = sum(r["late_retain"] for r in rows) / len(rows) if rows else float("nan")
        print(f"| {m} | " + " | ".join(cells) + f" | {ma:.3f} | {mr:.3f} | {f'{min(eff):.0f}-{max(eff):.0f}' if eff else 'see measure'} |")
    if missing:
        print("\nMissing runs:", missing); sys.exit(2)
    g1 = sum(all(acq(b, s) - acq(a, s) >= -0.02 for a, b in zip(DOSES, DOSES[1:])) for s in SEEDS)
    g2 = sum(acq(8, s) >= 0.95 for s in SEEDS)
    g3 = sum(acq(4, s) - acq(0, s) >= 0.05 and acq(8, s) - acq(4, s) >= 0.05 for s in SEEDS)
    print()
    for name, n in (("G1 no fall as M rises (tolerance 0.02)", g1), ("G2 M=8 >= 0.95", g2), ("G3 M=4 at least 0.05 from both ends", g3)):
        print(f"{name}: {n}/5 seeds -> {'holds' if n >= 4 else 'does not hold'}")


if __name__ == "__main__":
    main()
