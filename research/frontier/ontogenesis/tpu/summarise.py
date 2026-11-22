"""Collect the result files of the TPU baseline runs into one table. Usage: summarise.py <directory with *.json>"""
import glob, json, sys
rows = []
for f in sorted(glob.glob(sys.argv[1].rstrip("/") + "/dense-*.json")):
    d = json.load(open(f))
    if not d.get("curve"): continue
    e = d["curve"][-1]
    rows.append((d["params"], d["name"], d.get("finished", False), d["lr"], d["seed"], e["step"], d["steps"], e["tokens"], e["hours"], e.get("tokens_per_second"),
                 d.get("device", ""), {k: e[k] for k in ("web", "textbook", "math", "code", "mix")}, [(c["tokens"], c["mix"]) for c in d["curve"]]))
print("| Run | Parameters | Tokens seen | Web | Textbook | Maths | Code | Mix | Hours | Tokens/s | Done |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for p, name, fin, lr, seed, step, steps, tok, hrs, tps, dev, b, _ in sorted(rows):
    print(f"| {name} | {p/1e6:.1f}M | {tok/1e6:.0f}M | {b['web']:.4f} | {b['textbook']:.4f} | {b['math']:.4f} | {b['code']:.4f} | {b['mix']:.4f} | {hrs:.2f} | {tps} | {'yes' if fin else f'{step}/{steps}'} |")
if "--curves" in sys.argv:
    for p, name, *_, curve in sorted(rows): print(name, " ".join(f"{t/1e6:.0f}M:{m:.4f}" for t, m in curve))
