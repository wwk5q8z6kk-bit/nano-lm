# Fieldwise breakdown of the own-stack held-out copying gap (drafting-surfaced analysis;
# writing_audit.md item 8). Reuses the VALIDATED scorer from rescore_anchors.py (same
# frozen checkpoints, same ChatML/argmax pipeline, same m0-m4 instances) and adds
# per-field held/seen recall so we can see whether the ~18-pt gap is concentrated in a
# few fields or spread across all five. Re-scoring only; no new training. Local/MPS.
import json, os, importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location("ra", os.path.join(os.path.dirname(__file__), "rescore_anchors.py"))
ra = importlib.util.module_from_spec(spec); spec.loader.exec_module(ra)
FIELDS, RE = ra.FIELDS, ra.RE

def fieldwise(m, items):
    # per field: [held_correct, held_total, seen_correct, seen_total]
    acc = {f: [0, 0, 0, 0] for f in FIELDS}
    for it in items:
        out = ra.generate(m, ra.prompt_ids(it["convo"][0]["content"]))
        text = ra.tok.decode(out[len(ra.prompt_ids(it["convo"][0]["content"])):]).strip()
        mm = RE.match(text)
        if not mm: continue
        pred = dict(zip(FIELDS, [g.strip() for g in mm.groups()]))
        for f in FIELDS:
            hit = int(pred[f] == it["tuple"][f])
            if it["held_values"]: acc[f][0] += hit; acc[f][1] += 1
            else: acc[f][2] += hit; acc[f][3] += 1
    return acc

def run(tag):
    m, _ = ra.load(tag)
    # accumulate per-field across the 5 fresh instances, then per-instance gap for SD
    per_inst = []
    for k in range(5):
        acc = fieldwise(m, ra.fresh[k])
        gaps = {f: (acc[f][2]/max(1,acc[f][3]) - acc[f][0]/max(1,acc[f][1]))*100 for f in FIELDS}
        per_inst.append(gaps)
    print(f"\n=== {tag}: per-field held-out gap (mean +/- SD over m0-m4), pts ===")
    out = {}
    for f in FIELDS:
        vals = [pi[f] for pi in per_inst]
        mean, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
        out[f] = {"gap_mean": mean, "gap_sd": sd, "per_instance": vals}
        print(f"  {f:4s}  {mean:6.1f} +/- {sd:4.1f}")
    total = sum(out[f]["gap_mean"] for f in FIELDS) / len(FIELDS)
    print(f"  (mean over fields = {total:.1f}; matches aggregate ~18)")
    return {"tag": tag, "checkpoint": ra.CFG[tag]["file"], "per_field": out,
            "note": "gap = seen-value recall - held-value recall, per field, over 5 instances (m0-m4)"}

if __name__ == "__main__":
    res = {t: run(t) for t in ("nano", "scale")}
    outp = os.path.join(os.path.dirname(__file__), "results_fieldwise_anchors.json")
    json.dump(res, open(outp, "w"), indent=1)
    print(f"\n-> {outp}")
