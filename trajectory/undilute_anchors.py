# CLEAN (undiluted) per-field held-out-VALUE gap for the own-stack anchors (resolves
# AAEA finding A2). The paper's gap splits seen/held at the DIALOGUE level, so held-bucket
# fields are diluted with seen values (68% cc / 30% med / 21% alg actually held-out). Here
# we split by the ACTUAL field value: among value-bearing items (all under held templates),
# held-value-recall vs seen-value-recall, per field. This isolates copying a novel lexical
# value from copying a familiar one, template held fixed. dur/sev have no held-out values
# -> undefined (correctly). Re-scoring only; reuses the validated scorer.
import json, os, importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location("ra", os.path.join(os.path.dirname(__file__), "rescore_anchors.py"))
ra = importlib.util.module_from_spec(spec); spec.loader.exec_module(ra)
FIELDS, RE = ra.FIELDS, ra.RE

HELD = {"cc": {"toothache", "neck pain", "heartburn"},
        "med": {"melatonin", "throat lozenges"}, "alg": {"sulfa drugs"}}
VALFIELDS = ["cc", "med", "alg"]          # the fields that HAVE held-out values

def clean(m, items):
    # per field: [held_correct, held_total, seen_correct, seen_total]  (value-bearing only)
    acc = {f: [0, 0, 0, 0] for f in VALFIELDS}
    for it in items:
        out = ra.generate(m, ra.prompt_ids(it["convo"][0]["content"]))
        text = ra.tok.decode(out[len(ra.prompt_ids(it["convo"][0]["content"])):]).strip()
        mm = RE.match(text)
        if not mm: continue
        pred = dict(zip(FIELDS, [g.strip() for g in mm.groups()]))
        for f in VALFIELDS:
            t = it["tuple"][f]
            if t == "none": continue           # absence axis, not value copying
            hit = int(pred[f] == t)
            if t in HELD[f]: acc[f][0] += hit; acc[f][1] += 1
            else:            acc[f][2] += hit; acc[f][3] += 1
    return acc

def run(tag):
    m, _ = ra.load(tag)
    per_inst = []
    for k in range(5):
        acc = clean(m, ra.fresh[k])
        gaps = {f: (acc[f][2] / max(1, acc[f][3]) - acc[f][0] / max(1, acc[f][1])) * 100 for f in VALFIELDS}
        counts = {f: {"held_n": acc[f][1], "seen_n": acc[f][3],
                      "held_recall": acc[f][0] / max(1, acc[f][1]),
                      "seen_recall": acc[f][2] / max(1, acc[f][3])} for f in VALFIELDS}
        per_inst.append({"gaps": gaps, "counts": counts})
    print(f"\n=== {tag}: CLEAN per-field held-out-VALUE gap (value-bearing items only) ===")
    out = {}
    for f in VALFIELDS:
        vals = [pi["gaps"][f] for pi in per_inst]
        mean, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
        hn = int(np.mean([pi["counts"][f]["held_n"] for pi in per_inst]))
        sn = int(np.mean([pi["counts"][f]["seen_n"] for pi in per_inst]))
        out[f] = {"gap_mean": mean, "gap_sd": sd, "avg_held_n": hn, "avg_seen_n": sn,
                  "per_instance": vals}
        print(f"  {f:4s}  {mean:6.1f} +/- {sd:4.1f}   (held n~{hn}/inst, seen n~{sn}/inst)")
    # clean aggregate over the value fields (pooled held vs seen value recall)
    agg_gaps = []
    for k in range(5):
        acc = clean(m, ra.fresh[k])
        hc = sum(acc[f][0] for f in VALFIELDS); ht = sum(acc[f][1] for f in VALFIELDS)
        sc = sum(acc[f][2] for f in VALFIELDS); st = sum(acc[f][3] for f in VALFIELDS)
        agg_gaps.append((sc / max(1, st) - hc / max(1, ht)) * 100)
    agg_mean, agg_sd = float(np.mean(agg_gaps)), float(np.std(agg_gaps, ddof=1))
    print(f"  CLEAN AGG (cc+med+alg value-bearing) {agg_mean:.1f} +/- {agg_sd:.1f}  "
          f"(vs diluted dialogue-level {'18.3' if tag=='nano' else '18.7'})")
    return {"tag": tag, "per_field_clean": out, "clean_aggregate": {"gap_mean": agg_mean, "gap_sd": agg_sd, "per_instance": agg_gaps},
            "note": "held-value vs seen-value recall among value-bearing items (excl 'none'), under held templates; dur/sev omitted (no held-out values)"}

if __name__ == "__main__":
    res = {t: run(t) for t in ("nano", "scale")}
    outp = os.path.join(os.path.dirname(__file__), "results_undilute_anchors.json")
    json.dump(res, open(outp, "w"), indent=1)
    print(f"\n-> {outp}")
