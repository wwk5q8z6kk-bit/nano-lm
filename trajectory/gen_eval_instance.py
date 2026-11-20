# Stage T instrument: generate a fresh eval instance from the v1 generator's
# distribution (build_scribe_data.py lines 119-130) under a new recorded seed.
#
# Method: exec the v1 source UP TO the "build sets" marker so all slot
# vocabularies, template families, and rendering functions come from the
# published generator verbatim (no duplication drift), then re-run only the
# eval loop. The training-set build, tokenization, and file writes in v1 are
# never executed.
#
# Instance-T seed: 20260717 (date-derived, fixed before generation, recorded
# in trajectory/PREREG.md).
import json, random, os

V1 = os.path.join(os.path.dirname(__file__), "..", "scribe", "build_scribe_data.py")
SEED = 20260717
OUT = os.path.join(os.path.dirname(__file__), "scribe_eval_T.json")

src = open(V1).read()
marker = "# ---------------- build sets ----------------"
assert marker in src, "v1 generator layout changed; re-verify before generating"
prefix = src.split(marker)[0]
ns = {}
exec(compile(prefix, V1, "exec"), ns)  # definitions only; no dataset writes

random.seed(SEED)
ns["random"] = random  # v1 functions reference module-level `random`; same object anyway

sample_tuple, make_convo = ns["sample_tuple"], ns["make_convo"]

# Eval loop, mirroring v1 lines 123-130 exactly (N_EVAL=40, first 20 held-value).
eval_items = []
for i in range(40):
    held_vals = i < 20
    t = sample_tuple(held_vals)
    eval_items.append({"tuple": {"cc": t["cc"][1], "dur": f"{t['n']} {t['unit']}", "sev": t["sev"],
                                 "med": t["med"] or "none", "alg": t["alg"] or "none"},
                       "held_values": held_vals,
                       "convo": make_convo(t, True)})

json.dump(eval_items, open(OUT, "w"), indent=1)

# Instrument QA: schema + collision report vs instance 0 (held template families
# are tiny, so individual dialogue collisions are possible; measured, not hidden).
inst0 = json.load(open(os.path.join(os.path.dirname(__file__), "..", "scribe", "scribe_eval.json")))
convos0 = {e["convo"][0]["content"] for e in inst0}
collisions = sum(1 for e in eval_items if e["convo"][0]["content"] in convos0)
held_n = sum(e["held_values"] for e in eval_items)
print(f"instance T: {len(eval_items)} items, {held_n} held-value; "
      f"seed {SEED}; byte-identical dialogue collisions vs instance 0: {collisions}/40")
