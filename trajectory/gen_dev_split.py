# Stage T instrument: dev split for Arm 2 prompt development + the frozen prompt.
# Pre-registered role (PREREG.md, prompt-freeze protocol): the prompt is developed
# ONLY against this split; dev items are never scored and never overlap either
# eval instance. Seed 20260718 (= instance-T seed + 1, recorded).
#
# Structure (design rationale recorded in PREREG Amendment 2):
#   items 0-4  "exemplar"    — SEEN values, TRAIN templates. Few-shot pool: held
#                              values must never appear in-context (would collapse
#                              the held/seen distinction for Arm 2), and
#                              train-template exemplars mirror Arm 1's training
#                              exposure (trained on train templates, evaluated on
#                              held templates) so the arms stay analogous.
#   items 5-9  "dev_target"  — HELD templates, items 5-6 held values. For prompt
#                              sanity checks only; never scored, never embedded.
#
# Also emits arm2_prompt.txt deterministically (exemplars 0-2 embedded).
import json, random, os

V1 = os.path.join(os.path.dirname(__file__), "..", "scribe", "build_scribe_data.py")
SEED = 20260718
HERE = os.path.dirname(os.path.abspath(__file__))

src = open(V1).read()
marker = "# ---------------- build sets ----------------"
assert marker in src, "v1 generator layout changed; re-verify before generating"
ns = {}
exec(compile(src.split(marker)[0], V1, "exec"), ns)
sample_tuple, make_convo = ns["sample_tuple"], ns["make_convo"]

random.seed(SEED)

items = []
for i in range(5):                                   # exemplars: seen values, train templates
    t = sample_tuple(False)
    items.append({"role": "exemplar",
                  "tuple": {"cc": t["cc"][1], "dur": f"{t['n']} {t['unit']}", "sev": t["sev"],
                            "med": t["med"] or "none", "alg": t["alg"] or "none"},
                  "held_values": False, "convo": make_convo(t, False)})
for i in range(5):                                   # sanity targets: held templates
    held_vals = i < 2
    t = sample_tuple(held_vals)
    items.append({"role": "dev_target",
                  "tuple": {"cc": t["cc"][1], "dur": f"{t['n']} {t['unit']}", "sev": t["sev"],
                            "med": t["med"] or "none", "alg": t["alg"] or "none"},
                  "held_values": held_vals, "convo": make_convo(t, True)})

json.dump(items, open(os.path.join(HERE, "scribe_dev.json"), "w"), indent=1)

# QA: no dialogue collisions with either eval instance
inst0 = json.load(open(os.path.join(HERE, "..", "scribe", "scribe_eval.json")))
instT = json.load(open(os.path.join(HERE, "scribe_eval_T.json")))
eval_convos = {e["convo"][0]["content"] for e in inst0} | {e["convo"][0]["content"] for e in instT}
collisions = sum(1 for e in items if e["convo"][0]["content"] in eval_convos)

# Frozen Arm 2 prompt (exemplars 0-2). {dialogue} is the substitution slot.
def dialogue_text(it):
    return it["convo"][0]["content"].replace("\nSummarize the visit.", "")

def summary_line(it):
    t = it["tuple"]
    return f"CC: {t['cc']} | DUR: {t['dur']} | SEV: {t['sev']} | MED: {t['med']} | ALG: {t['alg']}"

blocks = "\n\n".join(f"Dialogue:\n{dialogue_text(items[i])}\nSummary: {summary_line(items[i])}"
                     for i in range(3))
prompt = f"""You extract a structured visit summary from a clinic dialogue.

Output EXACTLY one line in this format and nothing else:
CC: <chief complaint> | DUR: <duration> | SEV: <severity> | MED: <medication or none> | ALG: <allergy or none>

Rules:
- Copy each value from the dialogue; do not infer, expand, or normalize beyond the examples.
- SEV is one of: mild, moderate, severe.
- If no medication or no allergy is mentioned, write: none
- Ignore small talk and irrelevant remarks.

{blocks}

Dialogue:
{{dialogue}}
Summary:"""

open(os.path.join(HERE, "arm2_prompt.txt"), "w").write(prompt)
print(f"dev split: {len(items)} items (5 exemplar / 5 dev_target, {sum(e['held_values'] for e in items)} held-value); "
      f"seed {SEED}; collisions vs eval instances: {collisions}/10; "
      f"prompt: {len(prompt)} chars, 3 exemplars embedded")
held_leak = any(v in prompt for v in
                ["toothache", "neck pain", "heartburn", "melatonin", "throat lozenges", "sulfa drugs"])
print(f"held values leaked into frozen prompt: {held_leak}")
