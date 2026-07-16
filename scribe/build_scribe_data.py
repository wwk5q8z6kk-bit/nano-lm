# nano-scribe data: synthetic clinic dialogues rendered from KNOWN fact tuples.
# Faithfulness is measurable by construction: the gate compares the model's summary
# fields to the generating tuple — no judge model needed.
# Contamination control (pre-registered in AUDIT.md):
#   - held-out TEMPLATE families (doctor + patient phrasings) never seen in training
#   - held-out SLOT VALUES (3 complaints, 2 meds, 1 allergy) never seen in training
import json, random
import numpy as np
from tokenizers import Tokenizer

random.seed(7)

# ---------------- slot vocabularies (dialog form -> canonical summary form) ----------------
CC_TRAIN = [("a cough", "cough"), ("a headache", "headache"), ("back pain", "back pain"),
            ("a sore throat", "sore throat"), ("chest pain", "chest pain"), ("dizziness", "dizziness"),
            ("a fever", "fever"), ("stomach pain", "stomach pain"), ("joint pain", "joint pain"),
            ("a rash", "rash"), ("fatigue", "fatigue"), ("shortness of breath", "shortness of breath"),
            ("an earache", "earache"), ("nausea", "nausea")]
CC_HELD = [("a toothache", "toothache"), ("neck pain", "neck pain"), ("heartburn", "heartburn")]

MED_TRAIN = ["ibuprofen", "paracetamol", "aspirin", "antacids", "cough syrup",
             "allergy pills", "naproxen", "vitamin c"]
MED_HELD = ["melatonin", "throat lozenges"]

ALG_TRAIN = ["penicillin", "peanuts", "pollen", "latex", "shellfish"]
ALG_HELD = ["sulfa drugs"]

SEV = ["mild", "moderate", "severe"]

# ---------------- template families ----------------
D_OPEN_TRAIN = ["Good morning, what brings you in today?", "Hello, what can I do for you?",
                "Hi there, what seems to be the trouble?", "So, tell me what's going on.",
                "What brings you to the clinic today?"]
D_OPEN_HELD = ["Morning — what's been bothering you?", "Come in, have a seat. What's the issue today?"]

P_CC_TRAIN = ["I've been having {cc}.", "I came in because of {cc}.", "It's {cc}, doctor.",
              "Well, I've got {cc} that won't go away.", "I'm dealing with {cc}."]
P_CC_HELD = ["Honestly, {cc} has been troubling me.", "It started as {cc} and hasn't stopped."]

D_DUR_TRAIN = ["How long has this been going on?", "When did it start?", "How many days has it been?"]
D_DUR_HELD = ["Since when have you had it?"]

P_DUR_TRAIN = ["For about {n} {unit} now.", "It started {n} {unit} ago.", "Around {n} {unit}."]
P_DUR_HELD = ["I'd say it's been {n} {unit}."]

D_SEV_TRAIN = ["How bad would you say it is?", "Is it mild, moderate, or severe?"]
D_SEV_HELD = ["On a scale from mild to severe, where is it?"]

P_SEV_TRAIN = ["I'd call it {sev}.", "It's {sev}, I would say.", "Pretty {sev}."]
P_SEV_HELD = ["Definitely {sev}."]

D_MED_TRAIN = ["Have you taken anything for it?", "Are you on any medication for this?"]
D_MED_HELD = ["Did you try any medicine?"]

P_MED_YES_TRAIN = ["I've been taking {med}.", "Just {med}.", "Some {med}, but it barely helps."]
P_MED_YES_HELD = ["Only {med} so far."]
P_MED_NO_TRAIN = ["No, nothing yet.", "I haven't taken anything."]
P_MED_NO_HELD = ["Nothing at all."]

D_ALG_TRAIN = ["Any allergies I should know about?", "Are you allergic to anything?"]
D_ALG_HELD = ["Do you have any known allergies?"]

P_ALG_YES_TRAIN = ["I'm allergic to {alg}.", "Yes, {alg}.", "Just {alg}."]
P_ALG_YES_HELD = ["I do — {alg}."]
P_ALG_NO_TRAIN = ["No allergies.", "Not that I know of."]
P_ALG_NO_HELD = ["None whatsoever."]

DISTRACT = ["The parking here was terrible today.", "It's been so cold this week.",
            "My cousin drove me here this morning.", "I almost rescheduled twice.",
            "The waiting room was quite full."]

def sample_tuple(held: bool):
    cc = random.choice(CC_HELD if (held and random.random() < 0.7) else CC_TRAIN)
    unit = random.choice(["days", "weeks"])
    n = random.randint(2, 14) if unit == "days" else random.randint(1, 6)
    sev = random.choice(SEV)
    med = None
    if random.random() < 0.6:
        med = random.choice(MED_HELD if (held and random.random() < 0.5) else MED_TRAIN)
    alg = None
    if random.random() < 0.5:
        alg = random.choice(ALG_HELD if (held and random.random() < 0.5) else ALG_TRAIN)
    return {"cc": cc, "n": n, "unit": unit, "sev": sev, "med": med, "alg": alg}

def pick(train_list, held_list, held: bool):
    return random.choice(held_list if held else train_list)

def render_dialogue(t, held: bool):
    lines = []
    lines.append("Doctor: " + pick(D_OPEN_TRAIN, D_OPEN_HELD, held))
    lines.append("Patient: " + pick(P_CC_TRAIN, P_CC_HELD, held).format(cc=t["cc"][0]))
    if random.random() < 0.4:
        lines.append("Patient: " + random.choice(DISTRACT))
    lines.append("Doctor: " + pick(D_DUR_TRAIN, D_DUR_HELD, held))
    lines.append("Patient: " + pick(P_DUR_TRAIN, P_DUR_HELD, held).format(n=t["n"], unit=t["unit"]))
    lines.append("Doctor: " + pick(D_SEV_TRAIN, D_SEV_HELD, held))
    lines.append("Patient: " + pick(P_SEV_TRAIN, P_SEV_HELD, held).format(sev=t["sev"]))
    lines.append("Doctor: " + pick(D_MED_TRAIN, D_MED_HELD, held))
    if t["med"]:
        lines.append("Patient: " + pick(P_MED_YES_TRAIN, P_MED_YES_HELD, held).format(med=t["med"]))
    else:
        lines.append("Patient: " + pick(P_MED_NO_TRAIN, P_MED_NO_HELD, held))
    lines.append("Doctor: " + pick(D_ALG_TRAIN, D_ALG_HELD, held))
    if t["alg"]:
        lines.append("Patient: " + pick(P_ALG_YES_TRAIN, P_ALG_YES_HELD, held).format(alg=t["alg"]))
    else:
        lines.append("Patient: " + pick(P_ALG_NO_TRAIN, P_ALG_NO_HELD, held))
    return "\n".join(lines)

def summary_of(t):
    return (f"CC: {t['cc'][1]} | DUR: {t['n']} {t['unit']} | SEV: {t['sev']} | "
            f"MED: {t['med'] or 'none'} | ALG: {t['alg'] or 'none'}")

def make_convo(t, held: bool):
    dia = render_dialogue(t, held)
    return [{"role": "user", "content": dia + "\nSummarize the visit."},
            {"role": "assistant", "content": summary_of(t)}]

# ---------------- build sets ----------------
N_TRAIN, N_EVAL = 8000, 40
train = [make_convo(sample_tuple(False), False) for _ in range(N_TRAIN)]

eval_items = []
for i in range(N_EVAL):
    held_vals = i < 20                       # half the eval set uses held-out slot values
    t = sample_tuple(held_vals)
    eval_items.append({"tuple": {"cc": t["cc"][1], "dur": f"{t['n']} {t['unit']}", "sev": t["sev"],
                                 "med": t["med"] or "none", "alg": t["alg"] or "none"},
                       "held_values": held_vals,
                       "convo": make_convo(t, True)})   # ALL eval dialogues use held-out templates

json.dump(eval_items, open("scribe_eval.json", "w"), indent=1)

# ---------------- tokenize with the posttrain tokenizer (V=4098, has role tokens) ----------------
tok = Tokenizer.from_file("tokenizer.json")
IMS, IME = "<|im_start|>", "<|im_end|>"
ims_id, ime_id = tok.token_to_id(IMS), tok.token_to_id(IME)
S = 512

def render_ids(convo):
    ids, mask = [], []
    for m in convo:
        head = tok.encode(f"{m['role']}\n", add_special_tokens=False).ids
        body = tok.encode(m["content"], add_special_tokens=False).ids
        seg = [ims_id] + head + body + [ime_id]
        train_flag = 1 if m["role"] == "assistant" else 0
        for j, tkn in enumerate(seg):
            ids.append(tkn)
            mask.append(1 if (train_flag and j >= 1 + len(head)) else 0)
    return ids, mask

X, M, dropped = [], [], 0
for c in train:
    ids, mask = render_ids(c)
    if len(ids) > S:
        dropped += 1; continue
    X.append(ids + [0] * (S - len(ids)))
    M.append(mask + [0] * (S - len(mask)))

np.save("scribe_x.npy", np.array(X, dtype=np.uint16))
np.save("scribe_mask.npy", np.array(M, dtype=np.uint8))
sup = int(np.array(M).sum())
print(f"train examples {len(X)} (dropped {dropped} >512tok); supervised tokens {sup}; "
      f"eval {len(eval_items)} (all held-out templates; {sum(e['held_values'] for e in eval_items)} with held-out values)",
      flush=True)
