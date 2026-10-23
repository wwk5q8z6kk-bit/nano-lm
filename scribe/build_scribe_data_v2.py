# nano-scribe v2 training data — the SINGLE pre-specified improvement sweep from AUDIT.md.
# Levers (all diversity, no bar changes): templates x2-3, structural variation (section
# order, acks, fused complaint+duration), compositional complaint pool (~190 values so
# CC cannot be solved as classification), med list 8->18, N 8000->12000.
# THE EVAL SET IS NOT REGENERATED — scribe_eval.json from v1 is reused byte-identical.
import random
import numpy as np
from tokenizers import Tokenizer

random.seed(11)

CC_TRAIN = [("a cough", "cough"), ("a headache", "headache"), ("back pain", "back pain"),
            ("a sore throat", "sore throat"), ("chest pain", "chest pain"), ("dizziness", "dizziness"),
            ("a fever", "fever"), ("stomach pain", "stomach pain"), ("joint pain", "joint pain"),
            ("a rash", "rash"), ("fatigue", "fatigue"), ("shortness of breath", "shortness of breath"),
            ("an earache", "earache"), ("nausea", "nausea")]
HELD_CANON = {"toothache", "neck pain", "heartburn"}   # eval-only; excluded everywhere below

# compositional pool: body part x sensation -> ~190 values; forces COPYING over classification
PARTS = ["shoulder", "knee", "elbow", "wrist", "ankle", "hip", "lower back", "upper back",
         "jaw", "eye", "ear", "leg", "arm", "foot", "hand", "finger", "heel", "rib", "scalp"]
SENS = ["pain", "ache", "stiffness", "soreness", "numbness", "tingling",
        "cramping", "burning", "swelling", "itching"]
CC_COMP = [(f"{p} {s}", f"{p} {s}") for p in PARTS for s in SENS if f"{p} {s}" not in HELD_CANON]

MED_TRAIN = ["ibuprofen", "paracetamol", "aspirin", "antacids", "cough syrup",
             "allergy pills", "naproxen", "vitamin c", "zinc tablets", "magnesium",
             "fish oil", "nasal spray", "eye drops", "hydrocortisone cream",
             "loratadine", "cetirizine", "famotidine", "saline rinse"]   # held-out: melatonin, throat lozenges
ALG_TRAIN = ["penicillin", "peanuts", "pollen", "latex", "shellfish"]    # held-out: sulfa drugs
SEV = ["mild", "moderate", "severe"]

D_OPEN = ["Good morning, what brings you in today?", "Hello, what can I do for you?",
          "Hi there, what seems to be the trouble?", "So, tell me what's going on.",
          "What brings you to the clinic today?", "Hello, how can I help you today?",
          "Good afternoon, what's the reason for your visit?", "Hi, what are we seeing you for today?",
          "Welcome back. What's the concern this time?", "Okay, walk me through what's been happening.",
          "What symptoms brought you in?", "Tell me why you booked the appointment."]
P_CC = ["I've been having {cc}.", "I came in because of {cc}.", "It's {cc}, doctor.",
        "Well, I've got {cc} that won't go away.", "I'm dealing with {cc}.",
        "There's this {cc} bothering me.", "Mainly {cc}.", "My problem is {cc}.",
        "I keep getting {cc}.", "It's been {cc} on and off.", "I woke up with {cc} recently.",
        "The main thing is {cc}."]
P_CC_DUR = ["I've had {cc} for {n} {unit} now.", "It's {cc} — started about {n} {unit} ago.",
            "I've been dealing with {cc} for roughly {n} {unit}."]
D_DUR = ["How long has this been going on?", "When did it start?", "How many days has it been?",
         "How long have you had it?", "And this began when?", "For how long now?",
         "When did you first notice it?", "How far back does this go?"]
P_DUR = ["For about {n} {unit} now.", "It started {n} {unit} ago.", "Around {n} {unit}.",
         "Maybe {n} {unit}.", "{n} {unit}, give or take.", "Roughly {n} {unit} now.",
         "It's been {n} {unit}.", "Close to {n} {unit}."]
D_SEV = ["How bad would you say it is?", "Is it mild, moderate, or severe?",
         "How intense is it?", "How would you rate it?", "Is it manageable or quite bad?",
         "How severe does it feel?"]
P_SEV = ["I'd call it {sev}.", "It's {sev}, I would say.", "Pretty {sev}.",
         "Honestly, {sev}.", "{sev}, most days.", "I'd rate it {sev}.",
         "It feels {sev} to me.", "Somewhere around {sev}."]
D_MED = ["Have you taken anything for it?", "Are you on any medication for this?",
         "Any medicines so far?", "Have you tried treating it with anything?",
         "Are you taking anything currently?", "Any over-the-counter remedies?"]
P_MED_YES = ["I've been taking {med}.", "Just {med}.", "Some {med}, but it barely helps.",
             "{med}, twice a day.", "I picked up {med} from the pharmacy.",
             "Only {med} for now.", "I tried {med} last week.", "{med} here and there."]
P_MED_NO = ["No, nothing yet.", "I haven't taken anything.", "Not taking anything for it.",
            "No medication so far.", "Nothing, I wanted to check with you first."]
D_ALG = ["Any allergies I should know about?", "Are you allergic to anything?",
         "Any drug or food allergies?", "Do we have any allergies on file for you?",
         "Anything you're allergic to?", "Any known reactions to medications?"]
P_ALG_YES = ["I'm allergic to {alg}.", "Yes, {alg}.", "Just {alg}.",
             "{alg} — found out years ago.", "I react badly to {alg}.",
             "Yes — {alg}, it's on my chart.", "Only {alg}.", "{alg}, unfortunately."]
P_ALG_NO = ["No allergies.", "Not that I know of.", "None that I'm aware of.",
            "No, no allergies.", "Nothing on record."]
ACK = ["I see.", "Alright.", "Okay, noted.", "Understood.", "Got it."]
DISTRACT = ["The parking here was terrible today.", "It's been so cold this week.",
            "My cousin drove me here this morning.", "I almost rescheduled twice.",
            "The waiting room was quite full.", "My neighbor recommended this clinic.",
            "Traffic on the bridge was awful.", "I nearly forgot my insurance card."]

def sample_tuple():
    cc = random.choice(CC_COMP) if random.random() < 0.5 else random.choice(CC_TRAIN)
    unit = random.choice(["days", "weeks"])
    n = random.randint(2, 14) if unit == "days" else random.randint(1, 6)
    return {"cc": cc, "n": n, "unit": unit, "sev": random.choice(SEV),
            "med": random.choice(MED_TRAIN) if random.random() < 0.6 else None,
            "alg": random.choice(ALG_TRAIN) if random.random() < 0.5 else None}

def render_dialogue(t):
    lines = ["Doctor: " + random.choice(D_OPEN)]
    fused = random.random() < 0.3
    if fused:
        lines.append("Patient: " + random.choice(P_CC_DUR).format(cc=t["cc"][0], n=t["n"], unit=t["unit"]))
    else:
        lines.append("Patient: " + random.choice(P_CC).format(cc=t["cc"][0]))
    if random.random() < 0.4:
        lines.append("Patient: " + random.choice(DISTRACT))
    if not fused:
        if random.random() < 0.3: lines.append("Doctor: " + random.choice(ACK))
        lines.append("Doctor: " + random.choice(D_DUR))
        lines.append("Patient: " + random.choice(P_DUR).format(n=t["n"], unit=t["unit"]))
    lines.append("Doctor: " + random.choice(D_SEV))
    lines.append("Patient: " + random.choice(P_SEV).format(sev=t["sev"]))
    med_block = [("Doctor: " + random.choice(D_MED),
                  "Patient: " + (random.choice(P_MED_YES).format(med=t["med"]) if t["med"]
                                 else random.choice(P_MED_NO)))]
    alg_block = [("Doctor: " + random.choice(D_ALG),
                  "Patient: " + (random.choice(P_ALG_YES).format(alg=t["alg"]) if t["alg"]
                                 else random.choice(P_ALG_NO)))]
    blocks = med_block + alg_block
    random.shuffle(blocks)                      # section-order variation
    for dq, pa in blocks:
        if random.random() < 0.2: lines.append("Doctor: " + random.choice(ACK))
        lines.append(dq); lines.append(pa)
    return "\n".join(lines)

def summary_of(t):
    return (f"CC: {t['cc'][1]} | DUR: {t['n']} {t['unit']} | SEV: {t['sev']} | "
            f"MED: {t['med'] or 'none'} | ALG: {t['alg'] or 'none'}")

N_TRAIN = 12000
convos = []
for _ in range(N_TRAIN):
    t = sample_tuple()
    convos.append([{"role": "user", "content": render_dialogue(t) + "\nSummarize the visit."},
                   {"role": "assistant", "content": summary_of(t)}])

tok = Tokenizer.from_file("tokenizer.json")
ims_id, ime_id = tok.token_to_id("<|im_start|>"), tok.token_to_id("<|im_end|>")
S = 512

def render_ids(convo):
    ids, mask = [], []
    for m in convo:
        head = tok.encode(f"{m['role']}\n", add_special_tokens=False).ids
        body = tok.encode(m["content"], add_special_tokens=False).ids
        seg = [ims_id] + head + body + [ime_id]
        tf = 1 if m["role"] == "assistant" else 0
        for j, tkn in enumerate(seg):
            ids.append(tkn); mask.append(1 if (tf and j >= 1 + len(head)) else 0)
    return ids, mask

X, M, dropped = [], [], 0
for c in convos:
    ids, mask = render_ids(c)
    if len(ids) > S: dropped += 1; continue
    X.append(ids + [0] * (S - len(ids))); M.append(mask + [0] * (S - len(mask)))

np.save("scribe_x.npy", np.array(X, dtype=np.uint16))
np.save("scribe_mask.npy", np.array(M, dtype=np.uint8))
print(f"v2 train examples {len(X)} (dropped {dropped}); CC value space = {len(CC_TRAIN) + len(CC_COMP)}; "
      f"eval set NOT regenerated (v1 scribe_eval.json reused)", flush=True)
