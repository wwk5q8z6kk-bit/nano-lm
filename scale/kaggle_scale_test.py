# Stage S — ~10M-param scale test. Runs end-to-end on a free Kaggle/Colab GPU (T4/P100).
# Bars PRE-REGISTERED in scale/AUDIT.md (committed to the repo before this ran).
# Phases: [1] pretrain ~200M FineWeb tokens  [2] scribe finetune (v2 recipe)  [3] gate.
# Usage (Kaggle notebook, GPU on):  !pip -q install datasets tokenizers
#                                   !python kaggle_scale_test.py
# Resume-safe: checkpoints every 1000 steps; rerun the same cell to continue.
import json, math, os, random, re, time, urllib.request
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0); random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
assert dev == "cuda", "No GPU — enable the accelerator (Kaggle: Settings > Accelerator > GPU)"

RAW = "https://raw.githubusercontent.com/wwk5q8z6kk-bit/nano-lm/master"
for path, url in [("tokenizer.json", f"{RAW}/sft/tokenizer.json"),
                  ("scribe_eval.json", f"{RAW}/scribe/scribe_eval.json")]:
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path); print("fetched", path, flush=True)

from tokenizers import Tokenizer
tok = Tokenizer.from_file("tokenizer.json")
IMS, IME = tok.token_to_id("<|im_start|>"), tok.token_to_id("<|im_end|>")

# ---------------- model: ~10M params (d=320, L=8, GQA 8q:2kv) ----------------
V = 4098
d, L, H, KV, hd, ff, S = 320, 8, 8, 2, 40, 864, 512

def rope(q, k):
    t = torch.arange(S, device=dev, dtype=torch.float32)
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=dev).float() / hd))
    f = torch.outer(t, inv); cos, sin = f.cos()[None, None], f.sin()[None, None]
    def rot(x):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
    return rot(q), rot(k)

class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.n1, s.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        s.q, s.k, s.v, s.o = (nn.Linear(d, H*hd, bias=False), nn.Linear(d, KV*hd, bias=False),
                              nn.Linear(d, KV*hd, bias=False), nn.Linear(H*hd, d, bias=False))
        s.g, s.u, s.dn = nn.Linear(d, ff, bias=False), nn.Linear(d, ff, bias=False), nn.Linear(ff, d, bias=False)
    def forward(s, x):
        B = x.shape[0]; h = s.n1(x)
        q = s.q(h).view(B, S, H, hd).transpose(1, 2); k = s.k(h).view(B, S, KV, hd).transpose(1, 2); v = s.v(h).view(B, S, KV, hd).transpose(1, 2)
        q, k = rope(q, k)
        k, v = k.repeat_interleave(H//KV, 1), v.repeat_interleave(H//KV, 1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.o(a.transpose(1, 2).reshape(B, S, H*hd))
        h = s.n2(x)
        return x + s.dn(F.silu(s.g(h)) * s.u(h))

class GPT(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb = nn.Embedding(V, d); s.blocks = nn.ModuleList(Block() for _ in range(L)); s.nf = nn.RMSNorm(d)
        for p in s.parameters():
            if p.dim() >= 2: nn.init.normal_(p, std=0.02 / math.sqrt(2 * L))
        nn.init.normal_(s.emb.weight, std=0.02)
    def forward(s, x):
        h = s.emb(x)
        for b in s.blocks: h = b(h)
        return F.linear(s.nf(h), s.emb.weight)

# ---------------- phase 1: pretrain ----------------
TARGET_TOKENS, BATCH, PEAK_LR, WARM_FRAC, FLOOR = 200_000_000, 32, 1.8e-3, 0.03, 0.1
STEPS = TARGET_TOKENS // (BATCH * S)          # ~12.2k
CKPT = "scale10m_pretrain.pt"

if not os.path.exists("shard10m.npy"):
    print("streaming FineWeb (this takes ~10-20 min of CPU tokenization)...", flush=True)
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    ids, n_docs = [], 0
    for doc in ds:
        ids.extend(tok.encode(doc["text"], add_special_tokens=False).ids); ids.append(0)
        n_docs += 1
        if len(ids) >= TARGET_TOKENS + 2_000_000: break
        if n_docs % 20000 == 0: print(f"  {n_docs} docs, {len(ids)/1e6:.0f}M tokens", flush=True)
    np.save("shard10m.npy", np.array(ids, dtype=np.uint16))
    print(f"shard: {len(ids)/1e6:.0f}M tokens from {n_docs} docs", flush=True)

shard = np.load("shard10m.npy")
val_ids = torch.tensor(shard[-1_000_000:].astype(np.int64))
train_ids = torch.tensor(shard[:-1_000_000].astype(np.int64))

m = GPT().to(dev)
print(f"params={sum(p.numel() for p in m.parameters())/1e6:.2f}M  steps={STEPS}", flush=True)
decay = [p for p in m.parameters() if p.dim() >= 2]; nodecay = [p for p in m.parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
                        lr=PEAK_LR, betas=(0.9, 0.95), eps=1e-8)
scaler = torch.amp.GradScaler("cuda")
start_step = 0
if os.path.exists(CKPT):
    st = torch.load(CKPT, map_location=dev, weights_only=True)
    m.load_state_dict(st["m"]); opt.load_state_dict(st["o"]); scaler.load_state_dict(st["s"]); start_step = st["step"]
    print(f"resumed from step {start_step}", flush=True)

WARM = int(WARM_FRAC * STEPS)
def lr_at(t):
    if t < WARM: return PEAK_LR * t / max(1, WARM)
    p = (t - WARM) / max(1, STEPS - WARM)
    return PEAK_LR * (FLOOR + (1 - FLOOR) * 0.5 * (1 + math.cos(math.pi * p)))

def batch_from(ids_t):
    i = torch.randint(0, len(ids_t) - S - 1, (BATCH,))
    return torch.stack([ids_t[j:j+S] for j in i]).to(dev)

def ce_loss(logits, tgt):
    ce = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1))
    z = torch.logsumexp(logits.float(), -1)
    return ce + 1e-4 * (z ** 2).mean()

if start_step < STEPS:
    t0 = time.time()
    for step in range(start_step + 1, STEPS + 1):
        for g in opt.param_groups: g["lr"] = lr_at(step)
        x = batch_from(train_ids)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = m(x)[:, :-1]
            loss = ce_loss(logits, x[:, 1:])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        if step % 200 == 0 or step == start_step + 1:
            tput = (step - start_step) * BATCH * S / (time.time() - t0) / 1e3
            print(f"step {step:6d}/{STEPS}  loss {loss.item():.3f}  gnorm {gn.item():.2f}  {tput:.0f}k tok/s", flush=True)
        if step % 1000 == 0:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                vx = batch_from(val_ids); vl = ce_loss(m(vx)[:, :-1], vx[:, 1:]).item()
            print(f"  == val loss {vl:.3f} ==", flush=True)
            torch.save({"m": m.state_dict(), "o": opt.state_dict(), "s": scaler.state_dict(), "step": step}, CKPT)
    torch.save({"m": m.state_dict(), "o": opt.state_dict(), "s": scaler.state_dict(), "step": STEPS}, CKPT)
    print("pretrain done", flush=True)

base_sd = {k: v.clone() for k, v in m.state_dict().items()}    # base control for the gate

# ---------------- phase 2: scribe finetune (v2 recipe verbatim, inline) ----------------
random.seed(11)
CC_TRAIN = [("a cough", "cough"), ("a headache", "headache"), ("back pain", "back pain"),
            ("a sore throat", "sore throat"), ("chest pain", "chest pain"), ("dizziness", "dizziness"),
            ("a fever", "fever"), ("stomach pain", "stomach pain"), ("joint pain", "joint pain"),
            ("a rash", "rash"), ("fatigue", "fatigue"), ("shortness of breath", "shortness of breath"),
            ("an earache", "earache"), ("nausea", "nausea")]
HELD_CANON = {"toothache", "neck pain", "heartburn"}
PARTS = ["shoulder", "knee", "elbow", "wrist", "ankle", "hip", "lower back", "upper back",
         "jaw", "eye", "ear", "leg", "arm", "foot", "hand", "finger", "heel", "rib", "scalp"]
SENS = ["pain", "ache", "stiffness", "soreness", "numbness", "tingling",
        "cramping", "burning", "swelling", "itching"]
CC_COMP = [(f"{p} {s}", f"{p} {s}") for p in PARTS for s in SENS if f"{p} {s}" not in HELD_CANON]
MED_TRAIN = ["ibuprofen", "paracetamol", "aspirin", "antacids", "cough syrup",
             "allergy pills", "naproxen", "vitamin c", "zinc tablets", "magnesium",
             "fish oil", "nasal spray", "eye drops", "hydrocortisone cream",
             "loratadine", "cetirizine", "famotidine", "saline rinse"]
ALG_TRAIN = ["penicillin", "peanuts", "pollen", "latex", "shellfish"]
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
    if random.random() < 0.4: lines.append("Patient: " + random.choice(DISTRACT))
    if not fused:
        if random.random() < 0.3: lines.append("Doctor: " + random.choice(ACK))
        lines.append("Doctor: " + random.choice(D_DUR))
        lines.append("Patient: " + random.choice(P_DUR).format(n=t["n"], unit=t["unit"]))
    lines.append("Doctor: " + random.choice(D_SEV))
    lines.append("Patient: " + random.choice(P_SEV).format(sev=t["sev"]))
    med_block = [("Doctor: " + random.choice(D_MED),
                  "Patient: " + (random.choice(P_MED_YES).format(med=t["med"]) if t["med"] else random.choice(P_MED_NO)))]
    alg_block = [("Doctor: " + random.choice(D_ALG),
                  "Patient: " + (random.choice(P_ALG_YES).format(alg=t["alg"]) if t["alg"] else random.choice(P_ALG_NO)))]
    blocks = med_block + alg_block
    random.shuffle(blocks)
    for dq, pa in blocks:
        if random.random() < 0.2: lines.append("Doctor: " + random.choice(ACK))
        lines.append(dq); lines.append(pa)
    return "\n".join(lines)

def summary_of(t):
    return (f"CC: {t['cc'][1]} | DUR: {t['n']} {t['unit']} | SEV: {t['sev']} | "
            f"MED: {t['med'] or 'none'} | ALG: {t['alg'] or 'none'}")

def render_ids(convo):
    ids, mask = [], []
    for msg in convo:
        head = tok.encode(f"{msg['role']}\n", add_special_tokens=False).ids
        body = tok.encode(msg["content"], add_special_tokens=False).ids
        seg = [IMS] + head + body + [IME]
        tf = 1 if msg["role"] == "assistant" else 0
        for j, tkn in enumerate(seg):
            ids.append(tkn); mask.append(1 if (tf and j >= 1 + len(head)) else 0)
    return ids, mask

if not os.path.exists("scale10m_scribe.pt"):
    X, M = [], []
    for _ in range(12000):
        t = sample_tuple()
        c = [{"role": "user", "content": render_dialogue(t) + "\nSummarize the visit."},
             {"role": "assistant", "content": summary_of(t)}]
        ids, mask = render_ids(c)
        if len(ids) > S: continue
        X.append(ids + [0] * (S - len(ids))); M.append(mask + [0] * (S - len(mask)))
    X = torch.tensor(np.array(X, dtype=np.int64)); M = torch.tensor(np.array(M, dtype=np.int64))
    N = X.shape[0]; FSTEPS = (N // BATCH) * 3; FWARM = int(0.03 * FSTEPS); FLR = 1e-4
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=FLR, betas=(0.9, 0.95), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda")
    perm = torch.randperm(N)
    def flr_at(t):
        if t < FWARM: return FLR * t / max(1, FWARM)
        p = (t - FWARM) / max(1, FSTEPS - FWARM)
        return FLR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))
    t0 = time.time()
    for step in range(1, FSTEPS + 1):
        for g in opt.param_groups: g["lr"] = flr_at(step)
        i0 = (step * BATCH) % (N - BATCH); idx = perm[i0:i0 + BATCH]
        x, msk = X[idx].to(dev), M[idx].to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = m(x)[:, :-1]; tgt = x[:, 1:]; mtgt = msk[:, 1:]
            ce = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), reduction="none").reshape(tgt.shape)
            loss = (ce * mtgt).sum() / mtgt.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        if step % 200 == 0: print(f"scribe {step}/{FSTEPS}  loss {loss.item():.3f}", flush=True)
    torch.save(m.state_dict(), "scale10m_scribe.pt")
    print(f"scribe finetune done in {(time.time()-t0)/60:.1f} min", flush=True)
else:
    m.load_state_dict(torch.load("scale10m_scribe.pt", map_location=dev, weights_only=True))

# ---------------- phase 3: the pre-registered gate ----------------
def prompt_ids(user_content):
    return ([IMS] + tok.encode("user\n", add_special_tokens=False).ids
            + tok.encode(user_content, add_special_tokens=False).ids + [IME]
            + [IMS] + tok.encode("assistant\n", add_special_tokens=False).ids)

@torch.no_grad()
def generate(model, ids, max_new=64):
    ids = list(ids)
    for _ in range(max_new):
        if len(ids) >= S: break
        x = torch.tensor([ids + [0] * (S - len(ids))], device=dev)
        nxt = int(model(x)[0, len(ids) - 1].argmax())
        if nxt == IME: break
        ids.append(nxt)
    return ids

RE = re.compile(r"^CC: (.+?) \| DUR: (.+?) \| SEV: (.+?) \| MED: (.+?) \| ALG: (.+?)$")
FIELDS = ["cc", "dur", "sev", "med", "alg"]

def score(model, items, label):
    parsed = correct = halluc = omission = 0
    hc = ht = sc = st = 0
    for it in items:
        pids = prompt_ids(it["convo"][0]["content"])
        text = tok.decode(generate(model, pids)[len(pids):]).strip()
        mm = RE.match(text)
        if not mm: continue
        parsed += 1
        pred = dict(zip(FIELDS, [g.strip() for g in mm.groups()]))
        for f in FIELDS:
            t, p = it["tuple"][f], pred[f]
            hit = p == t
            if hit: correct += 1
            elif p == "none" and t != "none": omission += 1
            else: halluc += 1
            if it["held_values"]: ht += 1; hc += hit
            else: st += 1; sc += hit
    n = len(items); tf = n * 5
    print(f"[{label}] parse {parsed}/{n}={parsed/n:.0%}  recall {correct}/{tf}={correct/tf:.0%}  "
          f"halluc {halluc}/{tf}={halluc/tf:.1%}  omission {omission}", flush=True)
    gap = (sc/max(1,st) - hc/max(1,ht)) * 100
    print(f"          held-out-value recall {hc}/{ht}={hc/max(1,ht):.0%}  seen {sc}/{st}={sc/max(1,st):.0%}  GAP {gap:.0f} pts", flush=True)
    return parsed/n, correct/tf, halluc/tf, gap

items = json.load(open("scribe_eval.json"))
m.eval()
print("\n=== SCALE-10M SCRIBE (greedy = primary) ===", flush=True)
pr, rec, hal, gap = score(m, items, "10m-scribe")
base = GPT().to(dev); base.load_state_dict(base_sd); base.eval()
print("=== BASE CONTROL (10m pretrain, pre-scribe) ===", flush=True)
bpr, brec, bhal, _ = score(base, items, "10m-base")

print("\n--- verdict on pre-registered Stage-S bars ---", flush=True)
ok = pr >= 0.90 and rec >= 0.80 and hal <= 0.10
bf = not (bpr >= 0.90 and brec >= 0.80 and bhal <= 0.10)
print(f"  scribe clears bars: {ok}   base control fails: {bf}", flush=True)
print(f"  held-out-value GAP: {gap:.0f} pts  (nano-v2 baseline 22; <10 = capacity CONFIRMED, >=15 = weakened)", flush=True)
print("STAGE S GATE " + ("PASS" if ok and bf else "FAIL"), flush=True)
