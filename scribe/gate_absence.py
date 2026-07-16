# Stage A — absence-verifier axis, single measurement run.
# Bars PRE-REGISTERED in AUDIT.md (committed before this script ran):
#   residual halluc <=2.5% | presented precision >=95% | review load <=25%
# Verifier = Stage G presence rules + absence checks:
#   MED/ALG "none" -> flag if any lexicon term appears in patient utterances
#   CC/DUR/SEV "none" -> flag unconditionally (mandatory fields)
import json, re, torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer

torch.manual_seed(0)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
V = 4098
d, L, H, KV, hd, ff, S = 192, 6, 6, 2, 32, 512, 512

tok = Tokenizer.from_file("tokenizer.json")
IMS, IME = tok.token_to_id("<|im_start|>"), tok.token_to_id("<|im_end|>")

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
        s.q, s.k, s.v, s.o = nn.Linear(d, H*hd, bias=False), nn.Linear(d, KV*hd, bias=False), nn.Linear(d, KV*hd, bias=False), nn.Linear(H*hd, d, bias=False)
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
    def __init__(s, vocab):
        super().__init__()
        s.emb = nn.Embedding(vocab, d); s.blocks = nn.ModuleList(Block() for _ in range(L)); s.nf = nn.RMSNorm(d)
    def forward(s, x):
        h = s.emb(x)
        for b in s.blocks: h = b(h)
        return F.linear(s.nf(h), s.emb.weight)

m = GPT(V)
m.load_state_dict(torch.load("scribe.pt", map_location="cpu", weights_only=True))
m.to(dev).eval()

def prompt_ids(user_content):
    ids = [IMS] + tok.encode("user\n", add_special_tokens=False).ids \
        + tok.encode(user_content, add_special_tokens=False).ids + [IME]
    ids += [IMS] + tok.encode("assistant\n", add_special_tokens=False).ids
    return ids

@torch.no_grad()
def generate(ids, max_new=64):
    ids = list(ids)
    for _ in range(max_new):
        if len(ids) >= S: break
        x = torch.tensor([ids + [0] * (S - len(ids))], device=dev)
        nxt = int(m(x)[0, len(ids) - 1].argmax())
        if nxt == IME: break
        ids.append(nxt)
    return ids

RE = re.compile(r"^CC: (.+?) \| DUR: (.+?) \| SEV: (.+?) \| MED: (.+?) \| ALG: (.+?)$")
FIELDS = ["cc", "dur", "sev", "med", "alg"]

# Deployed-system lexicons (analogue of RxNorm / allergen registry): the task's FULL
# value vocabularies, independent of what training saw (includes eval-held-out terms).
MED_LEX = ["ibuprofen", "paracetamol", "aspirin", "antacids", "cough syrup",
           "allergy pills", "naproxen", "vitamin c", "zinc tablets", "magnesium",
           "fish oil", "nasal spray", "eye drops", "hydrocortisone cream",
           "loratadine", "cetirizine", "famotidine", "saline rinse",
           "melatonin", "throat lozenges"]
ALG_LEX = ["penicillin", "peanuts", "pollen", "latex", "shellfish", "sulfa drugs"]
LEX = {"med": MED_LEX, "alg": ALG_LEX}

def patient_text(dialogue):
    return " ".join(l[len("Patient:"):].lower() for l in dialogue.split("\n") if l.startswith("Patient:"))

items = json.load(open("scribe_eval.json"))
total = presented = flagged = 0
pres_correct = pres_halluc = pres_omission = 0
halluc_total = halluc_caught = 0
omission_total = omission_caught = 0
false_absence_flags = 0
unparsed = 0

for it in items:
    dia = it["convo"][0]["content"].rsplit("\nSummarize the visit.", 1)[0]
    ptext = patient_text(dia)
    pids = prompt_ids(it["convo"][0]["content"])
    out = generate(pids)
    text = tok.decode(out[len(pids):]).strip()
    mm = RE.match(text)
    total += 5
    if not mm:
        unparsed += 1; flagged += 5
        continue
    pred = dict(zip(FIELDS, [g.strip() for g in mm.groups()]))
    for f in FIELDS:
        t, p = it["tuple"][f], pred[f]
        is_halluc = (p != t) and not (p == "none" and t != "none")
        is_omission = (p == "none" and t != "none")
        if is_halluc: halluc_total += 1
        if is_omission: omission_total += 1
        if p == "none":
            if f in LEX:
                grounded = not any(term in ptext for term in LEX[f])   # flag if lexicon term spoken
            else:
                grounded = False                                        # mandatory field: always flag
        else:
            grounded = p.lower() in ptext                               # Stage G presence rule
        if grounded:
            presented += 1
            if p == t: pres_correct += 1
            elif is_omission: pres_omission += 1
            else: pres_halluc += 1
        else:
            flagged += 1
            if is_halluc: halluc_caught += 1
            if is_omission: omission_caught += 1
            if p == "none" and t == "none": false_absence_flags += 1

print(f"eval fields {total} (unparsed drafts: {unparsed} -> fully flagged)", flush=True)
print(f"model hallucination (pre-guardrail): {halluc_total}/{total} = {halluc_total/total:.1%}", flush=True)
print(f"model omissions (pre-guardrail): {omission_total}/{total} = {omission_total/total:.1%}", flush=True)
print(f"presented {presented}  flagged {flagged}  review load {flagged/total:.1%}", flush=True)
print(f"residual hallucination (presented): {pres_halluc}/{total} = {pres_halluc/total:.1%}", flush=True)
print(f"residual omissions (presented): {pres_omission}/{total} = {pres_omission/total:.1%}", flush=True)
print(f"presented precision: {pres_correct}/{presented} = {pres_correct/max(1,presented):.1%}", flush=True)
print(f"hallucination catch-rate: {halluc_caught}/{halluc_total} = {halluc_caught/max(1,halluc_total):.0%}", flush=True)
print(f"omission catch-rate: {omission_caught}/{omission_total} = {omission_caught/max(1,omission_total):.0%}", flush=True)
print(f"false absence-flags (pred none, truth none, flagged): {false_absence_flags}", flush=True)
print(f"recall among all fields (presented & correct): {pres_correct}/{total} = {pres_correct/total:.1%}", flush=True)

print("\n--- verdict on pre-registered Stage-A bars ---", flush=True)
b1 = pres_halluc / total <= 0.025
b2 = pres_correct / max(1, presented) >= 0.95
b3 = flagged / total <= 0.25
print(f"  residual halluc <=2.5%: {b1}   presented precision >=95%: {b2}   review load <=25%: {b3}", flush=True)
print("STAGE A PASS" if (b1 and b2 and b3) else "STAGE A FAIL", flush=True)
