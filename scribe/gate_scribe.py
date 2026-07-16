# nano-scribe faithfulness gate — bars PRE-REGISTERED in AUDIT.md before training:
#   parse>=90%  recall>=80%  hallucination<=10%  AND base control (dpo.pt) must fail.
# Ground truth is the generating fact tuple, so faithfulness is exact — no judge.
# Primary decoding: GREEDY (deployment path for extraction; deviation from prior
# stages' sampled-primary is deliberate and logged). Sampled K=4 = diagnostic.
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

def load(path):
    m = GPT(V); m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    m.to(dev).eval(); return m

def prompt_ids(convo_user):
    ids = []
    head = tok.encode("user\n", add_special_tokens=False).ids
    body = tok.encode(convo_user, add_special_tokens=False).ids
    ids += [IMS] + head + body + [IME]
    ids += [IMS] + tok.encode("assistant\n", add_special_tokens=False).ids
    return ids

@torch.no_grad()
def generate(m, ids, max_new=64, temp=0.0):
    ids = list(ids)
    for _ in range(max_new):
        if len(ids) >= S: break
        x = torch.tensor([ids + [0] * (S - len(ids))], device=dev)
        logits = m(x)[0, len(ids) - 1]
        if temp == 0.0:
            nxt = int(logits.argmax())
        else:
            p = F.softmax(logits / temp, -1)
            sp, si = torch.sort(p, descending=True)
            c = torch.cumsum(sp, 0); k = int((c < 0.9).sum()) + 1     # top-p 0.9
            nxt = int(si[torch.multinomial(sp[:k] / sp[:k].sum(), 1)])
        if nxt == IME: break
        ids.append(nxt)
    return ids

RE = re.compile(r"^CC: (.+?) \| DUR: (.+?) \| SEV: (.+?) \| MED: (.+?) \| ALG: (.+?)$")
FIELDS = ["cc", "dur", "sev", "med", "alg"]

def score(m, items, temp=0.0, K=1, label=""):
    parsed = 0; correct = 0; omission = 0; halluc = 0; total_fields = 0
    held_correct = held_total = seen_correct = seen_total = 0
    samples = []
    for it in items:
        pids = prompt_ids(it["convo"][0]["content"])
        for _ in range(K):
            out = generate(m, pids, temp=temp)
            text = tok.decode(out[len(pids):]).strip()
            if len(samples) < 3: samples.append((text, it["tuple"]))
            mm = RE.match(text)
            total_fields += 5
            if not mm: continue
            parsed += 1
            pred = dict(zip(FIELDS, [g.strip() for g in mm.groups()]))
            for f in FIELDS:
                t, p = it["tuple"][f], pred[f]
                hit = (p == t)
                if hit: correct += 1
                elif p == "none" and t != "none": omission += 1
                else: halluc += 1                      # fabrication or substitution
                if it["held_values"]: held_total += 1; held_correct += hit
                else: seen_total += 1; seen_correct += hit
    n_out = len(items) * K
    pr = parsed / n_out
    rec = correct / total_fields
    hal = halluc / total_fields
    print(f"[{label}] parse {parsed}/{n_out}={pr:.0%}  recall {correct}/{total_fields}={rec:.0%}  "
          f"halluc {halluc}/{total_fields}={hal:.1%}  omission {omission}", flush=True)
    if held_total:
        print(f"          held-out-value recall {held_correct}/{held_total}={held_correct/held_total:.0%}  "
              f"seen-value recall {seen_correct}/{seen_total}={seen_correct/seen_total:.0%}", flush=True)
    for s, t in samples[:2]:
        print(f"          out: {s!r}\n          truth: {t}", flush=True)
    return pr, rec, hal

items = json.load(open("scribe_eval.json"))
print(f"eval: {len(items)} dialogues, held-out templates; {sum(i['held_values'] for i in items)} with held-out values\n", flush=True)

print("=== BASE CONTROL (dpo.pt, greedy) ===", flush=True)
base = load("../sft/dpo.pt")
bpr, brec, bhal = score(base, items, label="base/greedy")
del base

print("\n=== SCRIBE (scribe.pt, greedy = primary) ===", flush=True)
scribe = load("scribe.pt")
pr, rec, hal = score(scribe, items, label="scribe/greedy")

print("\n=== SCRIBE (sampled K=4 temp 0.7 = diagnostic) ===", flush=True)
score(scribe, items, temp=0.7, K=4, label="scribe/sampled")

print("\n--- verdict on pre-registered bars ---", flush=True)
scribe_pass = pr >= 0.90 and rec >= 0.80 and hal <= 0.10
base_fails = not (bpr >= 0.90 and brec >= 0.80 and bhal <= 0.10)
print(f"  scribe clears bars (parse>=90% recall>=80% halluc<=10%): {scribe_pass}", flush=True)
print(f"  base control fails bars (discrimination): {base_fails}", flush=True)
print("GATE PASS" if scribe_pass and base_fails else "GATE FAIL", flush=True)
