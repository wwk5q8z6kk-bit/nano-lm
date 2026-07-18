# Stage T-v2 — re-score the frozen nano/scale ANCHORS on the multi-instance
# instrument (PREREG_anchors.md). Re-SCORING only: the checkpoints are frozen v0.1
# release assets; only the eval set changes (inst0 single -> m0..m4 multi-instance).
#
# The model defs + ChatML/argmax scoring pipeline are the anchors' NATIVE own-stack
# pipeline, copied verbatim from scribe/gate_scribe.py (nano) and kaggle_scale_test.py
# (scale) — same rope/attention/decode, only the layer dims differ per model. Fidelity
# is proven in-band by the inst0 cross-check (nano gap ~22 vs gate_scribe_v2.log; scale
# gap ~23 vs Stage S). NOT the Pythia HF-generate path.
#
# Usage: python trajectory/rescore_anchors.py            (device from NANO_DEV, default mps)
import json, os, re, sys, hashlib
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer

torch.manual_seed(0)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKDIR = os.environ.get("NANO_CKDIR", os.path.join(REPO, "..", "..", "ckpt"))
DEV = os.environ.get("NANO_DEV", "mps" if torch.backends.mps.is_available() else "cpu")
V, S = 4098, 512
print(f"device={DEV}  ckdir={CKDIR}", flush=True)

tok = Tokenizer.from_file(os.path.join(REPO, "sft", "tokenizer.json"))
IMS, IME = tok.token_to_id("<|im_start|>"), tok.token_to_id("<|im_end|>")

# ---- model (verbatim math from gate_scribe.py / kaggle_scale_test.py; dims via cfg) ----
class Block(nn.Module):
    def __init__(s, d, H, KV, hd, ff):
        super().__init__()
        s.d, s.H, s.KV, s.hd = d, H, KV, hd
        s.n1, s.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        s.q = nn.Linear(d, H * hd, bias=False); s.k = nn.Linear(d, KV * hd, bias=False)
        s.v = nn.Linear(d, KV * hd, bias=False); s.o = nn.Linear(H * hd, d, bias=False)
        s.g = nn.Linear(d, ff, bias=False); s.u = nn.Linear(d, ff, bias=False); s.dn = nn.Linear(ff, d, bias=False)
    def forward(s, x, cos, sin):
        B = x.shape[0]; H, KV, hd = s.H, s.KV, s.hd; h = s.n1(x)
        q = s.q(h).view(B, S, H, hd).transpose(1, 2)
        k = s.k(h).view(B, S, KV, hd).transpose(1, 2)
        v = s.v(h).view(B, S, KV, hd).transpose(1, 2)
        def rot(t):
            t1, t2 = t[..., 0::2], t[..., 1::2]
            return torch.stack([t1 * cos - t2 * sin, t1 * sin + t2 * cos], dim=-1).flatten(-2)
        q, k = rot(q), rot(k)
        k, v = k.repeat_interleave(H // KV, 1), v.repeat_interleave(H // KV, 1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.o(a.transpose(1, 2).reshape(B, S, H * hd))
        h = s.n2(x)
        return x + s.dn(F.silu(s.g(h)) * s.u(h))

class GPT(nn.Module):
    def __init__(s, d, L, H, KV, hd, ff):
        super().__init__()
        s.hd = hd
        s.emb = nn.Embedding(V, d)
        s.blocks = nn.ModuleList(Block(d, H, KV, hd, ff) for _ in range(L))
        s.nf = nn.RMSNorm(d)
    def _rope(s, dev):
        t = torch.arange(S, device=dev, dtype=torch.float32)
        inv = 1.0 / (10000 ** (torch.arange(0, s.hd, 2, device=dev).float() / s.hd))
        f = torch.outer(t, inv)
        return f.cos()[None, None], f.sin()[None, None]
    def forward(s, x):
        cos, sin = s._rope(x.device)
        h = s.emb(x)
        for b in s.blocks: h = b(h, cos, sin)
        return F.linear(s.nf(h), s.emb.weight)

CFG = {
    "nano":  dict(file="scribe.pt",          d=192, L=6, H=6, KV=2, hd=32, ff=512),
    "scale": dict(file="scale10m_scribe.pt", d=320, L=8, H=8, KV=2, hd=40, ff=864),
}

def load(tag):
    c = CFG[tag]; path = os.path.join(CKDIR, c["file"])
    sd = torch.load(path, map_location="cpu", weights_only=True)
    m = GPT(c["d"], c["L"], c["H"], c["KV"], c["hd"], c["ff"])
    m.load_state_dict(sd)
    m.to(DEV).eval()
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    print(f"[{tag}] loaded {c['file']} params={sum(p.numel() for p in m.parameters())/1e6:.2f}M sha={sha[:12]}", flush=True)
    return m, sha

# ---- ChatML prompt + argmax greedy decode (verbatim from gate_scribe.py) ----
def prompt_ids(convo_user):
    ids = [IMS] + tok.encode("user\n", add_special_tokens=False).ids \
        + tok.encode(convo_user, add_special_tokens=False).ids + [IME]
    ids += [IMS] + tok.encode("assistant\n", add_special_tokens=False).ids
    return ids

@torch.no_grad()
def generate(m, ids, max_new=64):
    ids = list(ids)
    for _ in range(max_new):
        if len(ids) >= S: break
        x = torch.tensor([ids + [0] * (S - len(ids))], device=DEV)
        nxt = int(m(x)[0, len(ids) - 1].argmax())
        if nxt == IME: break
        ids.append(nxt)
    return ids

RE = re.compile(r"^CC: (.+?) \| DUR: (.+?) \| SEV: (.+?) \| MED: (.+?) \| ALG: (.+?)$")
FIELDS = ["cc", "dur", "sev", "med", "alg"]

@torch.no_grad()
def score(m, items, label=""):
    parsed = correct = halluc = omission = total_fields = 0
    hc = ht = sc = st = 0
    for it in items:
        out = generate(m, prompt_ids(it["convo"][0]["content"]))
        text = tok.decode(out[len(prompt_ids(it["convo"][0]["content"])):]).strip()
        total_fields += 5
        mm = RE.match(text)
        if not mm: continue
        parsed += 1
        pred = dict(zip(FIELDS, [g.strip() for g in mm.groups()]))
        for f in FIELDS:
            t, p = it["tuple"][f], pred[f]
            hit = (p == t)
            if hit: correct += 1
            elif p == "none" and t != "none": omission += 1
            else: halluc += 1
            if it["held_values"]: ht += 1; hc += hit
            else: st += 1; sc += hit
    n = len(items)
    held_rec = hc / max(1, ht); seen_rec = sc / max(1, st); gap = (seen_rec - held_rec) * 100
    r = {"n": n, "parse": parsed / n, "recall": correct / max(1, total_fields),
         "halluc": halluc / max(1, total_fields), "omission": omission,
         "held_correct": hc, "held_total": ht, "seen_correct": sc, "seen_total": st,
         "held_recall": held_rec, "seen_recall": seen_rec, "gap_pts": gap}
    print(f"[{label}] parse {parsed}/{n}={r['parse']:.0%} recall {r['recall']:.0%} "
          f"halluc {r['halluc']:.1%} | held {hc}/{ht}={held_rec:.0%} seen {sc}/{st}={seen_rec:.0%} "
          f"GAP {gap:.1f}", flush=True)
    return r

# ---- eval instances ----
inst0 = json.load(open(os.path.join(REPO, "scribe", "scribe_eval.json")))
fresh = [json.load(open(os.path.join(REPO, "trajectory", f"scribe_eval_m{k}.json"))) for k in range(5)]

CROSS = {"nano": {"log": "gate_scribe_v2.log", "gap": 22.0, "recall": 0.81},
         "scale": {"log": "scale/AUDIT.md Stage S", "gap": 23.0, "recall": 0.88}}

def run(tag):
    m, sha = load(tag)
    print(f"\n=== {tag} inst0 (v1, cross-check vs {CROSS[tag]['log']}: expect gap~{CROSS[tag]['gap']}, recall~{CROSS[tag]['recall']:.0%}) ===", flush=True)
    m0 = score(m, inst0, label=f"{tag}/inst0")
    print(f"\n=== {tag} multi-instance m0..m4 (v1 distribution, 100 held + 100 seen) ===", flush=True)
    mfresh = [score(m, fresh[k], label=f"{tag}/m{k}") for k in range(5)]
    gaps = [r["gap_pts"] for r in mfresh]
    gap_mean = float(np.mean(gaps)); gap_sd = float(np.std(gaps, ddof=1))
    inst0_gap = m0["gap_pts"]
    contam = inst0_gap < gap_mean - 2 * gap_sd
    print(f"\n  {tag} fresh gaps: {[round(g,1) for g in gaps]}", flush=True)
    print(f"  {tag} gap_mean {gap_mean:.2f} ± {gap_sd:.2f} SD  (2SD band [{gap_mean-2*gap_sd:.1f}, {gap_mean+2*gap_sd:.1f}])", flush=True)
    print(f"  {tag} inst0 gap {inst0_gap:.1f} vs fresh mean {gap_mean:.1f} — contamination(inst0 easier by>2SD): {contam}", flush=True)
    res = {"stage": "T-v2-anchors", "tag": tag, "checkpoint": CFG[tag]["file"], "sha256": sha,
           "prereg": "trajectory/PREREG_anchors.md", "device": DEV,
           "cross_check": {**CROSS[tag], "observed_inst0_gap": inst0_gap,
                           "observed_inst0_recall": m0["recall"], "observed_inst0_parse": m0["parse"]},
           "decoding": "greedy argmax, ChatML, stop on <|im_end|>, max_new 64",
           "instrument": "5x(100 held+100 seen), seeds 20260720-20260724; inst0 v1 40-dlg cross-check",
           "inst0": m0, "fresh_instances": mfresh, "fresh_gaps": gaps,
           "gap_mean": gap_mean, "gap_sd": gap_sd,
           "gap_2sd_band": [gap_mean - 2 * gap_sd, gap_mean + 2 * gap_sd],
           "contamination_flag": bool(contam),
           "versions": {"torch": torch.__version__, "numpy": np.__version__,
                        "python": sys.version.split()[0]}}
    out = os.path.join(REPO, "trajectory", f"results_anchors_v2_{tag}.json")
    json.dump(res, open(out, "w"), indent=1)
    print(f"  -> {out}", flush=True)
    return res

if __name__ == "__main__":
    for tag in ("nano", "scale"):
        run(tag)
