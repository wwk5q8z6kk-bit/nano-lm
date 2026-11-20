#!/usr/bin/env python3
"""E2 U3 — early-stop matched LoRA vs full-FT (PREREG_E2_lora_universes.md).

Cheapest falsifier of geometry talk: early-stop full-FT when diluted held-gap
plateaus on a probe instance; continue LoRA past that step. Compare gaps at the
matched early-stop step (and report LoRA-continued for context).

Frozen before first CUDA run:
  - Base: ownstack160m_pretrain.pt (same as kaggle_ownstack_160m{,_lora}.py)
  - Recipe: v2 scribe data, 3 epochs, LR 1e-4, micro8×accum4, S=512
  - Probe: diluted gap on m0 every PROBE_EVERY steps
  - Plateau: improvement < PLATEAU_EPS pts over last PLATEAU_WINDOW probes
  - Min steps before stop: max(MIN_FRAC * FSTEPS, PROBE_EVERY)
  - Final score: m0–m4 diluted + clean (mean±SD), same scorer as ownstack kernels
  - Decision (PREREG): SUPPORT U3 if matched early-stop erases LoRA advantage
    (delta_gap moves ≥5 pts toward zero while other cells N/A in this first cell);
    report UNRESOLVED if OOM / power fail.

Venue: CUDA ≥16GB (RTX 3090 24GB preferred cost/perf for 160M).
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

assert torch.cuda.is_available(), "CUDA required for E2 U3"
dev = "cuda"
SEED = int(os.environ.get("FT_SEED", "0"))
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

REPO = Path(os.environ.get("E2_REPO", "/workspace/nano-lm")).resolve()
sys.path.insert(0, str(REPO))

OUT = REPO / "checkpoints" / "e2_u3"
OUT.mkdir(parents=True, exist_ok=True)
TRAJ = REPO / "trajectory"
PRETRAIN = REPO / "checkpoints" / "chinchilla-160m" / "ownstack160m_pretrain.pt"
assert PRETRAIN.exists(), PRETRAIN

# ---- frozen hyperparams (match kaggle ownstack) ----
V = 4098
d, L, H, KV, hd, ff, S = 1024, 14, 16, 4, 64, 2752, 512
MICRO, ACCUM = 8, 4
BATCH = MICRO * ACCUM
FLR = 1e-4
PROBE_EVERY = 100
PLATEAU_WINDOW = 3
PLATEAU_EPS = 1.0  # pts
MIN_FRAC = 0.10
LORA_CFG = dict(r=16, alpha=32, dropout=0.0, targets=["q", "k", "v", "o", "g", "u", "dn"])

tok = Tokenizer.from_file(str(REPO / "sft" / "tokenizer.json"))
IMS = tok.token_to_id("<|im_start|>")
IME = tok.token_to_id("<|im_end|>")

RE_ROW = re.compile(r"^CC: (.+?) \| DUR: (.+?) \| SEV: (.+?) \| MED: (.+?) \| ALG: (.+?)$")
FIELDS = ["cc", "dur", "sev", "med", "alg"]
HELD = {
    "cc": {"toothache", "neck pain", "heartburn"},
    "med": {"melatonin", "throat lozenges"},
    "alg": {"sulfa drugs"},
}
VALFIELDS = ["cc", "med", "alg"]


def rope(q, k):
    t = torch.arange(S, device=dev, dtype=torch.float32)
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=dev).float() / hd))
    f = torch.outer(t, inv)
    cos, sin = f.cos()[None, None], f.sin()[None, None]

    def rot(x):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)

    return rot(q), rot(k)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.q = nn.Linear(d, H * hd, bias=False)
        self.k = nn.Linear(d, KV * hd, bias=False)
        self.v = nn.Linear(d, KV * hd, bias=False)
        self.o = nn.Linear(H * hd, d, bias=False)
        self.g = nn.Linear(d, ff, bias=False)
        self.u = nn.Linear(d, ff, bias=False)
        self.dn = nn.Linear(ff, d, bias=False)

    def forward(self, x):
        B = x.shape[0]
        h = self.n1(x)
        q = self.q(h).view(B, S, H, hd).transpose(1, 2)
        k = self.k(h).view(B, S, KV, hd).transpose(1, 2)
        v = self.v(h).view(B, S, KV, hd).transpose(1, 2)
        q, k = rope(q, k)
        k = k.repeat_interleave(H // KV, 1)
        v = v.repeat_interleave(H // KV, 1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, S, H * hd))
        h = self.n2(x)
        return x + self.dn(F.silu(self.g(h)) * self.u(h))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.blocks = nn.ModuleList(Block() for _ in range(L))
        self.nf = nn.RMSNorm(d)

    def forward(self, x):
        h = self.emb(x)
        for b in self.blocks:
            h = b(h)
        return F.linear(self.nf(h), self.emb.weight)


def load_base():
    m = GPT()
    st = torch.load(PRETRAIN, map_location="cpu", weights_only=True)
    m.load_state_dict(st["m"])
    m.to(dev)
    print(f"[u3] loaded pretrain step={st.get('step')} on {torch.cuda.get_device_name(0)}", flush=True)
    return m, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}


def build_data():
    v2_src = (REPO / "scribe" / "build_scribe_data_v2.py").read_text()
    prefix = v2_src.split("tok = Tokenizer.from_file")[0].replace(
        "from tokenizers import Tokenizer", ""
    )
    ns: dict = {}
    exec(compile(prefix, "v2[prefix]", "exec"), ns)
    convos = ns["convos"]
    assert len(convos) == 12000

    def render_ids(convo):
        ids, mask = [], []
        for msg in convo:
            head = tok.encode(f"{msg['role']}\n", add_special_tokens=False).ids
            body = tok.encode(msg["content"], add_special_tokens=False).ids
            seg = [IMS] + head + body + [IME]
            tf = 1 if msg["role"] == "assistant" else 0
            for j, tkn in enumerate(seg):
                ids.append(tkn)
                mask.append(1 if (tf and j >= 1 + len(head)) else 0)
        return ids, mask

    X, M = [], []
    for c in convos:
        ids, mask = render_ids(c)
        if len(ids) > S:
            continue
        X.append(ids + [0] * (S - len(ids)))
        M.append(mask + [0] * (S - len(mask)))
    X = torch.tensor(np.array(X, dtype=np.int64))
    M = torch.tensor(np.array(M, dtype=np.int64))
    N = X.shape[0]
    fsteps = (N // BATCH) * 3
    print(f"[u3] train rows={N} FSTEPS={fsteps}", flush=True)
    return X, M, fsteps


def prompt_ids(user_content):
    return (
        [IMS]
        + tok.encode("user\n", add_special_tokens=False).ids
        + tok.encode(user_content, add_special_tokens=False).ids
        + [IME]
        + [IMS]
        + tok.encode("assistant\n", add_special_tokens=False).ids
    )


@torch.no_grad()
def generate(model, ids, max_new=64):
    ids = list(ids)
    for _ in range(max_new):
        if len(ids) >= S:
            break
        x = torch.tensor([ids + [0] * (S - len(ids))], device=dev)
        nxt = int(model(x)[0, len(ids) - 1].argmax())
        if nxt == IME:
            break
        ids.append(nxt)
    return ids


@torch.no_grad()
def score(model, items):
    parsed = 0
    hc = ht = sc = st_ = 0
    cln = {f: [0, 0, 0, 0] for f in VALFIELDS}
    for it in items:
        p = prompt_ids(it["convo"][0]["content"])
        text = tok.decode(generate(model, p)[len(p) :]).strip()
        mm = RE_ROW.match(text)
        if not mm:
            continue
        parsed += 1
        pred = dict(zip(FIELDS, [g.strip() for g in mm.groups()]))
        for f in FIELDS:
            hit = int(pred[f] == it["tuple"][f])
            if it["held_values"]:
                ht += 1
                hc += hit
            else:
                st_ += 1
                sc += hit
        for f in VALFIELDS:
            t = it["tuple"][f]
            if t == "none":
                continue
            h = int(pred[f] == t)
            if t in HELD[f]:
                cln[f][0] += h
                cln[f][1] += 1
            else:
                cln[f][2] += h
                cln[f][3] += 1
    diluted = (sc / max(1, st_) - hc / max(1, ht)) * 100
    ch = sum(cln[f][0] for f in VALFIELDS)
    cht = sum(cln[f][1] for f in VALFIELDS)
    cs = sum(cln[f][2] for f in VALFIELDS)
    cst = sum(cln[f][3] for f in VALFIELDS)
    clean = (cs / max(1, cst) - ch / max(1, cht)) * 100
    return {
        "parse": parsed / len(items),
        "diluted_gap": diluted,
        "clean_gap": clean,
        "held_recall": hc / max(1, ht),
        "seen_recall": sc / max(1, st_),
    }


def score_fresh(model, fresh):
    fr = [score(model, fresh[k]) for k in range(5)]
    dg = [r["diluted_gap"] for r in fr]
    cg = [r["clean_gap"] for r in fr]
    return {
        "fresh_instances": fr,
        "diluted_fresh": dg,
        "clean_fresh": cg,
        "diluted_gap_mean": float(np.mean(dg)),
        "diluted_gap_sd": float(np.std(dg, ddof=1)),
        "clean_gap_mean": float(np.mean(cg)),
        "clean_gap_sd": float(np.std(cg, ddof=1)),
    }


def flr_at(t, fsteps):
    fwarm = int(0.03 * fsteps)
    if t < fwarm:
        return FLR * t / max(1, fwarm)
    p = (t - fwarm) / max(1, fsteps - fwarm)
    return FLR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def train_arm(kind: str, base_sd: dict, X, M, fsteps, probe_items, stop_at: int | None):
    """Train until stop_at (inclusive) or early-stop plateau (fullft only when stop_at is None).

    Returns (state_dict_cpu, meta).
    """
    from peft import LoraConfig, inject_adapter_in_model

    m = GPT()
    m.load_state_dict(base_sd)
    m.to(dev)
    if kind == "lora":
        cfg = LoraConfig(
            r=LORA_CFG["r"],
            lora_alpha=LORA_CFG["alpha"],
            lora_dropout=LORA_CFG["dropout"],
            target_modules=LORA_CFG["targets"],
        )
        m = inject_adapter_in_model(cfg, m)
        for n, p in m.named_parameters():
            p.requires_grad = "lora_" in n
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"[u3/{kind}] LoRA trainables {trainable/1e6:.3f}M", flush=True)
        opt = torch.optim.AdamW(
            (p for p in m.parameters() if p.requires_grad),
            lr=FLR,
            betas=(0.9, 0.95),
            weight_decay=0.0,
        )
    else:
        decay, nodecay = [], []
        for n, p in m.named_parameters():
            if not p.requires_grad:
                continue
            (nodecay if p.ndim < 2 else decay).append(p)
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
            lr=FLR,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        print(f"[u3/{kind}] full-FT params {sum(p.numel() for p in m.parameters())/1e6:.2f}M", flush=True)

    scaler = torch.amp.GradScaler("cuda")
    N = X.shape[0]
    perm = torch.randperm(N)
    probes = []
    early_step = None
    early_sd = None
    min_steps = max(int(MIN_FRAC * fsteps), PROBE_EVERY)
    limit = stop_at if stop_at is not None else fsteps
    snapshot_at = int(os.environ.get("U3_SNAPSHOT_AT", "0")) or None
    t0 = time.time()
    m.train()
    for step in range(1, limit + 1):
        for g in opt.param_groups:
            g["lr"] = flr_at(step, fsteps)
        i0 = (step * BATCH) % (N - BATCH)
        idx = perm[i0 : i0 + BATCH]
        opt.zero_grad(set_to_none=True)
        for a in range(ACCUM):
            sub = idx[a * MICRO : (a + 1) * MICRO]
            x, msk = X[sub].to(dev), M[sub].to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = m(x)[:, :-1]
                tgt = x[:, 1:]
                mtgt = msk[:, 1:]
                ce = F.cross_entropy(
                    logits.reshape(-1, V), tgt.reshape(-1), reduction="none"
                ).reshape(tgt.shape)
                loss = (ce * mtgt).sum() / mtgt.sum().clamp(min=1)
            scaler.scale(loss / ACCUM).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(
            (p for p in m.parameters() if p.requires_grad), 1.0
        )
        scaler.step(opt)
        scaler.update()
        if step % 200 == 0 or step == 1:
            print(
                f"[u3/{kind}] {step}/{limit} loss={loss.item():.3f} "
                f"({(time.time()-t0)/60:.1f} min)",
                flush=True,
            )
        if step % PROBE_EVERY == 0 or step == limit:
            m.eval()
            pr = score(m, probe_items)
            probes.append({"step": step, **pr})
            print(
                f"[u3/{kind}] probe step={step} diluted={pr['diluted_gap']:.2f} "
                f"parse={pr['parse']:.2f}",
                flush=True,
            )
            m.train()
            if (
                kind == "fullft"
                and stop_at is None
                and early_step is None
                and step >= min_steps
                and len(probes) >= PLATEAU_WINDOW
            ):
                window = probes[-PLATEAU_WINDOW:]
                gaps = [p["diluted_gap"] for p in window]
                # plateau = held-gap not improving (diluted gap not falling) by ≥EPS
                best_prev = min(gaps[:-1])
                if gaps[-1] > best_prev - PLATEAU_EPS:
                    early_step = step
                    early_sd = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
                    print(
                        f"[u3/{kind}] EARLY-STOP plateau at step={early_step} "
                        f"gap={gaps[-1]:.2f} window={gaps}",
                        flush=True,
                    )
                    # U3: stop full-FT at plateau (do not continue wasting budget)
                    break
        if snapshot_at is not None and step == snapshot_at and early_sd is None:
            early_step = step
            early_sd = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            print(f"[u3/{kind}] snapshot at step={step}", flush=True)
        if stop_at is not None and step == stop_at and early_sd is None:
            early_step = step
            early_sd = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}

    final_sd = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    if early_sd is None:
        early_step = limit
        early_sd = final_sd
    meta = {
        "kind": kind,
        "probes": probes,
        "early_step": early_step,
        "final_step": limit,
        "train_secs": round(time.time() - t0),
        "gpu": torch.cuda.get_device_name(0),
    }
    return early_sd, final_sd, meta


def load_scored(sd, kind):
    from peft import LoraConfig, inject_adapter_in_model

    m = GPT()
    if kind == "lora":
        cfg = LoraConfig(
            r=LORA_CFG["r"],
            lora_alpha=LORA_CFG["alpha"],
            lora_dropout=LORA_CFG["dropout"],
            target_modules=LORA_CFG["targets"],
        )
        m = inject_adapter_in_model(cfg, m)
    m.load_state_dict(sd)
    m.to(dev).eval()
    return m


def decide(matched_fullft, matched_lora, continued_lora, behavioral_ref):
    """Apply PREREG U3 rule for this single cell (others N/A)."""
    ff = matched_fullft["diluted_gap_mean"]
    lo = matched_lora["diluted_gap_mean"]
    delta_matched = ff - lo  # LoRA advantage = positive if LoRA lower gap
    ref_delta = behavioral_ref["fullft"] - behavioral_ref["lora"]
    # Erasure: matched early-stop shrinks LoRA advantage by ≥5 pts vs behavioral ref
    erased = (ref_delta - abs(delta_matched)) >= 5.0 or abs(delta_matched) < 2.0
    cont = continued_lora["diluted_gap_mean"]
    verdict = "SUPPORT_U3" if erased else "U3_NOT_SUPPORTED"
    return {
        "verdict": verdict,
        "matched_delta_gap": delta_matched,
        "behavioral_ref_delta": ref_delta,
        "erasure_criterion": "matched |fullFT-LoRA| <2 OR ref_delta - |matched| ≥5",
        "erased": erased,
        "note": (
            "Single-cell U3; U1/U2/U4 not run. "
            "SUPPORT_U3 means early-stop matching erases LoRA advantage → "
            "geometry language not uniquely required."
        ),
        "continued_lora_gap": cont,
        "matched_fullft_gap": ff,
        "matched_lora_gap": lo,
    }


def main():
    os.system("pip uninstall -y -q torchao 2>/dev/null")
    print("=== E2 U3 early-stop matched pair ===", flush=True)
    print(f"venue gpu={torch.cuda.get_device_name(0)} mem={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}G", flush=True)

    _, base_sd = load_base()
    X, M, fsteps = build_data()
    fresh = [json.loads((TRAJ / f"scribe_eval_m{k}.json").read_text()) for k in range(5)]
    probe_items = fresh[0]
    behavioral_ref = {"fullft": 16.88, "lora": 7.08}  # prior measured means

    # Arm A: full-FT — stop at held-gap plateau
    print("=== ARM full-FT (early-stop on plateau) ===", flush=True)
    ff_early_sd, ff_final_sd, ff_meta = train_arm(
        "fullft", base_sd, X, M, fsteps, probe_items, stop_at=None
    )
    stop_step = ff_meta["early_step"]
    torch.save({"m": ff_early_sd, "meta": ff_meta}, OUT / "fullft_early.pt")
    # final == early when we break at plateau
    torch.save({"m": ff_final_sd, "meta": ff_meta}, OUT / "fullft_at_stop.pt")

    # Arm B: single LoRA run — snapshot at matched early-stop, continue to full FSTEPS
    print(f"=== ARM LoRA (snapshot@{stop_step}, continue to {fsteps}) ===", flush=True)
    os.environ["U3_SNAPSHOT_AT"] = str(stop_step)
    lo_early_sd, lo_final_sd, lo_meta = train_arm(
        "lora", base_sd, X, M, fsteps, probe_items, stop_at=fsteps
    )
    lo_meta_early = {**lo_meta, "snapshot_step": stop_step}
    lo_meta_full = lo_meta
    torch.save({"m": lo_early_sd, "meta": lo_meta_early}, OUT / "lora_early.pt")
    torch.save({"m": lo_final_sd, "meta": lo_meta_full}, OUT / "lora_final.pt")

    # Final multi-instance scores
    print("=== scoring m0-m4 ===", flush=True)
    m_ff = load_scored(ff_early_sd, "fullft")
    matched_fullft = score_fresh(m_ff, fresh)
    del m_ff
    torch.cuda.empty_cache()

    m_lo = load_scored(lo_early_sd, "lora")
    matched_lora = score_fresh(m_lo, fresh)
    del m_lo
    torch.cuda.empty_cache()

    m_lc = load_scored(lo_final_sd, "lora")
    continued_lora = score_fresh(m_lc, fresh)
    del m_lc

    decision = decide(matched_fullft, matched_lora, continued_lora, behavioral_ref)
    results = {
        "stage": "e2-u3-earlystop",
        "prereg": "trajectory/PREREG_E2_lora_universes.md",
        "venue": "runpod-cuda",
        "gpu": torch.cuda.get_device_name(0),
        "ft_seed": SEED,
        "fsteps": fsteps,
        "probe_every": PROBE_EVERY,
        "plateau": {"window": PLATEAU_WINDOW, "eps_pts": PLATEAU_EPS, "min_frac": MIN_FRAC},
        "early_stop_step": stop_step,
        "fullft_meta": ff_meta,
        "lora_early_meta": lo_meta_early,
        "lora_full_meta": lo_meta_full,
        "matched_fullft": matched_fullft,
        "matched_lora": matched_lora,
        "continued_lora": continued_lora,
        "behavioral_ref_gaps": behavioral_ref,
        "decision": decision,
    }
    out_path = TRAJ / "results_e2_u3_earlystop.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(decision, indent=2), flush=True)
    print(f"wrote {out_path}", flush=True)
    print("E2_U3_DONE", flush=True)


if __name__ == "__main__":
    main()
