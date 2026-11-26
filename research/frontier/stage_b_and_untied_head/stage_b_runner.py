"""Stage B (`RESEARCH-PLAN §20.5a`) Runner: `SmolLM2-135M -> 22 Gated DeltaNet + 8 GQA` Inheritance & Alignment Gate.

Executes all pre-registered Stage B arms (`P`, `A_all30`, `T` x3 seeds, `C1` x3 seeds, `C2` x1 seed, `R` x1 seed)
at an exact `250,347,520`-token student budget across all available Hopper/Ampere GPUs in parallel (`torch.cuda.device_count()`),
recording cryptographic SHA256 hashes of code, pinned teacher revision, and decontaminated corpus shards, plus
7-task zero-shot evaluation (`hellaswag, arc_easy, arc_challenge, piqa, winogrande, openbookqa, lambada`).

Supports `STAGEB_SMOKE=1` on CPU (`NaiveGDN`, synthetic shards/tasks) for deterministic pre-launch contract verification.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

SMOKE = os.environ.get("STAGEB_SMOKE") == "1"
OUT_DIR = Path(os.environ.get("STAGEB_OUT", "/workspace/out_stage_b"))
DATA_DIR = Path(os.environ.get("STAGEB_DATA", "/workspace/data_stage_b"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

PROBE_CODE = """
import torch, torch.nn.functional as F
import importlib.metadata as md
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
q = F.normalize(torch.randn(1, 64, 2, 64, device='cuda', dtype=torch.bfloat16), dim=-1).requires_grad_(True)
k = F.normalize(torch.randn(1, 64, 2, 64, device='cuda', dtype=torch.bfloat16), dim=-1).requires_grad_(True)
v = torch.randn(1, 64, 2, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
g = F.logsigmoid(torch.randn(1, 64, 2, device='cuda') + 2.0).requires_grad_(True)
b = torch.rand(1, 64, 2, device='cuda', dtype=torch.bfloat16, requires_grad=True)
o = chunk_gated_delta_rule(q, k, v, g, b, scale=0.125)
(o[0] if isinstance(o, tuple) else o).float().sum().backward()
torch.cuda.synchronize()
print("GDN_PROBE_OK triton=" + md.version("triton"))
"""

if not SMOKE and os.environ.get("STAGEB_INSTALL", "1") == "1":
    t_inst = time.time()
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "numpy", "pyarrow", "datasets", "huggingface_hub", "tokenizers", "transformers", "einops", "triton==3.7.1"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "flash-linear-attention==0.6.0", "fla-core==0.6.0"],
        check=True,
    )
    p = subprocess.run([sys.executable, "-c", PROBE_CODE], capture_output=True, text=True, timeout=180)
    if "GDN_PROBE_OK" not in p.stdout:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "triton==3.8.0"], check=False)
        p2 = subprocess.run([sys.executable, "-c", PROBE_CODE], capture_output=True, text=True, timeout=180)
        if "GDN_PROBE_OK" not in p2.stdout:
            raise RuntimeError(f"GDN kernel probe failed: {p.stdout} {p.stderr} / {p2.stdout} {p2.stderr}")
    print(json.dumps({"event": "install_ok", "seconds": round(time.time() - t_inst, 1), "probe": p.stdout.strip()}), flush=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TEACHER_MODEL = "HuggingFaceTB/SmolLM2-135M"
TEACHER_REV = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
KEEP_ATTN = (1, 2, 3, 4, 21, 22, 23, 24)
SOURCES = {
    "web": ("HuggingFaceFW/fineweb-edu", "sample/10BT/", "text", "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"),
    "textbook": ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2/", "text", "3ba9d605774198c5868892d7a8deda78031a781f"),
    "math": ("HuggingFaceTB/finemath", "finemath-4plus/", "text", "e92b25a616738fe95dc186b64dfb19f9c8525594"),
    "code": ("codeparrot/github-code-clean", "data/", "code", "c48d40f9e70f0196f8236901ee35807f7d6c44c0"),
}
MIX = {"web": 0.62, "textbook": 0.13, "math": 0.13, "code": 0.12}
CHANCE = {
    "hellaswag": 0.25,
    "arc_easy": 0.25,
    "arc_challenge": 0.25,
    "piqa": 0.50,
    "winogrande": 0.50,
    "openbookqa": 0.25,
    "lambada": 0.0,
}


def sha256_file(p: Path, max_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    rem = max_bytes
    with p.open("rb") as f:
        while True:
            n = (1 << 20) if rem is None else min(1 << 20, rem)
            if n <= 0:
                break
            b = f.read(n)
            if not b:
                break
            h.update(b)
            if rem is not None:
                rem -= len(b)
    return h.hexdigest()


def ref_gdn(q, k, v, g, beta, scale):
    B, T_, H, K = q.shape
    S = q.new_zeros(B, H, K, v.shape[-1])
    outs = []
    for t in range(T_):
        S = S * g[:, t].exp()[..., None, None]
        kv = torch.einsum("bhk,bhkv->bhv", k[:, t], S)
        S = S + torch.einsum("bhk,bhv->bhkv", k[:, t], (v[:, t] - kv) * beta[:, t][..., None])
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, t] * scale, S))
    return torch.stack(outs, 1)


class NaiveGDN(nn.Module):
    """CPU-only exact causal recurrence matching GatedDeltaNet projection names for SMOKE tests."""
    def __init__(s, hidden_size: int, num_heads: int, head_dim: int, **kw):
        super().__init__()
        s.h, s.d = num_heads, head_dim
        kd = num_heads * head_dim
        s.q_proj = nn.Linear(hidden_size, kd, bias=False)
        s.k_proj = nn.Linear(hidden_size, kd, bias=False)
        s.v_proj = nn.Linear(hidden_size, kd, bias=False)
        s.a_proj = nn.Linear(hidden_size, num_heads, bias=False)
        s.b_proj = nn.Linear(hidden_size, num_heads, bias=False)
        s.o_proj = nn.Linear(kd, hidden_size, bias=False)

    def forward(s, x, **kw):
        B, T_, _ = x.shape
        sh = lambda z: z.view(B, T_, s.h, s.d)
        q = F.normalize(sh(s.q_proj(x)), dim=-1)
        k = F.normalize(sh(s.k_proj(x)), dim=-1)
        v = sh(s.v_proj(x))
        g = F.logsigmoid(s.a_proj(x) + 3.0)
        beta = torch.sigmoid(s.b_proj(x))
        return s.o_proj(ref_gdn(q, k, v, g, beta, s.d ** -0.5).reshape(B, T_, -1)), None, None


def make_gdn(hidden: int, heads: int, head_dim: int, device: str):
    if SMOKE:
        return NaiveGDN(hidden, heads, head_dim).to(device)
    from fla.layers import GatedDeltaNet
    return GatedDeltaNet(hidden_size=hidden, num_heads=heads, head_dim=head_dim, expand_v=1, mode="chunk").to(device)


class GDNAttnWrapper(nn.Module):
    """Drop-in replacement for LlamaAttention inside HuggingFace LlamaDecoderLayer across transformers versions."""
    def __init__(self, gdn: nn.Module, n_ret: int = 2):
        super().__init__()
        self.gdn = gdn
        self.n_ret = max(1, n_ret)

    def forward(self, hidden_states=None, *args, **kwargs):
        out = self.gdn(hidden_states)
        y = out[0] if isinstance(out, tuple) else out
        if self.n_ret == 1:
            return y
        return (y,) + (None,) * (self.n_ret - 1)


PRINT_LOCK = threading.Lock()


def safe_print(obj_or_str):
    msg = obj_or_str if isinstance(obj_or_str, str) else json.dumps(obj_or_str)
    with PRINT_LOCK:
        print(msg, flush=True)


def emit_job_receipt(job_name: str, rec: dict):
    pretty = json.dumps(rec, indent=2)
    with PRINT_LOCK:
        print(f"STAGEB_REC_BEGIN {job_name}")
        print(pretty)
        print(f"STAGEB_REC_END {job_name}", flush=True)


def detect_attn_return_arity(teacher_cpu: nn.Module) -> int:
    arity = {"n": 2}
    def _hk(mod, args, kwargs, out):
        arity["n"] = len(out) if isinstance(out, tuple) else 1
    h = teacher_cpu.model.layers[0].self_attn.register_forward_hook(_hk, with_kwargs=True)
    try:
        with torch.no_grad():
            _ = teacher_cpu(input_ids=torch.zeros((1, 4), dtype=torch.long), use_cache=False)
    finally:
        h.remove()
    return arity["n"]


def inherit_projections(gdn: nn.Module, attn: nn.Module, heads: int, kv_heads: int, head_dim: int):
    rep = heads // kv_heads
    with torch.no_grad():
        gdn.q_proj.weight.copy_(attn.q_proj.weight)
        for name in ("k_proj", "v_proj"):
            w = getattr(attn, name).weight.view(kv_heads, head_dim, -1).repeat_interleave(rep, dim=0).reshape(heads * head_dim, -1)
            getattr(gdn, name).weight.copy_(w)
        gdn.o_proj.weight.copy_(attn.o_proj.weight)


def ngrams13(text: str) -> set[int]:
    w = text.split()
    return {hash(" ".join(w[i : i + 13])) for i in range(0, max(0, len(w) - 12), 5)}


def prepare_corpus_and_tasks(tok) -> dict:
    """Builds or loads decontaminated SmolLM2-tokenized train/val shards and 7-task zero-shot prompts."""
    train_p = DATA_DIR / "train_smollm.bin"
    meta_p = DATA_DIR / "corpus_meta.json"
    tasks_p = DATA_DIR / "tasks_text.json"

    if SMOKE:
        rng = np.random.default_rng(42)
        vocab = tok.vocab_size
        if not train_p.exists():
            train_p.write_bytes(rng.integers(0, vocab, 16384, dtype=np.uint16).tobytes())
        for d in MIX:
            vp = DATA_DIR / f"val_{d}.bin"
            if not vp.exists():
                vp.write_bytes(rng.integers(0, vocab, 2048, dtype=np.uint16).tobytes())
        if not tasks_p.exists():
            tasks_p.write_text(json.dumps({
                "arc_easy": [{"ctx": "Question: 1+1?\nAnswer:", "conts": [" 1", " 2", " 3", " 4"], "gold": 1}],
                "piqa": [{"ctx": "Question: Boil water\nAnswer:", "conts": [" heat kettle", " freeze pan"], "gold": 0}],
                "winogrande": [{"ctxs": ["The trophy fit the suitcase because the trophy", "The trophy fit the suitcase because the suitcase"], "cont": " was small.", "gold": 0}],
                "lambada": [{"ctx": "She opened the door and walked into the", "conts": [" room"], "gold": 0}],
            }))
        meta = {
            "train_tokens": 16384,
            "train_sha256": sha256_file(train_p),
            "val_sha256": {d: sha256_file(DATA_DIR / f"val_{d}.bin") for d in MIX},
            "tasks_sha256": sha256_file(tasks_p),
        }
        meta_p.write_text(json.dumps(meta, indent=2))
        return meta

    if meta_p.exists() and train_p.exists() and tasks_p.exists():
        return json.loads(meta_p.read_text())

    import pyarrow.parquet as pq
    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download

    t0 = time.time()
    api = HfApi()
    cache_hf = DATA_DIR / "hf_cache"
    cache_hf.mkdir(parents=True, exist_ok=True)
    eos_id = int(tok.eos_token_id)
    val_tokens_per_dom = 64 * 1024  # 65,536 tokens/domain = 262,144 held-out tokens
    total_train_target = 270_000_000  # > 250.35M required

    val_ngrams: set[int] = set()
    domain_train_chunks: dict[str, list[np.ndarray]] = {d: [] for d in MIX}

    for dom, (repo, prefix, col, rev) in SOURCES.items():
        all_files = sorted(f for f in api.list_repo_files(repo, repo_type="dataset", revision=rev) if f.startswith(prefix) and f.endswith(".parquet"))
        assert len(all_files) >= 2, f"{dom}: expected >=2 parquet shards, got {len(all_files)}"

        # Shard 0 -> strictly held-out validation ONLY
        f0 = hf_hub_download(repo, all_files[0], repo_type="dataset", revision=rev, cache_dir=str(cache_hf))
        tbl0 = pq.read_table(f0, columns=[col])
        val_ids: list[int] = []
        batch_texts: list[str] = []
        for s in tbl0[col].to_pylist():
            if not isinstance(s, str) or not (200 <= len(s) <= 100_000):
                continue
            val_ngrams.update(ngrams13(s))
            batch_texts.append(s)
            if len(batch_texts) >= 256:
                for enc in tok(batch_texts, add_special_tokens=False)["input_ids"]:
                    val_ids.extend(enc)
                    val_ids.append(eos_id)
                batch_texts.clear()
                if len(val_ids) >= val_tokens_per_dom + 1024:
                    break
        if batch_texts and len(val_ids) < val_tokens_per_dom + 1024:
            for enc in tok(batch_texts, add_special_tokens=False)["input_ids"]:
                val_ids.extend(enc)
                val_ids.append(eos_id)
        val_arr = np.asarray(val_ids[: val_tokens_per_dom + 1024], dtype=np.uint16)
        (DATA_DIR / f"val_{dom}.bin").write_bytes(val_arr.tobytes())
        print(json.dumps({"event": "val_built", "domain": dom, "tokens": int(len(val_arr))}), flush=True)

    # Shards 1..K -> decontaminated training tokens per domain
    for dom, (repo, prefix, col, rev) in SOURCES.items():
        all_files = sorted(f for f in api.list_repo_files(repo, repo_type="dataset", revision=rev) if f.startswith(prefix) and f.endswith(".parquet"))
        dom_quota = int(total_train_target * MIX[dom]) + 2_000_000
        collected = 0
        for shard_name in all_files[1:]:
            if collected >= dom_quota:
                break
            fp = hf_hub_download(repo, shard_name, repo_type="dataset", revision=rev, cache_dir=str(cache_hf))
            pf = pq.ParquetFile(fp)
            for rg in range(pf.num_row_groups):
                if collected >= dom_quota:
                    break
                texts = pf.read_row_group(rg, columns=[col])[col].to_pylist()
                clean_texts = [
                    s for s in texts
                    if isinstance(s, str) and (200 <= len(s) <= 100_000) and not (ngrams13(s) & val_ngrams)
                ]
                for i in range(0, len(clean_texts), 1024):
                    sub = clean_texts[i : i + 1024]
                    if not sub:
                        continue
                    encs = tok(sub, add_special_tokens=False)["input_ids"]
                    flat: list[int] = []
                    for e in encs:
                        flat.extend(e)
                        flat.append(eos_id)
                    arr = np.asarray(flat, dtype=np.uint16)
                    domain_train_chunks[dom].append(arr)
                    collected += len(arr)
                    if collected >= dom_quota:
                        break
        print(json.dumps({"event": "train_domain_built", "domain": dom, "tokens": collected}), flush=True)

    # Interleave in 64k blocks matching MIX shares exactly (as in build_corpus_v2.py)
    dom_concat = {d: np.concatenate(domain_train_chunks[d]) for d in MIX}
    del domain_train_chunks
    block = 65536
    block_counts = {d: max(1, round(block * w)) for d, w in MIX.items()}
    cursors = {d: 0 for d in MIX}
    n_blocks = total_train_target // sum(block_counts.values()) + 1
    out_mm = np.memmap(train_p, dtype=np.uint16, mode="w+", shape=(n_blocks * sum(block_counts.values()),))
    pos = 0
    for _ in range(n_blocks):
        for d in ("web", "textbook", "math", "code"):
            cnt = block_counts[d]
            src = dom_concat[d]
            c = cursors[d]
            if c + cnt > len(src):
                c = 0
            out_mm[pos : pos + cnt] = src[c : c + cnt]
            cursors[d] = c + cnt
            pos += cnt
    out_mm.flush()
    del out_mm, dom_concat

    # Build 7-task zero-shot formatted texts (identical to eval_prep.py)
    def hs_pre(t: str) -> str:
        t = t.strip().replace(" [title]", ". ")
        t = re.sub(r"\[.*?\]", "", t)
        return t.replace("  ", " ")

    tasks_dict = {}
    d_hs = load_dataset("Rowan/hellaswag", split="validation")
    tasks_dict["hellaswag"] = [{"ctx": hs_pre(x["activity_label"] + ": " + x["ctx_a"] + " " + x["ctx_b"].capitalize()), "conts": [" " + hs_pre(e) for e in x["endings"]], "gold": int(x["label"])} for x in d_hs]
    for name, cfg_n in (("arc_easy", "ARC-Easy"), ("arc_challenge", "ARC-Challenge")):
        d_arc = load_dataset("allenai/ai2_arc", cfg_n, split="test")
        tasks_dict[name] = [{"ctx": "Question: " + x["question"] + "\nAnswer:", "conts": [" " + c for c in x["choices"]["text"]], "gold": x["choices"]["label"].index(x["answerKey"])} for x in d_arc]
    d_piqa = load_dataset("ybisk/piqa", split="validation", revision="refs/convert/parquet")
    tasks_dict["piqa"] = [{"ctx": "Question: " + x["goal"] + "\nAnswer:", "conts": [" " + x["sol1"], " " + x["sol2"]], "gold": int(x["label"])} for x in d_piqa]
    d_wg = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    wg_rows = []
    for x in d_wg:
        idx = x["sentence"].index("_")
        suf = " " + x["sentence"][idx + 1 :].strip()
        wg_rows.append({"ctxs": [x["sentence"][:idx] + x["option1"], x["sentence"][:idx] + x["option2"]], "cont": suf, "gold": int(x["answer"]) - 1})
    tasks_dict["winogrande"] = wg_rows
    d_obqa = load_dataset("allenai/openbookqa", "main", split="test")
    tasks_dict["openbookqa"] = [{"ctx": x["question_stem"], "conts": [" " + c for c in x["choices"]["text"]], "gold": x["choices"]["label"].index(x["answerKey"].strip())} for x in d_obqa]
    d_lam = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    tasks_dict["lambada"] = [{"ctx": x["text"].rsplit(" ", 1)[0], "conts": [" " + x["text"].rsplit(" ", 1)[1]], "gold": 0} for x in d_lam]
    tasks_p.write_text(json.dumps(tasks_dict))

    meta = {
        "build_seconds": round(time.time() - t0, 1),
        "train_tokens": int(train_p.stat().st_size // 2),
        "train_head64mb_sha256": sha256_file(train_p, max_bytes=64 * 1024 * 1024),
        "val_sha256": {d: sha256_file(DATA_DIR / f"val_{d}.bin") for d in MIX},
        "tasks_sha256": sha256_file(tasks_p),
        "sources_pinned": SOURCES,
    }
    meta_p.write_text(json.dumps(meta, indent=2))
    return meta


@torch.no_grad()
def eval_heldout_nats(model: nn.Module, val_batches: dict[str, torch.Tensor], device: str, amp_dtype) -> dict:
    model.eval()
    out = {}
    use_amp = (device != "cpu")
    for dom, seqs in val_batches.items():
        tot_nll = 0.0
        tot_tok = 0
        for i in range(0, seqs.shape[0], 8):
            ids = seqs[i : i + 8].to(device)
            with torch.autocast(device_type="cuda" if use_amp else "cpu", dtype=amp_dtype, enabled=use_amp):
                lg = model(input_ids=ids[:, :-1], use_cache=False).logits
            nll = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1), reduction="sum")
            tot_nll += float(nll.item())
            tot_tok += int(ids[:, 1:].numel())
        out[dom] = round(tot_nll / tot_tok, 5)
    out["mix"] = round(sum(MIX[d] * out[d] for d in MIX), 5)
    return out


def build_tokenized_tasks(tok, tasks_path: Path) -> tuple[dict, list]:
    data = json.loads(tasks_path.read_text())
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    ents = []
    for task, rows in data.items():
        for e, r in enumerate(rows):
            pairs = [(c, r["cont"]) for c in r["ctxs"]] if task == "winogrande" else [(r["ctx"], k) for k in r["conts"]]
            for j, (ctx, cont) in enumerate(pairs):
                whole = enc(ctx + cont)[-1024:]
                n_ctx = max(1, len(whole) - len(enc(cont)))
                ents.append((task, e, j, whole, n_ctx, len(cont)))
    ents.sort(key=lambda x: len(x[3]))
    return data, ents


@torch.no_grad()
def score_7tasks_hf(model: nn.Module, tasks_bundle: tuple[dict, list], device: str, amp_dtype, batch_size: int = 32) -> dict:
    model.eval()
    data, ents = tasks_bundle
    S = {}
    use_amp = (device != "cpu")
    for i in range(0, len(ents), batch_size):
        b = ents[i : i + batch_size]
        L = max(len(x[3]) for x in b)
        X = torch.zeros((len(b), L), dtype=torch.long)
        for r, x in enumerate(b):
            X[r, : len(x[3])] = torch.tensor(x[3], dtype=torch.long)
        with torch.autocast(device_type="cuda" if use_amp else "cpu", dtype=amp_dtype, enabled=use_amp):
            logits = model(input_ids=X.to(device), use_cache=False).logits
        lp = torch.log_softmax(logits.float(), -1).cpu()
        for r, x in enumerate(b):
            ids = torch.tensor(x[3], dtype=torch.long)
            s_i, e_i = x[4], len(x[3])
            tl = lp[r, s_i - 1 : e_i - 1].gather(1, ids[s_i:e_i, None])[:, 0]
            greedy = bool((lp[r, s_i - 1 : e_i - 1].argmax(-1) == ids[s_i:e_i]).all())
            S[x[:3]] = (float(tl.sum()), greedy, x[5])

    res = {}
    for task, rows in data.items():
        if task == "lambada":
            ll = np.array([S[(task, e, 0)][0] for e in range(len(rows))])
            res[task] = {
                "acc": round(float(np.mean([S[(task, e, 0)][1] for e in range(len(rows))])), 4),
                "ppl": round(float(np.exp(-ll.mean())), 2),
                "n": len(rows),
            }
            continue
        acc, accn = [], []
        for e, r in enumerate(rows):
            k = 2 if task == "winogrande" else len(r["conts"])
            sc = np.array([S[(task, e, j)][0] for j in range(k)])
            acc.append(int(np.argmax(sc)) == r["gold"])
            if task != "winogrande":
                lens = np.maximum(np.array([S[(task, e, j)][2] for j in range(k)]), 1)
                accn.append(int(np.argmax(sc / lens)) == r["gold"])
        res[task] = {"acc": round(float(np.mean(acc)), 4), "n": len(rows), "chance": CHANCE[task]}
        if accn:
            res[task]["acc_norm"] = round(float(np.mean(accn)), 4)
    return res


def chunked_kd_ce_loss(student_hidden: torch.Tensor, lm_head_w: torch.Tensor, teacher_logits: torch.Tensor, targets: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """Memory-safe chunked 0.5 * CE + 0.5 * KL(teacher || student) over [B*T, d] without materializing full FP32 [B*T, V] graph."""
    flat_h = student_hidden.reshape(-1, student_hidden.shape[-1])
    flat_y = targets.reshape(-1)
    flat_tl = teacher_logits.reshape(-1, teacher_logits.shape[-1])
    tot = torch.zeros((), device=flat_h.device, dtype=torch.float32)

    def _chunk_loss(hc, yc, tlc):
        sl = (hc @ lm_head_w.T).float()
        slp = F.log_softmax(sl, dim=-1)
        ce = -slp.gather(1, yc[:, None])[:, 0].sum()
        tlf = tlc.float()
        tlp = F.log_softmax(tlf, dim=-1)
        tp = tlp.exp()
        kl = (tp * (tlp - slp)).sum(dim=-1).sum()
        return 0.5 * ce + 0.5 * kl

    for i in range(0, flat_h.shape[0], chunk):
        tot = tot + torch.utils.checkpoint.checkpoint(
            _chunk_loss,
            flat_h[i : i + chunk],
            flat_y[i : i + chunk],
            flat_tl[i : i + chunk],
            use_reentrant=False,
        )
    return tot / flat_y.numel()


def build_hybrid_student(teacher_cpu: nn.Module, arm: str, seed: int, device: str, attn_n_ret: int, convert_all_30: bool = False) -> tuple[nn.Module, list[int]]:
    """Constructs T, C1, C2, R, or A_all30 from teacher_cpu with deterministic seed control."""
    student = copy.deepcopy(teacher_cpu)
    cfg = student.config
    H, KV, hd, D, L = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads, cfg.hidden_size, cfg.num_hidden_layers

    if arm == "R":
        torch.manual_seed(seed + 700)
        student.apply(student._init_weights)

    conv_layers = list(range(L)) if convert_all_30 else [i for i in range(L) if i not in KEEP_ATTN]
    for i in conv_layers:
        torch.manual_seed(seed * 100 + i)
        gdn = make_gdn(D, H, hd, device="cpu")
        if arm in ("T", "A_all30"):
            inherit_projections(gdn, teacher_cpu.model.layers[i].self_attn, H, KV, hd)
        student.model.layers[i].self_attn = GDNAttnWrapper(gdn, n_ret=attn_n_ret)

    return student.to(device), conv_layers


def run_stage_b_job(
    job_name: str,
    arm: str,
    seed: int,
    device: str,
    teacher_cpu: nn.Module,
    attn_n_ret: int,
    tasks_bundle: tuple[dict, list],
    train_mm: np.ndarray,
    val_batches: dict[str, torch.Tensor],
    prov: dict,
    bs: int,
    seq: int,
    align_steps: int,
    e2e_steps: int,
) -> dict:
    out_file = OUT_DIR / f"stageb_{job_name}.json"
    if out_file.exists():
        try:
            ex = json.loads(out_file.read_text())
            if ex.get("completed") is True:
                safe_print({"event": "skip_completed", "job": job_name})
                emit_job_receipt(job_name, ex)
                return ex
        except Exception:
            pass

    amp_dtype = torch.bfloat16 if (device != "cpu" and torch.cuda.is_bf16_supported()) else torch.float16
    use_amp = (device != "cpu")
    use_scaler = use_amp and (amp_dtype == torch.float16)

    teacher = copy.deepcopy(teacher_cpu).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    convert_all_30 = (arm == "A_all30")
    student, conv_layers = build_hybrid_student(teacher_cpu, arm=arm, seed=seed, device=device, attn_n_ret=attn_n_ret, convert_all_30=convert_all_30)
    n_params = sum(p.numel() for p in student.parameters())

    rng = np.random.default_rng(seed + 1)
    tok_per_step = bs * seq
    curve = []
    t_train = 0.0
    nonfinite = 0

    # ---- Stage 1: Streaming Per-Layer Sublayer MSE Alignment (`T`, `C1`, `A_all30`) ----
    align_summary = None
    if align_steps > 0:
        gdn_modules = {i: student.model.layers[i].self_attn.gdn for i in conv_layers}
        align_params = [p for i in conv_layers for p in gdn_modules[i].parameters()]
        opt_a = torch.optim.AdamW(align_params, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.0, fused=use_amp)
        scaler_a = torch.amp.GradScaler("cuda", enabled=use_scaler)

        captured: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        hooks = []
        for i in conv_layers:
            def _mk_hook(layer_idx: int):
                def _hook(mod, args, kwargs, out):
                    h_in = kwargs.get("hidden_states", args[0] if args else None)
                    y_out = out[0] if isinstance(out, tuple) else out
                    captured[layer_idx] = (h_in.detach(), y_out.detach())
                return _hook
            hooks.append(teacher.model.layers[i].self_attn.register_forward_hook(_mk_hook(i), with_kwargs=True))

        t0 = time.perf_counter()
        last_rel_mse = 0.0
        for step in range(1, align_steps + 1):
            lr_now = 1e-3 * min(1.0, step / max(1, align_steps // 10)) * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * step / align_steps)))
            for gp in opt_a.param_groups:
                gp["lr"] = lr_now
            starts = rng.integers(0, len(train_mm) - (seq + 1), size=bs)
            batch_np = np.stack([train_mm[int(s) : int(s) + seq] for s in starts]).astype(np.int64)
            ids = torch.from_numpy(batch_np).to(device)

            captured.clear()
            with torch.no_grad(), torch.autocast(device_type="cuda" if use_amp else "cpu", dtype=amp_dtype, enabled=use_amp):
                _ = teacher(input_ids=ids, use_cache=False)

            opt_a.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if use_amp else "cpu", dtype=amp_dtype, enabled=use_amp):
                loss_a = torch.zeros((), device=device, dtype=torch.float32)
                for i in conv_layers:
                    h_in, y_tgt = captured[i]
                    y_pred = gdn_modules[i](h_in)[0]
                    loss_a = loss_a + ((y_pred.float() - y_tgt.float()) ** 2).sum() / ((y_tgt.float() ** 2).sum().clamp_min(1e-6))
                loss_mean = loss_a / len(conv_layers)

            nonfinite += int(not torch.isfinite(loss_mean))
            scaler_a.scale(loss_mean).backward()
            scaler_a.unscale_(opt_a)
            torch.nn.utils.clip_grad_norm_(align_params, 1.0)
            scaler_a.step(opt_a)
            scaler_a.update()
            last_rel_mse = float(loss_mean.detach().item())

        for h in hooks:
            h.remove()
        if use_amp:
            torch.cuda.synchronize(device)
        t_train += time.perf_counter() - t0
        ev_after_align = eval_heldout_nats(student, val_batches, device, amp_dtype)
        align_summary = {
            "align_steps": align_steps,
            "align_tokens": align_steps * tok_per_step,
            "mean_layer_rel_mse_final": round(last_rel_mse, 5),
            "post_align_heldout_nats": ev_after_align,
            "align_sec": round(t_train, 2),
        }
        curve.append({
            "phase": "post_stage1_align",
            "step": align_steps,
            "student_tokens": align_steps * tok_per_step,
            "train_sec": round(t_train, 2),
            **ev_after_align,
        })
        safe_print({"job": job_name, "event": "stage1_done", **align_summary})
        del opt_a

    # ---- Stage 2: End-to-End Distillation (`0.5 * CE + 0.5 * KL(P || S)`) ----
    if e2e_steps > 0:
        gdn_param_ids = {id(p) for i in conv_layers for p in student.model.layers[i].self_attn.parameters()}
        gdn_params = [p for p in student.parameters() if id(p) in gdn_param_ids]
        base_params = [p for p in student.parameters() if id(p) not in gdn_param_ids]
        lr_gdn = 5e-4
        lr_base = 5e-4 if arm == "R" else 1e-4
        opt_e = torch.optim.AdamW(
            [{"params": gdn_params, "lr": lr_gdn, "peak_lr": lr_gdn}, {"params": base_params, "lr": lr_base, "peak_lr": lr_base}],
            betas=(0.9, 0.95),
            weight_decay=0.01,
            fused=use_amp,
        )
        scaler_e = torch.amp.GradScaler("cuda", enabled=use_scaler)
        wu = max(1, min(200, e2e_steps // 10))
        eval_every = max(1, e2e_steps // 5)
        student.train()
        t0 = time.perf_counter()

        for step in range(1, e2e_steps + 1):
            mult = min(1.0, step / wu) * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * step / e2e_steps)))
            for gp in opt_e.param_groups:
                gp["lr"] = gp["peak_lr"] * mult

            starts = rng.integers(0, len(train_mm) - (seq + 1), size=bs)
            batch_np = np.stack([train_mm[int(s) : int(s) + seq + 1] for s in starts]).astype(np.int64)
            ids = torch.from_numpy(batch_np).to(device)
            x_in, y_tgt = ids[:, :-1], ids[:, 1:]

            with torch.no_grad(), torch.autocast(device_type="cuda" if use_amp else "cpu", dtype=amp_dtype, enabled=use_amp):
                t_logits = teacher(input_ids=x_in, use_cache=False).logits.detach()

            opt_e.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if use_amp else "cpu", dtype=amp_dtype, enabled=use_amp):
                s_hidden = student.model(input_ids=x_in, use_cache=False).last_hidden_state
                w_head = student.lm_head.weight
                loss_e = chunked_kd_ce_loss(s_hidden, w_head, t_logits, y_tgt, chunk=8192 if not SMOKE else 128)

            nonfinite += int(not torch.isfinite(loss_e))
            scaler_e.scale(loss_e).backward()
            scaler_e.unscale_(opt_e)
            gn = float(torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0))
            scaler_e.step(opt_e)
            scaler_e.update()

            if step % eval_every == 0 or step == e2e_steps:
                if use_amp:
                    torch.cuda.synchronize(device)
                t_train += time.perf_counter() - t0
                ev = eval_heldout_nats(student, val_batches, device, amp_dtype)
                student.train()
                tot_steps = align_steps + step
                row = {
                    "phase": "stage2_e2e",
                    "step": tot_steps,
                    "e2e_step": step,
                    "student_tokens": tot_steps * tok_per_step,
                    "train_sec": round(t_train, 2),
                    "tokens_per_second": round((tot_steps * tok_per_step) / max(t_train, 1e-6)),
                    "kd_ce_loss": round(float(loss_e.detach().item()), 4),
                    "grad_norm": round(gn, 4),
                    **ev,
                }
                curve.append(row)
                safe_print({"job": job_name, "dev": device, **row})
                t0 = time.perf_counter()

    # Final held-out & 7-task evaluation
    final_heldout = curve[-1] if curve else eval_heldout_nats(student, val_batches, device, amp_dtype)
    tasks_7 = score_7tasks_hf(student, tasks_bundle, device, amp_dtype, batch_size=32 if not SMOKE else 4)
    peak_mem_gb = round(torch.cuda.max_memory_allocated(device) / (2 ** 30), 3) if use_amp else 0.0

    rec = {
        "schema": "nanolm_stage_b_smollm2_135m_v1",
        "completed": True,
        "job_name": job_name,
        "arm": arm,
        "seed": seed,
        "params": n_params,
        "kept_attn_layers": list(KEEP_ATTN) if not convert_all_30 else [],
        "converted_gdn_layers": conv_layers,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(device) if use_amp else "cpu",
        "peak_gpu_gib": peak_mem_gb,
        "nonfinite_steps": nonfinite,
        "recipe": {
            "batch_size": bs,
            "seq": seq,
            "align_steps": align_steps,
            "e2e_steps": e2e_steps,
            "total_student_tokens": (align_steps + e2e_steps) * tok_per_step,
        },
        "provenance": {
            "code_sha256": CODE_SHA256,
            "teacher_model": TEACHER_MODEL,
            "teacher_revision": TEACHER_REV,
            **prov,
        },
        "align_summary": align_summary,
        "curve": curve,
        "final_heldout_nats": {k: final_heldout[k] for k in ("web", "textbook", "math", "code", "mix")},
        "downstream_7tasks": tasks_7,
    }
    out_file.write_text(json.dumps(rec, indent=2))
    emit_job_receipt(job_name, rec)
    del student, teacher
    if use_amp:
        torch.cuda.empty_cache()
    return rec


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM

    if SMOKE:
        cfg = LlamaConfig(
            vocab_size=512,
            hidden_size=64,
            intermediate_size=176,
            num_hidden_layers=30,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
            tie_word_embeddings=True,
        )
        torch.manual_seed(0)
        teacher_cpu = LlamaForCausalLM(cfg).eval()

        class DummyTok:
            vocab_size = 512
            eos_token_id = 1
            def __call__(self, s, add_special_tokens=False):
                if isinstance(s, str):
                    return {"input_ids": [max(2, ord(c) % 512) for c in s[:48]]}
                return {"input_ids": [[max(2, ord(c) % 512) for c in row[:48]] for row in s]}

        tok = DummyTok()
        bs, seq, align_s, e2e_s = 2, 32, 2, 2
        val_n_seq = 2
    else:
        tok = AutoTokenizer.from_pretrained(TEACHER_MODEL, revision=TEACHER_REV)
        teacher_cpu = AutoModelForCausalLM.from_pretrained(TEACHER_MODEL, revision=TEACHER_REV, torch_dtype=torch.float32).eval()
        bs = int(os.environ.get("STAGEB_BS", "32"))
        seq = int(os.environ.get("STAGEB_SEQ", "1024"))
        align_s = int(os.environ.get("STAGEB_ALIGN_STEPS", "640"))      # 20,971,520 tokens
        e2e_s = int(os.environ.get("STAGEB_E2E_STEPS", "7000"))         # 229,376,000 tokens -> total 250,347,520 tokens
        val_n_seq = 64

    prov = prepare_corpus_and_tasks(tok)
    train_mm = np.memmap(DATA_DIR / "train_smollm.bin", dtype=np.uint16, mode="r")
    val_batches = {}
    for d in MIX:
        arr = np.fromfile(DATA_DIR / f"val_{d}.bin", dtype=np.uint16)
        val_batches[d] = torch.from_numpy(arr[: val_n_seq * (seq + 1)].reshape(val_n_seq, seq + 1).astype(np.int64))
    tasks_path = DATA_DIR / "tasks_text.json"
    tasks_bundle = build_tokenized_tasks(tok, tasks_path)
    attn_n_ret = detect_attn_return_arity(teacher_cpu)

    n_gpus = torch.cuda.device_count() if (not SMOKE and torch.cuda.is_available()) else 0
    devices = [f"cuda:{i}" for i in range(n_gpus)] if n_gpus > 0 else ["cpu"]
    safe_print({"event": "stage_b_init", "devices": devices, "attn_n_ret": attn_n_ret, "prov": prov})

    # 1. Score unconverted Parent `P` and Stage-A `A_all30` reference first
    p_file = OUT_DIR / "stageb_P_parent.json"
    if not p_file.exists():
        dev0 = devices[0]
        amp_dt = torch.bfloat16 if (dev0 != "cpu" and torch.cuda.is_bf16_supported()) else torch.float16
        t_mod = copy.deepcopy(teacher_cpu).to(dev0).eval()
        p_nats = eval_heldout_nats(t_mod, val_batches, dev0, amp_dt)
        p_tasks = score_7tasks_hf(t_mod, tasks_bundle, dev0, amp_dt, batch_size=32 if not SMOKE else 4)
        p_rec = {
            "schema": "nanolm_stage_b_smollm2_135m_v1",
            "completed": True,
            "job_name": "P_parent",
            "arm": "P",
            "seed": 0,
            "params": sum(p.numel() for p in t_mod.parameters()),
            "final_heldout_nats": p_nats,
            "downstream_7tasks": p_tasks,
            "provenance": {"code_sha256": CODE_SHA256, "teacher_model": TEACHER_MODEL, "teacher_revision": TEACHER_REV, **prov},
        }
        p_file.write_text(json.dumps(p_rec, indent=2))
        safe_print({"event": "parent_scored", "nats": p_nats})
        emit_job_receipt("P_parent", p_rec)
        del t_mod
    else:
        emit_job_receipt("P_parent", json.loads(p_file.read_text()))

    # Build job queue: A_all30, then paired (T, C1) across 3 seeds, plus C2 and R at equal 250.35M tokens
    tot_steps = align_s + e2e_s
    job_specs = [
        ("A_all30_s1000", "A_all30", 1000, align_s, 0),
        ("T_s1000", "T", 1000, align_s, e2e_s),
        ("C1_s1000", "C1", 1000, align_s, e2e_s),
        ("T_s1001", "T", 1001, align_s, e2e_s),
        ("C1_s1001", "C1", 1001, align_s, e2e_s),
        ("T_s1002", "T", 1002, align_s, e2e_s),
        ("C1_s1002", "C1", 1002, align_s, e2e_s),
        ("C2_s1000", "C2", 1000, 0, tot_steps),
        ("R_s1000", "R", 1000, 0, tot_steps),
    ]

    # Run A_all30 first (~70s) so B2 reference is immediately persisted, then parallelize the 8 main runs across all GPUs!
    a_spec = job_specs[0]
    run_stage_b_job(a_spec[0], a_spec[1], a_spec[2], devices[0], teacher_cpu, attn_n_ret, tasks_bundle, train_mm, val_batches, prov, bs, seq, a_spec[3], a_spec[4])

    q_jobs: queue.Queue = queue.Queue()
    for sp in job_specs[1:]:
        q_jobs.put(sp)

    errors = []
    def worker(dev_str: str):
        while True:
            try:
                jn, arm, sd, a_st, e_st = q_jobs.get_nowait()
            except queue.Empty:
                return
            try:
                run_stage_b_job(jn, arm, sd, dev_str, teacher_cpu, attn_n_ret, tasks_bundle, train_mm, val_batches, prov, bs, seq, a_st, e_st)
            except Exception as exc:
                err = {"job": jn, "dev": dev_str, "error": repr(exc), "trace": traceback.format_exc()[-1000:]}
                safe_print({"event": "job_error", **err})
                errors.append(err)

    threads = [threading.Thread(target=worker, args=(d_s,)) for d_s in devices]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        raise RuntimeError(f"Stage B had {len(errors)} failed jobs: {errors}")

    bundle = {p.stem: json.loads(p.read_text()) for p in sorted(OUT_DIR.glob("stageb_*.json"))}
    (OUT_DIR / "stage_b_bundle.json").write_text(json.dumps(bundle, indent=2))
    safe_print("STAGEB_ALL_DONE")


if __name__ == "__main__":
    main()
