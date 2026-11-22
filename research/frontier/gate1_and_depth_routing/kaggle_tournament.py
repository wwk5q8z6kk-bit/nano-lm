"""Self-contained Kaggle GPU tournament runner for H-13 Block AttnRes & H-10 Coda Refinement.

Runs all 5 arms (`dense`, `value_res`, `block_attnres`, `coda_refine`, `dense10`) for one `SEED`
sequentially on the same physical GPU after a steady-state burn-in, using a Latin-square arm order
across seeds (1000, 1001, 1002) and exact Optax-parity Muon (`lr=4e-3, wd=0.1, floor=0.01`).
"""
from __future__ import annotations
import glob
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = int(os.environ.get("NANOLM_SEED", "1000"))
STEPS = int(os.environ.get("NANOLM_STEPS", "3000"))
EVAL_EVERY = int(os.environ.get("NANOLM_EVAL_EVERY", "500"))
EVAL_BATCHES = int(os.environ.get("NANOLM_EVAL_BATCHES", "20"))
WARMUP = int(os.environ.get("NANOLM_WARMUP", "300"))
BS = int(os.environ.get("NANOLM_BS", "64"))
SEQ = int(os.environ.get("NANOLM_SEQ", "512"))
MICRO = int(os.environ.get("NANOLM_MICRO", "1"))
LR = float(os.environ.get("NANOLM_LR", "0.004"))
LR_FLOOR = float(os.environ.get("NANOLM_LR_FLOOR", "0.01"))
WD = float(os.environ.get("NANOLM_WD", "0.1"))
VOCAB = int(os.environ.get("NANOLM_VOCAB", "32768"))
D_MODEL = int(os.environ.get("NANOLM_D", "256"))
N_LAYERS = int(os.environ.get("NANOLM_LAYERS", "8"))
N_HEAD = int(os.environ.get("NANOLM_HEADS", "4"))
N_KV = int(os.environ.get("NANOLM_KV", "2"))
D_FFN = int(os.environ.get("NANOLM_FFN", "704"))

LATIN_ORDERS = {
    1000: ["dense", "value_res", "block_attnres", "coda_refine", "dense10"],
    1001: ["block_attnres", "coda_refine", "dense10", "dense", "value_res"],
    1002: ["coda_refine", "dense10", "dense", "value_res", "block_attnres"],
}

ARM_SPECS = ("dense", "value_res", "block_attnres", "coda_refine", "dense10")


@dataclass(frozen=True)
class ModelCfg:
    arm: str = "dense"
    vocab: int = 32768
    d: int = 256
    layers: int = 8
    n_head: int = 4
    n_kv: int = 2
    ffn: int = 704
    rope_theta: float = 10000.0
    attnres_block_layers: int = 2
    coda_layers: int = 2


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normed * self.w.float()).to(x.dtype)


def rope_cache(T: int, hd: int, theta: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, device=device, dtype=torch.float32) / hd))
    freqs = torch.outer(torch.arange(T, device=device, dtype=torch.float32), inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    c = cos[None, None, : x.shape[2]].to(x.dtype)
    s = sin[None, None, : x.shape[2]].to(x.dtype)
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2)


class GQAAttention(nn.Module):
    def __init__(self, cfg: ModelCfg, use_value_res: bool):
        super().__init__()
        self.h = cfg.n_head
        self.kv = cfg.n_kv
        self.hd = cfg.d // cfg.n_head
        self.rep = self.h // self.kv
        self.q = nn.Linear(cfg.d, self.h * self.hd, bias=False)
        self.k = nn.Linear(cfg.d, self.kv * self.hd, bias=False)
        self.v = nn.Linear(cfg.d, self.kv * self.hd, bias=False)
        self.o = nn.Linear(self.h * self.hd, cfg.d, bias=False)
        self.qn = RMSNorm(self.hd)
        self.kn = RMSNorm(self.hd)
        self.use_value_res = use_value_res
        if use_value_res:
            self.lam = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor], v0: torch.Tensor | None):
        B, T, _ = x.shape
        q = self.qn(self.q(x).view(B, T, self.h, self.hd)).transpose(1, 2)
        k = self.kn(self.k(x).view(B, T, self.kv, self.hd)).transpose(1, 2)
        v = self.v(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        if v0 is None:
            v0_out = v
        else:
            v0_out = v0
            if self.use_value_res:
                lam = self.lam.to(v.dtype)
                v = lam * v + (1.0 - lam) * v0
        q = apply_rope(q, *rope)
        k = apply_rope(k, *rope)
        k_exp = k.repeat_interleave(self.rep, dim=1)
        v_exp = v.repeat_interleave(self.rep, dim=1)
        y = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True)
        return self.o(y.transpose(1, 2).contiguous().view(B, T, -1)), v0_out


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.gu = nn.Linear(cfg.d, 2 * cfg.ffn, bias=False)
        self.dn = nn.Linear(cfg.ffn, cfg.d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, u = self.gu(x).chunk(2, dim=-1)
        return self.dn(F.silu(g) * u)


class DepthRouter(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.key_norm = RMSNorm(d)
        self.q_depth = nn.Parameter(torch.zeros(d, dtype=torch.float32))
        self.scale = d ** -0.5

    def forward(self, states: list[torch.Tensor]) -> torch.Tensor:
        if len(states) == 1:
            return states[0]
        stacked = torch.stack(states, dim=2)
        keys = self.key_norm(stacked).float()
        scores = torch.einsum("btkd,d->btk", keys, self.q_depth.float()) * self.scale
        weights = F.softmax(scores, dim=-1).to(stacked.dtype)
        return torch.einsum("btk,btkd->btd", weights, stacked)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelCfg, layer_idx: int):
        super().__init__()
        self.n1 = RMSNorm(cfg.d)
        self.n2 = RMSNorm(cfg.d)
        use_vr = (cfg.arm == "value_res") and (layer_idx > 0)
        self.attn = GQAAttention(cfg, use_value_res=use_vr)
        self.mlp = SwiGLU(cfg)
        self.use_attnres = (cfg.arm == "block_attnres")
        if self.use_attnres:
            self.route_attn = DepthRouter(cfg.d)
            self.route_mlp = DepthRouter(cfg.d)

    def forward_standard(self, x: torch.Tensor, rope, v0):
        h_attn, v0 = self.attn(self.n1(x), rope, v0)
        x = x + h_attn
        x = x + self.mlp(self.n2(x))
        return x, v0

    def forward_attnres(self, completed_blocks: list[torch.Tensor], partial_block: torch.Tensor | None, rope, v0):
        states_1 = completed_blocks if partial_block is None else [*completed_blocks, partial_block]
        x_in1 = self.route_attn(states_1)
        u_attn, v0 = self.attn(self.n1(x_in1), rope, v0)
        partial_block = u_attn if partial_block is None else (partial_block + u_attn)

        states_2 = [*completed_blocks, partial_block]
        x_in2 = self.route_mlp(states_2)
        u_mlp = self.mlp(self.n2(x_in2))
        partial_block = partial_block + u_mlp
        return partial_block, v0


class NanoMechanismLM(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        if cfg.arm == "dense10":
            cfg = ModelCfg(
                arm="dense10", vocab=cfg.vocab, d=cfg.d, layers=cfg.layers + 2,
                n_head=cfg.n_head, n_kv=cfg.n_kv, ffn=cfg.ffn, rope_theta=cfg.rope_theta,
            )
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab, cfg.d)
        self.blocks = nn.ModuleList([TransformerBlock(cfg, i) for i in range(cfg.layers)])
        self.nf = RMSNorm(cfg.d)
        if cfg.arm == "block_attnres":
            self.route_final = DepthRouter(cfg.d)
        if cfg.arm == "coda_refine":
            self.refine_gate = nn.Parameter(torch.zeros(cfg.d, dtype=torch.float32))
        self._init_weights()

    def _init_weights(self) -> None:
        res_scale = 1.0 / math.sqrt(2 * self.cfg.layers)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        for name, p in self.named_parameters():
            if p.ndim == 2 and name != "emb.weight":
                nn.init.normal_(p, mean=0.0, std=0.02)
                if name.endswith((".o.weight", ".dn.weight")):
                    with torch.no_grad():
                        p.mul_(res_scale)

    def forward_logits(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        rope = rope_cache(T, self.cfg.d // self.cfg.n_head, self.cfg.rope_theta, idx.device)
        x0 = self.emb(idx)
        v0: torch.Tensor | None = None

        if self.cfg.arm == "block_attnres":
            completed: list[torch.Tensor] = [x0]
            partial: torch.Tensor | None = None
            blk_sz = self.cfg.attnres_block_layers
            for i, blk in enumerate(self.blocks):
                partial, v0 = blk.forward_attnres(completed, partial, rope, v0)
                if (i + 1) % blk_sz == 0 and (i + 1) < self.cfg.layers:
                    assert partial is not None
                    completed.append(partial)
                    partial = None
            assert partial is not None
            completed.append(partial)
            x = self.route_final(completed)
        else:
            x = x0
            for blk in self.blocks:
                x, v0 = blk.forward_standard(x, rope, v0)
            if self.cfg.arm == "coda_refine":
                x_ref = x
                for blk in self.blocks[-self.cfg.coda_layers :]:
                    x_ref, _ = blk.forward_standard(x_ref, rope, v0)
                gate = self.refine_gate.to(x.dtype)
                x = x + gate * (x_ref - x)

        return F.linear(self.nf(x), self.emb.weight)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.forward_logits(idx)
        if targets is None:
            return logits
        return F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))


def newton_schulz_5_batched(g_batch: torch.Tensor, steps: int = 5, eps: float = 1e-8) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g_batch.float()
    transposed = x.shape[1] > x.shape[2]
    if transposed:
        x = x.transpose(1, 2)
    norms = x.flatten(1).norm(dim=1).view(-1, 1, 1)
    x = x / (norms + eps)
    for _ in range(steps):
        gram = torch.bmm(x, x.transpose(1, 2))
        x = a * x + torch.bmm(b * gram + c * torch.bmm(gram, gram), x)
    if transposed:
        x = x.transpose(1, 2)
    return x


class OptaxParityMuon(torch.optim.Optimizer):
    def __init__(self, named_params: Iterable[tuple[str, nn.Parameter]], lr: float = 4e-3, wd: float = 0.1):
        params_list = []
        self.param_is_muon: dict[int, bool] = {}
        for name, p in named_params:
            if not p.requires_grad:
                continue
            params_list.append(p)
            self.param_is_muon[id(p)] = (p.ndim == 2) and ("blocks." in name)
        super().__init__(params_list, dict(lr=lr, wd=wd, muon_beta=0.95, adam_betas=(0.9, 0.95), eps=1e-8, consistent_rms=0.2, ns_steps=5))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, wd = group["lr"], group["wd"]
            beta = group["muon_beta"]
            b1, b2 = group["adam_betas"]
            eps = group["eps"]
            c_rms = group["consistent_rms"]
            ns_steps = group["ns_steps"]
            muon_buckets: dict[tuple[int, int], list[tuple[nn.Parameter, torch.Tensor]]] = {}
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()
                st = self.state[p]
                if len(st) == 0:
                    st["step"] = 0
                    if self.param_is_muon[id(p)]:
                        st["mu"] = torch.zeros_like(p, dtype=torch.float32)
                    else:
                        st["m"] = torch.zeros_like(p, dtype=torch.float32)
                        st["v"] = torch.zeros_like(p, dtype=torch.float32)
                st["step"] += 1
                t = st["step"]
                if self.param_is_muon[id(p)]:
                    mu = st["mu"]
                    mu.mul_(beta).add_(g, alpha=1.0 - beta)
                    mu_hat = beta * (mu / (1.0 - (beta ** (t + 1)))) + (1.0 - beta) * (g / (1.0 - (beta ** t)))
                    muon_buckets.setdefault((p.shape[0], p.shape[1]), []).append((p, mu_hat))
                else:
                    m, v = st["m"], st["v"]
                    m.mul_(b1).add_(g, alpha=1.0 - b1)
                    v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
                    m_hat = b1 * (m / (1.0 - (b1 ** (t + 1)))) + (1.0 - b1) * (g / (1.0 - (b1 ** t)))
                    v_hat = v / (1.0 - (b2 ** t))
                    upd = m_hat / (v_hat.sqrt() + eps) + wd * p.float()
                    p.add_(upd.to(p.dtype), alpha=-lr)
            for (r_dim, c_dim), items in muon_buckets.items():
                stacked = torch.stack([mh for _, mh in items], dim=0)
                orthos = newton_schulz_5_batched(stacked, steps=ns_steps, eps=eps)
                shape_scale = c_rms * math.sqrt(max(r_dim, c_dim))
                for idx_p, (p, _) in enumerate(items):
                    upd = orthos[idx_p] * shape_scale + wd * p.float()
                    p.add_(upd.to(p.dtype), alpha=-lr)
        return loss


def find_data_root() -> Path:
    env = os.environ.get("NANOLM_DATA")
    if env and Path(env).exists():
        return Path(env)
    for cand in sorted(glob.glob("/kaggle/input/**/manifest.json", recursive=True)):
        return Path(cand).parent
    local = Path("/Volumes/Express1M2/nanolm-lab/data/v1")
    if local.exists():
        return local
    raise SystemExit("Cannot find nanolm-corpus-v1-800m manifest.json")


def sample_batch(data: np.ndarray, batch_size: int, seq_len: int, rng: np.random.Generator, dev: torch.device):
    ix = rng.integers(0, len(data) - seq_len - 1, batch_size)
    arr = np.stack([data[i : i + seq_len + 1] for i in ix]).astype(np.int64)
    t = torch.from_numpy(arr).to(dev, non_blocking=True)
    return t[:, :-1], t[:, 1:]


@torch.no_grad()
def evaluate_bpb_and_ranks(model: NanoMechanismLM, val_arrays: dict[str, np.ndarray], manifest: dict, dev: torch.device, compute_ranks: bool = False):
    model.eval()
    out_bpb: dict[str, float] = {}
    dom_ranks: dict[str, dict[str, float]] = {}
    use_amp = (dev.type == "cuda")
    for dom in ("web", "textbook", "math", "code"):
        d = val_arrays[dom]
        rng = np.random.default_rng(0)
        bpt = manifest["domains"][dom]["val"]["bytes_per_token"]
        nll_sum = 0.0
        r1_bits = r2_8_bits = r9_64_bits = r65_bits = 0.0
        top1_hits = top64_hits = total_toks = 0
        for _ in range(EVAL_BATCHES):
            x, y = sample_batch(d, 16, SEQ, rng, dev)
            if use_amp:
                with torch.autocast("cuda", dtype=torch.float16):
                    lg = model.forward_logits(x).float()
            else:
                lg = model.forward_logits(x).float()
            lp = F.log_softmax(lg, dim=-1)
            true_lp = lp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
            nll_batch = -true_lp
            nll_sum += float(nll_batch.mean())
            if compute_ranks:
                true_lg = lg.gather(-1, y.unsqueeze(-1))
                rnk = (lg > true_lg).sum(dim=-1)
                bits = nll_batch / math.log(2.0)
                r1_bits += float(bits[rnk == 0].sum())
                r2_8_bits += float(bits[(rnk >= 1) & (rnk < 8)].sum())
                r9_64_bits += float(bits[(rnk >= 8) & (rnk < 64)].sum())
                r65_bits += float(bits[rnk >= 64].sum())
                top1_hits += int((rnk == 0).sum())
                top64_hits += int((rnk < 64).sum())
                total_toks += y.numel()
        out_bpb[dom] = round(nll_sum / float(EVAL_BATCHES) / math.log(2.0) / bpt, 4)
        if compute_ranks:
            scale = 1.0 / (total_toks * bpt)
            dom_ranks[dom] = {
                "rank1_calib": round(r1_bits * scale, 4),
                "rank2_8_rerank": round(r2_8_bits * scale, 4),
                "rank9_64_mid": round(r9_64_bits * scale, 4),
                "rank65plus_miss": round(r65_bits * scale, 4),
                "top1_acc": round(top1_hits / total_toks, 4),
                "top64_acc": round(top64_hits / total_toks, 4),
            }
    weights = manifest["mix"]
    out_bpb["mix"] = round(sum(weights[k] * out_bpb[k] for k in weights), 4)
    mix_ranks = {}
    if compute_ranks:
        for rk in ("rank1_calib", "rank2_8_rerank", "rank9_64_mid", "rank65plus_miss", "top1_acc", "top64_acc"):
            mix_ranks[rk] = round(sum(weights[k] * dom_ranks[k][rk] for k in weights), 4)
    model.train()
    return out_bpb, dom_ranks, mix_ranks


def gpu_burn_in(dev: torch.device, seconds: float = 15.0) -> None:
    if dev.type != "cuda":
        return
    a = torch.randn(2048, 2048, device=dev, dtype=torch.float16)
    b = torch.randn(2048, 2048, device=dev, dtype=torch.float16)
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) < seconds:
        _ = a @ b
    torch.cuda.synchronize()


def inspect_mechanism_diagnostics(model: NanoMechanismLM) -> dict:
    arm = model.cfg.arm
    if arm == "value_res":
        return {"lambdas": [round(float(blk.attn.lam.detach().cpu()), 4) for blk in model.blocks[1:]]}
    if arm == "block_attnres":
        q_norms = []
        for blk in model.blocks:
            q_norms.append(round(float(blk.route_attn.q_depth.detach().norm().cpu()), 4))
            q_norms.append(round(float(blk.route_mlp.q_depth.detach().norm().cpu()), 4))
        q_norms.append(round(float(model.route_final.q_depth.detach().norm().cpu()), 4))
        return {"q_depth_l2_norms": q_norms}
    if arm == "coda_refine":
        g = model.refine_gate.detach().cpu()
        return {
            "refine_gate_rms": round(float(g.pow(2).mean().sqrt()), 4),
            "refine_gate_abs_max": round(float(g.abs().max()), 4),
            "refine_gate_mean": round(float(g.mean()), 4),
        }
    return {}


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    root = find_data_root()
    out_dir = Path(os.environ.get("NANOLM_OUT", "/kaggle/working" if Path("/kaggle/working").exists() else "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"seed_{SEED}.json"

    manifest = json.loads((root / "manifest.json").read_text())
    train_data = np.memmap(root / "train.bin", dtype=np.uint16, mode="r")
    val_arrays = {dom: np.fromfile(root / f"val_{dom}.bin", dtype=np.uint16) for dom in ("web", "textbook", "math", "code")}

    arm_order = LATIN_ORDERS.get(SEED, list(ARM_SPECS))
    gpu_name = torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
    print(json.dumps({"event": "start", "seed": SEED, "gpu": gpu_name, "steps": STEPS, "arm_order": arm_order}), flush=True)

    gpu_burn_in(dev, seconds=15.0 if dev.type == "cuda" else 0.0)

    report = {
        "schema": "nanolm.h13_h10_muon_tournament.v1",
        "seed": SEED,
        "gpu": gpu_name,
        "recipe": {
            "opt": "OptaxParityMuon",
            "lr": LR,
            "lr_floor": LR_FLOOR,
            "wd": WD,
            "warmup": WARMUP,
            "bs": BS,
            "seq": SEQ,
            "micro": MICRO,
            "steps": STEPS,
            "tokens_per_arm": STEPS * BS * SEQ,
        },
        "arm_order": arm_order,
        "arms": {},
    }

    for arm in arm_order:
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(SEED)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(SEED)
        rng = np.random.default_rng(SEED + 1)

        model = NanoMechanismLM(
            ModelCfg(
                arm=arm,
                vocab=VOCAB,
                d=D_MODEL,
                layers=N_LAYERS,
                n_head=N_HEAD,
                n_kv=N_KV,
                ffn=D_FFN,
            )
        ).to(dev)
        params = sum(p.numel() for p in model.parameters())
        opt = OptaxParityMuon(model.named_parameters(), lr=LR, wd=WD)
        scaler = torch.amp.GradScaler("cuda", init_scale=4096.0) if dev.type == "cuda" else None

        curve = []
        train_sec = 0.0
        skipped_steps = 0

        for step in range(1, STEPS + 1):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t_s = time.perf_counter()

            lr_t = LR * min(1.0, step / WARMUP) * (LR_FLOOR + (1.0 - LR_FLOOR) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, step / STEPS))))
            for grp in opt.param_groups:
                grp["lr"] = lr_t

            x, y = sample_batch(train_data, BS, SEQ, rng, dev)
            opt.zero_grad(set_to_none=True)
            loss_val = 0.0
            for xm, ym in zip(x.chunk(MICRO), y.chunk(MICRO)):
                if scaler is not None:
                    with torch.autocast("cuda", dtype=torch.float16):
                        lm = model(xm, ym) / MICRO
                    scaler.scale(lm).backward()
                else:
                    lm = model(xm, ym) / MICRO
                    lm.backward()
                loss_val += float(lm.detach())

            if scaler is not None:
                scaler.unscale_(opt)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                scale_before = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                if scaler.get_scale() < scale_before:
                    skipped_steps += 1
            else:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                opt.step()

            if dev.type == "cuda":
                torch.cuda.synchronize()
            train_sec += (time.perf_counter() - t_s)

            if step % EVAL_EVERY == 0 or step == STEPS:
                is_final = (step == STEPS)
                bpb, dom_ranks, mix_ranks = evaluate_bpb_and_ranks(model, val_arrays, manifest, dev, compute_ranks=is_final)
                toks = step * BS * SEQ
                row = {
                    "step": step,
                    "tokens": toks,
                    "train_sec": round(train_sec, 2),
                    "tokens_per_second": round(toks / max(train_sec, 1e-6)),
                    "train_loss": round(loss_val, 4),
                    "grad_norm": round(grad_norm, 4),
                    "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / (2 ** 30), 3) if dev.type == "cuda" else 0.0,
                    **bpb,
                }
                curve.append(row)
                print(json.dumps({"seed": SEED, "arm": arm, **row}), flush=True)
                arm_rec = {
                    "params": params,
                    "skipped_amp_steps": skipped_steps,
                    "curve": curve,
                }
                if is_final:
                    arm_rec["rank_buckets_mix"] = mix_ranks
                    arm_rec["rank_buckets_domains"] = dom_ranks
                    arm_rec["diagnostics"] = inspect_mechanism_diagnostics(model)
                report["arms"][arm] = arm_rec
                out_file.write_text(json.dumps(report, indent=2))

        del model, opt, scaler

    report["completed"] = True
    out_file.write_text(json.dumps(report, indent=2))
    print(json.dumps({"event": "all_arms_done", "seed": SEED, "out": str(out_file)}), flush=True)


if __name__ == "__main__":
    main()
