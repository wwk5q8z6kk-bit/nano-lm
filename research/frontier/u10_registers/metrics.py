"""Selected research-only LM metrics and losses from the Antigravity U06, U08 and U09 runners."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from model import HSPALieSO3Mixer, UnifiedBlockLM


def _cosine_lr(step: int, total_steps: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * float(step + 1) / float(max(1, warmup))
    prog = float(step - warmup) / float(max(1, total_steps - warmup))
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog)))


@torch.no_grad()
def eval_bpb_multi_horizon(
    model: UnifiedBlockLM,
    manifest: dict[str, Any],
    dev: torch.device,
    data_dir: Path,
    lengths: tuple[int, ...] = (128, 256, 512),
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for L in lengths:
        dom_bpb: dict[str, float] = {}
        bs = 4 if L <= 256 else 2
        windows = 24 if L <= 256 else 16
        for dom in ("web", "textbook", "math", "code"):
            arr = np.memmap(data_dir / f"val_{dom}.bin", dtype=np.uint16, mode="r")
            stride = max(1, (len(arr) - L - 1) // windows)
            starts = [i * stride for i in range(windows)]
            tot_nats, tot_tok = 0.0, 0
            for i in range(0, len(starts), bs):
                xb = np.stack([arr[s : s + L + 1] for s in starts[i : i + bs]]).astype(np.int64)
                tx = torch.from_numpy(xb).to(dev)
                out_l = model(tx[:, :-1])
                V = model.vocab_size
                if model.spec.has_pointer:
                    nll = F.nll_loss(out_l.reshape(-1, V), tx[:, 1:].reshape(-1), reduction="sum")
                else:
                    nll = F.cross_entropy(out_l.reshape(-1, V), tx[:, 1:].reshape(-1), reduction="sum")
                tot_nats += float(nll.item())
                tot_tok += int(tx[:, 1:].numel())
            bpt = manifest["domains"][dom]["val"]["bytes_per_token"]
            dom_bpb[dom] = round((tot_nats / tot_tok) / math.log(2.0) / bpt, 5)
        out[f"L_{L}"] = dom_bpb
    return out


@torch.no_grad()
def eval_copy_probes(model: UnifiedBlockLM, vocab: int, dev: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for L_half in (32, 64, 128, 256):
        g = torch.Generator().manual_seed(1234 + L_half)
        xs = torch.randint(64, vocab - 64, (16, L_half), generator=g)
        xc = torch.cat([xs, xs], dim=1)
        bs = 4 if L_half <= 128 else 2
        corr, tot_loss, tot_c = 0, 0.0, 0
        for i in range(0, len(xc), bs):
            xb = xc[i : i + bs].to(dev)
            pred = model(xb[:, :-1])
            pred_half = pred[:, L_half:, :].float()
            tgt_half = xb[:, L_half + 1 :]
            if model.spec.has_pointer:
                loss = F.nll_loss(pred_half.reshape(-1, vocab), tgt_half.reshape(-1), reduction="sum")
            else:
                loss = F.cross_entropy(pred_half.reshape(-1, vocab), tgt_half.reshape(-1), reduction="sum")
            tot_loss += float(loss.item())
            corr += int((pred_half.argmax(dim=-1) == tgt_half).sum().item())
            tot_c += int(tgt_half.numel())
        out[f"copy_{L_half}"] = {"acc": round(corr / tot_c, 4), "loss": round(tot_loss / tot_c, 4)}
    return out


@torch.no_grad()
def eval_real_haystack_mqar(model: UnifiedBlockLM, vocab: int, dev: torch.device, data_dir: Path) -> dict[str, Any]:
    web_arr = np.memmap(data_dir / "val_web.bin", dtype=np.uint16, mode="r")
    res: dict[str, Any] = {}
    for band, (lo_id, hi_id) in (("common_ids", (200, 2000)), ("rare_tail_ids", (20000, vocab - 64))):
        band_out: dict[str, float] = {}
        for T_ctx, n_needles in ((128, 4), (256, 4), (512, 4)):
            rng = np.random.default_rng(777 + T_ctx * 13 + n_needles + lo_id)
            n_trials = 24
            bs = 4 if T_ctx <= 256 else 2
            seqs, targets = [], []
            for b in range(n_trials):
                offset = (b * 997) % (len(web_arr) - T_ctx - 10)
                ctx = np.array(web_arr[offset : offset + T_ctx], dtype=np.int64)
                used = set(ctx.tolist())
                cand: list[int] = []
                while len(cand) < 2 * n_needles:
                    tok = int(rng.integers(lo_id, hi_id))
                    if tok not in used:
                        used.add(tok)
                        cand.append(tok)
                pairs = [(cand[2 * i], cand[2 * i + 1]) for i in range(n_needles)]
                positions = np.linspace(8, int(T_ctx * 0.6), n_needles, dtype=int)
                for p, (kt, vt) in zip(positions, pairs):
                    ctx[p] = kt
                    ctx[p + 1] = vt
                qk, qv = pairs[int(rng.integers(0, n_needles))]
                ctx[-1] = qk
                seqs.append(ctx)
                targets.append(qv)
            preds_list = []
            for i in range(0, n_trials, bs):
                xb = torch.tensor(np.stack(seqs[i : i + bs]), dtype=torch.long, device=dev)
                preds_list.append(model(xb)[:, -1, :].argmax(dim=-1).cpu())
            preds = torch.cat(preds_list, dim=0)
            tgt = torch.tensor(targets, dtype=torch.long)
            band_out[f"T{T_ctx}_N{n_needles}"] = round(float((preds == tgt).float().mean().item()), 4)
        res[band] = band_out
    return res


def _compute_u08_loss(
    model: UnifiedBlockLM,
    x: torch.Tensor,
    y: torch.Tensor,
    aux_base_weight: float,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    V = model.vocab_size
    if aux_base_weight > 0.0 and model.spec.has_pointer:
        mix_lp, base_lg = model(x, return_base_logits=True)
        loss_mix = F.nll_loss(mix_lp.reshape(-1, V), y.reshape(-1), ignore_index=ignore_index)
        loss_base = F.cross_entropy(base_lg.reshape(-1, V), y.reshape(-1), ignore_index=ignore_index)
        return loss_mix + aux_base_weight * loss_base
    out = model(x)
    if model.spec.has_pointer:
        return F.nll_loss(out.reshape(-1, V), y.reshape(-1), ignore_index=ignore_index)
    return F.cross_entropy(out.reshape(-1, V), y.reshape(-1), ignore_index=ignore_index)


def _effective_rank_cpu(mat: torch.Tensor) -> float:
    """Effective rank exp(H(s^2 / sum(s^2))) for [T, D] matrix."""
    s = torch.linalg.svdvals(mat.detach().cpu().float())
    p = s.pow(2)
    p = p / p.sum().clamp_min(1e-12)
    ent = -(p * p.clamp_min(1e-12).log()).sum().item()
    return float(math.exp(ent))


def _matrix_health(normed_h: torch.Tensor) -> tuple[float, float, int]:
    """Return (mean_head_eff_rank, mean_offdiag_cos, hd) for normalized [B, T, H, hd]."""
    B, T, H, hd = normed_h.shape
    eye = torch.eye(T, dtype=torch.bool)
    ranks: list[float] = []
    cos_vals: list[float] = []
    for b in range(B):
        xh = normed_h[b].transpose(0, 1).cpu().float()  # [H, T, hd]
        gram = torch.matmul(xh, xh.transpose(-1, -2))
        offdiag = gram.masked_select(~eye.unsqueeze(0).expand_as(gram)).view(H, -1)
        cos_vals.append(float(offdiag.mean().item()))
        for hi in range(H):
            ranks.append(_effective_rank_cpu(xh[hi]))
    return round(sum(ranks) / len(ranks), 2), round(sum(cos_vals) / len(cos_vals), 5), int(hd)


def _annealed_aux_weight(step: int, total_steps: int, start_w: float = 0.25, end_w: float = 0.0) -> float:
    frac = min(1.0, max(0.0, float(step) / float(max(1, total_steps - 1))))
    return end_w + 0.5 * (start_w - end_w) * (1.0 + math.cos(math.pi * frac))


@torch.no_grad()
def audit_layer_ranks_v2(model: UnifiedBlockLM, vocab: int, dev: torch.device, data_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Measure uncentered AND time-centered L0->L3 rank, cosine isotropy, h_in RMS, and ||mix||/||h|| on val_code@256 and copy_128@256."""
    code_arr = np.memmap(data_dir / "val_code.bin", dtype=np.uint16, mode="r")
    stride = (len(code_arr) - 257) // 4
    code_xb = torch.tensor(
        np.stack([code_arr[i * stride : i * stride + 256] for i in range(4)]).astype(np.int64),
        dtype=torch.long,
        device=dev,
    )
    g = torch.Generator().manual_seed(4321)
    xs = torch.randint(64, vocab - 64, (4, 128), generator=g)
    copy_xb = torch.cat([xs, xs], dim=1).to(dev)

    out: dict[str, list[dict[str, Any]]] = {}
    for pname, tokens in (("val_code_256", code_xb), ("copy_128_256", copy_xb)):
        B, T = tokens.shape
        D = model.d_model
        anchor = F.normalize(model.emb(tokens).float(), dim=-1).to(model.emb.weight.dtype)
        h = (math.sqrt(float(D)) * anchor) if model.spec.scaled_residual else anchor
        hist = [h]
        layers: list[dict[str, Any]] = []
        for li, blk in enumerate(model.blocks):
            if model.spec.attn_res:
                stacked = torch.stack(hist, dim=0)
                k_norm = model.depth_norm(stacked).float()
                w = torch.softmax((k_norm * model.depth_q[li].float()).sum(dim=-1), dim=0)
                h_in = (w.unsqueeze(-1) * stacked.float()).sum(dim=0).to(h.dtype)
            else:
                h_in = h

            xn = blk.n1(h_in)
            mix = blk.mix
            mix_y = mix(xn)

            hd = D // mix.h
            mix_4d = mix_y.float().view(B, T, mix.h, hd)
            mix_uncent = F.normalize(mix_4d, dim=-1)
            mix_cent = F.normalize(mix_4d - mix_4d.mean(dim=1, keepdim=True), dim=-1)
            mu_r, mu_c, max_r = _matrix_health(mix_uncent)
            mc_r, mc_c, _ = _matrix_health(mix_cent)

            hin_4d = h_in.float().view(B, T, mix.h, hd)
            hin_uncent = F.normalize(hin_4d, dim=-1)
            hin_cent = F.normalize(hin_4d - hin_4d.mean(dim=1, keepdim=True), dim=-1)
            hu_r, hu_c, _ = _matrix_health(hin_uncent)
            hc_r, hc_c, _ = _matrix_health(hin_cent)

            h_norm_mean = float(h_in.float().norm(dim=-1).mean().item())
            mix_norm_mean = float(mix_y.float().norm(dim=-1).mean().item())
            h_rms = round(float(h_in.float().pow(2).mean().sqrt().item()), 4)
            mix_ratio = round(mix_norm_mean / max(1e-8, h_norm_mean), 4)

            entry: dict[str, Any] = {
                "layer": li,
                "max_rank": max_r,
                "state_eff_rank": mu_r,
                "mix_rank_uncent": mu_r,
                "mix_cos_uncent": mu_c,
                "mix_rank_cent": mc_r,
                "mix_cos_cent": mc_c,
                "h_in_rank_uncent": hu_r,
                "h_in_cos_uncent": hu_c,
                "h_in_rank_cent": hc_r,
                "h_in_cos_cent": hc_c,
                "h_in_rms": h_rms,
                "mix_to_h_ratio": mix_ratio,
            }
            if isinstance(mix, HSPALieSO3Mixer):
                xc = (xn + mix.conv(xn)) if mix.conv is not None else xn
                u_4d = mix.w_in(xc).float().view(B, T, mix.h, mix.r * 3)
                uk_r, uk_c, um = _matrix_health(F.normalize(u_4d, dim=-1))
                ck_r, ck_c, _ = _matrix_health(F.normalize(u_4d - u_4d.mean(dim=1, keepdim=True), dim=-1))
                entry.update({
                    "key_kind": "hspa_u_in",
                    "key_rank_uncent": uk_r,
                    "key_cos_uncent": uk_c,
                    "key_rank_cent": ck_r,
                    "key_cos_cent": ck_c,
                    "key_max_rank": um,
                })
            layers.append(entry)
            h = blk(h_in)
            hist.append(h)
        out[pname] = layers
    return out
