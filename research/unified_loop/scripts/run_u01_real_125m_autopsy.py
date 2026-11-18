#!/usr/bin/env python3
"""EXP-U01: Forensic Mechanistic Autopsy of Real 600M-Token Checkpoints (S0_125m, S1_125m, X4_46m).

Read-only inspection of:
  - /Volumes/Express1M2/nanolm-lab/runs/{S0_dense_125m, S1_nanolm_v2_125m, X4_nanolm_v2_muon}__muon__tokens__600M__s0/final.pt
  - /Volumes/Express1M2/nanolm-lab/data/v1/val_{web,textbook,math,code}.bin
  - /Users/mac/Projects/nano-lm/checkpoints/anchors/*.pt & chinchilla-160m/*.pt (structure & param audit)
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

WORKTREE_ROOT = Path(__file__).resolve().parents[3]
MAIN_REPO = Path("/Users/mac/Projects/nano-lm")
LAB_WT = MAIN_REPO / ".worktrees" / "nanolm-lab"
sys.path.insert(0, str(LAB_WT))

from lab.arch.nextgen import NextGenCfg, NextGenDeltaMixer, NextGenLM  # noqa: E402

DATA_DIR = Path("/Volumes/Express1M2/nanolm-lab/data/v1")
RUNS_DIR = Path("/Volumes/Express1M2/nanolm-lab/runs")
OUT_JSON = WORKTREE_ROOT / "research" / "unified_loop" / "experiments" / "U01_real_125m_autopsy" / "results.json"


@torch.no_grad()
def evaluate_canonical_chunked(
    model: NextGenLM,
    data: Path,
    manifest: dict[str, Any],
    dev: torch.device,
    dtype: torch.dtype,
    seq: int = 1024,
    windows: int = 48,
    bs: int = 4,
) -> dict[str, Any]:
    """Memory-safe exact equivalent of lab.arch.train.evaluate with bs-chunked copy eval."""
    model.eval()
    out: dict[str, Any] = {"bpb": {}}
    for dom, meta in manifest["domains"].items():
        arr = np.fromfile(data / f"val_{dom}.bin", dtype=np.uint16)
        n = min(windows, (len(arr) - 1) // (seq + 1))
        idx = np.arange(n) * ((len(arr) - seq - 1) // n)
        tot, cnt = 0.0, 0
        pos_loss = torch.zeros(seq, device=dev)
        for i in range(0, n, bs):
            xb = np.stack([arr[j : j + seq + 1] for j in idx[i : i + bs]]).astype(np.int64)
            x = torch.from_numpy(xb).to(dev)
            with torch.autocast(dev.type, dtype=dtype, enabled=dtype != torch.float32):
                lg = model(x[:, :-1])
            l = F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), x[:, 1:].reshape(-1), reduction="none").view(x.size(0), seq)
            pos_loss += l.sum(0)
            tot += float(l.sum().item())
            cnt += int(l.numel())
        bpt = meta["val"]["bytes_per_token"]
        out["bpb"][dom] = round((tot / cnt) / math.log(2) / bpt, 5)
        pl = (pos_loss / n).cpu()
        if dom == "web":
            out["web_loss_pos_0_64"] = round(float(pl[:64].mean().item()), 4)
            out["web_loss_pos_last_256"] = round(float(pl[-256:].mean().item()), 4)
    w = {"web": 0.4, "textbook": 0.2, "math": 0.2, "code": 0.2}
    out["bpb_composite"] = round(sum(out["bpb"][k] * w[k] for k in w), 5)

    g = torch.Generator().manual_seed(1234)
    L = seq // 2
    xs = torch.randint(64, manifest["vocab"] - 64, (32, L), generator=g)
    xc = torch.cat([xs, xs], 1)
    tot_l, corr_c, tot_c = 0.0, 0, 0
    for i in range(0, len(xc), bs):
        xb = xc[i : i + bs].to(dev)
        with torch.autocast(dev.type, dtype=dtype, enabled=dtype != torch.float32):
            lg = model(xb[:, :-1])
        lg_h = lg[:, L:].float()
        tg_h = xb[:, L + 1 :]
        tot_l += float(F.cross_entropy(lg_h.reshape(-1, lg_h.size(-1)), tg_h.reshape(-1), reduction="sum").item())
        corr_c += int((lg_h.argmax(-1) == tg_h).sum().item())
        tot_c += int(tg_h.numel())
    out["copy_loss_second_half"] = round(tot_l / tot_c, 4)
    out["copy_acc_second_half"] = round(corr_c / tot_c, 4)
    return out


def load_nextgen_ckpt(run_name: str, dev: torch.device) -> tuple[NextGenLM, dict[str, Any]]:
    ckpt_path = RUNS_DIR / run_name / "final.pt"
    result_path = RUNS_DIR / run_name / "result.json"
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = NextGenCfg(**ck["cfg"])
    model = NextGenLM(cfg)
    model.load_state_dict(ck["model"])
    model = model.to(dev).eval()
    reported = json.loads(result_path.read_text()) if result_path.exists() else {}
    return model, reported


def effective_rank(mat: torch.Tensor) -> float:
    """Effective rank exp(H(s^2 / sum(s^2))) for [T, hd] matrix."""
    s = torch.linalg.svdvals(mat.float())
    p = s.pow(2)
    p = p / p.sum().clamp_min(1e-12)
    ent = -(p * p.clamp_min(1e-12).log()).sum().item()
    return float(math.exp(ent))


@torch.no_grad()
def inspect_delta_mechanisms(model: NextGenLM, tokens: torch.Tensor, dev: torch.device, dtype: torch.dtype) -> list[dict[str, Any]]:
    """Trace exact internal pre-norm keys, cosines, effective ranks, decay, beta, g-gate, and v_gate per layer."""
    B, T = tokens.shape
    x = model.emb(tokens.to(dev))
    hd = model.c.d // model.c.n_head
    from lab.arch.model import rope_cache
    rope = rope_cache(T, hd, model.c.rope_theta, dev)
    v0 = None
    layer_stats: list[dict[str, Any]] = []

    for li, blk in enumerate(model.blocks):
        xn = blk.n1(x)
        if blk.kind == "D":
            mix: NextGenDeltaMixer = blk.mix
            with torch.autocast(dev.type, dtype=dtype, enabled=dtype != torch.float32):
                qkv = mix.qkv(xn)
            if mix.sharp_v:
                qk, _ = qkv.split([2 * model.c.d, model.c.d], dim=-1)
                with torch.autocast(dev.type, dtype=dtype, enabled=dtype != torch.float32):
                    qk_conv = mix.conv(qk)
                q_c, k_c = qk_conv.split(model.c.d, dim=-1)
            else:
                with torch.autocast(dev.type, dtype=dtype, enabled=dtype != torch.float32):
                    q_c, k_c, _ = mix.conv(qkv).split(model.c.d, dim=-1)

            k_f = F.normalize(k_c.view(B, T, mix.h, mix.hd).float(), dim=-1)  # [B, T, H, hd]
            # Per-head off-diagonal cosine and effective rank over positions
            k_h = k_f[0].transpose(0, 1)  # [H, T, hd]
            gram = torch.matmul(k_h, k_h.transpose(-1, -2))  # [H, T, T]
            eye = torch.eye(T, device=dev, dtype=torch.bool).unsqueeze(0)
            offdiag = gram.masked_select(~eye.expand_as(gram)).view(mix.h, -1)
            per_head_cos = offdiag.mean(dim=-1).tolist()
            mean_offdiag_cos = float(offdiag.mean().item())
            frac_pos_entries = float((k_f > 0).float().mean().item())
            eff_ranks = [effective_rank(k_h[h_i]) for h_i in range(mix.h)]

            loga = (-F.softplus(mix.a(xn.float())) * mix.A_log.exp()).transpose(1, 2)
            decay = loga.exp()
            beta = torch.sigmoid(mix.b(xn.float())).transpose(1, 2)
            g_act = F.silu(mix.g(xn.float()))
            v_alpha = torch.sigmoid(mix.v_gate(xn.float())) if mix.gated_vres else None

            layer_stats.append({
                "layer": li,
                "kind": "D",
                "mean_offdiag_key_cos": round(mean_offdiag_cos, 5),
                "min_head_key_cos": round(min(per_head_cos), 5),
                "max_head_key_cos": round(max(per_head_cos), 5),
                "mean_key_eff_rank": round(sum(eff_ranks) / len(eff_ranks), 2),
                "max_possible_rank": mix.hd,
                "frac_positive_key_coords": round(frac_pos_entries, 4),
                "mean_decay_a": round(float(decay.mean().item()), 4),
                "mean_beta": round(float(beta.mean().item()), 4),
                "mean_abs_silu_gate": round(float(g_act.abs().mean().item()), 4),
                "frac_gate_near_zero_lt_0.05": round(float((g_act.abs() < 0.05).float().mean().item()), 4),
                "mean_v_alpha": round(float(v_alpha.mean().item()), 4) if v_alpha is not None else None,
            })
        else:
            if blk.mix.gated_vres:
                v_alpha = torch.sigmoid(blk.mix.v_gate(xn.float()))
                layer_stats.append({
                    "layer": li,
                    "kind": "A",
                    "mean_v_alpha": round(float(v_alpha.mean().item()), 4),
                })
            else:
                layer_stats.append({"layer": li, "kind": "A", "mean_v_alpha": None})

        with torch.autocast(dev.type, dtype=dtype, enabled=dtype != torch.float32):
            x, v0, _ = blk(x, rope, v0)
    return layer_stats


@torch.no_grad()
def eval_length_extrapolation(
    model: NextGenLM,
    manifest: dict[str, Any],
    dev: torch.device,
    dtype: torch.dtype,
    lengths: tuple[int, ...] = (256, 512, 1024, 2048),
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for L in lengths:
        res_L: dict[str, Any] = {"bpb": {}}
        for dom in ("web", "code"):
            arr = np.fromfile(DATA_DIR / f"val_{dom}.bin", dtype=np.uint16)
            windows = 24 if L <= 1024 else 16
            bs = 4 if L <= 1024 else 2
            n = min(windows, (len(arr) - 1) // (L + 1))
            idx = np.arange(n) * ((len(arr) - L - 1) // n)
            tot, cnt = 0.0, 0
            for i in range(0, n, bs):
                xb = np.stack([arr[j : j + L + 1] for j in idx[i : i + bs]]).astype(np.int64)
                x = torch.from_numpy(xb).to(dev)
                with torch.autocast(dev.type, dtype=dtype, enabled=dtype != torch.float32):
                    lg = model(x[:, :-1])
                loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)).float(), x[:, 1:].reshape(-1), reduction="sum")
                tot += float(loss.item())
                cnt += int(x[:, 1:].numel())
            bpt = manifest["domains"][dom]["val"]["bytes_per_token"]
            res_L["bpb"][dom] = round((tot / cnt) / math.log(2) / bpt, 5)

        # Copy probe at L_copy = L // 2 -> sequence length L
        L_half = L // 2
        g = torch.Generator().manual_seed(1234 + L)
        xs = torch.randint(64, manifest["vocab"] - 64, (16, L_half), generator=g)
        xc = torch.cat([xs, xs], dim=1)
        corr_c, tot_loss, tot_c = 0, 0.0, 0
        for i in range(0, len(xc), 4):
            xb = xc[i : i + 4].to(dev)
            with torch.autocast(dev.type, dtype=dtype, enabled=dtype != torch.float32):
                lgc = model(xb[:, :-1])
            lg_half = lgc[:, L_half:, :].float()
            tgt_half = xb[:, L_half + 1 :]
            tot_loss += float(F.cross_entropy(lg_half.reshape(-1, lg_half.size(-1)), tgt_half.reshape(-1), reduction="sum").item())
            corr_c += int((lg_half.argmax(-1) == tgt_half).sum().item())
            tot_c += int(tgt_half.numel())
        res_L["copy_loss"] = round(tot_loss / tot_c, 4)
        res_L["copy_acc"] = round(corr_c / tot_c, 4)
        out[f"L_{L}"] = res_L
    return out


@torch.no_grad()
def eval_needle_mqar_real_distractors(
    model: NextGenLM,
    manifest: dict[str, Any],
    dev: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Zero-shot Needle-in-a-Haystack / MQAR with real val_web background distractors.

    Places `n_needles` (K_j, V_j) pairs into a real text haystack of length `T_ctx`,
    then repeats a query key `K_target` at the end and checks whether top-1 logit predicts `V_target`.
    Separately tests common-range token IDs `[200..2000]` vs rare tail token IDs `[20000..32000]`.
    """
    from lab.arch.model import rope_cache
    web_arr = np.fromfile(DATA_DIR / "val_web.bin", dtype=np.uint16)
    vocab = manifest["vocab"]
    results: dict[str, Any] = {}
    for band_name, (lo_id, hi_id) in (("common_ids", (200, 2000)), ("rare_tail_ids", (20000, vocab - 64))):
        band_res: dict[str, Any] = {}
        for T_ctx, n_needles in ((256, 1), (256, 4), (512, 4), (512, 8), (1024, 8)):
            rng = np.random.default_rng(777 + T_ctx * 13 + n_needles + lo_id)
            n_trials = 24
            seqs = []
            targets = []
            for b in range(n_trials):
                offset = (b * 997) % (len(web_arr) - T_ctx - 10)
                ctx = web_arr[offset : offset + T_ctx].astype(np.int64).copy()
                # Ensure unique keys & values disjoint from background window
                used = set(ctx.tolist())
                cand = []
                while len(cand) < 2 * n_needles:
                    tok = int(rng.integers(lo_id, hi_id))
                    if tok not in used:
                        used.add(tok)
                        cand.append(tok)
                pairs = [(cand[2 * i], cand[2 * i + 1]) for i in range(n_needles)]
                # Spread pairs in first 60% of context
                positions = np.linspace(8, int(T_ctx * 0.6), n_needles, dtype=int)
                for p, (k_tok, v_tok) in zip(positions, pairs):
                    ctx[p] = k_tok
                    ctx[p + 1] = v_tok
                q_idx = int(rng.integers(0, n_needles))
                q_key, q_val = pairs[q_idx]
                ctx[-1] = q_key
                seqs.append(ctx)
                targets.append(q_val)
            all_logits = []
            hd = model.c.d // model.c.n_head
            for i in range(0, n_trials, 4):
                xb = torch.tensor(np.stack(seqs[i : i + 4]), dtype=torch.long, device=dev)
                x_h = model.emb(xb)
                rope = rope_cache(T_ctx, hd, model.c.rope_theta, dev)
                v0 = None
                for blk in model.blocks:
                    x_h, v0, _ = blk(x_h, rope, v0)
                h_last = model.nf(x_h[:, -1:, :])
                lg_last = (h_last @ model.emb.weight.T if model.head is None else model.head(h_last)).squeeze(1).float()
                all_logits.append(lg_last)
            lg = torch.cat(all_logits, dim=0)
            tgt = torch.tensor(targets, dtype=torch.long, device=dev)
            acc = float((lg.argmax(dim=-1) == tgt).float().mean().item())
            nll = float(F.cross_entropy(lg, tgt).item())
            band_res[f"T{T_ctx}_N{n_needles}"] = {"acc": round(acc, 4), "nll": round(nll, 4)}
        results[band_name] = band_res
    return results


def audit_legacy_checkpoints() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for rel in (
        "checkpoints/anchors/nano_v01_scribe.pt",
        "checkpoints/anchors/scale10m_scribe.pt",
        "checkpoints/anchors/scribe.pt",
        "checkpoints/chinchilla-160m/ownstack160m_scribe.pt",
        "checkpoints/chinchilla-160m/ownstack160m_pretrain.pt",
    ):
        p = MAIN_REPO / rel
        if not p.exists():
            continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        sd = ck.get("model") or ck.get("model_state_dict") or ck.get("state_dict") or (
            ck if isinstance(ck, dict) and all(isinstance(v, torch.Tensor) for v in ck.values()) else {}
        )
        nparams = sum(v.numel() for v in sd.values() if isinstance(v, torch.Tensor))
        emb_shape = list(sd["emb.weight"].shape) if "emb.weight" in sd else None
        n_blocks = len({k.split(".")[1] for k in sd if k.startswith("blocks.")})
        out[rel] = {
            "size_mb": round(p.stat().st_size / (1024 ** 2), 2),
            "param_count": int(nparams),
            "emb_shape": emb_shape,
            "num_blocks": n_blocks,
            "top_keys": list(ck.keys())[:10] if isinstance(ck, dict) else [],
            "step": ck.get("step") if isinstance(ck, dict) else None,
        }
    return out


def main() -> int:
    torch.set_num_threads(4)
    t0 = time.perf_counter()
    dev = torch.device("cpu")
    dtype = torch.float32
    manifest = json.loads((DATA_DIR / "manifest.json").read_text())

    # Prepare one fixed real code/web probe batch (4 x 512) for internal mechanism inspection
    code_arr = np.fromfile(DATA_DIR / "val_code.bin", dtype=np.uint16)
    probe_code = torch.tensor(
        np.stack([code_arr[i * 512 : (i + 1) * 512] for i in range(4)]).astype(np.int64),
        dtype=torch.long,
    )
    g_copy = torch.Generator().manual_seed(999)
    half_c = torch.randint(64, manifest["vocab"] - 64, (4, 256), generator=g_copy)
    probe_copy = torch.cat([half_c, half_c], dim=1)

    models_to_audit = [
        "X4_nanolm_v2_muon__muon__tokens__600M__s0",
        "S0_dense_125m__muon__tokens__600M__s0",
        "S1_nanolm_v2_125m__muon__tokens__600M__s0",
    ]

    report: dict[str, Any] = {
        "schema": "nanolm.unified_loop.u01.v1",
        "device": str(dev),
        "dtype": str(dtype),
        "paid_compute_usd": 0.0,
        "legacy_checkpoints_verified": audit_legacy_checkpoints(),
        "models": {},
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    for run_name in models_to_audit:
        mt0 = time.perf_counter()
        print(f"[U-01] Auditing {run_name} on {dev}...", flush=True)
        model, reported = load_nextgen_ckpt(run_name, dev)
        rep = model.param_report()

        # 1. Exact reproduction of canonical 1024 eval (48 windows, bs=4, copy L=512)
        repro = evaluate_canonical_chunked(model, DATA_DIR, manifest, dev, dtype, seq=1024, windows=48, bs=4)
        print(f"  [1/4] repro done in {time.perf_counter()-mt0:.1f}s: BPB={repro['bpb_composite']} copy={repro['copy_acc_second_half']}", flush=True)

        # 2. Internal mechanism inspection on real val_code & random copy
        mech_code = inspect_delta_mechanisms(model, probe_code, dev, dtype)
        mech_copy = inspect_delta_mechanisms(model, probe_copy, dev, dtype)
        print(f"  [2/4] mech probes done in {time.perf_counter()-mt0:.1f}s", flush=True)

        # 3. Length extrapolation (256 -> 2048)
        extrap = eval_length_extrapolation(model, manifest, dev, dtype, lengths=(256, 512, 1024, 2048))
        print(f"  [3/4] extrap done in {time.perf_counter()-mt0:.1f}s: L2048 web={extrap['L_2048']['bpb']['web']} copy1024={extrap['L_2048']['copy_acc']}", flush=True)

        # 4. Zero-shot MQAR needle retrieval with real distractors
        mqar = eval_needle_mqar_real_distractors(model, manifest, dev, dtype)
        print(f"  [4/4] mqar done in {time.perf_counter()-mt0:.1f}s", flush=True)

        report["models"][run_name] = {
            "cfg": vars(model.c),
            "param_report": rep,
            "reported_final": reported.get("final"),
            "reproduced_eval_1024": repro,
            "internal_mechanisms_val_code": mech_code,
            "internal_mechanisms_copy_probe": mech_copy,
            "length_extrapolation": extrap,
            "needle_mqar_zero_shot": mqar,
            "elapsed_s": round(time.perf_counter() - mt0, 2),
        }
        report["total_elapsed_s"] = round(time.perf_counter() - t0, 2)
        OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  [saved] {run_name} -> {OUT_JSON}", flush=True)
        del model

    print(f"[U-01] Complete in {report['total_elapsed_s']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
