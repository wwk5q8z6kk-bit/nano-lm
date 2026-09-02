# Stage 2 — supervised fine-tuning per Recipe ③ §Stage 2 and sft.md.
# Trains policy (init from pretrain/ckpt.pt) on (sft_x.npy, sft_y.npy), optimizing only the completion tokens.
# Resizes embedding to 4098 for <|im_start|> (4096) and <|im_end|> (4097) per §Chat tokens.
import sys
import os

if __name__ == '__main__':
    import math, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
    import model_nano as M

    LR = 3e-4         # vault target (standard ~1e-5 DPO/RL is much lower)
    B, EPOCHS = 32, 3
    BASE = "../pretrain/ckpt.pt"
    dev = M.dev
    V, d = M.V, M.d

    base = torch.load(BASE, map_location="cpu", weights_only=True)
    old_emb = base["emb.weight"]                            # (4096, 192)
    # the exact token logic: old_emb gets exactly old_emb, new rows get old_emb.mean(0)
    new_emb = torch.zeros(V, d)
    new_emb[:4096] = old_emb
    new_emb[4096:] = old_emb.mean(dim=0, keepdim=True)
    base["emb.weight"] = new_emb

    m = M.GPT(); m.load_state_dict(base); m.to(dev)
    print(f"loaded base, resized emb {old_emb.shape} -> {new_emb.shape} (new rows <- <|endoftext|>); params={sum(p.numel() for p in m.parameters())/1e6:.2f}M")

    X = torch.tensor(np.load("sft_x.npy").astype(np.int64))
    Y = torch.tensor(np.load("sft_y.npy").astype(np.int64))
    N = X.shape[0]
    steps = (N * EPOCHS) // B
    print(f"{N} examples, {steps} steps, {EPOCHS} epochs, batch {B}", flush=True)

    opt = torch.optim.AdamW(m.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.0)

    t0 = time.time(); m.train()
    for step in range(1, steps + 1):
        idx = torch.randint(0, N, (B,))
        x, y = X[idx].to(dev), Y[idx].to(dev)
        logits = m(x)                                       # (B, S, V)
        loss = F.cross_entropy(logits.view(-1, V), y.view(-1), ignore_index=-1)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step % 50 == 0:
            print(f"step {step:4d}/{steps}  loss {loss.item():.4f}  {(time.time()-t0)/60:.1f}min", flush=True)

    torch.save(m.state_dict(), "sft.pt")
    print(f"SFT done in {(time.time()-t0)/60:.1f} min -> sft.pt", flush=True)

