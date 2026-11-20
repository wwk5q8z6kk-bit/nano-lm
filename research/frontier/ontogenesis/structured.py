"""Two more arms on the block interface used by DEMOS (batch: st [B,G,Ns,3], q [B,G,Q,2]).

GraphMP   weight-tied message passing over statement nodes with a learned forgetting gate.
          This is the conventional model DEMOS reduces to if its constraints add nothing:
          softmax-normalised messages, free (non-gradient) updates, no energy.
KeyStore  exact keyed memory with latest-write-wins.
          mode="token":   key = the key symbol itself (an oracle database; no key learning)
          mode="learned": key = G learned discrete codes of the key embedding (straight-through)
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F


class GraphMP(nn.Module):
    def __init__(self, symbols, d=64, heads=4, store_steps=2, probe_steps=3):
        super().__init__()
        self.d, self.h, self.store_steps, self.probe_steps = d, heads, store_steps, probe_steps
        self.sym, self.op = nn.Embedding(symbols + 1, d), nn.Embedding(5, d)
        self.enc = nn.Sequential(nn.Linear(3 * d + 1, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.gate = nn.Linear(d, 1)
        self.qenc = nn.Linear(d, d)
        self.qs, self.ks, self.vs = (nn.Linear(d, d, bias=False) for _ in range(3))   # store<-store
        self.qp, self.kp, self.vp = (nn.Linear(d, d, bias=False) for _ in range(3))   # probe<-store
        self.fs = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.fp = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.hop = nn.Linear(d, d)
        self.ns, self.np_ = nn.LayerNorm(d), nn.LayerNorm(d)
        self.out = nn.Linear(d, d); self.null = nn.Linear(d, 1)
        self.pair_evals = 0

    def _attend(self, q, k, v, x_from, x_to, logg, mask=None):
        B, n, D = x_from.shape; m = x_to.shape[1]; H = self.h
        qq = q(x_to).view(B, m, H, D // H).transpose(1, 2)
        kk = k(x_from).view(B, n, H, D // H).transpose(1, 2)
        vv = v(x_from).view(B, n, H, D // H).transpose(1, 2)
        s = qq @ kk.transpose(-1, -2) / (D // H) ** 0.5 + logg.view(B, 1, 1, n)
        if mask is not None: s = s.masked_fill(mask, -1e9)
        self.pair_evals += B * m * n
        return (s.softmax(-1) @ vv).transpose(1, 2).reshape(B, m, D)

    def forward(self, batch, prune_eps=None, record=None):
        st, q = batch["st"], batch["q"]
        Bn, G, Ns, _ = st.shape
        x = torch.zeros(Bn, 0, self.d, device=st.device); pad = torch.zeros(Bn, 0, dtype=torch.bool, device=st.device)
        outs, live = [], []
        for g in range(G):
            op, k, v = st[:, g, :, 0], st[:, g, :, 1], st[:, g, :, 2]
            t = (g * Ns + torch.arange(Ns, device=st.device).float()).expand(Bn, Ns) / (G * Ns)
            xn = self.enc(torch.cat([self.op(op), self.sym(k), self.sym(v), t.unsqueeze(-1)], -1))
            x, pad = torch.cat([x, xn], 1), torch.cat([pad, op == 0], 1)
            for _ in range(self.store_steps):
                logg = F.logsigmoid(self.gate(x)).squeeze(-1).masked_fill(pad, -1e9)
                msg = self._attend(self.qs, self.ks, self.vs, self.ns(x), self.ns(x), logg)
                x = x + self.fs(torch.cat([x, msg], -1))
            logg = F.logsigmoid(self.gate(x)).squeeze(-1).masked_fill(pad, -1e9)
            if prune_eps is not None:
                keep = logg[0].exp() > prune_eps
                x, pad, logg = x[:, keep], pad[:, keep], logg[:, keep]
            live.append(float((logg.exp() > (prune_eps or 0.05)).float().sum(-1).mean()))
            zq = self.qenc(self.sym(q[:, g, :, 0])); hops = q[:, g, :, 1]
            for hop in range(int(hops.max())):
                active = (hops > hop).float().unsqueeze(-1)
                if hop > 0: zq = zq + active * self.hop(zq)
                for _ in range(self.probe_steps):
                    msg = self._attend(self.qp, self.kp, self.vp, self.ns(x), self.np_(zq), logg)
                    zq = zq + active * self.fp(torch.cat([zq, msg], -1))
            outs.append(torch.cat([self.null(zq), self.out(zq) @ self.sym.weight[1:].T], -1))
        self.last_live = live
        return torch.stack(outs, 1)


class KeyStore(nn.Module):
    def __init__(self, symbols, d=64, mode="token", groups=2, codes=64):
        super().__init__()
        self.d, self.mode, self.G, self.C, self.S = d, mode, groups, codes, symbols
        self.sym, self.op = nn.Embedding(symbols + 1, d), nn.Embedding(5, d)
        self.val = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.wkey = nn.Linear(d, groups * codes)       # write key from the key symbol embedding
        self.qkey = nn.Linear(d, groups * codes)       # first-hop query key
        self.nkey = nn.Linear(d, groups * codes)       # next-hop key from a retrieved payload
        self.tok_next = nn.Linear(d, symbols + 1)      # token mode: next key symbol from payload
        self.absent = nn.Parameter(torch.zeros(d))
        self.out = nn.Linear(d, d); self.null = nn.Linear(d + 1, 1)
        self.pair_evals = 0                            # counts table lookups, not pairs

    def _codes(self, logits):
        """Straight-through one-hot codes [.., G, C]."""
        lg = logits.view(*logits.shape[:-1], self.G, self.C)
        p = lg.softmax(-1)
        hard = F.one_hot(p.argmax(-1), self.C).float()
        return hard + p - p.detach()

    def _lookup(self, qcode, wcode, wtime, wvalid):
        """Latest valid write whose codes equal the query codes. qcode [B,Q,G,C], wcode [B,n,G,C]."""
        match = torch.einsum("bqgc,bngc->bqgn", qcode, wcode).prod(2)              # [B,Q,n], exactly 0/1 forward
        m = match * wvalid.unsqueeze(1)
        score = m.detach() * (1.0 + wtime).unsqueeze(1)
        j = score.argmax(-1)                                                       # latest matching write
        found = torch.gather(m, 2, j.unsqueeze(-1)).squeeze(-1)                    # 1 if any match (carries ST gradient)
        self.pair_evals += qcode.shape[0] * qcode.shape[1]
        return j, found

    def forward(self, batch, prune_eps=None, record=None):
        st, q = batch["st"], batch["q"]
        Bn, G, Ns, _ = st.shape
        op, k, v = st[..., 0].flatten(1), st[..., 1].flatten(1), st[..., 2].flatten(1)   # [B, G*Ns]
        n = op.shape[1]
        time = torch.arange(n, device=st.device).float().expand(Bn, n) / n
        payload = self.val(torch.cat([self.op(op), self.sym(v)], -1))                    # DEL carries its own op embedding
        if self.mode == "token":
            wcode = F.one_hot(k, self.S + 1).float().unsqueeze(2)                        # [B,n,1,S+1]
        else:
            wcode = self._codes(self.wkey(self.sym(k)))
        outs = []
        for g in range(G):
            valid = ((torch.arange(n, device=st.device) < (g + 1) * Ns).float().expand(Bn, n)) * (op != 0).float()
            key, hops = q[:, g, :, 0], q[:, g, :, 1]
            if self.mode == "token": qc = F.one_hot(key, self.S + 1).float().unsqueeze(2)
            else: qc = self._codes(self.qkey(self.sym(key)))
            cur = self.absent.expand(Bn, key.shape[1], self.d); ok = torch.ones_like(key, dtype=torch.float)
            for hop in range(int(hops.max())):
                active = (hops > hop).float()
                if hop > 0:
                    if self.mode == "token":
                        lg = self.tok_next(cur); p = lg.softmax(-1)
                        qc = (F.one_hot(p.argmax(-1), self.S + 1).float() + p - p.detach()).unsqueeze(2)
                    else: qc = self._codes(self.nkey(cur))
                j, found = self._lookup(qc, wcode, time, valid)
                got = torch.gather(payload, 1, j.unsqueeze(-1).expand(-1, -1, self.d))
                is_del = torch.gather(op, 1, j) == 2
                hit = found * (~is_del).float()
                new = hit.unsqueeze(-1) * got + (1 - hit.unsqueeze(-1)) * self.absent
                cur = torch.where(active.unsqueeze(-1) > 0, new, cur)
                ok = torch.where(active > 0, ok * hit, ok)
            null_logit = self.null(torch.cat([cur, ok.unsqueeze(-1)], -1))
            outs.append(torch.cat([null_logit, self.out(cur) @ self.sym.weight[1:].T], -1))
        self.last_live = None
        return torch.stack(outs, 1)
