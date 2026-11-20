"""DEMOS, minimal faithful form: particles with position z, mass m and a fixed arrival time.

Store dynamics (between injections) are the gradient flow of
  E = sum_i m_i V(z_i) + 1/2 sum_{i!=j} m_i m_j W(z_i, z_j),   W symmetric
  Psi_i = dE/dm_i,   dz_i/dt = -(1/m_i) dE/dz_i = -grad Psi_i,   dm_i/dt = -alpha m_i (Psi_i - kappa)
with both derivatives taken by autograd of the implemented E, so E - kappa*sum(m) is the
Lyapunov function of the continuous flow by construction.

Documented approximations (see PROTOCOL.md):
  A1 statements are injected a block at a time, not one per token;
  A2 after an injection only new-old and new-new pair terms are evaluated (old-old assumed settled);
  A3 queries are transient probe particles moving in the field of a frozen store;
  A4 the conservative term is a learned antisymmetric linear map applied between hops;
  A5 particles are pruned (m < eps) only at inference; in training mass multiplies every interaction.
No softmax or other normalisation over particles is used anywhere.
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
from bindings import SET, DEL, NULL


class Demos(nn.Module):
    def __init__(self, symbols, d=64, heads=4, proj=16, store_steps=2, probe_steps=3,
                 m_max=2.0, symmetric_only=False, mass_dynamics=True, move=True, hop_mode="euler"):
        super().__init__()
        self.d, self.heads, self.proj = d, heads, proj
        self.store_steps, self.probe_steps, self.m_max = store_steps, probe_steps, m_max
        self.mass_dynamics, self.move, self.hop_mode = mass_dynamics, move, hop_mode
        self.hop_lin = nn.Linear(d, d)                                 # used only by hop_mode='linear' (non-conservative)
        self.sym = nn.Embedding(symbols + 1, d)
        self.op = nn.Embedding(5, d)
        self.enc = nn.Sequential(nn.Linear(3 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.mass0 = nn.Linear(3 * d, 1)
        self.qenc = nn.Linear(d, d)
        # store-store interaction: symmetric, function of C_r [z_i - z_j ; t_i - t_j]
        self.C = nn.Parameter(torch.randn(heads, proj, d + 1) / d ** 0.5)
        self.c = nn.Parameter(torch.zeros(heads))                      # signed strengths
        self.V = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        # probe-store wells (two species): -softplus(a_r) exp(-|A_r z_q - B_r z_j|^2 / 2)
        self.A = nn.Parameter(torch.randn(heads, proj, d) / d ** 0.5)
        self.B = nn.Parameter(torch.randn(heads, proj, d) / d ** 0.5)
        self.a = nn.Parameter(torch.zeros(heads))
        self.J = nn.Parameter(torch.zeros(d, d))                       # antisymmetrised in use
        self.log_eta = nn.Parameter(torch.tensor(0.0))
        self.log_eta_q = nn.Parameter(torch.tensor(0.0))
        self.log_alpha = nn.Parameter(torch.tensor(-1.0))
        self.kappa = nn.Parameter(torch.tensor(0.0))
        self.out = nn.Linear(d, d)
        self.null = nn.Sequential(nn.Linear(d + 1, d), nn.GELU(), nn.Linear(d, 1))
        self.pair_evals = 0                                            # secondary cost counter

    # ---- energies -------------------------------------------------------------------
    def _pair(self, za, ta, zb, tb):
        """Symmetric store-store W for all pairs: [B, na, nb]."""
        x = torch.cat([za, ta.unsqueeze(-1)], -1)
        y = torch.cat([zb, tb.unsqueeze(-1)], -1)
        px = torch.einsum("hpd,bnd->bhnp", self.C, x)
        py = torch.einsum("hpd,bnd->bhnp", self.C, y)
        d2 = (px.unsqueeze(3) - py.unsqueeze(2)).pow(2).sum(-1)       # [B,h,na,nb]
        self.pair_evals += d2[:, 0].numel()
        return (self.c.view(1, -1, 1, 1) * torch.exp(-0.5 * d2)).sum(1)

    def energy(self, z, m, t, new):
        """E with pair terms restricted to pairs that involve a new particle (A2)."""
        e = (m * self.V(z).squeeze(-1)).sum(-1)
        zn, mn, tn = z[:, new:], m[:, new:], t[:, new:]
        w = self._pair(zn, tn, z, t)                                   # new vs all
        mm = mn.unsqueeze(2) * m.unsqueeze(1)
        nn_ = z.shape[1] - new
        eye = torch.zeros(nn_, z.shape[1], device=z.device)
        eye[torch.arange(nn_), new + torch.arange(nn_)] = 1.0
        mm = mm * (1 - eye)                                            # no self term
        half = torch.ones(nn_, z.shape[1], device=z.device); half[:, new:] = 0.5   # new-new counted twice
        return e + (mm * w * half).sum((1, 2))

    def flow(self, z, m, t, new, record=None):
        eta, alpha = self.log_eta.exp(), self.log_alpha.exp()
        for _ in range(self.store_steps):
            E = self.energy(z, m, t, new)
            gz, gm = torch.autograd.grad(E.sum(), [z, m], create_graph=self.training)
            if record is not None:
                record.append((E.detach(), m.detach().sum(-1)))
            if self.move:
                z = z - eta * gz / m.clamp_min(1e-3).unsqueeze(-1)
            if self.mass_dynamics:
                m = (m * torch.exp(-alpha * (gm - self.kappa))).clamp(max=self.m_max)
        return z, m

    def probe(self, zq, z, m, hops):
        """zq [B,Q,d]; hops [B,Q]. Probes move in the frozen store field; J acts between hops."""
        eta = self.log_eta_q.exp()
        J = self.J - self.J.T
        pz = torch.einsum("hpd,bnd->bhnp", self.B, z)
        depth = torch.zeros(zq.shape[:2], device=zq.device)
        for hop in range(int(hops.max())):
            active = (hops > hop).float().unsqueeze(-1)
            if hop > 0:
                if self.hop_mode == "euler":      # first-order step of the conservative term (A4)
                    nxt = zq + zq @ J.T
                elif self.hop_mode == "expm":     # exact rotation generated by the same antisymmetric J
                    nxt = zq @ torch.linalg.matrix_exp(J).T
                else:                             # 'linear': unconstrained map, can erase (not conservative)
                    nxt = self.hop_lin(zq)
                zq = active * nxt + (1 - active) * zq
            for _ in range(self.probe_steps):
                pq = torch.einsum("hpd,bqd->bhqp", self.A, zq)
                d2 = (pq.unsqueeze(3) - pz.unsqueeze(2)).pow(2).sum(-1)          # [B,h,Q,n]
                self.pair_evals += d2[:, 0].numel()
                well = F.softplus(self.a).view(1, -1, 1, 1) * torch.exp(-0.5 * d2) * m.view(m.shape[0], 1, 1, -1)
                # force = -dE/dzq with E = -sum well  ->  pulls A zq toward B z_j
                diff = pz.unsqueeze(2) - pq.unsqueeze(3)                         # [B,h,Q,n,p]
                force = torch.einsum("bhqn,bhqnp,hpd->bqd", well, diff, self.A)
                zq = zq + active * eta * force
                depth = well.sum((1, 3))
        return zq, depth

    # ---- forward --------------------------------------------------------------------
    def forward(self, batch, prune_eps=None, record=None):
        """batch: st [B,G,Ns,3] (op,k,v), q [B,G,Q,2] (key,hops). Returns logits [B,G,Q,S+1]."""
        st, q = batch["st"], batch["q"]
        Bn, G, Ns, _ = st.shape
        z = torch.zeros(Bn, 0, self.d, device=st.device); m = torch.zeros(Bn, 0, device=st.device)
        t = torch.zeros(Bn, 0, device=st.device)
        outs, live = [], []
        for g in range(G):
            op, k, v = st[:, g, :, 0], st[:, g, :, 1], st[:, g, :, 2]
            feat = torch.cat([self.op(op), self.sym(k), self.sym(v)], -1)
            zn = self.enc(feat)
            mn = self.m_max * torch.sigmoid(self.mass0(feat).squeeze(-1)) * (op != 0).float()
            tn = (g * Ns + torch.arange(Ns, device=st.device).float()).expand(Bn, Ns) / (G * Ns)
            new = z.shape[1]
            z, m, t = torch.cat([z, zn], 1), torch.cat([m, mn], 1), torch.cat([t, tn], 1)
            if not z.requires_grad:
                z = z.requires_grad_(True); m = m.requires_grad_(True)
            z, m = self.flow(z, m, t, new, record)
            if prune_eps is not None:                                  # A5: inference only, batch of one
                keep = m[0] > prune_eps
                z, m, t = z[:, keep], m[:, keep], t[:, keep]
            live.append(float((m > (prune_eps or 0.05)).float().sum(-1).mean()))
            zq = self.qenc(self.sym(q[:, g, :, 0]))
            zq, depth = self.probe(zq, z, m, q[:, g, :, 1])
            sym_logits = self.out(zq) @ self.sym.weight[1:].T
            null_logit = self.null(torch.cat([zq, depth.unsqueeze(-1)], -1))
            outs.append(torch.cat([null_logit, sym_logits], -1))
        self.last_live = live
        return torch.stack(outs, 1)
