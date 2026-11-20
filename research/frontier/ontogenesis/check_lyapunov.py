"""Independent numerical check of the DEMOS stability statement on the implemented energy.

Claim: for the continuous flow dz = -grad Psi, dm = -alpha m (Psi - kappa), with W symmetric,
L = E - kappa * sum(m) is non-increasing, while E alone need not be.
Uses the model's own energy and autograd; small explicit steps approximate continuous time.
"""
import torch
from demos import Demos
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
worst_L, E_up = 0.0, 0
for trial in range(20):
    model = Demos(symbols=64, d=16, heads=3, proj=8).double()
    with torch.no_grad():
        model.c.normal_(); model.kappa.fill_(float(torch.randn(())))
        if trial % 2:                                # repulsive pairs and a high price: 0 < Psi < kappa is reachable
            model.c.abs_(); model.kappa.fill_(5.0)
    n = 12
    z = torch.randn(1, n, 16, requires_grad=True); m = (torch.rand(1, n) + 0.2).requires_grad_(True)
    t = torch.rand(1, n)
    dt, alpha, kappa = 1e-3, 0.5, model.kappa.detach()
    prev_L = prev_E = None
    for step in range(400):
        E = model.energy(z, m, t, new=0)            # new=0: every pair is evaluated
        gz, gm = torch.autograd.grad(E.sum(), [z, m])
        L = (E - kappa * m.sum(-1)).item()
        if prev_L is not None:
            worst_L = max(worst_L, L - prev_L); E_up += int(E.item() > prev_E + 1e-12)
        prev_L, prev_E = L, E.item()
        z = (z - dt * gz / m.unsqueeze(-1)).detach().requires_grad_(True)
        m = (m - dt * alpha * m * (gm - kappa)).detach().requires_grad_(True)
print(f"largest single-step increase of E - kappa*M over 20 systems x 400 steps: {worst_L:.3e}")
print(f"steps on which E alone increased: {E_up} of {20*399}")
