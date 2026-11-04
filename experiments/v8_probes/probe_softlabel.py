"""Probe 4: posterior soft labels for unexposed pairs in iALS (first E-step of an exposure EM).

Q_ui = P(Y=1 | O=0, u, i) under a MAR exposure model: Q = pi (1 - p) / (1 - p pi), with pi the
exposure-conditional additive prior (no IPS) and p the logistic popularity exposure model (log only).
min sum c (t - x_u^T y_i)^2: positives c = 1 + alpha, t = 1; exposed negatives c = 1, t = 0;
unexposed c = 1, t = kappa * Q (kappa = 0 is iALS on the same grid).
"""
import itertools
import sys
import time

import torch

sys.path.insert(0, "/tmp/claude-0/-home-user-causal-learning-graph-recommendation/198950e7-01e5-589f-abc7-a16b1bf09c95/scratchpad/probes")
from harness import Selector, load, report  # noqa: E402
from drup.estimation import clip_propensity, popularity_propensity  # noqa: E402

ds, OUT = sys.argv[1], sys.argv[2]
torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 2)
D = load(ds)
O, Y, pi, rows = D["O"], D["Y"], D["pi"], D["rows"]
P = clip_propensity(popularity_propensity(O.double())[0], 0.05).float()
Q = (pi * (1 - P) / (1 - P * pi).clamp_min(1e-6)).clamp(0, 1) * (1 - O)
pos = O * Y
m, n = O.shape


def solve(Xout, F, C, T, reg, chunk):
    d = F.shape[1]
    eye = reg * torch.eye(d)
    Ft = F.T.contiguous()
    for s in range(0, C.shape[0], chunk):
        c = C[s:s + chunk]
        A = torch.matmul(Ft[None] * c[:, None, :], F) + eye
        b = (c * T[s:s + chunk]) @ F
        Xout[s:s + chunk] = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)


grid = dict(coat=dict(dim=[64], alpha=[30.0, 100.0, 300.0], reg=[100.0, 300.0], kappa=[0.0, 0.5, 1.0, 2.0]),
            yahoo=dict(dim=[64], alpha=[10.0, 30.0], reg=[30.0, 100.0], kappa=[0.0, 0.5, 1.0, 2.0]))[ds]
chunk = 1024
sel = Selector(D, "soft")
base = Selector(D, "iALS same grid")
keys = list(grid)
for vals in itertools.product(*[grid[k] for k in keys]):
    cfg = dict(zip(keys, vals))
    t0 = time.time()
    C = 1.0 + cfg["alpha"] * pos
    T = pos + cfg["kappa"] * Q
    g = torch.Generator().manual_seed(0)
    U = 0.1 * torch.randn(m, cfg["dim"], generator=g)
    V = 0.1 * torch.randn(n, cfg["dim"], generator=g)
    CT, TT = C.T.contiguous(), T.T.contiguous()
    best, bad = -1, 0
    for ep in range(12):
        solve(U, V, C, T, cfg["reg"], chunk)
        solve(V, U, CT, TT, cfg["reg"], chunk)
        S = U[rows] @ V.T
        v = sel.add(S, dict(cfg, epoch=ep))
        if cfg["kappa"] == 0.0:
            base.add(S, dict(cfg, epoch=ep))
        if v > best:
            best, bad = v, 0
        else:
            bad += 1
            if bad >= 3:
                break
    print(f"  {cfg} ep{ep} val {best:.4f} [{time.time()-t0:.0f}s]", flush=True)
report(D, "softlabel", sel.result(), OUT)
report(D, "iALS_same_grid_soft", base.result(), OUT)
