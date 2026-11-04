"""Probe 2: three-state iALS with the exposure-conditional prior offset.

min sum_ui c_ui (t_ui - beta z_ui - x_u^T y_i)^2 + reg (|X|^2 + |Y|^2)
c = 1 + alpha (logged positive), w_n (exposed negative), 1 (unexposed)
t = 1 (positive), tau_n (exposed negative), 0 (unexposed)
w_n = 1, tau_n = 0 is iALS-p (iALS + prior) exactly. Dense exact ALS (batched GEMM).
"""
import itertools
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/tmp/claude-0/-home-user-causal-learning-graph-recommendation/198950e7-01e5-589f-abc7-a16b1bf09c95/scratchpad/probes")
from harness import Selector, load, report  # noqa: E402

ds, OUT = sys.argv[1], sys.argv[2]
tag = sys.argv[3] if len(sys.argv) > 3 else "ts"
torch.set_num_threads(int(sys.argv[4]) if len(sys.argv) > 4 else 4)
D = load(ds)
O, Y, Z, rows = D["O"], D["Y"], D["Z"], D["rows"]
pos = O * Y
neg = O * (1 - Y)
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


if ds == "coat":
    grid = dict(dim=[32, 64], alpha=[30.0, 100.0, 300.0], reg=[100.0, 300.0, 1000.0], beta=[0.5, 1.0],
                wn=[0.25, 1.0, 4.0], taun=[0.0, 0.3])
    chunk, maxep = 512, 15
elif ds == "yahoo":
    grid = dict(dim=[64, 128], alpha=[3.0, 10.0, 30.0, 100.0], reg=[10.0, 100.0, 1000.0], beta=[0.0, 0.5],
                wn=[0.25, 1.0, 4.0, 16.0], taun=[0.0])
    chunk, maxep = 1024, 12
elif ds == "kuairand":
    grid = dict(dim=[64], alpha=[3.0, 10.0, 30.0], reg=[100.0, 1000.0], beta=[0.0, 0.5],
                wn=[1.0, 4.0, 16.0], taun=[0.0])
    chunk, maxep = 64, 10
else:
    grid = dict(dim=[64], alpha=[10.0], reg=[300.0, 1000.0, 3000.0], beta=[0.5, 1.0], wn=[0.25, 1.0, 4.0, 16.0],
                taun=[0.0])
    chunk, maxep = 64, 8
if tag != "ts":     # restricted grid passed as python dict literal
    grid.update(eval(tag))
sel = Selector(D, "tsials")
sub = {"iALS(same grid)": Selector(D, "a"), "iALS+prior(same grid)": Selector(D, "b")}
keys = list(grid)
for vals in itertools.product(*[grid[k] for k in keys]):
    cfg = dict(zip(keys, vals))
    t0 = time.time()
    C = (1.0 + cfg["alpha"] * pos + (cfg["wn"] - 1.0) * neg)
    T = pos + cfg["taun"] * neg - cfg["beta"] * Z
    g = torch.Generator().manual_seed(0)
    U = 0.1 * torch.randn(m, cfg["dim"], generator=g)
    V = 0.1 * torch.randn(n, cfg["dim"], generator=g)
    CT, TT = C.T.contiguous(), T.T.contiguous()
    best, bad = -1, 0
    for ep in range(maxep):
        solve(U, V, C, T, cfg["reg"], chunk)
        solve(V, U, CT, TT, cfg["reg"], chunk)
        S = U[rows] @ V.T + cfg["beta"] * Z[rows]
        v = sel.add(S, dict(cfg, epoch=ep))
        if cfg["wn"] == 1.0 and cfg["taun"] == 0.0:
            sub["iALS+prior(same grid)"].add(S, dict(cfg, epoch=ep))
            if cfg["beta"] == 0.0:
                sub["iALS(same grid)"].add(S, dict(cfg, epoch=ep))
        if v > best:
            best, bad = v, 0
        else:
            bad += 1
            if bad >= 3:
                break
    del C, T, CT, TT
    print(f"  {cfg} ep{ep} val {best:.4f} [{time.time()-t0:.0f}s]", flush=True)
report(D, f"tsials_{tag[:20]}", sel.result(), OUT)
for k, sv in sub.items():
    if sv.n:
        report(D, k.replace(" ", "_").replace("(", "").replace(")", ""), sv.result(), OUT)
