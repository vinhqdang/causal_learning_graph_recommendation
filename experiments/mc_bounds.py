"""How loose are the variance and concentration bounds of Theorem 6?

Small synthetic MNAR model (all propensities >= tau, so no clipping bias).
For every (u, i) we compute the walk masses Lambda_e(u, i) of the corrected
3-hop estimator by enumeration and compare
  (b) Var bound  (1 + 1/tau)^2 / tau * sum_e eps_e^2 Lambda_e^2
      with the Monte-Carlo variance of T_hat, and
  (d) the McDiarmid half-width t at level 0.05 with the empirical 95% quantile
      of |T_hat - E T_hat|.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drup.propagation import edge_estimate, three_hop  # noqa: E402

g = torch.Generator().manual_seed(3)
dt = torch.float64
m, n, reps = 12, 15, 20000
U, V = torch.randn(m, 3, generator=g, dtype=dt), torch.randn(n, 3, generator=g, dtype=dt)
prob = torch.sigmoid(U @ V.T - 0.5)
Y = (prob > torch.rand(m, n, generator=g, dtype=dt)).to(dt)
out = {}
for tau in (0.3, 0.1):
    P = (tau + (0.9 - tau) * torch.rand(m, n, generator=g, dtype=dt) ** 2)
    Yhat = (prob + 0.2 * torch.randn(m, n, generator=g, dtype=dt)).clamp(0, 1)
    C = Yhat.sum(1).pow(-0.5)[:, None] * Yhat.sum(0).pow(-0.5)[None, :]
    eps = (Y - Yhat).abs()
    B = max(1.0, 1.0 / tau)
    vb, tb = torch.zeros(m, n, dtype=dt), torch.zeros(m, n, dtype=dt)
    for u in range(m):
        for i in range(n):
            lam = torch.zeros(m, n, dtype=dt)
            for j in range(n):
                for v in range(m):
                    w = C[u, j] * C[v, j] * C[v, i]
                    for e in {(u, j), (v, j), (v, i)}:
                        lam[e] += w
            vb[u, i] = (1 + 1 / tau) ** 2 / tau * (eps ** 2 * lam ** 2).sum()
            ce = eps * B ** 2 * lam / tau
            tb[u, i] = torch.sqrt(0.5 * (ce ** 2).sum() * np.log(2 / 0.05))
    samples = torch.zeros(reps, m, n, dtype=dt)
    for r in range(reps):
        O = (torch.rand(m, n, generator=g, dtype=dt) < P).to(dt)
        samples[r] = three_hop(edge_estimate(O, Y, P, Yhat), C, correct=True)
    var = samples.var(0)
    dev = (samples - samples.mean(0)).abs()
    q95 = torch.quantile(dev.reshape(reps, -1), 0.95, dim=0).reshape(m, n)
    out[str(tau)] = {"var_bound_over_mc_median": float((vb / var).median()),
                     "var_bound_over_mc_min": float((vb / var).min()),
                     "t_bound_over_q95_median": float((tb / q95).median()),
                     "t_bound_over_q95_min": float((tb / q95).min())}
    print(tau, out[str(tau)], flush=True)
json.dump(out, open("results/v2/mc_bounds.json", "w"), indent=1)
