"""Monte-Carlo check of the walk-correction theorem.

For a fixed potential-outcome graph Y, propensities P, imputation Yhat and
edge weights C, we resample exposures O ~ Bernoulli(P) many times and compare
the empirical mean of each 3-hop estimator with the oracle T* computed on Y.

Estimators
    naive-obs : propagation over the observed graph O*Y (standard LightGCN)
    IPS       : inverse-propensity adjacency (NAVIP-style), no walk correction
    IPS+WC    : IPS adjacency with the idempotent walk correction
    DR        : doubly robust adjacency, no walk correction
    DR+WC     : doubly robust adjacency with walk correction (DRUP)
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drup.propagation import edge_estimate, three_hop  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=30)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--reps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/mc_unbiasedness.json")
    a = ap.parse_args()

    g = torch.Generator().manual_seed(a.seed)
    dt = torch.float64
    m, n = a.m, a.n
    # Low-rank preference structure plus popularity-driven exposure (MNAR).
    U = torch.randn(m, 3, generator=g, dtype=dt)
    V = torch.randn(n, 3, generator=g, dtype=dt)
    Y = (torch.sigmoid(U @ V.T - 0.5) > torch.rand(m, n, generator=g, dtype=dt)).to(dt)
    pop = torch.rand(n, generator=g, dtype=dt) ** 2
    act = torch.rand(m, generator=g, dtype=dt)
    P = (0.05 + 0.9 * act[:, None] * pop[None, :]) * (0.5 + 0.5 * Y)
    P = P.clamp(0.03, 0.95)
    Yhat = (torch.sigmoid(U @ V.T - 0.5) + 0.2 * torch.randn(m, n, generator=g, dtype=dt)).clamp(0, 1)
    # Fixed normalisation computed from the oracle graph.
    du = Y.sum(1).clamp_min(1)
    di = Y.sum(0).clamp_min(1)
    C = du.pow(-0.5)[:, None] * di.pow(-0.5)[None, :]

    target = three_hop(Y, C, correct=False)  # Y is binary, so no repeats bias
    names = ["naive-obs", "IPS", "IPS+WC", "DR", "DR+WC"]
    sums = {k: torch.zeros(m, n, dtype=dt) for k in names}
    sq = {k: torch.zeros(m, n, dtype=dt) for k in names}
    for _ in range(a.reps):
        O = (torch.rand(m, n, generator=g, dtype=dt) < P).to(dt)
        ests = {
            "naive-obs": three_hop(O * Y, C, correct=False),
            "IPS": three_hop(edge_estimate(O, Y, P), C, correct=False),
            "IPS+WC": three_hop(edge_estimate(O, Y, P), C, correct=True),
            "DR": three_hop(edge_estimate(O, Y, P, Yhat), C, correct=False),
            "DR+WC": three_hop(edge_estimate(O, Y, P, Yhat), C, correct=True),
        }
        for k, v in ests.items():
            sums[k] += v
            sq[k] += v * v
    out = {}
    scale = target.abs().mean().item()
    for k in names:
        mean = sums[k] / a.reps
        var = sq[k] / a.reps - mean ** 2
        se = (var / a.reps).sqrt()
        bias = mean - target
        out[k] = {
            "rel_abs_bias": (bias.abs().mean() / scale).item(),
            "max_abs_z": (bias.abs() / se.clamp_min(1e-12)).max().item(),
            "frac_|z|>3": ((bias.abs() / se.clamp_min(1e-12)) > 3).double().mean().item(),
            "rel_rmse": ((bias ** 2 + var).mean().sqrt() / scale).item(),
        }
        print(f"{k:10s} rel|bias|={out[k]['rel_abs_bias']:.4f}  "
              f"frac|z|>3={out[k]['frac_|z|>3']:.4f}  rel-RMSE={out[k]['rel_rmse']:.3f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"config": vars(a), "results": out}, f, indent=2)


if __name__ == "__main__":
    main()
