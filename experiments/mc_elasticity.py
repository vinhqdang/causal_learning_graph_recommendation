"""Monte-Carlo check of exposure invariance (Theorem 4).

Half of the items are "treated": their exposure propensities are multiplied
by pi in (0, 1]. For each estimator we estimate the exposure elasticity
    e = d log E[s_ui] / d log pi
of the expected 3-hop score of treated candidate items (u, i unexposed),
by finite differences between pi = 1 and pi = 0.5.
Theory: Obs  e >= 1 (popularity amplification),
        IPS / DR adjacency without walk correction: e < 0 (over-correction),
        IPS+WC / DRUP: e = 0 (exposure invariance).
"""

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drup.propagation import edge_estimate, three_hop  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=60)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/mc_elasticity.json")
    a = ap.parse_args()
    g = torch.Generator().manual_seed(a.seed)
    dt = torch.float64
    m, n = a.m, a.n
    U = torch.randn(m, 3, generator=g, dtype=dt)
    V = torch.randn(n, 3, generator=g, dtype=dt)
    prob = torch.sigmoid(U @ V.T - 0.5)
    Y = (prob > torch.rand(m, n, generator=g, dtype=dt)).to(dt)
    pop = torch.rand(n, generator=g, dtype=dt) ** 2
    act = torch.rand(m, generator=g, dtype=dt)
    P0 = (0.1 + 0.8 * act[:, None] * pop[None, :]).clamp(0.05, 0.95)
    Yhat = (prob + 0.2 * torch.randn(m, n, generator=g, dtype=dt)).clamp(0, 1)
    du, di = Yhat.sum(1), Yhat.sum(0)
    C = du.pow(-0.5)[:, None] * di.pow(-0.5)[None, :]   # exposure-independent
    treated = torch.arange(n) % 2 == 0

    names = ["Obs", "IPS", "IPS+WC", "DR", "DRUP"]

    def mean_scores(pi):
        P = torch.where(treated[None, :], P0 * pi, P0)
        acc = {k: torch.zeros(m, n, dtype=dt) for k in names}
        cnt = torch.zeros(m, n, dtype=dt)
        for _ in range(a.reps):
            O = (torch.rand(m, n, generator=g, dtype=dt) < P).to(dt)
            unexp = 1.0 - O   # condition on the candidate being unexposed
            W_ips = edge_estimate(O, Y, P)
            W_dr = edge_estimate(O, Y, P, Yhat)
            ests = {
                "Obs": three_hop(O * Y, C, correct=False) + 0 * C,
                "IPS": three_hop(W_ips, C, correct=False),
                "IPS+WC": three_hop(W_ips, C, correct=True),
                "DR": three_hop(W_dr, C, correct=False),
                "DRUP": three_hop(W_dr, C, correct=True),
            }
            for k, v in ests.items():
                acc[k] += v * unexp
            cnt += unexp
        return {k: acc[k] / cnt.clamp_min(1) for k in names}

    e1 = mean_scores(1.0)
    e05 = mean_scores(0.5)
    out = {}
    for k in names:
        num = e1[k][:, treated].abs().mean()
        den = e05[k][:, treated].abs().mean()
        # control items: their propensities are unchanged
        ctl = e05[k][:, ~treated].abs().mean() / e1[k][:, ~treated].abs().mean()
        el = (math.log(num) - math.log(den)) / math.log(2.0)
        out[k] = {"elasticity_treated": el, "control_ratio": ctl.item()}
        print(f"{k:7s} elasticity(treated) = {el:+.3f}   control E-ratio = {ctl.item():.3f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"config": vars(a), "results": out}, f, indent=2)


if __name__ == "__main__":
    main()
