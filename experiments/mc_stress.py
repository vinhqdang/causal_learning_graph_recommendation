"""Monte-Carlo stress tests beyond the assumptions of Theorem 1.

Scenarios (3-hop unless stated):
  indep     independent Bernoulli exposures with outcome-dependent
            propensities (the setting of Theorem 1, as reference)
  fixedrow  every user is exposed to exactly k_u items, drawn without
            replacement with probabilities proportional to p (fixed-size
            slates, as in Coat). Propensities are the true inclusion
            probabilities, estimated by a separate pilot run.
  misprop   independent exposures, but the estimator uses a misspecified
            propensity p_hat = p * exp(0.5 z); with an accurate imputation
            (DR is still unbiased, IPS is not) and with a noisy one.
For every estimator we report the relative bias against the full-exposure
target, both over all entries and conditionally on unexposed candidates
(O_ui = 0, where the target is F* with Y_ui replaced by Yhat_ui for DR and
by 0 for IPS, Corollary 2), and the within-user ranking agreement (Kendall
tau) with the target on unexposed candidates.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drup.khop import khop  # noqa: E402
from drup.propagation import edge_estimate  # noqa: E402


def kendall_rows(A, B, mask):
    """Mean Kendall tau between rows of A and B over the entries in mask."""
    taus = []
    for a, b, mk in zip(A, B, mask):
        idx = torch.nonzero(mk).flatten()
        if len(idx) < 3:
            continue
        x, y = a[idx], b[idx]
        dx = torch.sign(x[:, None] - x[None, :])
        dy = torch.sign(y[:, None] - y[None, :])
        k = len(idx)
        taus.append(float((dx * dy).sum() / (k * (k - 1))))
    return float(np.mean(taus)) if taus else float("nan")


def setup(m, n, seed, pscale):
    g = torch.Generator().manual_seed(seed)
    dt = torch.float64
    U = torch.randn(m, 3, generator=g, dtype=dt)
    V = torch.randn(n, 3, generator=g, dtype=dt)
    prob = torch.sigmoid(U @ V.T - 0.5)
    Y = (prob > torch.rand(m, n, generator=g, dtype=dt)).to(dt)
    pop = torch.rand(n, generator=g, dtype=dt) ** 2
    act = torch.rand(m, generator=g, dtype=dt)
    P = ((0.05 + 0.9 * act[:, None] * pop[None, :]) * (0.5 + 0.5 * Y) * pscale).clamp(0.01, 0.95)
    Yhat = (prob + 0.2 * torch.randn(m, n, generator=g, dtype=dt)).clamp(0, 1)
    D = Yhat
    C = D.sum(1).pow(-0.5)[:, None] * D.sum(0).pow(-0.5)[None, :]
    return g, Y, P, Yhat, C


def fixedrow_sampler(P, rng):
    m, n = P.shape
    k = P.sum(1).round().clamp(1, n - 1).long().numpy()
    w = P.numpy()

    def draw():
        O = np.zeros((m, n))
        for u in range(m):
            O[u, rng.choice(n, k[u], replace=False, p=w[u] / w[u].sum())] = 1.0
        return torch.from_numpy(O)
    return draw


def run(a, scenario):
    g, Y, P, Yhat, C = setup(a.m, a.n, a.seed, a.pscale)
    m, n = Y.shape
    K = a.K
    rng = np.random.default_rng(a.seed + 7)
    if scenario == "fixedrow":
        draw = fixedrow_sampler(P, rng)
        pilot = torch.zeros(m, n, dtype=torch.float64)
        for _ in range(a.pilot):
            pilot += draw()
        Ptrue = (pilot / a.pilot).clamp_min(1.0 / a.pilot)
    else:
        def draw():
            return (torch.rand(m, n, generator=g, dtype=torch.float64) < P).to(torch.float64)
        Ptrue = P
    Phat = Ptrue
    Yh = Yhat
    if scenario.startswith("misprop"):
        z = torch.randn(m, n, generator=g, dtype=torch.float64)
        Phat = (Ptrue * torch.exp(0.5 * z)).clamp(0.01, 1.0)
        if scenario == "misprop_oracleY":
            Yh = Y.clone()
    names = ["IPS", "IPS+WC", "DR", "DRUP"]
    target = khop(Y, C, K, correct=False)
    sums = {k: torch.zeros(m, n, dtype=torch.float64) for k in names}
    csum = {k: torch.zeros(m, n, dtype=torch.float64) for k in names}
    cnt = torch.zeros(m, n, dtype=torch.float64)
    ktau = {k: [] for k in names}
    for rep in range(a.reps):
        O = draw()
        Wi = edge_estimate(O, Y, Phat)
        Wd = edge_estimate(O, Y, Phat, Yh)
        est = {"IPS": khop(Wi, C, K, correct=False), "IPS+WC": khop(Wi, C, K, correct=True),
               "DR": khop(Wd, C, K, correct=False), "DRUP": khop(Wd, C, K, correct=True)}
        un = 1.0 - O
        for k, v in est.items():
            sums[k] += v
            csum[k] += v * un
        cnt += un
        if rep < a.rank_reps:
            for k, v in est.items():
                ktau[k].append(kendall_rows(v, target, un > 0))
    scale = target.abs().mean().item()
    # conditional targets on unexposed candidates: Y_ui -> Yhat_ui (DR), -> 0 (IPS),
    # each repeated factor entering once (idempotent reduction, correct=True)
    def cond_target(repl):
        T = torch.zeros(m, n, dtype=torch.float64)
        for u in range(m):
            for i in range(n):
                Yc = Y.clone()
                Yc[u, i] = repl[u, i]
                T[u, i] = khop(Yc, C, K, rows=torch.tensor([u]), correct=True)[0, i]
        return T
    tgt_dr = cond_target(Yh)
    tgt_ips = cond_target(torch.zeros_like(Y))
    out = {}
    ok = cnt > 0
    for k in names:
        mean = sums[k] / a.reps
        cmean = csum[k] / cnt.clamp_min(1)
        ct = tgt_ips if k.startswith("IPS") else tgt_dr
        out[k] = {"rel_bias": float((mean - target).abs().mean() / scale),
                  "rel_bias_unexposed": float((cmean - ct)[ok].abs().mean() / scale),
                  "kendall_tau": float(np.nanmean(ktau[k]))}
    out["_info"] = {"median_p": float(Ptrue.median()), "frac_p<0.05": float((Ptrue < 0.05).double().mean())}
    if scenario == "fixedrow":
        k = Ptrue.sum(1)
        out["_info"]["rho_bound_mean"] = float(((n - k) / (k * (n - 1))).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=30)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--reps", type=int, default=20000)
    ap.add_argument("--rank_reps", type=int, default=2000)
    ap.add_argument("--pilot", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--pscale", type=float, default=1.0)
    ap.add_argument("--scenarios", nargs="+", default=["indep", "fixedrow", "misprop_oracleY", "misprop"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = {}
    for sc in a.scenarios:
        out[sc] = run(a, sc)
        print(sc, json.dumps(out[sc]), flush=True)
    path = a.out or f"results/v2/mc_stress_K{a.K}_p{a.pscale}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"config": vars(a), "results": out}, f, indent=1)


if __name__ == "__main__":
    main()
