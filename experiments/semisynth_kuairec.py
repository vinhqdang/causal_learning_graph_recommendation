"""Semi-synthetic score-level test on KuaiRec's fully observed small matrix.

The small matrix (1,411 users x 3,327 items, 99.6% observed) gives the
potential outcomes Y (watch ratio >= 2; the few unobserved pairs count as 0).
Exposure logs are drawn from a known MNAR mechanism fitted to the big
matrix: logit p_ui = c + 1.0 z_u + 1.5 z_i + gamma * Y_ui, with z the
standardised logit exposure rates of the user and the item in the big log and
c chosen for a target density. Because Y and p are known, the full-exposure
target F* = a C*Y + b (C*Y)(C*Y)^T(C*Y) is known, and so is every decision
that depends on it.

For every draw of the log we build IPS / IPS+WC / DR / DRUP scores with
nuisances fitted on an independent log (indep), cross-fitted on the same log
(xfit), or on a 20% sample split (split); with true or estimated (logistic
exposure model) propensities; and with degree weights C either fixed a priori
(the true-degree normalisation, which the target uses) or taken from the
imputation (the target keeps the fixed C). We record

  * score-level bias (averaged over draws) of the three-hop term and of the
    full score over all pairs and on unexposed candidates,
  * the error of the estimated full-exposure utility sum_x s of two fixed
    allocations (random and log-popular 20 items per user),
  * the regret, under F*, of the exposure-capped allocation computed from the
    estimated scores (cap: twice the fair share, K = 20), a decision that
    compares scores across users, and
  * nDCG@20 against Y on unexposed candidates.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drup.estimation import (clip_propensity, crossfit, fold_ids, lowrank_imputation,  # noqa: E402
                             popularity_propensity)
from drup.khop import khop  # noqa: E402
from drup.propagation import degree_weights, edge_estimate, three_hop  # noqa: E402
from drup.rerank import rerank_exact, uniform_caps  # noqa: E402

DT = torch.float64
ROOT = "data/raw/KuaiRec 2.0/data"


def load(threshold=2.0):
    small = pd.read_csv(os.path.join(ROOT, "small_matrix.csv"), usecols=["user_id", "video_id", "watch_ratio"])
    big = pd.read_csv(os.path.join(ROOT, "big_matrix.csv"), usecols=["user_id", "video_id"])
    users, items = np.sort(small.user_id.unique()), np.sort(small.video_id.unique())
    ui, ii = {u: k for k, u in enumerate(users)}, {i: k for k, i in enumerate(items)}
    g = small.groupby(["user_id", "video_id"]).watch_ratio.mean().reset_index()
    Y = torch.zeros(len(users), len(items), dtype=DT)
    Y[g.user_id.map(ui).values, g.video_id.map(ii).values] = torch.as_tensor(
        (g.watch_ratio.values >= threshold).astype(np.float64))
    # big-log exposure rates of these users and items
    bu = big.user_id.value_counts()
    bi = big.video_id.value_counts()
    nu, ni = big.video_id.nunique(), big.user_id.nunique()
    ru = np.array([bu.get(u, 0) for u in users]) / nu
    ri = np.array([bi.get(i, 0) for i in items]) / ni
    pop_big = torch.as_tensor(np.array([bi.get(i, 0) for i in items]), dtype=DT)
    return Y, ru, ri, pop_big


def propensities(ru, ri, Y, density, gamma, pmin=1e-4):
    def z(x):
        lx = np.log(np.clip(x, 1e-4, 1 - 1e-4) / (1 - np.clip(x, 1e-4, 1 - 1e-4)))
        return torch.as_tensor((lx - lx.mean()) / lx.std(), dtype=DT)
    base = 1.0 * z(ru)[:, None] + 1.5 * z(ri)[None, :] + gamma * (Y - Y.mean())
    lo, hi = -15.0, 5.0
    for _ in range(60):                      # bisection on the intercept
        c = (lo + hi) / 2
        if float(torch.sigmoid(base + c).mean()) > density:
            hi = c
        else:
            lo = c
    return torch.sigmoid(base + c).clamp(pmin, 0.95)


def ndcg_unexposed(S, Y, O, k=20):
    Sm = torch.where(O > 0, torch.full_like(S, -1e18), S)
    top = torch.topk(Sm, k, dim=1).indices
    rel = Y.gather(1, top)
    disc = 1.0 / torch.log2(torch.arange(2, k + 2, dtype=DT))
    npos = ((1 - O) * Y).sum(1)
    ideal = torch.stack([disc[: int(min(n, k))].sum() for n in npos])
    ok = npos > 0
    return float(((rel * disc).sum(1)[ok] / ideal[ok]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--density", type=float, default=0.13)
    ap.add_argument("--gamma", type=float, default=1.0, help="outcome dependence of the exposure")
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--pmin", type=float, default=0.02, help="lower bound of the true propensities")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--q", type=float, default=0.2)
    ap.add_argument("--rerank_items", type=int, default=300)
    ap.add_argument("--k5", action="store_true", help="also five hops (indep, true p, fixed C)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time()
    Y, ru, ri, pop_big = load()
    m, n = Y.shape
    P = propensities(ru, ri, Y, a.density, a.gamma, a.pmin)
    g = torch.Generator().manual_seed(a.seed)
    print(f"Y {m}x{n}, positive rate {float(Y.mean()):.3f}; propensity mean {float(P.mean()):.3f}, "
          f"median {float(P.median()):.4f}, below tau {float((P < a.tau).double().mean()):.3f}", flush=True)
    # target with a fixed normalisation (true degrees)
    Cfix = degree_weights(Y, a.alpha, D=Y)
    T3 = three_hop(Y, Cfix, correct=False)
    S1 = Cfix * Y
    c1, c3 = float(S1.abs().mean()), float(T3.abs().mean())
    Fstar = S1 / c1 + a.beta * T3 / c3
    T5 = khop(Y, Cfix, 5, correct=False) if a.k5 else None
    # fixed allocations for utility estimation and the rerank candidate subset
    rnd = torch.argsort(torch.rand(m, n, generator=g), dim=1)
    alloc_rand = rnd[:, :20]
    alloc_pop = torch.argsort(pop_big + 1e-9 * torch.rand(n, generator=g), descending=True)[:20].expand(m, 20)
    rr_items = rnd[:, 20:20 + a.rerank_items]
    rr_mask0 = torch.zeros(m, n, dtype=torch.bool).scatter_(1, rr_items, True)

    edges = ("IPS", "DR")
    acc = {}
    rows = {}
    cu = torch.zeros(m, n)

    def add(key, name, val):
        acc.setdefault(key, {}).setdefault(name, []).append(val)

    for rep in range(a.reps):
        tr = time.time()
        O = (torch.rand(m, n, generator=g, dtype=DT) < P).to(DT)
        O2 = (torch.rand(m, n, generator=g, dtype=DT) < P).to(DT)
        folds = fold_ids(O.shape, 10, seed=rep)
        A = (torch.rand(m, n, generator=g, dtype=DT) < a.q)
        un = (1 - O) > 0
        cu += un.float()
        rr_mask = rr_mask0 & un
        rr_cap = uniform_caps(rr_mask, 20, 2.0)
        al_best, _ = rerank_exact(Fstar, rr_mask, 20, rr_cap)
        u_opt = float(Fstar.gather(1, al_best).sum())
        for src in ("indep", "xfit", "split"):
            for prop in ("true", "est"):
                if src == "indep":
                    Pr = P if prop == "true" else popularity_propensity(O2)[0]
                    Pb = clip_propensity(Pr, a.tau)
                    Yh = lowrank_imputation(O2, Y, Pb, rank=a.rank, ridge=5.0)
                elif src == "xfit":
                    Pr = P if prop == "true" else crossfit(lambda M: popularity_propensity(O, mask=M)[0], O, folds, 10)
                    Pb = clip_propensity(Pr, a.tau)
                    Yh = crossfit(lambda M: lowrank_imputation(O * M, Y, Pb, rank=a.rank, ridge=5.0), O, folds, 10)
                else:
                    Am = A.to(DT)
                    Pr = P if prop == "true" else popularity_propensity(O, mask=Am)[0]
                    Pb = clip_propensity(Pr, a.tau)
                    Yh = lowrank_imputation(O * Am, Y, Pb, rank=a.rank, ridge=5.0)
                print(f"  {src}/{prop} nuisances [{time.time() - tr:.0f}s]", flush=True)
                for cdeg in (("fixed", "yhat") if prop == "true" else ("fixed",)):
                    C = Cfix if cdeg == "fixed" else degree_weights(Yh, a.alpha, D=Yh)
                    for edge in edges:
                        W = edge_estimate(O, Y, Pb, Yh if edge == "DR" else None)
                        if src == "split":
                            W = torch.where(A, Yh, W)
                        for corr in (False, True):
                            name = {("IPS", False): "IPS", ("IPS", True): "IPS+WC",
                                    ("DR", False): "DR", ("DR", True): "DRUP"}[(edge, corr)]
                            key = f"{src}/{prop}/{cdeg}"
                            T = three_hop(W, C, correct=corr)
                            s = C * W / c1 + a.beta * T / c3
                            r = rows.setdefault(key, {}).setdefault(name, {
                                "dT": torch.zeros(m, n), "qT": torch.zeros(m, n), "ds": torch.zeros(m, n),
                                "dsu": torch.zeros(m, n)})
                            r["dT"] += (T - T3).float()
                            r["qT"] += ((T - T3) ** 2).float()
                            r["ds"] += (s - Fstar).float()
                            r["dsu"] += ((s - Fstar) * un).float()
                            for an, al in (("rand", alloc_rand), ("pop", alloc_pop)):
                                est, tru = float(s.gather(1, al).sum()), float(Fstar.gather(1, al).sum())
                                add(key, f"{name}/util_{an}", (est - tru) / tru)
                            add(key, f"{name}/ndcg20", ndcg_unexposed(s, Y, O))
                            if prop == "true" and cdeg == "fixed":
                                al, _ = rerank_exact(s, rr_mask, 20, rr_cap)
                                u_est = float(Fstar.gather(1, al).sum())
                                add(key, f"{name}/cap_regret", (u_opt - u_est) / u_opt)
                            if a.k5 and src == "indep" and prop == "true" and cdeg == "fixed" and edge == "DR":
                                T5e = khop(W, C, 5, correct=corr)
                                r5 = rows.setdefault(key + "/K5", {}).setdefault(name, {
                                    "dT": torch.zeros(m, n)})
                                r5["dT"] += (T5e - T5).float()
        print(f"rep {rep + 1}/{a.reps} [{time.time() - tr:.0f}s]", flush=True)
    out = {"config": vars(a), "info": {"m": m, "n": n, "p_mean": float(P.mean()), "p_median": float(P.median()),
                                       "frac_p_below_tau": float((P < a.tau).double().mean()),
                                       "pos_rate": float(Y.mean())}, "results": {}}
    sT, sF = float(T3.abs().mean()), float(Fstar.abs().mean())
    for key, byname in rows.items():
        for name, r in byname.items():
            res = {}
            if key.endswith("/K5"):
                res["rel_bias_T5"] = float((r["dT"] / a.reps).abs().mean() / T5.abs().mean())
            else:
                res["rel_bias_T"] = float((r["dT"] / a.reps).abs().mean() / sT)
                # Monte-Carlo noise floor: mean standard error of the entry-wise bias
                var = (r["qT"] / a.reps - (r["dT"] / a.reps) ** 2).clamp_min(0)
                res["rel_se_T"] = float((var / max(a.reps - 1, 1)).sqrt().mean() / sT)
                res["rel_rmse_T"] = float((r["qT"] / a.reps).mean().sqrt() / T3.pow(2).mean().sqrt())
                res["rel_bias_s"] = float((r["ds"] / a.reps).abs().mean() / sF)
                ok = cu > 0
                res["rel_bias_s_unexposed"] = float((r["dsu"][ok] / cu[ok]).abs().mean() / sF)
                for mt, vals in acc.get(key, {}).items():
                    if mt.startswith(name + "/"):
                        v = np.array(vals)
                        res[mt.split("/", 1)[1]] = {"mean": float(v.mean()), "rmse": float(np.sqrt((v ** 2).mean())),
                                                    "se": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0}
            out["results"].setdefault(key, {})[name] = res
            print(key, name, json.dumps({k: (v if not isinstance(v, dict) else round(v["mean"], 4)) for k, v in res.items()}),
                  flush=True)
    path = a.out or f"results/v3/semisynth_kuairec_d{a.density}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"done [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
