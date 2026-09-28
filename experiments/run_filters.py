"""Training-free propagation estimators on unbiased test data.

Methods (all share the same 1-hop + beta * 3-hop propagation form):
    Pop        item positive count
    Impute     outcome imputation Yhat only (the DR nuisance model)
    Obs        propagation over the observed positive graph (linear LightGCN)
    IPS        inverse-propensity adjacency, no walk correction (NAVIP-style)
    IPS+WC     IPS adjacency + idempotent walk correction
    DR         doubly robust adjacency, no walk correction
    DRUP       doubly robust adjacency + walk correction (proposed)

Hyper-parameters are selected per split on the validation part of the
unbiased data (by ndcg@K) and the selected configuration is reported on the
held-out test part. Results are averaged over several random splits.
"""

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drup.data import load_coat, load_yahoo, split_test  # noqa: E402
from drup.estimation import clip_propensity, get_propensity  # noqa: E402
from drup.pipeline import impute  # noqa: E402
from drup.metrics import evaluate  # noqa: E402
from drup.khop import khop  # noqa: E402
from drup.propagation import degree_weights, edge_estimate, three_hop  # noqa: E402

BETAS = [0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 1e3]
GAMMAS = [0.03, 0.1, 0.3, 1.0, 3.0]          # weight of the 5-hop term


def unit(x):
    return x / x.abs().mean(1, keepdim=True).clamp_min(1e-12)


def propagate(O, Y, P, Yhat, alpha, correct, rows, deg="W", cv=1.0):
    W = edge_estimate(O, Y, P, Yhat, cv)
    C = degree_weights(W, alpha, D=Yhat if deg == "Yhat" else None)
    s1 = (C * W)[rows]
    s3 = three_hop(W, C, rows=rows, correct=correct)
    return unit(s1), unit(s3)


def propagate5(O, Y, P, Yhat, alpha, correct, rows, deg="W", cv=1.0):
    W = edge_estimate(O, Y, P, Yhat, cv)
    C = degree_weights(W, alpha, D=Yhat if deg == "Yhat" else None)
    s1 = (C * W)[rows]
    s3 = three_hop(W, C, rows=rows, correct=correct)
    s5 = khop(W, C, 5, rows=rows, correct=correct)
    return unit(s1), unit(s3), unit(s5)


def configs(method, a):
    if method in ("Pop", "Impute"):
        grid = {"lam": a.lams, "imp": a.imps} if method == "Impute" else {}
    elif method == "Obs":
        grid = {"alpha": a.alphas}
    elif method.startswith("IPS"):
        grid = {"alpha": a.alphas, "floor": a.floors}
    else:
        grid = {"alpha": a.alphas, "floor": a.floors, "lam": a.lams, "deg": a.degs, "imp": a.imps,
                "cv": a.cvs}
    keys = list(grid)
    for vals in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, vals))


def score_bank(d, method, cfg, P_raw, rows):
    """Returns a list of (cfg_with_beta, scores[rows]) for one base config."""
    O, Y = d["O"], d["Y"]
    if method == "Pop":
        s = (O * Y).sum(0, keepdim=True).expand(len(rows), -1)
        return [(dict(cfg), s)]
    if method == "Impute":
        P = clip_propensity(P_raw, 0.05)
        return [(dict(cfg), impute(O, Y, P, cfg)[rows])]
    if method == "Obs":
        s1, s3 = propagate(O, Y, torch.ones_like(O), None, cfg["alpha"], False, rows)
    else:
        P = clip_propensity(P_raw, cfg["floor"])
        Yhat = None
        if method.startswith("DR"):
            Yhat = impute(O, Y, P, cfg)
        correct = method in ("IPS+WC", "DRUP", "DRUP-5hop")
        if method.endswith("5hop"):
            s1, s3, s5 = propagate5(O, Y, P, Yhat, cfg["alpha"], correct, rows, cfg.get("deg", "W"),
                                    cfg.get("cv", 1.0))
            out = []
            for b in BETAS[:-1]:
                for gm in GAMMAS:
                    out.append((dict(cfg, beta=b, gamma=gm), s1 + b * s3 + gm * s5))
            return out
        s1, s3 = propagate(O, Y, P, Yhat, cfg["alpha"], correct, rows, cfg.get("deg", "W"),
                           cfg.get("cv", 1.0) if Yhat is not None else 1.0)
    out = []
    for b in BETAS:
        c = dict(cfg, beta=b)
        out.append((c, s1 + b * s3 if b < 1e3 else s3))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="coat")
    ap.add_argument("--prop", default="given", help="given | pop")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--frac_val", type=float, default=0.3)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    ap.add_argument("--floors", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--lams", type=float, nargs="+", default=[1.0, 5.0, 20.0])
    ap.add_argument("--degs", nargs="+", default=["W", "Yhat"])
    ap.add_argument("--imps", nargs="+", default=["add"], help="imputation: add | lr")
    ap.add_argument("--cvs", type=float, nargs="+", default=[1.0],
                    help="control-variate weight of the imputation (0 = IPS, 1 = DR)")
    ap.add_argument("--methods", nargs="+",
                    default=["Pop", "Impute", "Obs", "IPS", "IPS+WC", "DR", "DRUP"])
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dt = getattr(torch, a.dtype)
    if a.dataset == "coat":
        d = load_coat()
        ks, key, by = (5, 10), "ndcg@5", "entry"
    elif a.dataset == "yahoo":
        d = load_yahoo()
        ks, key, by = (5, 10), "ndcg@5", "user"
    else:
        d = torch.load("data/raw/kuairec.pt", weights_only=False)
        ks, key, by = (10, 20, 50), "ndcg@20", "user"
    d["O"], d["Y"] = d["O"].to(dt), d["Y"].to(dt)
    P_raw = get_propensity(d, d["O"], d["Y"], a.prop)
    rows_users = sorted({u for u, _, _ in d["test"]})
    rows = torch.tensor(rows_users)
    row_of = {u: k for k, u in enumerate(rows_users)}

    splits = [split_test(d["test"], a.frac_val, s, by=by) for s in range(a.seeds)]
    results = {}
    for method in a.methods:
        t0 = time.time()
        # val/test metrics for every configuration, for every split
        bank = []
        for cfg in configs(method, a):
            for c, s in score_bank(d, method, cfg, P_raw, rows):
                vm = [evaluate(s, v, ks, row_of)[key] for v, _ in splits]
                tm = [evaluate(s, t, ks, row_of) for _, t in splits]
                bank.append((c, vm, tm))
        per_split = []
        chosen = []
        for sidx in range(a.seeds):
            best = max(bank, key=lambda b: b[1][sidx])
            per_split.append(best[2][sidx])
            chosen.append(best[0])
        agg = {m: (float(np.mean([r[m] for r in per_split])), float(np.std([r[m] for r in per_split])))
               for m in per_split[0]}
        results[method] = {"test": agg, "chosen": chosen, "per_split": per_split}
        txt = "  ".join(f"{m}={v[0]:.4f}±{v[1]:.4f}" for m, v in agg.items())
        print(f"{method:8s} {txt}   [{time.time() - t0:.0f}s] e.g. {chosen[0]}", flush=True)
    out = a.out or f"results/filters_{a.dataset}_{a.prop}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"args": vars(a), "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
