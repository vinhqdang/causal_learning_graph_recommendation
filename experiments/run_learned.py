"""Trained baselines on the same splits / protocol as run_filters.py."""

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
from drup.estimation import baseline_imputation, clip_propensity, get_propensity  # noqa: E402
from drup.learned import MF, LightGCN, train  # noqa: E402
from drup.metrics import evaluate  # noqa: E402

METHODS = {
    "MF": ("mf", "naive"),
    "IPS-MF": ("mf", "ips"),
    "DR-MF": ("mf", "dr"),
    "LightGCN": ("lgn", "naive"),
    "NAVIP": ("navip", "naive"),
    "DR-LightGCN": ("lgn", "dr"),
}


def build(kind, m, n, d, O, Y, P):
    if kind == "mf":
        return MF(m, n, d)
    eu, ei = torch.nonzero(O * Y > 0, as_tuple=True)
    w = torch.ones(len(eu)) if kind == "lgn" else (1.0 / P[eu, ei]).float()
    return LightGCN(m, n, d, eu, ei, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="coat")
    ap.add_argument("--prop", default="given")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--frac_val", type=float, default=0.3)
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-2, 3e-3])
    ap.add_argument("--wds", type=float, nargs="+", default=[1e-4, 1e-3])
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--floor", type=float, default=0.05)
    ap.add_argument("--pairs_per_epoch", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--tune_first_split", action="store_true",
                    help="grid-search on split 0 only, then reuse the best config on all splits")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    if a.dataset == "coat":
        d = load_coat()
        ks, key, by = (5, 10), "ndcg@5", "entry"
    elif a.dataset == "yahoo":
        d = load_yahoo()
        ks, key, by = (5, 10), "ndcg@5", "user"
    else:
        d = torch.load("data/raw/kuairec.pt", weights_only=False)
        ks, key, by = (10, 20, 50), "ndcg@20", "user"
    O, Y = d["O"].float(), d["Y"].float()
    m, n = O.shape
    P = get_propensity(d, O.double(), Y.double(), a.prop).float()
    P = clip_propensity(P, a.floor)
    Yhat = baseline_imputation(O.double(), Y.double(), P.double(), lam=5.0).float()
    rows_users = sorted({u for u, _, _ in d["test"]})
    rows = torch.tensor(rows_users)
    row_of = {u: k for k, u in enumerate(rows_users)}
    splits = [split_test(d["test"], a.frac_val, s, by=by) for s in range(a.seeds)]

    results = {}
    for name in a.methods:
        kind, loss = METHODS[name]
        t0 = time.time()
        per_split, chosen = [], []
        grid = list(itertools.product(a.lrs, a.wds))
        for sidx, (val, tst) in enumerate(splits):
            best = (-1, None, None)
            if a.tune_first_split and sidx > 0:
                grid = [(chosen[0]["lr"], chosen[0]["wd"])]
            for lr, wd in grid:
                torch.manual_seed(sidx)
                model = build(kind, m, n, a.dim, O, Y, P)

                def vf(mod):
                    return evaluate(mod.full_scores(rows), val, ks, row_of)[key]
                v = train(model, O, Y, P, Yhat, loss, lr, wd, a.epochs, a.batch, vf,
                          patience=a.patience, pairs_per_epoch=a.pairs_per_epoch, seed=sidx)
                if v > best[0]:
                    with torch.no_grad():
                        best = (v, {"lr": lr, "wd": wd}, evaluate(model.full_scores(rows), tst, ks, row_of))
            per_split.append(best[2])
            chosen.append(best[1])
            print(f"  {name} split {sidx}: val {best[0]:.4f} test {best[2][key]:.4f} {best[1]}", flush=True)
        agg = {k: (float(np.mean([r[k] for r in per_split])), float(np.std([r[k] for r in per_split])))
               for k in per_split[0]}
        results[name] = {"test": agg, "chosen": chosen, "per_split": per_split}
        txt = "  ".join(f"{k}={v[0]:.4f}±{v[1]:.4f}" for k, v in agg.items())
        print(f"{name:12s} {txt}   [{time.time() - t0:.0f}s]", flush=True)
    out = a.out or f"results/learned_{a.dataset}_{a.prop}.json"
    with open(out, "w") as f:
        json.dump({"args": vars(a), "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
