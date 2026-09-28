"""Exposure-constrained re-ranking on top of each propagation operator.

For cap factors c (cap = ceil(c * R K / #items); c = inf is no constraint)
we report nDCG@K / Recall@K on the unbiased test data, the concentration of
top-K exposure (Gini, coverage, max share), the certified relative
optimality gap of the re-ranker and whether all caps hold.
"""

import argparse
import gc
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drup import fat  # noqa: E402
from drup.data import load_coat, load_yahoo  # noqa: E402
from drup.estimation import get_propensity  # noqa: E402
from drup.metrics import evaluate  # noqa: E402
from drup.pipeline import build, scores_with_consts  # noqa: E402
from drup.rerank import rerank, rerank_exact, uniform_caps  # noqa: E402
sys.path.insert(0, os.path.dirname(__file__))
from run_fat import chosen_config  # noqa: E402

METHODS = ["Obs", "IPS", "DR", "DRUP"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="coat")
    ap.add_argument("--prop", default="given")
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--factors", type=float, nargs="+", default=[float("inf"), 20, 10, 5, 3, 2, 1.5])
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--solver", default="exact", help="exact (min-cost flow) | dual (subgradient + repair)")
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--filters_json", default=None, help="where to read the selected configurations")
    ap.add_argument("--tag", default="", help="suffix of the output file")
    a = ap.parse_args()
    dt = getattr(torch, a.dtype)
    if a.dataset == "coat":
        d = load_coat()
        K = 5
    elif a.dataset == "yahoo":
        d = load_yahoo()
        K = 5
    else:
        d = torch.load("data/raw/kuairec.pt", weights_only=False)
        K = 20
    O, Y = d["O"].to(dt), d["Y"].to(dt)
    n = O.shape[1]
    P_raw = get_propensity(d, O, Y, a.prop)
    test = [t for t in d["test"] if len(t[1]) >= K]
    rows_users = sorted({u for u, _, _ in test})
    rows = torch.tensor(rows_users)
    row_of = {u: k for k, u in enumerate(rows_users)}
    mask = torch.zeros(len(rows_users), n, dtype=torch.bool)
    for u, items, _ in test:
        mask[row_of[u], torch.as_tensor(items)] = True
    fj = a.filters_json or f"results/filters_{a.dataset}_{a.prop}.json"
    cfgs = {m: chosen_config(fj, m) for m in a.methods}
    out = {"K": K, "configs": cfgs, "results": {}}
    for mth in a.methods:
        M = build(O, Y, P_raw, mth, cfgs[mth])
        S = scores_with_consts(M, rows)[0].double()
        del M
        gc.collect()
        res = []
        for fct in a.factors:
            cap = uniform_caps(mask, K, fct)
            try:
                if a.solver == "exact" and not math.isinf(fct):
                    alloc, info = rerank_exact(S, mask, K, cap)
                else:
                    alloc, info = rerank(S, mask, K, cap, iters=a.iters)
            except (RuntimeError, AssertionError) as e:
                res.append({"factor": fct, "error": str(e)})
                print(mth, fct, "infeasible", flush=True)
                continue
            S2 = S.clone()
            big = float(S.abs().max()) * 10 + 10
            ar = torch.arange(K, dtype=torch.float64)
            S2.scatter_(1, alloc, big - ar.expand(len(rows_users), K))
            ev = evaluate(S2, test, (K,), row_of)
            freq = np.bincount(alloc.flatten().numpy(), minlength=n).astype(float)
            cand = mask.any(0).numpy()
            r = {"factor": fct, "cap": float(cap.max()), **ev,
                 "gini": fat.gini(freq[cand]), "coverage": float((freq[cand] > 0).mean()),
                 "max_share": float(freq.max() / len(rows_users)),
                 "coverage_lower_bound": (len(rows_users) * K / float(cap.max())) / int(cand.sum()),
                 **info}
            res.append(r)
            print(mth, f"c={fct}", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                                    if k in (f"ndcg@{K}", "gini", "coverage", "coverage_lower_bound",
                                             "rel_gap", "feasible")}, flush=True)
        out["results"][mth] = res
        del S
        gc.collect()
    with open(f"results/rerank_{a.dataset}_{a.prop}{a.tag}.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
