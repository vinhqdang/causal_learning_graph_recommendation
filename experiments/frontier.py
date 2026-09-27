"""Accuracy / item-fairness trade-off over each method's hyper-parameter grid.

For every configuration we compute ndcg@K on the unbiased test data and the
concentration of top-K recommendations (Gini@K, coverage@K). A method's
Pareto front says how much accuracy it keeps at a given level of exposure
concentration; we also report the best ndcg reachable under Gini caps.
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drup import fat  # noqa: E402
from drup.data import load_coat  # noqa: E402
from drup.estimation import popularity_propensity  # noqa: E402
from drup.metrics import evaluate  # noqa: E402
from drup.pipeline import build, raw_parts  # noqa: E402

BETAS = [0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 1e3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="coat")
    ap.add_argument("--prop", default="given")
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    ap.add_argument("--floors", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--lams", type=float, nargs="+", default=[1.0, 5.0])
    a = ap.parse_args()
    dt = getattr(torch, a.dtype)
    if a.dataset == "coat":
        d = load_coat()
        K = 5
    else:
        d = torch.load("data/raw/kuairec.pt", weights_only=False)
        K = 20
    O, Y = d["O"].to(dt), d["Y"].to(dt)
    n = O.shape[1]
    P_raw = d["P_given"].to(dt) if (a.prop == "given" and d.get("P_given") is not None) \
        else popularity_propensity(O)[0]
    test = d["test"]
    rows_users = sorted({u for u, _, _ in test})
    rows = torch.tensor(rows_users)
    row_of = {u: k for k, u in enumerate(rows_users)}
    cand = np.zeros(n, dtype=bool)
    for _, items, _ in test:
        cand[items] = True

    grids = {
        "Obs": [{"alpha": al} for al in a.alphas],
        "IPS": [{"alpha": al, "floor": f} for al in a.alphas for f in a.floors],
        "DR": [{"alpha": al, "floor": f, "lam": l, "deg": "W"}
               for al, f, l in itertools.product(a.alphas, a.floors, a.lams)],
        "DRUP": [{"alpha": al, "floor": f, "lam": l, "deg": "Yhat"}
                 for al, f, l in itertools.product(a.alphas, a.floors, a.lams)],
    }
    points = {m: [] for m in grids}
    for mth, cfgs in grids.items():
        for cfg in cfgs:
            M = build(O, Y, P_raw, mth, cfg)
            s1, s3 = raw_parts(M, rows)
            s1 = s1 / s1.abs().mean(1, keepdim=True).clamp_min(1e-12)
            s3 = s3 / s3.abs().mean(1, keepdim=True).clamp_min(1e-12)
            for b in BETAS:
                S = s3 if b >= 1e3 else s1 + b * s3
                nd = evaluate(S, test, (K,), row_of)[f"ndcg@{K}"]
                top = fat.topk_lists(S, test, K, row_of)
                freq = np.zeros(n)
                for lst in top.values():
                    freq[lst] += 1
                points[mth].append({"cfg": dict(cfg, beta=b), "ndcg": nd,
                                    "gini": fat.gini(freq[cand]),
                                    "coverage": float((freq[cand] > 0).mean())})
        print(mth, "done", len(points[mth]), flush=True)

    caps = [0.3, 0.4, 0.5, 0.6, 1.0]
    summary = {}
    for mth, pts in points.items():
        summary[mth] = {str(c): max([p["ndcg"] for p in pts if p["gini"] <= c], default=None) for c in caps}
        print(mth, "best ndcg under Gini cap:", {k: (round(v, 4) if v else None) for k, v in summary[mth].items()})
    with open(f"results/frontier_{a.dataset}_{a.prop}.json", "w") as f:
        json.dump({"K": K, "summary": summary, "points": points}, f, indent=1)


if __name__ == "__main__":
    main()
