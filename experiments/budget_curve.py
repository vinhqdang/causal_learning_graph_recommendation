"""Expected test score as a function of the tuning budget.

For a method with N configurations and a budget of k configurations drawn at
random without replacement, the configuration with the best validation score
among the k is the j-th best overall with probability C(N-j, k-1) / C(N, k).
The expected test score of the selected configuration is computed exactly per
split and averaged over splits (expected validation performance, Dodge et al.
2019). Two units of budget: 'config' (a trained configuration keeps its
best epoch on each split; every value of beta is a configuration of DRUP) and
'eval' (every validation evaluation counts, i.e. every (configuration, epoch)
of a trained model and every configuration of a training-free operator).
"""

import argparse
import json
from fractions import Fraction
from math import comb

import numpy as np


def collapse(rows):
    """(cfg without epoch) -> per split (val, test) of the best epoch."""
    by = {}
    for r in rows:
        key = json.dumps({k: v for k, v in r["cfg"].items() if k != "epoch"}, sort_keys=True)
        cur = by.setdefault(key, [(-np.inf, 0.0)] * len(r["val"]))
        by[key] = [max(c, (v, t)) for c, v, t in zip(cur, r["val"], r["test"])]
    return list(by.values())


def expected(cfgs, k):
    n_split = len(cfgs[0])
    N = len(cfgs)
    out = []
    for s in range(n_split):
        vt = sorted(((c[s][0], c[s][1]) for c in cfgs), key=lambda x: -x[0])
        tot = comb(N, k)
        w = np.array([float(Fraction(comb(N - j, k - 1), tot)) for j in range(1, N + 1)])
        out.append(float((w * np.array([t for _, t in vt])).sum()))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--banks", nargs="+", required=True)
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4, 8, 9, 16, 18, 36, 64, 128, 384, 512, 1440])
    ap.add_argument("--out", required=True)
    ap.add_argument("--unit", default="config", choices=["config", "eval"])
    a = ap.parse_args()
    res = {}
    for path in a.banks:
        j = json.load(open(path))
        for mth, rows in j["banks"].items():
            if rows and "epoch" in rows[0]["cfg"] and a.unit == "config":
                cfgs = collapse(rows)
            else:
                cfgs = [list(zip(r["val"], r["test"])) for r in rows]
            N = len(cfgs)
            curve = {k: expected(cfgs, k) for k in sorted(set(a.budgets) | {N}) if k <= N}
            res[mth] = {"n_configs": N, "curve": curve}
            print(mth, N, " ".join(f"{k}:{v:.4f}" for k, v in curve.items()), flush=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
