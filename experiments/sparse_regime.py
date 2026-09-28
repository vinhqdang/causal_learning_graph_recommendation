"""Where does the walk correction change rankings? (KuaiRec, real data)

The logged exposures are thinned by a known factor q (O' = O * Bernoulli(q)),
which makes the log sparser and every propensity smaller; the nuisances are
then re-fitted (cross-fitted) on the thinned log. For each q we select, on
validation data, the configuration of the uncorrected and the corrected
operator (3 and 5 hops; DR and IPS ends of the control-variate family) and
report test nDCG, together with the agreement of the corrected and
uncorrected rankings at the same configuration (top-K overlap and Kendall
tau on the candidate items).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drup.data import split_test  # noqa: E402
from drup.estimation import clip_propensity  # noqa: E402
from drup.khop import khop  # noqa: E402
from drup.metrics import evaluate  # noqa: E402
from drup.pipeline import Nuisance  # noqa: E402
from drup.propagation import degree_weights, edge_estimate, three_hop  # noqa: E402

BETAS = [0.0, 0.03, 0.1, 0.3, 1.0, 3.0]
GAMMAS = [0.0, 0.1, 0.3, 1.0]


def gm(x):
    return float(x.abs().mean().clamp_min(1e-12))


def agreement(S1, S2, test, row_of, K):
    ov, kt = [], []
    A, B = S1.numpy(), S2.numpy()
    for u, items, _ in test:
        if len(items) < 3:
            continue
        a, b = A[row_of[u], items], B[row_of[u], items]
        ta, tb = set(np.argsort(-a)[:K].tolist()), set(np.argsort(-b)[:K].tolist())
        ov.append(len(ta & tb) / K)
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        kt.append(np.corrcoef(ra, rb)[0, 1])      # Spearman rank correlation
    return float(np.mean(ov)), float(np.nanmean(kt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qs", type=float, nargs="+", default=[1.0, 0.3, 0.1, 0.03])
    ap.add_argument("--floors", type=float, nargs="+", default=[0.2, 0.05, 0.01])
    ap.add_argument("--cvs", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--xfit", type=int, default=5)
    ap.add_argument("--out", default="results/v2/sparse_regime_kuairec.json")
    a = ap.parse_args()
    d = torch.load("data/raw/kuairec.pt", weights_only=False)
    O0, Y = d["O"].float(), d["Y"].float()
    test = d["test"]
    rows_users = sorted({u for u, _, _ in test})
    rows = torch.tensor(rows_users)
    row_of = {u: k for k, u in enumerate(rows_users)}
    splits = [split_test(test, 0.3, s, by="user") for s in range(a.seeds)]
    key, ks = "ndcg@20", (20,)
    out = {"args": vars(a), "rows": []}
    g = torch.Generator().manual_seed(99)
    for q in a.qs:
        O = O0 if q >= 1 else O0 * (torch.rand(O0.shape, generator=g) < q).float()
        dd = dict(d, O=O)
        t0 = time.time()
        nz = Nuisance(dd, O, Y, "pop", K=a.xfit, seed=1)
        print(f"q={q}: density {float(O.mean()):.4f}, nuisances {time.time() - t0:.0f}s", flush=True)
        bank = {m: [] for m in ("DR", "DRUP", "DR-5hop", "DRUP-5hop")}
        pairs = []
        for fl in a.floors:
            for cv in a.cvs:
                cfg = {"floor": fl, "lam": 20.0, "imp": "add", "alpha": 0.5, "cv": cv}
                P = clip_propensity(nz.P, fl)
                Yd = nz.imputation(fl, cfg)
                W = edge_estimate(O, Y, P, Yd, cv)
                C = degree_weights(W, 0.5, D=Yd)
                s1 = (C * W)[rows]
                s1 = s1 / gm(s1)
                parts = {}
                for corr in (False, True):
                    s3 = three_hop(W, C, rows=rows, correct=corr)
                    s5 = khop(W, C, 5, rows=rows, correct=corr)
                    parts[corr] = (s3 / gm(s3), s5 / gm(s5))
                for b in BETAS:
                    for gmm in GAMMAS:
                        S = {c: s1 + b * parts[c][0] + gmm * parts[c][1] for c in (False, True)}
                        c = dict(cfg, beta=b, gamma=gmm)
                        five = gmm > 0
                        for corr in (False, True):
                            name = ("DRUP" if corr else "DR") + ("-5hop" if five else "")
                            vm = [evaluate(S[corr], v, ks, row_of)[key] for v, _ in splits]
                            tm = [evaluate(S[corr], t, ks, row_of)[key] for _, t in splits]
                            bank[name].append((c, vm, tm))
                        if b in (0.3, 1.0) and gmm in (0.0, 0.3):
                            ov, sp = agreement(S[False], S[True], splits[0][1], row_of, 20)
                            pairs.append({"cfg": c, "overlap@20": ov, "spearman": sp})
                print(f"  floor {fl} cv {cv} done [{time.time() - t0:.0f}s]", flush=True)
        row = {"q": q, "density": float(O.mean()),
               "median_p_logged": float(nz.P[O > 0].median())}
        for m, b in bank.items():
            test_vals, chosen = [], []
            for s in range(a.seeds):
                best = max(b, key=lambda x: x[1][s])
                test_vals.append(best[2][s])
                chosen.append(best[0])
            row[m] = {"mean": float(np.mean(test_vals)), "std": float(np.std(test_vals)),
                      "per_split": test_vals, "chosen": chosen}
        row["agreement"] = pairs
        out["rows"].append(row)
        print(json.dumps({k: (v["mean"] if isinstance(v, dict) and "mean" in v else v)
                          for k, v in row.items() if k != "agreement"}), flush=True)
        print("  agreement (min overlap, min spearman):",
              min(p["overlap@20"] for p in pairs), min(p["spearman"] for p in pairs), flush=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
