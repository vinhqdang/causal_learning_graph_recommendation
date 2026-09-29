"""Training-free propagation estimators on unbiased test data.

Methods (the propagation operators share the 1-hop + beta * 3-hop form):
    Pop        item positive count
    Impute     outcome imputation Yhat only (the DR nuisance model)
    Obs        propagation over the observed positive graph (linear LightGCN;
               the normalisation exponent alpha makes it a linear r-AdjNorm)
    IPS        inverse-propensity adjacency, no walk correction (NAVIP-style)
    IPS+WC     IPS adjacency + idempotent walk correction
    DR         doubly robust adjacency, no walk correction
    DRUP       doubly robust adjacency + walk correction (proposed)
    DR-5hop / DRUP-5hop   the same with an added 5-hop term
    EASE, GF-CF           closed-form item models on the logged graph
    EASE-DR, GF-CF-DR     the same models on the doubly robust matrix W
    BSPM, BSPM-DR         blurring-sharpening filter (Choi et al., 2023) on the
                          logged and on the doubly robust graph

Protocol (v2):
  * all nuisances (propensity model, imputation) are cross-fitted over K
    random folds of pairs when --xfit K > 1;
  * the degree weights of every debiased operator come from the imputation
    Yhat or from cross-fitted edge estimates (deg = Wx: row / column sums of
    W over the other folds); both still depend on the log (see
    experiments/mc_protocol.py for the resulting gap to Assumption 2);
  * the 1-hop and 3-hop terms are combined with *global* constants
    (score = s1 / c1 + beta * s3 / c3, c = mean |term| over all scored users),
    i.e. a fixed linear combination;
  * hyper-parameters are selected per split on the validation part of the
    unbiased data; per-user test metrics of the selected configuration are
    stored for user-level significance tests (experiments/significance.py).
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
from drup.estimation import clip_propensity, popularity_propensity  # noqa: E402
from drup.khop import khop  # noqa: E402
from drup.metrics import evaluate  # noqa: E402
from drup.pipeline import Nuisance, _impute, edge_weights  # noqa: E402
from drup.propagation import degree_weights, edge_estimate, three_hop  # noqa: E402

BETAS = [0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 1e3]
GAMMAS = [0.03, 0.1, 0.3, 1.0, 3.0]          # weight of the 5-hop term
EASE_LAMS = [10.0, 100.0, 1000.0]
GFCF = [(0.0, 0), (0.3, 64), (0.3, 256), (1.0, 256)]   # (weight of ideal low-pass, rank)
DR_FAMILY = ("DR", "DRUP", "DR-5hop", "DRUP-5hop", "EASE-DR", "GF-CF-DR", "BSPM-DR")
BSPM_GRID = [(tb, ts, w) for tb in (1.0, 2.0) for ts in (0.0, 0.1, 0.3) for w in (0.0, 0.3)]
SPLIT = ("DR-split", "DRUP-split")


class SplitNuisance:
    """Sample split (Assumption 2 on the rest of the log): the propensity model
    and the imputation are fitted on a random fraction q of the pairs (mask A),
    the edge estimates of those pairs are replaced by the imputation, and the
    degree weights and the score constants come from the imputation. Given the
    split and the exposures in A, the remaining edge estimates are independent
    of every nuisance, and Theorem 1 applies with delta_e = Yhat_e - Y_e on A."""

    def __init__(self, d, O, Y, prop, q, seed=0):
        g = torch.Generator().manual_seed(seed + 12345)
        self.A = torch.rand(O.shape, generator=g) < q
        self.O, self.Y = O, Y
        Am = self.A.to(O.dtype)
        if prop == "given" and d.get("P_given") is not None:
            self.P = d["P_given"].to(O.dtype)
        else:
            self.P = popularity_propensity(O, mask=Am)[0]
        self._cache = {}

    def imputation(self, floor, cfg):
        key = (floor, cfg["lam"], cfg.get("imp", "add"))
        if key not in self._cache:
            self._cache.clear()
            self._cache[key] = _impute(self.O, self.Y, clip_propensity(self.P, floor), cfg,
                                       mask=self.A.to(self.O.dtype))
        return self._cache[key]


def gmean(x):
    return float(x.abs().mean().clamp_min(1e-12))


def propagate(W, C, correct, rows, hops=3):
    s1 = (C * W)[rows]
    s3 = three_hop(W, C, rows=rows, correct=correct)
    out = [s1 / gmean(s1), s3 / gmean(s3)]
    if hops == 5:
        s5 = khop(W, C, 5, rows=rows, correct=correct)
        out.append(s5 / gmean(s5))
    return out


def ease(X, rows, lam):
    G = X.T @ X
    G.diagonal().add_(lam)
    Pm = torch.linalg.inv(G)
    B = -Pm / torch.diagonal(Pm)[None, :]
    B.fill_diagonal_(0.0)
    return X[rows] @ B


def gfcf(X, rows, D, weights_ranks):
    """GF-CF (Shen et al., 2021): linear filter R~^T R~ plus an ideal
    low-pass filter from the top singular vectors of R~, with
    R~ = D_u^-1/2 X D_i^-1/2 and degrees from D."""
    du = D.sum(1).clamp_min(1.0).pow(-0.5)
    di = D.sum(0).clamp_min(1.0).pow(-0.5)
    Rt = du[:, None] * X * di[None, :]
    lin = (X[rows] @ Rt.T) @ Rt
    out = []
    kmax = max(k for _, k in weights_ranks)
    V = None
    if kmax > 0:
        _, _, V = torch.svd_lowrank(Rt, q=kmax + 10, niter=4)
    for w, k in weights_ranks:
        s = lin / gmean(lin)
        if w > 0:
            Vk = V[:, :k]
            ideal = ((X[rows] * di[None, :]) @ Vk) @ (Vk.T / di[None, :])
            s = s + w * ideal / gmean(ideal)
        out.append(({"w": w, "k": k}, s))
    return out


def bspm(X, rows, D, grid, steps=4, rank=256):
    """Blurring-sharpening process (Choi et al., SIGIR 2023) on the normalised
    item Gram matrix P = R~^T R~ (R~ as in GF-CF): Euler steps of the heat
    equation dx/dt = x (P - I) for time T_b (blurring, optionally with the
    ideal low-pass term of GF-CF with weight w), then of dx/dt = -x P for time
    T_s (sharpening)."""
    du = D.sum(1).clamp_min(1.0).pow(-0.5)
    di = D.sum(0).clamp_min(1.0).pow(-0.5)
    Rt = du[:, None] * X * di[None, :]
    Pm = Rt.T @ Rt
    x0 = X[rows]
    V = None
    if any(w > 0 for _, _, w in grid):
        _, _, V = torch.svd_lowrank(Rt, q=rank + 10, niter=4)
        V = V[:, :rank]
    out = []
    for tb, ts, w in grid:
        x = x0.clone()
        h = tb / steps
        for _ in range(steps):
            dx = x @ Pm - x
            if w > 0:
                dx = dx + w * (((x * di[None, :]) @ V) @ (V.T / di[None, :]))
            x = x + h * dx
        hs = ts / steps
        for _ in range(steps if ts > 0 else 0):
            x = x - hs * (x @ Pm)
        out.append(({"tb": tb, "ts": ts, "w": w}, x))
    return out


def configs(method, a):
    if method == "Pop":
        grid = {}
    elif method == "Impute":
        grid = {"lam": a.lams, "imp": a.imps}
    elif method in ("Obs", "EASE", "GF-CF", "BSPM"):
        grid = {"alpha": a.alphas} if method == "Obs" else {}
    elif method.startswith("IPS"):
        grid = {"floor": a.floors, "lam": a.lams, "alpha": a.alphas, "deg": a.degs}
    elif method in ("EASE-DR", "GF-CF-DR", "BSPM-DR"):
        grid = {"floor": a.floors, "lam": a.lams, "imp": a.imps, "cv": a.cvs}
    elif method in SPLIT:
        grid = {"floor": a.floors, "lam": a.lams, "imp": a.imps, "alpha": a.alphas, "cv": a.cvs}
    else:
        grid = {"floor": a.floors, "lam": a.lams, "imp": a.imps, "alpha": a.alphas, "cv": a.cvs,
                "deg": a.degs}
    keys = list(grid)
    for vals in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, vals))


def score_bank(d, method, cfg, nz, rows):
    """Returns a list of (cfg_with_hyper, scores[rows]) for one base config."""
    O, Y = d["O"], d["Y"]
    if method == "Pop":
        return [(dict(cfg), (O * Y).sum(0, keepdim=True).expand(len(rows), -1))]
    if method == "Impute":
        return [(dict(cfg), nz.imputation(0.05, cfg)[rows])]
    if method == "BSPM":
        X = O * Y
        return [(dict(cfg, **h), sc) for h, sc in bspm(X, rows, X, BSPM_GRID)]
    if method in ("EASE", "GF-CF"):
        X = O * Y
        if method == "EASE":
            return [(dict(cfg, lam_e=l), ease(X, rows, l)) for l in EASE_LAMS]
        return [(dict(cfg, **h), s) for h, s in gfcf(X, rows, X, GFCF)]
    if method == "Obs":
        W = O * Y
        C = degree_weights(W, cfg["alpha"])
        parts = propagate(W, C, False, rows)
    elif method in SPLIT:
        P = clip_propensity(nz.P, cfg["floor"])
        Yd = nz.imputation(cfg["floor"], dict(cfg, imp=cfg.get("imp", "add")))
        W = torch.where(nz.A, Yd, edge_estimate(O, Y, P, Yd, cfg.get("cv", 1.0)))
        C = degree_weights(W, cfg["alpha"], D=Yd)
        correct = method == "DRUP-split"
        s1 = (C * W)[rows]
        s3 = three_hop(W, C, rows=rows, correct=correct)
        # constants from the imputed graph, so that the score is a fixed
        # function of the edge estimates given the nuisances
        c1 = gmean((C * Yd)[rows])
        c3 = gmean(three_hop(Yd, C, rows=rows, correct=correct))
        parts = [s1 / c1, s3 / c3]
    else:
        P = clip_propensity(nz.P, cfg["floor"])
        Yd = nz.imputation(cfg["floor"], dict(cfg, imp=cfg.get("imp", "add")))
        dr = method in DR_FAMILY
        W = edge_estimate(O, Y, P, Yd if dr else None, cfg.get("cv", 1.0) if dr else 1.0)
        if method == "EASE-DR":
            return [(dict(cfg, lam_e=l), ease(W, rows, l)) for l in EASE_LAMS]
        if method == "BSPM-DR":
            return [(dict(cfg, **h), sc) for h, sc in bspm(W, rows, Yd, BSPM_GRID)]
        if method == "GF-CF-DR":
            return [(dict(cfg, **h), s) for h, s in gfcf(W, rows, Yd, GFCF)]
        C = edge_weights(nz, W, Yd, cfg["alpha"], cfg.get("deg", "Yhat"))
        correct = method in ("IPS+WC", "DRUP", "DRUP-5hop")
        parts = propagate(W, C, correct, rows, hops=5 if method.endswith("5hop") else 3)
    out = []
    if len(parts) == 3:
        s1, s3, s5 = parts
        for b in BETAS[:-1]:
            for gm in GAMMAS:
                out.append((dict(cfg, beta=b, gamma=gm), s1 + b * s3 + gm * s5))
        return out
    s1, s3 = parts
    for b in BETAS:
        out.append((dict(cfg, beta=b), s1 + b * s3 if b < 1e3 else s3))
    return out


def load_dataset(name, dt):
    if name == "coat":
        d, ks, key, by = load_coat(), (5, 10), "ndcg@5", "entry"
    elif name == "yahoo":
        d, ks, key, by = load_yahoo(), (5, 10), "ndcg@5", "user"
    elif name == "kuairand":
        d = torch.load("data/raw/kuairand.pt", weights_only=False)
        ks, key, by = (5, 10, 20), "ndcg@10", "user"
    else:
        d = torch.load("data/raw/kuairec.pt", weights_only=False)
        ks, key, by = (10, 20, 50), "ndcg@20", "user"
    d["O"], d["Y"] = d["O"].to(dt), d["Y"].to(dt)
    return d, ks, key, by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="coat")
    ap.add_argument("--prop", default="given", help="given | pop")
    ap.add_argument("--xfit", type=int, default=5, help="cross-fitting folds (<= 1: none)")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--frac_val", type=float, default=0.3)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    ap.add_argument("--floors", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--lams", type=float, nargs="+", default=[1.0, 5.0, 20.0])
    ap.add_argument("--imps", nargs="+", default=["add"], help="imputation: add | lr")
    ap.add_argument("--degs", nargs="+", default=["Yhat", "Wx"],
                    help="degree source: imputation (Yhat) or cross-fitted edge estimates (Wx)")
    ap.add_argument("--cvs", type=float, nargs="+", default=[1.0],
                    help="control-variate weight of the imputation (0 = IPS, 1 = DR)")
    ap.add_argument("--methods", nargs="+",
                    default=["Pop", "Impute", "Obs", "EASE", "GF-CF", "IPS", "IPS+WC", "DR", "DRUP",
                             "EASE-DR", "GF-CF-DR"])
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--out", default=None)
    ap.add_argument("--redo", action="store_true", help="recompute methods already in --out")
    ap.add_argument("--split_seed", type=int, default=0, help="seed of the sample split A")
    ap.add_argument("--split", type=float, default=0.2,
                    help="fraction of pairs used for the nuisances of DR-split / DRUP-split")
    ap.add_argument("--dump_bank", default=None,
                    help="also write every configuration's validation and test score per split "
                         "(for tuning-budget curves)")
    a = ap.parse_args()

    dt = getattr(torch, a.dtype)
    d, ks, key, by = load_dataset(a.dataset, dt)
    t0 = time.time()
    todo0 = [mth for mth in a.methods]
    nz = (Nuisance(d, d["O"], d["Y"], a.prop, K=a.xfit, seed=0)
          if any(mth not in SPLIT for mth in todo0) else None)
    nzs = SplitNuisance(d, d["O"], d["Y"], a.prop, a.split, seed=a.split_seed) if any(mth in SPLIT for mth in todo0) else None
    print(f"nuisance setup (xfit={a.xfit}) {time.time() - t0:.0f}s", flush=True)
    rows_users = sorted({u for u, _, _ in d["test"]})
    rows = torch.tensor(rows_users)
    row_of = {u: k for k, u in enumerate(rows_users)}

    splits = [split_test(d["test"], a.frac_val, s, by=by) for s in range(a.seeds)]
    results = {}
    if a.out and os.path.exists(a.out):
        results = json.load(open(a.out)).get("results", {})
    # Group the work by imputation key so that every cross-fitted imputation
    # is computed once and shared by all methods that use it.
    def nkey(method, cfg):
        if method in ("Pop", "Obs", "EASE", "GF-CF", "BSPM"):
            return None
        if method == "Impute":
            return (0.05, cfg["lam"], cfg.get("imp", "add"))
        if method in SPLIT:
            return ("split", cfg["floor"], cfg["lam"], cfg.get("imp", "add"))
        return (cfg["floor"], cfg["lam"], cfg.get("imp", "add"))
    todo = [mth for mth in a.methods if mth not in results or a.redo]
    jobs = {}
    for method in todo:
        for cfg in configs(method, a):
            jobs.setdefault(nkey(method, cfg), []).append((method, cfg))
    banks = {mth: [] for mth in todo}
    times = {mth: 0.0 for mth in todo}
    order = sorted(jobs, key=lambda k: (k is not None, str(k)))
    for kk, key_ in enumerate(order):
        tk = time.time()
        for method, cfg in jobs[key_]:
            t0 = time.time()
            for c, s in score_bank(d, method, cfg, nzs if method in SPLIT else nz, rows):
                vm = [evaluate(s, v, ks, row_of)[key] for v, _ in splits]
                tm = []
                for _, t in splits:
                    agg, pu = evaluate(s, t, ks, row_of, per_user=key)
                    # per-user values as a compact array (user order is fixed per split)
                    tm.append((agg, np.fromiter(pu.values(), dtype=np.float32), tuple(pu)))
                banks[method].append((c, vm, tm))
            times[method] += time.time() - t0
        print(f"[{kk + 1}/{len(order)}] nuisance {key_} done [{time.time() - tk:.0f}s]", flush=True)
    for method in todo:
        bank = banks[method]
        per_split, chosen, val_best, per_user = [], [], [], []
        for sidx in range(a.seeds):
            best = max(bank, key=lambda b: b[1][sidx])
            per_split.append(best[2][sidx][0])
            per_user.append(dict(zip(best[2][sidx][2], best[2][sidx][1].tolist())))
            chosen.append(best[0])
            val_best.append(best[1][sidx])
        agg = {m: (float(np.mean([r[m] for r in per_split])), float(np.std([r[m] for r in per_split])))
               for m in per_split[0]}
        results[method] = {"test": agg, "chosen": chosen, "per_split": per_split, "val_best": val_best,
                           "per_user": [{str(u): v for u, v in pu.items()} for pu in per_user],
                           "n_configs": len(bank)}
        txt = "  ".join(f"{m}={v[0]:.4f}±{v[1]:.4f}" for m, v in agg.items())
        print(f"{method:9s} {txt}   [{times[method]:.0f}s, {len(bank)} cfgs] e.g. {chosen[0]}", flush=True)
    if a.dump_bank:
        dump = {mth: [{"cfg": c, "val": vm, "test": [t[0][key] for t in tm]} for c, vm, tm in banks[mth]]
                for mth in todo}
        os.makedirs(os.path.dirname(a.dump_bank), exist_ok=True)
        with open(a.dump_bank, "w") as f:
            json.dump({"args": vars(a), "key": key, "banks": dump}, f)
        if a.out is None:
            return
    out = a.out or f"results/v2/filters_{a.dataset}_{a.prop}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"args": vars(a), "results": results}, f)


if __name__ == "__main__":
    main()
