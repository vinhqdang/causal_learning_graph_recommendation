"""Does the walk correction change rankings? Dose-response on semi-synthetic KuaiRec.

Pre-specified question: with true propensities, a-priori degree weights and an
independent log for the imputation (the setting of the theorems), is DRUP more
accurate than the uncorrected DR adjacency, at the same beta, and how does the
paired difference depend on beta (weight of the three-hop term) and on the
heterogeneity of the exposure (kappa, item-side coefficient; gamma, outcome
dependence)? Every cell is reported; nothing is selected over the grid.

Metrics per draw: nDCG@20 against Y on unexposed pairs, and the overlap of the
top-20 unexposed items with the top-20 of the full-exposure target F*.
"""
import argparse, json, os, sys, time
import numpy as np, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from semisynth_kuairec import load, ndcg_unexposed, DT  # noqa: E402
from drup.estimation import clip_propensity, lowrank_imputation  # noqa: E402
from drup.propagation import degree_weights, edge_estimate, three_hop  # noqa: E402


def props(ru, ri, Y, density, kappa, gamma, pmin):
    def z(x):
        x = np.clip(x, 1e-4, 1 - 1e-4)
        lx = np.log(x / (1 - x))
        return torch.as_tensor((lx - lx.mean()) / lx.std(), dtype=DT)
    base = z(ru)[:, None] + kappa * z(ri)[None, :] + gamma * (Y - Y.mean())
    lo, hi = -15.0, 5.0
    for _ in range(60):
        c = (lo + hi) / 2
        if float(torch.sigmoid(base + c).mean()) > density:
            hi = c
        else:
            lo = c
    return torch.sigmoid(base + c).clamp(pmin, 0.95)


def overlap(S, F, O, k=20):
    big = -1e18
    a = torch.topk(torch.where(O > 0, torch.full_like(S, big), S), k, dim=1).indices
    b = torch.topk(torch.where(O > 0, torch.full_like(F, big), F), k, dim=1).indices
    hit = (a[:, :, None] == b[:, None, :]).any(2).double().sum(1) / k
    return float(hit.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--density", type=float, default=0.13)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--kappas", default="0.5,1.5,3.0")
    ap.add_argument("--gammas", default="0,1")
    ap.add_argument("--betas", default="0.3,1,3,10,1000")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/v3/rank_dose.json")
    a = ap.parse_args()
    betas = [float(b) for b in a.betas.split(",")]
    Y, ru, ri, _ = load()
    m, n = Y.shape
    Cfix = degree_weights(Y, 0.5, D=Y)
    T3 = three_hop(Y, Cfix, correct=False)
    c1, c3 = float((Cfix * Y).abs().mean()), float(T3.abs().mean())
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for kappa in [float(x) for x in a.kappas.split(",")]:
        for gamma in [float(x) for x in a.gammas.split(",")]:
            cell = f"k{kappa}_g{gamma}_d{a.density}"
            if cell in res and len(res[cell]["rows"]) >= a.reps:
                continue
            P = props(ru, ri, Y, a.density, kappa, gamma, 0.02)
            Pb = clip_propensity(P, a.tau)
            g = torch.Generator().manual_seed(a.seed)
            rows = []
            t0 = time.time()
            for rep in range(a.reps):
                O = (torch.rand(m, n, generator=g, dtype=DT) < P).to(DT)
                O2 = (torch.rand(m, n, generator=g, dtype=DT) < P).to(DT)
                Yh = lowrank_imputation(O2, Y, Pb, rank=16, ridge=5.0)
                out = {}
                Ws = {e: edge_estimate(O, Y, Pb, Yh if e == "DR" else None) for e in ("IPS", "DR")}
                Ts = {(e, c): three_hop(Ws[e], Cfix, correct=c) for e in Ws for c in (False, True)}
                for beta in betas:
                    F = Cfix * Y / c1 + beta * T3 / c3
                    for (e, c), T in Ts.items():
                        s = Cfix * Ws[e] / c1 + beta * T / c3
                        nm = {("IPS", False): "IPS", ("IPS", True): "IPS+WC", ("DR", False): "DR", ("DR", True): "DRUP"}[(e, c)]
                        out[f"{nm}|{beta}|ndcg"] = ndcg_unexposed(s, Y, O)
                        out[f"{nm}|{beta}|ovl"] = overlap(s, F, O)
                rows.append(out)
            res[cell] = {"kappa": kappa, "gamma": gamma, "density": a.density,
                         "median_p": float(P.median()), "rows": rows}
            json.dump(res, open(a.out, "w"))
            print(cell, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
