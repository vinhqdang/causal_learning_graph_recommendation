"""Monte-Carlo check of the experimental protocol against Assumption 2.

Theorem 1 needs nuisances (propensities, imputation, degree weights, and the
constants a, b of the score) that do not depend on the log. The experiments
instead cross-fit the nuisances over ten folds of pairs on the same log, may
take degrees from cross-fitted edge estimates (deg = 'Wx'), and scale the one-
and three-hop terms by constants c1, c3 computed from the scores. Cross-fitting
over pairs removes the dependence of a nuisance on its own pair, but a product
of several edge estimates can still depend on the nuisances through the other
edges. This script measures the resulting bias.

For every draw of the log O ~ Bernoulli(P) the nuisances are fitted
  indep    on an independent log O' of the same distribution (Assumption 2),
  xfit     on the same log, cross-fitted over ten folds of pairs (protocol),
  insample on the same log without cross-fitting,
  splitQ   on a random fraction Q% of the pairs of the same log (sample split);
           the edge estimates of those pairs are replaced by the imputation,
           so that the remaining edge estimates are independent of the
           nuisances and Theorem 1 applies with delta_e = Yhat_e - Y_e on the
           imputed pairs,
with the true propensities or with the logistic exposure model, degrees from
the imputation (Yhat) or cross-fitted edge estimates (Wx), and IPS or DR edges.
Bias is measured against the full-exposure target computed with the same
weights C, T* = (C*Y)(C*Y)^T(C*Y), and, for the full score
s = s1 / c1 + beta * s3 / c3, against F* with constants from the target graph.
Score constants are taken from the scores (data), or from the imputed graph
C*Yhat (a nuisance, so within Assumption 2 whenever the imputation is).
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drup.estimation import (baseline_imputation, clip_propensity, crossfit, fold_ids,  # noqa: E402
                             popularity_propensity)
from drup.pipeline import crossfit_degree_weights  # noqa: E402
from drup.propagation import degree_weights, edge_estimate, three_hop  # noqa: E402

DT = torch.float64


def setup(m, n, seed, pmult=1.0):
    g = torch.Generator().manual_seed(seed)
    U = torch.randn(m, 3, generator=g, dtype=DT)
    V = torch.randn(n, 3, generator=g, dtype=DT)
    Y = (torch.sigmoid(U @ V.T - 0.5) > torch.rand(m, n, generator=g, dtype=DT)).to(DT)
    act = torch.rand(m, generator=g, dtype=DT)
    pop = torch.rand(n, generator=g, dtype=DT) ** 2
    P = ((0.05 + 0.6 * act[:, None] * pop[None, :] + 0.1 * pop[None, :]) * pmult).clamp(0.03, 0.9)
    return g, Y, P


def fit_nuisance(O, Y, P, prop, tau, lam, folds):
    """(clipped propensity, imputation) fitted on log O; folds=None: no cross-fitting."""
    if prop == "true":
        Pr = P
    elif folds is None:
        Pr = popularity_propensity(O)[0]
    else:
        Pr = crossfit(lambda M: popularity_propensity(O, mask=M)[0], O, folds, 10)
    Pb = clip_propensity(Pr, tau)
    if folds is None:
        Yh = baseline_imputation(O, Y, Pb, lam=lam)
    else:
        Yh = crossfit(lambda M: baseline_imputation(O * M, Y, Pb, lam=lam), O, folds, 10)
    return Pb, Yh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=40)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--lam", type=float, default=5.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--pmult", type=float, default=1.0, help="scales all propensities")
    ap.add_argument("--sources", nargs="+", default=["indep", "xfit", "insample"])
    ap.add_argument("--props", nargs="+", default=["true", "pop"])
    ap.add_argument("--out", default="results/v2/mc_protocol.json")
    a = ap.parse_args()
    g, Y, P = setup(a.m, a.n, a.seed, a.pmult)
    print(f"mean exposures per user {float(P.sum(1).mean()):.1f}, median propensity {float(P.median()):.3f}",
          flush=True)
    sources = a.sources
    keys = [(src, prop, deg, edge) for src in sources for prop in a.props
            for deg in ("Yhat", "Wx") for edge in ("IPS", "DR")]
    keys = [k for k in keys if not (k[0].startswith("split") and k[2] == "Wx")]
    acc = {k: {n_: torch.zeros(a.m, a.n, dtype=DT) for n_ in ("T", "T0", "Th", "s_data", "s_yhat", "Fstar", "Tabs")}
           for k in keys}
    for rep in range(a.reps):
        O = (torch.rand(a.m, a.n, generator=g, dtype=DT) < P).to(DT)
        O2 = (torch.rand(a.m, a.n, generator=g, dtype=DT) < P).to(DT)
        folds = fold_ids(O.shape, 10, seed=rep)
        for src in sources:
            for prop in a.props:
                if src == "indep":
                    Pb, Yh = fit_nuisance(O2, Y, P, prop, a.tau, a.lam, None)
                elif src.startswith("split"):
                    A = (torch.rand(a.m, a.n, generator=g, dtype=DT) < int(src[5:]) / 100).to(DT)
                    if prop == "true":
                        Pb = clip_propensity(P, a.tau)
                    else:
                        Pb = clip_propensity(popularity_propensity(O, mask=A)[0], a.tau)
                    Yh = baseline_imputation(O * A, Y, Pb, lam=a.lam)
                else:
                    Pb, Yh = fit_nuisance(O, Y, P, prop, a.tau, a.lam, folds if src == "xfit" else None)
                for edge in ("IPS", "DR"):
                    W = edge_estimate(O, Y, Pb, Yh if edge == "DR" else None)
                    if src.startswith("split"):
                        W = torch.where(A > 0, Yh, W)      # pairs used for the nuisances are imputed
                    for deg in ("Yhat", "Wx"):
                        if src.startswith("split") and deg == "Wx":
                            continue
                        if deg == "Yhat":
                            C = degree_weights(W, a.alpha, D=Yh)
                        elif src == "indep":
                            # degrees from edge estimates of the independent log
                            C = degree_weights(W, a.alpha, D=edge_estimate(O2, Y, Pb, Yh if edge == "DR" else None))
                        elif src == "xfit":
                            C = crossfit_degree_weights(W, a.alpha, folds)
                        else:
                            C = degree_weights(W, a.alpha)
                        T = three_hop(W, C, correct=True)
                        T0 = three_hop(W, C, correct=False)
                        CY = C * Y
                        Ts = three_hop(Y, C, correct=False)
                        s1 = C * W
                        # constants from the scores (protocol) and from the imputed graph
                        c1d, c3d = s1.abs().mean(), T.abs().mean()
                        c1y, c3y = (C * Yh).abs().mean(), three_hop(Yh, C, correct=True).abs().mean()
                        c1s, c3s = CY.abs().mean(), Ts.abs().mean()
                        r = acc[(src, prop, deg, edge)]
                        if src.startswith("split"):
                            # hybrid target of Theorem 1: imputed pairs contribute Yhat
                            r["Th"] += T - three_hop(torch.where(A > 0, Yh, Y), C, correct=True)
                        r["T"] += T - Ts
                        r["T0"] += T0 - Ts
                        r["Tabs"] += Ts
                        r["s_data"] += (s1 / c1d + a.beta * T / c3d) - (CY / c1s + a.beta * Ts / c3s)
                        r["s_yhat"] += (s1 / c1y + a.beta * T / c3y) - (CY / c1y + a.beta * Ts / c3y)
                        r["Fstar"] += CY / c1s + a.beta * Ts / c3s
        if (rep + 1) % 500 == 0:
            print("rep", rep + 1, flush=True)
    out = {}
    for k in keys:
        r = acc[k]
        sc_T = (r["Tabs"] / a.reps).abs().mean()
        sc_F = (r["Fstar"] / a.reps).abs().mean()
        out["/".join(k)] = {
            "rel_bias_T_corrected": float((r["T"] / a.reps).abs().mean() / sc_T),
            "rel_bias_T_uncorrected": float((r["T0"] / a.reps).abs().mean() / sc_T),
            "rel_bias_s_data_consts": float((r["s_data"] / a.reps).abs().mean() / sc_F),
            "rel_bias_s_yhat_consts": float((r["s_yhat"] / a.reps).abs().mean() / sc_F),
        }
        if k[0].startswith("split"):
            out["/".join(k)]["rel_bias_T_vs_hybrid_target"] = float((r["Th"] / a.reps).abs().mean() / sc_T)
        print("/".join(k), json.dumps(out["/".join(k)]), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"config": vars(a), "results": out}, f, indent=1)


if __name__ == "__main__":
    main()
