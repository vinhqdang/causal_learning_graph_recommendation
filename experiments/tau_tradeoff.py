"""Accuracy vs exposure invariance as a function of the propensity clip tau.

Theorem 4 needs pi * p >= tau: when most propensities lie below the clip, the
edge estimates collapse to the logged graph up to a constant and exposure
invariance is lost, while variance falls (Theorem 3). For the DRUP
configuration selected on validation we vary only tau and report nDCG on the
unbiased test data and the rank shift of items whose exposures are thinned by
a real do(p <- p/2) intervention (0 = invariant; logged graph for reference).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from drup.data import load_coat, load_yahoo  # noqa: E402
from drup.estimation import get_propensity  # noqa: E402
from drup.metrics import evaluate  # noqa: E402
from drup.estimation import clip_propensity  # noqa: E402
from drup.pipeline import build, scores_with_consts  # noqa: E402
from drup.propagation import edge_estimate  # noqa: E402
from run_fat import chosen_config, within_user_rank  # noqa: E402


def shift(S0, S1, test, row_of, n, treated):
    a0, c0 = within_user_rank(S0, test, row_of, n)
    a1, c1 = within_user_rank(S1, test, row_of, n)
    ok = c0 > 0
    tr, ct = treated & ok, (~treated) & ok
    return (a1[tr].sum() / c1[tr].sum() - a0[tr].sum() / c0[tr].sum()) \
        - (a1[ct].sum() / c1[ct].sum() - a0[ct].sum() / c0[ct].sum())


def score_elasticity(S0, S1, test, row_of, treated):
    """Theorem 4 is about expected scores: estimate
    eta = [log mean_t(S1)/mean_t(S0) - log mean_c(S1)/mean_c(S0)] / log(1/2)
    over the unbiased candidate entries of treated (t) and control (c) items.
    eta = 1 for exposure-proportional scores, 0 for exposure invariance."""
    acc = {"t0": 0.0, "t1": 0.0, "c0": 0.0, "c1": 0.0}
    A0, A1 = S0.numpy(), S1.numpy()
    for u, items, _ in test:
        if len(items) == 0:
            continue
        r = row_of[u]
        tr = treated[items]
        acc["t0"] += A0[r, items[tr]].sum()
        acc["t1"] += A1[r, items[tr]].sum()
        acc["c0"] += A0[r, items[~tr]].sum()
        acc["c1"] += A1[r, items[~tr]].sum()
    return float((np.log(acc["t1"] / acc["t0"]) - np.log(acc["c1"] / acc["c0"])) / np.log(0.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="yahoo")
    ap.add_argument("--prop", default="pop")
    ap.add_argument("--filters_json", default=None)
    ap.add_argument("--method", default="DRUP")
    ap.add_argument("--taus", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--deg", default=None, help="override the degree source (W | Yhat)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--frozen", action="store_true",
                    help="keep C and Yhat fitted on the original log (Assumption A2)")
    a = ap.parse_args()
    dt = getattr(torch, a.dtype)
    if a.dataset == "coat":
        d, kn = load_coat(), 5
    elif a.dataset == "yahoo":
        d, kn = load_yahoo(), 5
    else:
        d, kn = torch.load("data/raw/kuairec.pt", weights_only=False), 20
    O, Y = d["O"].to(dt), d["Y"].to(dt)
    m, n = O.shape
    P_raw = get_propensity(d, O, Y, a.prop)
    test = d["test"]
    rows_users = sorted({u for u, _, _ in test})
    rows = torch.tensor(rows_users)
    row_of = {u: k for k, u in enumerate(rows_users)}
    base = chosen_config(a.filters_json or f"results/filters_{a.dataset}_{a.prop}.json", a.method)
    if a.deg:
        base = dict(base, deg=a.deg)
    print("base config", base, " propensity quantiles on logged pairs:",
          np.quantile(P_raw[O > 0].numpy(), [0.1, 0.5, 0.9]).round(4).tolist(), flush=True)
    g = torch.Generator().manual_seed(123)
    interventions = []
    for _ in range(a.reps):
        treated = torch.rand(n, generator=g) < 0.5
        keep = (torch.rand(m, n, generator=g) < 0.5).to(dt)
        O2 = O * torch.where(treated[None, :], keep, torch.ones_like(keep))
        P2 = torch.where(treated[None, :], P_raw * 0.5, P_raw)
        interventions.append((treated.numpy(), O2, P2))
    out = {"base": base, "rows": []}
    settings = [("Obs", dict(base))] + [(a.method, dict(base, floor=t)) for t in a.taus]
    for mth, cfg in settings:
        S = scores_with_consts(build(O, Y, P_raw, mth, cfg), rows)[0]
        nd = evaluate(S, test, (kn,), row_of)[f"ndcg@{kn}"]
        sh, el = [], []
        M0 = build(O, Y, P_raw, mth, cfg) if a.frozen else None
        for tr, O2, P2 in interventions:
            if a.frozen:
                # same nuisances as on the original log; only W sees the intervention
                P2c = torch.ones_like(O2) if mth == "Obs" else clip_propensity(P2, cfg["floor"])
                W2 = O2 * Y if mth == "Obs" else edge_estimate(O2, Y, P2c, M0["Yhat"], cfg.get("cv", 1.0))
                M2 = dict(M0, W=W2)
                S2 = scores_with_consts(M2, rows)[0]
            else:
                S2 = scores_with_consts(build(O2, Y, P2, mth, cfg), rows)[0]
            sh.append(shift(S, S2, test, row_of, n, tr))
            el.append(score_elasticity(S, S2, test, row_of, tr))
        frac_clipped = float((P_raw[O > 0] < cfg.get("floor", 0.0)).double().mean()) if mth != "Obs" else 1.0
        r = {"method": mth, "tau": cfg.get("floor"), f"ndcg@{kn}": nd, "shift": float(np.mean(sh)),
             "shift_std": float(np.std(sh)), "elasticity": float(np.mean(el)),
             "elasticity_std": float(np.std(el)), "frac_logged_pairs_clipped": frac_clipped}
        out["rows"].append(r)
        print(r, flush=True)
    with open(f"results/tau_tradeoff_{a.dataset}{a.tag}.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
