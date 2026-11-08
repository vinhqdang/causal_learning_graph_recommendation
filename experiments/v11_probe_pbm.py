"""v11 probe P3+ (paper/v7/DESIGN.md Section 3): variants of the detection x attractiveness item appeal (DAX).

Model on the pre-cut standard log (temporal protocol): an impression of item i to user u in tab t is clicked iff the
user examines it (probability theta_u * kappa_t) and the item is attractive to the user's stratum (alpha_{i,s(u)}):

    P(click | u, i, t) = theta_u * kappa_t * alpha_{i, s(u)}

fitted by EM (latent examination / attraction, the regression-EM of click models; the occupancy model's detection
x presence factorisation). Priors (all MAP, Beta pseudo-counts):
    alpha_i      ~ centred on a content prior c_i, strength `prior` (c_i = tag mean, or a Fay-Herriot style
                   regression of logit alpha on content covariates)
    alpha_{i,s}  ~ centred on alpha_i, strength `prior_s` (MRP-style cells; None = no stratum deviation)
    theta_u      ~ centred on a scaled lifelong click rate of u, strength `k_life` (0 = in-pool data only)
    kappa_t      tab factor (tab-specific examination; fixed to 1 when tab=False); the test is scored at tab 1.
Score added to the reproduced LEAF scores: z(logit alpha_{i, s(u)}) per user, weight on validation.
python3 experiments/v11_probe_pbm.py      outputs results/v11_probes/probe_pbm.{txt,json}
"""
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v11_common import KEY, OUTDIR, ROOT, Ctx, Out, bundle, fmt, pair, stored  # noqa: E402
from v11_probe_items import GAMMAS, PerSplit, item_covariates, user_strata  # noqa: E402

SC = os.environ.get("V11_SCORES", "/tmp/v11_scores")
JS = os.path.join(OUTDIR, "probe_pbm.json")
CONTENT = ("tag", "upload_type", "video_type", "dur", "aspect", "music_type", "a_imp", "a_ctr", "a_lvr")


def lifelong_rate(B):
    L = np.load(os.path.join(ROOT, "data", "raw", "lifelong_27k.npz"), allow_pickle=True)
    tot = pd.DataFrame(L["tot_pre"], index=L["tot_pre_user"], columns=["n", "c", "l"]).reindex(B["users_raw"])
    r = ((tot.c + 1) / (tot.n + 2)).fillna(((tot.c + 1) / (tot.n + 2)).mean()).values
    return torch.as_tensor(r, dtype=torch.float64)


def content_centre(al, ni, cov, ridge=10.0):
    """Fay-Herriot style prior mean: weighted ridge regression of logit(alpha) on one-hot content covariates."""
    n = len(al)
    cols, off = [], 0
    for g in CONTENT:
        v = cov[g]
        cols.append(v + off)
        off += int(v.max()) + 1
    idx = torch.stack(cols, 1)                                     # (n, G)
    X = torch.zeros(n, off, dtype=torch.float64)
    X.scatter_(1, idx, 1.0)
    y = torch.logit(al)
    w = (ni + 1.0).sqrt()
    mu = (w * y).sum() / w.sum()
    Xw = X * w[:, None]
    beta = torch.linalg.solve(Xw.T @ X + ridge * torch.eye(off, dtype=torch.float64), Xw.T @ (y - mu))
    return torch.sigmoid(mu + X @ beta)


def fit_dax(B, cov, strata, prior=80.0, centre="tag", prior_s=None, k_life=0.0, tab=False, iters=40, life=None):
    lg = B["log"]
    m, n = B["shape"]
    u = torch.as_tensor(lg.u.values.astype(np.int64))
    i = torch.as_tensor(lg.i.values.astype(np.int64))
    tb = torch.as_tensor(lg.tab.values.astype(np.int64))
    y = torch.as_tensor(lg.is_click.values, dtype=torch.float64)
    S = int(strata.max()) + 1
    su = strata[u]
    T = int(tb.max()) + 1
    th = torch.full((m,), 0.7, dtype=torch.float64)
    ka = torch.ones(T, dtype=torch.float64)
    al = torch.full((n,), 0.6, dtype=torch.float64)
    als = torch.full((n, S), 0.6, dtype=torch.float64)
    tag = cov["tag"]
    G = int(tag.max()) + 1
    nu = torch.bincount(u, minlength=m).double()
    ni = torch.bincount(i, minlength=n).double()
    nis = torch.zeros(n * S, dtype=torch.float64).index_add_(0, i * S + su, torch.ones_like(y)).view(n, S)
    if k_life > 0:
        # prior centre of theta: lifelong click rate rescaled to the in-pool mean examination (updated each round)
        lr = life / life.mean()
    for it in range(iters):
        a = als[i, su] if prior_s is not None else al[i]
        t = (th[u] * ka[tb]).clamp(1e-6, 1 - 1e-6)
        den = (1 - t * a).clamp_min(1e-9)
        pE = torch.where(y > 0, torch.ones_like(t), t * (1 - a) / den)
        pR = torch.where(y > 0, torch.ones_like(t), (1 - t) * a / den)
        # theta_u (examination of the user, given the tab factor)
        sE = torch.zeros(m, dtype=torch.float64).index_add_(0, u, pE)
        sK = torch.zeros(m, dtype=torch.float64).index_add_(0, u, ka[tb])
        if k_life > 0:
            c0 = (lr * (sE.sum() / sK.sum())).clamp(1e-3, 1.0)
            th = ((sE + k_life * c0) / (sK + k_life)).clamp(1e-3, 1.0)
        else:
            th = ((sE + 1.0) / (sK + 2.0)).clamp(1e-3, 1.0)
        if tab:
            kE = torch.zeros(T, dtype=torch.float64).index_add_(0, tb, pE)
            kN = torch.zeros(T, dtype=torch.float64).index_add_(0, tb, th[u])
            ka = (kE + 1.0) / (kN + 1.0)
            ka = ka / ka[1]                                            # tab 1 (the test scenario) is the reference
        # alpha_i with a content-centred prior
        sR = torch.zeros(n, dtype=torch.float64).index_add_(0, i, pR)
        if centre == "tag" or it < 5:
            tm = (torch.zeros(G, dtype=torch.float64).index_add_(0, tag, sR) + 1.0) / \
                 (torch.zeros(G, dtype=torch.float64).index_add_(0, tag, ni) + 2.0)
            c = tm[tag]
        else:
            c = content_centre(al, ni, cov)
        al = ((sR + prior * c) / (ni + prior)).clamp(1e-4, 1 - 1e-4)
        if prior_s is not None:
            sRs = torch.zeros(n * S, dtype=torch.float64).index_add_(0, i * S + su, pR).view(n, S)
            als = ((sRs + prior_s * al[:, None]) / (nis + prior_s)).clamp(1e-4, 1 - 1e-4)
    return torch.logit(al).float(), (torch.logit(als).float() if prior_s is not None else None), ka


def main():
    out = Out(os.path.join(OUTDIR, "probe_pbm.txt"))
    out("=" * 100)
    out(f"v11 DAX (detection x attractiveness) probes [{time.strftime('%Y-%m-%d %H:%M')}]")
    t0 = time.time()
    B = bundle()
    C = Ctx(B)
    m, n = B["shape"]
    cov = item_covariates(B)
    strata = user_strata(B, m)
    life = lifelong_rate(B)
    ST = stored()
    res = json.load(open(JS)) if os.path.exists(JS) else {}
    bases = {bt: [C.rowstd(torch.load(os.path.join(SC, f"{bt}_s{s}.pt"))) for s in range(5)] for bt in ("lgn", "tied")}

    def report(name, rec):
        res[name] = rec
        json.dump(res, open(JS + ".tmp", "w"))
        os.replace(JS + ".tmp", JS)
        out(f"  {name:52s} test {rec['test']:.4f}  chosen {[c for c in rec['chosen']]}")
        for ref in ("QP family", "LEAF-LGN", "r-AdjNorm"):
            out(f"      - {ref:10s} {fmt(pair(rec, ST[ref]))}")

    def score(a, als):
        if als is None:
            return C.rowstd(C.item(a))
        return C.rowstd(C.table(lambda rows, cand: als[cand, strata[rows][:, None].expand_as(cand)]))

    def run(name, grid, base_tags=("lgn",)):
        cands = []
        for cfg in grid:
            t1 = time.time()
            a, als, ka = fit_dax(B, cov, strata, life=life, **cfg)
            cands.append((cfg, score(a, als)))
            out(f"    fit {cfg} [{time.time() - t1:.0f}s]" + (f" tab factors {ka.numpy().round(2)}" if cfg.get("tab") else ""))
        for bt in base_tags:
            sel = PerSplit(C)
            for cfg, Z in cands:
                for g in GAMMAS:
                    sel.add(bases[bt], [g * z for z in Z] if g else None, dict(cfg, gamma=g))
            report(f"LEAF-{bt.upper()} + {name}", sel.record())
        return cands

    P = (40.0, 80.0, 160.0)
    run("DAX tag prior (P3 re-run)", [dict(prior=p) for p in P])
    run("DAX content (Fay-Herriot) prior", [dict(prior=p, centre="content") for p in P])
    run("DAX + lifelong theta prior", [dict(prior=80.0, k_life=k) for k in (20.0, 100.0, 500.0)])
    run("DAX + tab examination", [dict(prior=p, tab=True) for p in P])
    run("DAX + stratum cells (MRP)", [dict(prior=80.0, prior_s=ps) for ps in (20.0, 80.0, 320.0)])
    # joint: every component, small grid
    grid = [dict(prior=p, centre=c, k_life=k, tab=t, prior_s=ps) for p, c, k, t, ps in
            itertools.product((80.0, 160.0), ("tag", "content"), (0.0, 100.0), (False, True), (None, 80.0))]
    run("DAX full grid", grid, ("lgn", "tied"))
    out(f"done [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
