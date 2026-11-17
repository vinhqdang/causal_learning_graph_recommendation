"""EM approximation check of DAX on KuaiRand-27K pre-cut data (v16, Part 2).

The theta and kappa M-steps of drup.dax.fit_dax are ratio forms, theta_u = (sum P(E=1) + k c_u) / (sum kappa + k) and
kappa_t = (sum P(E=1) + 1) / (sum theta + 1) (normalised by kappa_1). They are the exact maximisers only under a Poisson
approximation (or when kappa is shared by all impressions of a user). This script measures the gap:

(0) the fitted configuration: the DAX configuration chosen most often in results/v11/dax_lgn_post.json (LEAF-LGN+DAX);
    the old path of drup/dax.py is checked bit-identical on the full data against the HEAD copy of the module;
(i)   at the fitted values, the E-step P(E=1) is recomputed and both M-steps (ratio and exact Bernoulli) are applied to it:
      distribution of the relative differences of theta and of kappa, and the Spearman correlation of the two thetas;
(ii)  the exact-M-step fit (exact_mstep=True): Spearman / Pearson of logit alpha between the two fits, and the observed-data
      Bernoulli log-likelihood of the pre-cut impressions under each fit, log P(y | theta kappa alpha);
(iii) homogeneity of kappa: the share of each user's pre-cut impressions on the user's most common tab (the scenario),
      and the pre-cut click rate.

Writes the report to --out (results/v16/em_check.txt) and prints it. Only the pre-cut standard log is read.
"""

import argparse
import collections
import importlib.util
import json
import os
import resource
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from drup import dax as DX  # noqa: E402
from drup.data2 import load_kuairand_rich  # noqa: E402
from drup.leaf import CUT_MS  # noqa: E402

FIT_FILE = os.path.join(ROOT, "results", "v11", "dax_lgn_post.json")
HEAD_DAX = None   # path of the HEAD copy of drup/dax.py (--head_dax)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def chosen_config(path):
    chosen = json.load(open(path))["results"]["LEAF-LGN+DAX"]["chosen"]
    cnt = collections.Counter(json.dumps(c["dax"], sort_keys=True) for c in chosen)
    key, k = cnt.most_common(1)[0]
    return json.loads(key), k, len(chosen), cnt


def dist(x):
    x = np.asarray(x, dtype=np.float64)
    return f"median {np.median(x):.4g}, p90 {np.percentile(x, 90):.4g}, max {np.max(x):.4g}"


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def bound_diag(label, pE, th_in, ka_in, u, tb, m, T, pos, neg, say):
    """Where the exact steps sit on their brackets. kappa_t is bounded by (1 - 1e-9) / max_j theta_j over the tab (every
    p_j = theta_j kappa_t must stay below 1); theta_u by min(1, (1 - 1e-9) / max_j kappa_j) over the user's impressions.
    A maximiser on the bracket edge means the Bernoulli objective keeps increasing up to the model constraint p <= 1."""
    tu = th_in[u]
    mt = np.zeros(T)
    np.maximum.at(mt, tb, tu)
    hi_t = (1 - 1e-9) / np.maximum(mt, 1e-12)
    kr = DX.exact_kappa(torch.as_tensor(pE), torch.as_tensor(tb), torch.as_tensor(tu), T).numpy()
    at_k = np.flatnonzero(kr >= hi_t * (1 - 1e-6))
    mk = np.zeros(m)
    np.maximum.at(mk, u, ka_in[tb])
    hi_u = np.minimum(1.0, (1 - 1e-9) / np.maximum(mk, 1e-12))
    th_ex = DX.exact_theta(torch.as_tensor(pE), torch.as_tensor(u), torch.as_tensor(ka_in[tb]), m,
                           torch.as_tensor(pos), torch.as_tensor(neg)).numpy()
    has = np.bincount(u, minlength=m) > 0
    at_u = has & (th_ex >= hi_u - 1e-6)
    say(f"  bounds [{label}]: tabs with exact kappa on its bracket edge {at_k.tolist()} "
        f"(raw kappa {np.round(kr[at_k], 4).tolist()}, edge {np.round(hi_t[at_k], 4).tolist()}); "
        f"users with exact theta on its edge {int(at_u.sum())} "
        f"(edge 1 for {int(np.sum(at_u & (hi_u >= 1 - 1e-9)))}, edge (1-1e-9)/max kappa for {int(np.sum(at_u & (hi_u < 1 - 1e-9)))})")
    return kr


def loglik(y, th, ka, las, u, i, tb, su):
    """Observed-data Bernoulli log-likelihood sum_j log P(y_j | p_j alpha_j), p_j = theta_u kappa_t, alpha the stratum
    alpha of the impression's user (the same alpha the EM uses)."""
    p = th[u] * ka[tb]
    a = sigmoid(las[i, su])
    q = np.clip(p * a, 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(q) + (1 - y) * np.log(1 - q)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "v16", "em_check.txt"))
    ap.add_argument("--head_dax", default=None, help="path of the HEAD copy of drup/dax.py (bit-identity check)")
    ap.add_argument("--src", default=FIT_FILE)
    a = ap.parse_args()
    L = []

    def say(*x):
        s = " ".join(str(v) for v in x)
        print(s, flush=True)
        L.append(s)

    t_all = time.time()
    cfg, nch, S, cnt = chosen_config(a.src)
    say("EM check of DAX (v16, Part 2)")
    say(f"source: {os.path.relpath(a.src, ROOT)}; chosen DAX configurations over {S} splits:")
    for k, v in cnt.most_common():
        say(f"  {v} x {k}")
    say(f"fitted configuration (most frequent, {nch} of {S}): {json.dumps(cfg, sort_keys=True)}")

    d = load_kuairand_rich(version="pure", n_users=None, train_until=CUT_MS, dense_dtype=torch.float32,
                           with_signals=False)
    log = DX.pre_cut(d["log"])
    m, n = len(d["users_raw"]), len(d["item_feat"])
    tag_codes = DX.tag_codes(d["item_feat"])
    life = DX.lifelong_rate(d["users_raw"])
    strata = DX.user_strata(d["users_raw"])
    content = DX.item_content_codes(d["item_feat"], d["items_raw"])
    u = log.u.values.astype(np.int64)
    i = log.i.values.astype(np.int64)
    tb = log.tab.values.astype(np.int64)
    y = log.is_click.values.astype(np.float64)
    su = strata[u]
    T = int(tb.max()) + 1
    N = len(y)
    say(f"data: {m} users, {n} items, {N} pre-cut standard impressions, {T} tabs; "
        f"loaded in {time.time() - t_all:.0f}s")
    kw = dict(content=content, life=life, strata=strata)

    # (0) old path: the ratio form is bit-identical to the HEAD module on the full data
    if a.head_dax:
        spec = importlib.util.spec_from_file_location("dax_head", a.head_dax)
        H = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(H)
        ref = H.fit_dax_dataset(d, **kw, **cfg)
        new = DX.fit_dax_dataset(d, **kw, **cfg)
        same = all(torch.equal(x, z) for x, z in zip(ref[:3], new[:3])) and torch.equal(ref[3], new[3])
        say(f"(0) default (ratio) path vs HEAD module on the full data: bit-identical = {same}")

    # ratio fit (the production arm)
    t0 = time.time()
    la_t, th_t, ka_t, las_t = DX.fit_dax_dataset(d, **kw, **cfg)
    la, th, ka, las = la_t.double().numpy(), th_t.double().numpy(), ka_t.numpy(), las_t.double().numpy()
    say(f"ratio fit: {time.time() - t0:.1f}s, kappa (normalised, kappa_1 = 1) = {np.round(ka, 4).tolist()}")

    # (i) both M-steps at the fitted values: E-step from (theta, kappa, alpha) of the fit
    alpha = sigmoid(las[i, su])
    t = np.clip(th[u] * ka[tb], 1e-6, 1 - 1e-6)
    den = np.maximum(1 - t * alpha, 1e-9)
    pE = np.where(y > 0, 1.0, t * (1 - alpha) / den)
    lr = life / life.mean()
    sE = np.bincount(u, weights=pE, minlength=m)
    sK = np.bincount(u, weights=ka[tb], minlength=m)
    k = float(cfg["k_life"])
    if k > 0:
        c0 = np.clip(lr * (sE.sum() / sK.sum()), 1e-3, 1.0)
        th_ratio = np.clip((sE + k * c0) / (sK + k), 1e-3, 1.0)
        pos, neg = k * c0, k * (1 - c0)
    else:
        th_ratio = np.clip((sE + 1.0) / (sK + 2.0), 1e-3, 1.0)
        pos, neg = np.ones(m), np.ones(m)
    pt = torch.as_tensor(pE)
    th_exact = DX.exact_theta(pt, torch.as_tensor(u), torch.as_tensor(ka[tb]), m,
                              torch.as_tensor(pos), torch.as_tensor(neg)).numpy()
    has = np.bincount(u, minlength=m) > 0
    rel_th = np.abs(th_ratio - th_exact) / th_exact
    say("(i) M-steps at the fitted values (same P(E=1) and kappa for both)")
    say(f"  users {m}, with impressions {int(has.sum())}; P(E=1) over impressions: mean {pE.mean():.4f}")
    say(f"  theta relative difference |ratio - exact| / exact, all users: {dist(rel_th)}")
    say(f"  theta relative difference, users with impressions: {dist(rel_th[has])}")
    say(f"  theta absolute difference: median {np.median(np.abs(th_ratio - th_exact)):.4g}, "
        f"max {np.max(np.abs(th_ratio - th_exact)):.4g}")
    say(f"  theta at the bounds: ratio {int(np.sum((th_ratio <= 1e-3) | (th_ratio >= 1.0)))}, "
        f"exact {int(np.sum((th_exact <= 1e-3) | (th_exact >= 1.0)))} users")
    say(f"  Spearman(ratio theta, exact theta) = {spearman(th_ratio, th_exact):.6f}; "
        f"Pearson = {pearson(th_ratio, th_exact):.6f}")
    say(f"  fitted theta (last ratio step) vs exact at the same P(E=1): Spearman = {spearman(th, th_exact):.6f}")
    # kappa at the fitted theta (same input for both forms)
    kE = np.bincount(tb, weights=pE, minlength=T)
    kN = np.bincount(tb, weights=th[u], minlength=T)
    kap_ratio = (kE + 1.0) / (kN + 1.0)
    kap_ratio = kap_ratio / kap_ratio[1]
    kap_exact_raw = DX.exact_kappa(pt, torch.as_tensor(tb), torch.as_tensor(th[u]), T).numpy()
    kap_exact = kap_exact_raw / kap_exact_raw[1]
    rel_k = np.abs(kap_ratio - kap_exact) / kap_exact
    say(f"  kappa (normalised) ratio {np.round(kap_ratio, 5).tolist()}")
    say(f"  kappa (normalised) exact {np.round(kap_exact, 5).tolist()}")
    say(f"  kappa relative difference per tab: {np.round(rel_k, 5).tolist()}")
    bound_diag("(i) fitted values", pE, th, ka, u, tb, m, T, pos, neg, say)

    # (ii) exact-M-step fit
    t0 = time.time()
    la_e_t, th_e_t, ka_e_t, las_e_t = DX.fit_dax_dataset(d, **kw, **DX.dax_variant(cfg, "exact_em"))
    la_e, th_e, ka_e = la_e_t.double().numpy(), th_e_t.double().numpy(), ka_e_t.numpy()
    las_e = las_e_t.double().numpy()
    say(f"(ii) exact-M-step fit: {time.time() - t0:.1f}s, kappa = {np.round(ka_e, 4).tolist()}")
    say(f"  logit alpha (item level), Spearman(ratio, exact) = {spearman(la, la_e):.6f}; "
        f"Pearson = {pearson(la, la_e):.6f}")
    say(f"  logit alpha (item x stratum), Spearman = {spearman(las.ravel(), las_e.ravel()):.6f}; "
        f"Pearson = {pearson(las.ravel(), las_e.ravel()):.6f}")
    say(f"  theta: Spearman(ratio fit, exact fit) = {spearman(th, th_e):.6f}")
    ll_r = loglik(y, th, ka, las, u, i, tb, su)
    ll_e = loglik(y, th_e, ka_e, las_e, u, i, tb, su)
    say(f"  observed-data Bernoulli log-likelihood of the {N} pre-cut impressions: ratio fit {ll_r:.2f} "
        f"(mean {ll_r / N:.6f}); exact fit {ll_e:.2f} (mean {ll_e / N:.6f}); exact - ratio = {ll_e - ll_r:.2f}")

    # the exact fit's own E-step state: where its M-steps sit on the brackets
    alpha_e = sigmoid(las_e[i, su])
    t_e = np.clip(th_e[u] * ka_e[tb], 1e-6, 1 - 1e-6)
    den_e = np.maximum(1 - t_e * alpha_e, 1e-9)
    pE_e = np.where(y > 0, 1.0, t_e * (1 - alpha_e) / den_e)
    sE_e = np.bincount(u, weights=pE_e, minlength=m)
    sK_e = np.bincount(u, weights=ka_e[tb], minlength=m)
    if k > 0:
        c0_e = np.clip(lr * (sE_e.sum() / sK_e.sum()), 1e-3, 1.0)
        pos_e, neg_e = k * c0_e, k * (1 - c0_e)
    else:
        pos_e, neg_e = np.ones(m), np.ones(m)
    bound_diag("(ii) exact fit, its own E-step state", pE_e, th_e, ka_e, u, tb, m, T, pos_e, neg_e, say)

    # (iii) homogeneity of kappa per user and pre-cut click rate
    cnt_ut = pd.crosstab(u, tb)
    tot = cnt_ut.sum(1).values
    mx = cnt_ut.max(1).values
    share = mx / tot
    say("(iii) scenario homogeneity per user (tab = scenario; users with pre-cut impressions)")
    say(f"  users with impressions {len(tot)}; tabs with impressions {T}; "
        f"impressions per user median {np.median(tot):.0f}, mean {tot.mean():.1f}")
    say(f"  share of the user's impressions on the user's most common tab: median {np.median(share):.4f}, "
        f"mean {share.mean():.4f}, p10 {np.percentile(share, 10):.4f}, p90 {np.percentile(share, 90):.4f}")
    say(f"  impression-weighted share on the most common tab: {mx.sum() / tot.sum():.4f}; "
        f"users with a single tab: {100 * np.mean(share == 1):.2f}%")
    say(f"  tab shares over all pre-cut impressions: "
        f"{np.round(np.bincount(tb, minlength=T) / N, 4).tolist()}")
    say(f"  pre-cut click rate: {y.mean():.5f} ({int(y.sum())} clicks / {N} impressions); "
        f"user click rate median {np.median(np.bincount(u, weights=y, minlength=m)[has] / np.bincount(u, minlength=m)[has]):.4f}")

    say(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0:.0f} MB; "
        f"total {time.time() - t_all:.0f}s")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
