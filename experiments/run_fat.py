"""Fairness / accountability / transparency / privacy experiments.

Sections (all on the unbiased test users):
  fairness     accuracy + PRU, exposure-conditional bias, Gini@K, coverage@K,
               user-group NDCG gap
  intervene    real-data intervention do(p_i <- p_i / 2): exposures of a random
               half of the items are thinned with probability 1/2 and the
               change of their mean within-user rank and the score elasticity
               eta are measured, with the nuisances (i) frozen, (ii) re-fitted
               with the known propensity change, (iii) re-fitted with
               re-estimated propensities
  explain      exactness of attributions, deletion curves, minimal
               counterfactual explanation sizes (DRUP, frozen nuisances)
  attack       shilling attack with frozen nuisances + certified bound, and a
               check of the certificates when the nuisances are re-fitted on
               the poisoned log
  privacy      jointly private DRUP: population-level nuisances fitted on a
               public 10% of the (non-test) users, G released with the
               analytic Gaussian mechanism, fixed denoising ranks

Every section uses one configuration per operator (the one selected most
often across the validation splits of run_filters.py) and all unbiased test
data, so its accuracy differs from the split-averaged accuracy table.
"""

import argparse
import collections
import gc
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drup import fat  # noqa: E402
from drup.data import load_coat, load_yahoo  # noqa: E402
from drup.estimation import clip_propensity, public_nuisance  # noqa: E402
from drup.metrics import evaluate  # noqa: E402
from drup.pipeline import Nuisance, build, scores_with_consts  # noqa: E402
from drup.propagation import edge_estimate  # noqa: E402
from drup.propagation import item_gram, local_three_hop  # noqa: E402

METHODS = ["Obs", "IPS", "DR", "DRUP"]


def chosen_config(path, method):
    res = json.load(open(path))["results"][method]["chosen"]
    cnt = collections.Counter(json.dumps(c, sort_keys=True) for c in res)
    return json.loads(cnt.most_common(1)[0][0])


def per_user_ndcg(S, test, row_of, k):
    out = {}
    for u, items, rel in test:
        if len(items) and rel.sum() > 0:
            out[u] = evaluate(S, [(u, items, rel)], (k,), row_of)[f"ndcg@{k}"]
    return out


def within_user_rank(S, test, row_of, n):
    """Mean within-user percentile rank (1 = top) of every item over users."""
    acc, cnt = np.zeros(n), np.zeros(n)
    Sn = S.numpy()
    for u, items, _ in test:
        if len(items) < 2:
            continue
        s = Sn[row_of[u], items]
        r = np.argsort(np.argsort(s)) / (len(items) - 1)
        acc[items] += r
        cnt[items] += 1
    return acc, cnt


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
    ap.add_argument("--dataset", default="coat")
    ap.add_argument("--prop", default="given")
    ap.add_argument("--sections", nargs="+", default=["fairness", "intervene", "explain", "attack", "privacy"])
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--n_explain", type=int, default=200)
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--filters_json", default=None)
    ap.add_argument("--tag", default="", help="suffix of the output file")
    ap.add_argument("--xfit", type=int, default=10)
    ap.add_argument("--refit_xfit", type=int, default=5, help="cross-fitting folds when re-fitting")
    ap.add_argument("--refit_attack", type=int, default=1, help="check certificates under re-fitting")
    a = ap.parse_args()
    dt = getattr(torch, a.dtype)
    if a.dataset == "coat":
        d = load_coat()
        K, kn = 5, 5
        feats = np.loadtxt("data/raw/coat/user_item_features/user_features.ascii")
        groups = {u: int(feats[u, 1] == 1) for u in range(feats.shape[0])}  # women = 1
        gname = "gender (men vs women)"
    elif a.dataset == "yahoo":
        d = load_yahoo()
        K, kn = 5, 5
        act = d["O"].sum(1).numpy()
        med = np.median(act[[u for u, _, _ in d["test"]]])
        groups = {u: int(act[u] > med) for u, _, _ in d["test"]}
        gname = "activity (inactive vs active)"
    else:
        d = torch.load("data/raw/kuairec.pt", weights_only=False)
        K, kn = 20, 20
        act = d["O"].sum(1).numpy()
        med = np.median(act[[u for u, _, _ in d["test"]]])
        groups = {u: int(act[u] > med) for u, _, _ in d["test"]}
        gname = "activity (inactive vs active)"
    O, Y = d["O"].to(dt), d["Y"].to(dt)
    m, n = O.shape
    nz = Nuisance(d, O, Y, a.prop, K=a.xfit, seed=0)
    P_raw = nz.P
    test = d["test"]
    rows_users = sorted({u for u, _, _ in test})
    rows = torch.tensor(rows_users)
    row_of = {u: k for k, u in enumerate(rows_users)}
    global METHODS
    if a.methods:
        METHODS = a.methods
    fpath = a.filters_json or f"results/v2/filters_{a.dataset}_{a.prop}.json"
    cfgs = {mth: chosen_config(fpath, mth) for mth in METHODS}
    item_pop = (O * Y).sum(0).numpy()
    # true item quality on the unbiased data
    qa, qc = np.zeros(n), np.zeros(n)
    for u, items, rel in test:
        qa[items] += rel
        qc[items] += 1
    quality = np.where(qc > 0, qa / np.maximum(qc, 1), 0.0)
    out = {"dataset": a.dataset, "configs": cfgs}
    path = f"results/v2/fat_{a.dataset}_{a.prop}{a.tag}.json"

    def save():
        """Merge this run's sections into the result file (after every section)."""
        merged = json.load(open(path)) if os.path.exists(path) else {}
        merged.update(out)
        with open(path, "w") as f:
            json.dump(merged, f, indent=1)

    # Only one model is kept in memory at a time (KuaiRec matrices are large).
    cache = {}

    def get_model(mth):
        if mth not in cache:
            cache.clear()
            gc.collect()
            cache[mth] = build(O, Y, nz, mth, cfgs[mth])
        return cache[mth]
    S, consts = {}, {}
    for mth in METHODS:
        S[mth], consts[mth] = scores_with_consts(get_model(mth), rows)

    if "fairness" in a.sections:
        res = {}
        for mth in METHODS:
            ev = evaluate(S[mth], test, (kn,), row_of)
            top = fat.topk_lists(S[mth], test, K, row_of)
            freq = np.zeros(n)
            for lst in top.values():
                freq[lst] += 1
            cand = np.zeros(n, dtype=bool)
            for _, items, _ in test:
                cand[items] = True
            pu = per_user_ndcg(S[mth], test, row_of, kn)
            gap, g0, g1 = fat.group_gap(pu, groups)
            # bootstrap over users (1,000 resamples) for the group gap
            us = np.array([u for u in pu if u in groups])
            vals = np.array([pu[u] for u in us])
            grp = np.array([groups[u] for u in us])
            rng_b = np.random.default_rng(0)
            boots = []
            for _ in range(1000):
                ix = rng_b.integers(0, len(us), len(us))
                v, gg = vals[ix], grp[ix]
                if (gg == 0).any() and (gg == 1).any():
                    boots.append(v[gg == 0].mean() - v[gg == 1].mean())
            gap_ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
            n_g = [int((grp == 0).sum()), int((grp == 1).sum())]
            res[mth] = {
                f"ndcg@{kn}": ev[f"ndcg@{kn}"],
                "PRU": fat.popularity_rank_correlation(S[mth], test, item_pop, row_of),
                "ECB": fat.exposure_conditional_bias(S[mth], test, item_pop, quality, row_of),
                f"gini@{K}": fat.gini(freq[cand]),
                f"coverage@{K}": float((freq[cand] > 0).mean()),
                "group_gap": gap, "group0": g0, "group1": g1, "group_gap_ci95": gap_ci, "group_sizes": n_g,
            }
            print("fairness", mth, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[mth].items()}, flush=True)
        out["fairness"] = {"groups": gname, "results": res}
        save()

    if "intervene" in a.sections:
        variants = ("frozen", "refit_known", "refit_estimated")
        res = {v: {mth: {"shift": [], "eta": []} for mth in METHODS} for v in variants}
        clip_frac = {mth: [] for mth in METHODS}
        g = torch.Generator().manual_seed(123)
        for r in range(a.reps):
            treated = torch.rand(n, generator=g) < 0.5
            keep = (torch.rand(m, n, generator=g) < 0.5).to(dt)
            thin = torch.where(treated[None, :], keep, torch.ones_like(keep))
            O2 = O * thin
            tr_np = treated.numpy()
            # (ii) known intervention: the propensity of treated items is halved
            P2 = torch.where(treated[None, :], P_raw * 0.5, P_raw)
            cache.clear()
            gc.collect()
            nz_known = Nuisance({"P_given": P2}, O2, Y, "given", K=a.refit_xfit, seed=0)
            # (iii) propensities re-estimated on the intervened log
            if a.prop == "given":
                nz_est = None
            else:
                nz_est = Nuisance(d, O2, Y, a.prop, K=a.refit_xfit, seed=0)
            for mth in METHODS:
                M0 = get_model(mth)
                cfg = cfgs[mth]
                for v in variants:
                    if v == "frozen":
                        if mth == "Obs":
                            W2 = O2 * Y
                        else:
                            P2c = clip_propensity(P2, cfg["floor"])
                            W2 = edge_estimate(O2, Y, P2c, M0["Yhat"], cfg.get("cv", 1.0))
                        S2 = scores_with_consts(dict(M0, W=W2), rows)[0]
                    elif v == "refit_known":
                        S2 = scores_with_consts(build(O2, Y, nz_known, mth, cfg), rows)[0]
                    else:
                        if nz_est is None:
                            continue
                        S2 = scores_with_consts(build(O2, Y, nz_est, mth, cfg), rows)[0]
                    a0, c0 = within_user_rank(S[mth], test, row_of, n)
                    a1, c1 = within_user_rank(S2, test, row_of, n)
                    ok = c0 > 0
                    tr = tr_np & ok
                    ct = (~tr_np) & ok
                    shift = (a1[tr].sum() / c1[tr].sum() - a0[tr].sum() / c0[tr].sum()) \
                        - (a1[ct].sum() / c1[ct].sum() - a0[ct].sum() / c0[ct].sum())
                    res[v][mth]["shift"].append(float(shift))
                    res[v][mth]["eta"].append(score_elasticity(S[mth], S2, test, row_of, tr_np))
                    del S2
                if mth != "Obs":
                    lp = (O2 > 0) & treated[None, :]
                    clip_frac[mth].append(float((P2[lp] < cfg["floor"]).double().mean()))
            del O2, P2, thin, keep, nz_known, nz_est
            gc.collect()
        out["intervene"] = {v: {mth: {k: {"mean": float(np.mean(x)), "std": float(np.std(x)), "all": x}
                                      for k, x in dct.items() if x}
                                for mth, dct in res[v].items()} for v in variants}
        out["intervene"]["frac_treated_logged_below_clip"] = {k: float(np.mean(v)) for k, v in clip_frac.items() if v}
        for v in variants:
            for mth in METHODS:
                r_ = out["intervene"][v][mth]
                if r_:
                    print("intervene", v, mth, {k: round(x["mean"], 4) for k, x in r_.items()}, flush=True)
        save()

    M = get_model("DRUP")
    W, C, Yhat = M["W"], M["C"], M["Yhat"]
    G = item_gram(W, C, correct=True)
    beta = M["cfg"]["beta"]
    c1, c3 = consts["DRUP"]

    def user_scores(u, w_row, Gloo):
        s3 = fat.exact_three_hop(w_row, C[u], Gloo)
        s1 = C[u] * w_row
        return s3 / c3 if beta >= 1e3 else s1 / c1 + beta * s3 / c3

    # W^cv = O Y / P + cv (Yhat - O Yhat / P) is DR with the shrunk imputation
    # cv * Yhat, so every DR formula below applies with Yhat -> cv * Yhat.
    cv_drup = M["cfg"].get("cv", 1.0)

    if "explain" in a.sections:
        rng = np.random.default_rng(0)
        users = [t for t in test if len(t[1]) > 1]
        users = [users[k] for k in rng.permutation(len(users))[: a.n_explain]]
        comp_err, cf_sizes, no_cf = [], [], 0
        curves = {"DRUP-attribution": [], "random": [], "item-similarity": []}
        ks = [1, 2, 3, 5, 10]
        Wt = C * W
        norms = Wt.norm(dim=0).clamp_min(1e-12)
        for u, items, rel in users:
            Gloo = fat.leave_one_out_operator(G, W[u], C[u])
            s = user_scores(u, W[u], Gloo)
            full = S["DRUP"][row_of[u]]
            comp_err.append(float((s[items] - full[items]).abs().max() / full[items].abs().max().clamp_min(1e-12)))
            order = items[np.argsort(-s[items].numpy())]
            i, k2 = int(order[0]), int(order[1])
            logged = (O[u] > 0).to(dt)
            w0 = cv_drup * Yhat[u]      # unlogged value under the control-variate weight
            phi = fat.contributions(i, W[u], w0, C[u], Gloo, logged) * (beta / c3 if beta < 1e3 else 1 / c3)
            # exactness of additivity for a random subset
            idx = torch.nonzero(logged).flatten()
            sub = idx[torch.randperm(len(idx))[: max(1, len(idx) // 3)]]
            w2 = W[u].clone()
            w2[sub] = w0[sub]
            s2 = user_scores(u, w2, Gloo)
            comp_err[-1] = max(comp_err[-1], float(abs(s[i] - phi[sub].sum() - s2[i]) / abs(s[i]).clamp_min(1e-12)))
            sim = (Wt[:, idx] * Wt[:, i:i + 1]).sum(0) / (norms[idx] * norms[i])
            rankings = {
                "DRUP-attribution": idx[torch.argsort(phi[idx], descending=True)],
                "random": idx[torch.randperm(len(idx))],
                "item-similarity": idx[torch.argsort(sim, descending=True)],
            }
            for name, rk in rankings.items():
                row = []
                for kk in ks:
                    w3 = W[u].clone()
                    w3[rk[:kk]] = w0[rk[:kk]]
                    s3_ = user_scores(u, w3, Gloo)[items]
                    row.append(float((s3_ > s3_[list(items).index(i)]).sum()))  # new rank of i (0 = top)
                curves[name].append(row)
            removed, _ = fat.minimal_counterfactual(i, k2, W[u], w0, C[u], Gloo, logged)
            if removed is None:
                no_cf += 1
            else:
                cf_sizes.append(len(removed))
        out["explain"] = {
            "max_rel_error_completeness_additivity": float(np.max(comp_err)),
            "deletion_rank_of_top1": {nm: dict(zip(map(str, ks), np.mean(v, 0).tolist())) for nm, v in curves.items()},
            "minimal_cf_size_mean": float(np.mean(cf_sizes)) if cf_sizes else None,
            "minimal_cf_size_median": float(np.median(cf_sizes)) if cf_sizes else None,
            "frac_no_cf": no_cf / len(users),
        }
        print("explain", json.dumps(out["explain"]), flush=True)
        save()

    def attack_eval(Mm, s_clean, cm3, targets, budgets, L, fillers):
        """Frozen-nuisance attack: realised hits and Theorem-8 certificates.
        Returns hit[b], cert[b] (per target), violations, max bound excess
        and the per-(target, budget) set of certified users."""
        Wm, Cm, Pm = Mm["W"], Mm["C"], Mm["P"]
        mth = Mm["method"]
        bm = Mm["cfg"].get("beta", 1.0)
        tau = Mm["cfg"].get("floor", 1.0) if mth != "Obs" else 1.0
        Wr, Cr = Wm[rows], Cm[rows]
        # degrees from W (logged graph) or from cross-fitted W are attacker-controlled
        Dsrc = Mm["Ydeg"] if (Mm.get("Ydeg") is not None and Mm["cfg"].get("deg", "Yhat") == "Yhat") else Wm
        di = Dsrc.sum(0).clamp_min(1.0)
        alpha = Mm["cfg"]["alpha"]
        hit = {b: [] for b in budgets}
        cert = {b: [] for b in budgets}
        cert_users = {}
        viol, max_excess = 0, -float("inf")
        yh_v = Mm["cfg"].get("cv", 1.0) * Mm["Yhat"].mean(0) if Mm["Yhat"] is not None \
            else torch.zeros(n, dtype=dt)
        for t in targets:
            t = int(t)
            o_v = torch.zeros(n, dtype=dt)
            o_v[fillers] = 1.0
            o_v[t] = 1.0
            if mth == "Obs":
                w_v = o_v.clone()
            else:
                pv = torch.full((n,), float(Pm[:, t].mean()), dtype=dt)
                pv[fillers] = Pm[:, fillers].mean(0)
                pv = pv.clamp_min(tau)
                w_v = yh_v + o_v * (1.0 - yh_v) / pv
            dv = (w_v if Dsrc is Wm else Mm["Ydeg"].mean(0)).sum().clamp_min(1.0)
            c_v = dv ** (-alpha) * di ** (-(1 - alpha))
            wt_v = c_v * w_v
            wt_u = Cr * Wr
            if Mm["correct"]:
                dS3 = (wt_u @ wt_v)[:, None] * wt_v[None, :] - wt_u * (wt_v * wt_v)[None, :] \
                    + wt_u * (c_v * c_v * w_v)[None, :]
            else:
                dS3 = (wt_u @ wt_v)[:, None] * wt_v[None, :]
            kind = "Obs" if mth == "Obs" else ("IPS" if mth.startswith("IPS") else "DR")
            if Dsrc is Wm:   # attacker-controlled degree, only d_v >= 1 known
                c_lo, c_hi = torch.zeros(n, dtype=dt), di ** (-(1 - alpha))
            else:            # degree from the imputation: c_v known exactly
                c_lo = c_hi = c_v
            blo, bhi, ulo, uhi = fat.fake_user_effect_bounds(
                wt_u, c_lo, c_hi, yh_v, tau, L + 1, kind, corrected=Mm["correct"], return_unlogged=True)
            sc3 = (1.0 if bm >= 1e3 else bm) / cm3
            blo, bhi, ulo = blo * sc3, bhi * sc3, ulo * sc3
            eff = dS3 * sc3
            scale_b = float(torch.maximum(bhi.abs().max(), blo.abs().max()))
            excess = float(torch.maximum((eff - bhi).max(), (blo - eff).max()))
            max_excess = max(max_excess, excess / max(scale_b, 1e-30))
            assert excess <= 1e-4 * scale_b, (mth, excess, scale_b)
            for b in budgets:
                s_att = s_clean + b * eff
                hits, certs, cu = [], [], set()
                for uu, items, _ in test:
                    r = row_of[uu]
                    if O[uu, t] > 0:
                        continue
                    unexp = unexp_of[uu]
                    kth = torch.topk(s_att[r, unexp], K).values[-1]
                    in_top = bool(s_att[r, t] >= kth)
                    hits.append(in_top)
                    # certificate: t cannot enter the top-K under ANY b fake
                    # profiles of at most L+1 interactions (two valid tests)
                    up_t = s_clean[r, t] + b * bhi[r, t]
                    lo_any = s_clean[r, unexp] + b * blo[r, unexp]
                    certified = bool(up_t < torch.topk(lo_any, K + 1).values[-1])
                    need = K + 1 + b * (L + 1)
                    if not certified and need <= len(unexp):
                        lo_unl = s_clean[r, unexp] + b * ulo[r, unexp]
                        certified = bool(up_t < torch.topk(lo_unl, need).values[-1])
                    certs.append(certified)
                    if certified:
                        cu.add(uu)
                    if certified and in_top:
                        viol += 1
                hit[b].append(float(np.mean(hits)))
                cert[b].append(float(np.mean(certs)))
                cert_users[(t, b)] = cu
        return hit, cert, viol, max_excess, cert_users, o_v

    if "attack" in a.sections:
        res = {}
        # target: a low-popularity item that is a candidate for many test users
        cand_cnt = np.zeros(n)
        for _, items, _ in test:
            cand_cnt[items] += 1
        elig = np.nonzero(cand_cnt >= np.percentile(cand_cnt[cand_cnt > 0], 50))[0]
        elig = elig[np.argsort(item_pop[elig])]
        targets = elig[int(0.1 * len(elig)): int(0.1 * len(elig)) + 5]
        L = 50 if a.dataset == "kuairec" else 20
        fillers = np.argsort(-item_pop)[:L]
        budgets = {"coat": [0, 1, 2, 5, 10, 20, 50], "yahoo": [0, 1, 2, 5, 10, 20, 50, 100]}.get(
            a.dataset, [0, 1, 2, 5, 20, 50, 100, 200])
        unexp_of = {uu: torch.nonzero(O[uu] == 0).flatten() for uu, _, _ in test}
        for mth in METHODS:
            Mm = get_model(mth)
            hit, cert, viol, max_excess, _, _ = attack_eval(Mm, S[mth], consts[mth][1], targets, budgets,
                                                            L, fillers)
            res[mth] = {"hit@K": {str(b): float(np.mean(v)) for b, v in hit.items()},
                        "certified_frac": {str(b): float(np.mean(v)) for b, v in cert.items()},
                        "certificate_violations": viol, "tau": Mm["cfg"].get("floor", 1.0),
                        "max_rel_bound_excess": max_excess}
            print("attack", mth, json.dumps(res[mth]), flush=True)
        out["attack"] = {"K": K, "L": L, "targets": targets.tolist(), "results": res}
        save()

        if a.refit_attack:
            # Certificates assume frozen nuisances. Here the nuisances
            # (propensity model when it is estimated, imputation, degree
            # weights) are re-fitted on the poisoned log, with no
            # cross-fitting, and the realised top-K of the re-fitted system is
            # checked against the frozen certificates of the clean system.
            rb = {"coat": [1, 2, 5, 10], "yahoo": [1, 5, 20, 100]}.get(a.dataset, [5, 50, 200])
            tg = targets[:2] if a.dataset == "kuairec" else targets
            nz0 = Nuisance(d, O, Y, a.prop, K=0)
            refit = {}
            for mth in [x for x in METHODS if x != "Obs"]:
                cache.clear()
                gc.collect()
                M0 = build(O, Y, nz0, mth, cfgs[mth])
                S0, cs0 = scores_with_consts(M0, rows)
                hit_f, cert_f, _, _, cert_users, _ = attack_eval(M0, S0, cs0[1], tg, rb, L, fillers)
                viol, hits_r, n_cert = 0, {b: [] for b in rb}, 0
                for t in tg:
                    t = int(t)
                    o_v = torch.zeros(n, dtype=dt)
                    o_v[fillers] = 1.0
                    o_v[t] = 1.0
                    for b in rb:
                        Oa = torch.cat([O, o_v[None, :].expand(b, -1)])
                        Ya = torch.cat([Y, o_v[None, :].expand(b, -1)])
                        da = dict(d, O=Oa, Y=Ya)
                        if a.prop == "given":
                            Pf = M0["P"].mean(0, keepdim=True).expand(b, -1)
                            da["P_given"] = torch.cat([nz.P, Pf])
                        nza = Nuisance(da, Oa, Ya, a.prop, K=0)
                        Ma = build(Oa, Ya, nza, mth, cfgs[mth])
                        Sa = scores_with_consts(Ma, rows)[0]
                        del Ma, nza, Oa, Ya
                        hs = []
                        for uu, items, _ in test:
                            if O[uu, t] > 0:
                                continue
                            r = row_of[uu]
                            unexp = unexp_of[uu]
                            in_top = bool(Sa[r, t] >= torch.topk(Sa[r, unexp], K).values[-1])
                            hs.append(in_top)
                            if uu in cert_users[(t, b)]:
                                n_cert += 1
                                if in_top:
                                    viol += 1
                        hits_r[b].append(float(np.mean(hs)))
                        del Sa
                        gc.collect()
                refit[mth] = {"budgets": rb, "hit_frozen": {str(b): float(np.mean(hit_f[b])) for b in rb},
                              "hit_refit": {str(b): float(np.mean(v)) for b, v in hits_r.items()},
                              "certified_frac_frozen": {str(b): float(np.mean(cert_f[b])) for b in rb},
                              "certified_pairs": n_cert, "violations_under_refit": viol}
                print("attack-refit", mth, json.dumps(refit[mth]), flush=True)
            out["attack_refit"] = refit
            save()

    if "privacy" in a.sections:
        # Population-level nuisances from a public 10% of the non-test users;
        # user-level parts from each user's own row; G over the private rows
        # only; clipping radius R from the public users; analytic Gaussian
        # noise; denoising ranks fixed in advance (all reported).
        from drup.data import split_test
        _, ptest = split_test(test, 0.3, 0, by="entry" if a.dataset == "coat" else "user")
        cfg = dict(cfgs["DRUP"], imp="add")
        gpub = np.random.default_rng(11)
        test_users = set(row_of)
        cand_pub = np.array([u for u in range(m) if u not in test_users], dtype=np.int64)
        npub = max(10, int(0.1 * m))
        if len(cand_pub) < npub:            # Coat: every user is a test user
            cand_pub = np.arange(m, dtype=np.int64)
        pub_idx = gpub.choice(cand_pub, size=npub, replace=False)
        pub = torch.zeros(m, dtype=torch.bool)
        pub[torch.as_tensor(pub_idx, dtype=torch.long)] = True
        ptest = [t for t in ptest if not bool(pub[t[0]])]   # evaluate private users only
        priv = torch.nonzero(~pub).flatten()
        Pp, Yhp, Cp = public_nuisance(O, Y, pub, a.prop, cfg["floor"], cfg["lam"], cfg["alpha"],
                                      P_given=d.get("P_given"))
        Wp = edge_estimate(O, Y, Pp, Yhp, cfg.get("cv", 1.0))
        R = float((Cp * Wp)[pub].norm(dim=1).median())
        bp = cfg.get("beta", 1.0)
        s1p = (Cp * Wp)[rows]
        c1p = float(s1p.abs().mean())
        g = torch.Generator().manual_seed(7)
        ranks = [8, 32, 128, None]

        def utility(Gd, c3p):
            s3 = local_three_hop(Wp[rows], Cp[rows], Gd)
            sc = s3 / c3p if bp >= 1e3 else s1p / c1p + bp * s3 / c3p
            return evaluate(sc, ptest, (kn,), row_of)[f"ndcg@{kn}"]
        G_np, _, _ = fat.dp_item_operator(Wp, Cp, float("inf"), 1e-5, R, users=priv)
        c3p = float(local_three_hop(Wp[rows], Cp[rows], G_np).abs().mean())
        res = {"inf": {"ndcg": {str(rk): utility(fat.low_rank_denoise(G_np, rk), c3p) for rk in ranks},
                       "sigma": 0.0}}
        print("privacy eps inf", res["inf"], flush=True)
        for eps in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
            vals = {str(rk): [] for rk in ranks}
            for _ in range(3):
                Gd, _, sigma = fat.dp_item_operator(Wp, Cp, eps, 1e-5, R, generator=g, users=priv)
                comps = fat.spectral_components(Gd, max(r_ for r_ in ranks if r_))
                for rk in ranks:
                    vals[str(rk)].append(utility(fat.low_rank_denoise(Gd, rk, comps), c3p))
            res[str(eps)] = {"ndcg": {k: float(np.mean(v)) for k, v in vals.items()},
                             "std": {k: float(np.std(v)) for k, v in vals.items()}, "sigma": sigma}
            print("privacy eps", eps, res[str(eps)], flush=True)
        out["privacy"] = {"R": R, "delta": 1e-5, "n_public": int(pub.sum()), "config": cfg, "results": res,
                          "non_private_full_nuisance_ndcg": evaluate(S["DRUP"], ptest, (kn,), row_of)[f"ndcg@{kn}"],
                          "pop_ndcg": evaluate((O * Y).sum(0, keepdim=True).expand(len(rows), -1),
                                               ptest, (kn,), row_of)[f"ndcg@{kn}"]}
        print("privacy reference", out["privacy"]["non_private_full_nuisance_ndcg"], "pop",
              out["privacy"]["pop_ndcg"], flush=True)

    save()


if __name__ == "__main__":
    main()
