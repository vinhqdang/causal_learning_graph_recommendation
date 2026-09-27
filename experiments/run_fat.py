"""Fairness / accountability / transparency / privacy experiments.

Sections (all on the unbiased test users):
  fairness     accuracy + PRU, exposure-conditional bias, Gini@K, coverage@K,
               user-group NDCG gap
  intervene    real-data intervention do(p_i <- p_i / 2): exposures of a random
               half of the items are thinned with probability 1/2 and the
               change of their mean within-user rank is measured
               (exposure elasticity; Theorem 4 predicts 0 for DRUP)
  explain      exactness of attributions, deletion curves, minimal
               counterfactual explanation sizes (DRUP)
  attack       shilling attack with frozen nuisances + certified bound (Thm 7)
  privacy      utility of the (epsilon, delta)-DP public operator (Thm 8)
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
from drup.data import load_coat  # noqa: E402
from drup.estimation import popularity_propensity  # noqa: E402
from drup.metrics import evaluate  # noqa: E402
from drup.pipeline import build, scores_with_consts  # noqa: E402
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="coat")
    ap.add_argument("--prop", default="given")
    ap.add_argument("--sections", nargs="+", default=["fairness", "intervene", "explain", "attack", "privacy"])
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--n_explain", type=int, default=200)
    a = ap.parse_args()
    dt = getattr(torch, a.dtype)
    if a.dataset == "coat":
        d = load_coat()
        K, kn = 5, 5
        feats = np.loadtxt("data/raw/coat/user_item_features/user_features.ascii")
        groups = {u: int(feats[u, 1] == 1) for u in range(feats.shape[0])}  # women = 1
        gname = "gender (men vs women)"
    else:
        d = torch.load("data/raw/kuairec.pt", weights_only=False)
        K, kn = 20, 20
        act = d["O"].sum(1).numpy()
        med = np.median(act[[u for u, _, _ in d["test"]]])
        groups = {u: int(act[u] > med) for u, _, _ in d["test"]}
        gname = "activity (inactive vs active)"
    O, Y = d["O"].to(dt), d["Y"].to(dt)
    m, n = O.shape
    P_raw = d["P_given"].to(dt) if (a.prop == "given" and d.get("P_given") is not None) \
        else popularity_propensity(O)[0]
    test = d["test"]
    rows_users = sorted({u for u, _, _ in test})
    rows = torch.tensor(rows_users)
    row_of = {u: k for k, u in enumerate(rows_users)}
    fpath = f"results/filters_{a.dataset}_{a.prop}.json"
    cfgs = {mth: chosen_config(fpath, mth) for mth in METHODS}
    item_pop = (O * Y).sum(0).numpy()
    # true item quality on the unbiased data
    qa, qc = np.zeros(n), np.zeros(n)
    for u, items, rel in test:
        qa[items] += rel
        qc[items] += 1
    quality = np.where(qc > 0, qa / np.maximum(qc, 1), 0.0)
    out = {"dataset": a.dataset, "configs": cfgs}
    path = f"results/fat_{a.dataset}_{a.prop}.json"

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
            cache[mth] = build(O, Y, P_raw, mth, cfgs[mth])
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
            res[mth] = {
                f"ndcg@{kn}": ev[f"ndcg@{kn}"],
                "PRU": fat.popularity_rank_correlation(S[mth], test, item_pop, row_of),
                "ECB": fat.exposure_conditional_bias(S[mth], test, item_pop, quality, row_of),
                f"gini@{K}": fat.gini(freq[cand]),
                f"coverage@{K}": float((freq[cand] > 0).mean()),
                "group_gap": gap, "group0": g0, "group1": g1,
            }
            print("fairness", mth, {k: round(v, 4) for k, v in res[mth].items()}, flush=True)
        out["fairness"] = {"groups": gname, "results": res}
        save()

    if "intervene" in a.sections:
        res = {mth: [] for mth in METHODS}
        g = torch.Generator().manual_seed(123)
        for r in range(a.reps):
            treated = torch.rand(n, generator=g) < 0.5
            keep = (torch.rand(m, n, generator=g) < 0.5).to(dt)
            thin = torch.where(treated[None, :], keep, torch.ones_like(keep))
            O2 = O * thin
            # Known intervention: propensity of treated items is halved.
            P2 = torch.where(treated[None, :], P_raw * 0.5, P_raw)
            cache.clear()
            gc.collect()
            for mth in METHODS:
                M2 = build(O2, Y, P2, mth, cfgs[mth])
                S2 = scores_with_consts(M2, rows)[0]
                del M2
                a0, c0 = within_user_rank(S[mth], test, row_of, n)
                a1, c1 = within_user_rank(S2, test, row_of, n)
                ok = c0 > 0
                tr = treated.numpy() & ok
                ct = (~treated.numpy()) & ok
                shift = (a1[tr].sum() / c1[tr].sum() - a0[tr].sum() / c0[tr].sum()) \
                    - (a1[ct].sum() / c1[ct].sum() - a0[ct].sum() / c0[ct].sum())
                res[mth].append(float(shift))
                del S2
            del O2, P2, thin, keep
            gc.collect()
        out["intervene"] = {mth: {"mean": float(np.mean(v)), "std": float(np.std(v)), "all": v}
                            for mth, v in res.items()}
        for mth in METHODS:
            print("intervene", mth, out["intervene"][mth]["mean"], "+-", out["intervene"][mth]["std"], flush=True)
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
            w0 = Yhat[u]
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

    if "attack" in a.sections:
        res = {}
        # target: a low-popularity item that is a candidate for many test users
        cand_cnt = np.zeros(n)
        for _, items, _ in test:
            cand_cnt[items] += 1
        elig = np.nonzero(cand_cnt >= np.percentile(cand_cnt[cand_cnt > 0], 50))[0]
        elig = elig[np.argsort(item_pop[elig])]
        targets = elig[int(0.1 * len(elig)): int(0.1 * len(elig)) + 5]
        L = 20 if a.dataset == "coat" else 50
        fillers = np.argsort(-item_pop)[:L]
        budgets = [0, 1, 2, 5, 10, 20, 50] if a.dataset == "coat" else [0, 1, 2, 5, 20, 50, 100, 200]
        unexp_of = {uu: torch.nonzero(O[uu] == 0).flatten() for uu, _, _ in test}
        for mth in METHODS:
            Mm = get_model(mth)
            Wm, Cm, Pm = Mm["W"], Mm["C"], Mm["P"]
            cm1, cm3 = consts[mth]
            bm = Mm["cfg"].get("beta", 1.0)
            tau = Mm["cfg"].get("floor", 1.0) if mth != "Obs" else 1.0
            Wr = Wm[rows]
            Cr = Cm[rows]
            s_clean = S[mth]
            # item-side weights for a fake user (frozen item degrees, d_v >= 1)
            Dsrc = Mm["Yhat"] if (Mm["cfg"].get("deg") == "Yhat" and Mm["Yhat"] is not None) else Wm
            di = Dsrc.sum(0).clamp_min(1.0)
            alpha = Mm["cfg"]["alpha"]
            hit = {b: [] for b in budgets}
            cert = {b: [] for b in budgets}
            viol = 0
            max_excess = -float("inf")
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
                    yh_v = Mm["Yhat"].mean(0) if Mm["Yhat"] is not None else torch.zeros(n, dtype=dt)
                    w_v = yh_v + o_v * (1.0 - yh_v) / pv
                dv = (w_v if Dsrc is Wm else Mm["Yhat"].mean(0) if Mm["Yhat"] is not None else w_v).sum().clamp_min(1.0)
                c_v = dv ** (-alpha) * di ** (-(1 - alpha))
                wt_v = c_v * w_v
                wt_u = Cr * Wr
                if Mm["correct"]:
                    # ONE fake user: wt_u @ contrib(w_v) without forming (n, n)
                    dS3 = (wt_u @ wt_v)[:, None] * wt_v[None, :] - wt_u * (wt_v * wt_v)[None, :] \
                        + wt_u * (c_v * c_v * w_v)[None, :]
                else:
                    dS3 = (wt_u @ wt_v)[:, None] * wt_v[None, :]
                # certified bounds for ANY profile with <= L+1 logged items
                kind = "Obs" if mth == "Obs" else ("IPS" if mth.startswith("IPS") else "DR")
                yh_v = Mm["Yhat"].mean(0) if Mm["Yhat"] is not None else torch.zeros(n, dtype=dt)
                if Dsrc is Wm:   # attacker-controlled degree, only d_v >= 1 known
                    c_lo, c_hi = torch.zeros(n, dtype=dt), di ** (-(1 - alpha))
                else:            # degree from the imputation: c_v known exactly
                    c_lo = c_hi = c_v
                blo, bhi, ulo, uhi = fat.fake_user_effect_bounds(
                    wt_u, c_lo, c_hi, yh_v, tau, L + 1, kind, corrected=Mm["correct"],
                    return_unlogged=True)
                sc3 = (1.0 if bm >= 1e3 else bm) / cm3
                blo, bhi, ulo = blo * sc3, bhi * sc3, ulo * sc3
                # sanity: the realised single-profile effect lies inside the bound
                # (relative tolerance for float32 round-off)
                eff = dS3 * sc3
                scale_b = float(torch.maximum(bhi.abs().max(), blo.abs().max()))
                excess = float(torch.maximum((eff - bhi).max(), (blo - eff).max()))
                max_excess = max(max_excess, excess / max(scale_b, 1e-30))
                assert excess <= 1e-4 * scale_b, (mth, excess, scale_b)
                for b in budgets:
                    s_att = s_clean + b * dS3 * sc3
                    hits, certs = [], []
                    for uu, items, _ in test:
                        r = row_of[uu]
                        if O[uu, t] > 0:
                            continue
                        unexp = unexp_of[uu]
                        sc = s_att[r, unexp]
                        kth = torch.topk(sc, K).values[-1]
                        in_top = bool(s_att[r, t] >= kth)
                        hits.append(in_top)
                        # certificate: t cannot enter top-K under ANY b fake profiles
                        # b fake users log at most b*(L+1) items; every other
                        # competitor keeps its unlogged-state lower bound. The
                        # (K+1+b(L+1))-th largest such bound leaves >= K
                        # competitors (other than t) above it whatever the attack.
                        # Two valid certificates; t is certified if either holds.
                        up_t = s_clean[r, t] + b * bhi[r, t]
                        # (i) every competitor at its worst-case (logged) lower bound
                        lo_any = s_clean[r, unexp] + b * blo[r, unexp]
                        certified = bool(up_t < torch.topk(lo_any, K + 1).values[-1])
                        # (ii) at most b(L+1) competitors can be logged by the fakes
                        need = K + 1 + b * (L + 1)
                        if not certified and need <= len(unexp):
                            lo_unl = s_clean[r, unexp] + b * ulo[r, unexp]
                            certified = bool(up_t < torch.topk(lo_unl, need).values[-1])
                        certs.append(certified)
                        if certified and in_top:
                            viol += 1
                    hit[b].append(float(np.mean(hits)))
                    cert[b].append(float(np.mean(certs)))
            res[mth] = {"hit@K": {str(b): float(np.mean(v)) for b, v in hit.items()},
                        "certified_frac": {str(b): float(np.mean(v)) for b, v in cert.items()},
                        "certificate_violations": viol, "tau": tau,
                        "max_rel_bound_excess": max_excess}
            print("attack", mth, json.dumps(res[mth]), flush=True)
        out["attack"] = {"K": K, "L": L, "targets": targets.tolist(), "results": res}
        save()

    if "privacy" in a.sections:
        # The rank of the post-processing denoiser is chosen on a validation
        # part of the unbiased data; utility is reported on the rest.
        from drup.data import split_test
        pval, ptest = split_test(test, 0.3, 0, by="entry" if a.dataset == "coat" else "user")
        res = {}
        Wt = C * W
        R = float(Wt.norm(dim=1).median())
        g = torch.Generator().manual_seed(7)
        ranks = [None, 4, 8, 16, 32, 64, 128]
        s1 = (C * W)[rows]

        def utility(Gd, part):
            s3 = local_three_hop(W[rows], C[rows], Gd)
            sc = s3 / c3 if beta >= 1e3 else s1 / c1 + beta * s3 / c3
            return evaluate(sc, part, (kn,), row_of)[f"ndcg@{kn}"]
        for eps in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, float("inf")]:
            vals, chosen_r = [], []
            for r in range(3 if eps < float("inf") else 1):
                if eps == float("inf"):
                    _, k, _ = fat.dp_item_operator(W, C, 1.0, 1e-5, R, generator=g)
                    Gd = item_gram(W * k[:, None], C, correct=True)
                    sigma = 0.0
                else:
                    Gd, k, sigma = fat.dp_item_operator(W, C, eps, 1e-5, R, generator=g)
                comps = fat.spectral_components(Gd, max(r_ for r_ in ranks if r_))
                best = max(ranks, key=lambda rk: utility(fat.low_rank_denoise(Gd, rk, comps), pval))
                chosen_r.append(best)
                vals.append(utility(fat.low_rank_denoise(Gd, best, comps), ptest))
            res[str(eps)] = {"ndcg": float(np.mean(vals)), "std": float(np.std(vals)), "sigma": sigma,
                             "ranks": chosen_r}
            print("privacy eps", eps, res[str(eps)], flush=True)
        out["privacy"] = {"R": R, "delta": 1e-5, "results": res,
                          "non_private_ndcg": evaluate(S["DRUP"], ptest, (kn,), row_of)[f"ndcg@{kn}"],
                          "pop_ndcg": evaluate((O * Y).sum(0, keepdim=True).expand(len(rows), -1),
                                               ptest, (kn,), row_of)[f"ndcg@{kn}"]}
        print("privacy reference", out["privacy"]["non_private_ndcg"], "pop", out["privacy"]["pop_ndcg"])

    save()


if __name__ == "__main__":
    main()
