"""v11 probes (paper/v7/DESIGN.md Section 3): item-appeal estimators borrowed from other fields, all fitted on the
pre-cut standard log only (temporal protocol), added to the reproduced LEAF-LGN / LEAF-tied scores with the weight
(and the estimator's own hyperparameters) chosen on validation per split. Paired with the stored per-user results of
results/v10 (QP family, r-AdjNorm, LEAF-LGN).

  P1 MRP      multilevel logistic regression of pre-cut clicks: logit = mu + b_u + tag + author + item (+ content
              covariates = Fay-Herriot prior, P2) (+ item x user-stratum deviations, post-stratified to all users)
  P3 PBM-EM   detection x attractiveness: P(click) = theta_u * alpha_i (regression-EM of click models / occupancy
              detection), alpha_i with a Beta prior centred on its tag mean
  P5 TRAJ     per-item log-linear trend of daily pre-cut clicks, slope shrunk to the tag mean, forecast to the middle
              of the random period (demand / popularity forecasting)
  OFF-PROTOCOL diagnostic UC: item click rate from the validation users' random impressions (uses post-cut random
              data; reported only to price the protocol, never a candidate)
python3 experiments/v11_probe_items.py [P1 P3 P5 UC COMBO]     outputs results/v11_probes/probe_items.{txt,json}
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
from v11_common import (CUT_MS, DAY_MS, KEY, OUTDIR, ROOT, Ctx, Out, bundle, fmt, pair,  # noqa: E402
                        stored)

SC = os.environ.get("V11_SCORES", "/tmp/v11_scores")
GAMMAS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
JS = os.path.join(OUTDIR, "probe_items.json")


class PerSplit:
    """Validation selection where split s uses its own base score (the stored LEAF selection of that split)."""

    def __init__(self, C):
        self.C = C
        self.sel = [C.selector() for _ in C.splits]
        self.nc = 0

    def add(self, bases, extra, cfg):
        """bases[s]: bucketed base score of split s; extra: bucketed addition (shared) or None."""
        self.nc += 1
        S_cache = {}
        for s, sel in enumerate(self.sel):
            key = id(bases[s])
            if key not in S_cache:
                S_cache[key] = bases[s] if extra is None else [b + e for b, e in zip(bases[s], extra)]
            sel.add(S_cache[key], cfg, active=[j == s for j in range(len(self.sel))])

    def record(self):
        bs = [sel.best[s] for s, sel in enumerate(self.sel)]
        rid = self.C.row_ids
        return {"per_user": [{str(int(rid[k])): v for k, v in b["per_user"].items()} for b in bs],
                "per_split": [{KEY: b["test"]} for b in bs], "val_best": [b["val"] for b in bs],
                "chosen": [b["cfg"] for b in bs], "test": float(np.mean([b["test"] for b in bs])),
                "n_configs": self.nc}


# ------------------------------------------------------------------------------------------------------------------
# item-appeal estimators (pre-cut log only)
# ------------------------------------------------------------------------------------------------------------------
def item_covariates(B):
    """Content covariates of the pool items (codes), from the item features and the pre-cut lifelong tables."""
    it = B["item_feat"]
    n = len(it)
    vf = pd.read_csv(os.path.join(ROOT, "data", "raw", "KuaiRand-Pure", "data", "video_features_basic_pure.csv"))
    vf = vf.set_index("video_id").reindex(B["items_raw"])
    L = np.load(os.path.join(ROOT, "data", "raw", "lifelong_27k.npz"), allow_pickle=True)
    vals = pd.Index(L["author_id_values"].astype(str))
    agg = np.zeros((len(vals), 3))
    np.add.at(agg, L["author_id_pre_k"], L["author_id_pre_ncl"].astype(np.float64))
    code = vals.get_indexer(pd.to_numeric(it["author_id"], errors="coerce").astype("Int64").astype(str))
    A = np.vstack([agg, np.zeros((1, 3))])[np.where(code < 0, len(vals), code)]
    cov = {
        "tag": pd.factorize(it["tag"].astype(str))[0],
        "author": pd.factorize(it["author_id"].astype(str))[0],
        "upload_type": pd.factorize(it["upload_type"].astype(str))[0],
        "video_type": pd.factorize(it["video_type"].astype(str))[0],
        "dur": pd.qcut(it["video_duration"].fillna(it["video_duration"].median()).rank(method="first"), 10,
                       labels=False).values,
        "aspect": pd.factorize((vf.server_height / vf.server_width).round(2).astype(str).values)[0],
        "music_type": pd.factorize(vf.music_type.fillna(-1).astype(str).values)[0],
        "a_imp": np.digitize(np.log1p(A[:, 0]), np.quantile(np.log1p(A[:, 0]), np.linspace(0.1, 0.9, 9))),
        "a_ctr": np.digitize((A[:, 1] + 1) / (A[:, 0] + 2), np.quantile((A[:, 1] + 1) / (A[:, 0] + 2),
                                                                       np.linspace(0.1, 0.9, 9))),
        "a_lvr": np.digitize((A[:, 2] + 1) / (A[:, 0] + 2), np.quantile((A[:, 2] + 1) / (A[:, 0] + 2),
                                                                       np.linspace(0.1, 0.9, 9))),
    }
    out = {}
    for k, v in cov.items():
        v = np.asarray(v, dtype=np.int64)
        out[k] = torch.as_tensor(np.where(v < 0, v.max() + 1, v))      # missing value -> its own code
    return out


def user_strata(B, m, k=5):
    """User strata by the lifelong (pre-cut) click rate, the usual MRP cell variable here; users without history
    form their own stratum."""
    L = np.load(os.path.join(ROOT, "data", "raw", "lifelong_27k.npz"), allow_pickle=True)
    tot = pd.DataFrame(L["tot_pre"], index=L["tot_pre_user"], columns=["n", "c", "l"]).reindex(B["users_raw"])
    r = (tot.c + 1) / (tot.n + 2)
    s = np.full(m, k, dtype=np.int64)
    ok = tot.n.notna().values
    s[ok] = pd.qcut(r[ok].rank(method="first"), k, labels=False).values
    return torch.as_tensor(s)


def fit_mrp(B, cov, strata, lam_e, lam_h=5.0, content=True, lam_d=None, iters=60):
    """Penalised (multilevel, MAP) logistic regression of pre-cut clicks. Returns item appeal a (n,), the stratum
    deviations d (n, S) or None, and the stratum mean user effects."""
    lg = B["log"]
    m, n = B["shape"]
    u = torch.as_tensor(lg.u.values)
    i = torch.as_tensor(lg.i.values)
    y = torch.as_tensor(lg.is_click.values, dtype=torch.float32)
    groups = ["tag", "author"] + (["upload_type", "video_type", "dur", "aspect", "music_type", "a_imp", "a_ctr",
                                   "a_lvr"] if content else [])
    P = {"mu": torch.zeros(1), "b": torch.zeros(m), "e": torch.zeros(n)}
    for g in groups:
        P[g] = torch.zeros(int(cov[g].max()) + 1)
    S = int(strata.max()) + 1
    if lam_d is not None:
        P["d"] = torch.zeros(n, S)
    for v in P.values():
        v.requires_grad_(True)
    su = strata[u]
    lam = {"b": 1.0, "e": lam_e, "tag": 1.0, "author": lam_h, "d": lam_d}

    def appeal():
        a = P["e"].clone()
        for g in groups:
            a = a + P[g][cov[g]]
        return a

    opt = torch.optim.LBFGS(list(P.values()), lr=1.0, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        a = appeal()
        z = P["mu"] + P["b"][u] + a[i]
        if lam_d is not None:
            z = z + P["d"][i, su]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction="sum")
        reg = lam["b"] * (P["b"] ** 2).sum() + lam["e"] * (P["e"] ** 2).sum()
        for g in groups:
            reg = reg + lam.get(g, 1.0) * (P[g] ** 2).sum()
        if lam_d is not None:
            reg = reg + lam_d * (P["d"] ** 2).sum()
        tot = loss + reg
        tot.backward()
        return tot
    opt.step(closure)
    with torch.no_grad():
        a = appeal()
        bm = torch.zeros(S).index_add_(0, strata, P["b"]) / torch.bincount(strata, minlength=S).clamp_min(1)
        d = P["d"].detach() if lam_d is not None else None
    return a.detach(), d, (P["mu"].detach() + bm)


def fit_pbm(B, cov, prior=20.0, iters=40):
    """Detection x attractiveness (regression-EM of position-bias click models, user-specific examination):
    P(click) = theta_u * alpha_i; alpha_i gets a Beta prior of strength `prior` centred on its tag mean."""
    lg = B["log"]
    m, n = B["shape"]
    u = torch.as_tensor(lg.u.values)
    i = torch.as_tensor(lg.i.values)
    y = torch.as_tensor(lg.is_click.values, dtype=torch.float64)
    th = torch.full((m,), 0.7, dtype=torch.float64)
    al = torch.full((n,), 0.6, dtype=torch.float64)
    tag = cov["tag"]
    T = int(tag.max()) + 1
    nu = torch.bincount(u, minlength=m).double()
    ni = torch.bincount(i, minlength=n).double()
    for _ in range(iters):
        t, a = th[u], al[i]
        den = (1 - t * a).clamp_min(1e-9)
        pE = torch.where(y > 0, torch.ones_like(t), t * (1 - a) / den)
        pR = torch.where(y > 0, torch.ones_like(t), (1 - t) * a / den)
        th = ((torch.zeros(m, dtype=torch.float64).index_add_(0, u, pE) + 1.0) / (nu + 2.0)).clamp(1e-3, 1 - 1e-3)
        sR = torch.zeros(n, dtype=torch.float64).index_add_(0, i, pR)
        tm = (torch.zeros(T, dtype=torch.float64).index_add_(0, tag, sR) + 1.0) / \
             (torch.zeros(T, dtype=torch.float64).index_add_(0, tag, ni) + 2.0)
        al = ((sR + prior * tm[tag]) / (ni + prior)).clamp(1e-3, 1 - 1e-3)
    return torch.logit(al).float()


def fit_traj(B, cov, shrink=5.0, horizon_days=8.0, window=7):
    """Per-item log-linear trend of daily pre-cut clicks over the last `window` days before the cut, slope shrunk to
    its tag mean (weighted least squares on log(1 + clicks)), extrapolated `horizon_days` past the cut."""
    lg = B["log"]
    n = B["shape"][1]
    day = ((lg.time_ms.values - CUT_MS) // DAY_MS).astype(int)       # -14 .. -1
    keep = day >= -window
    D = np.zeros((n, window))
    np.add.at(D, (lg.i.values[keep], day[keep] + window), lg.is_click.values[keep])
    Yl = np.log1p(D)
    t = np.arange(window) - (window - 1) / 2.0
    slope = (Yl * t).sum(1) / (t ** 2).sum()
    level = Yl.mean(1)
    tag = cov["tag"].numpy()
    w = D.sum(1)
    tm = pd.Series(slope * w).groupby(tag).sum() / (pd.Series(w).groupby(tag).sum() + 1e-9)
    s_sh = (w * slope + shrink * tm.reindex(tag).values) / (w + shrink)
    fc = level + s_sh * ((window - 1) / 2.0 + horizon_days)
    return torch.as_tensor(fc, dtype=torch.float32)


# ------------------------------------------------------------------------------------------------------------------
def main():
    what = sys.argv[1:] or ["P1", "P3", "P5", "UC", "COMBO"]
    out = Out(os.path.join(OUTDIR, "probe_items.txt"))
    out("=" * 100)
    out(f"v11 item-appeal probes {what} [{time.strftime('%Y-%m-%d %H:%M')}]")
    t0 = time.time()
    B = bundle()
    C = Ctx(B)
    m, n = B["shape"]
    cov = item_covariates(B)
    strata = user_strata(B, m)
    ST = stored()
    res = json.load(open(JS)) if os.path.exists(JS) else {}
    bases = {}
    for tag in ("lgn", "tied"):
        bases[tag] = [C.rowstd(torch.load(os.path.join(SC, f"{tag}_s{s}.pt"))) for s in range(5)]
    pop = torch.load(os.path.join(SC, "pop.pt"))

    def zitem(v):
        return C.rowstd(C.item(v))

    def report(name, rec):
        res[name] = rec
        json.dump(res, open(JS + ".tmp", "w"))
        os.replace(JS + ".tmp", JS)
        out(f"  {name:44s} test {rec['test']:.4f}  chosen {rec['chosen'][0]}")
        for ref in ("QP family", "LEAF-LGN", "r-AdjNorm"):
            out(f"      - {ref:10s} {fmt(pair(rec, ST[ref]))}")

    def probe(name, cands, base_tags=("lgn", "tied")):
        """cands: list of (cfg, bucketed standardised item score). Item alone, and base + gamma * item."""
        sel = PerSplit(C)
        for cfg, Z in cands:
            sel.add([Z] * 5, None, dict(cfg))
        report(f"{name} alone", sel.record())
        for bt in base_tags:
            sel = PerSplit(C)
            for cfg, Z in cands:
                for g in GAMMAS:
                    sel.add(bases[bt], [g * z for z in Z] if g else None, dict(cfg, gamma=g))
            report(f"LEAF-{bt.upper()} + {name}", sel.record())

    # reference: the reproduced LEAF bases under this harness (must equal the stored values)
    for bt in ("lgn", "tied"):
        sel = PerSplit(C)
        sel.add(bases[bt], None, {})
        r = sel.record()
        out(f"  reproduced LEAF-{bt}: test {r['test']:.4f} (stored {ST['LEAF-LGN' if bt == 'lgn' else 'LEAF-tied']['test']:.4f})")
    # control: popularity again on top (LEAF already has it)
    probe("POP (control)", [({}, zitem(pop))], ("lgn",))

    if "P1" in what:
        cands = []
        for lam_e, content, lam_d in itertools.product((1.0, 5.0, 20.0), (False, True), (None, 20.0)):
            t1 = time.time()
            a, d, bm = fit_mrp(B, cov, strata, lam_e, content=content, lam_d=lam_d)
            cfg = dict(lam_e=lam_e, content=content, lam_d=lam_d)
            cands.append((dict(cfg, form="item"), zitem(a)))
            if d is not None:
                # post-stratified appeal over all users (strata shares), and the user's own cell (personalised)
                w = torch.bincount(strata, minlength=d.shape[1]).float()
                w = w / w.sum()
                ps = (torch.sigmoid(bm[None, :] + a[:, None] + d) * w[None, :]).sum(1)
                cands.append((dict(cfg, form="poststrat"), zitem(torch.logit(ps.clamp(1e-6, 1 - 1e-6)))))
                own = C.table(lambda rows, cand: a[cand] + d[cand, strata[rows][:, None].expand_as(cand)])
                cands.append((dict(cfg, form="own_cell"), C.rowstd(own)))
            out(f"    MRP fit lam_e {lam_e} content {content} lam_d {lam_d} [{time.time() - t1:.0f}s]")
        probe("P1 MRP", cands)
    if "P3" in what:
        cands = [({"prior": p}, zitem(fit_pbm(B, cov, prior=p))) for p in (5.0, 20.0, 80.0, 200.0, 500.0, 2000.0)]
        probe("P3 PBM-EM", cands)
    if "P5" in what:
        cands = [({"shrink": s, "h": h}, zitem(fit_traj(B, cov, s, h))) for s in (2.0, 10.0) for h in (4.0, 8.0, 12.0)]
        probe("P5 TRAJ", cands, ("lgn",))
    if "COMBO" in what and "P1" in what and "P3" in what:
        # best P1 and P3 configurations of split 0 on LEAF-LGN, jointly re-weighted on validation
        c1 = res["LEAF-LGN + P1 MRP"]["chosen"][0]
        a, d, bm = fit_mrp(B, cov, strata, c1["lam_e"], content=c1["content"], lam_d=c1["lam_d"])
        Z1 = zitem(a)
        Z3 = zitem(fit_pbm(B, cov, prior=res["LEAF-LGN + P3 PBM-EM"]["chosen"][0]["prior"]))
        sel = PerSplit(C)
        for g1, g3 in itertools.product(GAMMAS, GAMMAS):
            sel.add(bases["lgn"], [g1 * x + g3 * y for x, y in zip(Z1, Z3)], dict(g1=g1, g3=g3))
        report("LEAF-LGN + P1 + P3", sel.record())
    if "UC" in what:
        # OFF-PROTOCOL diagnostic: per split, the item click rate of the validation users' random impressions
        tl = B["test_log"]
        out("  OFF-PROTOCOL (post-cut random data of validation users; not a candidate method):")
        sel = PerSplit(C)
        for s, (val, _) in enumerate(C.splits):
            vrows = set(C.row_ids[[k for k, _, _ in val]].tolist())
            mk = tl.u.isin(vrows).values
            Ni = np.bincount(tl.i[mk], minlength=n).astype(float)
            Ci = np.bincount(tl.i[mk], weights=tl.is_click[mk], minlength=n)
            own = tl[mk].groupby(["u", "i"]).is_click.agg(["sum", "count"])
            mu0 = Ci.sum() / Ni.sum()
            for a0 in (5.0, 20.0, 80.0):
                def tab(rows, cand, a0=a0, own=own):
                    idx = pd.MultiIndex.from_arrays([np.repeat(rows.numpy(), cand.shape[1]), cand.numpy().ravel()])
                    o = own.reindex(idx).fillna(0.0)
                    c = cand.numpy().ravel()
                    q = (Ci[c] - o["sum"].values + a0 * mu0) / (Ni[c] - o["count"].values + a0)   # leave-own-out
                    return torch.logit(torch.as_tensor(q.reshape(cand.shape)).clamp(1e-4, 1 - 1e-4))
                Z = C.rowstd(C.table(tab))
                for g in GAMMAS:
                    S = [b + g * z for b, z in zip(bases["lgn"][s], Z)]
                    sel.sel[s].add(S, dict(a0=a0, gamma=g), active=[j == s for j in range(5)])
                    sel.nc += 1
        report("OFF-PROTOCOL LEAF-LGN + validation-user uniform rate", sel.record())
    out(f"done [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
