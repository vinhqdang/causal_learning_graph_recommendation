"""DIAGNOSTIC (uses random-exposure test labels; never a method, never used for selection): where does the gap between
LEAF-LGN and the leave-one-user-out item oracle live on the temporal all-user KuaiRand setting?

Quantities (nDCG@10 over all evaluated users, LEAF-LGN = the reproduced split-0 selection of results/v10/leaf_lgn.json,
saved by experiments/v11_leaf_scores.py):
  1. levels: LEAF-LGN, the item oracle (LOO uniform click rate), the time-local oracle (LOO rate within +-3 days of the
     impression), oracle from the first half of the random period only, LEAF + a * oracle (best a on all users).
  2. item-level agreement (Spearman with the uniform rate q_i): pre-cut log statistics, LEAF's implied item effect,
     post-cut standard-log statistics (concurrent: excluded by the protocol; measures drift), out-of-pool author
     statistics (pre-cut, from data/raw/lifelong_27k.npz).
  3. attribution: r_i = residual of z(q_i) on LEAF's item effect. The full correction S_LEAF + a * r_i(LOO) recovers
     G_all. A factor's share = gain of S_LEAF + a * E[r | bin of the factor] (bin means cross-fitted over two halves
     of the items) / G_all; a is chosen on all users (diagnostic). Item groups: share of G_all recovered when the
     exact LOO correction is applied only to items of the group. User groups: exact additive split of G_all.
  4. per test day: oracle gain by the day of the random impression.
python3 experiments/v11_diag_gap.py   (output results/v11_probes/diag_gap.txt)
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v11_common import CUT_MS, DAY_MS, KEY, OUTDIR, ROOT, Ctx, Out, bundle, dense_OY  # noqa: E402
from drup.metrics import UserMetrics  # noqa: E402

SC = os.environ.get("V11_SCORES", "/tmp/v11_scores")
PURE = os.path.join(ROOT, "data", "raw", "KuaiRand-Pure", "data")


class AllUsers:
    """Per-user nDCG@10 of every evaluated user for bucketed scores (no selection)."""

    def __init__(self, C):
        self.C = C
        self.um = []
        for b, bk in enumerate(C.layout.buckets):
            lst = [(r, np.arange(len(C.test[k][1])), C.test[k][2]) for r, k in enumerate(bk["k"])]
            lst = [x for x in lst if x[2].sum() > 0]
            self.um.append(UserMetrics(lst, (10,), None, "cpu") if lst else None)
        self.uid = np.concatenate([C.row_ids[C.layout.buckets[b]["k"][um.users]] for b, um in enumerate(self.um)
                                   if um is not None])

    def __call__(self, S):
        """S: list of (U_b, L_b) scores; candidates are positions, so gather is the identity."""
        out = []
        for b, um in enumerate(self.um):
            if um is None:
                continue
            out.append(um.compute(S[b])[KEY].numpy())
        return np.concatenate(out)


def pos_scores(C, S):
    """Bucketed (U_b, L_b) scores are already per position."""
    return S


def main():
    out = Out(os.path.join(OUTDIR, "diag_gap.txt"))
    out("=" * 100)
    out("DIAGNOSTIC: uses test labels. Never a method or a selection input.")
    B = bundle()
    C = Ctx(B)
    A = AllUsers(C)
    n = C.n
    tl = B["test_log"]
    lg = B["log"]
    # ----- uniform item rates (all random impressions of the dataset users) ----------------------------------
    N_i = np.bincount(tl.i, minlength=n).astype(float)
    C_i = np.bincount(tl.i, weights=tl.is_click, minlength=n)
    mu = C_i.sum() / N_i.sum()
    a0 = 10.0
    q = (C_i + a0 * mu) / (N_i + a0)
    out(f"random impressions {len(tl)}, items with any {int((N_i > 0).sum())}, mean per item {N_i.mean():.1f}, "
        f"uniform CTR {mu:.4f}")
    # per (u, i) own counts for LOO
    own = tl.groupby(["u", "i"]).is_click.agg(["sum", "count"])

    def loo_rate(bk, Ni=N_i, Ci=C_i):
        rows = np.repeat(bk["rows"].numpy(), bk["cand"].shape[1])
        idx = pd.MultiIndex.from_arrays([rows, bk["cand"].numpy().ravel()])
        o = own.reindex(idx).fillna(0.0)
        cand = bk["cand"].numpy().ravel()
        r = (Ci[cand] - o["sum"].values + a0 * mu) / (Ni[cand] - o["count"].values + a0)
        return torch.as_tensor(r.reshape(bk["cand"].shape), dtype=torch.float32)

    orc = [loo_rate(bk) for bk in C.layout.buckets]
    lgn = torch.load(os.path.join(SC, "lgn_s0.pt"))
    pop = torch.load(os.path.join(SC, "pop.pt"))
    leaf = C.rowstd(lgn)
    v_leaf = A(leaf)
    v_orc = A(orc)
    out(f"\n1. Levels (all {len(v_leaf)} evaluated users, nDCG@10)")
    out(f"  LEAF-LGN (split-0 configuration)     {v_leaf.mean():.4f}")
    out(f"  item oracle (LOO uniform rate)       {v_orc.mean():.4f}   gap {1000 * (v_orc - v_leaf).mean():+.2f}e-3")
    v_pop = A(C.item(pop))
    out(f"  popularity (pre-cut clicks)          {v_pop.mean():.4f}")
    # time-local oracle: LOO rate within +-3 days of the candidate's random impression
    tday = ((tl.time_ms.values - CUT_MS) // DAY_MS).astype(int)
    nd = tday.max() + 1
    Nd = np.zeros((nd, n))
    Cd = np.zeros((nd, n))
    np.add.at(Nd, (tday, tl.i.values), 1.0)
    np.add.at(Cd, (tday, tl.i.values), tl.is_click.values)
    out(f"  random period: {nd} days; impressions per day {np.bincount(tday).tolist()}")
    for half in (1, 3):
        S = []
        for b, bk in enumerate(C.layout.buckets):
            ct = C.cand_time[b].numpy()
            day = np.clip(((ct - CUT_MS) // DAY_MS).astype(int), 0, nd - 1)
            cs = np.cumsum(np.vstack([np.zeros((1, n)), Nd]), 0)
            cc = np.cumsum(np.vstack([np.zeros((1, n)), Cd]), 0)
            lo, hi = np.clip(day - half, 0, nd), np.clip(day + half + 1, 0, nd)
            cand = bk["cand"].numpy()
            Nw = cs[hi, cand] - cs[lo, cand]
            Cw = cc[hi, cand] - cc[lo, cand]
            rows = np.repeat(bk["rows"].numpy(), cand.shape[1])
            idx = pd.MultiIndex.from_arrays([rows, cand.ravel()])
            o = own.reindex(idx).fillna(0.0)
            Nw = Nw - o["count"].values.reshape(cand.shape)
            Cw = Cw - o["sum"].values.reshape(cand.shape)
            # shrink to the whole-period LOO rate
            S.append(torch.as_tensor((Cw + 20.0 * orc[b].numpy()) / (Nw + 20.0), dtype=torch.float32))
        out(f"  time-local oracle (+-{half} days, shrunk to the period rate) {A(S).mean():.4f}")
    # early-half oracle: rate from the first 8 days only (no LOO needed for users' later impressions: LOO anyway)
    early = tday < nd // 2
    Ne = np.bincount(tl.i[early], minlength=n).astype(float)
    Ce = np.bincount(tl.i[early], weights=tl.is_click[early], minlength=n)
    own_e = tl[early].groupby(["u", "i"]).is_click.agg(["sum", "count"])

    def loo_e(bk):
        rows = np.repeat(bk["rows"].numpy(), bk["cand"].shape[1])
        idx = pd.MultiIndex.from_arrays([rows, bk["cand"].numpy().ravel()])
        o = own_e.reindex(idx).fillna(0.0)
        cand = bk["cand"].numpy().ravel()
        r = (Ce[cand] - o["sum"].values + a0 * mu) / (Ne[cand] - o["count"].values + a0)
        return torch.as_tensor(r.reshape(bk["cand"].shape), dtype=torch.float32)
    orc_e = [loo_e(bk) for bk in C.layout.buckets]
    late_mask = [torch.as_tensor(((C.cand_time[b].numpy() - CUT_MS) // DAY_MS) >= nd // 2) for b in
                 range(len(C.layout.buckets))]
    out(f"  oracle from the first {nd // 2} days only (all candidates)   {A(orc_e).mean():.4f}")

    # LEAF + a * oracle
    zo = C.rowstd([torch.logit(o.clamp(1e-4, 1 - 1e-4)) for o in orc])
    best = None
    for a in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        v = A([l + a * z for l, z in zip(leaf, zo)]).mean()
        out(f"  LEAF + {a} * z(oracle)              {v:.4f}")
        if best is None or v > best[1]:
            best = (a, v)
    a_star = best[0]
    v_full = A([l + a_star * z for l, z in zip(leaf, zo)])
    G_all = (v_full - v_leaf).mean()
    out(f"  -> G_all (LEAF + {a_star} z(oracle) minus LEAF) = {1000 * G_all:+.2f}e-3; oracle alone minus LEAF "
        f"{1000 * (v_orc - v_leaf).mean():+.2f}e-3")

    # ----- item-level frame ------------------------------------------------------------------------------------
    it = B["item_feat"].copy()
    it["q"] = q
    it["n_rand"] = N_i
    # LEAF item effect: mean of its standardised score over all candidate cells that hold the item
    s_sum = np.zeros(n)
    s_cnt = np.zeros(n)
    for b, bk in enumerate(C.layout.buckets):
        mk = C.mask[b].numpy()
        np.add.at(s_sum, bk["cand"].numpy()[mk], leaf[b].numpy()[mk])
        np.add.at(s_cnt, bk["cand"].numpy()[mk], 1.0)
    it["leaf_item"] = s_sum / np.maximum(s_cnt, 1)
    pre = lg  # already strictly before the cut
    g = pre.groupby("i")
    it["pre_imp"] = g.size().reindex(range(n)).fillna(0).values
    it["pre_clk"] = g.is_click.sum().reindex(range(n)).fillna(0).values
    it["pre_lv"] = g.long_view.sum().reindex(range(n)).fillna(0).values
    it["pre_users"] = g.u.nunique().reindex(range(n)).fillna(0).values
    it["pre_ctr"] = (it.pre_clk + 1) / (it.pre_imp + 2)
    t1 = pre[pre.tab == 1].groupby("i")
    it["pre_tab1_imp"] = t1.size().reindex(range(n)).fillna(0).values
    it["pre_tab1_ctr"] = ((t1.is_click.sum().reindex(range(n)).fillna(0) + 1) / (it.pre_tab1_imp + 2)).values
    last3 = pre[pre.time_ms >= CUT_MS - 3 * DAY_MS].groupby("i")
    it["pre_last3_imp"] = last3.size().reindex(range(n)).fillna(0).values
    it["pre_last3_clk"] = last3.is_click.sum().reindex(range(n)).fillna(0).values
    # user-standardised click residual: click minus the user's pre-cut click rate (exposure-shift adjustment)
    urate = pre.groupby("u").is_click.mean()
    pre_res = pre.is_click.values - urate.reindex(pre.u.values).values
    it["pre_userres"] = pd.Series(pre_res).groupby(pre.i.values).mean().reindex(range(n)).fillna(0).values
    # concurrent standard log (post-cut; DIAGNOSTIC of drift; excluded from every method)
    post = pd.read_csv(os.path.join(PURE, "log_standard_4_22_to_5_08_pure.csv"),
                       usecols=["user_id", "video_id", "is_click", "long_view", "tab"])
    post = post[post.user_id.isin(B["users_raw"])]
    post["i"] = np.searchsorted(B["items_raw"], post.video_id.values)
    gp = post.groupby("i")
    it["post_imp"] = gp.size().reindex(range(n)).fillna(0).values
    it["post_clk"] = gp.is_click.sum().reindex(range(n)).fillna(0).values
    it["post_ctr"] = (it.post_clk + 1) / (it.post_imp + 2)
    it["growth"] = np.log1p(it.post_imp) - np.log1p(it.pre_imp)
    # out-of-pool author statistics (pre-cut lifelong tables, summed over users)
    L = np.load(os.path.join(ROOT, "data", "raw", "lifelong_27k.npz"), allow_pickle=True)
    for key in ("author_id", "music_id", "tag"):
        vals = pd.Index(L[f"{key}_values"].astype(str))
        kk = L[f"{key}_pre_k"]
        ncl = L[f"{key}_pre_ncl"].astype(np.float64)
        agg = np.zeros((len(vals), 3))
        np.add.at(agg, kk, ncl)
        if key == "tag":
            ks = it[key].astype(str).str.split(",").str[0]
        else:
            ks = pd.to_numeric(it[key], errors="coerce").astype("Int64").astype(str)
        code = vals.get_indexer(ks)
        A_ = np.vstack([agg, np.zeros((1, 3))])[np.where(code < 0, len(vals), code)]
        it[f"{key}_oop_imp"] = A_[:, 0]
        it[f"{key}_oop_ctr"] = (A_[:, 1] + 1) / (A_[:, 0] + 2)
        it[f"{key}_oop_lvr"] = (A_[:, 2] + 1) / (A_[:, 0] + 2)
    it["duration"] = it.video_duration.fillna(it.video_duration.median())
    it["aspect"] = (B["item_feat"].index * 0).astype(float)
    vf = pd.read_csv(os.path.join(PURE, "video_features_basic_pure.csv")).set_index("video_id").reindex(B["items_raw"])
    it["aspect"] = (vf.server_height / vf.server_width).values
    it["upload_dt"] = vf.upload_dt.values
    it["music_type"] = vf.music_type.fillna(-1).values
    ok = it.n_rand >= 50
    out(f"\n2. Item-level Spearman with the uniform rate q (items with >= 50 random impressions: {int(ok.sum())})")
    # split-half reliability
    rng = np.random.default_rng(0)
    users = tl.u.unique()
    half = set(rng.choice(users, len(users) // 2, replace=False).tolist())
    hm = tl.u.isin(half).values
    q1 = (np.bincount(tl.i[hm], weights=tl.is_click[hm], minlength=n) + 1) / (np.bincount(tl.i[hm], minlength=n) + 2)
    q2 = (np.bincount(tl.i[~hm], weights=tl.is_click[~hm], minlength=n) + 1) / (np.bincount(tl.i[~hm], minlength=n) + 2)
    rel = stats.spearmanr(q1[ok], q2[ok])[0]
    out(f"  split-half reliability of q: {rel:.3f}  (Spearman-Brown full-data ceiling {2 * rel / (1 + rel):.3f})")
    cols = ["leaf_item", "pre_clk", "pre_imp", "pre_users", "pre_lv", "pre_ctr", "pre_tab1_ctr", "pre_last3_clk",
            "pre_userres", "post_clk", "post_imp", "post_ctr", "growth", "author_id_oop_imp", "author_id_oop_ctr",
            "author_id_oop_lvr", "music_id_oop_ctr", "tag_oop_ctr", "duration", "aspect", "music_type"]
    for c in cols:
        rho = stats.spearmanr(it.loc[ok, c], it.loc[ok, "q"])[0]
        out(f"  {c:22s} {rho:+.3f}")
    # first-half vs second-half uniform rates (within-period drift)
    q_e = (Ce + 1) / (Ne + 2)
    Nl, Cl = N_i - Ne, C_i - Ce
    q_l = (Cl + 1) / (Nl + 2)
    ok2 = (Ne >= 30) & (Nl >= 30)
    out(f"  uniform rate, first {nd // 2} days vs later days: Spearman {stats.spearmanr(q_e[ok2], q_l[ok2])[0]:.3f} "
        f"({int(ok2.sum())} items with >= 30 in both); same-size random-user halves: see reliability above")
    # same-size comparison: random half of early impressions vs random half of early impressions
    em = early & hm
    em2 = early & ~hm
    qa = (np.bincount(tl.i[em], weights=tl.is_click[em], minlength=n) + 1) / (np.bincount(tl.i[em], minlength=n) + 2)
    qb = (np.bincount(tl.i[em2], weights=tl.is_click[em2], minlength=n) + 1) / (np.bincount(tl.i[em2], minlength=n) + 2)
    lm = (~early) & hm
    qc = (np.bincount(tl.i[lm], weights=tl.is_click[lm], minlength=n) + 1) / (np.bincount(tl.i[lm], minlength=n) + 2)
    nE = np.bincount(tl.i[em], minlength=n)
    nE2 = np.bincount(tl.i[em2], minlength=n)
    nL = np.bincount(tl.i[lm], minlength=n)
    ok3 = (nE >= 15) & (nE2 >= 15) & (nL >= 15)
    out(f"  matched-size check ({int(ok3.sum())} items): early(user half A) vs early(user half B) "
        f"{stats.spearmanr(qa[ok3], qb[ok3])[0]:.3f}; early(half B) vs late(half A) {stats.spearmanr(qb[ok3], qc[ok3])[0]:.3f}")

    # ----- 3. attribution ---------------------------------------------------------------------------------------
    zq = np.log(q / (1 - q))
    zq = (zq - zq.mean()) / zq.std()
    w = np.sqrt(N_i)
    X1 = np.vstack([np.ones(n), it.leaf_item.values]).T
    coef = np.linalg.lstsq(X1 * w[:, None], zq * w, rcond=None)[0]
    r = zq - X1 @ coef
    out(f"\n3. Attribution. item residual r = z(q) - {coef[1]:.3f} * LEAF item effect - {coef[0]:.3f}; "
        f"var(z q) explained by LEAF item effect: {1 - np.average(r ** 2, weights=w) / np.average((zq - np.average(zq, weights=w)) ** 2, weights=w):.3f}")
    # LOO residual per candidate: z of the LOO rate minus the same projection
    zl = [torch.logit(o.clamp(1e-4, 1 - 1e-4)) for o in orc]
    zm, zs = np.log(q / (1 - q)).mean(), np.log(q / (1 - q)).std()
    rl = [((z - zm) / zs - torch.as_tensor(X1 @ coef, dtype=torch.float32)[bk["cand"]])
          for z, bk in zip(zl, C.layout.buckets)]

    def gain(corr, mask=None):
        best = (0.0, 0.0)
        for a in (0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5):
            S = [l + a * (c if mask is None else c * m) for l, c, m in
                 zip(leaf, corr, mask if mask is not None else [None] * len(leaf))]
            v = (A(S) - v_leaf).mean()
            if v > best[0]:
                best = (v, a)
        return best
    Gr, ar = gain(rl)
    out(f"  full residual correction S_LEAF + a r(LOO): {1000 * Gr:+.2f}e-3 (a={ar})  [reference G_all of z(oracle): "
        f"{1000 * G_all:+.2f}e-3]")

    def binned(col, nb=10, cat=False):
        v = it[col]
        if cat:
            b = pd.factorize(v.astype(str))[0]
        else:
            b = pd.qcut(v.rank(method="first"), nb, labels=False).values
        # cross-fitted bin means: items split in two halves; each half gets the bin means of the other
        h = np.random.default_rng(1).random(n) < 0.5
        est = np.zeros(n)
        for part in (h, ~h):
            other = ~part
            df = pd.DataFrame({"b": b[other], "r": r[other], "w": w[other]})
            means = df.groupby("b").apply(lambda x: np.average(x.r, weights=x.w + 1e-9), include_groups=False)
            cnt = df.groupby("b").size()
            means = means * cnt / (cnt + 5)    # shrink small bins
            est[part] = pd.Series(b[part]).map(means).fillna(0).values
        R2 = 1 - np.average((r - est) ** 2, weights=w) / np.average(r ** 2, weights=w)
        g_, a_ = gain([torch.as_tensor(est, dtype=torch.float32)[bk["cand"]] for bk in C.layout.buckets])
        return R2, g_, a_
    out("  factor (bins)                 R2 of residual (cross-fit)   nDCG gain x1e-3   share of full residual gain")
    factors = [("pre_clk", 10, False), ("pre_imp", 10, False), ("pre_ctr", 10, False), ("pre_tab1_ctr", 10, False),
               ("pre_last3_clk", 10, False), ("pre_userres", 10, False),
               ("post_clk [post-cut, leak]", 10, False), ("post_ctr [post-cut, leak]", 10, False),
               ("growth [post-cut, leak]", 10, False),
               ("author_id_oop_ctr", 10, False), ("author_id_oop_imp", 10, False), ("author_id_oop_lvr", 10, False),
               ("music_id_oop_ctr", 10, False), ("duration", 10, False), ("aspect", 5, False), ("upload_dt", 0, True),
               ("upload_type", 0, True), ("video_type", 0, True), ("music_type", 0, True), ("tag", 0, True)]
    for name, nb, cat in factors:
        col = name.split(" ")[0]
        R2, g_, a_ = binned(col, nb, cat)
        out(f"  {name:30s} {R2:+.3f}                      {1000 * g_:+6.2f}           {g_ / Gr:5.2f}")
    # all factors jointly, gradient boosting, cross-fitted (pre-cut factors only, then with post-cut)
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        pre_cols = ["pre_clk", "pre_imp", "pre_users", "pre_lv", "pre_ctr", "pre_tab1_ctr", "pre_tab1_imp",
                    "pre_last3_clk", "pre_last3_imp", "pre_userres", "author_id_oop_imp", "author_id_oop_ctr",
                    "author_id_oop_lvr", "music_id_oop_imp", "music_id_oop_ctr", "tag_oop_ctr", "duration", "aspect",
                    "music_type", "leaf_item"]
        Xc = it[pre_cols].copy()
        for c in ("upload_dt", "upload_type", "video_type", "tag"):
            Xc[c] = pd.factorize(it[c].astype(str))[0]
        for label, cols_ in (("all pre-cut item factors (GBM)", list(Xc.columns)),
                             ("pre-cut + post-cut [leak] (GBM)", list(Xc.columns) + ["post_clk", "post_imp", "post_ctr",
                                                                                    "growth"])):
            Xm = pd.concat([Xc, it[["post_clk", "post_imp", "post_ctr", "growth"]]], axis=1)[cols_].values
            est = np.zeros(n)
            folds = np.random.default_rng(2).integers(0, 5, n)
            for f in range(5):
                tr, te = folds != f, folds == f
                mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                    min_samples_leaf=40, random_state=0)
                mdl.fit(Xm[tr], r[tr], sample_weight=w[tr])
                est[te] = mdl.predict(Xm[te])
            R2 = 1 - np.average((r - est) ** 2, weights=w) / np.average(r ** 2, weights=w)
            g_, a_ = gain([torch.as_tensor(est, dtype=torch.float32)[bk["cand"]] for bk in C.layout.buckets])
            out(f"  {label:30s} {R2:+.3f}                      {1000 * g_:+6.2f}           {g_ / Gr:5.2f}")
    except ImportError:
        out("  (sklearn missing: joint factor model skipped)")

    # item groups: exact correction only on the group's items
    out("\n  item groups (exact LOO residual correction applied only to the group's items; share of the full gain)")
    for col, nb in (("pre_imp", 5), ("pre_clk", 5), ("author_id_oop_imp", 4), ("duration", 4), ("q", 5)):
        bins = pd.qcut(it[col].rank(method="first"), nb, labels=False).values
        for b_ in range(nb):
            sel = torch.as_tensor(bins == b_)
            mask = [sel[bk["cand"]].float() for bk in C.layout.buckets]
            g_, _ = gain(rl, mask)
            lo, hi = it[col][bins == b_].min(), it[col][bins == b_].max()
            out(f"  {col:18s} bin {b_} [{lo:.4g}, {hi:.4g}] ({int((bins == b_).sum())} items): {1000 * g_:+6.2f}e-3 "
                f"share {g_ / Gr:5.2f}")
    # user groups: exact additive split of the oracle gains
    out("\n  user groups (additive: share of G_all = LEAF + a z(oracle) minus LEAF; and oracle-alone minus LEAF)")
    uu = A.uid
    uact = lg.groupby("u").size().reindex(uu).fillna(0).values
    L2 = np.load(os.path.join(ROOT, "data", "raw", "lifelong_27k.npz"), allow_pickle=True)
    life = pd.Series(L2["tot_pre"][:, 0], index=L2["tot_pre_user"]).reindex(B["users_raw"][uu]).fillna(0).values
    ncand = np.array([len(C.test[k][1]) for b, bk in enumerate(C.layout.buckets) if A.um[b] is not None
                      for k in bk["k"][A.um[b].users]])
    for name, v in (("pre-cut in-pool impressions", uact), ("lifelong impressions", life), ("candidates", ncand)):
        bins = pd.qcut(pd.Series(v).rank(method="first"), 5, labels=False).values
        for b_ in range(5):
            mk = bins == b_
            out(f"  {name:28s} q{b_} [{v[mk].min():.0f}, {v[mk].max():.0f}] users {int(mk.sum())}: LEAF "
                f"{v_leaf[mk].mean():.4f}, oracle {v_orc[mk].mean():.4f}; share of G_all "
                f"{(v_full - v_leaf)[mk].sum() / (v_full - v_leaf).sum():.2f}, share of (oracle - LEAF) "
                f"{(v_orc - v_leaf)[mk].sum() / (v_orc - v_leaf).sum():.2f}")
    # 4. by test day: the oracle correction applied only to candidates impressed on early / late days
    out("\n4. Test time: exact residual correction applied only to candidates by the day of their random impression")
    for lo_, hi_ in ((0, 4), (4, 8), (8, 12), (12, 17)):
        mask = [torch.as_tensor(((C.cand_time[b].numpy() - CUT_MS) // DAY_MS >= lo_) &
                                ((C.cand_time[b].numpy() - CUT_MS) // DAY_MS < hi_)).float()
                for b in range(len(C.layout.buckets))]
        share_cells = sum(float((m * C.mask[b]).sum()) for b, m in enumerate(mask)) / sum(float(m.sum()) for m in C.mask)
        g_, _ = gain(rl, mask)
        out(f"  days {lo_:2d}-{hi_ - 1:2d} after the cut ({share_cells:.2f} of candidates): {1000 * g_:+6.2f}e-3 "
            f"share {g_ / Gr:5.2f}")
    it.to_csv(os.path.join("/tmp/claude-0" if os.path.isdir("/tmp/claude-0") else "/tmp", "v11_item_frame.csv"))
    out("item frame written to the scratch directory (v11_item_frame.csv)")


if __name__ == "__main__":
    main()
