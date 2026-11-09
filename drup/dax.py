"""DAX: detection-adjusted attractiveness of the items (paper/v7/DESIGN.md, Sections 5 and 6).

A click on the pre-cut standard log is modelled as the product of an examination and an attraction:

    E ~ Bernoulli(theta_u * kappa_t)     the user examines the shown item (tab-specific factor kappa_t)
    R ~ Bernoulli(alpha_i)               the item is attractive to a random viewer
    y = E * R

fitted by EM with closed-form M-steps (fit_dax). alpha_i has a Beta prior centred on its tag mean (or on a
content regression, or on one global mean for the ablation); theta_u has a prior centred on the
user's lifelong click rate (out-of-pool history before the cut-off). The estimate is an item appeal under a
uniform insertion, used as an extra channel of a collaborative backbone: post hoc (a weight on validation) or
trained jointly (DAXChannel inside LEAFLGNDAX).

Only the pre-cut standard log is read (fit_dax_dataset filters on the cut-off whatever the input holds); test
labels, validation labels and the random-exposure log never enter the fit.

Pieces:
  fit_dax              EM of the detection x attractiveness model (float64; index_add_ accumulations)
  fit_dax_dataset      the same on a drup.data2 dict: pre-cut log, tag codes, lifelong rates, optional strata
  content_centre       Fay-Herriot style prior centre: ridge regression of logit alpha on content covariates
  item_content_codes   the content covariates of the pool items (tag, upload type, duration, author history, ...)
  fit_additive         MRP-style additive logistic model with tab effects (the fair additive arm, no detection)
  dax_variant          the configuration of an ablation arm (no_tab, no_life_prior, no_tag_prior; None = full)
  within_code_perm / all_perm   negative controls of the item appeal (gate B2)
  masked_rowstd        per-row standardisation over the real candidates of a bucket
  DAXChannel           standardised item appeal with the interface of drup.leaf.LifelongTables
  LEAFLGNDAX           LEAFLGN plus one learned weight on the DAX channel (joint variant)
"""

import os

import numpy as np
import pandas as pd
import torch

from drup.leaf import CUT_MS, LEAFLGN

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
LIFELONG = os.path.join(ROOT, "data", "raw", "lifelong_27k.npz")
VIDEO = os.path.join(ROOT, "data", "raw", "KuaiRand-Pure", "data", "video_features_basic_pure.csv")
CONTENT = ("tag", "upload_type", "video_type", "dur", "aspect", "music_type", "a_imp", "a_ctr", "a_lvr")


def tag_codes(item_feat):
    """Integer tag code of every pool item (first tag). An item without a tag gets its own code (the largest + 1),
    as in the probes: missing values are a group of their own, not a missing index."""
    v = pd.factorize(item_feat["tag"].astype(str))[0]
    return np.where(v < 0, v.max() + 1, v)


def pre_cut(log, cut=CUT_MS):
    """Rows of the standard log strictly before the cut-off (the only rows DAX reads)."""
    return log[log.time_ms.values < cut]


def fit_dax(log, n_users, n_items, tag_codes, life_rate=None, prior=80.0, k_life=100.0, tab=True, centre="tag",
            content_codes=None, strata=None, prior_s=None, iters=40):
    """EM of the detection x attractiveness model on the training impressions `log` (columns u, i, tab, is_click;
    already restricted to the pre-cut period by the caller, see fit_dax_dataset).

    prior      strength of the Beta prior on alpha_i (pseudo impressions), centred on c_i
    centre     "tag": c_i = pooled attraction of the tag of i; "content": Fay-Herriot ridge regression on
               content_codes (tag centre in the first 5 iterations); "global": one pooled mean (ablation)
    k_life     strength of the prior on theta_u, centred on life_rate (lifelong click rate rescaled to the in-pool
               mean examination); 0 = no lifelong prior, theta_u gets a Beta(2, 2) pseudo-count prior
    tab        tab-specific examination kappa_t (kappa of the first tab with impressions, 1, is the reference)
    strata     optional (n_users,) user strata and prior_s: item x stratum alpha_{i,s} shrunk to alpha_i (MRP cells)

    Returns (logit_alpha (n,), theta (m,), kappa (T,), logit_alpha_strata (n, S) or None), float32 except kappa.
    The M-steps are closed form: theta_u = (sum P(E=1) + k c_u) / (sum kappa + k), kappa_t = sum P(E=1) over tab t
    / sum theta_u over tab t, alpha_i = (sum P(R=1) + a c_i) / (n_i + a).
    """
    m, n = int(n_users), int(n_items)
    u = torch.as_tensor(log.u.values.astype(np.int64))
    i = torch.as_tensor(log.i.values.astype(np.int64))
    tb = torch.as_tensor(log.tab.values.astype(np.int64))
    y = torch.as_tensor(log.is_click.values, dtype=torch.float64)
    tag = torch.as_tensor(np.asarray(tag_codes), dtype=torch.long)
    G = int(tag.max()) + 1
    T = int(tb.max()) + 1
    if k_life > 0:
        if life_rate is None:
            raise ValueError("k_life > 0 needs life_rate")
        lr = torch.as_tensor(np.asarray(life_rate), dtype=torch.float64)
        lr = lr / lr.mean()
    if centre == "content" and content_codes is None:
        raise ValueError("centre content needs content_codes")
    cov = None if content_codes is None else {k: torch.as_tensor(np.asarray(v), dtype=torch.long)
                                             for k, v in content_codes.items()}
    use_s = prior_s is not None
    if use_s:
        S = int(strata.max()) + 1
        su = torch.as_tensor(np.asarray(strata), dtype=torch.long)[u]
    th = torch.full((m,), 0.7, dtype=torch.float64)
    ka = torch.ones(T, dtype=torch.float64)
    al = torch.full((n,), 0.6, dtype=torch.float64)
    als = torch.full((n, S), 0.6, dtype=torch.float64) if use_s else None
    ni = torch.zeros(n, dtype=torch.float64).index_add_(0, i, torch.ones(len(i), dtype=torch.float64))
    if use_s:
        nis = torch.zeros(n * S, dtype=torch.float64).index_add_(0, i * S + su,
                                                               torch.ones(len(i), dtype=torch.float64)).view(n, S)
    for it in range(iters):
        a = als[i, su] if use_s else al[i]
        t = (th[u] * ka[tb]).clamp(1e-6, 1 - 1e-6)
        den = (1 - t * a).clamp_min(1e-9)
        pE = torch.where(y > 0, torch.ones_like(t), t * (1 - a) / den)
        pR = torch.where(y > 0, torch.ones_like(t), (1 - t) * a / den)
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
        sR = torch.zeros(n, dtype=torch.float64).index_add_(0, i, pR)
        if centre == "global":
            c = torch.full((n,), float((sR.sum() + 1.0) / (ni.sum() + 2.0)), dtype=torch.float64)
        elif centre == "tag" or it < 5:
            tm = (torch.zeros(G, dtype=torch.float64).index_add_(0, tag, sR) + 1.0) / \
                 (torch.zeros(G, dtype=torch.float64).index_add_(0, tag, ni) + 2.0)
            c = tm[tag]
        else:
            c = content_centre(al, ni, cov)
        al = ((sR + prior * c) / (ni + prior)).clamp(1e-4, 1 - 1e-4)
        if use_s:
            sRs = torch.zeros(n * S, dtype=torch.float64).index_add_(0, i * S + su, pR).view(n, S)
            als = ((sRs + prior_s * al[:, None]) / (nis + prior_s)).clamp(1e-4, 1 - 1e-4)
    la_s = torch.logit(als).float() if use_s else None
    return torch.logit(al).float(), th.float(), ka, la_s


def content_centre(al, ni, cov, ridge=10.0):
    """Fay-Herriot style prior mean: weighted ridge regression of logit(alpha) on one-hot content covariates
    (the weights are sqrt(n_i + 1); the intercept is the weighted mean)."""
    n = len(al)
    cols, off = [], 0
    for g in CONTENT:
        v = cov[g]
        cols.append(v + off)
        off += int(v.max()) + 1
    idx = torch.stack(cols, 1)
    X = torch.zeros(n, off, dtype=torch.float64)
    X.scatter_(1, idx, 1.0)
    y = torch.logit(al)
    w = (ni + 1.0).sqrt()
    mu = (w * y).sum() / w.sum()
    Xw = X * w[:, None]
    beta = torch.linalg.solve(Xw.T @ X + ridge * torch.eye(off, dtype=torch.float64), Xw.T @ (y - mu))
    return torch.sigmoid(mu + X @ beta)


def lifelong_rate(users_raw, path=LIFELONG):
    """Lifelong click rate (C_u + 1) / (N_u + 2) of the dataset users, from the out-of-pool history before the cut-off
    (table tot_pre of experiments/v9_build_lifelong.py); users without history get the mean rate."""
    L = np.load(path, allow_pickle=True)
    tot = pd.DataFrame(L["tot_pre"], index=L["tot_pre_user"], columns=["n", "c", "l"]).reindex(np.asarray(users_raw))
    r = (tot.c + 1) / (tot.n + 2)
    return r.fillna(r.mean()).values


def user_strata(users_raw, path=LIFELONG, k=5):
    """User strata by the lifelong click rate (quantile groups); users without history form their own stratum k."""
    L = np.load(path, allow_pickle=True)
    tot = pd.DataFrame(L["tot_pre"], index=L["tot_pre_user"], columns=["n", "c", "l"]).reindex(np.asarray(users_raw))
    r = (tot.c + 1) / (tot.n + 2)
    s = np.full(len(users_raw), k, dtype=np.int64)
    ok = tot.n.notna().values
    s[ok] = pd.qcut(r[ok].rank(method="first"), k, labels=False).values
    return s


def item_content_codes(item_feat, items_raw, video=VIDEO, path=LIFELONG):
    """Content covariates of the pool items as integer codes (a missing value is its own code): tag, upload type,
    video type, duration decile, aspect ratio, music type, and the out-of-pool author history (impressions, click
    rate, long-view rate, quantile groups). The author history is the pre-cut lifelong table."""
    it = item_feat
    vf = pd.read_csv(video).set_index("video_id").reindex(np.asarray(items_raw))
    L = np.load(path, allow_pickle=True)
    vals = pd.Index(L["author_id_values"].astype(str))
    agg = np.zeros((len(vals), 3))
    np.add.at(agg, L["author_id_pre_k"], L["author_id_pre_ncl"].astype(np.float64))
    code = vals.get_indexer(pd.to_numeric(it["author_id"], errors="coerce").astype("Int64").astype(str))
    A = np.vstack([agg, np.zeros((1, 3))])[np.where(code < 0, len(vals), code)]
    ctr = (A[:, 1] + 1) / (A[:, 0] + 2)
    lvr = (A[:, 2] + 1) / (A[:, 0] + 2)
    cov = {
        "tag": tag_codes(it),
        "upload_type": pd.factorize(it["upload_type"].astype(str))[0],
        "video_type": pd.factorize(it["video_type"].astype(str))[0],
        "dur": pd.qcut(it["video_duration"].fillna(it["video_duration"].median()).rank(method="first"), 10,
                       labels=False).values,
        "aspect": pd.factorize((vf.server_height / vf.server_width).round(2).astype(str).values)[0],
        "music_type": pd.factorize(vf.music_type.fillna(-1).astype(str).values)[0],
        "a_imp": np.digitize(np.log1p(A[:, 0]), np.quantile(np.log1p(A[:, 0]), np.linspace(0.1, 0.9, 9))),
        "a_ctr": np.digitize(ctr, np.quantile(ctr, np.linspace(0.1, 0.9, 9))),
        "a_lvr": np.digitize(lvr, np.quantile(lvr, np.linspace(0.1, 0.9, 9))),
    }
    out = {}
    for k, v in cov.items():
        v = np.asarray(v, dtype=np.int64)
        out[k] = np.where(v < 0, v.max() + 1, v)
    return out


def fit_dax_dataset(d, cut=CUT_MS, content=None, life=None, strata=None, **kw):
    """fit_dax on a drup.data2 dict. Reads d["log"] (filtered to time_ms < cut), d["item_feat"]["tag"] and
    d["users_raw"]; nothing else of d is read. life / strata / content: precomputed inputs (see lifelong_rate,
    user_strata, item_content_codes), computed from the lifelong tables when None."""
    log = pre_cut(d["log"], cut)
    assert len(log) == 0 or int(log.time_ms.max()) < cut, "a post-cut row reached the DAX fit"
    m, n = len(d["users_raw"]), len(d["item_feat"])
    tag = tag_codes(d["item_feat"])
    if kw.get("k_life", 100.0) > 0 and life is None:
        life = lifelong_rate(d["users_raw"])
    if kw.get("prior_s") is not None and strata is None:
        strata = user_strata(d["users_raw"])
    return fit_dax(log, m, n, tag, life_rate=life, content_codes=content, strata=strata, **kw)


def dax_variant(cfg, ablation=None):
    """Configuration of one DAX ablation arm (gate B2, DESIGN.md Section 6). ablation None is the switch off: the
    configuration is returned unchanged, so the arm is the full DAX arm. Otherwise the named switch takes its off value:
    no_tab (no examination factor kappa_t), no_life_prior (k_life = 0, no lifelong prior on theta_u), no_tag_prior
    (alpha_i centred on the global mean, not on its tag)."""
    if ablation is None:
        return dict(cfg)
    if ablation == "no_tab":
        return dict(cfg, tab=False)
    if ablation == "no_life_prior":
        return dict(cfg, k_life=0.0)
    if ablation == "no_tag_prior":
        return dict(cfg, centre="global")
    raise ValueError(f"unknown DAX ablation {ablation!r}")


def fit_additive(log, n_users, n_items, tag_codes, lam_e=5.0, lam_b=1.0, lam_g=1.0, tab=True, iters=60):
    """MAP logistic regression of the pre-cut clicks, additive on the logit scale (the MRP-style arm):
        logit P(click) = mu + b_u + g_tag(i) + e_i + t_kappa   (ridge on b, g and e; t unpenalised).
    Returns the item appeal g_tag(i) + e_i (n,) as float32; the user effects are constant within a candidate list
    and do not change the ranking."""
    m, n = int(n_users), int(n_items)
    u = torch.as_tensor(log.u.values.astype(np.int64))
    i = torch.as_tensor(log.i.values.astype(np.int64))
    tb = torch.as_tensor(log.tab.values.astype(np.int64))
    y = torch.as_tensor(log.is_click.values, dtype=torch.float64)
    tag = torch.as_tensor(np.asarray(tag_codes), dtype=torch.long)
    G, T = int(tag.max()) + 1, int(tb.max()) + 1
    P = {k: torch.zeros(s, dtype=torch.float64, requires_grad=True)
         for k, s in (("mu", 1), ("b", m), ("g", G), ("e", n), ("t", T))}
    opt = torch.optim.LBFGS(list(P.values()), lr=1.0, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = P["mu"] + P["b"][u] + P["g"][tag][i] + P["e"][i]
        if tab:
            z = z + P["t"][tb]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction="sum")
        reg = lam_b * (P["b"] ** 2).sum() + lam_g * (P["g"] ** 2).sum() + lam_e * (P["e"] ** 2).sum()
        tot = loss + reg
        tot.backward()
        return tot
    opt.step(closure)
    with torch.no_grad():
        appeal = P["g"][tag] + P["e"]
    return appeal.detach().float()


def within_code_perm(codes, seed=0):
    """Permutation of the items that moves every item's appeal to another item of its own code group (the tag mean
    is kept, the item part is destroyed). Apply as values[perm]."""
    codes = np.asarray(codes)
    rng = np.random.default_rng(seed)
    perm = np.arange(len(codes))
    for c in np.unique(codes):
        idx = np.flatnonzero(codes == c)
        perm[idx] = idx[rng.permutation(len(idx))]
    return perm


def all_perm(n, seed=0):
    """Permutation of all items (every item receives another item's appeal). Apply as values[perm]."""
    return np.random.default_rng(seed).permutation(n)


def masked_rowstd(S, mask):
    """Per-row standardisation over the real candidates (mask True); padded cells are returned as 0."""
    s = torch.where(mask, S, torch.zeros_like(S))
    cnt = mask.sum(1, keepdim=True).clamp_min(1)
    mu = s.sum(1, keepdim=True) / cnt
    var = torch.where(mask, (S - mu) ** 2, torch.zeros_like(S)).sum(1, keepdim=True) / cnt
    return torch.where(mask, (S - mu) / var.sqrt().clamp_min(1e-9), torch.zeros_like(S))


class DAXChannel:
    """Standardised DAX item appeal with the interface of drup.leaf.LifelongTables: pairs(u, i), compact(rows, cand)
    and dense_rows(rows). The appeal is standardised over the pool items (mean 0, sd 1 of logit alpha), so the
    weight of the joint variant is on a fixed scale. With stratum cells, alpha_{i, s(u)} is used for user u."""

    def __init__(self, logit_alpha, strata=None, logit_alpha_strata=None):
        z = torch.as_tensor(np.asarray(logit_alpha), dtype=torch.float64)
        mu, sd = z.mean(), z.std().clamp_min(1e-12)
        self.n = len(z)
        self.z = ((z - mu) / sd).float()
        self.strata = None if strata is None else torch.as_tensor(np.asarray(strata), dtype=torch.long)
        self.zs = None
        if logit_alpha_strata is not None:
            zs = torch.as_tensor(np.asarray(logit_alpha_strata), dtype=torch.float64)
            self.zs = ((zs - mu) / sd).float()

    def pairs(self, u, i):
        """Channel value of the pairs (u, i), broadcast as in indexing (float32)."""
        if self.zs is None:
            return self.z[i]
        return self.zs[i, self.strata[u]]

    def compact(self, rows, cand):
        """(U, L) channel values for the users rows[k] (dataset ids) and the candidates cand[k, j]."""
        return self.pairs(rows[:, None], cand)

    def dense_rows(self, rows):
        """(len(rows), n): the channel for all pool items of the users rows."""
        if self.zs is None:
            return self.z[None, :].expand(len(rows), self.n)
        return self.zs[:, self.strata[rows]].T


class LEAFLGNDAX(LEAFLGN):
    """LEAF-LGN with one more fixed channel, the DAX item appeal, and a learned weight g_dax (zero at start)."""

    def __init__(self, base, tabs, pop, dax, mask=None):
        super().__init__(base, tabs, pop, mask)
        self.dax = dax
        self.g_dax = torch.nn.Parameter(torch.zeros(1))

    def forward(self, u, i, emb=None):
        return super().forward(u, i, emb) + self.g_dax * self.dax.pairs(u, i)

    def full_scores(self, rows):
        return super().full_scores(rows) + self.g_dax * self.dax.dense_rows(rows)
