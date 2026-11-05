"""LEAF: lifelong exposure-conditional affinity transfer (paper/v6/DESIGN.md, Sections 5 and 6).

Pieces used by experiments/run_leaf.py:
  LifelongTables      per (user, key value) lifelong lift and familiarity on the content keys tag, author_id and
                      music_id, from the tables of experiments/v9_build_lifelong.py (out-of-pool history only)
  collective_gram /   closed-form collective linear autoencoder (5.2): Gram accumulation over user chunks and the
  collective_solve    zero-diagonal solution; collective_ease is the dense form used by the tests
  LEAFLGN             LightGCN with the channel weights and the popularity offset (5.2b)
  fit_bpr_policy      BPR loop with the policy channel (copy of the bpr branch of drup.learned.train)
  Layout, BucketSelector   candidate layout bucketed by list length, and per-split validation selection on it
  protocol_split, check_temporal_protocol   the temporal protocol and its checks

Inputs of the lifelong channels are the npz tables, the user ids and the pool item features only; the test list
and the validation labels are never read here.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from drup.data import split_test
from drup.learned import LightGCN

KEYS = ("tag", "author_id", "music_id")
# channel c = 2 * key index + type; type 0 = exposure-conditional lift, type 1 = familiarity log(1 + n)
CHANNELS = ("tag_lift", "tag_fam", "author_id_lift", "author_id_fam", "music_id_lift", "music_id_fam")
CUT_MS = 1650556800000          # 2022-04-22 00:00 UTC+8, first day of the random exposure period


NA_KEY = "<NA>"                 # key value of a missing tag, author or music id: no lifelong row, so a zero channel
MISSING_KEYS = (NA_KEY, "nan")  # "nan" is what an earlier build stored for missing tags; no pool item maps to it


def key_strings(col, key):
    """Key value of every pool item, as the strings stored in the lifelong tables (same as the builder). A pool item
    without a key value (missing tag, author or music id) maps to NA_KEY and gets a zero channel (see LifelongTables)."""
    if key == "tag":
        return col.where(col.notna(), NA_KEY).astype(str).str.split(",").str[0]
    return pd.to_numeric(col, errors="coerce").astype("Int64").astype(str)


def cyclic_perm(m, seed=0):
    """A permutation of range(m) without fixed points: every user takes the lifelong table of another user."""
    order = np.random.default_rng(seed).permutation(m)
    perm = np.empty(m, dtype=np.int64)
    perm[order] = np.roll(order, -1)
    return perm


class LifelongTables:
    """Dense lifelong channels for the dataset users and the pool items.

    npz: the tables of experiments/v9_build_lifelong.py (or a dict with the same keys); users_raw: raw user id of
    every dataset row; item_feat: DataFrame of the pool items in dataset order (columns tag, author_id, music_id);
    window: "pre" (impressions before the cut-off, the protocol of the paper) or "all" (concurrent protocol);
    s: Beta-binomial shrinkage of the lift; shuffle: every user takes the table of another user (negative control).

    For a user u and pool item i with key value v = kappa_k(i):
        lift  = (c_uv + s r_u) / (n_uv + s) - r_u,   r_u = (C_u + 1) / (N_u + 2)   (lifelong base rate)
        fam   = log(1 + n_uv)
    Every channel is divided by the standard deviation of its nonzero entries over all (user, pool item) cells,
    and stored in float16. Pool items without a key value, and key-user pairs without impressions, are zero.
    """

    def __init__(self, npz, users_raw, item_feat, window="pre", s=5.0, shuffle=False, seed=0, chunk=2048):
        users_raw = np.asarray(users_raw)
        m, n = len(users_raw), len(item_feat)
        self.m, self.n, self.window, self.s, self.shuffle = m, n, window, float(s), bool(shuffle)
        tot = np.asarray(npz[f"tot_{window}"], dtype=np.float64)
        base_all = pd.Series((tot[:, 1] + 1.0) / (tot[:, 0] + 2.0), index=np.asarray(npz[f"tot_{window}_user"]))
        base = base_all.reindex(users_raw).fillna(base_all.mean()).values
        urow = pd.Index(users_raw)
        perm = cyclic_perm(m, seed) if shuffle else None
        self.tab, self.ci, self.count, self.scale, self.vals = {}, {}, {}, {}, {}
        for k in KEYS:
            vals = pd.Index(np.asarray(npz[f"{k}_values"]).astype(str))
            self.vals[k] = vals.values
            V = len(vals)
            code = vals.get_indexer(key_strings(item_feat[k], k))
            self.ci[k] = torch.as_tensor(np.where(code < 0, V, code), dtype=torch.long)
            ur = urow.get_indexer(np.asarray(npz[f"{k}_{window}_u"]))
            kk = np.asarray(npz[f"{k}_{window}_k"]).astype(np.int64)
            ncl = np.asarray(npz[f"{k}_{window}_ncl"], dtype=np.float64)
            keep = ur >= 0
            ur, kk, ncl = ur[keep], kk[keep], ncl[keep]
            if len(np.unique(ur * V + kk)) != len(ur):
                raise ValueError(f"{k}: duplicated (user, value) rows in the {window} table")
            b = base[ur]
            T = np.zeros((m, V + 1, 2), dtype=np.float32)          # column V is the zero padding of "no value"
            T[ur, kk, 0] = (ncl[:, 1] + self.s * b) / (ncl[:, 0] + self.s) - b
            T[ur, kk, 1] = np.log1p(ncl[:, 0])
            cnt = np.zeros(V)
            np.add.at(cnt, kk, ncl[:, 0])
            self.count[k] = cnt                                       # impressions of each value, dataset users
            Tt = torch.from_numpy(T)
            del T
            s1, s2, c1 = torch.zeros(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64), \
                torch.zeros(2, dtype=torch.float64)
            for c0 in range(0, m, chunk):
                G = Tt[c0:c0 + chunk][:, self.ci[k]].double()
                s1 += G.sum((0, 1))
                s2 += (G * G).sum((0, 1))
                c1 += (G != 0).double().sum((0, 1))
            mean = s1 / c1.clamp_min(1.0)
            std = (s2 / c1.clamp_min(1.0) - mean * mean).clamp_min(1e-24).sqrt()
            out = torch.empty((m, V + 1, 2), dtype=torch.float16)
            for c0 in range(0, m, chunk):
                out[c0:c0 + chunk] = (Tt[c0:c0 + chunk] / std.float()).half()
            del Tt
            if perm is not None:
                out = out[torch.as_tensor(perm)]
            self.tab[k], self.scale[k] = out, std

    def pairs(self, u, i):
        """(..., 6) float32 channel values of the pairs (u, i), broadcast as in indexing; channel order CHANNELS."""
        return torch.cat([self.tab[k][u, self.ci[k][i]].float() for k in KEYS], -1)

    def compact(self, rows, cand):
        """(U, L, 6) channel values for user rows[k] and the candidate items cand[k, j]."""
        return torch.cat([self.tab[k][rows[:, None], self.ci[k][cand]].float() for k in KEYS], -1)

    def dense_rows(self, rows, weights, chunk=2048):
        """(len(rows), n): sum over channels of weights[c] * channel c, for all pool items of the users rows."""
        out = torch.zeros(len(rows), self.n)
        for r0 in range(0, len(rows), chunk):
            rr = rows[r0:r0 + chunk]
            S = 0.0
            for ki, k in enumerate(KEYS):
                T = self.tab[k][rr][:, self.ci[k]].float()          # (r, n, 2)
                S = S + T @ weights[2 * ki:2 * ki + 2]
            out[r0:r0 + len(rr)] = S
        return out


class PhiFull:
    """Lift and familiarity of the key values with at least min_imp impressions among the training users, as dense
    user columns (the lifelong-key block Phi of the free-W variant, LEAF-full)."""

    def __init__(self, tabs, families, min_imp=20):
        self.tabs = tabs
        self.idx = {k: torch.as_tensor(np.flatnonzero((tabs.count[k] >= min_imp) & ~np.isin(tabs.vals[k], MISSING_KEYS)),
                                       dtype=torch.long) for k in families}
        self.p = 2 * sum(len(v) for v in self.idx.values())

    def chunk(self, rows):
        parts = [self.tabs.tab[k][rows][:, idx].float().reshape(len(rows), -1) for k, idx in self.idx.items()]
        return torch.cat(parts, 1)


def rowstd(S):
    S = S - S.mean(1, keepdim=True)
    return S / S.std(1, keepdim=True).clamp_min(1e-9)


def _inputs(O, Y, w, r0, r1):
    """Input Z = X + w N and target X of the users r0:r1, with X = O * Y (clicks) and N = O - X (shown, no click)."""
    o = O[r0:r1]
    x = o * Y[r0:r1].to(o.dtype)
    return x + w * (o - x), x


def collective_gram(O, Y, w, phi=None, chunk=2048):
    """G = A'A and C = A'X for A = [Z, Phi], accumulated over user chunks (no dense m x d matrix)."""
    m, n = O.shape
    d = n + (0 if phi is None else phi.p)
    G, C = torch.zeros(d, d, dtype=O.dtype), torch.zeros(d, n, dtype=O.dtype)
    for r0 in range(0, m, chunk):
        r1 = min(m, r0 + chunk)
        z, x = _inputs(O, Y, w, r0, r1)
        A = z if phi is None else torch.cat([z, phi.chunk(torch.arange(r0, r1)).to(O.dtype)], 1)
        G += A.T @ A
        C += A.T @ x
    return G, C


def collective_solve(G, C, n, lam, lam_phi=0.0):
    """Minimiser of ||X - Z B - Phi W||^2 + lam ||B||^2 + lam_phi ||W||^2 with diag(B) = 0 (DESIGN 5.2):
    Theta = Theta_hat - P[:, :n] diag(mu), P = (G + Lambda)^-1, Theta_hat = P C, mu_j = Theta_hat_jj / P_jj.
    G is restored before returning. Returns B (n x n) and W (p x n, or None when there is no Phi block).
    Precision: G, C, the Cholesky factor, P and Theta are float32, the dtype of the inputs. A float64 solve needs
    twice the memory; at d = n + p of the free-W variant (about 22K) that is about 4 dense d x d copies, more than
    the RAM left beside the running jobs, and the tied and LEAF-LGN results (computed in float32) must not change."""
    d = G.shape[0]
    reg = torch.full((d,), float(lam_phi), dtype=G.dtype)
    reg[:n] = float(lam)
    G.diagonal().add_(reg)
    try:
        L = torch.linalg.cholesky(G)
    finally:
        G.diagonal().sub_(reg)
    P = torch.cholesky_inverse(L)
    del L
    Th = P @ C
    mu = torch.diagonal(Th)[:n] / torch.diagonal(P)[:n]
    Th -= P[:, :n] * mu[None, :]
    return Th[:n], (Th[n:] if d > n else None)


def collective_ease(O, Y, w, lam, phi=None, lam_phi=0.0):
    """Dense convenience form of collective_gram + collective_solve (used by the tests)."""
    G, C = collective_gram(O, Y, w, phi)
    return collective_solve(G, C, O.shape[1], lam, lam_phi)


def compact_base(O, Y, w, B, rows, cand, phi=None, W=None, chunk=2048):
    """Row-standardised closed-form scores (Z B [+ Phi W]) of the users rows, gathered on the candidates cand."""
    out = torch.empty(cand.shape, dtype=torch.float32)
    for r0 in range(0, len(rows), chunk):
        rr = rows[r0:r0 + chunk]
        o = O[rr].to(B.dtype)
        x = o * Y[rr].to(B.dtype)
        S = (x + w * (o - x)) @ B
        if phi is not None and W is not None:
            S = S + phi.chunk(rr).to(B.dtype) @ W
        out[r0:r0 + len(rr)] = torch.gather(rowstd(S), 1, cand[r0:r0 + len(rr)])
    return out


class LEAFLGN(torch.nn.Module):
    """LightGCN (base) plus beta * pop_i plus the six lifelong channels with learned weights gain (6, zero at start).
    mask (6,) switches channels off (ablations). forward / full_scores follow the drup.learned interface."""

    def __init__(self, base, tabs, pop, mask=None):
        super().__init__()
        self.base = base
        self.tabs = tabs
        self.register_buffer("pop", pop.float())
        self.register_buffer("mask", torch.ones(6) if mask is None else torch.as_tensor(mask, dtype=torch.float32))
        self.gain = torch.nn.Parameter(torch.zeros(6))
        self.beta = torch.nn.Parameter(torch.zeros(1))

    def embeddings(self):
        return self.base.embeddings()

    def forward(self, u, i, emb=None):
        out = self.base(u, i, emb)
        return out + self.beta * self.pop[i] + self.tabs.pairs(u, i) @ (self.gain * self.mask)

    def full_scores(self, rows):
        out = self.base.full_scores(rows) + self.beta * self.pop[None, :]
        return out + self.tabs.dense_rows(rows, self.gain * self.mask)


def make_lightgcn(m, n, X, dim, layers, seed=0):
    """LightGCN on the clicked pairs of the training log, as drup.learned / run_learned build it."""
    torch.manual_seed(seed)
    eu, ei = torch.nonzero(X > 0, as_tuple=True)
    return LightGCN(m, n, dim, eu, ei, torch.ones(len(eu)), layers=layers, r=0.5)


def fit_bpr_policy(model, O, Y, w, lr, wd, epochs, batch, on_epoch, seed=0, log=None):
    """BPR with the policy channel. A copy of the 'bpr' branch of drup.learned.train with one change: the positives
    are the clicked pairs (weight 1) and the shown-not-clicked pairs (weight w, realised by keeping each of them
    with probability w in every epoch). Negatives are uniform items outside the positives (clicked pairs, and
    shown-not-clicked pairs when w > 0), one resample as in train. on_epoch(model, ep) returns True to stop."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    m, n = O.shape
    X = (O * Y) > 0
    N = (O > 0) & ~X if w > 0 else torch.zeros_like(X)
    xu, xi = torch.nonzero(X, as_tuple=True)
    nu, ni = torch.nonzero(N, as_tuple=True)
    rej = X | N
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        if len(nu):
            keep = torch.rand(len(nu), generator=g) < w
            eu, ei = torch.cat([xu, nu[keep]]), torch.cat([xi, ni[keep]])
        else:
            eu, ei = xu, xi
        perm = torch.randperm(len(eu), generator=g)
        su, si = eu[perm], ei[perm]
        for s in range(0, len(su), batch):
            u, i = su[s:s + batch], si[s:s + batch]
            emb = model.embeddings()
            j = torch.randint(0, n, (len(u),), generator=g)
            j = torch.where(rej[u, j], torch.randint(0, n, (len(u),), generator=g), j)
            fp, fneg = model(u, i, emb), model(u, j, emb)
            loss = -F.logsigmoid(fp - fneg).mean()
            reg = wd * sum((e ** 2).sum() for e in (emb[0][u], emb[1][i], emb[1][j])) / len(u)
            opt.zero_grad()
            (loss + reg).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            stop = on_epoch(model, ep)
        if log:
            log(f"  epoch {ep} loss {float(loss):.4f}")
        if stop:
            break


class Layout:
    """Candidate layout of the test users, bucketed by candidate-list length (edges as in the probes): bucket b holds
    the users rows_b (dataset rows), their candidates cand_b (U_b x L_b, zero padded) and their positions k in the
    test list. Score matrices are kept per bucket, so memory follows the number of candidates, not U x max L."""

    EDGES = (32, 64, 128, 256, 100000)

    def __init__(self, test):
        lengths = np.array([len(it) for _, it, _ in test], dtype=np.int64)
        bucket = np.searchsorted(np.asarray(self.EDGES), lengths, side="left")
        self.buckets, self.bucket_of, self.local_of = [], np.zeros(len(test), np.int64), np.zeros(len(test), np.int64)
        for b in range(len(self.EDGES)):
            ks = np.flatnonzero(bucket == b)
            if len(ks) == 0:
                continue
            L = int(lengths[ks].max())
            cand = torch.zeros(len(ks), max(L, 1), dtype=torch.long)
            for r, k in enumerate(ks):
                cand[r, :lengths[k]] = torch.as_tensor(np.asarray(test[k][1], dtype=np.int64))
            self.bucket_of[ks] = len(self.buckets)
            self.local_of[ks] = np.arange(len(ks))
            self.buckets.append({"k": ks, "rows": torch.as_tensor(np.array([test[k][0] for k in ks], dtype=np.int64)),
                                 "cand": cand})


class BucketSelector:
    """Validation selection of one (configuration, epoch) per split, on bucketed score matrices (see Layout). A score
    argument is a list with one (U_b x L_b) matrix per bucket. Test values of the selected configuration are kept per
    user. Same values as drup.metrics.evaluate (checked against the probes)."""

    def __init__(self, layout, splits, ks, key, row_ids):
        from drup.metrics import UserMetrics
        self.layout, self.key, self.row_ids, self.n, self.nc = layout, key, np.asarray(row_ids), 0, 0
        nb = len(layout.buckets)

        def parts(lst):
            groups = [[] for _ in range(nb)]
            for k, it, rel in lst:
                if len(it) > 0 and rel.sum() > 0:
                    groups[layout.bucket_of[k]].append((int(layout.local_of[k]), it, rel))
            return [UserMetrics(g, ks, None, "cpu") if g else None for g in groups]
        self.parts = [(parts(v), parts(t)) for v, t in splits]
        self.best = [{"val": -1.0, "cfg": None, "test": None, "per_user": None} for _ in splits]

    def add(self, S, cfg, active=None, new_config=True):
        """Adds one evaluation (a configuration, or one epoch of it); returns the validation value of every split
        (None if not active). new_config is False for the later epochs of a configuration already counted in nc."""
        self.n += 1
        self.nc += int(new_config)
        vals = []
        for s, (vp, tp) in enumerate(self.parts):
            if active is not None and not active[s]:
                vals.append(None)
                continue
            tot, cnt = 0.0, 0
            for b, um in enumerate(vp):
                if um is not None:
                    tot += float(um.compute(S[b])[self.key].sum())
                    cnt += len(um.users)
            v = tot / max(cnt, 1)
            vals.append(v)
            if v > self.best[s]["val"]:
                per_user = {}
                for b, um in enumerate(tp):
                    if um is None:
                        continue
                    pu = um.compute(S[b])[self.key].tolist()
                    per_user.update(zip(self.layout.buckets[b]["k"][um.users].tolist(), pu))
                self.best[s] = {"val": v, "cfg": dict(cfg), "test": float(np.mean(list(per_user.values()))),
                                "per_user": per_user}
        return vals

    def result(self):
        tests = [b["test"] for b in self.best]
        return {"test_mean": float(np.mean(tests)), "test_sd": float(np.std(tests)),
                "val_mean": float(np.mean([b["val"] for b in self.best])),
                "chosen": [b["cfg"] for b in self.best],
                "per_split": [{self.key: b["test"]} for b in self.best],
                "val_best": [b["val"] for b in self.best],
                "per_user": [{str(int(self.row_ids[k])): v for k, v in b["per_user"].items()} for b in self.best],
                "n_evals": self.n, "n_configs": self.nc}

    def state(self):
        return {"n": self.n, "nc": self.nc, "best": [{"val": b["val"], "cfg": b["cfg"], "test": b["test"],
                                       "per_user": None if b["per_user"] is None else
                                       [[int(k), v] for k, v in b["per_user"].items()]} for b in self.best]}

    def load_state(self, st):
        self.n = st["n"]
        # partial files written before nc existed count evaluations only: exact for the tied variant (one evaluation
        # per configuration), an upper bound for LEAF-LGN (epochs)
        self.nc = st.get("nc", st["n"])
        self.best = [{"val": b["val"], "cfg": b["cfg"], "test": b["test"],
                      "per_user": None if b["per_user"] is None else {k: v for k, v in b["per_user"]}}
                     for b in st["best"]]


def protocol_split(test, seed, frac_val=0.3):
    """Validation / test split of the random-exposure users, as in every stored KuaiRand row (split_test by user).
    The two parts share no user, so every user's candidate list lands in exactly one part."""
    val, tst = split_test(test, frac_val, seed, by="user")
    uv, ut = {u for u, _, _ in val}, {u for u, _, _ in tst}
    assert not (uv & ut) and len(uv) + len(ut) == len(test), "validation and test users overlap or are missing"
    return val, tst


def check_temporal_protocol(d, cut=CUT_MS):
    """Temporal protocol on a drup.data2 dict: every training impression precedes the cut-off, every random
    (test) impression follows it, and the leak counts of the loader are zero."""
    lg, tl = d["log"], d["test_log"]
    max_train, min_test = int(lg.time_ms.max()), int(tl.time_ms.min())
    assert max_train < cut <= min_test, (max_train, cut, min_test)
    leak = d["meta"]["leak"]
    assert leak["ranked_pairs_present_in_training_log"] == 0
    assert leak["test_events_present_in_training_log"] == 0
    return {"max_train_time_ms": max_train, "cut_ms": int(cut), "min_test_time_ms": min_test}
